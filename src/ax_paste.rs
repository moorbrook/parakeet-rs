//! Direct text insertion into the focused app — no clipboard, no
//! `⌘V`, no race. Two layered paths:
//!
//! 1. **Accessibility (`AXUIElementSetAttributeValue(AXSelectedText)`).**
//!    Apple's own Voice Dictation uses this. The standard AX
//!    semantics: replaces the current selection, or inserts at the
//!    caret if nothing is selected. The focused app's text engine
//!    handles the insertion natively. Works in: Safari, Chrome,
//!    Slack, Notes, TextEdit, Mail, Cursor, VS Code (native text
//!    fields), and most Cocoa apps.
//!
//! 2. **Synthetic Unicode keystroke
//!    (`CGEventCreateKeyboardEvent` + `CGEventKeyboardSetUnicodeString`).**
//!    Fallback when AX fails. Injects a key-down/key-up pair whose
//!    attached Unicode string IS the text. The focused app's text
//!    input system sees "user typed this" and processes it. Works in:
//!    Ghostty, iTerm2, Terminal.app, most Electron apps with custom
//!    input handlers, and basically anything that supports keyboard
//!    input at all.
//!
//! `insert_text` tries (1) then (2); if both fail it surfaces an
//! error so the caller can show a user-visible status.
//!
//! Compared to the old clipboard+⌘V path:
//!
//! - **No NSPasteboard writes.** The user's clipboard is untouched;
//!   nothing to save or restore.
//! - **No `⌘V` chord.** Nothing goes through the global event queue,
//!   so the propagation races we hit (write-to-read on the up-side,
//!   restore-before-read on the down-side) simply don't exist.
//! - **No `enigo`/`TSM` main-thread requirement.** Both paths work
//!   from any thread.
//!
//! **Real-world failure modes:**
//!
//! - Password fields intentionally reject both AX and synthetic
//!   keystrokes.
//! - Apps with very aggressive input filtering (some games, some
//!   accessibility-blocking apps) may reject both paths.
//! - If the Accessibility permission isn't granted, AX queries fail
//!   AND `CGEventPost` is rate-limited. `permissions::check_all`
//!   gates startup on Accessibility, so in normal operation this
//!   doesn't happen.

use anyhow::{anyhow, bail, Result};
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::string::{CFString, CFStringRef};
use core_graphics::event::{CGEvent, CGEventTapLocation};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

// AXUIElement is an opaque CFType; we only ever pass pointers around.
#[repr(C)]
struct __AXUIElement {
    _private: [u8; 0],
}
type AXUIElementRef = *const __AXUIElement;

type AXError = i32;
const KAX_ERROR_SUCCESS: AXError = 0;

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> AXError;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> AXError;
}

/// Insert `text` at the focused element's caret (or replace its
/// current selection). Tries Accessibility first; falls back to a
/// synthetic Unicode keystroke if AX rejects the focused element
/// (terminals, custom-input Electron apps).
///
/// Empty input is a no-op success (so it composes cleanly with
/// streaming chunk callbacks).
pub fn insert_text(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    match insert_via_ax(text) {
        Ok(()) => Ok(()),
        Err(ax_err) => {
            log::debug!("AX insert rejected, trying synthetic keystroke: {ax_err:#}");
            insert_via_keystroke(text).map_err(|ks_err| {
                anyhow!(
                    "both delivery paths failed — AX: {ax_err:#}; keystroke: {ks_err:#}"
                )
            })
        }
    }
}

/// Accessibility-based insertion via `AXUIElementSetAttributeValue
/// (focused, AXSelectedText, text)`. The cleanest path for native
/// Cocoa apps and most browsers.
fn insert_via_ax(text: &str) -> Result<()> {
    // SAFETY: all AX calls take CFType pointers we own (system-wide
    // handle, focused element) plus CFString refs whose Rust-side
    // lifetime brackets the call. CFRelease balances the Copy-rule
    // retains.
    unsafe {
        let system = AXUIElementCreateSystemWide();
        if system.is_null() {
            bail!("AXUIElementCreateSystemWide returned null");
        }

        let focused_attr = CFString::new("AXFocusedUIElement");
        let mut focused_ref: CFTypeRef = std::ptr::null();
        let err = AXUIElementCopyAttributeValue(
            system,
            focused_attr.as_concrete_TypeRef(),
            &mut focused_ref,
        );
        CFRelease(system as CFTypeRef);
        if err != KAX_ERROR_SUCCESS {
            return Err(anyhow!("AXFocusedUIElement read error {err} ({})", ax_err_name(err)));
        }
        if focused_ref.is_null() {
            bail!("no focused UI element (nothing has keyboard focus?)");
        }

        let selected_attr = CFString::new("AXSelectedText");
        let value = CFString::new(text);
        let err = AXUIElementSetAttributeValue(
            focused_ref as AXUIElementRef,
            selected_attr.as_concrete_TypeRef(),
            value.as_concrete_TypeRef() as CFTypeRef,
        );
        CFRelease(focused_ref);
        if err != KAX_ERROR_SUCCESS {
            return Err(anyhow!(
                "AXSelectedText set error {err} ({}) — focused element doesn't support text replacement (terminal? canvas UI?)",
                ax_err_name(err)
            ));
        }
    }
    Ok(())
}

/// Synthetic Unicode keystroke via
/// `CGEventCreateKeyboardEvent` + `CGEventKeyboardSetUnicodeString`.
/// The focused app sees a key-down/key-up pair whose attached
/// Unicode string IS the text — its standard text-input pipeline
/// processes it as user typing. The keycode itself (we use `0`)
/// doesn't matter because the Unicode string overrides it for
/// text-aware apps.
///
/// This is what makes dictation work in terminals (Ghostty,
/// iTerm2, Terminal.app) — they don't expose `AXSelectedText` but
/// they do consume keyboard events normally.
fn insert_via_keystroke(text: &str) -> Result<()> {
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|()| anyhow!("CGEventSource::new failed"))?;
    // Keycode 0 = 'a' on QWERTY, but `set_string` overrides what
    // text-aware apps actually receive. The key event still needs a
    // valid keycode so apps that look at it (rare for text input)
    // don't crash.
    let keydown = CGEvent::new_keyboard_event(source.clone(), 0, true)
        .map_err(|()| anyhow!("CGEventCreateKeyboardEvent keydown failed"))?;
    keydown.set_string(text);
    keydown.post(CGEventTapLocation::HID);
    // Matching keyup so the focused app doesn't observe a dangling
    // key-down. Some apps (and some accessibility tools) care.
    let keyup = CGEvent::new_keyboard_event(source, 0, false)
        .map_err(|()| anyhow!("CGEventCreateKeyboardEvent keyup failed"))?;
    keyup.set_string(text);
    keyup.post(CGEventTapLocation::HID);
    Ok(())
}

/// Best-effort human-readable name for the small set of AX error
/// codes we actually see in practice. Anything unrecognised falls
/// back to the numeric code in the caller's error message.
fn ax_err_name(err: AXError) -> &'static str {
    match err {
        -25204 => "kAXErrorAPIDisabled",
        -25205 => "kAXErrorActionUnsupported",
        -25206 => "kAXErrorAttributeUnsupported",
        -25207 => "kAXErrorCannotComplete",
        -25208 => "kAXErrorNoValue",
        -25209 => "kAXErrorParameterizedAttributeUnsupported",
        -25210 => "kAXErrorNotEnoughPrecision",
        -25211 => "kAXErrorNotImplemented",
        -25212 => "kAXErrorIllegalArgument",
        -25213 => "kAXErrorInvalidUIElement",
        -25214 => "kAXErrorInvalidUIElementObserver",
        -25200 => "kAXErrorFailure",
        _ => "AXError",
    }
}
