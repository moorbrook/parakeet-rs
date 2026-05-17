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
///
/// Every call logs which path was tried and the outcome at INFO
/// level so the path Ghostty / iTerm2 / VS Code / etc. actually
/// take is visible in `log stream`.
pub fn insert_text(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let preview = text_preview(text);
    match insert_via_ax(text) {
        Ok(()) => {
            log::info!("ax_paste: AX insert OK ({} chars: {preview})", text.len());
            Ok(())
        }
        Err(ax_err) => {
            log::info!("ax_paste: AX rejected ({ax_err:#}); trying CGEvent keystroke");
            insert_via_keystroke(text)
                .map(|()| {
                    log::info!(
                        "ax_paste: CGEvent keystroke OK ({} chars: {preview})",
                        text.len()
                    );
                })
                .map_err(|ks_err| {
                    log::error!(
                        "ax_paste: BOTH paths failed ({} chars: {preview}) — AX: {ax_err:#}; keystroke: {ks_err:#}",
                        text.len()
                    );
                    anyhow!(
                        "both delivery paths failed — AX: {ax_err:#}; keystroke: {ks_err:#}"
                    )
                })
        }
    }
}

/// Short, log-safe preview of `text` — first ~32 chars, no newlines.
fn text_preview(text: &str) -> String {
    const MAX: usize = 32;
    let mut out = String::with_capacity(MAX + 4);
    out.push('"');
    let mut count = 0;
    for ch in text.chars() {
        if count >= MAX {
            out.push_str("…");
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push('?'),
            c => out.push(c),
        }
        count += 1;
    }
    out.push('"');
    out
}

/// Accessibility-based insertion via `AXUIElementSetAttributeValue
/// (focused, AXSelectedText, text)`. The cleanest path for native
/// Cocoa apps and most browsers. Logs focused-element diagnostics
/// (role, subrole, parent app pid) before the set so failures are
/// debuggable.
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

        // Telemetry: what does AX think is focused? Helps diagnose
        // apps where insertion silently doesn't land.
        let role = read_string_attr(focused_ref as AXUIElementRef, "AXRole")
            .unwrap_or_else(|| "?".into());
        let subrole = read_string_attr(focused_ref as AXUIElementRef, "AXSubrole")
            .unwrap_or_else(|| "-".into());
        log::info!("ax_paste: focused element role={role} subrole={subrole}");

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
                "AXSelectedText set error {err} ({}) on role={role} subrole={subrole}",
                ax_err_name(err)
            ));
        }
    }
    Ok(())
}

/// Read a string AX attribute on `element`, returning `None` on any
/// error or non-string value. Used for telemetry only — failures
/// here are non-fatal.
///
/// SAFETY: caller must pass a valid `AXUIElementRef`. CFString
/// wrap-under-create-rule consumes the +1 retain returned by
/// `AXUIElementCopyAttributeValue`, so no manual `CFRelease`.
unsafe fn read_string_attr(element: AXUIElementRef, attr: &str) -> Option<String> {
    let attr_cf = CFString::new(attr);
    let mut value_ref: CFTypeRef = std::ptr::null();
    // SAFETY: `element` is the caller's valid AXUIElementRef;
    // `attr_cf` outlives the call; `value_ref` is initialised here.
    let err = unsafe {
        AXUIElementCopyAttributeValue(element, attr_cf.as_concrete_TypeRef(), &mut value_ref)
    };
    if err != KAX_ERROR_SUCCESS || value_ref.is_null() {
        return None;
    }
    // SAFETY: AX returned a +1 retained CFStringRef; wrap-under-
    // create-rule transfers ownership to the Rust wrapper which
    // CFReleases on drop.
    let cf_str = unsafe { CFString::wrap_under_create_rule(value_ref as CFStringRef) };
    Some(cf_str.to_string())
}

/// Synthetic Unicode keystroke via
/// `CGEventCreateKeyboardEvent` + `CGEventKeyboardSetUnicodeString`.
/// The focused app sees a key-down/key-up pair whose attached
/// Unicode string IS the text — its standard text-input pipeline
/// processes it as user typing. The keycode itself (we use `0`)
/// doesn't matter because the Unicode string overrides it for
/// text-aware apps.
///
/// Posts to `AnnotatedSession` (not `HID`) because some terminals
/// (Ghostty included) tap HID-level events to implement their own
/// keymap and may not see the unicode string the way text-input
/// frameworks at the session layer do. AnnotatedSession is what
/// text-replacement / dictation APIs use under the hood.
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
    keydown.post(CGEventTapLocation::AnnotatedSession);
    log::info!("ax_paste: keydown posted to AnnotatedSession");
    // Matching keyup so the focused app doesn't observe a dangling
    // key-down. Some apps (and some accessibility tools) care.
    let keyup = CGEvent::new_keyboard_event(source, 0, false)
        .map_err(|()| anyhow!("CGEventCreateKeyboardEvent keyup failed"))?;
    keyup.set_string(text);
    keyup.post(CGEventTapLocation::AnnotatedSession);
    log::info!("ax_paste: keyup posted to AnnotatedSession");
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
