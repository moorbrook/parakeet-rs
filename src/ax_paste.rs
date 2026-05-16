//! Direct text insertion into the focused app via the macOS
//! Accessibility (AX) API — no clipboard, no `⌘V`, no race.
//!
//! Mechanics:
//!
//! 1. `AXUIElementCreateSystemWide` → a system-wide AX handle.
//! 2. Read `AXFocusedUIElement` → the AX element that currently has
//!    keyboard focus (a text field, web input, IDE editor, etc.).
//! 3. Set `AXSelectedText` on that element to our text. The standard
//!    AX semantics: replaces the current selection, or inserts at
//!    the caret if nothing is selected. The focused app's text engine
//!    handles the insertion natively — exactly what happens when a
//!    user types.
//!
//! Compared to the clipboard+⌘V path:
//!
//! - **No NSPasteboard writes.** The user's clipboard is untouched;
//!   there's nothing to save or restore.
//! - **No ⌘V chord.** Nothing goes through the global event queue,
//!   so the propagation races we hit (write-to-read on the up-side,
//!   restore-before-read on the down-side) simply don't exist.
//! - **No `enigo`/`TSM` main-thread requirement.** AX calls cross the
//!   process boundary via the focused app's `XPC` AX service; they
//!   work from any thread.
//!
//! **What can fail:**
//!
//! - The focused element may not advertise `AXSelectedText` (rare on
//!   native macOS apps; common in some Electron / canvas-based UIs
//!   with custom input handlers). `insert_text` returns an error in
//!   that case so the caller can fall back to clipboard+⌘V.
//! - Password fields intentionally reject programmatic input.
//! - If the Accessibility permission isn't granted, all AX queries
//!   fail. `permissions::check_all` already gates startup on this, so
//!   in normal operation the permission is granted.

use anyhow::{anyhow, bail, Result};
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::string::{CFString, CFStringRef};

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
/// current selection). Returns `Ok(())` if the AX insert succeeded;
/// `Err(...)` otherwise — caller is expected to fall back to a
/// clipboard+⌘V paste.
///
/// Empty input is a no-op success (so it composes cleanly with
/// streaming chunk callbacks).
pub fn insert_text(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
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
                "AXSelectedText set error {err} ({}) — focused element may not support text replacement",
                ax_err_name(err)
            ));
        }
    }
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
