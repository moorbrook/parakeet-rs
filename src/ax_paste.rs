//! Direct text insertion into the focused app via a synthetic
//! Unicode keystroke (`CGEventCreateKeyboardEvent` +
//! `CGEventKeyboardSetUnicodeString`).
//!
//! ## Why not the Accessibility API
//!
//! An earlier revision used `AXUIElementSetAttributeValue(AXSelectedText)`
//! as the primary path (the same path Apple's Voice Dictation uses)
//! with a keystroke fallback for apps that rejected AX. Observed
//! failure: Ghostty's focused element is an `AXTextArea` representing
//! the rendered scrollback view — `AXSelectedText` returns SUCCESS
//! but the write never reaches the PTY input pipe. AX silently drops
//! the write, returns "OK", and the user sees nothing pasted.
//!
//! Trusting the AX success code is therefore unsafe. Without a
//! reliable way to verify the insertion landed (reading back
//! `AXValue` is slow, race-prone, and many elements don't expose it),
//! we'd need to detect "AX lies for this app" out-of-band — bundle-id
//! allowlists, hardcoded terminal IDs, etc. — and the maintenance
//! burden never ends.
//!
//! The synthetic keystroke goes through the standard `NSResponder`
//! / WebView / PTY input pipeline that every app already handles.
//! Coverage in practice:
//!
//! - Terminals: Ghostty, iTerm2, Terminal.app, Warp, Alacritty
//! - Browsers: Safari, Chrome, Firefox, Arc (text fields + web inputs)
//! - Native Cocoa: TextEdit, Notes, Mail, Messages, Pages
//! - Electron: Slack, Discord, VS Code, Cursor
//! - IDE editors: Xcode, JetBrains family
//!
//! What it doesn't reach: password fields (intentional), apps with
//! aggressive input filtering (some games, accessibility-blocking
//! utilities), apps with no keyboard handling at all.
//!
//! ## Event post location
//!
//! `CGEventTapLocation::AnnotatedSession` (not `HID`) so the event
//! lands at the session layer where text-input frameworks consume the
//! attached Unicode string. `HID` works for some apps but terminals
//! (Ghostty in particular) tap HID for their own keymap and may not
//! surface the unicode string properly.

use anyhow::{anyhow, Result};
use core_graphics::event::{CGEvent, CGEventTapLocation};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

/// Insert `text` at the focused element's caret. Returns `Ok(())`
/// if the synthesised keystroke was successfully posted to the OS
/// event queue. **Does not (cannot) verify the focused app actually
/// processed it** — `CGEventPost` is fire-and-forget.
///
/// Empty input is a no-op success (so it composes cleanly with
/// streaming chunk callbacks).
///
/// Every call logs the chunk size + a short preview at INFO level
/// for observability.
pub fn insert_text(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let preview = text_preview(text);

    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|()| anyhow!("CGEventSource::new failed"))?;
    // Keycode 0 is 'a' on QWERTY but it doesn't matter — text-aware
    // apps read the attached unicode string (set via
    // `CGEventKeyboardSetUnicodeString` under `CGEvent::set_string`),
    // not the keycode. The key event still needs a valid keycode so
    // apps that look at the raw keycode don't crash on `nil`.
    let keydown = CGEvent::new_keyboard_event(source.clone(), 0, true)
        .map_err(|()| anyhow!("CGEventCreateKeyboardEvent keydown failed"))?;
    keydown.set_string(text);
    keydown.post(CGEventTapLocation::AnnotatedSession);

    // Matching keyup so the focused app doesn't observe a dangling
    // key-down (some apps and accessibility tools track key state).
    let keyup = CGEvent::new_keyboard_event(source, 0, false)
        .map_err(|()| anyhow!("CGEventCreateKeyboardEvent keyup failed"))?;
    keyup.set_string(text);
    keyup.post(CGEventTapLocation::AnnotatedSession);

    log::info!(
        "ax_paste: keystroke posted ({} chars: {preview}) → AnnotatedSession",
        text.len()
    );
    Ok(())
}

/// Short, log-safe preview of `text` — first ~32 chars, no
/// newlines, control chars escaped. Used for telemetry only.
fn text_preview(text: &str) -> String {
    const MAX: usize = 32;
    let mut out = String::with_capacity(MAX + 4);
    out.push('"');
    for (count, ch) in text.chars().enumerate() {
        if count >= MAX {
            out.push('…');
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push('?'),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
