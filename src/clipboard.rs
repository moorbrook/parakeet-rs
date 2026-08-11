//! Last-resort clipboard write, used **only** when keystroke delivery
//! failed.
//!
//! ## Why this exists at all
//!
//! [ADR-0019](../docs/ADR.md#0019--paste-delivery-synthetic-unicode-keystroke)
//! deliberately keeps the happy path off the clipboard: writing the
//! transcript to the pasteboard and synthesising ⌘V races with whatever
//! the user copied a moment ago, and restoring the previous contents
//! afterwards is itself racy. None of that reasoning applies once
//! `ax_paste::insert_text` has already failed — at that point the
//! alternative to touching the clipboard is *losing the user's speech
//! entirely*, which is strictly worse than clobbering a clipboard entry.
//!
//! So: never on success, always on failure. We do **not** snapshot and
//! restore the previous pasteboard contents (the qwen-scribe approach) —
//! the transcript needs to stay on the clipboard until the user pastes
//! it, which is exactly the window in which a restore would destroy it.
//!
//! ## Threading
//!
//! Called from the `transcribe` worker rather than the main queue: the
//! delivery-failure path must not depend on the AppKit run loop being
//! responsive. `NSPasteboard` is not a main-thread-only class (unlike
//! the `NSStatusItem` / `NSMenu` objects in `menubar.rs`, which we do
//! bounce to main), and the pasteboard server is a separate process
//! reached by IPC.
//!
//! The whole body runs inside an `autoreleasepool`: Cocoa requires a
//! pool on any secondary thread that touches its objects, and a Rust
//! `std::thread` has none by default. Without it the autoreleased
//! objects here leak until the thread exits.

use anyhow::{anyhow, Result};
use objc2::rc::autoreleasepool;
use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
use objc2_foundation::NSString;

/// Replace the general pasteboard's contents with `text`.
///
/// Empty input is a no-op success so callers can pass a possibly-empty
/// transcript without branching.
pub fn put(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    autoreleasepool(|_| {
        let pb = NSPasteboard::generalPasteboard();
        // `clearContents` is mandatory before `setString:forType:` —
        // AppKit rejects writes to a pasteboard whose change count wasn't
        // bumped by the current owner, and the failure is a silent `false`
        // return rather than an exception.
        pb.clearContents();
        // SAFETY: both arguments are valid retained Objective-C objects for
        // the duration of this synchronous call; the pasteboard type is the
        // framework's public string constant.
        let ok = unsafe { pb.setString_forType(&NSString::from_str(text), NSPasteboardTypeString) };
        if ok {
            log::info!("clipboard: wrote {} chars as rescue copy", text.len());
            Ok(())
        } else {
            Err(anyhow!("NSPasteboard setString:forType: returned false"))
        }
    })
}
