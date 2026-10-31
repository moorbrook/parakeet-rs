//! Deliver a transcript to the focused window. Production hands the
//! text to [`crate::ax_paste::insert_text`], which posts a synthetic
//! Unicode keystroke via `CGEventKeyboardSetUnicodeString` at the
//! `AnnotatedSession` event-tap layer. See `ax_paste.rs` for the why,
//! and [ADR-0019](../docs/ADR.md#0019--paste-delivery-synthetic-unicode-keystroke)
//! for the path-not-taken history (clipboard+⌘V → races, AX-first →
//! Ghostty silently swallowed the write).
//!
//! `deliver` routes through the [`TextSink`] seam. Production wires
//! [`AxKeystrokeSink`] in; tests wire in a recording sink that captures
//! the exact insert calls without needing Accessibility entitlements or
//! a focused window.

use anyhow::Result;

/// Seam between the paste path and the OS-level text injection.
/// Production wires [`AxKeystrokeSink`]; tests wire a recording sink.
pub trait TextSink: Send {
    fn insert(&mut self, text: &str) -> Result<()>;
}

/// Production sink: posts a synthetic Unicode keystroke through
/// `CGEventPost` at the `AnnotatedSession` layer. See `ax_paste.rs`.
pub struct AxKeystrokeSink;

impl TextSink for AxKeystrokeSink {
    fn insert(&mut self, text: &str) -> Result<()> {
        crate::ax_paste::insert_text(text)
    }
}

pub fn deliver(text: &str) -> Result<()> {
    deliver_via(text, &mut AxKeystrokeSink)
}

fn deliver_via<S: TextSink + ?Sized>(text: &str, sink: &mut S) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    sink.insert(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test sink that records every `insert` call.
    #[derive(Default)]
    struct RecordingSink {
        calls: Vec<String>,
    }

    impl TextSink for RecordingSink {
        fn insert(&mut self, text: &str) -> Result<()> {
            self.calls.push(text.to_string());
            Ok(())
        }
    }

    #[test]
    fn deliver_routes_empty_text_through_without_calling_sink() {
        // Empty input is a documented no-op success: an empty transcript
        // pastes nothing.
        let mut sink = RecordingSink::default();
        deliver_via("", &mut sink).unwrap();
        assert!(sink.calls.is_empty());
    }

    #[test]
    fn deliver_routes_text_through_the_sink_once() {
        let mut sink = RecordingSink::default();
        deliver_via("hello world", &mut sink).unwrap();
        assert_eq!(sink.calls, vec!["hello world"]);
    }
}
