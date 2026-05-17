//! Deliver a transcript to the focused window. Production hands every
//! chunk to [`crate::ax_paste::insert_text`], which posts a synthetic
//! Unicode keystroke via `CGEventKeyboardSetUnicodeString` at the
//! `AnnotatedSession` event-tap layer. See `ax_paste.rs` for the why,
//! and [ADR-0019](../docs/ADR.md#0019--paste-delivery-synthetic-unicode-keystroke)
//! for the path-not-taken history (clipboard+⌘V → races, AX-first →
//! Ghostty silently swallowed the write).
//!
//! ## Where the bytes go
//!
//! Both `deliver` and `Streamer` route through the [`TextSink`] seam.
//! Production wires [`AxKeystrokeSink`] in; tests wire in a recording
//! sink that captures the exact insert calls without needing
//! Accessibility entitlements or a focused window.
//!
//! Two shapes:
//!
//! - **`deliver`** — one-shot. Used when polish is OFF (no streaming
//!   benefit), or as the not-loaded / error fallback when polish is
//!   ON but couldn't run.
//! - **`Streamer`** — incremental. Buffers chunks until a word
//!   boundary, then delivers them one at a time. Used when polish
//!   is ON so the user sees text appear as the LLM generates instead
//!   of all at once at the end.

use anyhow::Result;

/// Seam between the streamer / one-shot paste path and the OS-level
/// text injection. Production wires [`AxKeystrokeSink`]; tests wire a
/// recording sink to make the word-boundary buffering, commit, and
/// abort semantics observable without the OS.
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

/// Incremental paste sink. Each `push` appends to an internal tail
/// buffer and flushes through the [`TextSink`] when the buffer ends
/// on a word boundary. Word-boundary batching reduces the number of
/// `CGEventPost` roundtrips without adding visible latency — the
/// LLM emits tokens faster than the user can read them, so chunking
/// to whole words is invisible.
pub struct Streamer {
    /// Tail buffer for tokens that haven't been flushed yet because
    /// they don't end on a word boundary. Avoids pasting fragments
    /// like "thi" then "s is the" — one cohesive insert per word.
    pending: String,
    /// Set to true once `push` actually delivers a chunk. Lets
    /// `deliver_cleaned` decide whether to fall back to raw paste
    /// on a polish error/panic (when `fired` is true, partial
    /// cleaned text is already on screen; falling back to raw would
    /// duplicate, so the caller keeps the partial instead).
    fired: bool,
    sink: Box<dyn TextSink>,
}

impl Streamer {
    /// Construct a streamer that delivers via the production keystroke
    /// sink. Convenience wrapper around `with_sink` for the production
    /// call site.
    pub fn start() -> Result<Self> {
        Self::with_sink(Box::new(AxKeystrokeSink))
    }

    /// Construct a streamer that delivers to the given sink. Tests use
    /// a recording sink; production uses [`AxKeystrokeSink`].
    pub fn with_sink(sink: Box<dyn TextSink>) -> Result<Self> {
        Ok(Self {
            pending: String::new(),
            fired: false,
            sink,
        })
    }

    /// True iff `push` has actually inserted at least one chunk. The
    /// `deliver_cleaned` fallback path uses this to avoid pasting raw
    /// on top of partially-inserted cleaned output when polish fails
    /// or panics mid-stream — a partial cleaned result is strictly
    /// less bad than `partial + raw` appended.
    pub fn has_fired(&self) -> bool {
        self.fired
    }

    /// Buffer `chunk` and flush through the sink when the buffer ends
    /// on a word boundary (whitespace or sentence-terminating
    /// punctuation).
    pub fn push(&mut self, chunk: &str) -> Result<()> {
        if chunk.is_empty() {
            return Ok(());
        }
        self.pending.push_str(chunk);
        // Only flush at a word boundary so partial words don't appear
        // and then get extended (less visual noise than chunk-by-
        // chunk, even though the keystroke path doesn't flicker the
        // way ⌘V did).
        let has_boundary = self
            .pending
            .chars()
            .last()
            .map(|c| c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '!' | '?'))
            .unwrap_or(false);
        if !has_boundary {
            return Ok(());
        }
        self.flush_now()
    }

    /// Successful end-of-stream. Flushes any buffered tail (the
    /// model's last fragment before EOS is often a word with no
    /// trailing space).
    ///
    /// **Don't call this on a polish error/panic path** — flushing
    /// the tail there would visibly commit a fragment of cleaned
    /// output AND allow the caller's error branch to additionally
    /// paste raw, producing a `partial + raw` mess. Use `abort`
    /// instead.
    pub fn commit(mut self) -> Result<()> {
        if !self.pending.is_empty() {
            self.flush_now()?;
        }
        Ok(())
    }

    /// Aborted end-of-stream. Discards the pending tail (does NOT
    /// flush). Used when polish errored or panicked mid-stream so
    /// the caller can decide between "keep the partial cleaned text
    /// already on screen" and "paste raw as fallback" without an
    /// extra fragment muddying the state.
    ///
    /// Keystroke-based delivery has nothing to undo here — the
    /// clipboard was never touched.
    pub fn abort(self) {
        // Pending is dropped; that's the entire abort action.
    }

    fn flush_now(&mut self) -> Result<()> {
        let chunk: String = std::mem::take(&mut self.pending);
        self.sink.insert(&chunk)?;
        self.fired = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Test sink that records every `insert` call and lets the test
    /// inspect the calls + optionally inject an error on the Nth call.
    #[derive(Default, Clone)]
    struct RecordingSink {
        inner: Arc<Mutex<RecordingState>>,
    }

    #[derive(Default)]
    struct RecordingState {
        calls: Vec<String>,
        fail_on_call: Option<usize>,
    }

    impl RecordingSink {
        fn calls(&self) -> Vec<String> {
            self.inner.lock().unwrap().calls.clone()
        }
        fn boxed(&self) -> Box<dyn TextSink> {
            Box::new(self.clone())
        }
    }

    impl TextSink for RecordingSink {
        fn insert(&mut self, text: &str) -> Result<()> {
            let mut g = self.inner.lock().unwrap();
            g.calls.push(text.to_string());
            if let Some(n) = g.fail_on_call {
                if g.calls.len() == n {
                    return Err(anyhow::anyhow!("synthetic sink failure"));
                }
            }
            Ok(())
        }
    }

    #[test]
    fn streamer_buffers_until_word_boundary_then_flushes() {
        // "hello" — no boundary char — must NOT flush.
        // "hello " — space at end — MUST flush as "hello ".
        // After flush, pending is empty.
        let sink = RecordingSink::default();
        let mut s = Streamer::with_sink(sink.boxed()).unwrap();
        s.push("hello").unwrap();
        assert!(sink.calls().is_empty(), "must not flush mid-word");
        assert!(!s.has_fired());
        s.push(" ").unwrap();
        assert_eq!(sink.calls(), vec!["hello "]);
        assert!(s.has_fired());
    }

    #[test]
    fn streamer_flushes_on_sentence_punctuation() {
        // Each of `. , ; : ! ?` is a flushable boundary. A regression
        // that drops one fails this test loudly.
        for punct in [".", ",", ";", ":", "!", "?"] {
            let sink = RecordingSink::default();
            let mut s = Streamer::with_sink(sink.boxed()).unwrap();
            s.push("done").unwrap();
            s.push(punct).unwrap();
            assert_eq!(
                sink.calls(),
                vec![format!("done{punct}")],
                "punctuation {punct:?} should flush",
            );
        }
    }

    #[test]
    fn streamer_skips_flush_when_chunk_ends_mid_word() {
        // Mutation-survivable: if the boundary check accidentally
        // returns true for letters, "hell" would flush and this fails.
        let sink = RecordingSink::default();
        let mut s = Streamer::with_sink(sink.boxed()).unwrap();
        s.push("hell").unwrap();
        s.push("o").unwrap();
        assert!(sink.calls().is_empty(), "no boundary, no flush");
        assert!(!s.has_fired());
    }

    #[test]
    fn streamer_commit_flushes_dangling_tail() {
        // EOS without a trailing space — typical of the LLM's last
        // fragment. `commit` MUST flush; otherwise the user loses the
        // final word.
        let sink = RecordingSink::default();
        let mut s = Streamer::with_sink(sink.boxed()).unwrap();
        s.push("trailing").unwrap();
        assert!(sink.calls().is_empty());
        s.commit().unwrap();
        assert_eq!(sink.calls(), vec!["trailing"]);
    }

    #[test]
    fn streamer_abort_discards_tail() {
        // Polish error mid-stream: caller calls `abort`, not `commit`.
        // The pending tail must NOT be flushed — the caller will paste
        // raw as the fallback, and the tail would just be duplicated
        // content.
        let sink = RecordingSink::default();
        let mut s = Streamer::with_sink(sink.boxed()).unwrap();
        s.push("polished ").unwrap();
        s.push("tail").unwrap();
        // The space flushed "polished "; "tail" is still buffered.
        assert_eq!(sink.calls(), vec!["polished "]);
        s.abort();
        // Still just the one flushed chunk — abort didn't push "tail".
        assert_eq!(sink.calls(), vec!["polished "]);
    }

    #[test]
    fn streamer_has_fired_stays_false_for_empty_pushes() {
        let sink = RecordingSink::default();
        let mut s = Streamer::with_sink(sink.boxed()).unwrap();
        s.push("").unwrap();
        s.push("").unwrap();
        assert!(!s.has_fired());
        assert!(sink.calls().is_empty());
    }

    #[test]
    fn streamer_push_propagates_sink_error() {
        // Polish-mid-flow sink failure (e.g. a transient OS error)
        // bubbles up to `push`'s caller, which then drives the
        // error/abort fallback. The fired state from the prior
        // successful insert is preserved so the caller can decide
        // "keep partial vs paste raw".
        let sink = RecordingSink::default();
        sink.inner.lock().unwrap().fail_on_call = Some(2);
        let mut s = Streamer::with_sink(sink.boxed()).unwrap();
        s.push("first ").unwrap();
        let err = s.push("second ").unwrap_err();
        assert!(
            err.to_string().contains("synthetic sink failure"),
            "expected sink error to surface, got: {err}"
        );
        assert!(s.has_fired(), "fired flag must survive the error");
    }

    #[test]
    fn deliver_routes_empty_text_through_without_calling_sink() {
        // Empty input is a documented no-op success — `deliver_cleaned`
        // can call `deliver("")` when the LLM produces nothing and we
        // want to fall back to raw paste of an empty transcript.
        let mut sink = RecordingSink::default();
        deliver_via("", &mut sink).unwrap();
        assert!(sink.calls().is_empty());
    }

    #[test]
    fn deliver_routes_text_through_the_sink_once() {
        let mut sink = RecordingSink::default();
        deliver_via("hello world", &mut sink).unwrap();
        assert_eq!(sink.calls(), vec!["hello world"]);
    }
}
