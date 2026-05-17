//! Deliver a transcript to the focused window. Hands every chunk to
//! [`crate::ax_paste::insert_text`], which posts a synthetic Unicode
//! keystroke via `CGEventKeyboardSetUnicodeString` at the
//! `AnnotatedSession` event-tap layer. See `ax_paste.rs` for the why,
//! and [ADR-0019](../docs/ADR.md#0019--paste-delivery-synthetic-unicode-keystroke)
//! for the path-not-taken history (clipboard+⌘V → races, AX-first →
//! Ghostty silently swallowed the write).
//!
//! Two shapes:
//!
//! - **`deliver`** — one-shot. Used when cleanup is OFF (no streaming
//!   benefit), or as the not-loaded / error fallback when cleanup is
//!   ON but couldn't run.
//! - **`Streamer`** — incremental. Buffers chunks until a word
//!   boundary, then delivers them one at a time. Used when cleanup
//!   is ON so the user sees text appear as the LLM generates instead
//!   of all at once at the end.

use anyhow::Result;

pub fn deliver(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    crate::ax_paste::insert_text(text)
}

/// Incremental paste sink. Each `push` appends to an internal tail
/// buffer and flushes to the focused app when the buffer ends on a
/// word boundary. Word-boundary batching reduces the number of
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
}

impl Streamer {
    /// Construct a fresh streamer. No I/O — the keystroke path needs
    /// no pre-stream setup.
    pub fn start() -> Result<Self> {
        Ok(Self {
            pending: String::new(),
            fired: false,
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

    /// Buffer `chunk` and flush via AX when the buffer ends on a
    /// word boundary (whitespace or sentence-terminating punctuation).
    pub fn push(&mut self, chunk: &str) -> Result<()> {
        if chunk.is_empty() {
            return Ok(());
        }
        self.pending.push_str(chunk);
        // Only flush at a word boundary so partial words don't appear
        // and then get extended (less visual noise than chunk-by-
        // chunk, even though AX doesn't flicker the way ⌘V did).
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
    /// AX-based delivery has nothing to undo here — the clipboard
    /// was never touched.
    pub fn abort(self) {
        // Pending is dropped; that's the entire abort action.
    }

    fn flush_now(&mut self) -> Result<()> {
        let chunk: String = std::mem::take(&mut self.pending);
        crate::ax_paste::insert_text(&chunk)?;
        self.fired = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Streamer behaviour is exercised via integration: see
    // `polish_streaming_*` tests in `cleanup.rs` plus the manual
    // smoke described in `docs/latency-plan.md`. AX itself isn't
    // testable without a real focused element on a live macOS
    // session, so we don't try.
}
