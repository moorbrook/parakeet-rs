//! Deliver a transcript to the focused window via the macOS
//! Accessibility API (`AXUIElementSetAttributeValue(AXSelectedText)`).
//!
//! This module used to manage clipboard save/restore + a synthetic
//! `⌘V` chord, with a tail of races we kept chasing (write-to-read
//! propagation, restore-before-read in the target app's keyDown
//! handler, the `enigo`→TSM main-thread crash). All of that is gone —
//! AX writes straight to the focused element's text engine, the same
//! way Apple's own Voice Dictation works.
//!
//! Two shapes:
//!
//! - **`deliver`** — one-shot. Hands `text` to AX. Used when cleanup
//!   is OFF (no streaming benefit), or as the not-loaded / error
//!   fallback when cleanup is ON but couldn't run.
//! - **`Streamer`** — incremental. Buffers chunks until a word
//!   boundary, then hands them to AX one at a time. Used when cleanup
//!   is ON so the user sees text appear as the LLM generates instead
//!   of all at once at the end.
//!
//! `inject_mode` in settings is accepted for forward compatibility
//! with older settings.json files but ignored — there's only one
//! delivery path now.

use std::time::Instant;

use anyhow::Result;

pub fn deliver(text: &str, _inject_mode: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    crate::ax_paste::insert_text(text)
}

/// Incremental paste sink. Each `push` appends to an internal tail
/// buffer and flushes to the focused app via `ax_paste::insert_text`
/// when the buffer ends on a word boundary. Word-boundary batching
/// reduces the number of AX XPC roundtrips without adding visible
/// latency — the LLM emits tokens faster than the user can read
/// them, so chunking to whole words is invisible.
pub struct Streamer {
    last_push_at: Option<Instant>,
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
    /// Construct a fresh streamer. No I/O — the AX path doesn't need
    /// any pre-stream setup.
    pub fn start() -> Result<Self> {
        Ok(Self {
            last_push_at: None,
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
        self.last_push_at = Some(Instant::now());
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
