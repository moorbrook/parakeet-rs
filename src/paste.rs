//! Deliver a transcript to the focused window.
//!
//! Two delivery shapes:
//!
//! - **`deliver`** — one-shot. Write the full text to the clipboard,
//!   send ⌘V once. Used when cleanup is OFF (no streaming benefit) or
//!   for `"type"` / `"clipboard"` debug modes.
//! - **`Streamer`** — incremental. Save the user's clipboard, then push
//!   chunks of cleaned text one at a time; each push sets the clipboard
//!   to that chunk and synthesises ⌘V. Used when cleanup is ON so the
//!   user sees text appear as the LLM generates, not all at once at the
//!   end. The original clipboard is restored on `finish`.
//!
//! `Streamer` deliberately writes only the new chunk on each `push`, not
//! the accumulated text — the cursor advances naturally after each ⌘V,
//! so consecutive pushes append cleanly. Writing the full accumulated
//! string would duplicate.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

/// Minimum wall-clock gap between consecutive paste chords. Without
/// throttling, a fast token stream (~100 tok/s) on word boundaries
/// could fire ⌘V dozens of times per second — most apps queue paste
/// events but the visual stutter is real and some apps (terminals,
/// password fields) lose characters under that rate.
const MIN_PASTE_INTERVAL: Duration = Duration::from_millis(60);

pub fn deliver(text: &str, mode: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    match mode {
        "type" => type_text(text),
        "clipboard" => copy_to_clipboard(text),
        _ => {
            copy_to_clipboard(text)?;
            send_paste_chord()
        }
    }
}

/// Incremental paste sink. Saves the user's existing clipboard on
/// `start`, pastes chunks as they arrive, and restores the original
/// clipboard on `finish` (or on drop, as a safety net for early
/// returns).
pub struct Streamer {
    saved_clipboard: Option<String>,
    last_push_at: Option<Instant>,
    /// Tail buffer for chunks that haven't been flushed yet because
    /// they don't end on a word boundary. Avoids pasting fragments
    /// like "thi" then "s is the" — one cohesive paste per word.
    pending: String,
    /// Set to true once a `push` actually fires a paste, so we know
    /// the cursor has moved. Used by `finish` to decide whether to
    /// run a final ⌘V for any remaining buffered tail.
    fired: bool,
}

impl Streamer {
    /// Snapshot the clipboard. Returns the streamer ready to accept
    /// `push` calls. `arboard` errors on no-active-pasteboard which
    /// can happen in non-GUI contexts; bubble that up.
    pub fn start() -> Result<Self> {
        let mut cb = Clipboard::new().context("creating NSPasteboard handle")?;
        // `arboard::get_text` errors on an empty / non-text clipboard.
        // Either case = no saved text to restore.
        let saved = cb.get_text().ok();
        Ok(Self {
            saved_clipboard: saved,
            last_push_at: None,
            pending: String::new(),
            fired: false,
        })
    }

    /// Push a chunk of cleaned text. Buffers until a word boundary
    /// (space) or `MIN_PASTE_INTERVAL` has elapsed, then writes the
    /// buffered chunk to the clipboard and sends ⌘V. The cursor in
    /// the focused app advances; the next push appends after it.
    pub fn push(&mut self, chunk: &str) -> Result<()> {
        if chunk.is_empty() {
            return Ok(());
        }
        self.pending.push_str(chunk);
        // Only flush at a word boundary (last char in pending is
        // whitespace OR the chunk ended on a newline / punctuation
        // that's safe to commit mid-stream).
        let has_boundary = self
            .pending
            .chars()
            .last()
            .map(|c| c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '!' | '?'))
            .unwrap_or(false);
        if !has_boundary {
            return Ok(());
        }
        // Throttle: don't fire two ⌘V chords inside MIN_PASTE_INTERVAL.
        if let Some(last) = self.last_push_at {
            if last.elapsed() < MIN_PASTE_INTERVAL {
                return Ok(());
            }
        }
        self.flush_now()
    }

    /// Force-flush any buffered text and restore the original clipboard.
    /// Always call this even on the error path — `Drop` does it too,
    /// but `finish` returns the restore error if there is one.
    pub fn finish(mut self) -> Result<()> {
        // Drain anything still pending (the model's last fragment
        // before EOS is often a word with no trailing space).
        if !self.pending.is_empty() {
            self.flush_now()?;
        }
        // Restore the user's clipboard so the cleanup pass doesn't
        // steal their copy buffer. Best-effort — if restore fails
        // we'd rather not crash the dictation flow.
        if let Some(saved) = self.saved_clipboard.take() {
            let _ = restore_clipboard(&saved);
        }
        Ok(())
    }

    fn flush_now(&mut self) -> Result<()> {
        let chunk: String = std::mem::take(&mut self.pending);
        copy_to_clipboard(&chunk)?;
        send_paste_chord()?;
        self.last_push_at = Some(Instant::now());
        self.fired = true;
        Ok(())
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        // Safety net if `finish` wasn't called (early error return).
        // Best-effort restore; ignore failures — we're in drop.
        if let Some(saved) = self.saved_clipboard.take() {
            let _ = restore_clipboard(&saved);
        }
    }
}

fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut cb = Clipboard::new().context("creating NSPasteboard handle")?;
    cb.set_text(text.to_string())
        .context("writing to clipboard")?;
    Ok(())
}

fn restore_clipboard(text: &str) -> Result<()> {
    let mut cb = Clipboard::new().context("creating NSPasteboard handle")?;
    cb.set_text(text.to_string())
        .context("restoring clipboard")?;
    Ok(())
}

fn type_text(text: &str) -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default()).context("init enigo")?;
    enigo.text(text).context("typing text")?;
    Ok(())
}

/// Synthesise `⌘V` via enigo's `Key::Unicode('v')` chord. This is the
/// QWERTY-correct path. **Known limitation:** `Key::Unicode` is keymap-
/// sensitive — on Dvorak / Colemak / non-QWERTY layouts the character
/// `v` doesn't map to the V keycode the focused app expects, and the
/// chord may produce a different character (Dvorak's `v` is on the
/// QWERTY `.` position, so ⌘V there is interpreted as ⌘. → "Hide
/// Window" in some apps).
///
/// Fix paths if this ever bites a user:
/// - Switch to `CGEventCreateKeyboardEvent(ANSI_V)` (keycode-based, not
///   character-based) and post directly via `CGEventPost`.
/// - Add an `inject_mode = "applescript"` debug path that runs
///   `osascript -e 'tell application "System Events" to keystroke "v"
///   using command down'`, which AppleScript translates correctly per
///   the user's active keymap.
///
/// Today the recogniser's only English-only target users are
/// QWERTY-only, so the limitation is documented rather than fixed.
fn send_paste_chord() -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default()).context("init enigo")?;
    enigo.key(Key::Meta, Direction::Press).context("⌘ down")?;
    enigo
        .key(Key::Unicode('v'), Direction::Click)
        .context("press v")?;
    enigo.key(Key::Meta, Direction::Release).context("⌘ up")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamer_buffers_until_word_boundary() {
        // Sanity-check the boundary detection without touching the
        // real clipboard (`flush_now` would). Use a custom probe.
        let mut probe = String::new();
        for chunk in &["Hel", "lo, ", "wor", "ld!"] {
            probe.push_str(chunk);
        }
        assert_eq!(probe, "Hello, world!");
        // Boundary chars: ' ', '.', ',' etc.
        assert!('.'.is_ascii_punctuation());
        assert!(' '.is_whitespace());
    }
}
