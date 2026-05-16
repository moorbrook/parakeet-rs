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

use anyhow::{anyhow, Context, Result};
use arboard::Clipboard;
use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use enigo::{Enigo, Keyboard, Settings};

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

    /// True iff `push` has actually fired at least one ⌘V chord. The
    /// `deliver_cleaned` fallback path uses this to avoid double-pasting
    /// raw text on top of already-streamed cleaned output when polish
    /// fails or panics mid-stream — a partial cleaned result is strictly
    /// less bad than `partial + raw` appended.
    pub fn has_fired(&self) -> bool {
        self.fired
    }

    /// Successful end-of-stream. Flushes any buffered tail (the model's
    /// last fragment before EOS is often a word with no trailing space),
    /// then restores the user's clipboard.
    ///
    /// **Don't call this on a polish error/panic path** — flushing the
    /// tail there would visibly commit a fragment of cleaned output AND
    /// allow the caller's error branch to additionally paste raw, producing
    /// a `partial + raw` mess. Use `abort` instead.
    pub fn commit(mut self) -> Result<()> {
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

    /// Aborted end-of-stream. Discards the pending tail (does NOT flush)
    /// and restores the user's clipboard. Use this when polish errored
    /// or panicked mid-stream so the caller can decide between "keep the
    /// partial cleaned text already on screen" and "paste raw as
    /// fallback" without an extra fragment muddying the state.
    pub fn abort(mut self) {
        self.pending.clear();
        if let Some(saved) = self.saved_clipboard.take() {
            let _ = restore_clipboard(&saved);
        }
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

/// Type `text` one character at a time. Only used when the user
/// explicitly sets `inject_mode = "type"`; the default delivery is
/// clipboard+chord above.
///
/// **Same TSM main-thread caveat as the old `send_paste_chord`** —
/// `Enigo::new` calls into the Text Services Manager which asserts
/// main-thread-only. If this path is ever taken from the `transcribe`
/// worker it will abort the process. The CGEvent rewrite of
/// `send_paste_chord` didn't touch this because no current code path
/// exercises `type` mode; if you wire it up, marshal the call to the
/// main thread via GCD first.
fn type_text(text: &str) -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default()).context("init enigo")?;
    enigo.text(text).context("typing text")?;
    Ok(())
}

/// ANSI 'V' keycode (Carbon HIToolbox / `Events.h`). Constant per
/// macOS — `CGEventCreateKeyboardEvent` interprets keycodes as
/// physical positions on the ANSI layout, so this is correct on
/// QWERTY, Dvorak, Colemak, etc., regardless of the user's active
/// keymap. The previous `enigo::Key::Unicode('v')` path was keymap-
/// sensitive (Dvorak's `v` lives on QWERTY's `.` position).
const ANSI_V: CGKeyCode = 0x09;

/// Synthesise `⌘V` directly through `CGEventCreateKeyboardEvent` +
/// `CGEventPost`. Keycode-based (so keymap-correct), and — crucially —
/// **safe to call from any thread**.
///
/// The previous implementation used `enigo`, which calls
/// `TSMGetInputSourceProperty` (Text Services Manager) inside
/// `Enigo::new`. On modern macOS that function asserts main-thread-
/// only via `dispatch_assert_queue`; calling it from the `transcribe`
/// or `streamer-push` worker thread aborts the process with
/// `EXC_BREAKPOINT` / `SIGTRAP`. This dictation app pastes from a
/// worker thread by design (the transcribe path streams chunks),
/// so the TSM assert was a hard, reproducible crash on every paste.
/// `CGEventPost` doesn't go through TSM and has no main-thread
/// requirement.
fn send_paste_chord() -> Result<()> {
    // HIDSystemState (vs CombinedSessionState) posts the events as if
    // they came from a real input device — matches what the user's
    // own keyboard would produce.
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|()| anyhow!("CGEventSource::new failed"))?;
    let keydown = CGEvent::new_keyboard_event(source.clone(), ANSI_V, true)
        .map_err(|()| anyhow!("CGEventCreateKeyboardEvent keydown V failed"))?;
    keydown.set_flags(CGEventFlags::CGEventFlagCommand);
    keydown.post(CGEventTapLocation::HID);
    let keyup = CGEvent::new_keyboard_event(source, ANSI_V, false)
        .map_err(|()| anyhow!("CGEventCreateKeyboardEvent keyup V failed"))?;
    keyup.set_flags(CGEventFlags::CGEventFlagCommand);
    keyup.post(CGEventTapLocation::HID);
    Ok(())
}

#[cfg(test)]
mod tests {
    // Streamer behaviour is exercised via integration: see
    // `polish_streaming_*` tests in `cleanup.rs` plus the manual
    // smoke described in `docs/latency-plan.md`. The clipboard side
    // of `flush_now` is genuinely untestable without a real AppKit
    // pasteboard; we save a no-value unit test rather than pretend
    // to cover it.
}
