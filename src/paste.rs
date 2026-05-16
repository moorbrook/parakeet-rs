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

/// Delay between writing to NSPasteboard and firing the ⌘V chord.
///
/// `pasteboardd` propagates clipboard updates to subscribed apps
/// asynchronously. If our ⌘V hits the focused app's keyDown handler
/// before that propagation lands, the app reads the OLD clipboard —
/// in this app's case, whatever the prior dictation pasted. Old
/// `enigo`-based chord had enough TSM overhead to mask this; the
/// CGEvent-direct rewrite is microseconds and exposed the race.
///
/// 35 ms picked empirically — 20 ms was enough on some target apps
/// but not on slower ones (the user saw the prior dictation pasted
/// in place of the current chunk). Alfred / Raycast use the 30-50 ms
/// range for the same reason.
const PASTEBOARD_SETTLE_DELAY: Duration = Duration::from_millis(35);

/// Delay between the LAST `flush_now` chord and `restore_clipboard`
/// in `Streamer::commit`. Without it, the restore can overwrite the
/// clipboard with the saved (pre-dictation) value BEFORE the focused
/// app has dequeued the queued ⌘V event and read the pasteboard —
/// the app then pastes the saved value (the prior dictation's output)
/// instead of the chunk we just wrote.
///
/// Longer than `PASTEBOARD_SETTLE_DELAY` because the target app's
/// event-dequeue latency can be substantial under load (Slack /
/// Chrome on a slow frame can take >60 ms to process the keyDown).
const RESTORE_SETTLE_DELAY: Duration = Duration::from_millis(120);

pub fn deliver(text: &str, mode: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    match mode {
        "type" => type_text(text),
        "clipboard" => copy_to_clipboard(text),
        _ => {
            // Try direct AX insertion first — no clipboard write, no
            // ⌘V chord, no race. Falls through to clipboard+⌘V if the
            // focused app doesn't expose `AXSelectedText`
            // (Electron-with-custom-input, canvas-based UIs).
            match crate::ax_paste::insert_text(text) {
                Ok(()) => Ok(()),
                Err(e) => {
                    log::debug!("AX insert failed, falling back to clipboard+⌘V: {e:#}");
                    copy_to_clipboard(text)?;
                    std::thread::sleep(PASTEBOARD_SETTLE_DELAY);
                    send_paste_chord()
                }
            }
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
    /// Set to true once a `push` actually inserted text (via either
    /// AX or clipboard+⌘V). Used by `commit` / `abort` to decide
    /// whether the cursor has moved.
    fired: bool,
    /// Set when the clipboard+⌘V fallback path was used at least
    /// once. Drives `RESTORE_SETTLE_DELAY` + `restore_clipboard` in
    /// `commit` / `abort` — both are skipped when AX worked for
    /// every chunk, so the clipboard is untouched and there's
    /// nothing to wait for or restore.
    clipboard_dirty: bool,
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
            clipboard_dirty: false,
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
        // Clipboard restore is only needed if we actually wrote to
        // the clipboard during the stream (i.e. AX fallback fired
        // at least once). When AX worked for every chunk, the
        // user's clipboard is already untouched — nothing to wait
        // for, nothing to restore.
        if self.clipboard_dirty {
            // The focused app may still be dequeuing the last ⌘V;
            // wait before overwriting the clipboard with the saved
            // value, otherwise the app's paste handler reads the
            // saved value (= prior transcript) instead of the chunk
            // we just wrote.
            std::thread::sleep(RESTORE_SETTLE_DELAY);
            if let Some(saved) = self.saved_clipboard.take() {
                let _ = restore_clipboard(&saved);
            }
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
        // Same condition as `commit`: only wait + restore if we
        // actually touched the clipboard. AX-only streams skip both.
        if self.clipboard_dirty {
            std::thread::sleep(RESTORE_SETTLE_DELAY);
            if let Some(saved) = self.saved_clipboard.take() {
                let _ = restore_clipboard(&saved);
            }
        }
    }

    fn flush_now(&mut self) -> Result<()> {
        let chunk: String = std::mem::take(&mut self.pending);
        // Try AX insert first — no clipboard write, no ⌘V chord, no
        // race against pasteboardd / the focused app's keyDown
        // handler. Falls back to clipboard+⌘V if AX fails (most
        // common when the focused element doesn't support
        // `AXSelectedText` — Electron-with-custom-input, canvas
        // editors). Mixing modes per chunk is safe: AX inserts at
        // the caret, ⌘V also inserts at the caret, so chunks land
        // in order regardless of how each one was delivered.
        if let Err(e) = crate::ax_paste::insert_text(&chunk) {
            log::debug!("AX insert failed, falling back to clipboard+⌘V: {e:#}");
            copy_to_clipboard(&chunk)?;
            std::thread::sleep(PASTEBOARD_SETTLE_DELAY);
            send_paste_chord()?;
            self.clipboard_dirty = true;
        }
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
