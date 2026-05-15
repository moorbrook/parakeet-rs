//! Deliver a finalised transcript to the focused window.
//!
//! Default mode is "paste": write the text to the system clipboard via
//! `NSPasteboard` (through `arboard`), then synthesise a ⌘V keypress via
//! `enigo` so the focused app pastes it. Two other modes available for
//! debugging: `"type"` types the string keystroke-by-keystroke (slow), and
//! `"clipboard"` only writes to the pasteboard without pressing ⌘V.

use anyhow::{Context, Result};
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

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

fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut cb = Clipboard::new().context("creating NSPasteboard handle")?;
    cb.set_text(text.to_string())
        .context("writing to clipboard")?;
    Ok(())
}

fn type_text(text: &str) -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default()).context("init enigo")?;
    enigo.text(text).context("typing text")?;
    Ok(())
}

fn send_paste_chord() -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default()).context("init enigo")?;
    enigo.key(Key::Meta, Direction::Press).context("⌘ down")?;
    enigo
        .key(Key::Unicode('v'), Direction::Click)
        .context("press v")?;
    enigo.key(Key::Meta, Direction::Release).context("⌘ up")?;
    Ok(())
}
