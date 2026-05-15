use anyhow::{Context, Result};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use tauri::AppHandle;
use tauri_plugin_clipboard_manager::ClipboardExt;

/// Deliver transcribed `text` to the user according to `mode`.
///
/// Modes:
/// - "paste":     write to clipboard then synthesize Cmd/Ctrl+V
/// - "type":      type each character via enigo (slow, but no clipboard touch)
/// - "clipboard": write to clipboard only
pub fn deliver(app: &AppHandle, text: &str, mode: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    match mode {
        "type" => type_text(text),
        "clipboard" => copy_to_clipboard(app, text),
        _ => {
            copy_to_clipboard(app, text)?;
            send_paste_chord()
        }
    }
}

fn copy_to_clipboard(app: &AppHandle, text: &str) -> Result<()> {
    app.clipboard()
        .write_text(text.to_string())
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
