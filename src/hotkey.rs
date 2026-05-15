//! Global hotkey registration via the `global-hotkey` crate, which under the
//! hood calls `RegisterEventHotKey` on macOS — the same Carbon API Tauri's
//! plugin used. The wrapper here just parses our `"CmdOrCtrl+Shift+Space"`
//! token string, keeps the manager alive for the lifetime of the app, and
//! pumps events into a Rust closure on a background thread.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

pub struct HotkeyHandle {
    _manager: GlobalHotKeyManager,
}

/// Parse a token of the form `CmdOrCtrl+Shift+Space` into a `HotKey`. Keep
/// in lock-step with the strings the menu/glyph rendering produces.
fn parse(spec: &str) -> Result<HotKey> {
    let mut mods = Modifiers::empty();
    let mut code: Option<Code> = None;
    for raw in spec.split('+').map(str::trim) {
        match raw.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "cmdorctrl" | "commandorcontrol" | "super" | "meta" => {
                mods |= Modifiers::META
            }
            "ctrl" | "control" => mods |= Modifiers::CONTROL,
            "alt" | "option" => mods |= Modifiers::ALT,
            "shift" => mods |= Modifiers::SHIFT,
            other => {
                code = Some(match other {
                    "space" => Code::Space,
                    "enter" | "return" => Code::Enter,
                    "tab" => Code::Tab,
                    "esc" | "escape" => Code::Escape,
                    "backspace" => Code::Backspace,
                    s if s.len() == 1 => match s.chars().next().unwrap().to_ascii_lowercase() {
                        'a' => Code::KeyA, 'b' => Code::KeyB, 'c' => Code::KeyC,
                        'd' => Code::KeyD, 'e' => Code::KeyE, 'f' => Code::KeyF,
                        'g' => Code::KeyG, 'h' => Code::KeyH, 'i' => Code::KeyI,
                        'j' => Code::KeyJ, 'k' => Code::KeyK, 'l' => Code::KeyL,
                        'm' => Code::KeyM, 'n' => Code::KeyN, 'o' => Code::KeyO,
                        'p' => Code::KeyP, 'q' => Code::KeyQ, 'r' => Code::KeyR,
                        's' => Code::KeyS, 't' => Code::KeyT, 'u' => Code::KeyU,
                        'v' => Code::KeyV, 'w' => Code::KeyW, 'x' => Code::KeyX,
                        'y' => Code::KeyY, 'z' => Code::KeyZ,
                        '0' => Code::Digit0, '1' => Code::Digit1, '2' => Code::Digit2,
                        '3' => Code::Digit3, '4' => Code::Digit4, '5' => Code::Digit5,
                        '6' => Code::Digit6, '7' => Code::Digit7, '8' => Code::Digit8,
                        '9' => Code::Digit9,
                        c => return Err(anyhow!("unsupported single-char key: {c}")),
                    },
                    other => return Err(anyhow!("unsupported key token: {other}")),
                });
            }
        }
    }
    let code = code.ok_or_else(|| anyhow!("hotkey missing a key"))?;
    Ok(HotKey::new(Some(mods), code))
}

/// Register the hotkey and spawn a background polling thread that calls
/// `on_press` whenever it fires. The returned handle owns the manager; drop
/// it (or just keep it alive for the program) to unregister.
pub fn register(
    spec: &str,
    on_press: Arc<dyn Fn() + Send + Sync + 'static>,
) -> Result<HotkeyHandle> {
    let manager = GlobalHotKeyManager::new().context("creating GlobalHotKeyManager")?;
    let hotkey = parse(spec).with_context(|| format!("parsing hotkey: {spec}"))?;
    manager.register(hotkey).context("registering hotkey")?;

    let receiver = GlobalHotKeyEvent::receiver();
    thread::Builder::new()
        .name("hotkey-pump".into())
        .spawn(move || loop {
            // try_recv + a short sleep keeps the thread responsive without
            // burning a core. The Carbon RegisterEventHotKey hook lands
            // events on the main thread; the receiver channel just relays
            // them, so a 25 ms tick latency is invisible to the user.
            while let Ok(event) = receiver.try_recv() {
                if event.state == HotKeyState::Pressed {
                    on_press();
                }
            }
            thread::sleep(Duration::from_millis(25));
        })
        .context("spawning hotkey pump")?;

    Ok(HotkeyHandle { _manager: manager })
}
