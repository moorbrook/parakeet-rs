//! Global hotkey registration via the `global-hotkey` crate, which under
//! the hood calls `RegisterEventHotKey` on macOS. We expose both press and
//! release callbacks so the app can implement Hold-to-dictate UX where
//! `release == paste`.
//!
//! Caps Lock is intentionally NOT supported here: macOS surfaces it as a
//! sticky toggle through Carbon, so `global-hotkey` can't see momentary
//! down/up events. Real momentary Caps Lock would need a `CGEventTap`;
//! that's deferred to a follow-up.

use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use parking_lot::Mutex;

/// Long-lived hotkey registration. Drop the handle to unregister.
pub struct HotkeyHandle {
    manager: GlobalHotKeyManager,
    current: Mutex<Option<HotKey>>,
}

impl HotkeyHandle {
    /// Swap the active hotkey out for a new one. Used by the Settings UI
    /// after the user records a new combo.
    pub fn rebind(&self, new_spec: &str) -> Result<()> {
        let new_hk = parse(new_spec).with_context(|| format!("parsing hotkey: {new_spec}"))?;
        let mut slot = self.current.lock();
        if let Some(old) = slot.take() {
            let _ = self.manager.unregister(old);
        }
        self.manager
            .register(new_hk)
            .context("registering new hotkey")?;
        *slot = Some(new_hk);
        Ok(())
    }
}

/// Callbacks fired from a background thread when the hotkey press/release
/// edge crosses. Wrapped in `Arc<dyn Fn(...)>` so they can be cloned cheaply
/// into the polling thread.
pub type EventFn = Arc<dyn Fn() + Send + Sync + 'static>;

static PUMP_STARTED: OnceLock<()> = OnceLock::new();

/// Register the hotkey and start the background event-pump thread. Returns
/// a handle that owns the manager; keep it alive for the lifetime of the
/// program (or call `rebind` to change the hotkey on the fly).
pub fn register(
    spec: &str,
    on_press: EventFn,
    on_release: EventFn,
) -> Result<HotkeyHandle> {
    let manager = GlobalHotKeyManager::new().context("creating GlobalHotKeyManager")?;
    let hotkey = parse(spec).with_context(|| format!("parsing hotkey: {spec}"))?;
    manager.register(hotkey).context("registering hotkey")?;

    // Spawn the event-pump exactly once per process — `GlobalHotKeyEvent::receiver()`
    // returns a singleton channel that's safe to read from one thread only.
    if PUMP_STARTED.set(()).is_ok() {
        let receiver = GlobalHotKeyEvent::receiver();
        thread::Builder::new()
            .name("hotkey-pump".into())
            .spawn(move || loop {
                while let Ok(event) = receiver.try_recv() {
                    match event.state {
                        HotKeyState::Pressed => on_press(),
                        HotKeyState::Released => on_release(),
                    }
                }
                // 25 ms tick keeps the thread responsive without burning a
                // core. The Carbon RegisterEventHotKey hook lands events on
                // the main thread; the receiver channel relays them.
                thread::sleep(Duration::from_millis(25));
            })
            .context("spawning hotkey pump")?;
    }

    Ok(HotkeyHandle {
        manager,
        current: Mutex::new(Some(hotkey)),
    })
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
                    "f1" => Code::F1, "f2" => Code::F2, "f3" => Code::F3,
                    "f4" => Code::F4, "f5" => Code::F5, "f6" => Code::F6,
                    "f7" => Code::F7, "f8" => Code::F8, "f9" => Code::F9,
                    "f10" => Code::F10, "f11" => Code::F11, "f12" => Code::F12,
                    "f13" => Code::F13, "f14" => Code::F14, "f15" => Code::F15,
                    "f16" => Code::F16, "f17" => Code::F17, "f18" => Code::F18,
                    "f19" => Code::F19,
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
