//! Global hotkey detection. Two sources cover all the keys we care about:
//!
//! 1. **`CGEventTap` at the HID level** sees every regular keyboard event
//!    (`KeyDown` / `KeyUp` / `FlagsChanged`) with full left/right modifier
//!    discrimination. Handles chord hotkeys like `⌘⇧Space` and Caps Lock.
//! 2. **`NSEvent.addGlobalMonitorForEventsMatchingMask:handler:`** sees
//!    system-defined events (`NSEventType::SystemDefined`, subtype 8 =
//!    `AuxControlButtons`). Handles media keys like Eject, Play, Vol+/Vol-,
//!    Brightness, which never appear in CGEvent KeyDown streams.
//!
//! **Permissions.** macOS 10.15+ requires the **Input Monitoring** TCC
//! permission for HID-level taps *and* global NSEvent monitors. The
//! Accessibility permission we already grab for the ⌘V paste chord is *not*
//! sufficient. On first launch the user is prompted to allow Parakeet in
//! System Settings → Privacy & Security → Input Monitoring.

use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;

// Public CoreGraphics APIs (macOS 10.15+) for checking and requesting
// the Input Monitoring permission. Not exposed by the `core-graphics`
// crate, so we declare them directly.
unsafe extern "C" {
    /// Returns true if the calling process is allowed to listen to events
    /// via `CGEventTap`. Equivalent to checking the Input Monitoring TCC
    /// permission.
    fn CGPreflightListenEventAccess() -> bool;
    /// Triggers the Input Monitoring permission dialog if not already
    /// granted. Returns the current state. The user's decision is
    /// asynchronous — they grant it in System Settings, then must restart
    /// the app for the tap to actually start delivering events.
    fn CGRequestListenEventAccess() -> bool;
}

use anyhow::{anyhow, Context, Result};
use block2::RcBlock;
use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
use core_graphics::event::{
    CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement,
    CGEventType, CGKeyCode, EventField, KeyCode,
};
use objc2_app_kit::{NSEvent, NSEventMask, NSEventSubtype, NSEventType};
use objc2_foundation::MainThreadMarker;
use parking_lot::Mutex;

/// Closure type that the tap fires on press/release edges.
pub type EventFn = Arc<dyn Fn() + Send + Sync + 'static>;

/// The hotkey can be three shapes today:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// Standard chord — modifier(s) + a non-modifier key, e.g. `⌘⇧Space`.
    Chord {
        required_mods: CGEventFlags,
        main_key: CGKeyCode,
    },
    /// Caps Lock as a dictation toggle. Each physical tap of the key flips
    /// the AlphaShift bit and produces a single `FlagsChanged` event; we
    /// alternate `on_press` / `on_release` on consecutive taps and swallow
    /// the event so the system's Caps Lock state never actually toggles.
    /// (True press-and-hold momentary on Caps Lock would need IOHIDManager;
    /// the toggle path here matches how Wispr Flow and SuperWhisper do it.)
    CapsLockToggle,
    /// The Eject key. Reaches us via the `NSEvent` system-defined monitor
    /// (subtype 8 = `AUX_CONTROL_BUTTONS`, key type `NX_KEYTYPE_EJECT = 14`).
    /// Other media keys (Play, Next, Volume, etc.) intentionally aren't
    /// supported — Parakeet has no use for them as dictation triggers, and
    /// binding them would steal a system function the user actually wants.
    Eject,
}

pub struct HotkeyHandle {
    binding: Arc<Mutex<Binding>>,
}

impl HotkeyHandle {
    /// Swap the active hotkey for a new one. Both the CGEventTap thread and
    /// the NSEvent monitor read the binding through this shared `Arc<Mutex>`
    /// on every event, so the change takes effect on the next keypress.
    pub fn rebind(&self, new_spec: &str) -> Result<()> {
        let new = parse(new_spec).with_context(|| format!("parsing hotkey: {new_spec}"))?;
        *self.binding.lock() = new;
        Ok(())
    }
}

static TAP_STARTED: OnceLock<()> = OnceLock::new();
static NS_MONITOR_INSTALLED: OnceLock<()> = OnceLock::new();
// The NSEvent monitor handle returned by AppKit isn't `Send` (it's a raw
// objc reference to a NSCFType), so it can't live in a static. AppKit
// holds its own strong reference via the monitor's internal list — so as
// long as we don't drop our handle, the monitor stays installed. We
// `std::mem::forget` it after install so it stays alive for the program's
// lifetime without needing a static.

/// Install both detectors and return a handle. Subsequent calls in the same
/// process replace the binding rather than spawning new detectors. The
/// `MainThreadMarker` is required because `NSEvent::addGlobalMonitorFor…`
/// must be invoked on the main thread.
pub fn register(
    spec: &str,
    on_press: EventFn,
    on_release: EventFn,
    mtm: MainThreadMarker,
) -> Result<HotkeyHandle> {
    let initial = parse(spec).with_context(|| format!("parsing hotkey: {spec}"))?;
    let binding = Arc::new(Mutex::new(initial));

    // ---------- 0. Input Monitoring preflight ----------
    // CGEventTapCreate returns a "valid" mach port even without Input
    // Monitoring permission, but the tap never delivers events. Check up
    // front so we can surface a clear log line and trigger the system
    // prompt instead of failing silently.
    let granted = unsafe { CGPreflightListenEventAccess() };
    if !granted {
        log::warn!(
            "Input Monitoring permission not yet granted. Requesting it now — \
             macOS will open System Settings → Privacy & Security → Input \
             Monitoring. Enable Parakeet there and relaunch the app for the \
             hotkey to start working."
        );
        let _ = unsafe { CGRequestListenEventAccess() };
        // The request returns immediately; the user has to grant it in
        // System Settings and then relaunch. We continue starting the
        // detectors anyway — they'll be inert until permission lands and
        // the app restarts.
    } else {
        log::info!("Input Monitoring permission granted");
    }

    // ---------- 1. CGEventTap thread ----------
    if TAP_STARTED.set(()).is_ok() {
        let binding_for_tap = binding.clone();
        let on_press_for_tap = on_press.clone();
        let on_release_for_tap = on_release.clone();
        thread::Builder::new()
            .name("hotkey-tap".into())
            .spawn(move || {
                crate::qos::set_user_interactive();
                run_tap(binding_for_tap, on_press_for_tap, on_release_for_tap);
            })
            .context("spawn hotkey-tap thread")?;
    }

    // ---------- 2. NSEvent system-defined monitor (media keys) ----------
    if NS_MONITOR_INSTALLED.set(()).is_ok() {
        install_media_key_monitor(mtm, binding.clone(), on_press, on_release)?;
    }

    Ok(HotkeyHandle { binding })
}

/// CGEventTap callback runs on its own dedicated thread (we spawn it). The
/// callback fires the user-supplied closures on each press / release edge.
fn run_tap(binding: Arc<Mutex<Binding>>, on_press: EventFn, on_release: EventFn) {
    // Edge-detection state for chord hotkeys. CGEventTap delivers a KeyDown
    // for every auto-repeat tick; we only want the initial press.
    let chord_held = Cell::new(false);
    // Toggle state for Caps Lock. Each FlagsChanged event on Caps Lock
    // alternates press → release → press → release.
    let caps_held = Cell::new(false);

    let tap = CGEventTap::new(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        // `ListenOnly` so the tap never modifies or swallows events.
        // `Default` mode (which would let us suppress Caps Lock's
        // AlphaShift toggle) trips a stricter TCC pathway on macOS that
        // silently drops the tap into "permission pending" even when
        // Input Monitoring is allowed — observed empirically: no events
        // delivered at all. We accept the Caps Lock state toggling for
        // now in exchange for the tap actually working.
        CGEventTapOptions::ListenOnly,
        vec![
            CGEventType::KeyDown,
            CGEventType::KeyUp,
            CGEventType::FlagsChanged,
        ],
        move |_proxy, event_type, event| {
            let bind = *binding.lock();
            let flags = event.get_flags();
            let keycode =
                event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE) as CGKeyCode;

            // Trace every keyboard event the tap sees. Useful for
            // diagnosing "I pressed X but nothing happened" — e.g. F-keys
            // being intercepted as media keys by the system. Turn on with
            // RUST_LOG=parakeet_rs::hotkey=trace.
            log::trace!(
                "tap event: type={:?} keycode={} flags={:#010x}",
                event_type,
                keycode,
                flags.bits()
            );

            match bind {
                Binding::Chord {
                    required_mods,
                    main_key,
                } => {
                    match event_type {
                        CGEventType::KeyDown
                            if keycode == main_key
                                && mods_match(flags, required_mods)
                                && !chord_held.get() =>
                        {
                            chord_held.set(true);
                            on_press();
                        }
                        CGEventType::KeyUp if keycode == main_key && chord_held.get() => {
                            chord_held.set(false);
                            on_release();
                        }
                        _ => {}
                    }
                    Some(event.clone())
                }
                Binding::CapsLockToggle => {
                    if matches!(event_type, CGEventType::FlagsChanged)
                        && keycode == KeyCode::CAPS_LOCK
                    {
                        if caps_held.get() {
                            caps_held.set(false);
                            on_release();
                        } else {
                            caps_held.set(true);
                            on_press();
                        }
                        // In `ListenOnly` mode the return value is ignored
                        // and the AlphaShift state still toggles. Documented
                        // limitation; revisit when we figure out how to
                        // ask for the stricter TCC capability needed for
                        // `Default` mode tap-modify.
                        Some(event.clone())
                    } else {
                        Some(event.clone())
                    }
                }
                Binding::Eject => Some(event.clone()),
            }
        },
    );

    let tap = match tap {
        Ok(t) => t,
        Err(_) => {
            log::error!(
                "CGEventTap::new failed — usually means the Input Monitoring \
                 permission hasn't been granted. Open System Settings → \
                 Privacy & Security → Input Monitoring and enable Parakeet, \
                 then relaunch."
            );
            return;
        }
    };

    let loop_source = match tap.mach_port.create_runloop_source(0) {
        Ok(s) => s,
        Err(_) => {
            log::error!("create_runloop_source failed; hotkey detector inactive");
            return;
        }
    };
    let current = CFRunLoop::get_current();
    unsafe { current.add_source(&loop_source, kCFRunLoopCommonModes) };
    tap.enable();
    // Re-check Input Monitoring here so the line we log reflects whether the
    // tap will ACTUALLY deliver events, not just whether create() succeeded
    // (which it does even without permission).
    if unsafe { CGPreflightListenEventAccess() } {
        log::info!("hotkey tap active (CGEventTap at HID level, Input Monitoring granted)");
    } else {
        log::warn!(
            "hotkey tap created but Input Monitoring is not yet granted — events \
             will NOT be delivered. Grant permission in System Settings → Privacy \
             & Security → Input Monitoring and relaunch."
        );
    }
    CFRunLoop::run_current(); // blocks forever
}

/// NSEvent global monitor for media keys. Runs callbacks on the main thread.
fn install_media_key_monitor(
    _mtm: MainThreadMarker,
    binding: Arc<Mutex<Binding>>,
    on_press: EventFn,
    on_release: EventFn,
) -> Result<()> {
    // Per-monitor edge-detection state for media key auto-repeat.
    let media_held = Arc::new(Mutex::new(false));

    let media_held_for_block = media_held.clone();
    let block = RcBlock::new(move |event: NonNull<NSEvent>| {
        let event: &NSEvent = unsafe { event.as_ref() };
        if unsafe { event.r#type() } != NSEventType::SystemDefined {
            return;
        }
        // Subtype 8 (= NSEventSubtype::ScreenChanged numerically; the same
        // integer value is `NX_SUBTYPE_AUX_CONTROL_BUTTONS` when the event
        // is SystemDefined — Apple's constants are overloaded by integer).
        if unsafe { event.subtype() } != NSEventSubtype::ScreenChanged {
            return;
        }
        let data1 = unsafe { event.data1() };
        let keytype = ((data1 & 0xFFFF_0000) >> 16) as i32;
        let keystate = ((data1 & 0xFF00) >> 8) as i32;
        let is_down = keystate == 0x0A; // NX_KEYDOWN

        let bind = *binding.lock();
        // Only the Eject binding cares about system-defined events today.
        if !matches!(bind, Binding::Eject) {
            return;
        }
        // NX_KEYTYPE_EJECT = 14
        if keytype != 14 {
            return;
        }

        let mut held = media_held_for_block.lock();
        if is_down && !*held {
            *held = true;
            on_press();
        } else if !is_down && *held {
            *held = false;
            on_release();
        }
    });

    // SAFETY: The block is `'static` (closure captures only owned `Arc`s).
    // AppKit retains the block internally, so it lives as long as the
    // monitor is registered. We hold the returned handle to keep the
    // monitor alive for the program's life.
    let handle = unsafe {
        NSEvent::addGlobalMonitorForEventsMatchingMask_handler(NSEventMask::SystemDefined, &block)
    };
    let Some(handle) = handle else {
        log::warn!(
            "NSEvent global monitor install returned nil — Input Monitoring \
             permission probably not granted yet. Media-key bindings will be inert."
        );
        return Ok(());
    };

    // RAII: store the handle in a process-lifetime `OnceLock` so the
    // monitor lives for the program's life AND the handle gets
    // `removeMonitor:`-ed if the slot is ever taken out. Previously
    // this was `std::mem::forget(handle)` — clean from a leak-tools
    // perspective today, but blocks any future "rebind the media-key
    // monitor without process restart" use case.
    if MEDIA_KEY_MONITOR.set(MediaKeyMonitor { token: handle }).is_err() {
        log::warn!("media-key monitor already installed — ignoring duplicate install");
    } else {
        log::info!("media-key monitor active (NSEvent global, system-defined)");
    }
    Ok(())
}

/// RAII wrapper for the NSEvent global monitor token. On drop, sends
/// `+[NSEvent removeMonitor:]` so the system stops dispatching events
/// to our block. We keep one of these alive for the life of the
/// process via the `MEDIA_KEY_MONITOR` `OnceLock`.
struct MediaKeyMonitor {
    token: Retained<AnyObject>,
}

impl Drop for MediaKeyMonitor {
    fn drop(&mut self) {
        // `NSEvent::removeMonitor` returns void and takes an `id` —
        // safe to call with our retained token. AppKit handles its
        // own internal teardown of the block at this point.
        unsafe {
            NSEvent::removeMonitor(&self.token);
        }
    }
}

// SAFETY: `Retained<AnyObject>` isn't `Send + Sync` because Cocoa
// object access generally needs care across threads, but our specific
// access pattern is single-threaded: this token is *set* exactly
// once (on the main thread, from `install_media_key_monitor`), is
// only *read* by `Drop` (which only runs when the `OnceLock` itself
// is being torn down, which doesn't happen — the static lives for
// the program's life), and is never aliased or mutated thereafter.
// `removeMonitor:` itself is documented as thread-safe.
unsafe impl Send for MediaKeyMonitor {}
unsafe impl Sync for MediaKeyMonitor {}

static MEDIA_KEY_MONITOR: std::sync::OnceLock<MediaKeyMonitor> = std::sync::OnceLock::new();

/// Side-agnostic modifier match. The required modifier bits must all be set
/// on the event; AlphaShift (Caps Lock state), NumericPad, Help, SecondaryFn
/// and the NonCoalesced bit don't disqualify a chord.
fn mods_match(actual: CGEventFlags, required: CGEventFlags) -> bool {
    let mod_mask = CGEventFlags::CGEventFlagShift
        | CGEventFlags::CGEventFlagControl
        | CGEventFlags::CGEventFlagAlternate
        | CGEventFlags::CGEventFlagCommand;
    (actual & mod_mask) == (required & mod_mask)
}

/// Parse a token of the form `CmdOrCtrl+Shift+Space`, or a single bare key
/// name like `CapsLock`, `Eject`, `F5`, `Space`.
pub fn parse(spec: &str) -> Result<Binding> {
    let trimmed = spec.trim();
    if !trimmed.contains('+') {
        match trimmed.to_ascii_lowercase().as_str() {
            "capslock" | "caps_lock" | "caps-lock" => return Ok(Binding::CapsLockToggle),
            "eject" => return Ok(Binding::Eject),
            _ => {} // fall through to chord parser
        }
    }

    let mut required_mods = CGEventFlags::empty();
    let mut main_key: Option<CGKeyCode> = None;
    for raw in trimmed.split('+').map(str::trim) {
        match raw.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "cmdorctrl" | "commandorcontrol" | "super" | "meta" => {
                required_mods |= CGEventFlags::CGEventFlagCommand
            }
            "ctrl" | "control" => required_mods |= CGEventFlags::CGEventFlagControl,
            "alt" | "option" => required_mods |= CGEventFlags::CGEventFlagAlternate,
            "shift" => required_mods |= CGEventFlags::CGEventFlagShift,
            other => {
                main_key = Some(parse_key(other)?);
            }
        }
    }
    let main_key =
        main_key.ok_or_else(|| anyhow!("hotkey missing a key (no non-modifier component)"))?;
    Ok(Binding::Chord {
        required_mods,
        main_key,
    })
}

fn parse_key(s: &str) -> Result<CGKeyCode> {
    Ok(match s {
        "space" => KeyCode::SPACE,
        "enter" | "return" => KeyCode::RETURN,
        "tab" => KeyCode::TAB,
        "esc" | "escape" => KeyCode::ESCAPE,
        "backspace" | "delete" => KeyCode::DELETE,
        "f1" => KeyCode::F1,
        "f2" => KeyCode::F2,
        "f3" => KeyCode::F3,
        "f4" => KeyCode::F4,
        "f5" => KeyCode::F5,
        "f6" => KeyCode::F6,
        "f7" => KeyCode::F7,
        "f8" => KeyCode::F8,
        "f9" => KeyCode::F9,
        "f10" => KeyCode::F10,
        "f11" => KeyCode::F11,
        "f12" => KeyCode::F12,
        "f13" => KeyCode::F13,
        "f14" => KeyCode::F14,
        "f15" => KeyCode::F15,
        "f16" => KeyCode::F16,
        "f17" => KeyCode::F17,
        "f18" => KeyCode::F18,
        "f19" => KeyCode::F19,
        "f20" => KeyCode::F20,
        // Bare letter / digit / punctuation keys map to Carbon virtual
        // keycodes from HIToolbox/Events.h. Inline since core-graphics
        // `KeyCode` only exposes constants for special keys.
        "a" => 0x00,
        "s" => 0x01,
        "d" => 0x02,
        "f" => 0x03,
        "h" => 0x04,
        "g" => 0x05,
        "z" => 0x06,
        "x" => 0x07,
        "c" => 0x08,
        "v" => 0x09,
        "b" => 0x0B,
        "q" => 0x0C,
        "w" => 0x0D,
        "e" => 0x0E,
        "r" => 0x0F,
        "y" => 0x10,
        "t" => 0x11,
        "1" => 0x12,
        "2" => 0x13,
        "3" => 0x14,
        "4" => 0x15,
        "6" => 0x16,
        "5" => 0x17,
        "=" => 0x18,
        "9" => 0x19,
        "7" => 0x1A,
        "-" => 0x1B,
        "8" => 0x1C,
        "0" => 0x1D,
        "]" => 0x1E,
        "o" => 0x1F,
        "u" => 0x20,
        "[" => 0x21,
        "i" => 0x22,
        "p" => 0x23,
        "l" => 0x25,
        "j" => 0x26,
        "'" => 0x27,
        "k" => 0x28,
        ";" => 0x29,
        "\\" => 0x2A,
        "," => 0x2B,
        "/" => 0x2C,
        "n" => 0x2D,
        "m" => 0x2E,
        "." => 0x2F,
        "`" => 0x32,
        other => return Err(anyhow!("unsupported key token: {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_chord_with_modifiers() {
        let b = parse("CmdOrCtrl+Shift+Space").unwrap();
        let Binding::Chord {
            required_mods,
            main_key,
        } = b
        else {
            panic!("expected Chord, got {b:?}");
        };
        assert_eq!(main_key, KeyCode::SPACE);
        assert!(required_mods.contains(CGEventFlags::CGEventFlagCommand));
        assert!(required_mods.contains(CGEventFlags::CGEventFlagShift));
        assert!(!required_mods.contains(CGEventFlags::CGEventFlagControl));
    }

    #[test]
    fn parse_bare_function_key() {
        let b = parse("F5").unwrap();
        let Binding::Chord {
            required_mods,
            main_key,
        } = b
        else {
            panic!("expected Chord");
        };
        assert_eq!(main_key, KeyCode::F5);
        assert!(required_mods.is_empty());
    }

    #[test]
    fn parse_caps_lock() {
        assert!(matches!(
            parse("CapsLock").unwrap(),
            Binding::CapsLockToggle
        ));
        assert!(matches!(
            parse("caps_lock").unwrap(),
            Binding::CapsLockToggle
        ));
    }

    #[test]
    fn parse_eject() {
        assert!(matches!(parse("Eject").unwrap(), Binding::Eject));
        assert!(matches!(parse("eject").unwrap(), Binding::Eject));
    }

    #[test]
    fn parse_rejects_modifier_only_chord() {
        // "Shift" alone has no main key; should error rather than silently
        // produce a useless binding.
        assert!(parse("Shift").is_err());
        assert!(parse("Shift+Ctrl").is_err());
    }

    #[test]
    fn mods_match_is_side_agnostic_and_ignores_capslock() {
        let required = CGEventFlags::CGEventFlagCommand | CGEventFlags::CGEventFlagShift;
        // Cmd + Shift, no AlphaShift — should match.
        assert!(mods_match(required, required));
        // Same Cmd + Shift but with Caps Lock state set — irrelevant, should match.
        assert!(mods_match(
            required | CGEventFlags::CGEventFlagAlphaShift,
            required
        ));
        // Missing Shift — should NOT match.
        assert!(!mods_match(CGEventFlags::CGEventFlagCommand, required));
        // Extra Ctrl — should NOT match.
        assert!(!mods_match(
            required | CGEventFlags::CGEventFlagControl,
            required
        ));
    }
}
