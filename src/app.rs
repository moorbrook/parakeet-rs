//! Top-level coordinator. Owns the recogniser, the active dictation session,
//! the settings store, and the menubar/hotkey wiring. Talks to AppKit only
//! through `crate::menubar` and `crate::settings_ui`; everything else is
//! plain Rust.
//!
//! There is exactly one `App` per process. It lives behind an `Arc` inside
//! the `AppHandle` global so the AppKit menu-action selectors (which can't
//! carry Rust state through the Objective-C runtime) can reach back into it.

use std::sync::Arc;
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::asr::Asr;
use crate::cleanup;
use crate::hotkey::HotkeyHandle;
use crate::hud;
use crate::menubar;
use crate::model_fetch::{self, Progress};
use crate::settings::{CleanupMode, Settings, SettingsStore, TriggerMode};
use crate::streamer::{self, Mode as StreamerMode, Outcome, OutcomeRx, Session};
use crate::{paste, performance, warmup};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DictationState {
    /// Model still downloading / loading.
    ModelLoading,
    /// Ready, no active session.
    Idle,
    /// Currently capturing audio.
    Listening,
    /// Capture stopped, ASR running.
    Transcribing,
    /// ASR done; LLM cleanup pass running.
    Polishing,
}

pub struct App {
    pub session: Mutex<Option<Session>>,
    pub settings: SettingsStore,
    /// Set once the model is downloaded and the recogniser is warm.
    pub asr: Mutex<Option<Arc<Asr>>>,
    pub current_state: Mutex<DictationState>,
    /// Lives for the program's lifetime; held here so the Settings UI can
    /// rebind it after the user records a new hotkey combo.
    pub hotkey: Mutex<Option<HotkeyHandle>>,
}

impl App {
    pub fn new(settings: SettingsStore) -> Self {
        Self {
            session: Mutex::new(None),
            settings,
            asr: Mutex::new(None),
            current_state: Mutex::new(DictationState::ModelLoading),
            hotkey: Mutex::new(None),
        }
    }

    /// Hotkey-press edge. Behaviour depends on the configured TriggerMode:
    /// - Tap mode: toggle (start a session, or cancel one in flight).
    /// - Hold mode: always start a session. Release is the commit edge.
    pub fn on_hotkey_press(self: &Arc<Self>) {
        let mode = effective_trigger_mode(&self.settings.load());
        let active = self.session.lock().is_some();
        match (mode, active) {
            (TriggerMode::Tap, true) => {
                if let Some(s) = self.session.lock().as_ref() {
                    s.cancel();
                }
            }
            (TriggerMode::Tap, false) => self.start_session(StreamerMode::VadAutoStop),
            (TriggerMode::Hold, true) => {
                // Spurious second press while already recording — ignore;
                // the user is still holding the key from the first edge.
            }
            (TriggerMode::Hold, false) => self.start_session(StreamerMode::Manual),
        }
    }

    /// Hotkey-release edge. Only meaningful in Hold mode; in Tap mode the
    /// auto-key-repeat or release noise doesn't change anything.
    pub fn on_hotkey_release(self: &Arc<Self>) {
        if effective_trigger_mode(&self.settings.load()) != TriggerMode::Hold {
            return;
        }
        if let Some(s) = self.session.lock().as_ref() {
            s.finalize();
        }
    }

    fn start_session(self: &Arc<Self>, mode: StreamerMode) {
        if self.asr.lock().is_none() {
            log::warn!("hotkey ignored: model still loading");
            return;
        }
        let vad_path = self.settings.vad_path();
        let (session, outcome_rx) = match streamer::start(&vad_path, mode) {
            Ok(pair) => pair,
            Err(e) => {
                log::error!("start session failed: {e:#}");
                return;
            }
        };
        self.set_state(DictationState::Listening);

        // Park the command half in app.session for the lifetime of the
        // recording. on_hotkey_release / on_hotkey_press reach the active
        // session through here. The watcher owns only the outcome
        // receiver — it can't accidentally interfere with the command
        // path.
        {
            let mut slot = self.session.lock();
            let prev = slot.replace(session);
            debug_assert!(prev.is_none(), "session slot was not cleared");
        }

        let app = self.clone();
        std::thread::Builder::new()
            .name("session-watcher".into())
            .spawn(move || {
                let OutcomeRx(rx) = outcome_rx;
                let outcome = rx.recv().ok();
                // Drop the command half NOW that the recording is done.
                // This both shuts down the capture thread (Drop sends a
                // Cancel) and clears `app.session` so the next hotkey
                // press sees an idle slot.
                drop(app.session.lock().take());
                match outcome {
                    Some(Outcome::Speech {
                        samples,
                        sample_rate,
                    }) => {
                        app.set_state(DictationState::Transcribing);
                        app.transcribe_and_paste(samples, sample_rate);
                    }
                    Some(Outcome::Cancelled) => app.on_session_finished(),
                    Some(Outcome::NoSpeech) => {
                        log::info!("VAD heard nothing; aborting session");
                        app.on_session_finished();
                    }
                    Some(Outcome::Error(e)) => {
                        log::error!("session error: {e:#}");
                        app.on_session_finished();
                    }
                    None => app.on_session_finished(),
                }
            })
            .expect("spawn session-watcher thread");
    }

    fn transcribe_and_paste(self: &Arc<Self>, samples: Vec<f32>, sample_rate: u32) {
        let app = self.clone();
        std::thread::Builder::new()
            .name("transcribe".into())
            .spawn(move || {
                let asr = match app.asr.lock().clone() {
                    Some(a) => a,
                    None => {
                        log::error!("transcribe: model gone");
                        app.on_session_finished();
                        return;
                    }
                };
                let raw = match asr.recognize(&samples, sample_rate) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!("recognise failed: {e:#}");
                        app.on_session_finished();
                        return;
                    }
                };

                // Run the cleanup pass if enabled. Failure is non-fatal
                // — the worst case is "user gets the raw transcript",
                // which is the same as if cleanup were off. Surface the
                // error in the menu bar but keep going.
                let settings = app.settings.load();
                let cleaned = if matches!(settings.cleanup_mode, CleanupMode::Off) {
                    raw
                } else {
                    app.set_state(DictationState::Polishing);
                    match cleanup::polish(&raw, &settings) {
                        Ok(t) => t,
                        Err(e) => {
                            log::error!("cleanup failed: {e:#}");
                            menubar::set_status_text(&format!(
                                "Cleanup failed — using raw transcript ({e})"
                            ));
                            raw
                        }
                    }
                };

                if let Err(e) = paste::deliver(&cleaned, &settings.inject_mode) {
                    log::error!("paste failed: {e:#}");
                }
                app.on_session_finished();
            })
            .expect("spawn transcribe thread");
    }

    fn on_session_finished(self: &Arc<Self>) {
        let new_state = if self.asr.lock().is_some() {
            DictationState::Idle
        } else {
            DictationState::ModelLoading
        };
        self.set_state(new_state);
    }

    pub fn set_state(&self, new_state: DictationState) {
        *self.current_state.lock() = new_state;
        self.refresh_menu();
        hud::show_state(new_state);
    }

    pub fn refresh_menu(&self) {
        let current = *self.current_state.lock();
        let asr_ready = self.asr.lock().is_some();
        let s = self.settings.load();
        let shortcut = glyphs_for_shortcut(&s.hotkey);
        // Show what we'll actually do, not what the user requested. Caps
        // Lock forces Hold-mode semantics (tap-to-start, tap-to-stop)
        // because the OS doesn't surface true press/release events for it.
        let effective_mode = effective_trigger_mode(&s);
        menubar::refresh(current, asr_ready, &shortcut, effective_mode);
    }

    /// Persist new settings AND apply runtime side-effects (rebinding the
    /// global hotkey if it changed).
    pub fn apply_settings(self: &Arc<Self>, new: Settings) -> anyhow::Result<()> {
        // Refuse to persist an empty / unparseable hotkey. The Settings
        // UI also validates before calling us, but this is the canonical
        // chokepoint — any future caller (CLI flag, scripted import)
        // can't bypass it.
        if new.hotkey.trim().is_empty() {
            anyhow::bail!("refusing to save an empty hotkey");
        }
        crate::hotkey::parse(&new.hotkey)
            .map_err(|e| anyhow::anyhow!("hotkey {:?} is not parseable: {e}", new.hotkey))?;

        let prev = self.settings.load();
        self.settings.save(&new)?;
        if prev.hotkey != new.hotkey {
            if let Some(handle) = self.hotkey.lock().as_ref() {
                handle.rebind(&new.hotkey)?;
            }
        }
        self.refresh_menu();
        Ok(())
    }

    /// Run the first-run model download + page-touch + recogniser load +
    /// dummy warmup decode on the tokio runtime. Transitions state from
    /// ModelLoading → Idle on success.
    pub async fn spawn_model_setup(self: Arc<Self>) {
        let model_dir = self.settings.model_dir();
        let vad_path = self.settings.vad_path();

        let on_progress: model_fetch::ProgressFn = Arc::new(|p| match p {
            Progress::Status(s) => menubar::set_status_text(&s),
            Progress::Chunk {
                file,
                bytes,
                total,
                fraction,
            } => {
                let pct = (fraction * 100.0) as u32;
                menubar::set_status_text(&format!(
                    "{file}: {} / {} ({pct}%)",
                    fmt_bytes(bytes),
                    fmt_bytes(total)
                ));
            }
        });

        if !self.settings.model_present() {
            if let Err(e) =
                model_fetch::ensure_model(&model_dir, &vad_path, on_progress.clone()).await
            {
                log::error!("model fetch failed: {e:#}");
                menubar::set_status_text(&format!("Model download failed: {e}"));
                return;
            }
        }

        let encoder_path = self.settings.encoder_path();
        let decoder_path = self.settings.decoder_path();
        let joiner_path = self.settings.joiner_path();
        let tokens_path = self.settings.tokens_path();

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Arc<Asr>> {
            menubar::set_status_text("Warming page cache…");
            let _ = warmup::page_touch(&encoder_path)?;

            menubar::set_status_text("Loading recogniser (CoreML)…");
            let threads = performance::performance_core_count();
            log::info!("OfflineRecognizer: {threads} threads, provider=coreml");
            let asr = Asr::load(
                &encoder_path,
                &decoder_path,
                &joiner_path,
                &tokens_path,
                threads,
            )?;

            menubar::set_status_text("Pre-warming graph…");
            warmup::dummy_decode(&asr)?;
            Ok(Arc::new(asr))
        })
        .await;

        match result {
            Ok(Ok(asr)) => {
                *self.asr.lock() = Some(asr);
                self.set_state(DictationState::Idle);
            }
            Ok(Err(e)) => {
                log::error!("model setup failed: {e:#}");
                menubar::set_status_text(&format!("Setup failed: {e}"));
            }
            Err(e) => {
                log::error!("setup task panicked: {e}");
                menubar::set_status_text(&format!("Setup panicked: {e}"));
            }
        }
    }
}

/// Singleton accessor used by the AppKit menu-action selectors which can't
/// carry Rust state through the Objective-C runtime.
pub struct AppHandle;

static APP: OnceLock<Arc<App>> = OnceLock::new();

impl AppHandle {
    pub fn set(app: Arc<App>) -> Result<(), Arc<App>> {
        APP.set(app)
    }
    pub fn get() -> Option<Arc<App>> {
        APP.get().cloned()
    }
}

/// Render the stored shortcut token as a glyph string for the menu.
/// Examples:
///   `CmdOrCtrl+Shift+Space` → `⌘⇧Space`
///   `CapsLock`              → `⇪`
///   `Eject`                 → `⏏`
pub fn glyphs_for_shortcut(token: &str) -> String {
    let trimmed = token.trim();
    if !trimmed.contains('+') {
        match trimmed.to_ascii_lowercase().as_str() {
            "capslock" | "caps_lock" | "caps-lock" => return "⇪".to_string(),
            "eject" => return "⏏".to_string(),
            _ => {} // fall through to chord renderer
        }
    }

    // Collect modifier flags first so we can render in macOS HIG canonical
    // order (⌃ ⌥ ⇧ ⌘) regardless of which order the user typed them.
    let mut has_ctrl = false;
    let mut has_alt = false;
    let mut has_shift = false;
    let mut has_cmd = false;
    let mut key = String::new();
    for part in trimmed.split('+').map(str::trim) {
        match part.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "cmdorctrl" | "commandorcontrol" | "super" | "meta" => {
                has_cmd = true
            }
            "ctrl" | "control" => has_ctrl = true,
            "alt" | "option" => has_alt = true,
            "shift" => has_shift = true,
            other => {
                key = match other {
                    "space" => "Space".to_string(),
                    "enter" | "return" => "⏎".to_string(),
                    "tab" => "⇥".to_string(),
                    "esc" | "escape" => "⎋".to_string(),
                    other => other.to_uppercase(),
                };
            }
        }
    }
    let mut out = String::new();
    if has_ctrl {
        out.push('⌃');
    }
    if has_alt {
        out.push('⌥');
    }
    if has_shift {
        out.push('⇧');
    }
    if has_cmd {
        out.push('⌘');
    }
    out.push_str(&key);
    out
}

/// True if the hotkey token names Caps Lock. The CGEventTap surfaces only
/// one event per physical Caps Lock tap (the modifier toggle), so the
/// Tap/Hold trigger-mode distinction doesn't apply — `effective_trigger_mode`
/// forces Hold so the second tap can finalise the session.
pub fn is_capslock_token(token: &str) -> bool {
    matches!(
        token.trim().to_ascii_lowercase().as_str(),
        "capslock" | "caps_lock" | "caps-lock"
    )
}

/// The trigger mode we actually use, after applying the Caps-Lock override.
/// For every other binding this just returns the stored `trigger_mode`.
pub fn effective_trigger_mode(s: &Settings) -> TriggerMode {
    if is_capslock_token(&s.hotkey) {
        TriggerMode::Hold
    } else {
        s.trigger_mode
    }
}

fn fmt_bytes(n: u64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.0} KB", n as f64 / 1024.0)
    } else if n < 1024 * 1024 * 1024 {
        format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", n as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::TriggerMode;

    #[test]
    fn glyphs_render_chord_modifiers_in_canonical_order() {
        // Token order is whatever the user typed; the glyph order is
        // always ⌃ ⌥ ⇧ ⌘ → key per macOS HIG, so two equivalent token
        // orderings render identically.
        assert_eq!(glyphs_for_shortcut("CmdOrCtrl+Shift+Space"), "⇧⌘Space");
        assert_eq!(glyphs_for_shortcut("Shift+CmdOrCtrl+Space"), "⇧⌘Space");
        assert_eq!(glyphs_for_shortcut("Alt+CmdOrCtrl+E"), "⌥⌘E");
        assert_eq!(glyphs_for_shortcut("Ctrl+Alt+Shift+Cmd+A"), "⌃⌥⇧⌘A");
    }

    #[test]
    fn glyphs_render_named_keys() {
        assert_eq!(glyphs_for_shortcut("F5"), "F5");
        assert_eq!(glyphs_for_shortcut("Enter"), "⏎");
        assert_eq!(glyphs_for_shortcut("Escape"), "⎋");
    }

    #[test]
    fn glyphs_render_special_bindings() {
        assert_eq!(glyphs_for_shortcut("CapsLock"), "⇪");
        assert_eq!(glyphs_for_shortcut("caps-lock"), "⇪");
        assert_eq!(glyphs_for_shortcut("Eject"), "⏏");
    }

    #[test]
    fn is_capslock_token_matches_aliases() {
        assert!(is_capslock_token("CapsLock"));
        assert!(is_capslock_token("caps_lock"));
        assert!(is_capslock_token("caps-lock"));
        assert!(is_capslock_token("  CAPSLOCK  "));
        assert!(!is_capslock_token("Eject"));
        assert!(!is_capslock_token("CmdOrCtrl+Shift+Space"));
    }

    #[test]
    fn effective_mode_forces_hold_for_capslock() {
        let s = Settings {
            hotkey: "CapsLock".into(),
            trigger_mode: TriggerMode::Tap,
            ..Settings::default()
        };
        assert_eq!(effective_trigger_mode(&s), TriggerMode::Hold);
        // Stored mode is untouched (the override is runtime-only).
        assert_eq!(s.trigger_mode, TriggerMode::Tap);
    }

    #[test]
    fn parse_rejects_empty_and_whitespace_hotkey() {
        // The validation that apply_settings runs lives on
        // `crate::hotkey::parse`. Pin both ends — empty rejected,
        // whitespace rejected — so a future refactor can't silently
        // re-open the "next launch bricks" footgun.
        assert!(crate::hotkey::parse("").is_err());
        assert!(crate::hotkey::parse("   ").is_err());
        // And the canonical happy paths still pass.
        assert!(crate::hotkey::parse("CmdOrCtrl+Shift+Space").is_ok());
        assert!(crate::hotkey::parse("F5").is_ok());
        assert!(crate::hotkey::parse("CapsLock").is_ok());
    }

    #[test]
    fn effective_mode_passes_through_for_other_bindings() {
        let mut s = Settings {
            hotkey: "F5".into(),
            trigger_mode: TriggerMode::Hold,
            ..Settings::default()
        };
        assert_eq!(effective_trigger_mode(&s), TriggerMode::Hold);
        s.trigger_mode = TriggerMode::Tap;
        assert_eq!(effective_trigger_mode(&s), TriggerMode::Tap);
    }
}
