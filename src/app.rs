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
use crate::hotkey::HotkeyHandle;
use crate::menubar;
use crate::model_fetch::{self, Progress};
use crate::settings::{Settings, SettingsStore, TriggerMode};
use crate::streamer::{self, Mode as StreamerMode, Outcome, Session};
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
        let session = match streamer::start(&vad_path, mode) {
            Ok(s) => s,
            Err(e) => {
                log::error!("start session failed: {e:#}");
                return;
            }
        };
        self.set_state(DictationState::Listening);

        {
            let mut slot = self.session.lock();
            let prev = slot.replace(session);
            debug_assert!(prev.is_none(), "session slot was not cleared");
        }

        let app = self.clone();
        std::thread::Builder::new()
            .name("session-watcher".into())
            .spawn(move || {
                let session = app.session.lock().take();
                let outcome = match session {
                    Some(s) => s.outcome_rx.recv().ok(),
                    None => None,
                };
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
                let text = match asr.recognize(&samples, sample_rate) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!("recognise failed: {e:#}");
                        app.on_session_finished();
                        return;
                    }
                };
                let mode = app.settings.load().inject_mode;
                if let Err(e) = paste::deliver(&text, &mode) {
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

    let mut mods = String::new();
    let mut key = String::new();
    for part in trimmed.split('+').map(str::trim) {
        let g = match part.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "cmdorctrl" | "commandorcontrol" | "super" | "meta" => "⌘",
            "ctrl" | "control" => "⌃",
            "alt" | "option" => "⌥",
            "shift" => "⇧",
            other => {
                key = match other {
                    "space" => "Space".to_string(),
                    "enter" | "return" => "⏎".to_string(),
                    "tab" => "⇥".to_string(),
                    "esc" | "escape" => "⎋".to_string(),
                    other => other.to_uppercase(),
                };
                continue;
            }
        };
        mods.push_str(g);
    }
    format!("{mods}{key}")
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
