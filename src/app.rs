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
use crate::cleanup::{self, CleanupBackend, LlamaCleanup};
use crate::hotkey::HotkeyHandle;
use crate::hud;
use crate::menubar;
use crate::model_fetch::{self, Progress};
use crate::performance::PhaseTimer;
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
    /// Set once the cleanup GGUF is loaded + warmed. `None` while
    /// loading or when `cleanup_mode == Off` (no point warming what
    /// the user didn't enable). `transcribe_and_paste` falls back to
    /// raw paste when the slot is empty. Boxed behind the
    /// [`CleanupBackend`] trait so tests can swap in a fake without
    /// needing a real GGUF on disk.
    pub llm: Mutex<Option<Arc<dyn CleanupBackend>>>,
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
            llm: Mutex::new(None),
            current_state: Mutex::new(DictationState::ModelLoading),
            hotkey: Mutex::new(None),
        }
    }

    /// Hotkey-press edge. Behaviour depends on the configured TriggerMode:
    /// - Tap mode: toggle (start a session, or cancel one in flight).
    /// - Hold mode: always start a session. Release is the commit edge.
    ///
    /// # Lock-discipline invariant
    ///
    /// This runs on the **main thread** when invoked from the NSEvent
    /// global-monitor block (`hotkey::install_media_key_monitor`), so
    /// any lock held here also blocks the AppKit run loop. Every lock
    /// acquired in this function and the helpers it calls must be
    /// **uncontended-short**:
    /// - `self.settings.load()` clones from a `parking_lot::Mutex`
    ///   inside `SettingsStore` — held only for the clone duration.
    /// - `self.session.lock()` is briefly held to check `is_some()` and
    ///   to call `s.cancel()` (a single channel send). No I/O.
    /// - `self.asr.lock()` (inside `start_session`) is held to check
    ///   `is_none()` and clone the `Arc`; no I/O.
    /// - Spawning the session-watcher thread releases all locks before
    ///   the spawn returns.
    ///
    /// Do not introduce blocking I/O, file reads, or network calls
    /// inside any lock acquired by this path — doing so freezes the
    /// menu bar.
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
                        timer,
                    }) => {
                        app.set_state(DictationState::Transcribing);
                        app.transcribe_and_paste(samples, sample_rate, timer);
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

    fn transcribe_and_paste(
        self: &Arc<Self>,
        samples: Vec<f32>,
        sample_rate: u32,
        mut timer: PhaseTimer,
    ) {
        let app = self.clone();
        std::thread::Builder::new()
            .name("transcribe".into())
            .spawn(move || {
                let Some(asr) = app.asr.lock().clone() else {
                    log::error!("transcribe: model gone");
                    app.on_session_finished();
                    return;
                };
                timer.mark_asr_start();
                let raw = match asr.recognize(&samples, sample_rate) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!("recognise failed: {e:#}");
                        app.on_session_finished();
                        return;
                    }
                };
                timer.mark_asr_done();

                let settings = app.settings.load();
                let result = deliver_cleaned(&app, &raw, &settings);
                // Marks *last*-chunk paste under streaming; perceived
                // first-chunk latency is much lower. See ADR-0018.
                timer.mark_paste_done();
                timer.emit();
                if let Err(e) = result {
                    log::error!("deliver failed: {e:#}");
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
    pub fn apply_settings(self: &Arc<Self>, new: &Settings) -> anyhow::Result<()> {
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
        self.settings.save(new)?;
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

        // Cleanup LLM loads in a second pass, ONLY if cleanup is
        // enabled in settings. Skipping when off means a fresh install
        // doesn't pay 1.2 GB of resident memory for a feature the user
        // hasn't turned on. The Settings UI toggle should re-trigger
        // this load when flipped to On (TODO: wire that hook).
        if matches!(self.settings.load().cleanup_mode, CleanupMode::On) {
            self.clone().spawn_llm_setup().await;
        }
    }

    /// Load + warm the cleanup GGUF off the main thread. Idempotent —
    /// returns early if `app.llm` is already populated. Called from
    /// `spawn_model_setup` today.
    ///
    /// **Not single-flight.** The `is_some()` check and the eventual
    /// write into `self.llm` are not atomic; two concurrent callers
    /// would both observe `None`, both load the 1.2 GB GGUF, and race
    /// at write-time. Today only the boot path invokes this, so the
    /// race is unreachable. **Before wiring the Settings UI toggle
    /// hook (TODO in `spawn_model_setup`), gate this on a `Mutex<bool>`
    /// "loading-in-progress" flag** so the toggle can't trigger a
    /// duplicate load while the boot load is still in flight.
    pub async fn spawn_llm_setup(self: Arc<Self>) {
        if self.llm.lock().is_some() {
            return; // already loaded
        }
        let model_path = self.settings.cleanup_model_path();
        if !self.settings.cleanup_model_present() {
            // TODO: extend model_fetch.rs to pull the GGUF. For v1
            // the user runs `scripts/fetch-cleanup-model.sh` or
            // bench_llm's curl one-liner.
            log::warn!(
                "cleanup model missing at {} — cleanup will fail back to raw paste",
                model_path.display()
            );
            menubar::set_status_text("Cleanup model not downloaded");
            return;
        }
        let result =
            tokio::task::spawn_blocking(move || -> anyhow::Result<Arc<dyn CleanupBackend>> {
                menubar::set_status_text("Loading cleanup model…");
                let llm = LlamaCleanup::load(&model_path)?;
                menubar::set_status_text("Warming cleanup model…");
                llm.warmup()?;
                Ok(Arc::new(llm))
            })
            .await;
        match result {
            Ok(Ok(llm)) => {
                *self.llm.lock() = Some(llm);
                menubar::set_status_text("Cleanup ready");
                self.refresh_menu();
            }
            Ok(Err(e)) => {
                log::error!("cleanup model setup failed: {e:#}");
                menubar::set_status_text(&format!("Cleanup setup failed: {e}"));
            }
            Err(e) => {
                log::error!("cleanup setup task panicked: {e}");
                menubar::set_status_text(&format!("Cleanup setup panicked: {e}"));
            }
        }
    }
}

/// Deliver `raw` to the focused app, optionally piping through the
/// cleanup LLM with streaming-paste. Split out of `transcribe_and_paste`
/// so the streaming-paste choreography (start streamer → push chunks →
/// finish) stays readable.
///
/// Behaviour matrix:
///
/// | cleanup_mode | LLM loaded? | path |
/// |---|---|---|
/// | Off | — | one-shot `paste::deliver(raw)` |
/// | On  | yes | streaming polish → `paste::Streamer` |
/// | On  | no  | one-shot paste of `raw`, status text explains |
///
/// Cleanup failures fall back to raw paste — the user always sees their
/// transcript, never nothing.
fn deliver_cleaned(app: &App, raw: &str, settings: &Settings) -> anyhow::Result<()> {
    if matches!(settings.cleanup_mode, CleanupMode::Off) {
        return paste::deliver(raw, &settings.inject_mode);
    }
    let llm = match app.llm.lock().clone() {
        Some(l) => l,
        None => {
            // Cleanup was enabled but the model isn't loaded — pasting
            // raw is the right fallback (better than nothing). Status
            // text already explained the load failure.
            log::warn!("cleanup enabled but model unavailable; pasting raw");
            return paste::deliver(raw, &settings.inject_mode);
        }
    };
    app.set_state(DictationState::Polishing);

    let mut streamer = paste::Streamer::start()?;
    let outcome = run_polish_isolated(llm.as_ref(), raw, settings, |chunk| {
        streamer
            .push(chunk)
            .map_err(|e| anyhow::anyhow!("streamer push: {e}"))
    });
    // Capture fired-state BEFORE `finish` consumes the streamer.
    // If any chunks were pasted, falling back to raw would append
    // raw text on top of partial cleaned output — strictly worse than
    // just keeping the partial.
    let any_streamed = streamer.has_fired();
    let finish_result = streamer.finish();
    match outcome {
        PolishOutcome::Ok => finish_result,
        PolishOutcome::Error(e) => {
            log::error!("cleanup pipeline failed: {e:#}");
            if any_streamed {
                menubar::set_status_text(&format!(
                    "Cleanup failed mid-stream — partial output kept ({e})"
                ));
                Ok(())
            } else {
                menubar::set_status_text(&format!(
                    "Cleanup failed — using raw transcript ({e})"
                ));
                paste::deliver(raw, &settings.inject_mode)
            }
        }
        PolishOutcome::Panicked(msg) => {
            log::error!("cleanup panic caught: {msg}");
            // We leave the model loaded — a single panic doesn't mean
            // the weights are corrupt, and reloading would cost ~250 ms.
            if any_streamed {
                menubar::set_status_text("Cleanup panicked mid-stream — partial output kept");
                Ok(())
            } else {
                menubar::set_status_text("Cleanup panicked — using raw transcript");
                paste::deliver(raw, &settings.inject_mode)
            }
        }
    }
}

/// Outcome of one `polish_streaming` call run inside `catch_unwind`.
///
/// Splitting the success / typed-error / panic-payload triplet into a
/// concrete enum lets us unit-test the panic-isolation seam without
/// dragging `paste::Streamer` (which touches the system clipboard)
/// into the test.
#[derive(Debug)]
enum PolishOutcome {
    Ok,
    Error(anyhow::Error),
    Panicked(String),
}

/// Run the cleanup backend with a `catch_unwind` boundary. `on_chunk`
/// is called from inside the unwind boundary — if it panics, the panic
/// is caught the same way a polish-internal panic would be.
///
/// "Panic isolation" here means **Rust panics** from `llama-cpp-2`'s
/// safe wrapper layer (e.g. an `assert!` inside the binding, a tokenise
/// failure that the binding maps to `unwrap`). It does NOT catch C++
/// exceptions, SIGSEGV from the GGML backend, or `abort()` from the
/// underlying llama.cpp C++ — those bypass Rust's unwinding machinery
/// entirely. The fallback is best-effort, not bulletproof.
fn run_polish_isolated<F>(
    backend: &dyn CleanupBackend,
    text: &str,
    settings: &Settings,
    mut on_chunk: F,
) -> PolishOutcome
where
    F: FnMut(&str) -> anyhow::Result<()>,
{
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cleanup::polish_streaming(backend, text, settings, &mut on_chunk)
    }));
    match result {
        Ok(Ok(())) => PolishOutcome::Ok,
        Ok(Err(e)) => PolishOutcome::Error(e),
        Err(payload) => PolishOutcome::Panicked(panic_message(&*payload)),
    }
}

/// Extract a printable message from a `catch_unwind` payload — handles
/// both `&'static str` and `String` panic types, returns a generic
/// label for anything else (rare).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
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
    fn panic_message_extracts_static_str_payload() {
        // `panic!("literal")` payload comes through as `&'static str`.
        let result: std::thread::Result<()> = std::panic::catch_unwind(|| {
            panic!("synthetic panic from cleanup");
        });
        let payload = result.unwrap_err();
        // `&*payload` derefs the Box so `downcast_ref` sees the inner
        // type (`&'static str`) instead of the Box itself.
        assert_eq!(panic_message(&*payload), "synthetic panic from cleanup");
    }

    #[test]
    fn panic_message_extracts_owned_string_payload() {
        // `panic!("{}", String::from(...))` payload comes through as
        // owned `String`. Both branches of `panic_message` must work.
        let result: std::thread::Result<()> = std::panic::catch_unwind(|| {
            let dynamic = String::from("dynamic cleanup error");
            panic!("{}", dynamic);
        });
        let payload = result.unwrap_err();
        assert_eq!(panic_message(&*payload), "dynamic cleanup error");
    }

    #[test]
    fn panic_message_handles_unknown_payload_type() {
        // `panic_any` with a custom type (rare in practice, but FFI
        // surfaces sometimes emit non-string panic payloads).
        let result: std::thread::Result<()> = std::panic::catch_unwind(|| {
            std::panic::panic_any(42i32);
        });
        let payload = result.unwrap_err();
        assert_eq!(panic_message(&*payload), "non-string panic payload");
    }

    // Test backends for the §6-7 panic-isolation acceptance criterion.
    // Real `LlamaCleanup` can't be constructed without a GGUF on disk,
    // so we drive `run_polish_isolated` through fakes that exercise
    // each `PolishOutcome` arm directly.
    struct OkBackend;
    impl crate::cleanup::CleanupBackend for OkBackend {
        fn polish_into(
            &self,
            text: &str,
            on_chunk: &mut dyn FnMut(&str) -> anyhow::Result<()>,
        ) -> anyhow::Result<()> {
            on_chunk(&format!("polished: {text}"))
        }
        fn warmup(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct ErroringBackend;
    impl crate::cleanup::CleanupBackend for ErroringBackend {
        fn polish_into(
            &self,
            _text: &str,
            _on_chunk: &mut dyn FnMut(&str) -> anyhow::Result<()>,
        ) -> anyhow::Result<()> {
            Err(anyhow::anyhow!("simulated polish error"))
        }
        fn warmup(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct PanickingBackend;
    impl crate::cleanup::CleanupBackend for PanickingBackend {
        fn polish_into(
            &self,
            _text: &str,
            _on_chunk: &mut dyn FnMut(&str) -> anyhow::Result<()>,
        ) -> anyhow::Result<()> {
            panic!("simulated llama-cpp safe-wrapper panic");
        }
        fn warmup(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn run_polish_isolated_returns_ok_when_backend_succeeds() {
        let backend = OkBackend;
        let settings = Settings {
            cleanup_mode: CleanupMode::On,
            ..Settings::default()
        };
        let mut captured = String::new();
        let outcome = run_polish_isolated(&backend, "hello", &settings, |c| {
            captured.push_str(c);
            Ok(())
        });
        assert!(matches!(outcome, PolishOutcome::Ok));
        assert_eq!(captured, "polished: hello");
    }

    #[test]
    fn run_polish_isolated_classifies_typed_error_separately_from_panic() {
        // An `Err(...)` from the backend should NOT trip catch_unwind —
        // it's a typed failure, not a panic. The fallback paste is
        // still raw, but the user-facing message includes the error.
        let backend = ErroringBackend;
        let settings = Settings {
            cleanup_mode: CleanupMode::On,
            ..Settings::default()
        };
        let mut captured = String::new();
        let outcome = run_polish_isolated(&backend, "hello", &settings, |c| {
            captured.push_str(c);
            Ok(())
        });
        match outcome {
            PolishOutcome::Error(e) => {
                assert!(e.to_string().contains("simulated polish error"));
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn run_polish_isolated_catches_panic_and_recovers_payload() {
        // The §6-7 acceptance: a panic inside the polish call MUST be
        // caught here so the dictation pipeline can fall back to raw
        // paste. This is the load-bearing test for ADR-0018's panic
        // isolation tradeoff.
        let backend = PanickingBackend;
        let settings = Settings {
            cleanup_mode: CleanupMode::On,
            ..Settings::default()
        };
        let mut captured = String::new();
        let outcome = run_polish_isolated(&backend, "hello", &settings, |c| {
            captured.push_str(c);
            Ok(())
        });
        match outcome {
            PolishOutcome::Panicked(msg) => {
                assert!(
                    msg.contains("simulated llama-cpp safe-wrapper panic"),
                    "expected the panic payload, got: {msg}"
                );
            }
            other => panic!("expected Panicked, got {other:?}"),
        }
    }

    #[test]
    fn run_polish_isolated_off_mode_bypasses_backend_panic() {
        // CleanupMode::Off should short-circuit BEFORE reaching the
        // backend — even a PanickingBackend should never be invoked.
        // Caller's `on_chunk` sees the raw text exactly once.
        let backend = PanickingBackend;
        let settings = Settings {
            cleanup_mode: CleanupMode::Off,
            ..Settings::default()
        };
        let mut captured = String::new();
        let outcome = run_polish_isolated(&backend, "raw text", &settings, |c| {
            captured.push_str(c);
            Ok(())
        });
        assert!(matches!(outcome, PolishOutcome::Ok));
        assert_eq!(captured, "raw text");
    }

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
