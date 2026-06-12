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
use crate::dictation_fsm::{DictationFsm, HoldPressOutcome, TapPressOutcome};
use crate::hotkey::HotkeyHandle;
use crate::hud;
use crate::llm_manager::{FinalizeOutcome, LlmManager, LoadClaim};
use crate::menubar;
use crate::model_fetch::{self, Progress};
use crate::performance::PhaseTimer;
use crate::polish::{self, LlamaPolish, PolishBackend};
use crate::settings::{PolishMode, Settings, SettingsStore, TriggerMode};
use crate::streamer::{self, Mode as StreamerMode, Outcome, OutcomeRx};
use crate::{paste, performance, warmup};

pub use crate::dictation_fsm::DictationState;

pub struct App {
    /// State machine that owns the (state, session, pending_terminate)
    /// triple atomically. See `crate::dictation_fsm` for the lock-
    /// protocol rationale.
    pub fsm: DictationFsm,
    pub settings: SettingsStore,
    /// Set once the model is downloaded and the recogniser is warm.
    pub asr: Mutex<Option<Arc<Asr>>>,
    /// Polish-LLM lifecycle: `Disabled` → `Loading` → `Ready` → back
    /// to `Disabled` on toggle Off or load failure. See
    /// `crate::llm_manager` for the single-mutex state machine that
    /// replaced the previous `Mutex<Option<...>>` + `Mutex<bool>` pair.
    pub llm: LlmManager,
    /// Lives for the program's lifetime; held here so the Settings UI can
    /// rebind it after the user records a new hotkey combo.
    pub hotkey: Mutex<Option<HotkeyHandle>>,
}

impl App {
    pub fn new(settings: SettingsStore) -> Self {
        Self {
            fsm: DictationFsm::new(),
            settings,
            asr: Mutex::new(None),
            llm: LlmManager::new(),
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
    /// any lock held here also blocks the AppKit run loop. The FSM's
    /// internal mutex is the only one taken on the press path; it's
    /// held only for the read-modify-write of the (state, session,
    /// pending_terminate) triple and never across I/O. The mic open
    /// and Silero VAD load happen on a worker thread spawned by
    /// `start_session`, not under any lock.
    ///
    /// Do not introduce blocking I/O, file reads, or network calls
    /// inside any lock acquired by this path — doing so freezes the
    /// menu bar.
    pub fn on_hotkey_press(self: &Arc<Self>) {
        let mode = effective_trigger_mode(&self.settings.load());
        match mode {
            TriggerMode::Tap => match self.fsm.on_press_tap() {
                TapPressOutcome::ClaimedListening => {
                    self.announce_state(DictationState::Listening);
                    self.start_session(StreamerMode::VadAutoStop);
                }
                TapPressOutcome::CancelledLive | TapPressOutcome::QueuedCancel => {
                    // The FSM already routed the cancel; nothing
                    // further to do. The session-watcher (live case)
                    // or starter (gap case) will resolve state.
                }
                TapPressOutcome::Ignored(state) => {
                    log::debug!("hotkey press ignored from state {state:?}");
                }
            },
            TriggerMode::Hold => match self.fsm.on_press_hold() {
                HoldPressOutcome::ClaimedListening => {
                    self.announce_state(DictationState::Listening);
                    self.start_session(StreamerMode::Manual);
                }
                HoldPressOutcome::Ignored(state) => {
                    log::debug!("hotkey press ignored from state {state:?}");
                }
            },
        }
    }

    /// Hotkey-release edge. Only meaningful in Hold mode; in Tap mode the
    /// auto-key-repeat or release noise doesn't change anything.
    pub fn on_hotkey_release(self: &Arc<Self>) {
        if effective_trigger_mode(&self.settings.load()) != TriggerMode::Hold {
            return;
        }
        // FSM decides finalise-vs-queue under one lock so the starter
        // can't slip an install/drain in between.
        let _ = self.fsm.on_release_hold();
    }

    /// Spawn a worker thread that opens the mic + loads Silero VAD and,
    /// once both are ready, populates `app.session` and starts the
    /// session-watcher. Assumes `try_claim_listening` has already
    /// transitioned state to Listening.
    ///
    /// The cold-start work (~100-300 ms for CPAL + Silero) runs on the
    /// worker, NOT on the caller's thread. This matters because the
    /// caller of `on_hotkey_press` is often the CGEventTap callback,
    /// which has a hard ~250 ms budget before macOS disables the tap
    /// (`kCGEventTapDisabledByTimeout`). Doing the load inline would
    /// kill all subsequent hotkey events until the app restarts.
    ///
    /// During the worker's setup window, a press/release edge may
    /// arrive (Tap-cancel or Hold-release). Those edges are queued in
    /// `pending_terminate` by `on_hotkey_press` / `on_hotkey_release`
    /// when the session slot is empty, and drained here as soon as
    /// the slot is populated.
    fn start_session(self: &Arc<Self>, mode: StreamerMode) {
        let next = self.resting_state();
        if matches!(next, DictationState::ModelLoading) {
            // Defensive: a press got past `try_claim_listening` even
            // though the recogniser isn't loaded. Reachable only via a
            // race against the boot transition (asr is set before the
            // state goes Idle), so unlikely — but the wrong recovery
            // state would mislead the menubar.
            log::warn!("hotkey ignored: model still loading");
            self.fsm.abort_starter(next);
            self.announce_state(next);
            return;
        }
        let vad_path = self.settings.vad_path();
        self.spawn_supervised("session-starter", move |app| {
            app.start_session_blocking(mode, vad_path);
        });
    }

    /// Synchronous body of `start_session`. Runs on the
    /// `session-starter` worker; never call from the AppKit main
    /// thread or the CGEventTap callback — `streamer::start` opens
    /// the mic and loads Silero VAD (~100-300 ms cold).
    fn start_session_blocking(self: Arc<Self>, mode: StreamerMode, vad_path: std::path::PathBuf) {
        let (session, outcome_rx) = match streamer::start(&vad_path, mode) {
            Ok(pair) => pair,
            Err(e) => {
                log::error!("start session failed: {e:#}");
                let next = self.resting_state();
                self.fsm.abort_starter(next);
                self.announce_state(next);
                return;
            }
        };

        // Install the freshly-built session and atomically drain any
        // press/release edge that arrived during the cold-start gap.
        // The FSM holds the (session, pending_terminate) pair behind
        // one lock so the drain can't slip past the install; the
        // drained edge is delivered through the freshly-installed
        // session under that same lock.
        let _drained = self.fsm.install_session(session);

        self.spawn_supervised("session-watcher", move |app| {
            let OutcomeRx(rx) = outcome_rx;
            let outcome = rx.recv().ok();

            // FSM atomically transitions state out of Listening AND
            // clears the session slot. This closes two opposite TOCTOU
            // windows:
            //
            //   (a) `state=Listening` AND `session=None` (queue writes
            //       leak to next session).
            //   (b) `state=Idle/Transcribing` AND `session=Some`
            //       (hotkey claims a slot the old watcher is about to
            //       clear; the old watcher's drop then cancels the new
            //       session).
            let target_state = match &outcome {
                Some(Outcome::Speech { .. }) => DictationState::Transcribing,
                _ => app.resting_state(),
            };
            app.fsm.finish_session(target_state);
            // Menu + HUD refresh outside the FSM lock (AppKit calls
            // can be slow / re-entrant; never hold a Mutex across
            // them).
            app.announce_state(target_state);

            match outcome {
                Some(Outcome::Speech {
                    samples,
                    sample_rate,
                    timer,
                }) => {
                    app.transcribe_and_paste(samples, sample_rate, timer);
                }
                Some(Outcome::Cancelled) => {}
                Some(Outcome::NoSpeech) => {
                    log::info!("VAD heard nothing; aborting session");
                }
                Some(Outcome::Error(e)) => {
                    log::error!("session error: {e:#}");
                }
                None => {}
            }
        });
    }

    fn transcribe_and_paste(
        self: &Arc<Self>,
        samples: Vec<f32>,
        sample_rate: u32,
        mut timer: PhaseTimer,
    ) {
        self.spawn_supervised("transcribe", move |app| {
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
        });
    }

    fn on_session_finished(self: &Arc<Self>) {
        self.set_state(self.resting_state());
    }

    /// Force-recover after a worker thread panic. The three spawned
    /// worker threads (`session-starter`, `session-watcher`,
    /// `transcribe`) can panic at any FFI boundary (sherpa-onnx, cpal,
    /// llama.cpp, the `CGEventPost` keystroke synthesis in
    /// `ax_paste`). Without this recovery, a panic in
    /// any of them would leave the app stuck in `Listening` /
    /// `Transcribing` / `Polishing` forever — hotkey presses gated
    /// on `Idle` would be silently ignored and the user would have to
    /// quit.
    ///
    /// Called from `catch_unwind` handlers around each worker body.
    fn recover_from_panic(self: &Arc<Self>, source: &str, msg: &str) {
        log::warn!("recovering app state after {source} panic: {msg}");
        let next_state = self.resting_state();
        // Atomically clear session + pending_terminate + state under
        // the FSM lock so a concurrent hotkey press observes a clean
        // baseline.
        self.fsm.recover(next_state);
        self.announce_state(next_state);
        menubar::set_status_text(&format!("{source} crashed — try again"));
    }

    pub fn set_state(&self, new_state: DictationState) {
        self.fsm.set_state(new_state);
        self.announce_state(new_state);
    }

    /// The post-session resting state: `Idle` if the recogniser is
    /// loaded, `ModelLoading` otherwise. Used by every code path that
    /// has to compute "where do we go after this session/transcribe/
    /// recovery completes" — extracted so the answer is consistent and
    /// the asr lock is taken exactly once per decision.
    fn resting_state(&self) -> DictationState {
        if self.asr.lock().is_some() {
            DictationState::Idle
        } else {
            DictationState::ModelLoading
        }
    }

    /// Push the given state to both the menu-bar refresh and the HUD.
    /// Every state transition that surfaces to the user wants both
    /// updates back-to-back; extracted to keep them in lock-step (the
    /// pre-refactor code had a real bug where panic recovery refreshed
    /// the menu but not the HUD, leaving the listening waveform on
    /// screen).
    fn announce_state(&self, state: DictationState) {
        self.refresh_menu();
        hud::show_state(state);
    }

    /// Spawn a named worker thread whose body runs inside a
    /// `catch_unwind` boundary. On panic, `recover_from_panic(name, …)`
    /// is invoked so the dictation pipeline can return to Idle instead
    /// of soft-bricking. The body receives `Arc<App>` so it doesn't
    /// need to clone twice manually.
    fn spawn_supervised<F>(self: &Arc<Self>, name: &'static str, body: F)
    where
        F: FnOnce(Arc<App>) + Send + 'static,
    {
        self.spawn_supervised_with(name, body, |app, msg| {
            app.recover_from_panic(name, &msg);
        });
    }

    /// Variant of [`spawn_supervised`] with a caller-provided panic
    /// recovery closure. The default `spawn_supervised` calls
    /// `recover_from_panic`; the llm-toggle worker uses this to clear
    /// `llm_loading` instead (a polish-LLM panic shouldn't reset the
    /// whole dictation FSM).
    fn spawn_supervised_with<F, P>(self: &Arc<Self>, name: &'static str, body: F, on_panic: P)
    where
        F: FnOnce(Arc<App>) + Send + 'static,
        P: FnOnce(Arc<App>, String) + Send + 'static,
    {
        let app = Arc::clone(self);
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let app_for_panic = Arc::clone(&app);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    body(app);
                }));
                if let Err(payload) = result {
                    let msg = panic_message(&*payload);
                    on_panic(app_for_panic, msg);
                }
            })
            .unwrap_or_else(|e| panic!("spawn {name} thread: {e}"));
    }

    pub fn refresh_menu(&self) {
        let current = self.fsm.state();
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
        if prev.polish_mode != new.polish_mode {
            self.handle_polish_mode_change(new.polish_mode);
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

        let on_progress: model_fetch::ProgressFn = Arc::new(fetch_progress_to_menubar);

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

        // Polish LLM loads in a second pass, ONLY if polish is
        // enabled in settings. Skipping when off means a fresh install
        // doesn't pay 3.5 GB of resident memory for a feature the user
        // hasn't turned on. Settings-UI toggle (Off → On) re-triggers
        // a load via `apply_settings` → `handle_polish_mode_change`.
        if matches!(self.settings.load().polish_mode, PolishMode::On) {
            self.clone().spawn_llm_setup().await;
        }
    }

    /// Load + warm the polish GGUF off the main thread. Idempotent —
    /// returns early if `app.llm` is already populated or another load
    /// is in flight. Called from the boot path
    /// (`spawn_model_setup`) AND from the Settings-UI toggle
    /// (`apply_settings` → `handle_polish_mode_change`).
    ///
    /// Single-flight via `llm_loading: Mutex<bool>` — without that gate
    /// a rapid toggle (boot still loading + user flips Off→On) could
    /// fire two concurrent loads racing to write `llm`, each holding
    /// 3.5 GB of GGUF weights resident.
    pub async fn spawn_llm_setup(self: Arc<Self>) {
        if !matches!(self.llm.try_claim_load(), LoadClaim::Claimed) {
            return;
        }
        let app = Arc::clone(&self);
        let result = tokio::task::spawn_blocking(move || load_llm_blocking(&app.settings)).await;
        self.finalize_llm_load(result.unwrap_or_else(|e| Err(anyhow::anyhow!("task panic: {e}"))));
    }

    /// Apply a completed loader result. Reading `polish_mode` here
    /// (inside the manager's finalize critical section) is what wins
    /// the rapid-toggle race: if the user flipped Off while we were
    /// loading, settings.cache reads `Off` and the manager discards
    /// the freshly-loaded backend.
    fn finalize_llm_load(&self, result: anyhow::Result<Arc<dyn PolishBackend>>) {
        let keep_if_loaded = matches!(self.settings.load().polish_mode, PolishMode::On);
        let outcome = self.llm.finalize_load(result, keep_if_loaded);
        let status: Option<String> = match outcome {
            FinalizeOutcome::Stored => Some("Polish ready".to_string()),
            FinalizeOutcome::DiscardedDisabled => {
                log::info!("polish load completed but mode is now Off; discarding loaded model");
                None
            }
            FinalizeOutcome::Failed(e) => {
                log::error!("polish model setup failed: {e:#}");
                if keep_if_loaded {
                    Some(format!("Polish setup failed: {e}"))
                } else {
                    None
                }
            }
        };
        if let Some(text) = status {
            menubar::set_status_text(&text);
            self.refresh_menu();
        }
    }

    /// Settings-UI polish toggle hook. On→Off drops the loaded model
    /// (releases the 3.5 GB of weights); Off→On spawns a worker thread
    /// to load + warm, guarded by `LlmManager::try_claim_load`. Called
    /// from `apply_settings` when `polish_mode` changes.
    fn handle_polish_mode_change(self: &Arc<Self>, new_mode: PolishMode) {
        match new_mode {
            PolishMode::Off => {
                // Drop the Arc; if other threads hold clones (e.g. a
                // polish-in-flight) the model lives until they're done.
                self.llm.disable();
                menubar::set_status_text("Polish disabled");
                self.refresh_menu();
            }
            PolishMode::On => {
                if !matches!(self.llm.try_claim_load(), LoadClaim::Claimed) {
                    return;
                }
                self.spawn_supervised_with(
                    "llm-toggle-load",
                    move |app| {
                        let load_result = load_llm_blocking(&app.settings);
                        app.finalize_llm_load(load_result);
                    },
                    |app, msg| {
                        log::error!("llm-toggle-load panic: {msg}");
                        // Reset `Loading` → `Disabled` so a future On
                        // toggle can retry. Without this the toggle
                        // would silently no-op forever.
                        app.llm.clear_loading_after_panic();
                        menubar::set_status_text("Polish load crashed — try again");
                    },
                );
            }
        }
    }
}

/// Synchronous polish-LLM load. Used by both the boot path (via
/// `tokio::spawn_blocking`) and the Settings-toggle path (via
/// `std::thread::spawn`). Updates the menubar status text as it goes
/// so the user sees progress through the ~250 ms load + ~150 ms warm.
fn load_llm_blocking(settings: &SettingsStore) -> anyhow::Result<Arc<dyn PolishBackend>> {
    let model_path = settings.polish_model_path();
    if !settings.polish_model_present() {
        // First enable: auto-fetch the GGUF (~3.5 GB). Both callers run
        // on a dedicated blocking thread (`tokio::spawn_blocking` on the
        // boot path, supervised `std::thread` on the toggle path), so
        // driving the async downloader with a throwaway current-thread
        // runtime here is safe and keeps a single wiring point for both.
        // The caller's load-slot claim means toggle-spam can't start two
        // concurrent downloads; on failure the slot finalizes as Failed
        // and a later toggle retries (the .part cleanup in `download_to`
        // guarantees no corrupt leftover).
        use anyhow::Context as _;
        let on_progress: model_fetch::ProgressFn = Arc::new(fetch_progress_to_menubar);
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("building polish-download runtime")?
            .block_on(model_fetch::ensure_polish_model(&model_path, on_progress))
            .with_context(|| {
                format!("auto-downloading polish model to {}", model_path.display())
            })?;
    }
    menubar::set_status_text("Loading polish model…");
    let llm = LlamaPolish::load(&model_path)?;
    menubar::set_status_text("Warming polish model…");
    llm.warmup()?;
    Ok(Arc::new(llm))
}

/// Deliver `raw` to the focused app, optionally piping through the
/// polish LLM with streaming-paste. Split out of `transcribe_and_paste`
/// so the streaming-paste choreography (start streamer → push chunks →
/// finish) stays readable.
///
/// Behaviour matrix:
///
/// | polish_mode | LLM loaded? | path |
/// |---|---|---|
/// | Off | — | one-shot `paste::deliver(raw)` |
/// | On  | yes | streaming polish → `paste::Streamer` |
/// | On  | no  | one-shot paste of `raw`, status text explains |
///
/// Polish failures fall back to raw paste — the user always sees their
/// transcript, never nothing.
fn deliver_cleaned(app: &App, raw: &str, settings: &Settings) -> anyhow::Result<()> {
    if matches!(settings.polish_mode, PolishMode::Off) {
        return paste::deliver(raw);
    }
    let Some(llm) = app.llm.try_get() else {
        // Polish was enabled but the model isn't loaded — pasting
        // raw is the right fallback (better than nothing). Status
        // text already explained the load failure.
        log::warn!("polish enabled but model unavailable; pasting raw");
        return paste::deliver(raw);
    };
    app.set_state(DictationState::Polishing);

    let mut streamer = paste::Streamer::start()?;
    let outcome = run_polish_isolated(llm.as_ref(), raw, settings, |chunk| {
        streamer
            .push(chunk)
            .map_err(|e| anyhow::anyhow!("streamer push: {e}"))
    });
    match outcome {
        PolishOutcome::Ok => {
            // Success: flush the unbroken-boundary tail (often the
            // model's last fragment without a trailing space).
            streamer.commit()
        }
        PolishOutcome::Error(e) => {
            log::error!("polish pipeline failed: {e:#}");
            // Sample fired-state BEFORE `abort` consumes the streamer.
            // `abort` deliberately does NOT flush the pending tail, so
            // this snapshot won't drift under us the way a `commit`-then-
            // snapshot would.
            let any_streamed = streamer.has_fired();
            streamer.abort();
            if any_streamed {
                menubar::set_status_text(&format!(
                    "Polish failed mid-stream — partial output kept ({e})"
                ));
                Ok(())
            } else {
                menubar::set_status_text(&format!("Polish failed — using raw transcript ({e})"));
                paste::deliver(raw)
            }
        }
        PolishOutcome::Panicked(msg) => {
            log::error!("polish panic caught: {msg}");
            // We leave the model loaded — a single panic doesn't mean
            // the weights are corrupt, and reloading would cost ~250 ms.
            let any_streamed = streamer.has_fired();
            streamer.abort();
            if any_streamed {
                menubar::set_status_text("Polish panicked mid-stream — partial output kept");
                Ok(())
            } else {
                menubar::set_status_text("Polish panicked — using raw transcript");
                paste::deliver(raw)
            }
        }
    }
}

/// Outcome of one `polish_streaming` call run inside `catch_unwind`.
///
/// Splitting the success / typed-error / panic-payload triplet into a
/// concrete enum lets us unit-test the panic-isolation seam without
/// dragging `paste::Streamer` (which talks to the focused app via the
/// CGEvent keystroke pipeline; see ADR-0019) into the test.
#[derive(Debug)]
enum PolishOutcome {
    Ok,
    Error(anyhow::Error),
    Panicked(String),
}

/// Run the polish backend with a `catch_unwind` boundary. `on_chunk`
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
    backend: &dyn PolishBackend,
    text: &str,
    settings: &Settings,
    mut on_chunk: F,
) -> PolishOutcome
where
    F: FnMut(&str) -> anyhow::Result<()>,
{
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        polish::polish_streaming(backend, text, settings, &mut on_chunk)
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
                has_cmd = true;
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

/// Shared download-progress sink: routes `model_fetch::Progress` events
/// (ASR first-run fetch AND polish first-enable fetch) to the menubar
/// status text.
fn fetch_progress_to_menubar(p: Progress) {
    match p {
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
            panic!("synthetic panic from polish");
        });
        let payload = result.unwrap_err();
        // `&*payload` derefs the Box so `downcast_ref` sees the inner
        // type (`&'static str`) instead of the Box itself.
        assert_eq!(panic_message(&*payload), "synthetic panic from polish");
    }

    #[test]
    fn panic_message_extracts_owned_string_payload() {
        // `panic!("{}", String::from(...))` payload comes through as
        // owned `String`. Both branches of `panic_message` must work.
        let result: std::thread::Result<()> = std::panic::catch_unwind(|| {
            let dynamic = String::from("dynamic polish error");
            panic!("{}", dynamic);
        });
        let payload = result.unwrap_err();
        assert_eq!(panic_message(&*payload), "dynamic polish error");
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
    // Real `LlamaPolish` can't be constructed without a GGUF on disk,
    // so we drive `run_polish_isolated` through fakes that exercise
    // each `PolishOutcome` arm directly.
    struct OkBackend;
    impl crate::polish::PolishBackend for OkBackend {
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
    impl crate::polish::PolishBackend for ErroringBackend {
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
    impl crate::polish::PolishBackend for PanickingBackend {
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
            polish_mode: PolishMode::On,
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
            polish_mode: PolishMode::On,
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
            polish_mode: PolishMode::On,
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
        // PolishMode::Off should short-circuit BEFORE reaching the
        // backend — even a PanickingBackend should never be invoked.
        // Caller's `on_chunk` sees the raw text exactly once.
        let backend = PanickingBackend;
        let settings = Settings {
            polish_mode: PolishMode::Off,
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
