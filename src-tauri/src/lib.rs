mod asr;
mod audio;
mod model_fetch;
mod paste;
mod qos;
mod settings;
mod sf_symbol;
mod streamer;
mod vad;
mod warmup;

use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;
use tauri::{
    AppHandle, Emitter, Manager, WindowEvent,
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{TrayIcon, TrayIconBuilder},
};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
};

use crate::asr::Asr;
use crate::settings::{Settings, SettingsStore};
use crate::streamer::{Outcome, Session};

pub struct TrayHandles {
    pub tray: TrayIcon,
    pub status_item: MenuItem<tauri::Wry>,
    pub toggle_item: MenuItem<tauri::Wry>,
}

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

pub struct AppState {
    /// Active dictation session, if one is running.
    pub session: Mutex<Option<Session>>,
    pub settings: SettingsStore,
    /// Set once the model is downloaded and the recognizer is warm.
    pub asr: Mutex<Option<Arc<Asr>>>,
    pub tray: Mutex<Option<TrayHandles>>,
    pub current_state: Mutex<DictationState>,
}

#[derive(Serialize)]
struct SettingsView {
    hotkey: String,
    #[serde(rename = "injectMode")]
    inject_mode: String,
    language: String,
    #[serde(rename = "modelReady")]
    model_ready: bool,
    #[serde(rename = "modelPath")]
    model_path: String,
}

#[tauri::command]
fn get_settings(state: tauri::State<'_, Arc<AppState>>) -> SettingsView {
    let s = state.settings.load();
    SettingsView {
        hotkey: s.hotkey,
        inject_mode: s.inject_mode,
        language: s.language,
        model_ready: state.asr.lock().is_some(),
        model_path: state.settings.encoder_path().display().to_string(),
    }
}

#[tauri::command]
fn save_settings(
    app: AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    hotkey: String,
    #[allow(non_snake_case)] injectMode: String,
    language: String,
) -> Result<(), String> {
    let old = state.settings.load();
    let new = Settings {
        hotkey,
        inject_mode: injectMode,
        language,
    };
    state.settings.save(&new).map_err(|e| e.to_string())?;
    if old.hotkey != new.hotkey {
        rebind_hotkey(&app, &new.hotkey).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Render the stored shortcut token as a glyph string for the tray menu.
/// `CmdOrCtrl+Shift+Space` → `⌘⇧Space`.
fn glyphs_for_shortcut(token: &str) -> String {
    let mut mods = String::new();
    let mut key = String::new();
    for part in token.split('+').map(|s| s.trim()) {
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

fn set_state(app: &AppHandle, state: &Arc<AppState>, new_state: DictationState) {
    *state.current_state.lock() = new_state;
    refresh_tray(app, state);
}

fn refresh_tray(app: &AppHandle, state: &Arc<AppState>) {
    let current = *state.current_state.lock();
    let asr_ready = state.asr.lock().is_some();
    let shortcut = glyphs_for_shortcut(&state.settings.load().hotkey);

    // Idle ↔ Listening uses two SF Symbols: `mic` (outline) and `mic.fill`.
    // Transcribing reuses `mic.fill` while we don't have a separate spinner.
    // ModelLoading uses `arrow.down.circle` to signal "not ready yet".
    let (symbol, status_label, toggle_label) = match current {
        DictationState::ModelLoading => (
            "arrow.down.circle",
            "Model: downloading…",
            format!("Dictation unavailable ({shortcut})"),
        ),
        DictationState::Idle => (
            "mic",
            if asr_ready { "Model: ready" } else { "Model: loading…" },
            format!("Start Dictation  {shortcut}"),
        ),
        DictationState::Listening => (
            "mic.fill",
            "Model: ready",
            format!("Stop Dictation  {shortcut}"),
        ),
        DictationState::Transcribing => (
            "mic.fill",
            "Transcribing…",
            "Working…".to_string(),
        ),
    };

    if let Some(handles) = state.tray.lock().as_ref() {
        if let Some(img) = sf_symbol::load(symbol, 18.0) {
            let _ = handles.tray.set_icon(Some(img));
            let _ = handles.tray.set_icon_as_template(true);
        }
        let _ = handles.status_item.set_text(status_label);
        let _ = handles.toggle_item.set_text(&toggle_label);
        let _ = handles
            .toggle_item
            .set_enabled(!matches!(current, DictationState::Transcribing));
    }
    let _ = app.emit("tray-state", format!("{current:?}"));
}

fn parse_shortcut(spec: &str) -> anyhow::Result<Shortcut> {
    let mut mods = Modifiers::empty();
    let mut code: Option<Code> = None;
    for part in spec.split('+').map(|s| s.trim()) {
        match part.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "super" | "meta" => mods |= Modifiers::SUPER,
            "ctrl" | "control" => mods |= Modifiers::CONTROL,
            "alt" | "option" => mods |= Modifiers::ALT,
            "shift" => mods |= Modifiers::SHIFT,
            "cmdorctrl" | "commandorcontrol" => mods |= Modifiers::SUPER,
            key => {
                code = Some(match key {
                    "space" => Code::Space,
                    "enter" | "return" => Code::Enter,
                    "tab" => Code::Tab,
                    "esc" | "escape" => Code::Escape,
                    "backspace" => Code::Backspace,
                    other if other.len() == 1 => {
                        let c = other.chars().next().unwrap().to_ascii_lowercase();
                        match c {
                            'a' => Code::KeyA,
                            'b' => Code::KeyB,
                            'c' => Code::KeyC,
                            'd' => Code::KeyD,
                            'e' => Code::KeyE,
                            'f' => Code::KeyF,
                            'g' => Code::KeyG,
                            'h' => Code::KeyH,
                            'i' => Code::KeyI,
                            'j' => Code::KeyJ,
                            'k' => Code::KeyK,
                            'l' => Code::KeyL,
                            'm' => Code::KeyM,
                            'n' => Code::KeyN,
                            'o' => Code::KeyO,
                            'p' => Code::KeyP,
                            'q' => Code::KeyQ,
                            'r' => Code::KeyR,
                            's' => Code::KeyS,
                            't' => Code::KeyT,
                            'u' => Code::KeyU,
                            'v' => Code::KeyV,
                            'w' => Code::KeyW,
                            'x' => Code::KeyX,
                            'y' => Code::KeyY,
                            'z' => Code::KeyZ,
                            '0' => Code::Digit0,
                            '1' => Code::Digit1,
                            '2' => Code::Digit2,
                            '3' => Code::Digit3,
                            '4' => Code::Digit4,
                            '5' => Code::Digit5,
                            '6' => Code::Digit6,
                            '7' => Code::Digit7,
                            '8' => Code::Digit8,
                            '9' => Code::Digit9,
                            other => anyhow::bail!("unsupported key: {}", other),
                        }
                    }
                    other => anyhow::bail!("unsupported key token: {}", other),
                });
            }
        }
    }
    let code = code.ok_or_else(|| anyhow::anyhow!("hotkey missing a key"))?;
    Ok(Shortcut::new(Some(mods), code))
}

fn rebind_hotkey(app: &AppHandle, spec: &str) -> anyhow::Result<()> {
    let gs = app.global_shortcut();
    gs.unregister_all()?;
    let shortcut = parse_shortcut(spec)?;
    gs.register(shortcut)?;
    Ok(())
}

fn transcribe_and_paste(
    app: AppHandle,
    state: Arc<AppState>,
    samples: Vec<f32>,
    sample_rate: u32,
) {
    tauri::async_runtime::spawn(async move {
        let state_for_async = state.clone();
        let res: anyhow::Result<String> = async {
            let asr = state_for_async
                .asr
                .lock()
                .clone()
                .ok_or_else(|| anyhow::anyhow!("model not ready yet"))?;
            let text = tokio::task::spawn_blocking(move || asr.recognize(&samples, sample_rate))
                .await
                .map_err(|e| anyhow::anyhow!("recognise join error: {e}"))??;
            let mode = state_for_async.settings.load().inject_mode;
            paste::deliver(&app, &text, &mode)?;
            Ok(text)
        }
        .await;
        match res {
            Ok(text) => {
                let _ = app.emit("transcript", text);
            }
            Err(err) => {
                log::error!("dictation failed: {err:#}");
                let _ = app.emit("dictation-error", err.to_string());
            }
        }
        on_session_finished(&app, &state);
    });
}

fn start_session(app: &AppHandle, state: &Arc<AppState>) {
    if state.asr.lock().is_none() {
        let _ = app.emit(
            "dictation-error",
            "Model still downloading — try again in a moment.".to_string(),
        );
        return;
    }
    let vad_path = state.settings.vad_path();
    let session = match streamer::start(&vad_path) {
        Ok(s) => s,
        Err(e) => {
            log::error!("start session failed: {e:#}");
            let _ = app.emit("dictation-error", e.to_string());
            return;
        }
    };
    set_state(app, state, DictationState::Listening);

    // Stash the Session in AppState so a re-press of the hotkey can find &
    // cancel it; the watcher task below takes ownership back when the VAD
    // (or a cancel) produces an outcome.
    {
        let mut slot = state.session.lock();
        let prev = slot.replace(session);
        debug_assert!(prev.is_none(), "session slot was not cleared");
    }

    let app2 = app.clone();
    let state2 = state.clone();
    tauri::async_runtime::spawn(async move {
        let outcome = tokio::task::spawn_blocking({
            let state_for_blocking = state2.clone();
            move || {
                // Take the Session out so a fresh hotkey press finds an empty
                // slot and starts a new session correctly. Dropping the Session
                // here also tears down the VAD watcher thread via its Drop.
                let session = state_for_blocking.session.lock().take();
                session.map(|s| s.outcome_rx.recv().ok()).unwrap_or(None)
            }
        })
        .await
        .ok()
        .flatten();
        match outcome {
            Some(Outcome::Speech {
                samples,
                sample_rate,
            }) => {
                set_state(&app2, &state2, DictationState::Transcribing);
                transcribe_and_paste(app2, state2, samples, sample_rate);
            }
            Some(Outcome::Cancelled) => {
                on_session_finished(&app2, &state2);
            }
            Some(Outcome::NoSpeech) => {
                let _ = app2.emit(
                    "dictation-error",
                    "No speech detected — try pressing the hotkey closer to talking.".to_string(),
                );
                on_session_finished(&app2, &state2);
            }
            Some(Outcome::Error(e)) => {
                log::error!("session error: {e:#}");
                let _ = app2.emit("dictation-error", e.to_string());
                on_session_finished(&app2, &state2);
            }
            None => {
                on_session_finished(&app2, &state2);
            }
        }
    });
}

fn on_hotkey(app: &AppHandle, state: &Arc<AppState>) {
    // If a session is already running, the hotkey is a cancel.
    let active = state.session.lock().is_some();
    if active {
        if let Some(s) = state.session.lock().as_ref() {
            s.cancel();
        }
        // The watcher task will fire Outcome::Cancelled which clears state.
        return;
    }
    start_session(app, state);
}

fn on_session_finished(app: &AppHandle, state: &Arc<AppState>) {
    let new_state = if state.asr.lock().is_some() {
        DictationState::Idle
    } else {
        DictationState::ModelLoading
    };
    set_state(app, state, new_state);
}

fn build_tray(app: &AppHandle) -> tauri::Result<TrayHandles> {
    let status_item = MenuItem::with_id(app, "status", "Model: loading…", false, None::<&str>)?;
    let separator_1 = PredefinedMenuItem::separator(app)?;
    let toggle_item =
        MenuItem::with_id(app, "toggle", "Start Dictation", true, None::<&str>)?;
    let separator_2 = PredefinedMenuItem::separator(app)?;
    let settings_item = MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?;
    let separator_3 = PredefinedMenuItem::separator(app)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit Parakeet", true, None::<&str>)?;

    let menu = Menu::with_items(
        app,
        &[
            &status_item,
            &separator_1,
            &toggle_item,
            &separator_2,
            &settings_item,
            &separator_3,
            &quit_item,
        ],
    )?;

    // Initial icon: idle mic outline. If SF Symbols are unavailable for some
    // reason (e.g. macOS < 11), TrayIconBuilder requires *some* icon, so we
    // build it later and call set_icon. With macOS 11+ (our minimumSystem
    // is 11.0 per tauri.conf.json) this branch should never miss.
    let initial_icon = sf_symbol::load("mic", 18.0)
        .ok_or_else(|| tauri::Error::AssetNotFound("SF Symbol 'mic' missing".into()))?;

    let tray = TrayIconBuilder::with_id("main")
        .tooltip("Parakeet")
        .icon(initial_icon)
        .icon_as_template(true)
        .menu(&menu)
        .on_menu_event(|app, ev| match ev.id.as_ref() {
            "toggle" => {
                if let Some(state) = app.try_state::<Arc<AppState>>() {
                    on_hotkey(app, &state);
                }
            }
            "settings" => {
                if let Some(win) = app.get_webview_window("settings") {
                    let _ = win.show();
                    let _ = win.set_focus();
                }
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .build(app)?;
    Ok(TrayHandles {
        tray,
        status_item,
        toggle_item,
    })
}

/// Number of performance cores. M5 Pro = 10P/5E, so sherpa-onnx num_threads
/// should target the P-cores only.
fn performance_core_count() -> i32 {
    // sysctlbyname("hw.perflevel0.logicalcpu") returns P-core count on Apple
    // Silicon; falls back to half of total logicals if unavailable.
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let name = std::ffi::CString::new("hw.perflevel0.logicalcpu").unwrap();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        value
    } else {
        (num_cpus_total() / 2).max(2) as i32
    }
}

fn num_cpus_total() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn spawn_model_setup(app: AppHandle, state: Arc<AppState>) {
    tauri::async_runtime::spawn(async move {
        let model_dir = state.settings.model_dir();
        let vad_path = state.settings.vad_path();
        if !state.settings.model_present() {
            if let Err(e) = model_fetch::ensure_model(&app, &model_dir, &vad_path).await {
                log::error!("model fetch failed: {e:#}");
                let _ = app.emit("model-status", format!("Model download failed: {e}"));
                return;
            }
        }

        let encoder_path = state.settings.encoder_path();
        let decoder_path = state.settings.decoder_path();
        let joiner_path = state.settings.joiner_path();
        let tokens_path = state.settings.tokens_path();

        // Move heavy CPU work to a blocking task so we don't park the runtime.
        let app2 = app.clone();
        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Arc<Asr>> {
            // ds4-style page-touch: pull weights into the page cache. The
            // encoder is the bulk of the model (~620 MB of the 640 MB total);
            // touching just it captures most of the cold-read tax.
            let _ = app2.emit("model-status", "Warming page cache…");
            let _ = warmup::page_touch(&encoder_path)?;

            let _ = app2.emit("model-status", "Loading recognizer (CoreML)…");
            let threads = performance_core_count();
            log::info!("OfflineRecognizer: {threads} threads, provider=coreml");
            let asr = Asr::load(&encoder_path, &decoder_path, &joiner_path, &tokens_path, threads)?;

            let _ = app2.emit("model-status", "Pre-warming graph…");
            warmup::dummy_decode(&asr)?;

            Ok(Arc::new(asr))
        })
        .await;

        match result {
            Ok(Ok(asr)) => {
                *state.asr.lock() = Some(asr);
                let _ = app.emit("model-status", "Ready.");
                set_state(&app, &state, DictationState::Idle);
            }
            Ok(Err(e)) => {
                log::error!("model setup failed: {e:#}");
                let _ = app.emit("model-status", format!("Setup failed: {e}"));
            }
            Err(e) => {
                log::error!("setup task panicked: {e}");
                let _ = app.emit("model-status", format!("Setup panicked: {e}"));
            }
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        if let Some(state) = app.try_state::<Arc<AppState>>() {
                            on_hotkey(app, &state);
                        }
                    }
                })
                .build(),
        )
        .setup(|app| {
            let store = SettingsStore::new(app.handle().clone())?;
            let initial = store.load();
            let state = Arc::new(AppState {
                session: Mutex::new(None),
                settings: store,
                asr: Mutex::new(None),
                tray: Mutex::new(None),
                current_state: Mutex::new(DictationState::ModelLoading),
            });
            app.manage(state.clone());

            if let Err(e) = rebind_hotkey(app.handle(), &initial.hotkey) {
                log::warn!("could not register hotkey {}: {e:#}", initial.hotkey);
            }

            let handles = build_tray(app.handle())?;
            *state.tray.lock() = Some(handles);
            refresh_tray(app.handle(), &state);
            spawn_model_setup(app.handle().clone(), state);
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "settings" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_settings, save_settings])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
