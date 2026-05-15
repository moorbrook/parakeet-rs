// objc2 0.6 marked many AppKit methods safe that 0.5 left `unsafe`. Some of
// our wrapper blocks therefore now look unnecessary, but they document the
// boundary where Rust ↔ Objective-C lives. Suppress the lint at the crate
// level instead of editing every call site, since the next objc2 minor bump
// may re-tighten the safety annotations.
#![allow(unused_unsafe)]

//! parakeet-rs entry point.
//!
//! Single-binary, no Tauri, no WebKit. Sets up an `NSApplication` as a
//! menu-bar agent (no Dock icon, no main menu strip), installs the
//! `NSStatusItem`, registers the global hotkey, kicks off the model setup
//! on a tokio runtime, and runs the AppKit event loop forever.

mod app;
mod asr;
mod audio;
mod cleanup;
mod hotkey;
mod hud;
mod keychain;
mod menubar;
mod model_fetch;
mod objc_util;
mod paste;
mod performance;
mod permissions;
mod qos;
mod settings;
mod settings_ui;
mod sf_symbol;
mod streamer;
mod vad;
mod warmup;

use std::sync::Arc;

use anyhow::{Context, Result};
use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
use objc2_foundation::MainThreadMarker;

use crate::app::{App, AppHandle};
use crate::settings::SettingsStore;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Log panics before the (release-mode) abort handler eats them. This
    // is the only feedback channel for a Finder-launched LSUIElement app
    // that aborts on panic — stderr from a double-clicked .app is
    // visible only in Console.app.
    objc_util::install_panic_hook();

    // AppKit requires its first contact to be on the main thread. Rust's
    // entry point already is.
    let mtm = MainThreadMarker::new().context("main() must run on the main thread")?;

    // Pre-flight TCC permissions: Microphone, Accessibility, Input
    // Monitoring. Missing permissions show a native NSAlert (the previous
    // stderr-then-exit path was invisible for an LSUIElement app launched
    // from Finder). The alert lets the user open each System Settings
    // pane directly, then Quit — and we relaunch on the next attempt.
    let missing = permissions::check_all();
    if !missing.is_empty() {
        permissions::present_missing_alert_blocking(mtm, &missing);
        // Surface the same info on stderr for the curious / for CI.
        eprintln!(
            "Parakeet exiting: missing permission(s): {}",
            missing
                .iter()
                .map(|p| p.label())
                .collect::<Vec<_>>()
                .join(", ")
        );
        std::process::exit(1);
    }

    // Build the app state up-front so the menu-action handlers can reach it
    // via the AppHandle singleton.
    let settings = SettingsStore::new().context("init settings store")?;
    let app = Arc::new(App::new(settings));
    AppHandle::set(app.clone()).map_err(|_| anyhow::anyhow!("AppHandle already initialised"))?;

    // Status-bar menu (uses sf_symbol::load internally).
    menubar::install(mtm).context("install menu bar")?;
    // Recording-state HUD overlay. Hidden until the first state change.
    hud::install(mtm);

    // Hotkey: press/release edges call App::on_hotkey_press / on_hotkey_release.
    // In Tap mode only press matters; in Hold mode release is the commit edge.
    // `mtm` is required because the NSEvent global monitor for media keys
    // has to be installed on the main thread.
    let app_for_press = app.clone();
    let app_for_release = app.clone();
    let hotkey_handle = hotkey::register(
        &app.settings.load().hotkey,
        Arc::new(move || app_for_press.on_hotkey_press()),
        Arc::new(move || app_for_release.on_hotkey_release()),
        mtm,
    )
    .context("register global hotkey")?;
    // Stash the handle in AppState so the Settings UI can call `rebind`
    // when the user picks a new combo.
    *app.hotkey.lock() = Some(hotkey_handle);

    // Tokio runtime drives the model download + spawn_blocking ASR work
    // + the cleanup-pass HTTP call. Stash the handle on `App` so the
    // synchronous transcribe thread can `block_on` the cleanup call
    // without spinning up a second runtime per dictation.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("parakeet-tokio")
        .build()
        .context("build tokio runtime")?;
    *app.tokio_handle.lock() = Some(runtime.handle().clone());
    {
        let app = app.clone();
        runtime.spawn(async move {
            app.spawn_model_setup().await;
        });
    }
    // Keep the runtime alive for the life of the process. Without this,
    // dropping at the end of `main` would tear down our async tasks.
    std::mem::forget(runtime);

    // Initial menu paint reflecting "model loading" state.
    app.refresh_menu();

    // Become a UI-but-not-Dock agent and enter the AppKit run loop.
    let ns_app = NSApplication::sharedApplication(mtm);
    unsafe {
        ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
        ns_app.activate();
        ns_app.run();
    }

    Ok(())
}
