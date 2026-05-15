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
mod hotkey;
mod menubar;
mod model_fetch;
mod paste;
mod performance;
mod qos;
mod settings;
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

    // AppKit requires its first contact to be on the main thread. Rust's
    // entry point already is.
    let mtm = MainThreadMarker::new().context("main() must run on the main thread")?;

    // Build the app state up-front so the menu-action handlers can reach it
    // via the AppHandle singleton.
    let settings = SettingsStore::new().context("init settings store")?;
    let app = Arc::new(App::new(settings));
    AppHandle::set(app.clone())
        .map_err(|_| anyhow::anyhow!("AppHandle already initialised"))?;

    // Status-bar menu (uses sf_symbol::load internally).
    menubar::install(mtm).context("install menu bar")?;

    // Hotkey: a ⌘⇧Space press calls App::on_hotkey.
    let app_for_hotkey = app.clone();
    let _hotkey = hotkey::register(
        &app.settings.load().hotkey,
        Arc::new(move || app_for_hotkey.on_hotkey()),
    )
    .context("register global hotkey")?;
    // Hold the handle for the program's lifetime by leaking it. Cleaner
    // than threading a "shutdown" channel into `App` for personal use.
    std::mem::forget(_hotkey);

    // Tokio runtime drives the model download + spawn_blocking ASR work.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("parakeet-tokio")
        .build()
        .context("build tokio runtime")?;
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
