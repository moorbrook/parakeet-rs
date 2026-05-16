//! parakeet-rs entry point.
//!
//! Single-binary, no Tauri, no WebKit. Sets up the `NSApplication`,
//! installs the `NSApplicationDelegate` (which owns all post-launch
//! AppKit installation — see `src/app_delegate.rs`), and runs the
//! AppKit event loop forever.

// Same TEMPORARY allows as `lib.rs` — remove when the per-site
// audit completes (see `lib.rs` module doc).
#![allow(unused_unsafe)]
#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::Arc;

use anyhow::{Context, Result};
use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
use objc2_foundation::MainThreadMarker;

use parakeet_rs::app::{App, AppHandle};
use parakeet_rs::settings::SettingsStore;
use parakeet_rs::{app_delegate, objc_util, permissions};

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
    //
    // Runs BEFORE the delegate is installed so a permissions-missing
    // exit happens cleanly without the partial post-launch setup the
    // delegate would have done.
    let missing = permissions::check_all();
    if !missing.is_empty() {
        permissions::present_missing_alert_blocking(mtm, &missing);
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

    // Build the app state up-front so the delegate (and the menu-action
    // selectors it transitively wires up) can reach it via the
    // AppHandle singleton.
    let settings = SettingsStore::new().context("init settings store")?;
    let app = Arc::new(App::new(settings));
    AppHandle::set(app.clone()).map_err(|_| anyhow::anyhow!("AppHandle already initialised"))?;

    // Install the delegate. All menubar / hud / hotkey / model-fetch
    // setup now happens inside `applicationDidFinishLaunching:`, which
    // AppKit fires once the run loop is spinning.
    let ns_app = NSApplication::sharedApplication(mtm);
    let delegate = app_delegate::install(&ns_app, mtm);
    // The delegate must outlive the rest of main(). AppKit holds the
    // only other reference via setDelegate:; forgetting our Retained
    // makes sure it survives until process exit.
    std::mem::forget(delegate);

    // Become a UI-but-not-Dock agent and enter the AppKit run loop.
    // All three methods are `safe` on objc2-app-kit 0.3.
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
    ns_app.activate();
    ns_app.run();

    Ok(())
}
