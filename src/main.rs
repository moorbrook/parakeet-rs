//! Parakeet Dictation entry point.
//!
//! Single-binary, no Tauri, no WebKit. Sets up the `NSApplication`,
//! installs the `NSApplicationDelegate` (which owns all post-launch
//! AppKit installation — see `src/app_delegate.rs`), and runs the
//! AppKit event loop forever.

use std::sync::Arc;

use anyhow::{Context, Result};
use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
use objc2_foundation::MainThreadMarker;

use parakeet_dictation::app::{App, AppHandle};
use parakeet_dictation::settings::SettingsStore;
use parakeet_dictation::{app_delegate, objc_util};

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
