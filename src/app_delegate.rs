//! `NSApplicationDelegate` for parakeet-rs.
//!
//! Owns the application lifecycle hooks (`applicationDidFinishLaunching:`,
//! `applicationShouldHandleReopen:hasVisibleWindows:`,
//! `applicationWillTerminate:`). All AppKit installation that used to
//! live inline in `main.rs` now happens in `didFinishLaunching` so the
//! NSApplication has its run loop spun up before we register status
//! items, HUD panels, and global hotkeys.
//!
//! State (App, Settings, Hotkey handle) is reached via the
//! `crate::app::AppHandle` singleton, set in `main.rs` before the
//! delegate is installed. Keeping ivars empty avoids a retain-cycle
//! foot-gun between the delegate and the `Arc<App>` it would otherwise
//! reference.

use std::sync::Arc;

use anyhow::Context;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, MainThreadOnly};
use objc2_app_kit::{NSApplication, NSApplicationDelegate};
use objc2_foundation::{MainThreadMarker, NSNotification, NSObject, NSObjectProtocol};

use crate::app::AppHandle;
use crate::{hotkey, hud, menubar};

define_class!(
    /// Lives for the life of the process. AppKit holds the only strong
    /// reference (via `setDelegate:`); we keep our `Retained<AppDelegate>`
    /// alive in `main` via `mem::forget` for the same reason the tokio
    /// runtime is kept alive — dropping it would deallocate the delegate
    /// mid-event-loop.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "ParakeetAppDelegate"]
    #[ivars = ()]
    pub struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        /// Fires once, after the NSApplication run loop starts. This is
        /// the canonical place to install status items, HUD panels, and
        /// the global hotkey — before this, calling `[NSStatusBar
        /// systemStatusBar]` works but the resulting items don't render
        /// until the run loop is spinning.
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn application_did_finish_launching(&self, _notification: &NSNotification) {
            crate::objc_util::selector_guard("applicationDidFinishLaunching:", || {
                let mtm = match MainThreadMarker::new() {
                    Some(m) => m,
                    None => {
                        log::error!("delegate fired off main thread (impossible)");
                        return;
                    }
                };
                if let Err(e) = install_runtime_state(mtm) {
                    log::error!("applicationDidFinishLaunching install failed: {e:#}");
                }
            });
        }

        /// Dock-icon click or `open -a Parakeet.app` while the app is
        /// already running. Without an explicit hook, AppKit's default
        /// is to do nothing for an LSUIElement (no Dock icon, no
        /// windows) — which means the user has no way to surface the
        /// Settings window. We open it.
        ///
        /// `has_visible_windows: false` is the case that matters; with
        /// any visible window the user can already reach Settings via
        /// the menu bar.
        #[unsafe(method(applicationShouldHandleReopen:hasVisibleWindows:))]
        fn application_should_handle_reopen(
            &self,
            _sender: &NSApplication,
            has_visible_windows: bool,
        ) -> bool {
            crate::objc_util::selector_guard("applicationShouldHandleReopen:", || {
                if has_visible_windows {
                    return;
                }
                let Some(mtm) = MainThreadMarker::new() else {
                    return;
                };
                crate::settings_ui::open(mtm);
            });
            // Returning false tells AppKit "I handled it, don't do the
            // default behaviour". Default for an LSUIElement is no-op
            // anyway; explicit is clearer.
            false
        }

        /// Display reconfiguration: monitor un/replug, resolution
        /// change, side-of-mountain wakeup, etc. Re-position the HUD
        /// in case its old coordinates are now off-screen (e.g. the
        /// external monitor it was anchored to is gone).
        #[unsafe(method(applicationDidChangeScreenParameters:))]
        fn application_did_change_screen_parameters(&self, _notification: &NSNotification) {
            crate::objc_util::selector_guard("applicationDidChangeScreenParameters:", || {
                let Some(mtm) = MainThreadMarker::new() else {
                    return;
                };
                crate::hud::reposition_on_screen(mtm);
            });
        }

        /// Fires when the user invokes `quit:` from our menu, the
        /// hotkey rebind triggers a relaunch, or the OS asks us to
        /// shut down. Drop any active dictation session cleanly so
        /// the audio thread tears down instead of being abruptly
        /// killed mid-buffer.
        #[unsafe(method(applicationWillTerminate:))]
        fn application_will_terminate(&self, _notification: &NSNotification) {
            crate::objc_util::selector_guard("applicationWillTerminate:", || {
                if let Some(app) = AppHandle::get() {
                    // Drop the session if one is in flight. Dropping the
                    // Session sends Signal::Cancel through the watcher
                    // channel and joins the audio thread — see
                    // `streamer::Session::Drop`.
                    let _ = app.session.lock().take();
                }
                log::info!("parakeet-rs terminating");
            });
        }
    }
);

impl AppDelegate {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}

/// Body of `applicationDidFinishLaunching:` — pulled out so the
/// selector body stays small and the error-handling path is testable.
/// Order matches the previous inline-in-main.rs setup.
fn install_runtime_state(mtm: MainThreadMarker) -> anyhow::Result<()> {
    let app = AppHandle::get().context("AppHandle not initialised before delegate fired")?;

    menubar::install(mtm).context("install menu bar")?;
    hud::install(mtm);

    // Hotkey: press/release edges call App::on_hotkey_press / on_hotkey_release.
    // In Tap mode only press matters; in Hold mode release is the commit edge.
    let app_for_press = Arc::clone(&app);
    let app_for_release = Arc::clone(&app);
    let hotkey_handle = hotkey::register(
        &app.settings.load().hotkey,
        Arc::new(move || app_for_press.on_hotkey_press()),
        Arc::new(move || app_for_release.on_hotkey_release()),
        mtm,
    )
    .context("register global hotkey")?;
    *app.hotkey.lock() = Some(hotkey_handle);

    // Tokio runtime drives the model download + spawn_blocking ASR
    // load. Forgotten on purpose — see `main.rs` for the lifetime
    // explanation.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("parakeet-tokio")
        .build()
        .context("build tokio runtime")?;
    {
        let app = Arc::clone(&app);
        runtime.spawn(async move {
            app.spawn_model_setup().await;
        });
    }
    std::mem::forget(runtime);

    // Initial menu paint reflecting "model loading" state.
    app.refresh_menu();

    log::info!("parakeet-rs runtime installed");
    Ok(())
}

/// Build, retain, and install the delegate on `ns_app`. Caller must
/// `mem::forget` the returned `Retained` (or otherwise keep it alive)
/// so the delegate isn't released the moment `main` returns from setup.
pub fn install(ns_app: &NSApplication, mtm: MainThreadMarker) -> Retained<AppDelegate> {
    let delegate = AppDelegate::new(mtm);
    let proto = ProtocolObject::from_ref(&*delegate);
    unsafe {
        ns_app.setDelegate(Some(proto));
    }
    delegate
}
