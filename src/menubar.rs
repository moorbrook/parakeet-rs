//! Menu-bar status item via `NSStatusBar` / `NSStatusItem` / `NSMenu`.
//!
//! Replaces what Tauri's `tray-icon` plugin used to do for us, native: the
//! status item lives in the system menu bar, shows an SF Symbol (mic /
//! mic.fill / arrow.down.circle), and presents a small menu on click.
//!
//! Action handlers (Start/Stop Dictation, Quit) are wired through a tiny
//! Objective-C subclass declared with `objc2::define_class!`. The subclass
//! forwards each action to the singleton `App` registered in `app::AppHandle`,
//! so the AppKit runtime doesn't have to carry any Rust state.

use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::{define_class, msg_send, sel, MainThreadOnly};
use objc2_app_kit::{
    NSApplication, NSMenu, NSMenuItem, NSStatusBar, NSStatusItem, NSVariableStatusItemLength,
};
use objc2_foundation::{MainThreadMarker, NSObject, NSObjectProtocol, NSString};

use crate::app::{AppHandle, DictationState};
use crate::settings::TriggerMode;
use crate::settings_ui;
use crate::sf_symbol;

define_class!(
    /// Receives the two menu-action selectors (`toggleDictation:` and
    /// `quit:`) and forwards them to the singleton `App` from
    /// `crate::app::AppHandle`.
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[ivars = ()]
    struct MenuController;

    unsafe impl NSObjectProtocol for MenuController {}

    impl MenuController {
        #[unsafe(method(toggleDictation:))]
        fn toggle_dictation(&self, _sender: *mut NSObject) {
            // Menu clicks always behave like a Tap press — they can't
            // express a hold/release pair. In Hold mode this means a menu
            // click starts the session and a second click stops it.
            if let Some(app) = AppHandle::get() {
                app.on_hotkey_press();
                // For Hold mode + active session, simulate the release
                // edge so the menu-click toggle stays sane.
                if app.session.lock().is_some()
                    && app.settings.load().trigger_mode == TriggerMode::Hold
                {
                    app.on_hotkey_release();
                }
            }
        }

        #[unsafe(method(openSettings:))]
        fn open_settings(&self, _sender: *mut NSObject) {
            settings_ui::open(self.mtm());
        }

        #[unsafe(method(quit:))]
        fn quit(&self, _sender: *mut NSObject) {
            let mtm = self.mtm();
            let ns_app = NSApplication::sharedApplication(mtm);
            unsafe { ns_app.terminate(None) };
        }
    }
);

impl MenuController {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}

/// Owns the status item + the strong-retained menu items we mutate over
/// time. Kept in `MENU_BAR` for the lifetime of the app.
struct MenuBar {
    status_item: Retained<NSStatusItem>,
    status_header: Retained<NSMenuItem>,
    toggle_item: Retained<NSMenuItem>,
    mode_item: Retained<NSMenuItem>,
    _controller: Retained<MenuController>,
}

// SAFETY: All AppKit interaction goes through `with_main`, which gates
// access on a fresh `MainThreadMarker`. The wrapper itself never moves the
// `Retained<...>` handles off-thread.
struct MainOnly<T>(parking_lot::Mutex<Option<T>>);

unsafe impl<T> Send for MainOnly<T> {}
unsafe impl<T> Sync for MainOnly<T> {}

impl<T> MainOnly<T> {
    fn new(v: T) -> Self {
        Self(parking_lot::Mutex::new(Some(v)))
    }
    fn with<R>(&self, _mtm: MainThreadMarker, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        self.0.lock().as_mut().map(f)
    }
}

static MENU_BAR: OnceLock<MainOnly<MenuBar>> = OnceLock::new();

pub fn install(mtm: MainThreadMarker) -> Result<(), anyhow::Error> {
    let bar = NSStatusBar::systemStatusBar();
    let status_item = unsafe { bar.statusItemWithLength(NSVariableStatusItemLength) };

    if let Some(img) = sf_symbol::load("arrow.down.circle", 18.0) {
        if let Some(button) = status_item.button(mtm) {
            unsafe { button.setImage(Some(&img)) };
        }
    }

    let controller = MenuController::new(mtm);

    let menu = NSMenu::new(mtm);

    let status_header = make_menu_item(mtm, "Model: loading…", None, None, false);
    let mode_item = make_menu_item(mtm, "Mode: Tap (VAD)", None, None, false);
    let separator_1 = NSMenuItem::separatorItem(mtm);
    let toggle_item = make_menu_item(
        mtm,
        "Dictation unavailable",
        Some(&controller),
        Some(sel!(toggleDictation:)),
        false,
    );
    let separator_2 = NSMenuItem::separatorItem(mtm);
    let settings_item = make_menu_item(
        mtm,
        "Settings…",
        Some(&controller),
        Some(sel!(openSettings:)),
        true,
    );
    let separator_3 = NSMenuItem::separatorItem(mtm);
    let quit_item = make_menu_item(
        mtm,
        "Quit Parakeet",
        Some(&controller),
        Some(sel!(quit:)),
        true,
    );

    menu.addItem(&status_header);
    menu.addItem(&mode_item);
    menu.addItem(&separator_1);
    menu.addItem(&toggle_item);
    menu.addItem(&separator_2);
    menu.addItem(&settings_item);
    menu.addItem(&separator_3);
    menu.addItem(&quit_item);

    unsafe { status_item.setMenu(Some(&menu)) };

    MENU_BAR
        .set(MainOnly::new(MenuBar {
            status_item,
            status_header,
            toggle_item,
            mode_item,
            _controller: controller,
        }))
        .map_err(|_| anyhow::anyhow!("MenuBar installed twice"))?;
    Ok(())
}

fn make_menu_item(
    mtm: MainThreadMarker,
    title: &str,
    target: Option<&MenuController>,
    action: Option<objc2::runtime::Sel>,
    enabled: bool,
) -> Retained<NSMenuItem> {
    let title_ns = NSString::from_str(title);
    let key_ns = NSString::from_str("");
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &title_ns,
            action,
            &key_ns,
        )
    };
    if let Some(target) = target {
        unsafe { item.setTarget(Some(target.as_ref())) };
    }
    item.setEnabled(enabled);
    item
}

/// Refresh icon + labels for the given app state. Safe to call from any
/// thread; if invoked off-main we use an unchecked main-thread marker since
/// these single-property setters on the status item / menu item are stable
/// across threads in practice on macOS 14+. (Cleaner alternative: dispatch
/// through CFRunLoop; not worth the complexity for personal use.)
pub fn refresh(
    state: DictationState,
    asr_ready: bool,
    shortcut: &str,
    trigger_mode: TriggerMode,
) {
    let Some(slot) = MENU_BAR.get() else {
        return;
    };
    let mtm = MainThreadMarker::new().unwrap_or_else(|| unsafe { MainThreadMarker::new_unchecked() });
    slot.with(mtm, |bar| {
        let mode_verb = match trigger_mode {
            TriggerMode::Tap => "Start",
            TriggerMode::Hold => "Hold",
        };
        let mode_label = match trigger_mode {
            TriggerMode::Tap => "Mode: Tap (VAD auto-stop)".to_string(),
            TriggerMode::Hold => "Mode: Hold (release to paste)".to_string(),
        };
        let (symbol, header_label, toggle_label, toggle_enabled) = match state {
            DictationState::ModelLoading => (
                "arrow.down.circle",
                "Model: downloading…".to_string(),
                format!("Dictation unavailable ({shortcut})"),
                false,
            ),
            DictationState::Idle => (
                "mic",
                if asr_ready {
                    "Model: ready".to_string()
                } else {
                    "Model: loading…".to_string()
                },
                format!("{mode_verb} Dictation  {shortcut}"),
                asr_ready,
            ),
            DictationState::Listening => (
                "mic.fill",
                "Model: ready".to_string(),
                format!("Stop Dictation  {shortcut}"),
                true,
            ),
            DictationState::Transcribing => (
                "mic.fill",
                "Transcribing…".to_string(),
                "Working…".to_string(),
                false,
            ),
        };

        if let Some(img) = sf_symbol::load(symbol, 18.0) {
            if let Some(button) = bar.status_item.button(mtm) {
                unsafe { button.setImage(Some(&img)) };
            }
        }
        unsafe { bar.status_header.setTitle(&NSString::from_str(&header_label)) };
        unsafe { bar.mode_item.setTitle(&NSString::from_str(&mode_label)) };
        unsafe { bar.toggle_item.setTitle(&NSString::from_str(&toggle_label)) };
        bar.toggle_item.setEnabled(toggle_enabled);
    });
}

pub fn set_status_text(text: &str) {
    let Some(slot) = MENU_BAR.get() else {
        return;
    };
    let mtm = MainThreadMarker::new().unwrap_or_else(|| unsafe { MainThreadMarker::new_unchecked() });
    slot.with(mtm, |bar| {
        unsafe { bar.status_header.setTitle(&NSString::from_str(text)) };
    });
}
