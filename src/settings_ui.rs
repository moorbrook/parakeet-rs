//! Native NSWindow Settings UI.
//!
//! Opened from the menu bar's "Settings…" item. Lets the user:
//!
//! - Pick a new global hotkey by clicking the shortcut button and pressing
//!   any combo (the window catches the keystroke locally, doesn't fire the
//!   dictation flow).
//! - Switch between Tap (current VAD auto-stop) and Hold (press-and-hold,
//!   release-to-paste) trigger modes.
//!
//! Save persists to `~/Library/Application Support/com.parakeet.rs/settings.json`
//! AND rebinds the global hotkey live without restarting the app.
//!
//! Threading: every method here must run on the main thread. Use the
//! `open(mtm)` entry point — it's called from a `#[unsafe(method)]` on the
//! menu controller, which AppKit guarantees runs on main.

use std::cell::RefCell;

use dispatch2::DispatchQueue;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, sel, DefinedClass, MainThreadOnly};
use objc2_app_kit::{
    NSApplication, NSBackingStoreType, NSButton, NSColor, NSEvent, NSEventModifierFlags,
    NSEventType, NSPopUpButton, NSResponder, NSTextField, NSView, NSWindow, NSWindowDelegate,
    NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize, NSString,
};

use crate::app::{glyphs_for_shortcut, is_capslock_token, AppHandle};
use crate::settings::{CleanupMode, TriggerMode};

const WINDOW_W: f64 = 480.0;
const WINDOW_H: f64 = 440.0;
const ROW_H: f64 = 26.0;
const LABEL_W: f64 = 130.0;
const PAD: f64 = 20.0;

/// Models offered in the cleanup-model popup. Index matches the popup
/// menu item position; we keep both the user-visible label and the API
/// model id paired so the order is the only source of truth.
const CLEANUP_MODELS: &[(&str, &str)] = &[
    ("Haiku 4.5 (fast, recommended)", "claude-haiku-4-5"),
    ("Sonnet 4.6 (best quality)", "claude-sonnet-4-6"),
];

#[derive(Default)]
struct Ivars {
    /// Built-up token like `"CmdOrCtrl+Shift+Space"` while the user is
    /// recording a new combo. `None` outside of recording state.
    recording_token: RefCell<Option<String>>,
    shortcut_button: RefCell<Option<Retained<NSButton>>>,
    mode_popup: RefCell<Option<Retained<NSPopUpButton>>>,
    cleanup_popup: RefCell<Option<Retained<NSPopUpButton>>>,
    cleanup_model_popup: RefCell<Option<Retained<NSPopUpButton>>>,
    api_key_field: RefCell<Option<Retained<NSTextField>>>,
    // No window field. Storing a strong `Retained<NSWindow>` here while
    // the window also held a strong `Retained<SettingsController>` was a
    // textbook retain cycle that prevented either object from ever being
    // freed. Both lifetimes are now owned by `LIVE_SETTINGS` (a single
    // thread-local on the main thread) and routed through helpers.
}

/// Single-instance handle to the live settings window + its controller.
/// Held in a `thread_local!` because both objects are AppKit and must only
/// be touched from the main thread — that constraint also makes the
/// "exactly one instance" semantics automatic.
struct LiveSettings {
    controller: Retained<SettingsController>,
    window: Retained<RecordingWindow>,
}

thread_local! {
    static LIVE_SETTINGS: RefCell<Option<LiveSettings>> = const { RefCell::new(None) };
}

fn with_live_controller<R>(f: impl FnOnce(&SettingsController) -> R) -> Option<R> {
    LIVE_SETTINGS.with(|slot| slot.borrow().as_ref().map(|l| f(&l.controller)))
}

fn drop_live_settings() {
    LIVE_SETTINGS.with(|slot| {
        // Drop both handles in one shot — taking out of the RefCell first
        // so any windowWillClose callback re-entrant through here finds an
        // empty slot and short-circuits.
        let _ = slot.borrow_mut().take();
    });
}

define_class!(
    /// Window delegate + action target. One per Settings window.
    #[unsafe(super = NSResponder)]
    #[thread_kind = MainThreadOnly]
    #[ivars = Ivars]
    struct SettingsController;

    unsafe impl NSObjectProtocol for SettingsController {}

    unsafe impl NSWindowDelegate for SettingsController {
        #[unsafe(method(windowWillClose:))]
        fn window_will_close(&self, _note: &NSNotification) {
            crate::objc_util::selector_guard("windowWillClose:", || {
                // Defer the drop. AppKit is mid-dispatch on `self`
                // here; if this happens to be the call path that also
                // drops the only strong ref (e.g. user clicked the red
                // close button rather than Cancel), releasing
                // synchronously would dangle the receiver pointer ObjC
                // is still using. Bouncing through the main GCD queue
                // postpones the drop until after the selector has
                // fully returned.
                DispatchQueue::main().exec_async(drop_live_settings);
            });
        }
    }

    impl SettingsController {
        /// "Save" button action. Read controls, call App::apply_settings.
        /// Body in `save_inner` so the selector itself is just the
        /// panic-guard wrapper.
        #[unsafe(method(save:))]
        fn save(&self, _sender: *mut NSObject) {
            crate::objc_util::selector_guard("save:", || self.save_inner());
        }

        /// Cleanup-mode popup changed. Enable / disable the API key + model
        /// inputs to make it obvious they only matter when cleanup is on.
        #[unsafe(method(cleanupModeChanged:))]
        fn cleanup_mode_changed(&self, _sender: *mut NSObject) {
            crate::objc_util::selector_guard("cleanupModeChanged:", || {
                self.refresh_cleanup_enabled();
            });
        }

        #[unsafe(method(cancel:))]
        fn cancel(&self, _sender: *mut NSObject) {
            crate::objc_util::selector_guard("cancel:", || self.close_window());
        }

        /// "Record Shortcut" button. Flips the UI into recording state; the
        /// next NSEvent-driven key combo seen by `key_down` becomes the new
        /// hotkey token.
        #[unsafe(method(beginRecording:))]
        fn begin_recording(&self, _sender: *mut NSObject) {
            crate::objc_util::selector_guard("beginRecording:", || {
                *self.ivars().recording_token.borrow_mut() = Some(String::new());
                if let Some(btn) = self.ivars().shortcut_button.borrow().as_ref() {
                    unsafe { btn.setTitle(&NSString::from_str("Press a key combination…")) };
                }
            });
        }

        /// Captured by the custom NSWindow subclass below — see
        /// `RecordingWindow::keyDown`. Converts the NSEvent into a token
        /// like `"CmdOrCtrl+Shift+Space"` and updates the button label.
        #[unsafe(method(captureKey:))]
        fn capture_key(&self, event_obj: *mut NSObject) {
            crate::objc_util::selector_guard("captureKey:", || {
                if self.ivars().recording_token.borrow().is_none() {
                    return;
                }
                // Re-interpret the opaque sender as an NSEvent pointer.
                // The RecordingWindow forwards events through this
                // selector.
                let event: &NSEvent = unsafe { &*(event_obj as *const NSEvent) };
                if let Some(token) = ns_event_to_token(event) {
                    let glyphs = glyphs_for_shortcut(&token);
                    *self.ivars().recording_token.borrow_mut() = Some(token.clone());
                    if let Some(btn) = self.ivars().shortcut_button.borrow().as_ref() {
                        unsafe { btn.setTitle(&NSString::from_str(&glyphs)) };
                    }
                    // Recording a Caps Lock binding flips us into the
                    // locked-Hold UI state so the user understands
                    // their Tap/Hold choice no longer applies.
                    self.refresh_mode_popup_for(&token);
                }
            });
        }
    }
);

impl SettingsController {
    /// Inner body of the `save:` selector. Lives outside `define_class!`
    /// so it isn't ObjC-exposed.
    fn save_inner(&self) {
        let Some(app) = AppHandle::get() else {
            return;
        };
        let mut new = app.settings.load();

        if let Some(token) = self.ivars().recording_token.borrow().clone() {
            // begin_recording seeds recording_token with an empty
            // string. If the user clicks Record then clicks Save
            // without pressing anything, the token is "". Validating
            // here (and also belt-and-braces in App::apply_settings)
            // stops us persisting an unparseable binding that would
            // brick the next launch.
            let trimmed = token.trim();
            if !trimmed.is_empty() && crate::hotkey::parse(trimmed).is_ok() {
                new.hotkey = trimmed.to_string();
            } else if !trimmed.is_empty() {
                log::warn!("ignoring unparseable hotkey token from recorder: {trimmed:?}");
            }
        }
        if let Some(popup) = self.ivars().mode_popup.borrow().as_ref() {
            new.trigger_mode = match unsafe { popup.indexOfSelectedItem() } {
                1 => TriggerMode::Hold,
                _ => TriggerMode::Tap,
            };
        }
        if let Some(popup) = self.ivars().cleanup_popup.borrow().as_ref() {
            new.cleanup_mode = match unsafe { popup.indexOfSelectedItem() } {
                1 => CleanupMode::Anthropic,
                _ => CleanupMode::Off,
            };
        }
        if let Some(field) = self.ivars().api_key_field.borrow().as_ref() {
            let s: Retained<NSString> = unsafe { field.stringValue() };
            new.anthropic_api_key = s.to_string().trim().to_string();
        }
        if let Some(popup) = self.ivars().cleanup_model_popup.borrow().as_ref() {
            let idx = unsafe { popup.indexOfSelectedItem() };
            if let Some((_, id)) = idx
                .try_into()
                .ok()
                .and_then(|i: usize| CLEANUP_MODELS.get(i))
            {
                new.cleanup_model = (*id).to_string();
            }
        }

        if let Err(e) = app.apply_settings(new) {
            log::error!("save settings failed: {e:#}");
        }
        self.close_window();
    }
}

impl SettingsController {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(Ivars::default());
        unsafe { msg_send![super(this), init] }
    }

    /// Lock or unlock the trigger-mode popup based on the active binding.
    /// Caps Lock only fires a single FlagsChanged event per physical tap,
    /// so Tap vs Hold can't make a meaningful runtime difference — we
    /// force Hold semantics and visually grey out the choice.
    fn refresh_mode_popup_for(&self, token: &str) {
        let Some(popup) = self.ivars().mode_popup.borrow().clone() else {
            return;
        };
        if is_capslock_token(token) {
            unsafe {
                popup.selectItemAtIndex(1); // 1 = Hold
                popup.setEnabled(false);
            }
        } else {
            unsafe { popup.setEnabled(true) };
        }
    }

    /// Grey out the API key + model picker unless cleanup mode is on.
    /// Removes the "I typed my key but cleanup is off" footgun.
    fn refresh_cleanup_enabled(&self) {
        let on = match self.ivars().cleanup_popup.borrow().as_ref() {
            Some(popup) => {
                let idx = unsafe { popup.indexOfSelectedItem() };
                idx == 1
            }
            None => false,
        };
        if let Some(field) = self.ivars().api_key_field.borrow().as_ref() {
            unsafe { field.setEnabled(on) };
        }
        if let Some(popup) = self.ivars().cleanup_model_popup.borrow().as_ref() {
            unsafe { popup.setEnabled(on) };
        }
    }

    fn close_window(&self) {
        // Defer the close + drop to the next runloop tick.
        //
        // Why: callers reach `close_window` from inside an ObjC selector
        // on `self`. The only strong ref to `self` lives in
        // LIVE_SETTINGS. Dropping it synchronously here would release
        // `self` while ObjC is still mid-dispatch on it — a textbook
        // use-after-free, even though `self` is just a `&` borrow.
        //
        // `DispatchQueue::main().exec_async` runs the closure on the
        // next runloop iteration, after the current selector has
        // returned. By then ObjC no longer holds the receiver pointer.
        DispatchQueue::main().exec_async(move || {
            // Same take-before-close ordering as before, for the same
            // reason: windowWillClose fires synchronously from `close()`
            // and re-enters `drop_live_settings`; if the slot is already
            // empty that re-entry short-circuits.
            let live = LIVE_SETTINGS.with(|slot| slot.borrow_mut().take());
            if let Some(live) = live {
                unsafe { live.window.close() };
                // `live` drops here; the controller + window both
                // deallocate at this point.
            }
        });
    }
}

// A custom NSWindow that forwards keyDown / flagsChanged events to the
// active SettingsController during shortcut recording. AppKit normally
// swallows un-targeted keys, so we hook them at the window level.
//
// Critically: this subclass does NOT hold a strong reference to the
// controller. It reads the controller from `LIVE_SETTINGS` on each
// event. Storing the controller here strongly was half of the retain
// cycle that previously kept windows alive after close.
define_class!(
    #[unsafe(super = NSWindow)]
    #[thread_kind = MainThreadOnly]
    #[ivars = ()]
    struct RecordingWindow;

    unsafe impl NSObjectProtocol for RecordingWindow {}

    impl RecordingWindow {
        #[unsafe(method(keyDown:))]
        fn key_down(&self, event: &NSEvent) {
            crate::objc_util::selector_guard("RecordingWindow.keyDown:", || {
                forward_to_live_controller(event);
            });
        }

        // Caps Lock arrives via `flagsChanged:`, not `keyDown:`, because
        // the OS treats it as a modifier toggle. We only forward Caps Lock
        // specifically (keycode 57) so a user holding Shift while recording
        // doesn't accidentally commit "Shift" as the binding.
        #[unsafe(method(flagsChanged:))]
        fn flags_changed(&self, event: &NSEvent) {
            crate::objc_util::selector_guard("RecordingWindow.flagsChanged:", || {
                let keycode = unsafe { event.keyCode() };
                if keycode == 57 {
                    forward_to_live_controller(event);
                }
            });
        }
    }
);

fn forward_to_live_controller(event: &NSEvent) {
    let obj: *const NSEvent = event;
    let _ = with_live_controller(|c| unsafe {
        let _: () = msg_send![c, captureKey: obj as *mut NSObject];
    });
}

/// Open the Settings window, or focus the existing one if it's already up.
/// This is the only public entry point; the menu-action selector and any
/// future call site both flow through here.
pub fn open(mtm: MainThreadMarker) {
    // Dedupe: if a window is already live, just bring it back to the front.
    let already_open = LIVE_SETTINGS.with(|slot| slot.borrow().as_ref().map(|l| l.window.clone()));
    if let Some(existing) = already_open {
        let ns_app = NSApplication::sharedApplication(mtm);
        #[allow(deprecated)]
        unsafe {
            ns_app.activateIgnoringOtherApps(true)
        };
        existing.makeKeyAndOrderFront(None);
        return;
    }

    let controller = SettingsController::new(mtm);

    // Window frame: NSWindow expects bottom-left origin. We center after.
    let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(WINDOW_W, WINDOW_H));
    let mask =
        NSWindowStyleMask::Titled | NSWindowStyleMask::Closable | NSWindowStyleMask::Miniaturizable;
    let window: Retained<RecordingWindow> = unsafe {
        let alloc = RecordingWindow::alloc(mtm).set_ivars(());
        msg_send![
            super(alloc),
            initWithContentRect: frame,
            styleMask: mask,
            backing: NSBackingStoreType::Buffered,
            defer: false,
        ]
    };
    unsafe { window.setReleasedWhenClosed(false) };
    window.setTitle(&NSString::from_str("Parakeet Settings"));

    // Build form contents.
    let content = window.contentView().expect("window must have content view");
    let settings = AppHandle::get().map(|a| a.settings.load());

    // --- Row 1: Hotkey -----------------------------------------------------
    let row1_y = WINDOW_H - PAD - ROW_H - 30.0;
    add_label(mtm, &content, "Hotkey", PAD, row1_y);

    let shortcut_initial = settings
        .as_ref()
        .map(|s| glyphs_for_shortcut(&s.hotkey))
        .unwrap_or_else(|| "⌘⇧Space".to_string());
    let shortcut_btn = make_button(
        mtm,
        &shortcut_initial,
        &controller,
        sel!(beginRecording:),
        NSPoint::new(PAD + LABEL_W, row1_y - 4.0),
        NSSize::new(WINDOW_W - PAD * 2.0 - LABEL_W, ROW_H + 4.0),
    );
    unsafe { content.addSubview(&shortcut_btn) };
    *controller.ivars().shortcut_button.borrow_mut() = Some(shortcut_btn);

    // --- Row 2: Trigger mode ----------------------------------------------
    let row2_y = row1_y - ROW_H - 20.0;
    add_label(mtm, &content, "Trigger", PAD, row2_y);

    let popup = make_popup(
        mtm,
        &["Tap — VAD auto-stop", "Hold — release to paste"],
        match settings.as_ref().map(|s| s.trigger_mode) {
            Some(TriggerMode::Hold) => 1,
            _ => 0,
        },
        NSPoint::new(PAD + LABEL_W, row2_y - 4.0),
        NSSize::new(WINDOW_W - PAD * 2.0 - LABEL_W, ROW_H + 4.0),
    );
    unsafe { content.addSubview(&popup) };
    *controller.ivars().mode_popup.borrow_mut() = Some(popup);

    // If the persisted binding is Caps Lock, the trigger-mode popup starts
    // greyed out (locked to Hold). Plain bindings keep the user's choice.
    if let Some(s) = settings.as_ref() {
        controller.refresh_mode_popup_for(&s.hotkey);
    }

    // --- Row 3: Hint text --------------------------------------------------
    let row3_y = row2_y - ROW_H - 10.0;
    add_hint(
        mtm,
        &content,
        "Tap: press once to start, Parakeet stops when you finish speaking.\n\
         Hold: press and hold while speaking, release to paste.\n\
         Caps Lock: tap to start, tap again to paste. Trigger mode locked.",
        PAD,
        row3_y - 50.0,
        WINDOW_W - PAD * 2.0,
        60.0,
    );

    // --- Section divider: Cleanup -----------------------------------------
    let section_y = row3_y - 80.0;
    add_section_label(mtm, &content, "Post-processing", PAD, section_y);

    // --- Row 4: Cleanup mode popup ----------------------------------------
    let row4_y = section_y - ROW_H - 10.0;
    add_label(mtm, &content, "Cleanup", PAD, row4_y);
    let cleanup_popup = make_popup(
        mtm,
        &["Off — paste raw transcript", "Anthropic Claude"],
        match settings.as_ref().map(|s| s.cleanup_mode) {
            Some(CleanupMode::Anthropic) => 1,
            _ => 0,
        },
        NSPoint::new(PAD + LABEL_W, row4_y - 4.0),
        NSSize::new(WINDOW_W - PAD * 2.0 - LABEL_W, ROW_H + 4.0),
    );
    unsafe {
        cleanup_popup.setTarget(Some(controller.as_ref()));
        cleanup_popup.setAction(Some(sel!(cleanupModeChanged:)));
        content.addSubview(&cleanup_popup);
    }
    *controller.ivars().cleanup_popup.borrow_mut() = Some(cleanup_popup);

    // --- Row 5: API key (secure text field) -------------------------------
    let row5_y = row4_y - ROW_H - 10.0;
    add_label(mtm, &content, "API key", PAD, row5_y);
    let key_field = make_secure_field(
        mtm,
        settings
            .as_ref()
            .map(|s| s.anthropic_api_key.as_str())
            .unwrap_or(""),
        NSPoint::new(PAD + LABEL_W, row5_y - 4.0),
        NSSize::new(WINDOW_W - PAD * 2.0 - LABEL_W, ROW_H + 4.0),
    );
    unsafe { content.addSubview(&key_field) };
    *controller.ivars().api_key_field.borrow_mut() = Some(key_field);

    // --- Row 6: Model picker ----------------------------------------------
    let row6_y = row5_y - ROW_H - 10.0;
    add_label(mtm, &content, "Model", PAD, row6_y);
    let labels: Vec<&str> = CLEANUP_MODELS.iter().map(|(label, _)| *label).collect();
    let stored_model = settings
        .as_ref()
        .map(|s| s.cleanup_model.as_str())
        .unwrap_or("claude-haiku-4-5");
    let selected_model_idx = CLEANUP_MODELS
        .iter()
        .position(|(_, id)| *id == stored_model)
        .unwrap_or(0) as isize;
    let model_popup = make_popup(
        mtm,
        &labels,
        selected_model_idx,
        NSPoint::new(PAD + LABEL_W, row6_y - 4.0),
        NSSize::new(WINDOW_W - PAD * 2.0 - LABEL_W, ROW_H + 4.0),
    );
    unsafe { content.addSubview(&model_popup) };
    *controller.ivars().cleanup_model_popup.borrow_mut() = Some(model_popup);

    // Initial enabled-state sync: API key + model are greyed out when
    // cleanup is Off, so a fresh-install user doesn't think they need
    // to fill them in.
    controller.refresh_cleanup_enabled();

    // --- Row 7: Cleanup hint ----------------------------------------------
    let row7_y = row6_y - 30.0;
    add_hint(
        mtm,
        &content,
        "Cleanup removes filler words, fixes punctuation, and honours\n\
         commands like \"new paragraph\" and \"scratch that\". Your\n\
         API key is stored in the macOS Keychain, not on disk.",
        PAD,
        row7_y - 38.0,
        WINDOW_W - PAD * 2.0,
        46.0,
    );

    // --- Row 8: Buttons (bottom-right) ------------------------------------
    let btn_w = 90.0;
    let btn_h = 28.0;
    let btn_y = PAD;
    let save_btn = make_button(
        mtm,
        "Save",
        &controller,
        sel!(save:),
        NSPoint::new(WINDOW_W - PAD - btn_w, btn_y),
        NSSize::new(btn_w, btn_h),
    );
    unsafe { save_btn.setKeyEquivalent(&NSString::from_str("\r")) }; // Return
    let cancel_btn = make_button(
        mtm,
        "Cancel",
        &controller,
        sel!(cancel:),
        NSPoint::new(WINDOW_W - PAD * 2.0 - btn_w * 2.0 + 10.0, btn_y),
        NSSize::new(btn_w, btn_h),
    );
    unsafe { cancel_btn.setKeyEquivalent(&NSString::from_str("\u{1b}")) }; // Escape
    unsafe {
        content.addSubview(&save_btn);
        content.addSubview(&cancel_btn);
    }

    // Wire the window delegate and store both halves in the singleton
    // slot. No back-pointer from the window to the controller — that was
    // the retain cycle. The window's keyDown / flagsChanged forwarders
    // read the controller out of the slot on each event.
    let delegate_proto = ProtocolObject::from_ref(&*controller);
    window.setDelegate(Some(delegate_proto));
    LIVE_SETTINGS.with(|slot| {
        *slot.borrow_mut() = Some(LiveSettings {
            controller: controller.clone(),
            window: window.clone(),
        });
    });

    // Center + show. The agent app isn't otherwise frontmost, so activate.
    window.center();
    let ns_app = NSApplication::sharedApplication(mtm);
    #[allow(deprecated)]
    unsafe {
        ns_app.activateIgnoringOtherApps(true)
    };
    window.makeKeyAndOrderFront(None);
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn add_label(mtm: MainThreadMarker, parent: &NSView, text: &str, x: f64, y: f64) {
    let label = unsafe { NSTextField::labelWithString(&NSString::from_str(text), mtm) };
    unsafe {
        label.setFrame(NSRect::new(NSPoint::new(x, y), NSSize::new(LABEL_W, ROW_H)));
        label.setTextColor(Some(&NSColor::labelColor()));
        parent.addSubview(&label);
    }
}

fn add_hint(mtm: MainThreadMarker, parent: &NSView, text: &str, x: f64, y: f64, w: f64, h: f64) {
    let label = unsafe { NSTextField::labelWithString(&NSString::from_str(text), mtm) };
    unsafe {
        label.setFrame(NSRect::new(NSPoint::new(x, y), NSSize::new(w, h)));
        label.setTextColor(Some(&NSColor::secondaryLabelColor()));
        // 11pt secondary text for the description; the system label colour
        // gives appropriate contrast in light and dark.
        parent.addSubview(&label);
    }
}

/// Bold section heading, full-width above its rows. Used to break the
/// window up into Hotkey / Post-processing groups.
fn add_section_label(mtm: MainThreadMarker, parent: &NSView, text: &str, x: f64, y: f64) {
    let label = unsafe { NSTextField::labelWithString(&NSString::from_str(text), mtm) };
    unsafe {
        label.setFrame(NSRect::new(
            NSPoint::new(x, y),
            NSSize::new(WINDOW_W - PAD * 2.0, ROW_H),
        ));
        label.setTextColor(Some(&NSColor::labelColor()));
        // System font, bold, slightly larger than the row labels.
        let font = objc2_app_kit::NSFont::boldSystemFontOfSize(13.0);
        label.setFont(Some(&font));
        parent.addSubview(&label);
    }
}

/// NSSecureTextField for the API key. Same metrics as a regular NSTextField
/// but hides the contents like a password field.
fn make_secure_field(
    mtm: MainThreadMarker,
    initial: &str,
    origin: NSPoint,
    size: NSSize,
) -> Retained<NSTextField> {
    let frame = NSRect::new(origin, size);
    let field: Retained<objc2_app_kit::NSSecureTextField> = unsafe {
        let alloc = objc2_app_kit::NSSecureTextField::alloc(mtm);
        msg_send![alloc, initWithFrame: frame]
    };
    unsafe {
        field.setStringValue(&NSString::from_str(initial));
        field.setPlaceholderString(Some(&NSString::from_str("sk-ant-…")));
    }
    // Up-cast to NSTextField — the controller's ivars hold the base type so
    // the same `stringValue` accessor works for both regular and secure.
    let as_text: Retained<NSTextField> = unsafe { Retained::cast_unchecked(field) };
    as_text
}

fn make_button(
    mtm: MainThreadMarker,
    title: &str,
    target: &SettingsController,
    action: objc2::runtime::Sel,
    origin: NSPoint,
    size: NSSize,
) -> Retained<NSButton> {
    let btn = unsafe { NSButton::new(mtm) };
    unsafe {
        btn.setTitle(&NSString::from_str(title));
        btn.setFrame(NSRect::new(origin, size));
        btn.setTarget(Some(target.as_ref()));
        btn.setAction(Some(action));
        // System rounded button (Rounded is the modern alias of the
        // older "Rounded" bezel style; the deprecation note is just
        // about naming, the visual is unchanged).
        btn.setBezelStyle(objc2_app_kit::NSBezelStyle::Push);
    }
    btn
}

fn make_popup(
    mtm: MainThreadMarker,
    items: &[&str],
    selected: isize,
    origin: NSPoint,
    size: NSSize,
) -> Retained<NSPopUpButton> {
    let frame = NSRect::new(origin, size);
    let popup: Retained<NSPopUpButton> = unsafe {
        msg_send![
            NSPopUpButton::alloc(mtm),
            initWithFrame: frame,
            pullsDown: false,
        ]
    };
    for item in items {
        unsafe { popup.addItemWithTitle(&NSString::from_str(item)) };
    }
    if selected >= 0 && (selected as usize) < items.len() {
        unsafe { popup.selectItemAtIndex(selected) };
    }
    popup
}

/// Convert an NSEvent into our internal hotkey token. Accepts:
///   - `KeyDown` → chord token (`CmdOrCtrl+Shift+Space`, `F5`, etc.)
///   - `FlagsChanged` with keycode 57 → `CapsLock`
///
/// Returns None for modifier-only `KeyDown` events (so holding Shift
/// during recording doesn't commit a half-recorded combo).
fn ns_event_to_token(event: &NSEvent) -> Option<String> {
    let event_type = unsafe { event.r#type() };
    // Caps Lock arrives as a FlagsChanged event with keyCode 57.
    if event_type == NSEventType::FlagsChanged {
        if unsafe { event.keyCode() } == 57 {
            return Some("CapsLock".to_string());
        }
        return None;
    }
    if event_type != NSEventType::KeyDown {
        return None;
    }
    let flags = unsafe { event.modifierFlags() };
    let mut parts: Vec<String> = Vec::new();
    if flags.contains(NSEventModifierFlags::Command) {
        parts.push("CmdOrCtrl".to_string());
    }
    if flags.contains(NSEventModifierFlags::Option) {
        parts.push("Alt".to_string());
    }
    if flags.contains(NSEventModifierFlags::Shift) {
        parts.push("Shift".to_string());
    }
    if flags.contains(NSEventModifierFlags::Control)
        && !flags.contains(NSEventModifierFlags::Command)
    {
        parts.push("Ctrl".to_string());
    }

    let chars: Retained<NSString> = unsafe { event.charactersIgnoringModifiers() }?;
    let key_string = chars.to_string();
    let key: String = match key_string.as_str() {
        " " => "Space".to_string(),
        "\r" => "Enter".to_string(),
        "\t" => "Tab".to_string(),
        "\u{1b}" => "Escape".to_string(),
        "\u{7f}" => "Backspace".to_string(),
        // Function keys arrive as private-use codepoints.
        "\u{f704}" => "F1".to_string(),
        "\u{f705}" => "F2".to_string(),
        "\u{f706}" => "F3".to_string(),
        "\u{f707}" => "F4".to_string(),
        "\u{f708}" => "F5".to_string(),
        "\u{f709}" => "F6".to_string(),
        "\u{f70a}" => "F7".to_string(),
        "\u{f70b}" => "F8".to_string(),
        "\u{f70c}" => "F9".to_string(),
        "\u{f70d}" => "F10".to_string(),
        "\u{f70e}" => "F11".to_string(),
        "\u{f70f}" => "F12".to_string(),
        "\u{f710}" => "F13".to_string(),
        "\u{f711}" => "F14".to_string(),
        "\u{f712}" => "F15".to_string(),
        "\u{f713}" => "F16".to_string(),
        "\u{f714}" => "F17".to_string(),
        "\u{f715}" => "F18".to_string(),
        "\u{f716}" => "F19".to_string(),
        other if other.chars().count() == 1 => {
            let c = other.chars().next().unwrap();
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase().to_string()
            } else {
                return None;
            }
        }
        _ => return None,
    };
    parts.push(key);
    Some(parts.join("+"))
}
