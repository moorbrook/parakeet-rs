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

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, sel, DefinedClass, MainThreadOnly};
use objc2_app_kit::{
    NSApplication, NSBackingStoreType, NSButton, NSColor, NSEvent, NSEventModifierFlags,
    NSEventType, NSPopUpButton, NSResponder, NSTextField, NSView, NSWindow, NSWindowDelegate,
    NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize,
    NSString,
};

use crate::app::{glyphs_for_shortcut, AppHandle};
use crate::settings::TriggerMode;

const WINDOW_W: f64 = 460.0;
const WINDOW_H: f64 = 260.0;
const ROW_H: f64 = 26.0;
const LABEL_W: f64 = 130.0;
const PAD: f64 = 20.0;

#[derive(Default)]
struct Ivars {
    /// Built-up token like `"CmdOrCtrl+Shift+Space"` while the user is
    /// recording a new combo. `None` outside of recording state.
    recording_token: RefCell<Option<String>>,
    shortcut_button: RefCell<Option<Retained<NSButton>>>,
    mode_popup: RefCell<Option<Retained<NSPopUpButton>>>,
    window: RefCell<Option<Retained<NSWindow>>>,
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
            // Release our retained window so the app doesn't keep it alive
            // after the close button is clicked.
            *self.ivars().window.borrow_mut() = None;
        }
    }

    impl SettingsController {
        /// "Save" button action. Read controls, call App::apply_settings.
        #[unsafe(method(save:))]
        fn save(&self, _sender: *mut NSObject) {
            let Some(app) = AppHandle::get() else {
                return;
            };
            let mut new = app.settings.load();

            if let Some(token) = self.ivars().recording_token.borrow().clone() {
                new.hotkey = token;
            }
            if let Some(popup) = self.ivars().mode_popup.borrow().as_ref() {
                new.trigger_mode = match unsafe { popup.indexOfSelectedItem() } {
                    1 => TriggerMode::Hold,
                    _ => TriggerMode::Tap,
                };
            }

            if let Err(e) = app.apply_settings(new) {
                log::error!("save settings failed: {e:#}");
            }
            self.close_window();
        }

        #[unsafe(method(cancel:))]
        fn cancel(&self, _sender: *mut NSObject) {
            self.close_window();
        }

        /// "Record Shortcut" button. Flips the UI into recording state; the
        /// next NSEvent-driven key combo seen by `key_down` becomes the new
        /// hotkey token.
        #[unsafe(method(beginRecording:))]
        fn begin_recording(&self, _sender: *mut NSObject) {
            *self.ivars().recording_token.borrow_mut() = Some(String::new());
            if let Some(btn) = self.ivars().shortcut_button.borrow().as_ref() {
                unsafe { btn.setTitle(&NSString::from_str("Press a key combination…")) };
            }
        }

        /// Captured by the custom NSWindow subclass below — see
        /// `RecordingWindow::keyDown`. Converts the NSEvent into a token
        /// like `"CmdOrCtrl+Shift+Space"` and updates the button label.
        #[unsafe(method(captureKey:))]
        fn capture_key(&self, event_obj: *mut NSObject) {
            if self.ivars().recording_token.borrow().is_none() {
                return;
            }
            // Re-interpret the opaque sender as an NSEvent pointer. The
            // RecordingWindow forwards events through this selector.
            let event: &NSEvent = unsafe { &*(event_obj as *const NSEvent) };
            if let Some(token) = ns_event_to_token(event) {
                let glyphs = glyphs_for_shortcut(&token);
                *self.ivars().recording_token.borrow_mut() = Some(token);
                if let Some(btn) = self.ivars().shortcut_button.borrow().as_ref() {
                    unsafe { btn.setTitle(&NSString::from_str(&glyphs)) };
                }
            }
        }
    }
);

impl SettingsController {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(Ivars::default());
        unsafe { msg_send![super(this), init] }
    }

    fn close_window(&self) {
        // Take ownership of the Retained out of the RefCell *before* calling
        // `close()`. NSWindow::close fires `windowWillClose` synchronously,
        // which calls back into our delegate and tries to clear the same
        // RefCell. Holding an immutable borrow across that call panics with
        // `BorrowMutError` — exactly the crash we saw on Save.
        let w = self.ivars().window.borrow_mut().take();
        if let Some(w) = w {
            unsafe { w.close() };
        }
    }
}

// A custom NSWindow that forwards keyDown events to the SettingsController
// during shortcut recording. AppKit normally swallows un-targeted keys, so
// we hook them at the window level.
define_class!(
    #[unsafe(super = NSWindow)]
    #[thread_kind = MainThreadOnly]
    #[ivars = RefCell<Option<Retained<SettingsController>>>]
    struct RecordingWindow;

    unsafe impl NSObjectProtocol for RecordingWindow {}

    impl RecordingWindow {
        #[unsafe(method(keyDown:))]
        fn key_down(&self, event: &NSEvent) {
            let ctrl = self.ivars().borrow().clone();
            if let Some(c) = ctrl {
                let obj: *const NSEvent = event;
                let _: () = unsafe {
                    msg_send![&*c, captureKey: obj as *mut NSObject]
                };
            }
        }
    }
);

/// Open the Settings window (or focus it if already open).
pub fn open(mtm: MainThreadMarker) {
    let controller = SettingsController::new(mtm);

    // Window frame: NSWindow expects bottom-left origin. We center after.
    let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(WINDOW_W, WINDOW_H));
    let mask = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable;
    let window: Retained<RecordingWindow> = unsafe {
        let alloc = RecordingWindow::alloc(mtm).set_ivars(RefCell::new(Some(controller.clone())));
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
    let content = window
        .contentView()
        .expect("window must have content view");
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

    // --- Row 3: Hint text --------------------------------------------------
    let row3_y = row2_y - ROW_H - 10.0;
    add_hint(
        mtm,
        &content,
        "Tap: press once to start, Parakeet stops when you finish speaking.\nHold: press and hold while speaking, release to paste.",
        PAD,
        row3_y - 30.0,
        WINDOW_W - PAD * 2.0,
        40.0,
    );

    // --- Row 4: Buttons (bottom-right) ------------------------------------
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

    // Wire the window delegate and remember it.
    let delegate_proto = ProtocolObject::from_ref(&*controller);
    window.setDelegate(Some(delegate_proto));
    *controller.ivars().window.borrow_mut() =
        Some(unsafe { Retained::cast_unchecked::<NSWindow>(window.clone()) });

    // Center + show. The agent app isn't otherwise frontmost, so activate.
    window.center();
    let ns_app = NSApplication::sharedApplication(mtm);
    #[allow(deprecated)]
    unsafe { ns_app.activateIgnoringOtherApps(true) };
    window.makeKeyAndOrderFront(None);
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn add_label(mtm: MainThreadMarker, parent: &NSView, text: &str, x: f64, y: f64) {
    let label = unsafe {
        NSTextField::labelWithString(&NSString::from_str(text), mtm)
    };
    unsafe {
        label.setFrame(NSRect::new(NSPoint::new(x, y), NSSize::new(LABEL_W, ROW_H)));
        label.setTextColor(Some(&NSColor::labelColor()));
        parent.addSubview(&label);
    }
}

fn add_hint(
    mtm: MainThreadMarker,
    parent: &NSView,
    text: &str,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
) {
    let label = unsafe {
        NSTextField::labelWithString(&NSString::from_str(text), mtm)
    };
    unsafe {
        label.setFrame(NSRect::new(NSPoint::new(x, y), NSSize::new(w, h)));
        label.setTextColor(Some(&NSColor::secondaryLabelColor()));
        // 11pt secondary text for the description; the system label colour
        // gives appropriate contrast in light and dark.
        parent.addSubview(&label);
    }
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

/// Convert an NSEvent.keyDown into our internal `"CmdOrCtrl+Shift+Space"`
/// token. Returns None for modifier-only events (so holding shift doesn't
/// commit a half-recorded combo).
fn ns_event_to_token(event: &NSEvent) -> Option<String> {
    if unsafe { event.r#type() } != NSEventType::KeyDown {
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
        "\u{f704}" => "F1".to_string(), "\u{f705}" => "F2".to_string(),
        "\u{f706}" => "F3".to_string(), "\u{f707}" => "F4".to_string(),
        "\u{f708}" => "F5".to_string(), "\u{f709}" => "F6".to_string(),
        "\u{f70a}" => "F7".to_string(), "\u{f70b}" => "F8".to_string(),
        "\u{f70c}" => "F9".to_string(), "\u{f70d}" => "F10".to_string(),
        "\u{f70e}" => "F11".to_string(), "\u{f70f}" => "F12".to_string(),
        "\u{f710}" => "F13".to_string(), "\u{f711}" => "F14".to_string(),
        "\u{f712}" => "F15".to_string(), "\u{f713}" => "F16".to_string(),
        "\u{f714}" => "F17".to_string(), "\u{f715}" => "F18".to_string(),
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
