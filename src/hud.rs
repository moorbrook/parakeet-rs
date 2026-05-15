//! Floating recording-state HUD.
//!
//! A small pill at the bottom of the main display that shows the current
//! dictation state ("Listening…", "Transcribing…", "Polishing…") so the
//! user knows the hotkey actually registered. Hides itself on `Idle` /
//! `ModelLoading` so it's only visible when something is happening.
//!
//! Threading mirrors `menubar.rs`: a `thread_local!` slot on the main
//! thread owns the AppKit objects; `show_state()` is safe to call from
//! any thread and bounces onto the main GCD queue.
//!
//! Implementation notes:
//!
//! - The panel is a borderless, non-activating, `ignoresMouseEvents: true`
//!   `NSPanel`. Activating it would steal focus from the user's app and
//!   ruin the paste step that follows.
//! - The window level is `kCGFloatingWindowLevel` (3) so it sits above
//!   normal app windows but below the system menu bar / Spotlight.
//! - `collectionBehavior` includes `canJoinAllSpaces` + `stationary` so
//!   the HUD doesn't follow space-switching animations.

use std::cell::RefCell;

use dispatch2::DispatchQueue;
use objc2::rc::Retained;
use objc2::{define_class, msg_send, MainThreadOnly};
use objc2_app_kit::{
    NSBackingStoreType, NSColor, NSFont, NSPanel, NSScreen, NSTextAlignment, NSTextField,
    NSWindowCollectionBehavior, NSWindowStyleMask,
};
use objc2_foundation::{MainThreadMarker, NSObjectProtocol, NSPoint, NSRect, NSSize, NSString};

use crate::app::DictationState;

const HUD_W: f64 = 220.0;
const HUD_H: f64 = 44.0;
/// Gap between the bottom of the screen and the bottom of the HUD. The
/// Dock is auto-hidden for many users; 80px clears it comfortably for
/// "always visible" Dock setups too.
const BOTTOM_OFFSET: f64 = 80.0;
const CORNER_RADIUS: f64 = 12.0;

// NSPanel subclass with the `nonactivatingPanel` style mask set in its
// init path. The base NSWindow only exposes the bit at construction time;
// having a tiny subclass keeps the open code clean.
define_class!(
    #[unsafe(super = NSPanel)]
    #[thread_kind = MainThreadOnly]
    #[ivars = ()]
    struct HudPanel;

    unsafe impl NSObjectProtocol for HudPanel {}
);

struct Hud {
    panel: Retained<HudPanel>,
    label: Retained<NSTextField>,
    visible: bool,
}

thread_local! {
    static HUD: RefCell<Option<Hud>> = const { RefCell::new(None) };
}

/// Install the HUD. Must run on the main thread. Idempotent — subsequent
/// calls are no-ops so the menubar / startup path can call this safely.
pub fn install(mtm: MainThreadMarker) {
    HUD.with(|slot| {
        if slot.borrow().is_some() {
            return;
        }

        // Frame placeholder; `position_on_screen` rewrites it before we
        // ever show the panel. `frame.zero` is invalid for some AppKit
        // assertions, so use a sensible non-zero default.
        let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(HUD_W, HUD_H));
        let style = NSWindowStyleMask::Borderless | NSWindowStyleMask::NonactivatingPanel;
        let panel: Retained<HudPanel> = unsafe {
            let alloc = HudPanel::alloc(mtm).set_ivars(());
            msg_send![
                super(alloc),
                initWithContentRect: frame,
                styleMask: style,
                backing: NSBackingStoreType::Buffered,
                defer: false,
            ]
        };
        unsafe {
            panel.setOpaque(false);
            panel.setHasShadow(true);
            panel.setBackgroundColor(Some(&NSColor::clearColor()));
            panel.setIgnoresMouseEvents(true);
            // kCGFloatingWindowLevel = 3. Above normal app windows,
            // below Spotlight / status / dock.
            panel.setLevel(objc2_app_kit::NSFloatingWindowLevel);
            panel.setHidesOnDeactivate(false);
            panel.setCollectionBehavior(
                NSWindowCollectionBehavior::CanJoinAllSpaces
                    | NSWindowCollectionBehavior::Stationary
                    | NSWindowCollectionBehavior::IgnoresCycle,
            );
            panel.setReleasedWhenClosed(false);
        }

        // Content view: rounded dark background via CALayer. We rely on
        // the system's wantsLayer auto-creation rather than swapping in a
        // custom NSView subclass — saves a class definition for a UI
        // element that doesn't need any custom drawing.
        let content = panel
            .contentView()
            .expect("borderless panel must still have a content view");
        unsafe {
            content.setWantsLayer(true);
            if let Some(layer) = content.layer() {
                // CALayer's `setCornerRadius` / `setBackgroundColor` accessors
                // aren't exposed on the objc2-quartz-core types we get
                // through `content.layer()` without pulling in additional
                // feature flags. Just dispatch through `msg_send!` — these
                // are stable AppKit selectors with no risk of drift.
                let _: () = msg_send![&*layer, setCornerRadius: CORNER_RADIUS];
                let _: () = msg_send![&*layer, setMasksToBounds: true];
                // Translucent black with system-dark feel. Matches the
                // visual weight of HUD overlays elsewhere on macOS.
                let bg = NSColor::colorWithRed_green_blue_alpha(0.10, 0.10, 0.12, 0.92);
                let cg = bg.CGColor();
                let _: () = msg_send![&*layer, setBackgroundColor: &*cg];
            }
        }

        // Centred label inside the panel content.
        let label_frame = NSRect::new(NSPoint::new(12.0, 0.0), NSSize::new(HUD_W - 24.0, HUD_H));
        let label =
            unsafe { NSTextField::labelWithString(&NSString::from_str("Listening…"), mtm) };
        unsafe {
            label.setFrame(label_frame);
            label.setAlignment(NSTextAlignment::Center);
            label.setTextColor(Some(&NSColor::whiteColor()));
            let font = NSFont::systemFontOfSize(14.0);
            label.setFont(Some(&font));
            label.setDrawsBackground(false);
            label.setBordered(false);
            content.addSubview(&label);
        }

        position_on_screen(mtm, &panel);

        *slot.borrow_mut() = Some(Hud {
            panel,
            label,
            visible: false,
        });
    });
}

/// Show/hide + re-label the HUD based on the current dictation state.
/// Safe to call from any thread.
pub fn show_state(state: DictationState) {
    dispatch_to_main(move || show_state_main(state));
}

fn show_state_main(state: DictationState) {
    HUD.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(hud) = slot.as_mut() else {
            return;
        };

        let label = match state {
            DictationState::Listening => Some("●  Listening…"),
            DictationState::Transcribing => Some("Transcribing…"),
            DictationState::Polishing => Some("✨  Polishing…"),
            DictationState::Idle | DictationState::ModelLoading => None,
        };

        match label {
            Some(text) => {
                unsafe { hud.label.setStringValue(&NSString::from_str(text)) };
                if !hud.visible {
                    // Re-position before showing in case the user moved
                    // their main display since install (e.g. unplugged
                    // an external monitor).
                    if let Some(mtm) = MainThreadMarker::new() {
                        position_on_screen(mtm, &hud.panel);
                    }
                    unsafe { hud.panel.orderFrontRegardless() };
                    hud.visible = true;
                }
            }
            None => {
                if hud.visible {
                    unsafe { hud.panel.orderOut(None) };
                    hud.visible = false;
                }
            }
        }
    });
}

/// Center the panel horizontally on the main screen, with a fixed gap
/// from the bottom. NSScreen returns the visible frame in screen-space
/// coordinates (bottom-left origin), so we just compute the rect.
fn position_on_screen(mtm: MainThreadMarker, panel: &HudPanel) {
    let screen = match NSScreen::mainScreen(mtm) {
        Some(s) => s,
        None => return,
    };
    let visible = screen.visibleFrame();
    let x = visible.origin.x + (visible.size.width - HUD_W) / 2.0;
    let y = visible.origin.y + BOTTOM_OFFSET;
    let frame = NSRect::new(NSPoint::new(x, y), NSSize::new(HUD_W, HUD_H));
    unsafe { panel.setFrame_display(frame, false) };
}

fn dispatch_to_main<F: FnOnce() + Send + 'static>(f: F) {
    if MainThreadMarker::new().is_some() {
        f();
    } else {
        DispatchQueue::main().exec_async(f);
    }
}
