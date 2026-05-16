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
    NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState, NSVisualEffectView,
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

        // Content chrome: `NSVisualEffectView` with the HUD material.
        // Auto-adapts to light / dark mode, respects Increase Contrast
        // and Reduce Transparency in System Settings → Accessibility.
        // Replaces the previous hand-rolled CALayer setBackgroundColor
        // approach (hardcoded RGB, ignored every accessibility pref).
        let effect_view: Retained<NSVisualEffectView> = unsafe {
            let alloc = NSVisualEffectView::alloc(mtm);
            NSVisualEffectView::initWithFrame(
                alloc,
                NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(HUD_W, HUD_H)),
            )
        };
        unsafe {
            effect_view.setMaterial(NSVisualEffectMaterial::HUDWindow);
            effect_view.setBlendingMode(NSVisualEffectBlendingMode::BehindWindow);
            effect_view.setState(NSVisualEffectState::Active);
            // Rounded corners on the effect view's own layer. AppKit
            // gives `NSVisualEffectView` a backing layer for free
            // (wantsLayer = true is implicit for this class).
            if let Some(layer) = effect_view.layer() {
                let _: () = msg_send![&*layer, setCornerRadius: CORNER_RADIUS];
                let _: () = msg_send![&*layer, setMasksToBounds: true];
            }
            // Use the effect view as the panel's content. The label
            // becomes its only subview.
            panel.setContentView(Some(&effect_view));
        }

        // Centred label inside the effect view.
        let label_frame = NSRect::new(NSPoint::new(12.0, 0.0), NSSize::new(HUD_W - 24.0, HUD_H));
        let label =
            unsafe { NSTextField::labelWithString(&NSString::from_str("Listening…"), mtm) };
        unsafe {
            label.setFrame(label_frame);
            label.setAlignment(NSTextAlignment::Center);
            // White text reads on the HUDWindow material in both light
            // and dark modes; HUD material is intentionally dark in
            // both appearances per Apple's HIG.
            label.setTextColor(Some(&NSColor::whiteColor()));
            let font = NSFont::systemFontOfSize(14.0);
            label.setFont(Some(&font));
            label.setDrawsBackground(false);
            label.setBordered(false);
            effect_view.addSubview(&label);
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

/// Reposition the HUD on the current main screen. Called from the
/// `AppDelegate::applicationDidChangeScreenParameters:` hook when the
/// display configuration changes (monitor un/replug, resolution swap,
/// Spaces reshuffle). Safe to call when the HUD isn't visible —
/// `setFrame` just moves the panel for the next show.
pub fn reposition_on_screen(mtm: MainThreadMarker) {
    HUD.with(|slot| {
        if let Some(hud) = slot.borrow().as_ref() {
            position_on_screen(mtm, &hud.panel);
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
