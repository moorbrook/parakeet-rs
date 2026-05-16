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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use dispatch2::{DispatchQueue, DispatchTime};
use objc2::rc::Retained;
use objc2::{define_class, msg_send, MainThreadOnly};
use objc2_app_kit::{
    NSBackingStoreType, NSColor, NSFont, NSPanel, NSScreen, NSTextAlignment, NSTextField,
    NSView, NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState,
    NSVisualEffectView, NSWindowCollectionBehavior, NSWindowStyleMask,
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

// --- Waveform bars (Listening-state only) ---
//
// Live audio peak amplitude is published by the cpal callback in
// `audio.rs` via [`set_audio_level`] (lock-free `AtomicU32` of f32
// bits). While the HUD is in `Listening`, an animation tick scheduled
// on the main GCD queue reads the level at ~30 fps, computes a
// per-bar target height, and lerps each bar's frame toward it. The
// generation counter [`ANIMATION_GEN`] lets a stale tick from a prior
// Listening session bail out instead of fighting a freshly-installed
// one.

const BARS_COUNT: usize = 7;
const BAR_WIDTH: f64 = 6.0;
const BAR_GAP: f64 = 5.0;
const BAR_HEIGHT_MIN: f64 = 8.0;
const BAR_HEIGHT_MAX: f64 = 36.0;
/// Left edge of the leftmost bar inside the HUD's content view.
/// `HUD_W = 220`; label area is x ∈ [14, 126], bars area is
/// x ∈ [130, 202] (7*6 + 6*5 = 72 px wide). 18 px right margin.
const BARS_ORIGIN_X: f64 = 130.0;
/// Empirical gain applied to the raw mic peak before the compressive
/// curve. A typical conversational voice peaks at ~0.2-0.4 on a
/// MacBook built-in mic; whispering / late-night peaks at ~0.05-0.10.
/// `(level * BAR_LEVEL_GAIN).min(1.0)` clamps loud bursts to full
/// height instead of clipping math elsewhere; the `.sqrt()` in
/// `bar_tick` then lifts the quiet end so soft speech is still
/// clearly visible.
const BAR_LEVEL_GAIN: f32 = 4.0;
/// Lerp coefficient per tick — higher = snappier reactions, lower =
/// smoother. 0.5 at 30 fps tracks speech onsets cleanly without
/// looking jittery on background noise.
const BAR_LERP: f32 = 0.5;
/// Frame period for the bar animation.
const BAR_TICK: Duration = Duration::from_millis(33);

/// Current audio peak amplitude in [0.0, 1.0], updated from cpal's
/// realtime callback. f32 bits stored in an `AtomicU32` so the
/// write side is a single atomic store with no allocation.
static AUDIO_LEVEL: AtomicU32 = AtomicU32::new(0);

/// Monotonically-increasing animation epoch. Bumped by `show_state`
/// each time we enter Listening so a stale tick from a previous
/// Listening session can recognise itself and bail.
static ANIMATION_GEN: AtomicU64 = AtomicU64::new(0);

/// Set the current audio peak amplitude. Lock-free; safe to call from
/// any thread (in practice, from the cpal realtime audio callback).
pub fn set_audio_level(level: f32) {
    AUDIO_LEVEL.store(level.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

fn current_audio_level() -> f32 {
    f32::from_bits(AUDIO_LEVEL.load(Ordering::Relaxed))
}

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
    /// 7 thin vertical bars rendered to the right of the label. Each
    /// bar is a layer-backed `NSView` with a white background; its
    /// frame's height is animated by the bar tick to follow the live
    /// audio peak.
    bars: Vec<Retained<NSView>>,
    /// Smoothed per-bar heights, lerped each tick toward the
    /// audio-driven target. Kept in `f32` (not the frame's `f64`) to
    /// match the audio domain and avoid drift from repeated rounding.
    bar_heights: [f32; BARS_COUNT],
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

        // Label takes the left portion of the HUD; bars go in the right
        // portion. The label is left-aligned (was Center) so the
        // "● Listening…" text doesn't visually drift away from the bars.
        // BARS_ORIGIN_X is 130; label area is x ∈ [14, 126].
        let label_frame =
            NSRect::new(NSPoint::new(14.0, 0.0), NSSize::new(BARS_ORIGIN_X - 18.0, HUD_H));
        let label =
            unsafe { NSTextField::labelWithString(&NSString::from_str("Listening…"), mtm) };
        unsafe {
            label.setFrame(label_frame);
            label.setAlignment(NSTextAlignment::Left);
            // White text reads on the HUDWindow material in both light
            // and dark modes; HUD material is intentionally dark in
            // both appearances per Apple's HIG.
            label.setTextColor(Some(&NSColor::whiteColor()));
            let font = NSFont::systemFontOfSize(13.0);
            label.setFont(Some(&font));
            label.setDrawsBackground(false);
            label.setBordered(false);
            effect_view.addSubview(&label);
        }

        // 7 pill-shaped white bars to the right of the label. Each
        // starts at the minimum height (idle look); the bar tick
        // animates them while Listening.
        let mut bars: Vec<Retained<NSView>> = Vec::with_capacity(BARS_COUNT);
        for i in 0..BARS_COUNT {
            let x = BARS_ORIGIN_X + (i as f64) * (BAR_WIDTH + BAR_GAP);
            let y = (HUD_H - BAR_HEIGHT_MIN) / 2.0;
            let frame = NSRect::new(
                NSPoint::new(x, y),
                NSSize::new(BAR_WIDTH, BAR_HEIGHT_MIN),
            );
            let view: Retained<NSView> = unsafe {
                let alloc = NSView::alloc(mtm);
                NSView::initWithFrame(alloc, frame)
            };
            unsafe {
                view.setWantsLayer(true);
                if let Some(layer) = view.layer() {
                    // Pure white pills on the HUD material. 85%
                    // opacity reads as "active indicator" without
                    // overpowering the rest of the chrome.
                    //
                    // The `as *mut c_void` cast is to bypass an
                    // objc2-version mismatch between
                    // objc2-app-kit's `Retained<CGColor>` (newer
                    // objc2) and core-graphics 0.24's `CGColor`
                    // (older objc2) — both encode as the same C ABI
                    // pointer, but the Rust types don't unify. The
                    // runtime only sees the pointer.
                    let white_cg = NSColor::whiteColor().colorWithAlphaComponent(0.85).CGColor();
                    let cg_ptr = Retained::as_ptr(&white_cg) as *mut std::ffi::c_void;
                    let _: () = msg_send![&*layer, setBackgroundColor: cg_ptr];
                    let _: () = msg_send![&*layer, setCornerRadius: BAR_WIDTH / 2.0];
                }
                effect_view.addSubview(&view);
            }
            bars.push(view);
        }

        position_on_screen(mtm, &panel);

        *slot.borrow_mut() = Some(Hud {
            panel,
            label,
            bars,
            bar_heights: [BAR_HEIGHT_MIN as f32; BARS_COUNT],
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
    let mut start_animation = false;
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

        // Bars only animate during Listening. Show them while the
        // panel is up (Transcribing / Polishing keep them frozen at
        // the minimum height — recording is over, no more audio to
        // visualise). Hide them entirely when the panel hides.
        let want_animation = matches!(state, DictationState::Listening);
        let want_bars_visible = label.is_some();
        for bar in &hud.bars {
            unsafe { bar.setHidden(!want_bars_visible) };
        }

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

        if want_animation {
            // Bump the generation BEFORE scheduling the first tick so
            // any still-pending tick from a prior session sees the
            // mismatch and bails. The new tick we schedule below
            // captures the post-bump value as "its" generation.
            ANIMATION_GEN.fetch_add(1, Ordering::Relaxed);
            start_animation = true;
        } else {
            // Leaving Listening — bump generation to retire pending
            // ticks. The bars are left at whatever frame they were on
            // until they get hidden / re-animated next time.
            ANIMATION_GEN.fetch_add(1, Ordering::Relaxed);
            // Reset bar heights to minimum so the next Listening
            // state starts from a clean idle look.
            reset_bar_heights(slot.as_mut());
        }
    });

    if start_animation {
        let my_gen = ANIMATION_GEN.load(Ordering::Relaxed);
        schedule_bar_tick(my_gen);
    }
}

/// Animation tick: read the latest audio peak, compute per-bar
/// targets, lerp the current heights toward them, push the new
/// heights into each bar's frame. Re-schedules itself for
/// `BAR_TICK` later if the generation it was scheduled under is
/// still current; otherwise exits silently so a prior Listening
/// session's tick can't fight a fresh one.
fn schedule_bar_tick(my_gen: u64) {
    let when: DispatchTime = match BAR_TICK.try_into() {
        Ok(t) => t,
        Err(()) => return,
    };
    let _ = DispatchQueue::main().after(when, move || bar_tick(my_gen));
}

fn bar_tick(my_gen: u64) {
    if ANIMATION_GEN.load(Ordering::Relaxed) != my_gen {
        return;
    }
    // Two-stage response curve: linear gain to bring conversational
    // voice into the [0, 1] range, then `sqrt` to compress so even
    // quiet/whispered input (peak ~0.05-0.1) lifts the bars
    // noticeably. With gain=4.0 and sqrt:
    //   peak=0.05 → gained=0.20 → curved=0.45 (44% of max-min span)
    //   peak=0.15 → gained=0.60 → curved=0.77 (77%)
    //   peak=0.30 → gained=1.20 → clamp=1.00 → curved=1.00 (100%)
    let level = ((current_audio_level() * BAR_LEVEL_GAIN).min(1.0)).sqrt();
    HUD.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(hud) = slot.as_mut() else {
            return;
        };
        for (i, bar) in hud.bars.iter().enumerate() {
            // Subtle bell-curve profile so middle bars run a hair
            // taller — looks more "voice-shaped" than a flat row,
            // but only mildly so (outer bars at 85% of centre)
            // because the user complained that strong attenuation
            // made the outer bars invisible.
            let centre = (BARS_COUNT - 1) as f32 / 2.0;
            let dist = (i as f32 - centre).abs() / centre;
            let profile = 1.0 - 0.15 * dist;
            let target =
                (BAR_HEIGHT_MIN as f32) + (BAR_HEIGHT_MAX - BAR_HEIGHT_MIN) as f32 * level * profile;
            // Lerp toward target; clamp to [MIN, MAX].
            let smoothed = hud.bar_heights[i] + (target - hud.bar_heights[i]) * BAR_LERP;
            let h = smoothed.clamp(BAR_HEIGHT_MIN as f32, BAR_HEIGHT_MAX as f32);
            hud.bar_heights[i] = h;
            let h_f64 = f64::from(h);
            let x = BARS_ORIGIN_X + (i as f64) * (BAR_WIDTH + BAR_GAP);
            let y = (HUD_H - h_f64) / 2.0;
            unsafe {
                bar.setFrame(NSRect::new(
                    NSPoint::new(x, y),
                    NSSize::new(BAR_WIDTH, h_f64),
                ));
            }
        }
    });
    schedule_bar_tick(my_gen);
}

fn reset_bar_heights(hud: Option<&mut Hud>) {
    let Some(hud) = hud else { return };
    for (i, bar) in hud.bars.iter().enumerate() {
        hud.bar_heights[i] = BAR_HEIGHT_MIN as f32;
        let x = BARS_ORIGIN_X + (i as f64) * (BAR_WIDTH + BAR_GAP);
        let y = (HUD_H - BAR_HEIGHT_MIN) / 2.0;
        unsafe {
            bar.setFrame(NSRect::new(
                NSPoint::new(x, y),
                NSSize::new(BAR_WIDTH, BAR_HEIGHT_MIN),
            ));
        }
    }
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
