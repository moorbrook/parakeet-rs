//! Pre-flight TCC permission preflight.
//!
//! On every launch we (a) **request** every permission we need so first-run
//! actually triggers the macOS system prompts, then (b) **re-check** them
//! and refuse to start if any is still missing — pointing the user at the
//! exact System Settings pane that fixes it.
//!
//! Three permissions matter:
//!
//! - **Microphone** (`kTCCServiceMicrophone`) — for `cpal` to capture audio.
//! - **Accessibility** (`kTCCServiceAccessibility`) — so `enigo` can post
//!   the synthetic ⌘V paste chord into the focused app.
//! - **Input Monitoring** (`kTCCServiceListenEvent`) — so the global
//!   `CGEventTap` in `hotkey.rs` actually receives keyboard events.
//!   `CGEventTapCreate` silently returns a valid-looking mach port even
//!   when this is missing, so users see "the hotkey just does nothing"
//!   with no feedback. The preflight is what stops that footgun.

use std::sync::mpsc;
use std::time::Duration;

use anyhow::{bail, Result};
use objc2::class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2_foundation::{NSDictionary, NSNumber, NSObject, NSString};

unsafe extern "C" {
    /// `<CoreGraphics/CGEventSource.h>`. Returns true if the calling
    /// process is allowed to listen to events via `CGEventTap` — i.e. the
    /// Input Monitoring TCC permission is granted.
    fn CGPreflightListenEventAccess() -> bool;
    /// Triggers the Input Monitoring permission dialog if not already
    /// granted. Returns the current state (which is almost always still
    /// false the first time — the user has to flip it in System Settings
    /// and relaunch).
    fn CGRequestListenEventAccess() -> bool;
    /// `<ApplicationServices/HIServices/AXUIElement.h>`. Returns true if
    /// the calling process is trusted for Accessibility access. The
    /// options dict is allowed to contain `kAXTrustedCheckOptionPrompt`
    /// to surface a prompt on the first call.
    fn AXIsProcessTrustedWithOptions(options: *const NSDictionary<NSString, NSNumber>) -> bool;
    /// String constant `"AXTrustedCheckOptionPrompt"`. Static `*const`
    /// because the symbol is a global NSString published by HIServices.
    static kAXTrustedCheckOptionPrompt: *const NSString;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Microphone,
    Accessibility,
    InputMonitoring,
}

impl Permission {
    fn label(self) -> &'static str {
        match self {
            Self::Microphone => "Microphone",
            Self::Accessibility => "Accessibility",
            Self::InputMonitoring => "Input Monitoring",
        }
    }

    fn settings_url(self) -> &'static str {
        match self {
            Self::Microphone => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
            }
            Self::Accessibility => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            }
            Self::InputMonitoring => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
            }
        }
    }
}

fn mic_status() -> i32 {
    // `[AVCaptureDevice authorizationStatusForMediaType:@"soun"]`
    //   0 = NotDetermined, 1 = Restricted, 2 = Denied, 3 = Authorized
    unsafe {
        let cls = class!(AVCaptureDevice);
        let media_type = NSString::from_str("soun");
        msg_send![cls, authorizationStatusForMediaType: &*media_type]
    }
}

fn mic_granted() -> bool {
    mic_status() == 3
}

/// Trigger the system mic-permission prompt if status is `NotDetermined`.
/// Blocks briefly on a channel because `requestAccessForMediaType:` is
/// async — but the synchronous wait happens before AppKit is up so we're
/// not deadlocking a run loop.
fn request_mic() {
    if mic_status() != 0 {
        // Already determined (granted, denied, or restricted) — no prompt
        // would appear; don't waste a wait.
        return;
    }
    let (tx, rx) = mpsc::channel::<bool>();
    unsafe {
        let cls = class!(AVCaptureDevice);
        let media_type = NSString::from_str("soun");
        // The handler block fires on a background queue. We just forward
        // the result over the channel and unblock the main thread.
        let block = block2::RcBlock::new(move |granted: Bool| {
            let _ = tx.send(granted.as_bool());
        });
        let _: () = msg_send![
            cls,
            requestAccessForMediaType: &*media_type,
            completionHandler: &*block,
        ];
    }
    // Bound the wait so a broken framework can't hang startup forever.
    let _ = rx.recv_timeout(Duration::from_secs(60));
}

fn accessibility_granted_with_prompt() -> bool {
    unsafe {
        // Build `{ kAXTrustedCheckOptionPrompt: @YES }`. The prompt
        // option lifts the system Accessibility dialog (or, if denied,
        // takes the user to the right pane in System Settings) on the
        // first call per launch.
        let key = NSString::retain(kAXTrustedCheckOptionPrompt);
        let value = NSNumber::new_bool(true);
        let opts: Retained<NSDictionary<NSString, NSNumber>> =
            NSDictionary::from_slices(&[&*key], &[&*value]);
        AXIsProcessTrustedWithOptions(&*opts)
    }
}

fn accessibility_granted() -> bool {
    // Plain check, no prompt. Used for the post-request verification step.
    let null: *const NSDictionary<NSString, NSNumber> = std::ptr::null();
    unsafe { AXIsProcessTrustedWithOptions(null) }
}

/// Helper: hold an unretained NSString safely via the global symbol.
trait NSStringRetain {
    fn retain(raw: *const NSString) -> Retained<NSString>;
}
impl NSStringRetain for NSString {
    fn retain(raw: *const NSString) -> Retained<NSString> {
        // SAFETY: `kAXTrustedCheckOptionPrompt` is a non-NULL CFString
        // constant exported by HIServices and lives for the program's
        // lifetime; calling retain on it gives us a sound `Retained`.
        unsafe { Retained::retain(raw.cast_mut()) }.expect("global NSString constant was NULL")
    }
}

/// Run every preflight: prompt for anything that hasn't been decided yet,
/// then verify all three are granted. Returns Err with a clear message
/// if any is still missing after the prompts, listing each one with the
/// exact System Settings URL.
pub fn ensure_all() -> Result<()> {
    // ---- 1. Request what we can ----
    // Mic and Accessibility surface their first-time system dialogs from
    // these calls. Input Monitoring's prompt only fires from
    // `CGRequestListenEventAccess`; we surface it the first time the user
    // launches with a missing grant.
    request_mic();
    let _ = accessibility_granted_with_prompt();
    if !unsafe { CGPreflightListenEventAccess() } {
        let _ = unsafe { CGRequestListenEventAccess() };
    }

    // ---- 2. Re-check ----
    let mut missing: Vec<Permission> = Vec::new();
    if !mic_granted() {
        missing.push(Permission::Microphone);
    }
    if !accessibility_granted() {
        missing.push(Permission::Accessibility);
    }
    if !unsafe { CGPreflightListenEventAccess() } {
        missing.push(Permission::InputMonitoring);
    }

    if missing.is_empty() {
        log::info!("All required permissions granted: Microphone, Accessibility, Input Monitoring");
        return Ok(());
    }

    eprintln!();
    eprintln!("=============================================================");
    eprintln!(" Parakeet can't start — missing required permissions:");
    for p in &missing {
        eprintln!("   ✗ {}", p.label());
    }
    eprintln!();
    eprintln!(" Grant each one in System Settings → Privacy & Security:");
    for p in &missing {
        eprintln!("   {}: {}", p.label(), p.settings_url());
    }
    eprintln!();
    eprintln!(" Open the pane via Finder or run:");
    for p in &missing {
        eprintln!("   open '{}'", p.settings_url());
    }
    eprintln!();
    eprintln!(" Then relaunch Parakeet.");
    eprintln!("=============================================================");
    eprintln!();
    bail!(
        "missing {} required permission(s): {}",
        missing.len(),
        missing
            .iter()
            .map(|p| p.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

// Suppress dead-code warning on the now-unused intermediate symbol from
// the previous direct-call API. Kept around so external callers that ask
// for status without prompting still have it.
#[allow(dead_code)]
fn _unused() {
    let _ = (mic_status,);
    let _: &NSObject = unsafe { &*(std::ptr::null::<NSObject>()) };
}
