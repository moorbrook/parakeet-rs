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
    pub fn label(self) -> &'static str {
        match self {
            Self::Microphone => "Microphone",
            Self::Accessibility => "Accessibility",
            Self::InputMonitoring => "Input Monitoring",
        }
    }

    pub fn settings_url(self) -> &'static str {
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

/// Request every permission we need (showing system prompts the first
/// time per launch), then return the list of permissions that are still
/// missing after the prompts. Empty Vec means everything is granted and
/// the app can start.
///
/// This is preflight only — it doesn't present any UI of its own. The
/// caller is expected to decide what to do about the missing list
/// (typically: show an NSAlert and exit). Splitting the request from the
/// presentation lets `main.rs` route the failure through a native dialog
/// instead of stderr — important because for an `LSUIElement` app
/// launched from Finder, stderr is invisible.
pub fn check_all() -> Vec<Permission> {
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
    }
    missing
}

/// Show a native NSAlert listing the missing permissions, with one
/// "Open …" button per missing permission plus a "Quit" button. Loops
/// until the user picks Quit. Must be called on the main thread, after
/// `NSApplication::sharedApplication` exists.
///
/// For an `LSUIElement` app launched from Finder, this is the only
/// feedback the user gets that something needs to be granted — without
/// it, `eprintln! + exit` is invisible.
pub fn present_missing_alert_blocking(
    mtm: objc2_foundation::MainThreadMarker,
    missing: &[Permission],
) {
    use objc2_app_kit::{NSAlert, NSAlertStyle, NSApplication, NSApplicationActivationPolicy};
    use objc2_foundation::NSURL;

    if missing.is_empty() {
        return;
    }

    // The app is normally an Accessory (LSUIElement). The alert needs us
    // to be a Regular app momentarily, otherwise it appears behind the
    // frontmost window and the user never sees it. Snap back to Accessory
    // before exiting so nothing leaks into the Dock if the alert dance
    // ever runs as part of a larger flow.
    let ns_app = NSApplication::sharedApplication(mtm);
    unsafe {
        ns_app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
        ns_app.activate();
    }

    let mut detail = String::from(
        "Parakeet needs these permissions before it can start. Click an \
         \"Open\" button below to jump straight to the right pane in \
         System Settings, grant the permission, then relaunch Parakeet.\n\n\
         Missing:\n",
    );
    for p in missing {
        detail.push_str(&format!("  •  {}\n", p.label()));
    }

    let alert: Retained<NSAlert> = unsafe { NSAlert::new(mtm) };
    unsafe {
        alert.setMessageText(&NSString::from_str(
            "Parakeet can't start: missing permissions",
        ));
        alert.setInformativeText(&NSString::from_str(&detail));
        alert.setAlertStyle(NSAlertStyle::Warning);
        // One "Open <pane>" button per missing permission, then Quit at
        // the end so the rightmost button is the safe default.
        for p in missing {
            let label = format!("Open {}", p.label());
            let _ = alert.addButtonWithTitle(&NSString::from_str(&label));
        }
        let _ = alert.addButtonWithTitle(&NSString::from_str("Quit"));
    }

    loop {
        // NSAlert returns NSModalResponse values; the first added button
        // is `NSAlertFirstButtonReturn` (1000), the next 1001, etc.
        let response = unsafe { alert.runModal() };
        // `NSAlert::runModal` returns the new-style `NSModalResponse`
        // which is a plain isize. The first added button is 1000
        // (`NSAlertFirstButtonReturn`), the next 1001, etc.
        let idx = (response - 1000) as usize;
        if idx < missing.len() {
            // Open the requested System Settings pane via NSWorkspace,
            // then loop so the alert stays up while the user grants the
            // permission (they have to relaunch anyway, but at least
            // they don't lose the list).
            let url_str = NSString::from_str(missing[idx].settings_url());
            unsafe {
                if let Some(url) = NSURL::URLWithString(&url_str) {
                    let workspace = objc2_app_kit::NSWorkspace::sharedWorkspace();
                    let _: Bool = msg_send![&*workspace, openURL: &*url];
                }
            }
            // Loop and re-show — the user might want to open the next one.
            continue;
        }
        // Anything else (Quit button, or Esc / Cmd-. → NSModalResponseStop):
        // drop back to Accessory and let the caller exit the process.
        unsafe { ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory) };
        return;
    }
}

// Suppress dead-code warning on the now-unused intermediate symbol from
// the previous direct-call API. Kept around so external callers that ask
// for status without prompting still have it.
#[allow(dead_code)]
fn _unused() {
    let _ = (mic_status,);
    let _: &NSObject = unsafe { &*(std::ptr::null::<NSObject>()) };
}
