//! Pre-flight TCC permission checks. Run before the AppKit run loop starts.
//! If any required permission is missing we surface a clear error pointing
//! at the exact System Settings pane and exit. The user grants what's
//! missing, relaunches, and the app comes up working.
//!
//! Three permissions matter:
//!
//! - **Microphone** (`kTCCServiceMicrophone`) — for `cpal` to capture audio.
//! - **Accessibility** (`kTCCServiceAccessibility`) — so `enigo` can post
//!   the synthetic ⌘V paste chord into the focused app.
//! - **Input Monitoring** (`kTCCServiceListenEvent`) — so the global
//!   `CGEventTap` in `hotkey.rs` actually receives keyboard events.
//!   CGEventTapCreate silently returns a valid-looking mach port even when
//!   this is missing, so users see "the hotkey just does nothing" with no
//!   feedback. This preflight is what stops that footgun.

use anyhow::{Result, bail};
use objc2::class;
use objc2::msg_send;
use objc2_foundation::NSString;

unsafe extern "C" {
    /// `<CoreGraphics/CGEventSource.h>`. Returns true if the calling process
    /// is allowed to listen to events via `CGEventTap` — i.e. the Input
    /// Monitoring TCC permission is granted.
    fn CGPreflightListenEventAccess() -> bool;
    /// `<ApplicationServices/HIServices/AXUIElement.h>`. Returns true if the
    /// calling process is trusted for Accessibility access.
    fn AXIsProcessTrusted() -> bool;
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

    /// macOS Settings URL that lands on the right pane.
    fn settings_url(self) -> &'static str {
        match self {
            Self::Microphone => "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
            Self::Accessibility => "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
            Self::InputMonitoring => "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
        }
    }
}

fn mic_granted() -> bool {
    // Call `[AVCaptureDevice authorizationStatusForMediaType:@"soun"]`.
    // `@"soun"` is the literal `AVMediaTypeAudio` string constant.
    //   AVAuthorizationStatusNotDetermined = 0
    //   AVAuthorizationStatusRestricted    = 1
    //   AVAuthorizationStatusDenied        = 2
    //   AVAuthorizationStatusAuthorized    = 3
    unsafe {
        let cls = class!(AVCaptureDevice);
        let media_type = NSString::from_str("soun");
        let status: i32 = msg_send![cls, authorizationStatusForMediaType: &*media_type];
        status == 3
    }
}

fn accessibility_granted() -> bool {
    unsafe { AXIsProcessTrusted() }
}

fn input_monitoring_granted() -> bool {
    unsafe { CGPreflightListenEventAccess() }
}

/// Hard-check every required permission. On any miss, log a clear actionable
/// error pointing at the right System Settings pane and return Err. The
/// caller is responsible for exiting (or, in the future, opening the
/// onboarding UI to walk the user through granting).
pub fn ensure_all() -> Result<()> {
    let mut missing: Vec<Permission> = Vec::new();
    if !mic_granted() {
        missing.push(Permission::Microphone);
    }
    if !accessibility_granted() {
        missing.push(Permission::Accessibility);
    }
    if !input_monitoring_granted() {
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
