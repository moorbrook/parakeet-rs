//! Native macOS TCC permission state, onboarding, and recovery.
//!
//! This follows ZoomItForMac's useful permission architecture: querying state
//! is separate from requesting it, the app stays alive while permissions are
//! missing, a menu command always exposes current status, and returning from
//! System Settings refreshes the dialog. Unlike the old startup preflight, no
//! permission is requested merely because the process launched.
//!
//! Parakeet needs three TCC services:
//!
//! - **Input Monitoring** for the global `CGEventTap`. It is the only grant
//!   needed at launch, because without it the configured global hotkey cannot
//!   work. The menu remains usable without it.
//! - **Microphone** when the user first asks to dictate.
//! - **Accessibility** when dictation needs to deliver text with
//!   `CGEventPost`. We ask before recording so a completed transcript is not
//!   surprised by a delivery prompt.
//!
//! AVFoundation exposes microphone's full four-state authorization model.
//! CoreGraphics and Accessibility expose only granted/not-granted preflights,
//! so their UI truthfully says "Not granted" rather than inventing a denied vs
//! not-determined distinction macOS does not publish.

use std::cell::RefCell;

use block2::RcBlock;
use objc2::class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2::MainThreadOnly;
use objc2_app_kit::{NSAlert, NSAlertStyle, NSApplication, NSView, NSWorkspace};
use objc2_foundation::{
    MainThreadMarker, NSDictionary, NSNumber, NSPoint, NSRect, NSSize, NSString, NSURL,
};

unsafe extern "C" {
    /// `<CoreGraphics/CGEventSource.h>`.
    fn CGPreflightListenEventAccess() -> bool;
    /// Requests/registers Input Monitoring. The user's decision is external to
    /// this call and an event tap created before the grant needs a relaunch.
    fn CGRequestListenEventAccess() -> bool;
    /// `<ApplicationServices/HIServices/AXUIElement.h>`.
    fn AXIsProcessTrustedWithOptions(options: *const NSDictionary<NSString, NSNumber>) -> bool;
    static kAXTrustedCheckOptionPrompt: *const NSString;
}

const GENERIC_PRIVACY_SETTINGS_URL: &str =
    "x-apple.systempreferences:com.apple.preference.security";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    InputMonitoring,
    Microphone,
    Accessibility,
}

impl Permission {
    pub const ALL: [Self; 3] = [Self::InputMonitoring, Self::Microphone, Self::Accessibility];

    pub fn label(self) -> &'static str {
        match self {
            Self::InputMonitoring => "Input Monitoring",
            Self::Microphone => "Microphone",
            Self::Accessibility => "Accessibility",
        }
    }

    pub fn purpose(self) -> &'static str {
        match self {
            Self::InputMonitoring => "detect the global dictation hotkey",
            Self::Microphone => "record speech while a dictation is active",
            Self::Accessibility => "type the finished transcript into the focused app",
        }
    }

    pub fn settings_url(self) -> &'static str {
        match self {
            Self::InputMonitoring => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
            }
            Self::Microphone => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
            }
            Self::Accessibility => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionStatus {
    Granted,
    NotDetermined,
    Denied,
    Restricted,
    /// The underlying API exposes only a Boolean preflight.
    NotGranted,
}

impl PermissionStatus {
    pub fn is_granted(self) -> bool {
        self == Self::Granted
    }

    fn description(self) -> &'static str {
        match self {
            Self::Granted => "Granted",
            Self::NotDetermined => "Not requested",
            Self::Denied => "Denied — open System Settings to recover",
            Self::Restricted => "Restricted — managed by macOS or device policy",
            Self::NotGranted => "Not granted — request it or open System Settings",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PermissionState {
    pub input_monitoring: PermissionStatus,
    pub microphone: PermissionStatus,
    pub accessibility: PermissionStatus,
}

impl PermissionState {
    pub fn status(self, permission: Permission) -> PermissionStatus {
        match permission {
            Permission::InputMonitoring => self.input_monitoring,
            Permission::Microphone => self.microphone,
            Permission::Accessibility => self.accessibility,
        }
    }
}

/// State-query seam matching ZoomItForMac's `PermissionService`. UI policy and
/// tests consume a snapshot rather than reaching into TCC calls themselves.
pub trait PermissionService {
    fn current_state(&self) -> PermissionState;
}

pub struct SystemPermissionService;

impl PermissionService for SystemPermissionService {
    fn current_state(&self) -> PermissionState {
        PermissionState {
            input_monitoring: if unsafe { CGPreflightListenEventAccess() } {
                PermissionStatus::Granted
            } else {
                PermissionStatus::NotGranted
            },
            microphone: microphone_status(),
            accessibility: if accessibility_granted() {
                PermissionStatus::Granted
            } else {
                PermissionStatus::NotGranted
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermissionAction {
    Request,
    OpenSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DashboardScope {
    Startup,
    Dictation,
    #[default]
    All,
}

fn visible_permissions(scope: DashboardScope, state: PermissionState) -> Vec<Permission> {
    match scope {
        DashboardScope::Startup => vec![Permission::InputMonitoring],
        DashboardScope::Dictation => [Permission::Microphone, Permission::Accessibility]
            .into_iter()
            .filter(|permission| !state.status(*permission).is_granted())
            .collect(),
        DashboardScope::All => Permission::ALL.to_vec(),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DashboardModel {
    detail: String,
    permissions: Vec<Permission>,
}

fn dashboard_model(
    scope: DashboardScope,
    state: PermissionState,
    input_needs_relaunch: bool,
) -> DashboardModel {
    let permissions = visible_permissions(scope, state);
    let mut detail = String::new();
    for permission in &permissions {
        let status = state.status(*permission);
        detail.push_str(&format!(
            "{}: {}\n  Used to {}.\n",
            permission.label(),
            status.description(),
            permission.purpose(),
        ));
    }
    match scope {
        DashboardScope::Startup => detail.push_str(
            "\nThe global hotkey is the only capability needed now. Microphone and Accessibility are requested only when you start dictation. Until then, Parakeet remains available from the menu bar.",
        ),
        DashboardScope::Dictation => detail.push_str(
            "\nThese permissions are needed for the dictation you just requested. Parakeet will not begin recording until they are granted.",
        ),
        DashboardScope::All => detail.push_str(
            "\nParakeet requests a permission only after you choose its Grant button or use the feature that needs it.",
        ),
    }
    if input_needs_relaunch {
        detail.push_str(
            "\n\nInput Monitoring was granted after this process created its event tap. Quit and reopen Parakeet once to activate the global hotkey. The menu remains usable now.",
        );
    }
    DashboardModel {
        detail,
        permissions,
    }
}

fn primary_action(permission: Permission, status: PermissionStatus) -> PermissionAction {
    match (permission, status) {
        (Permission::Microphone, PermissionStatus::NotDetermined)
        | (Permission::InputMonitoring | Permission::Accessibility, PermissionStatus::NotGranted) => {
            PermissionAction::Request
        }
        _ => PermissionAction::OpenSettings,
    }
}

fn button_title(permission: Permission, status: PermissionStatus) -> String {
    let verb = match primary_action(permission, status) {
        PermissionAction::Request => "Grant",
        PermissionAction::OpenSettings => "Open",
    };
    let suffix = if matches!(
        primary_action(permission, status),
        PermissionAction::OpenSettings
    ) {
        " Settings…"
    } else {
        "…"
    };
    format!("{verb} {}{suffix}", permission.label())
}

fn startup_needs_onboarding(state: PermissionState) -> bool {
    !state.input_monitoring.is_granted()
}

fn dictation_ready(state: PermissionState) -> bool {
    state.microphone.is_granted() && state.accessibility.is_granted()
}

fn revoked_permissions(before: PermissionState, after: PermissionState) -> Vec<Permission> {
    Permission::ALL
        .into_iter()
        .filter(|permission| {
            before.status(*permission).is_granted() && !after.status(*permission).is_granted()
        })
        .collect()
}

#[derive(Default)]
struct PermissionUiState {
    last_state: Option<PermissionState>,
    refresh_when_active: bool,
    dialog_visible: bool,
    input_missing_at_install: bool,
    return_scope: DashboardScope,
}

thread_local! {
    static UI_STATE: RefCell<PermissionUiState> = RefCell::new(PermissionUiState::default());
}

/// Install permission-state observation after the app's menu and runtime are
/// alive. First launch presents an explanation only when Input Monitoring—the
/// grant needed for the global hotkey at that moment—is missing.
pub fn install(mtm: MainThreadMarker) {
    let state = SystemPermissionService.current_state();
    UI_STATE.with(|slot| {
        let mut ui = slot.borrow_mut();
        ui.last_state = Some(state);
        ui.input_missing_at_install = !state.input_monitoring.is_granted();
    });
    if let Some(scope) = permission_preview_scope() {
        present_dashboard(mtm, scope);
        return;
    }
    if startup_needs_onboarding(state) {
        present_dashboard(mtm, DashboardScope::Startup);
    }
}

fn permission_preview_scope() -> Option<DashboardScope> {
    let value = std::env::var("PARAKEET_PERMISSIONS_PREVIEW").ok()?;
    match value.as_str() {
        "startup" => Some(DashboardScope::Startup),
        "dictation" => Some(DashboardScope::Dictation),
        "all" => Some(DashboardScope::All),
        _ => {
            log::warn!(
                "ignoring unknown PARAKEET_PERMISSIONS_PREVIEW={value:?}; expected startup, dictation, or all"
            );
            None
        }
    }
}

/// Called from `applicationDidBecomeActive:`. It re-presents once after a
/// System Settings trip and also catches later revocation without forcing an
/// unexplained restart.
pub fn application_did_become_active(mtm: MainThreadMarker) {
    let current = SystemPermissionService.current_state();
    let should_present = UI_STATE.with(|slot| {
        let mut ui = slot.borrow_mut();
        let revoked = ui
            .last_state
            .is_some_and(|previous| !revoked_permissions(previous, current).is_empty());
        let should_present = (ui.refresh_when_active || revoked) && !ui.dialog_visible;
        let scope = if revoked {
            DashboardScope::All
        } else {
            ui.return_scope
        };
        ui.refresh_when_active = false;
        ui.last_state = Some(current);
        (should_present, scope)
    });
    if should_present.0 {
        present_dashboard(mtm, should_present.1);
    }
}

/// Gate only the start edge of dictation. Stop/cancel edges must remain usable
/// if a grant is revoked while recording. The menu can start dictation without
/// Input Monitoring; microphone and delivery still need to be ready.
pub fn ensure_dictation_ready() -> bool {
    let ready = dictation_ready(SystemPermissionService.current_state());
    if !ready {
        schedule_dashboard(DashboardScope::Dictation);
    }
    ready
}

/// Explicit menu action, equivalent to ZoomIt's "Check Permissions" command.
pub fn show_dashboard(mtm: MainThreadMarker) {
    present_dashboard(mtm, DashboardScope::All);
}

fn schedule_dashboard(scope: DashboardScope) {
    crate::objc_util::dispatch_to_main(move || {
        let Some(mtm) = MainThreadMarker::new() else {
            log::error!("permission dashboard dispatched off the main thread");
            return;
        };
        present_dashboard(mtm, scope);
    });
}

fn present_dashboard(mtm: MainThreadMarker, scope: DashboardScope) {
    let should_show = UI_STATE.with(|slot| {
        let mut ui = slot.borrow_mut();
        if ui.dialog_visible {
            false
        } else {
            ui.dialog_visible = true;
            true
        }
    });
    if !should_show {
        return;
    }

    let state = SystemPermissionService.current_state();
    let input_needs_relaunch = UI_STATE
        .with(|slot| slot.borrow().input_missing_at_install && state.input_monitoring.is_granted());
    let model = dashboard_model(scope, state, input_needs_relaunch);

    let alert = unsafe { NSAlert::new(mtm) };
    unsafe {
        alert.setMessageText(&NSString::from_str("Parakeet Permissions"));
        alert.setInformativeText(&NSString::from_str(&model.detail));
        alert.setAlertStyle(NSAlertStyle::Informational);
        let _ = alert.addButtonWithTitle(&NSString::from_str("Done"));
        for permission in &model.permissions {
            let permission = *permission;
            let title = button_title(permission, state.status(permission));
            let _ = alert.addButtonWithTitle(&NSString::from_str(&title));
        }
        // NSAlert may otherwise collapse to roughly 260 points and wrap every
        // sentence into a tall, hard-to-scan column. A transparent standard
        // accessory view gives AppKit a stable native minimum width without
        // replacing the alert's layout or controls.
        let spacer = NSView::initWithFrame(
            NSView::alloc(mtm),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(480.0, 1.0)),
        );
        alert.setAccessoryView(Some(&spacer));
    }
    let ns_app = NSApplication::sharedApplication(mtm);
    ns_app.activate();
    let response = unsafe { alert.runModal() };

    UI_STATE.with(|slot| {
        let mut ui = slot.borrow_mut();
        ui.dialog_visible = false;
        ui.last_state = Some(state);
    });

    let index = response - 1000;
    if index <= 0 {
        return;
    }
    let permission_index = usize::try_from(index - 1).ok();
    let Some(permission) = permission_index.and_then(|index| model.permissions.get(index).copied())
    else {
        return;
    };
    perform_action(permission, state.status(permission), scope);
}

fn perform_action(permission: Permission, status: PermissionStatus, scope: DashboardScope) {
    match primary_action(permission, status) {
        PermissionAction::OpenSettings => {
            arm_refresh_when_active(scope);
            if !open_settings(permission) {
                log::error!(
                    "failed to open {} or generic Privacy & Security settings",
                    permission.label()
                );
                schedule_dashboard(scope);
            }
        }
        PermissionAction::Request => match permission {
            Permission::Microphone => request_microphone_async(scope),
            Permission::Accessibility => {
                arm_refresh_when_active(scope);
                if request_accessibility() {
                    schedule_dashboard(scope);
                }
            }
            Permission::InputMonitoring => {
                arm_refresh_when_active(scope);
                if unsafe { CGRequestListenEventAccess() } {
                    schedule_dashboard(scope);
                }
            }
        },
    }
}

fn arm_refresh_when_active(scope: DashboardScope) {
    UI_STATE.with(|slot| {
        let mut ui = slot.borrow_mut();
        ui.refresh_when_active = true;
        ui.return_scope = scope;
    });
}

fn microphone_status() -> PermissionStatus {
    // `[AVCaptureDevice authorizationStatusForMediaType:@"soun"]`:
    // 0 = not determined, 1 = restricted, 2 = denied, 3 = authorized.
    let raw: i32 = unsafe {
        let cls = class!(AVCaptureDevice);
        let media_type = NSString::from_str("soun");
        msg_send![cls, authorizationStatusForMediaType: &*media_type]
    };
    match raw {
        0 => PermissionStatus::NotDetermined,
        1 => PermissionStatus::Restricted,
        2 => PermissionStatus::Denied,
        3 => PermissionStatus::Granted,
        other => {
            log::warn!("unknown AVAuthorizationStatus {other}; treating microphone as denied");
            PermissionStatus::Denied
        }
    }
}

fn request_microphone_async(scope: DashboardScope) {
    unsafe {
        let cls = class!(AVCaptureDevice);
        let media_type = NSString::from_str("soun");
        let block = RcBlock::new(move |_granted: Bool| schedule_dashboard(scope));
        let _: () = msg_send![
            cls,
            requestAccessForMediaType: &*media_type,
            completionHandler: &*block,
        ];
    }
}

fn accessibility_granted() -> bool {
    let null: *const NSDictionary<NSString, NSNumber> = std::ptr::null();
    unsafe { AXIsProcessTrustedWithOptions(null) }
}

fn request_accessibility() -> bool {
    unsafe {
        let key = ax_trusted_check_option_prompt_key();
        let value = NSNumber::new_bool(true);
        let options: Retained<NSDictionary<NSString, NSNumber>> =
            NSDictionary::from_slices(&[&*key], &[&*value]);
        AXIsProcessTrustedWithOptions(&*options)
    }
}

fn ax_trusted_check_option_prompt_key() -> Retained<NSString> {
    let from_symbol = unsafe { Retained::retain(kAXTrustedCheckOptionPrompt.cast_mut()) };
    from_symbol.unwrap_or_else(|| NSString::from_str("AXTrustedCheckOptionPrompt"))
}

fn open_settings(permission: Permission) -> bool {
    open_url(permission.settings_url()) || open_url(GENERIC_PRIVACY_SETTINGS_URL)
}

fn open_url(value: &str) -> bool {
    let value = NSString::from_str(value);
    let Some(url) = NSURL::URLWithString(&value) else {
        return false;
    };
    NSWorkspace::sharedWorkspace().openURL(&url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(
        input_monitoring: PermissionStatus,
        microphone: PermissionStatus,
        accessibility: PermissionStatus,
    ) -> PermissionState {
        PermissionState {
            input_monitoring,
            microphone,
            accessibility,
        }
    }

    #[test]
    fn launch_onboarding_only_requires_the_launch_time_capability() {
        assert!(!startup_needs_onboarding(state(
            PermissionStatus::Granted,
            PermissionStatus::NotDetermined,
            PermissionStatus::NotGranted,
        )));
        assert!(startup_needs_onboarding(state(
            PermissionStatus::NotGranted,
            PermissionStatus::Granted,
            PermissionStatus::Granted,
        )));
    }

    #[test]
    fn dictation_gate_requires_microphone_and_delivery_but_not_hotkey() {
        assert!(dictation_ready(state(
            PermissionStatus::NotGranted,
            PermissionStatus::Granted,
            PermissionStatus::Granted,
        )));
        assert!(!dictation_ready(state(
            PermissionStatus::Granted,
            PermissionStatus::Denied,
            PermissionStatus::Granted,
        )));
        assert!(!dictation_ready(state(
            PermissionStatus::Granted,
            PermissionStatus::Granted,
            PermissionStatus::NotGranted,
        )));
    }

    #[test]
    fn dashboard_scope_never_front_loads_future_permission_requests() {
        let all_missing = state(
            PermissionStatus::NotGranted,
            PermissionStatus::NotDetermined,
            PermissionStatus::NotGranted,
        );
        assert_eq!(
            visible_permissions(DashboardScope::Startup, all_missing),
            vec![Permission::InputMonitoring]
        );
        assert_eq!(
            visible_permissions(DashboardScope::Dictation, all_missing),
            vec![Permission::Microphone, Permission::Accessibility]
        );

        let microphone_granted = state(
            PermissionStatus::NotGranted,
            PermissionStatus::Granted,
            PermissionStatus::NotGranted,
        );
        assert_eq!(
            visible_permissions(DashboardScope::Dictation, microphone_granted),
            vec![Permission::Accessibility]
        );
        assert_eq!(
            visible_permissions(DashboardScope::All, microphone_granted),
            Permission::ALL
        );
    }

    #[test]
    fn contextual_dashboard_copy_matches_the_actions_it_offers() {
        let all_missing = state(
            PermissionStatus::NotGranted,
            PermissionStatus::NotDetermined,
            PermissionStatus::NotGranted,
        );
        let startup = dashboard_model(DashboardScope::Startup, all_missing, false);
        assert_eq!(startup.permissions, vec![Permission::InputMonitoring]);
        assert!(startup
            .detail
            .contains("global hotkey is the only capability needed now"));
        assert!(startup
            .detail
            .contains("requested only when you start dictation"));

        let dictation = dashboard_model(DashboardScope::Dictation, all_missing, false);
        assert_eq!(
            dictation.permissions,
            vec![Permission::Microphone, Permission::Accessibility]
        );
        assert!(dictation.detail.contains("dictation you just requested"));
        assert!(!dictation.detail.contains("Input Monitoring:"));

        let after_input_grant = dashboard_model(DashboardScope::Startup, all_missing, true);
        assert!(after_input_grant
            .detail
            .contains("Quit and reopen Parakeet once"));
        assert!(after_input_grant.detail.contains("menu remains usable now"));
    }

    #[test]
    fn microphone_four_state_policy_has_an_explicit_recovery_action() {
        assert_eq!(
            primary_action(Permission::Microphone, PermissionStatus::NotDetermined),
            PermissionAction::Request
        );
        for status in [
            PermissionStatus::Granted,
            PermissionStatus::Denied,
            PermissionStatus::Restricted,
        ] {
            assert_eq!(
                primary_action(Permission::Microphone, status),
                PermissionAction::OpenSettings
            );
        }
    }

    #[test]
    fn binary_permissions_request_when_missing_and_open_settings_when_granted() {
        for permission in [Permission::InputMonitoring, Permission::Accessibility] {
            assert_eq!(
                primary_action(permission, PermissionStatus::NotGranted),
                PermissionAction::Request
            );
            assert_eq!(
                primary_action(permission, PermissionStatus::Granted),
                PermissionAction::OpenSettings
            );
        }
    }

    #[test]
    fn revocation_detects_only_granted_to_missing_transitions() {
        let before = state(
            PermissionStatus::Granted,
            PermissionStatus::Granted,
            PermissionStatus::Granted,
        );
        let after = state(
            PermissionStatus::NotGranted,
            PermissionStatus::Denied,
            PermissionStatus::Granted,
        );
        assert_eq!(
            revoked_permissions(before, after),
            vec![Permission::InputMonitoring, Permission::Microphone]
        );
        assert!(revoked_permissions(after, after).is_empty());
    }

    #[test]
    fn each_permission_has_a_specific_settings_link_and_generic_fallback_exists() {
        for permission in Permission::ALL {
            assert!(permission
                .settings_url()
                .starts_with("x-apple.systempreferences:"));
            assert!(permission.settings_url().contains("Privacy_"));
        }
        assert_eq!(
            GENERIC_PRIVACY_SETTINGS_URL,
            "x-apple.systempreferences:com.apple.preference.security"
        );
    }
}
