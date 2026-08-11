# macOS permissions

Parakeet separates permission state, explanation, request, and recovery. It
does not call a macOS TCC request API just because the process launched, and it
does not exit when a permission is absent.

## User flow

| Moment | Permission offered | If deferred |
|---|---|---|
| First launch | Input Monitoring, for the global hotkey | Parakeet stays in the menu bar; Start Dictation remains available there |
| First Start Dictation | Only missing Microphone and/or Accessibility grants | No recording begins; the permission dashboard remains available |
| Check Permissions… | Current state and action for all three | No side effect until the user chooses a Grant/Open button |
| Return from System Settings | Refreshed state | Denied/restricted/missing state retains an explicit Settings recovery action |
| Previously granted permission is revoked | The dashboard reappears when Parakeet becomes active | Stop/cancel remains usable; a new recording is gated if delivery or microphone access is missing |

Input Monitoring is not required for menu-bar dictation. When it is granted
after the current process created its event tap, Parakeet explains the one
quit/reopen macOS needs to activate the global hotkey; it does not present an
unexplained generic restart instruction.

## State and action map

| Permission | Published state | Primary action |
|---|---|---|
| Microphone | Not requested | Grant, using `AVCaptureDevice.requestAccess` |
| Microphone | Denied | Open the Microphone privacy pane |
| Microphone | Restricted | Open the Microphone privacy pane; explain that macOS/device policy controls it |
| Microphone | Granted | Open the Microphone privacy pane so it can be reviewed or changed |
| Input Monitoring | Not granted | Grant/register with `CGRequestListenEventAccess` |
| Input Monitoring | Granted | Open the Input Monitoring privacy pane |
| Accessibility | Not granted | Grant/register with `AXIsProcessTrustedWithOptions` and the prompt option |
| Accessibility | Granted | Open the Accessibility privacy pane |

AVFoundation publishes all four microphone states. CoreGraphics and the
Accessibility preflight publish only a Boolean, so Parakeet deliberately says
“Not granted” instead of inventing a denied/not-determined distinction. Each
service-specific `x-apple.systempreferences:` link falls back to the generic
Privacy & Security pane if macOS rejects it.

Parakeet does not use or request Screen Recording or Camera.

## ZoomItForMac implementation map

The reference was Microsoft ZoomItForMac at commit
[`e14bc9b97e784a5208addc8031f9f1c17c6f3a7f`](https://github.com/microsoft/ZoomitForMac/tree/e14bc9b97e784a5208addc8031f9f1c17c6f3a7f),
reviewed on 2026-08-11. The useful patterns are in
[`PermissionService.swift`](https://github.com/microsoft/ZoomitForMac/blob/e14bc9b97e784a5208addc8031f9f1c17c6f3a7f/Sources/ZoomItMacCore/Permissions/PermissionService.swift)
and
[`AppController.swift`](https://github.com/microsoft/ZoomitForMac/blob/e14bc9b97e784a5208addc8031f9f1c17c6f3a7f/Sources/ZoomItMacCore/App/AppController.swift).

| ZoomIt pattern | Parakeet implementation | Intentional difference |
|---|---|---|
| `PermissionService` state seam | `PermissionService` and `PermissionState` in `src/permissions.rs` | Rust policy functions are unit-tested without AppKit |
| AVFoundation tri-state microphone | Four-state `PermissionStatus` | Parakeet preserves `restricted` rather than folding it into denied |
| Check Permissions command | Permanent menu item in `src/menubar.rs` | Shows Input Monitoring and Accessibility instead of Screen Recording/Camera |
| Specific Settings links | Per-permission URL plus generic fallback | Adds fallback rather than silently doing nothing on a rejected URL |
| Re-present on app activation | `applicationDidBecomeActive:` in `src/app_delegate.rs` | Also detects later revocation |
| Request optional capture grants when feature is selected | Dictation gate requests Microphone/Accessibility in context | Process launch never requests any grant |

The implementation is an independent Rust adaptation of the interaction
architecture; no Swift source is vendored.

## Signed release QA matrix

Use a consistently signed bundle so TCC identity does not change between
runs:

```bash
PARAKEET_SIGN_ID='Parakeet Local Dev' scripts/make-app.sh
codesign --verify --deep --strict target/release/bundle/osx/Parakeet.app
```

For visual QA without changing TCC state, launch the built executable with
`PARAKEET_PERMISSIONS_PREVIEW=startup`, `dictation`, or `all`. This chooses the
dashboard scope but still reads the real permission states and uses the real
actions:

```bash
PARAKEET_PERMISSIONS_PREVIEW=startup \
  target/release/bundle/osx/Parakeet.app/Contents/MacOS/parakeet-rs
```

For a clean-state test, these development commands remove the existing grants
for bundle identifier `com.parakeet.rs`:

```bash
tccutil reset ListenEvent com.parakeet.rs
tccutil reset Microphone com.parakeet.rs
tccutil reset Accessibility com.parakeet.rs
```

They intentionally change local privacy settings. Run them only when prepared
to grant the permissions again.

| Case | Procedure | Expected result | 2026-08-11 signed build |
|---|---|---|---|
| Clean launch | Reset all three, launch from `/Applications` | One explanatory Parakeet dialog offers only Input Monitoring; app/menu stay alive; no Microphone or Accessibility system prompt | Pending |
| Input Monitoring grant | Choose Grant Input Monitoring and enable Parakeet | Return refreshes status; dashboard explains one quit/reopen for global hotkey; menu dictation remains available | Pending |
| Contextual dictation grants | With Microphone/Accessibility reset, choose Start Dictation | Dashboard offers only the missing dictation grants; recording does not start prematurely | Pending |
| Microphone denied | Deny the first system request, then try again | State says Denied and button opens Microphone settings rather than repeating the system prompt | Pending |
| Microphone restricted | Exercise on a managed/restricted Mac; policy test covers state mapping elsewhere | State says Restricted and directs recovery to Settings/policy owner | Not reproducible on unmanaged test Mac |
| Accessibility missing | Disable Accessibility while app is running, reactivate Parakeet | Dashboard surfaces revocation; new dictation is gated; menu and stop/cancel remain usable | Pending |
| Microphone revoked | Disable Microphone while app is running, reactivate Parakeet | Dashboard surfaces revocation; new dictation is gated with an explicit recovery action | Pending |
| Input Monitoring revoked | Disable Input Monitoring while app is running, reactivate Parakeet | Dashboard surfaces revocation; menu dictation still works; global hotkey is unavailable | Pending |
| Deep-link fallback | Use each Open Settings action; unit tests retain generic fallback contract | Correct service pane opens, or generic Privacy & Security opens | Pending |
| Fully granted relaunch | Grant all three, quit, reopen | No onboarding dialog; global hotkey and menu dictation work without another prompt | Pending |

After QA, restore all three grants and leave the signed `/Applications` bundle
running. Record the tested commit and replace each Pending cell before release.
