# AppKit Checklist for Rust / `objc2-app-kit`

> Research snapshot from May 15, 2026. This is a general reference checklist, not a statement of current `parakeet-rs` implementation status. See the project [README](../README.md) and [ADR](../docs/ADR.md) for current behavior.

AppKit is a large, event-driven Objective-C framework with decades of conventions. When using it from Rust, keep the impedance mismatch explicit: AppKit is a dynamic, reentrant, main-thread-owned object runtime, while Rust is a static, ownership-checked, concurrency-safe systems language. Keep AppKit at the boundary, translate it into typed Rust state/events quickly, and preserve the `.app` bundle context AppKit expects.

## 1. App Startup, Bundle Context, Lifecycle, and Main Thread

- [ ] **Main-thread startup:** Acquire `objc2::MainThreadMarker` at startup with `MainThreadMarker::new().unwrap()` and pass it to APIs that require main-thread proof. Use `unsafe MainThreadMarker::new_unchecked()` only for tightly justified no-check contexts.

- [ ] **Develop and test inside a real `.app`:** Treat bundling as part of the dev loop, not only distribution. Launch a bundled app with `open -W path/to/MyApp.app` so Dock/menu activation, `Bundle.main`, `Info.plist`, document/URL events, TCC privacy prompts, resource lookup, and signing-sensitive capabilities behave like production.

- [ ] **Single source of bundle truth:** Keep bundle id, version, executable name, icons, privacy strings, document types, URL schemes, minimum macOS version, and `LSUIElement`/activation policy in `[package.metadata.bundle]`, `cargo-packager`, or an `xtask`/`justfile` template that generates `Contents/{Info.plist,MacOS,Resources}` consistently.

- [ ] **Ad-hoc-signed dev builds:** Provide a dev command such as `cargo xtask dev` that builds, bundles, ad-hoc signs with `codesign --sign -`, and relaunches the `.app`, so TCC and Launch Services see a stable app identity across iterations.

- [ ] **One `NSApplication` boot path:** For programmatic apps, use `NSApplication::sharedApplication(mtm)`, configure activation policy, install the delegate, create initial UI from the launch callback, and call `app.run()`. For storyboard/nib apps, use `NSApplication::main(mtm)` and register Rust-defined classes before AppKit loads them.

- [ ] **Application delegate:** Define an `AppDelegate` with `objc2::define_class!`, `#[thread_kind = MainThreadOnly]`, `#[ivars = AppState]`, `unsafe impl NSObjectProtocol`, and `unsafe impl NSApplicationDelegate`. Put launch, termination, reopen, open-file/open-url, user-activity, and app-service routing there rather than in random views.

- [ ] **Run-loop discipline:** Never block the main run loop with CPU-heavy Rust work or synchronous I/O. AppKit objects are not general-purpose concurrent data; move pure Rust data/work off-thread and post typed UI updates back to the main thread. Do not smuggle `!Send` AppKit objects through `Arc<Mutex<_>>`, globals, raw pointers, or erased wrapper types.

- [ ] **Main-queue bridge:** Use an explicit bridge such as `dispatch2::DispatchQueue::main().exec_async(...)` or another documented main-queue primitive for AppKit updates. Prefer channels carrying a `MainThreadCommand` enum that the app delegate/controller matches on the main thread.

- [ ] **Async-runtime policy:** Document one async integration model: message-passing back to the main queue, a local executor pumped from the AppKit/CFRunLoop, or another explicit design. Do not rely on ambient `tokio::spawn` from event handlers unless the wakeup path into the AppKit run loop is proven. Add tests that prove worker results wake the main-thread consumer without blocking the AppKit run loop.

- [ ] **Window/controller lifetime:** Prefer `NSWindowController` for normal app windows. If creating raw `NSWindow`s, decide ownership explicitly; for Rust-owned windows, call `setReleasedWhenClosed(false)` and retain them for as long as AppKit may use them. For every delegate, data source, target, observer, window, and block, write down who owns it, who may call back later, and what makes that callback stop.

- [ ] **State restoration:** Assign stable window/controller restoration identifiers. Decide what restores: open documents, selected item, scroll position, split sizes, search text, transient panels, and multi-display placement. Validate files/resources before restoring stale state, and regression-test quit/relaunch, crash/relaunch, and missing-file restore paths.

- [ ] **Termination and reopen semantics:** Decide whether the app quits after the last window closes, how Dock reopen works, and how files/URLs opened from Finder, Dock, `open`, or custom URL schemes are routed.

## 2. `objc2` Safety, Initialization, and Objective-C Interop

- [ ] **Current class macros:** Use `objc2::define_class!` for Rust-defined Objective-C classes. For AppKit/Foundation classes, use the generated types from `objc2-app-kit` / `objc2-foundation`; reserve `extern_class!` for Objective-C classes not already provided by framework crates. Do not use `declare_class!` examples for `objc2` 0.6+; the macro was removed/renamed and re-syntaxed.

- [ ] **Ownership types:** Treat `alloc()` results as `Allocated<T>` that must be initialized exactly once. Store owned Objective-C objects as `Retained<T>` or `Option<Retained<T>>`; do not hide ownership in raw pointers. Any raw pointer crossing should have a nearby comment naming the retain/borrow convention, owner, and lifetime bound.

- [ ] **Designated initializer chain:** For each `define_class!` subclass, implement an `init...` selector that consumes `Allocated<Self>`, initializes ivars with the current `objc2` pattern such as `this.set_ivars(...)`, calls the superclass designated initializer with `super(this)`, and returns `Retained<Self>`. Never let `Allocated<Self>` escape or drop uninitialized.

- [ ] **Superclass contracts:** For `NSView`, `NSWindow`, `NSViewController`, `NSDocument`, data sources, and delegates, identify the superclass designated initializer and lifecycle methods you rely on. Broken super-init chains can corrupt responder chains, layout, tracking areas, redraws, and delegate callbacks at runtime.

- [ ] **Main-thread-only UI classes:** Use `#[thread_kind = MainThreadOnly]` for AppKit-facing delegates, views, view controllers, window controllers, and data sources that touch UI state.

- [ ] **Interior mutability and reentrancy:** Because Objective-C objects are aliased and methods usually receive `&self`, store mutable ivars in `Cell`, `RefCell`, `Mutex`, atomics, or another deliberate interior-mutability type. Never hold a mutable borrow across calls into AppKit that can synchronously trigger layout, drawing, validation, KVO, notifications, delegate callbacks, or target/action reentry.

- [ ] **Generated bindings before raw `msg_send!`:** Prefer `objc2-app-kit` methods. If raw `msg_send!` is unavoidable, document selector validity, argument/return encodings, ownership convention, nullability, exception behavior, and thread requirements near the `unsafe` block. Treat selectors, optional protocol methods, responder actions, KVC/KVO key paths, notification names, pasteboard types, nib identifiers, and `Info.plist` declarations as dynamic runtime edges that need typed wrappers or smoke tests.

- [ ] **Feature flags:** Enable the `objc2-app-kit` per-class/per-protocol features for every AppKit type or protocol you use, such as `NSWindow`, `NSView`, or `NSApplicationDelegate`. Missing features usually appear as compile-time missing-type/missing-method errors, not mysterious linker behavior. Keep a curated feature set; do not blindly use `--all-features` as the app's correctness gate.

- [ ] **Panic boundaries:** The real contract is that Rust panics must never unwind into Objective-C/AppKit frames. `std::panic::catch_unwind` only catches with `panic = "unwind"`; under `panic = "abort"`, the process aborts. Choose the bundle target's panic strategy deliberately, wrap AppKit-invoked Rust callbacks accordingly, and convert recoverable failures into typed UI errors before crossing the Objective-C boundary.

- [ ] **NSException discipline:** Treat uncaught `NSException` as a programmer error comparable to abort, not a recoverable control path. Design wrappers that make range errors, nil errors, and KVC/key-path errors unreachable by construction; prefer `Option`, `Result`, enums, and bounds-checked helpers over Objective-C exception paths. Use `objc2::exception::catch` only where justified, note that it requires the `objc2` `"exception"` cargo feature, and prefer outer crash-reporting boundaries over continuing after AppKit invariants may be broken.

- [ ] **Selectors and optional protocols:** Prefer typed protocol implementations and generated bindings. When manually exposing selectors, keep names next to AppKit documentation and add smoke tests that exercise them via real AppKit dispatch, not only direct Rust method calls.

## 3. Ownership, Observers, Blocks, and Cleanup

- [ ] **Weak AppKit relationships:** Delegates, data sources, targets, observers, and controller links are often weak/non-owning. Keep Rust-side `Retained<T>` owners for anything AppKit may call later.

- [ ] **Cycles and weak back-pointers:** Break parent/back-pointer/callback cycles with `objc2::rc::Weak`; upgrade weak references only for the duration of a call.

- [ ] **Block capture discipline:** For Objective-C block callbacks, use `block2::RcBlock` when the block must be heap-promoted/retained and `block2::StackBlock` only when stack lifetime is sufficient. Do not capture a strong `Retained<Self>` from a block owned by `Self`; capture `Weak<T>`, an ID, immutable data, or a channel sender instead.

- [ ] **Completion-handler boundary:** Treat AppKit/Foundation completion handlers such as sheet completions, open/save panels, and URL/session callbacks as explicit FFI boundaries. Check callback thread, lifetime of captures, cancellation behavior, and whether UI work must be reposted to the main queue; test success, cancellation, late callback after owner drop, and reentrant callback cases.

- [ ] **Notification tokens:** Store notification/observer tokens in RAII wrappers and unregister in `Drop`. Avoid anonymous observers that cannot be removed.

- [ ] **KVO as interop, not architecture:** KVO is valid for documented KVO-compliant Objective-C/AppKit properties, but it is stringly typed, reentrant, and lifetime-sensitive. `objc2` does not provide a turnkey `KeyValueObservation` wrapper; if KVO is required, wrap `addObserver:forKeyPath:options:context:` / `removeObserver:` manually in an RAII type that unregisters exactly once in `Drop`, documents callback-thread assumptions, and tests normal change delivery, owner/observed-object teardown order, and duplicate/unbalanced removal.

- [ ] **Typed state over KVC:** Prefer typed Rust state, commands, channels, callbacks, or controller update methods over KVC/KVO/Cocoa Bindings for primary app architecture. If using KVC/key paths, hide them behind typed property accessors to avoid `NSUndefinedKeyException` at runtime.

- [ ] **Autorelease pools:** Wrap high-allocation loops, worker-thread Objective-C calls, and CLI/test entry points in `objc2::rc::autoreleasepool` when autoreleased objects can accumulate. In current `objc2`, the closure receives an `AutoreleasePool` token.

- [ ] **CoreFoundation ownership:** Prefer typed framework wrappers. If manually wrapping `CFTypeRef`, follow Create/Copy/Get conventions and call `CFRelease` only for owned results.

- [ ] **Temporary files and caches:** Clean up temporary/cache files on graceful shutdown and on next launch after crashes.

## 4. AppKit UI Architecture, Commands, and System Integration

- [ ] **Boundary architecture:** Keep domain state, persistence, validation, and background work in pure Rust modules. Let AppKit objects adapt that model to windows, views, menus, delegates, responder-chain commands, pasteboard, documents, and system services. Convert AppKit callbacks into typed Rust events/commands at the edge rather than letting the Objective-C object graph become the app's core state model.

- [ ] **Responder chain and global menu bar:** Use standard selectors such as `sel!(copy:)` and `nil`/`None` targets for standard commands so AppKit routes actions through the active responder chain.

- [ ] **Command validation:** Menu and toolbar items should enable, disable, and rename themselves from current context using `NSUserInterfaceValidations` / `NSMenuItemValidation`-style validation rather than stale global state. Unit-test validation logic as pure state transitions and smoke-test it through actual menu/toolbar items.

- [ ] **Toolbars as command surfaces:** Use `NSToolbar` / `NSToolbarDelegate` for primary window commands, assign stable toolbar identifiers, decide whether customization is allowed, persist customization where appropriate, and mirror toolbar commands in menus for discoverability and keyboard access.

- [ ] **Window vs view controllers:** Use `NSWindowController` for window lifecycle, chrome, toolbar, autosave names, screen placement, tabs, and restoration. Use `NSViewController` for content.

- [ ] **Sheets, alerts, panels, and popovers:** Prefer window/document-modal sheets for decisions scoped to one window; use `NSAlert` for simple confirmations/errors, `NSPanel` for utility/inspector windows, and `NSPopover` for contextual UI. Define dismissal, key-window, focus, and accessibility behavior explicitly.

- [ ] **First responder and keyboard navigation:** Decide which views can become first responder, support keyboard equivalents and tab navigation, and test without a mouse.

- [ ] **Text system:** Prefer `NSTextView` / TextKit for rich text, selection, spellchecking, find, undo, accessibility, input methods, and services. For custom text editing/canvas text, support marked text, composition, dead keys, Unicode input, selection ranges, writing direction, and standard editing commands rather than only `keyDown:`.

- [ ] **Mouse, cursor, and drag/drop:** Use `NSTrackingArea`, cursor rects, responder methods, `NSDraggingSource`, `NSDraggingDestination`, and pasteboard reader/writer protocols rather than polling mouse state.

- [ ] **Drag/drop payloads and file promises:** Use UTType-based payloads, multiple representations, lazy data providers, and file promises for dragging generated/exported files to Finder without prewriting them. Revalidate operations as modifier keys and destinations change.

- [ ] **Collections and virtualization:** Avoid legacy cell-based `NSTableView` APIs for new work. Use view-based `NSTableView`, `NSOutlineView`, `NSCollectionView`, `NSSplitViewController`, and `NSScrollView` conventions with reuse, lazy data access, keyboard navigation, drag/drop, and accessibility.

- [ ] **Dark mode, materials, and appearance:** Use semantic system colors/materials, not hardcoded RGB. Invalidate custom drawing when effective appearance changes; test vibrancy/sidebar materials and non-default accent colors.

- [ ] **Accessibility preferences:** Respect Reduce Motion, Increase Contrast, Differentiate Without Color, transparency/vibrancy preferences, and accent color. Avoid conveying state by color alone.

- [ ] **Window styling and safe areas:** Use full-size content views, transparent title bars, toolbars, and sidebars deliberately. Respect safe-area insets around traffic-light controls and title/toolbar regions.

- [ ] **Accessibility:** Provide roles, labels, values, actions, focus behavior, and custom accessibility elements for custom controls. Set stable accessibility identifiers for UI automation where appropriate, and test with VoiceOver, Accessibility Inspector, and scripted accessibility-tree smoke checks.

- [ ] **App services:** Use native macOS integration where appropriate: `NSUserActivity`/Handoff for resumable context, `NSServicesMenuRequestor` for Services menu support, `NSSharingService` for sharing, `NSHelpManager` for Help menu/searchable help, `NSWorkspace` for Finder/default-app/external-open workflows, and Continuity Camera for importing scans/images.

- [ ] **Dock, status items, and activation:** If using Dock badges/menus, status-bar items, accessory apps, or agent-style UI, define activation policy, menu-bar behavior, reopen behavior, accessibility visibility, and quit affordances explicitly.

## 5. Layout, Rendering, Assets, Color, and Printing

- [ ] **Auto Layout first:** Prefer constraints, stack views, intrinsic content size, and hugging/compression priorities over manual frames for standard controls. Call `setTranslatesAutoresizingMaskIntoConstraints(false)` when adding constraints programmatically.

- [ ] **Coordinate systems:** AppKit's default coordinate system is bottom-left. For custom top-left-oriented views, override `isFlipped` in an `NSView` subclass; do not assume all AppKit views are flipped.

- [ ] **Coordinate conversion:** Use AppKit conversion methods between view, window, screen, layer, and event spaces, especially across flipped views, multiple displays, and backing-scale changes.

- [ ] **Retina/backing scale:** One point is not one pixel. Use backing-scale APIs for custom drawing, hit testing, image generation, and pixel-aligned rendering.

- [ ] **Layer-backed vs layer-hosting:** `setWantsLayer(true)` changes drawing and invalidation behavior; it is not a universal performance switch. Decide explicitly between AppKit drawing, layer-backed views, and custom `CALayer`s.

- [ ] **Redraw discipline:** Invalidate only dirty regions, avoid allocations in draw calls, cache expensive layout/text shaping, and profile live resize and scrolling. For custom drawing, add deterministic render tests or golden-image tests with fixed appearance, scale, locale, font assumptions, and tolerances for antialiasing.

- [ ] **Image and icon assets:** Provide multi-resolution icons, template images where appropriate, dark-mode-compatible assets, asset-catalog or bundle-resource lookup policy, and correct `Info.plist` icon declarations.

- [ ] **Nib/storyboard/resource policy:** Decide whether UI is fully programmatic or uses nib/storyboard resources. If using nibs/storyboards, validate owner/outlet/action wiring, class names, localized resources, bundle lookup, and failure behavior. If avoiding Interface Builder, document replacement conventions for layout, localization, accessibility, and screenshots.

- [ ] **Color management:** Use `NSColor` / `CGColor` color spaces deliberately for imported/exported images, PDF, and print. Do not assume every asset or display path is plain sRGB.

- [ ] **Printing/export:** For document, drawing, or report apps, implement print/PDF/export from the model or rendering pipeline rather than screenshotting the current UI. Test print panel, page setup, margins, scaling, pagination, preview, and PDF/vector fidelity.

## 6. Documents, Data, User State, and File Workflows

- [ ] **Document model:** If editing user files, evaluate `NSDocument` / document-controller patterns for recent documents, autosave, window restoration, file coordination, duplicate/rename/move handling, and standard File menu behavior. Test file-open, duplicate, rename, move, autosave, conflict, permission-denied, and corrupt-file paths against temporary files.

- [ ] **Open files and URLs:** Register document types and URL schemes in `Info.plist`; route open-file/open-url events through the app delegate or document controller.

- [ ] **Pasteboard:** Use `NSPasteboard` with appropriate UTTypes, multiple representations, promised/lazy data, and privacy-conscious reads for large or sensitive payloads. Test round-trips with unique/private pasteboards where possible so tests do not consume or leak the user's real clipboard contents.

- [ ] **Undo/redo:** Integrate `NSUndoManager` or a native-feeling undo stack so standard Edit menu items work through the responder chain.

- [ ] **Preferences:** Use `NSUserDefaults` or a clearly versioned preferences store with explicit defaults, migrations, reset behavior, and, if needed, iCloud key-value sync policy. Run tests under isolated preference domains or temporary homes such as `CFFIXED_USER_HOME` so user defaults and recent-doc state do not leak across test runs.

- [ ] **Progress, cancellation, and errors:** Use native progress indicators or `NSProgress`-style state for long work, make operations cancellable, and present errors in the window/document context they affect. Normalize AppKit's mixed failure surfaces (`nil`, `NSError`, delegate callbacks, modal response codes, notifications, and exceptions-for-programmer-error) into explicit Rust `Result`/`Option`/enum outcomes before updating the model.

## 7. Localization and Internationalization

- [ ] **Localized resources:** Do not bury user-facing strings permanently in Rust source. Plan bundle resource lookup for localized `.strings` files or another localization pipeline that fits your Rust build.

- [ ] **Locale-sensitive formatting:** Use Foundation formatters or a Rust localization stack for dates, numbers, measurements, currencies, names, and sorting.

- [ ] **Variable-length and bidirectional text:** Prefer leading/trailing layout semantics, constraints, and intrinsic content size over hardcoded left/right positions and fixed widths.

- [ ] **Input and writing systems:** Test non-US keyboard layouts, IMEs, bidirectional text, composed Unicode characters, font fallback, localized resources, and pseudolocalized/long-string layouts for the app's supported languages.

## 8. Privacy, Security, Helpers, Bundling, and Distribution

- [ ] **Sandbox and file access:** Use `NSOpenPanel` / `NSSavePanel` for user-granted file access. If access must persist across launches, implement security-scoped bookmarks and release access when done.

- [ ] **Bundle structure:** Ensure every build path creates a valid `.app` with `Contents/Info.plist`, `Contents/MacOS/<executable>`, `Contents/Resources`, and `Contents/Frameworks` when embedding frameworks/helpers.

- [ ] **`Info.plist` completeness:** Verify bundle identity/version keys, executable name, minimum system version, icons, document types, URL schemes, high-resolution capability, app category, activation policy hints, and all required privacy usage descriptions.

- [ ] **Privacy usage descriptions:** Include the relevant `NS...UsageDescription` keys for camera, microphone, contacts, calendars, photos, location, Bluetooth, local network, Apple Events, and other protected resources. Treat missing privacy strings as release-blocking.

- [ ] **Hardened runtime and entitlements:** Sign with hardened runtime and a minimal explicit entitlements file. Review sandbox, network, file access, Apple Events, JIT, unsigned executable memory, automation, helper, and app-group permissions individually.

- [ ] **Helpers and extensions:** If functionality belongs outside the main app, evaluate app extensions, Finder Sync, Share extensions, Quick Look, Spotlight importers, XPC helpers, login items, or privileged helpers. Sign, sandbox, entitle, embed, version, and notarize helpers/extensions independently.

- [ ] **Signing, notarization, and Gatekeeper:** Sign each nested binary/framework/helper explicitly before signing the app; avoid `--deep` as a signing shortcut. Submit signed archives with `xcrun notarytool`, staple accepted tickets, and test the downloaded/quarantined artifact path rather than only the local build.

- [ ] **Universal binary policy:** Decide whether to ship arm64-only or universal arm64+x86_64; verify the executable and every embedded native dependency.

- [ ] **Update mechanism:** If distributing outside the Mac App Store, use a signed update mechanism such as Sparkle or an equivalent design with signature verification, rollback handling, and notarized update artifacts.

- [ ] **Network security:** Prefer HTTPS, configure App Transport Security exceptions only when necessary, and review certificate-pinning or custom trust requirements explicitly.

## 9. Logging, Diagnostics, Performance, and Supportability

- [ ] **Unified logging:** Bridge Rust logs to Apple's unified logging system, e.g. with `oslog` or equivalent, so logs appear in Console.app with subsystem/category metadata.

- [ ] **Crash reports and symbols:** Configure symbol generation explicitly. Rust release builds commonly embed DWARF in the Mach-O unless configured otherwise; use `[profile.release] split-debuginfo = "packed"` or run `dsymutil target/release/<binary> -o <binary>.dSYM`, preserve the resulting symbols, upload them to the crash reporter, and verify panic/crash symbolication.

- [ ] **Log levels and privacy:** Hide debug/trace logs in production, keep errors/faults visible, and avoid logging private user data.

- [ ] **Support diagnostics:** Provide a debug/support path that reports bundle path, resource lookup, entitlements, sandbox state, OS version, architecture, enabled features, update channel, relevant privacy authorization states, and active test-mode launch arguments/environment.

- [ ] **Minimize FFI chattiness:** Batch AppKit updates, coalesce invalidations, use model snapshots, and avoid thousands of subviews when custom drawing or virtualization fits better.

- [ ] **Lazy view loading:** Defer complex `NSViewController` hierarchies and heavy resources until display time via `loadView` / lifecycle hooks.

- [ ] **Profile release-like builds:** Use Instruments Time Profiler, Allocations, Leaks, Main Thread Checker, Core Animation, file/network instruments, and `xcrun xctrace record`/export workflows against the compiled `.app`, not only `cargo run`. Keep repeatable launch/performance scenarios for regression comparisons.

## 10. Testing, CI, Static Validation, and Release Gates

- [ ] **Test pyramid:** Keep most behavior in pure Rust model/controller tests that do not create AppKit objects. Use AppKit integration tests for lifecycle, responder-chain, run-loop, ownership, dynamic-dispatch edges, reentrancy, error translation, and binding behavior, and reserve full `.app` UI tests for user-visible flows.

- [ ] **TDD for risky behavior:** For bug fixes and new behavior, write a failing regression test first, watch it fail for the expected reason, implement the minimal fix, then run focused and full gates. Prioritize tests around unsafe FFI wrappers, selectors, lifecycle callbacks, async bridges, document I/O, preferences migrations, and security/privacy decisions.

- [ ] **Main-thread AppKit tests:** Put AppKit tests in a separate target or clearly marked module that runs on the main thread with `MainThreadMarker::new().unwrap()`. Run them serially, e.g. `cargo test appkit -- --test-threads=1`, and skip or fail clearly when a CI runner cannot provide a GUI session.

- [ ] **Autorelease and lifecycle tests:** Wrap Objective-C-heavy tests in `objc2::rc::autoreleasepool`; assert observer/token cleanup, weak-reference teardown, block capture behavior, and absence of retain cycles with leak tooling rather than relying on visual inspection.

- [ ] **Compile-fail and type-safety tests:** Use doctest `compile_fail` examples or `trybuild`-style UI tests for invariants that should be rejected by Rust: sending `MainThreadOnly` UI objects across threads, missing `MainThreadMarker`, stale selectors, wrong `msg_send!` signatures, missing feature flags, and invalid ownership conversions.

- [ ] **Debug-build FFI verification:** Run a debug test job that exercises raw `msg_send!`, selector, and nil/encoding-sensitive paths so `objc2` debug assertions can catch signature mistakes before optimized release builds hide them.

- [ ] **Headless vs GUI split:** Separate tests that can run in headless CI from tests requiring a logged-in macOS GUI session, accessibility permission, windows, TCC prompts, or real displays. Mark slow/flaky/manual tests explicitly instead of silently omitting them.

- [ ] **Bundled app smoke tests:** Build, ad-hoc sign, and launch the generated `.app` in CI or a local release gate with `open -W` or an equivalent harness. Verify activation, menu bar, Dock/reopen behavior, resource lookup, `Info.plist`, open-file/open-url routing, sandbox/entitlements, and clean termination.

- [ ] **XCTest / XCUIAutomation harness:** For end-to-end UI flows, maintain an Xcode UI-test harness or equivalent accessibility automation that launches the bundled Rust app with test launch arguments/environment, uses stable accessibility identifiers, waits with expectations/predicates rather than sleeps, and captures screenshots/log attachments on failure. XCTest supports unit, asynchronous, performance, and UI tests through XCTest/XCUIAutomation.

- [ ] **Interaction matrix:** UI automation should cover standard flows: first launch, menus and keyboard shortcuts, toolbar validation, sheets/alerts, open/save panels, drag/drop, pasteboard, undo/redo, state restoration, localization/dark-mode/accessibility variants, and clean quit.

- [ ] **Performance regression tests:** Use `criterion` or Rust benchmarks for pure computation and XCTest performance metrics / Instruments / `xctrace` for app launch, scrolling, layout, drawing, file import/export, memory, hitches, CPU, and signposted workflows. Store baselines carefully and compare release-like builds on stable hardware.

- [ ] **Rust gates:** Run `cargo fmt --all -- --check`, `cargo check --workspace --all-targets` with the curated AppKit feature set, target-specific checks such as `--target aarch64-apple-darwin` / `x86_64-apple-darwin`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace --all-targets`. Add a separate `--all-features` job only if the crate's features are intentionally mutually compatible with the host SDK.

- [ ] **Unsafe and dependency gates:** Enable Rust's `unsafe_op_in_unsafe_fn` lint and explicitly enable Clippy's pedantic `undocumented_unsafe_blocks` lint, e.g. with `#![warn(clippy::undocumented_unsafe_blocks)]` or `clippy.toml`. Periodically review `unsafe` with `cargo geiger` or similar. Run `cargo audit` or `cargo deny check advisories`, plus `cargo deny check` for licenses, bans, and sources. Use `cargo udeps` as an optional nightly advisory check for unused dependencies.

- [ ] **Bundle and plist gates:** Validate app layout with the same packaging script used in dev and release. Run `plutil -lint Contents/Info.plist`, inspect with `/usr/libexec/PlistBuddy -c Print Contents/Info.plist`, and check required keys/resources/localizations explicitly.

- [ ] **Entitlement and signing gates:** Validate entitlements with `codesign -d --entitlements :- MyApp.app` and diff against expected CI output. Verify every nested binary/framework/helper and the final app with `codesign --verify --strict --verbose=4`; `--deep` may be useful as an additional verification sweep but should not replace per-binary verification.

- [ ] **Notarization and Gatekeeper gates:** Submit with `xcrun notarytool submit --wait`, keep `xcrun notarytool log` output, staple/validate with `xcrun stapler`, and assess the quarantined app or image with `spctl --assess --type execute --verbose`.

- [ ] **Architecture and dynamic-library gates:** Use `file` / `lipo -info` recursively over `Contents/MacOS`, `Contents/Frameworks`, and helpers. Use `otool -L` / `otool -l` to catch unexpected local build paths and incorrect `@rpath`, `@executable_path`, or `@loader_path` usage.

- [ ] **Runtime/UI gates:** Run Accessibility Inspector / VoiceOver smoke tests, scripted accessibility-tree checks, Instruments Main Thread Checker, Leaks, Allocations, Zombies, Address Sanitizer, Thread Sanitizer where compatible, and debug malloc options such as `NSZombieEnabled`, `MallocStackLogging`, or `MallocScribble` for targeted diagnostics rather than every CI run.

- [ ] **Release artifact smoke test:** On a clean machine or fresh user account, download/unzip/mount the exact artifact users receive, verify quarantine/Gatekeeper behavior, launch the app through Finder/Dock and `open`, open a sample file/URL, exercise menus/toolbars/sheets, and collect logs.
