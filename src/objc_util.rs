//! Tiny helpers for the AppKit ↔ Rust boundary.
//!
//! Every `#[unsafe(method)]` ObjC selector body is wrapped in
//! [`selector_guard`] so a Rust panic can't unwind across ObjC frames
//! (Undefined Behaviour). With `panic = "unwind"` in `[profile.release]`
//! (required by the polish-pipeline panic-isolation acceptance
//! criterion in `docs/latency-plan.md` §7), the guard catches the panic
//! in BOTH debug and release builds — it logs the panic via `log::error!`
//! and returns control to AppKit so the menu-bar app keeps running.
//! `install_panic_hook` adds a backup logging path that fires even when
//! a panic site is outside any selector wrapper.

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Run `body` with a panic boundary. Intended to wrap the entire body
/// of every ObjC selector. The `name` is the selector name, used in
/// the log line so a panicking handler is identifiable in retrospect.
///
/// `AssertUnwindSafe` is required because most selectors capture
/// `&self` (an objc2 `Retained<…>`) and AppKit-owned references that
/// aren't auto-marked `UnwindSafe`. We accept this in exchange for not
/// having to plaster every selector with bespoke wrappers.
///
/// Works in both debug and release builds — the project's release
/// profile uses `panic = "unwind"` specifically so this and
/// `run_polish_isolated` in `app.rs` can actually catch panics.
pub fn selector_guard<F: FnOnce()>(name: &'static str, body: F) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(body)) {
        let msg = payload_message(&*payload);
        log::error!("panic in ObjC selector {name}: {msg}");
    }
}

fn payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "<non-string panic payload>"
    }
}

/// Install a process-wide panic hook that logs panic location + message
/// before the default handler runs. Useful as a backup for panic sites
/// outside any `selector_guard` wrapper (e.g. inside spawned worker
/// threads), where there's no other path to surface the failure in the
/// menu-bar app's log stream.
pub fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Forward to env_logger first so the panic ends up in the same
        // stream as everything else (and ends up in any log file the
        // user has configured).
        log::error!("PANIC: {info}");
        // Then chain to the default formatter so `cargo test` output,
        // `RUST_BACKTRACE`, etc. still work as expected.
        default(info);
    }));
}

/// Run `f` on the main thread — immediately when already there,
/// otherwise enqueued onto the main dispatch queue. Shared by the HUD
/// and menubar paths (both mutate AppKit state that is only legal to
/// touch on main).
pub fn dispatch_to_main<F: FnOnce() + Send + 'static>(f: F) {
    if objc2_foundation::MainThreadMarker::new().is_some() {
        f();
    } else {
        dispatch2::DispatchQueue::main().exec_async(f);
    }
}
