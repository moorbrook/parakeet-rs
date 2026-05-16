//! Tiny helpers for the AppKit ↔ Rust boundary.
//!
//! Every `#[unsafe(method)]` ObjC selector body is wrapped in
//! [`selector_guard`] so a Rust panic can't unwind across ObjC frames
//! (Undefined Behaviour). In a debug build (`panic = "unwind"`) the
//! guard catches the panic, logs it, and returns control to AppKit. In
//! a release build (`panic = "abort"`) `catch_unwind` is a no-op and
//! the process still aborts — but the panic hook installed by `main`
//! logs the panic info first so the abort isn't silent.

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
/// # **No-op in release builds**
///
/// `panic = "abort"` in `[profile.release]` (`Cargo.toml`) means
/// `catch_unwind` does NOT catch panics — `body` aborts the process
/// instead. The function still runs `body`, but the wrapper does
/// nothing meaningful at the panic boundary. `install_panic_hook` in
/// `main()` logs the panic message via `log::error!` *before* the
/// abort handler runs; that log line is the only release-mode
/// feedback path. In debug builds (`panic = "unwind"`) this guard
/// works as expected.
pub fn selector_guard<F: FnOnce()>(name: &'static str, body: F) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(body)) {
        let msg = payload_message(&payload);
        log::error!("panic in ObjC selector {name}: {msg}");
    }
}

fn payload_message(payload: &Box<dyn std::any::Any + Send>) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "<non-string panic payload>"
    }
}

/// Install a process-wide panic hook that logs panic location + message
/// before the default handler runs. With `panic = "abort"` in release,
/// this is the only way the user (and crash reporters) see *why* the
/// app vanished — the default abort prints nothing to a Finder-launched
/// LSUIElement app.
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
