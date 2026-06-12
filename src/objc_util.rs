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

#[cfg(test)]
mod tests {
    //! Lint-as-test: architectural invariants the compiler can't see.

    /// Every `#[unsafe(method...)]` ObjC selector body must route
    /// through [`super::selector_guard`]. A Rust panic unwinding across
    /// ObjC frames is Undefined Behaviour, and this is the kind of rule
    /// that erodes silently — a new selector added without the guard
    /// compiles and works fine until the first panic. Walk the source
    /// and fail the build listing any offender.
    #[test]
    fn every_objc_selector_is_panic_guarded() {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders = Vec::new();
        // Needle assembled at runtime so this test's own source (and
        // doc comments quoting the attribute) can't match themselves.
        let needle = ["#[unsafe(", "method"].concat();
        for entry in walk_rs_files(&src_dir) {
            // Strip comment lines: doc comments legitimately mention
            // the attribute when documenting this very invariant.
            let text: String = std::fs::read_to_string(&entry)
                .unwrap()
                .lines()
                .map(|l| {
                    if l.trim_start().starts_with("//") {
                        ""
                    } else {
                        l
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let mut search_from = 0;
            while let Some(rel) = text[search_from..].find(&needle) {
                let attr_at = search_from + rel;
                let fn_at = text[attr_at..]
                    .find("fn ")
                    .map(|i| attr_at + i)
                    .expect("selector attribute not followed by a fn");
                let body = fn_body(&text, fn_at);
                if !body.contains("selector_guard") {
                    let line = text[..attr_at].matches('\n').count() + 1;
                    offenders.push(format!("{}:{line}", entry.display()));
                }
                search_from = fn_at + 3;
            }
        }
        assert!(
            offenders.is_empty(),
            "ObjC selectors without a selector_guard panic boundary \
             (panic across ObjC frames is UB):\n  {}",
            offenders.join("\n  ")
        );
    }

    /// Return the brace-matched body of the fn whose `fn` keyword
    /// starts at `fn_at`.
    fn fn_body(text: &str, fn_at: usize) -> &str {
        let open = text[fn_at..].find('{').map(|i| fn_at + i).unwrap();
        let mut depth = 0usize;
        for (i, c) in text[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &text[open..=open + i];
                    }
                }
                _ => {}
            }
        }
        &text[open..]
    }

    fn walk_rs_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                out.extend(walk_rs_files(&path));
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
        out
    }
}
