//! `parakeet-rs` library crate. Re-exports modules so the binaries under
//! `src/bin/` can link them.
//!
//! Lint policy is declared in `Cargo.toml [lints.rust]` and
//! `[lints.clippy]`: `unsafe_op_in_unsafe_fn = warn` and
//! `undocumented_unsafe_blocks = warn`.
//!
//! ## Technical debt: unsafe-block audit pending
//!
//! Removing the previous `#![allow(unused_unsafe)]` (per AppKit
//! checklist item #2) surfaces ~54 `unsafe { }` wrappers that objc2 0.6
//! relaxed to safe, plus ~80 `// SAFETY:`-less FFI sites. Two crate-
//! wide allows below are **TEMPORARY** so the build stays green while
//! the audit lands incrementally. Cleanup pattern per site:
//!
//! - Wrapper around a now-safe method → drop the `unsafe { }`.
//! - Genuinely unsafe op (raw pointer, FFI invariant) → add
//!   `// SAFETY: <why>` immediately above the block.
//!
//! Remove both allows below as the per-site audit completes.
#![allow(unused_unsafe)]
#![allow(clippy::undocumented_unsafe_blocks)]

// Public modules — reachable from `src/main.rs` (the bundled binary
// target) and the benches under `src/bin/`. Both link this crate by
// name, so they need `pub` for the items they use.
pub mod app;
pub mod app_delegate;
pub mod asr;
pub mod dictation_fsm;
pub mod llm_manager;
pub mod objc_util;
pub mod performance;
pub mod permissions;
pub mod polish;
pub mod settings;
pub mod warmup;

// Internal modules — only referenced from inside the lib (the public
// modules above use them via `crate::*`). Keeping them private tightens
// the API surface and makes it obvious at a glance which modules are
// in the "stable" set vs. the implementation-detail set. If a new bin
// or main.rs callsite needs one of these, promote it to `pub mod`.
mod audio;
mod ax_paste;
mod hotkey;
mod hud;
mod menubar;
mod model_fetch;
mod paste;
mod qos;
mod settings_ui;
mod sf_symbol;
mod streamer;
mod vad;
