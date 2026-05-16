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

pub mod app;
pub mod app_delegate;
pub mod asr;
pub mod audio;
pub mod cleanup;
pub mod hotkey;
pub mod hud;
pub mod llm_warmup;
pub mod menubar;
pub mod model_fetch;
pub mod objc_util;
pub mod paste;
pub mod performance;
pub mod permissions;
pub mod qos;
pub mod settings;
pub mod settings_ui;
pub mod sf_symbol;
pub mod streamer;
pub mod vad;
pub mod warmup;
