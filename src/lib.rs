//! `parakeet-rs` library crate. Re-exports modules so the binaries under
//! `src/bin/` can link them.
//!
//! Lint policy is declared in `Cargo.toml [lints.rust]` and
//! `[lints.clippy]`: `unsafe_op_in_unsafe_fn = warn` and
//! `undocumented_unsafe_blocks = warn`.

// Public modules — reachable from `src/main.rs` (the bundled binary
// target) and the benches under `src/bin/`. Both link this crate by
// name, so they need `pub` for the items they use.
pub mod app;
pub mod app_delegate;
pub mod asr;
pub mod asr_eval;
pub mod asr_tuning;
pub mod coreml_worker;
pub mod dictation_fsm;
pub mod endpointing;
pub mod llm_manager;
pub mod objc_util;
pub mod performance;
pub mod permissions;
pub mod polish;
pub mod settings;
pub mod vocabulary;
pub mod warmup;
pub mod wav;

// Internal modules — only referenced from inside the lib (the public
// modules above use them via `crate::*`). Keeping them private tightens
// the API surface and makes it obvious at a glance which modules are
// in the "stable" set vs. the implementation-detail set. If a new bin
// or main.rs callsite needs one of these, promote it to `pub mod`.
mod audio;
mod ax_paste;
mod clipboard;
mod hotkey;
mod hud;
mod menubar;
pub mod model_fetch;
mod paste;
mod qos;
mod settings_ui;
mod sf_symbol;
pub mod streamer;
pub mod vad;
