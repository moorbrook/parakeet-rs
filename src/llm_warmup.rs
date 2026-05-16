//! Boot-time warmup for the cleanup LLM.
//!
//! Mirrors `src/warmup.rs` for the ASR side. Loads the GGUF, primes the
//! Metal kernel cache via a dummy 1-token decode, returns the wrapped
//! `LlamaCleanup` for `App::llm` to stash.
//!
//! Two distinct phases on the latency contract:
//!
//! 1. **`load`** — `LlamaCleanup::load` opens the GGUF (~250 ms cold)
//!    and asks llama.cpp to allocate Metal buffers. Done once per
//!    process; happens off the main thread in `App::spawn_model_setup`.
//! 2. **`dummy_polish`** — runs one tiny polish (input = "hi") through
//!    the full pipeline. CoreML / Metal kernel JIT happens here, not
//!    on the user's first real dictation. Wall-clock typically 100-200
//!    ms; throwaway-discarded.

use std::sync::Arc;

use anyhow::Result;

use crate::cleanup::{polish, LlamaCleanup};
use crate::settings::{CleanupMode, Settings};

/// Throwaway polish to JIT the Metal kernels. The output is discarded;
/// only the side effect (kernels compiled, buffers warm) is wanted.
/// One iteration is enough — the kernel cache persists for the life
/// of the `LlamaCleanup` instance.
pub fn dummy_polish(llm: &Arc<LlamaCleanup>) -> Result<()> {
    let warmup_settings = Settings {
        cleanup_mode: CleanupMode::On,
        ..Settings::default()
    };
    let _ = polish(llm, "hi", &warmup_settings)?;
    Ok(())
}
