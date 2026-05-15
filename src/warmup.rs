//! Startup warmup, ds4-style.
//!
//! Two passes:
//! 1. mmap the .onnx file and walk one byte per 16 KB page. The kernel populates
//!    the page cache before the first recognition needs it, so the first decode
//!    doesn't pay a cold-read tax (~hundreds of ms on a 1 GB model).
//! 2. Run one tiny silent decode through the recognizer. CoreML compiles its
//!    graph the first time it sees a shape; we eat that cost during startup so
//!    the first user press feels instant.

use std::path::Path;

use anyhow::{Context, Result};
use memmap2::Mmap;

use crate::asr::Asr;

/// Walk the mmap'd model file, touching one byte every `STRIDE` bytes.
pub fn page_touch(model_path: &Path) -> Result<u64> {
    let file = std::fs::File::open(model_path)
        .with_context(|| format!("opening {} for warmup", model_path.display()))?;
    let map = unsafe { Mmap::map(&file).context("mmap model file")? };

    // 16 KiB matches macOS' default page size on Apple Silicon.
    const STRIDE: usize = 16 * 1024;
    let bytes = &map[..];
    let mut acc: u64 = 0;
    let mut i = 0;
    while i < bytes.len() {
        // `std::hint::black_box` keeps LLVM from optimising the read away.
        acc = acc.wrapping_add(std::hint::black_box(bytes[i]) as u64);
        i += STRIDE;
    }
    Ok(acc)
}

/// Two-pass startup warmup against the recognizer:
/// 1. **Throwaway pass.** 0.5 s of silence pays the CoreML graph-compile cost
///    and primes CPU/ANE caches. Its timing is meaningless and we ignore it.
/// 2. **Measured pass.** 2 s of silence runs through the now-warm graph.
///    `recognize_with_timing` logs the resulting RTFx; that's the steady-state
///    number we want users (and ADR-0015 layer 3) to see in the log.
pub fn dummy_decode(asr: &Asr) -> Result<()> {
    // Pass 1: small sample, timing discarded.
    let throwaway = vec![0.0_f32; 16_000 / 2];
    let _ = asr.recognize_silent_warmup(&throwaway, 16_000)?;
    // Pass 2: longer sample, timing logged via `recognize`.
    let measured = vec![0.0_f32; 16_000 * 2];
    let _ = asr.recognize(&measured, 16_000)?;
    Ok(())
}
