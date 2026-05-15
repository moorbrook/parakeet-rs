//! Build script. Inspects the sherpa-onnx prebuilt library for the CoreML
//! execution-provider symbol (ADR-0015 layer 1).
//!
//! We use the SHARED variant of sherpa-onnx (per the `shared` feature on
//! `sherpa-onnx-sys`), which links against Microsoft's official
//! `libonnxruntime.dylib`. That dylib DOES export
//! `_OrtSessionOptionsAppendExecutionProvider_CoreML`. The static variant
//! does not (sherpa-onnx's static-onnxruntime cmake hardcodes
//! `-DSHERPA_ONNX_DISABLE_COREML`). See ADR-0012 for why we picked shared
//! over a from-source build.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell rustc this cfg is one we set ourselves; silences --check-cfg lint.
    println!("cargo:rustc-check-cfg=cfg(parakeet_coreml_ep_present)");

    // `sherpa-onnx-sys` emits `cargo:rustc-link-arg=-Wl,-rpath,@loader_path`
    // from its own build script — but Cargo does NOT propagate
    // `rustc-link-arg` directives from transitive build scripts into the
    // final binary's link command. Re-emit them here, from the binary's
    // own build script, so the resulting executable can actually find
    // `libonnxruntime.dylib` next to itself at runtime.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        // Also add the sherpa-onnx-prebuilt cache dir as an rpath so a
        // bare `cargo run` works without first copying dylibs alongside
        // the binary. Used as a fallback only — production .app bundles
        // ship the dylibs in Contents/Frameworks.
        if let Some(dir) = locate_libonnxruntime().and_then(|p| p.parent().map(|p| p.to_path_buf()))
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }

    let _ = check_coreml_ep();
}

fn check_coreml_ep() -> Result<(), String> {
    println!("cargo:rerun-if-env-changed=PARAKEET_REQUIRE_COREML");

    let lib = match locate_libonnxruntime() {
        Some(p) => p,
        None => {
            warn("libonnxruntime.a not in sherpa-onnx prebuilt cache yet; skipping CoreML EP check (will run after the next cargo build)");
            return Ok(());
        }
    };
    println!("cargo:rerun-if-changed={}", lib.display());

    let output = Command::new("nm")
        .arg("-gU")
        .arg(&lib)
        .output()
        .map_err(|e| format!("running nm: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Microsoft's onnxruntime dylib exports the CoreML EP via the C API
    // function `OrtSessionOptionsAppendExecutionProvider_CoreML`. The static
    // library exports a C++-mangled `CoreMLExecutionProvider` instead — both
    // count, so we accept either.
    let has_coreml = stdout.lines().any(|l| {
        l.contains("OrtSessionOptionsAppendExecutionProvider_CoreML")
            || l.contains("CoreMLExecutionProvider")
            || l.contains("CoreMLProviderFactory")
    });

    if has_coreml {
        println!("cargo:rustc-cfg=parakeet_coreml_ep_present");
        return Ok(());
    }

    let msg = format!(
        "CoreML EP symbol NOT found in {}. The linked libonnxruntime has \
         no CoreML support; provider=\"coreml\" in OfflineRecognizerConfig \
         will silently fall back to CPU. See docs/ADR.md ADR-0012 / ADR-0015.",
        lib.display()
    );
    if env::var("PARAKEET_REQUIRE_COREML").is_ok() {
        panic!("{msg}");
    }
    warn(&msg);
    Ok(())
}

fn warn(msg: &str) {
    println!("cargo:warning={msg}");
}

/// sherpa-onnx-sys caches the prebuilt under
/// `<target_dir>/sherpa-onnx-prebuilt/<archive>/lib/`. Walk up from
/// OUT_DIR to find the target dir, then look for the onnxruntime dylib
/// (preferred) or its static counterpart (legacy / fallback).
fn locate_libonnxruntime() -> Option<PathBuf> {
    let out_dir = env::var("OUT_DIR").ok()?;
    let target_dir = PathBuf::from(out_dir)
        .ancestors()
        .find(|p| p.file_name().and_then(|s| s.to_str()) == Some("target"))?
        .to_path_buf();

    let prebuilt_root = target_dir.join("sherpa-onnx-prebuilt");
    if !prebuilt_root.exists() {
        return None;
    }
    // Names to look for, in order of preference. The versioned dylib has
    // the actual symbols; the unversioned one is a symlink.
    let candidates = [
        "libonnxruntime.1.24.4.dylib",
        "libonnxruntime.dylib",
        "libonnxruntime.a",
    ];
    for entry in std::fs::read_dir(&prebuilt_root).ok()?.flatten() {
        for name in candidates {
            let candidate = entry.path().join("lib").join(name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}
