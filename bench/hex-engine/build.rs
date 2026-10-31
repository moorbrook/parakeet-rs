//! Cargo does not forward `rustc-link-arg` from a dependency's build script,
//! so the app library's sherpa-onnx rpath does not reach this binary. Re-emit
//! the same absolute rpath the app's build.rs uses: sherpa-onnx-sys copies its
//! dylibs into `<build-dir>/<profile>`, and OUT_DIR is
//! `<build-dir>/<profile>/build/<pkg>/out`.

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    if let Some(profile_dir) = env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .and_then(|out| out.ancestors().nth(3).map(std::path::Path::to_path_buf))
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", profile_dir.display());
    }
}
