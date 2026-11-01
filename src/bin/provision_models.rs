//! Run the app's Core ML model provisioning once, without the app: verify or
//! download the stock Parakeet Unified pack and the 2/5/8 s encoder buckets
//! into the data directory the app itself would use, then print which bucket
//! windows verified.
//!
//! The data root is `SettingsStore::new()`'s, which follows `$HOME`, so a
//! fresh install is `HOME=$(mktemp -d) provision_models`. The printed model
//! directory is what `bench_asr --model-dir` takes to show the bucketed
//! encoder cost.
//!
//! `--verify-only` checks the encoder buckets already on disk and touches
//! nothing else: no network, and the stock pack is not checked, because its
//! check repairs by downloading.

use std::sync::Arc;

use anyhow::{Context, Result};
use parakeet_dictation::model_fetch::{self, BucketFetch, Progress, ProgressFn};
use parakeet_dictation::settings::SettingsStore;

fn main() -> Result<()> {
    let verify_only = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--verify-only") => true,
        Some(other) => anyhow::bail!("unknown argument {other:?}; expected --verify-only"),
    };
    let settings = SettingsStore::new()?;
    let model_dir = settings.coreml_model_dir();
    let progress: ProgressFn = Arc::new(|progress| {
        if let Progress::Status(status) = progress {
            eprintln!("{status}");
        }
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building download runtime")?;
    let started = std::time::Instant::now();
    let fetch = if verify_only {
        BucketFetch::VerifyOnly
    } else {
        runtime.block_on(model_fetch::ensure_coreml_model(
            &model_dir,
            progress.clone(),
        ))?;
        BucketFetch::Download
    };
    let windows = runtime.block_on(model_fetch::ensure_encoder_buckets(
        &model_dir,
        &model_fetch::all_encoder_bucket_windows(),
        fetch,
        progress,
    ));
    println!("model_dir={}", model_dir.display());
    println!("encoder_buckets={windows:?}");
    println!("seconds={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
