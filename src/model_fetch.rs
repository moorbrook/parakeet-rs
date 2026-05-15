//! First-run download of the Parakeet TDT 0.6B v3 int8 transducer model plus
//! the Silero VAD model from Hugging Face / sherpa-onnx releases. ~640 MB +
//! ~2 MB total.
//!
//! Progress is reported through a caller-supplied `Progress` callback so the
//! menu-bar UI can update the disabled "Model: …" header item without this
//! module knowing anything about AppKit.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

const HF_REPO: &str = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main";
const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

/// Parakeet TDT triplet + tokens, all relative to the per-model dir.
const ASR_FILES: &[&str] = &[
    "tokens.txt",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "encoder.int8.onnx",
];

#[derive(Clone, Debug)]
pub enum Progress {
    /// Status text — drives the menu-bar header label.
    Status(String),
    /// Streaming chunk update — fires at most ~5 Hz while bytes flow.
    Chunk {
        file: String,
        bytes: u64,
        total: u64,
        fraction: f32,
    },
}

pub type ProgressFn = Arc<dyn Fn(Progress) + Send + Sync + 'static>;

pub async fn ensure_model(
    model_dir: &Path,
    vad_path: &Path,
    on_progress: ProgressFn,
) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;
    if let Some(parent) = vad_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let asr_missing: Vec<&&str> = ASR_FILES
        .iter()
        .filter(|name| !model_dir.join(name).exists())
        .collect();
    let vad_missing = !vad_path.exists();

    if asr_missing.is_empty() && !vad_missing {
        return Ok(());
    }

    on_progress(Progress::Status(
        "Downloading Parakeet TDT v3 + Silero VAD (~640 MB, first run only)…".to_string(),
    ));

    for name in &asr_missing {
        let url = format!("{HF_REPO}/{name}");
        let dest = model_dir.join(name);
        download_to(name, &url, &dest, &on_progress)
            .await
            .with_context(|| format!("downloading {name}"))?;
    }

    if vad_missing {
        download_to("silero_vad.onnx", SILERO_VAD_URL, vad_path, &on_progress)
            .await
            .context("downloading silero_vad.onnx")?;
    }

    on_progress(Progress::Status("Model ready.".to_string()));
    Ok(())
}

async fn download_to(
    label: &str,
    url: &str,
    dest: &Path,
    on_progress: &ProgressFn,
) -> Result<()> {
    let tmp = dest.with_extension("part");
    let client = reqwest::Client::builder()
        .user_agent("parakeet-rs/0.1")
        .build()?;
    let resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length().unwrap_or(0);

    let mut file = tokio::fs::File::create(&tmp).await?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if last_emit.elapsed() >= std::time::Duration::from_millis(200) {
            last_emit = std::time::Instant::now();
            let fraction = if total > 0 {
                downloaded as f32 / total as f32
            } else {
                0.0
            };
            on_progress(Progress::Chunk {
                file: label.to_string(),
                bytes: downloaded,
                total,
                fraction,
            });
        }
    }
    file.flush().await?;
    drop(file);

    tokio::fs::rename(&tmp, dest)
        .await
        .context("rename .part to final")?;

    let final_size = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    if total > 0 && final_size != total {
        return Err(anyhow!(
            "{label} short download: got {final_size} of {total} bytes"
        ));
    }
    Ok(())
}
