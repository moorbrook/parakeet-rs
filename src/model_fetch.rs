//! First-run download of the Parakeet TDT 0.6B v3 int8 transducer model plus
//! the Silero VAD model from Hugging Face / sherpa-onnx releases. ~640 MB +
//! ~2 MB total.
//!
//! Progress is reported through a caller-supplied `Progress` callback so the
//! menu-bar UI can update the disabled "Model: …" header item without this
//! module knowing anything about AppKit.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

const HF_REPO: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main";
const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
/// Polish-pass GGUF (ADR-0018, amended to the 4B). MUST stay in sync
/// with `Settings::polish_model_path` — the filename at the end of
/// this URL is the filename the loader expects on disk.
pub(crate) const POLISH_GGUF_URL: &str =
    "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q6_K.gguf";

/// Parakeet TDT triplet + tokens, all relative to the per-model dir.
/// `pub(crate)` so `settings.rs`'s tests can cross-check that this list
/// stays in sync with the four files `SettingsStore::model_present()`
/// gates startup on.
pub(crate) const ASR_FILES: &[&str] = &[
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

/// First-toggle-on download of the polish GGUF (~3.5 GB) to the path
/// `Settings::polish_model_path` expects. No-op if already on disk.
/// Shares `download_to`'s .part-then-rename + length-validation flow,
/// so a killed download or short read redownloads cleanly next time
/// instead of leaving a corrupt GGUF the loader chokes on.
pub async fn ensure_polish_model(dest: &Path, on_progress: ProgressFn) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    on_progress(Progress::Status(
        "Downloading polish model (~3.5 GB, first enable only)…".to_string(),
    ));
    download_to("Qwen3.5-4B-Q6_K.gguf", POLISH_GGUF_URL, dest, &on_progress)
        .await
        .context("downloading polish GGUF")?;
    on_progress(Progress::Status("Polish model ready.".to_string()));
    Ok(())
}

async fn download_to(label: &str, url: &str, dest: &Path, on_progress: &ProgressFn) -> Result<()> {
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

    // Validate the .part length BEFORE renaming. A short/truncated
    // download would otherwise produce a final file that `ensure_model`
    // accepts on the next launch (it only checks `exists()`), bricking
    // startup with a corrupt model that no UI path retries. Failed
    // validation removes the .part so the next launch redownloads.
    if total > 0 {
        let part_size = tokio::fs::metadata(&tmp)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        if part_size != total {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(anyhow!(
                "{label} short download: got {part_size} of {total} bytes (discarded)"
            ));
        }
    }

    tokio::fs::rename(&tmp, dest)
        .await
        .context("rename .part to final")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Guard rails for the download set. We can't hit the network in a unit
    //! test, but we *can* pin the upstream URLs (so an accidental edit is
    //! visible in PR review) and cross-check that the four files we
    //! download are exactly the four files `Settings::model_present()`
    //! requires — drift between those lists means a successful first-run
    //! fetch can still leave the recogniser refusing to load. (That
    //! cross-check lives in `settings::tests` so it can reach the private
    //! `SettingsStore` fields.)
    use super::*;

    #[test]
    fn hf_repo_url_is_the_canonical_int8_upload() {
        // Changing this URL means every existing user re-downloads ~640 MB
        // on their next launch. Catch the change in review, not in prod.
        assert_eq!(
            HF_REPO,
            "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main"
        );
    }

    #[test]
    fn silero_url_is_the_k2_fsa_release_asset() {
        // sherpa-onnx-bundled Silero VAD model. Mirrored on the project's
        // own release page (not Hugging Face) so the version is pinned.
        assert_eq!(
            SILERO_VAD_URL,
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
        );
    }

    #[test]
    fn polish_gguf_url_is_the_unsloth_q6_k_upload() {
        // Changing this URL silently changes what every user's first
        // toggle-on downloads — 3.5 GB. Catch it in review.
        assert_eq!(
            POLISH_GGUF_URL,
            "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q6_K.gguf"
        );
    }

    // The URL-filename ↔ `polish_model_path` filename cross-check lives
    // in `settings::tests` (next to the equivalent ASR_FILES check) so
    // it can build a synthetic `SettingsStore` without a real `$HOME`.

    #[test]
    fn asr_files_list_has_no_duplicates() {
        let mut copy: Vec<&&str> = ASR_FILES.iter().collect();
        copy.sort();
        let len_before = copy.len();
        copy.dedup();
        assert_eq!(len_before, copy.len(), "duplicate entry in ASR_FILES");
    }
}
