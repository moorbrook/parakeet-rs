//! Resident native Core ML ASR worker backend.
//!
//! The Swift worker owns FluidAudio's Parakeet Unified models for the process
//! lifetime. Rust sends raw mono Float32 samples over a framed pipe protocol,
//! avoiding per-utterance process startup, model loading, WAV encoding, and
//! temporary files.

use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use parking_lot::Mutex;
use serde::Deserialize;

use crate::asr::{Asr, AsrBackend, AsrBackendMetadata, Decoded};

const PROTOCOL_MAGIC: [u8; 4] = *b"PRKT";
const PROTOCOL_VERSION: u32 = 1;
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 384_000;
const MAX_AUDIO_SECONDS: u64 = 30 * 60;
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const WORKER_NAME: &str = "parakeet-coreml-worker";
pub const COREML_MODEL_FOLDER: &str = "parakeet-unified-en-0.6b";

/// Paths needed to start the native Core ML worker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoreMlWorkerConfig {
    pub worker_path: PathBuf,
    pub model_source: CoreMlModelSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreMlModelSource {
    ExistingDirectory(PathBuf),
    DownloadRoot(PathBuf),
}

impl CoreMlWorkerConfig {
    pub fn new(worker_path: impl Into<PathBuf>, model_directory: impl Into<PathBuf>) -> Self {
        Self {
            worker_path: worker_path.into(),
            model_source: CoreMlModelSource::ExistingDirectory(model_directory.into()),
        }
    }

    pub fn download_to(worker_path: impl Into<PathBuf>, model_root: impl Into<PathBuf>) -> Self {
        Self {
            worker_path: worker_path.into(),
            model_source: CoreMlModelSource::DownloadRoot(model_root.into()),
        }
    }

    pub fn set_existing_model_directory(&mut self, model_directory: impl Into<PathBuf>) {
        self.model_source = CoreMlModelSource::ExistingDirectory(model_directory.into());
    }

    pub fn set_download_root(&mut self, model_root: impl Into<PathBuf>) {
        self.model_source = CoreMlModelSource::DownloadRoot(model_root.into());
    }

    /// Discover the bundled worker and the standard FluidAudio model cache.
    ///
    /// Environment overrides keep benchmarks and development builds explicit;
    /// a bundled app finds the worker beside its own executable.
    pub fn discover() -> Result<Self> {
        let worker_path = match std::env::var_os("PARAKEET_COREML_WORKER") {
            Some(path) => PathBuf::from(path),
            None => discover_worker_path()?,
        };
        let model_directory = match std::env::var_os("PARAKEET_COREML_MODEL_DIR") {
            Some(path) => PathBuf::from(path),
            None => dirs::data_dir()
                .ok_or_else(|| anyhow!("macOS application-support directory is unavailable"))?
                .join("FluidAudio")
                .join("Models")
                .join(COREML_MODEL_FOLDER),
        };
        Ok(Self::new(worker_path, model_directory))
    }
}

fn discover_worker_path() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("locating the current executable")?;
    let sibling = executable
        .parent()
        .ok_or_else(|| anyhow!("current executable has no parent directory"))?
        .join(WORKER_NAME);
    if sibling.is_file() {
        return Ok(sibling);
    }

    let development = PathBuf::from("target").join("release").join(WORKER_NAME);
    if development.is_file() {
        return Ok(development);
    }
    Ok(sibling)
}

/// Start a worker and return it behind the stable app-facing ASR facade.
pub fn load_coreml_worker(config: &CoreMlWorkerConfig) -> Result<(Asr, f64)> {
    let backend = CoreMlWorkerBackend::spawn(config)?;
    let load_seconds = backend.load_seconds;
    Ok((Asr::from_backend(Arc::new(backend)), load_seconds))
}

struct CoreMlWorkerBackend {
    process: Mutex<WorkerProcess>,
    metadata: AsrBackendMetadata,
    load_seconds: f64,
}

impl CoreMlWorkerBackend {
    fn spawn(config: &CoreMlWorkerConfig) -> Result<Self> {
        validate_file(&config.worker_path, "Core ML worker")?;
        let (model_flag, model_path) = match &config.model_source {
            CoreMlModelSource::ExistingDirectory(path) => {
                validate_directory(path, "Core ML model")?;
                ("--model-dir", path)
            }
            CoreMlModelSource::DownloadRoot(path) => ("--model-root", path),
        };

        let mut child = Command::new(&config.worker_path)
            .arg(model_flag)
            .arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("starting {}", config.worker_path.display()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Core ML worker stdin was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Core ML worker stdout was not piped"))?;
        let mut process = WorkerProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = process
            .read_response()
            .context("waiting for Core ML worker readiness")?;
        ready.require_success("ready")?;
        let load_seconds = ready
            .load_seconds
            .ok_or_else(|| anyhow!("Core ML ready response omitted load_seconds"))?;

        Ok(Self {
            process: Mutex::new(process),
            metadata: AsrBackendMetadata {
                backend: "fluid-audio-worker".to_string(),
                model: "Parakeet Unified EN 0.6B offline 15s".to_string(),
                quantization: "int8 encoder".to_string(),
                execution_provider: "Core ML CPU+ANE".to_string(),
            },
            load_seconds,
        })
    }
}

impl AsrBackend for CoreMlWorkerBackend {
    fn metadata(&self) -> &AsrBackendMetadata {
        &self.metadata
    }

    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
        let sample_count = validate_request(samples.len(), sample_rate)?;

        let mut process = self.process.lock();
        process
            .write_request(samples, sample_rate, sample_count)
            .context("sending audio to Core ML worker")?;
        let response = process
            .read_response()
            .context("reading Core ML worker result")?;
        response.require_success("result")?;
        let decode_seconds = response
            .decode_seconds
            .ok_or_else(|| anyhow!("Core ML result omitted decode_seconds"))?
            + response.resample_seconds.unwrap_or(0.0);

        Ok(Decoded {
            text: response.text.unwrap_or_default(),
            audio_seconds: samples.len() as f32 / sample_rate as f32,
            decode_seconds: decode_seconds as f32,
        })
    }

    fn auxiliary_resident_bytes(&self) -> Result<u64> {
        let pid = self.process.lock().child.id();
        crate::performance::resident_bytes(pid)
            .with_context(|| format!("reading Core ML worker {pid} resident set"))
    }
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl WorkerProcess {
    fn write_request(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        sample_count: u32,
    ) -> Result<()> {
        let header = encode_header(sample_rate, sample_count);
        self.stdin.write_all(&header)?;

        // The worker protocol is explicitly little-endian. Encoding without a
        // raw-slice cast keeps this safe and portable; IPC cost is measured by
        // the outer benchmark and can be optimized only if it is material.
        let mut payload = Vec::with_capacity(std::mem::size_of_val(samples));
        for sample in samples {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        self.stdin.write_all(&payload)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_response(&mut self) -> Result<WorkerResponse> {
        let mut length = [0_u8; 4];
        self.stdout.read_exact(&mut length)?;
        let length = u32::from_le_bytes(length) as usize;
        if length > MAX_RESPONSE_BYTES {
            bail!("Core ML worker response is too large: {length} bytes");
        }
        let mut payload = vec![0_u8; length];
        self.stdout.read_exact(&mut payload)?;
        serde_json::from_slice(&payload).context("decoding Core ML worker response")
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    kind: String,
    ok: bool,
    text: Option<String>,
    error: Option<String>,
    load_seconds: Option<f64>,
    decode_seconds: Option<f64>,
    resample_seconds: Option<f64>,
}

impl WorkerResponse {
    fn require_success(&self, expected_kind: &str) -> Result<()> {
        if !self.ok {
            bail!(
                "Core ML worker {expected_kind} failed: {}",
                self.error.as_deref().unwrap_or("unknown worker error")
            );
        }
        if self.kind != expected_kind {
            bail!(
                "Core ML worker returned {:?}, expected {expected_kind:?}",
                self.kind
            );
        }
        Ok(())
    }
}

fn validate_file(path: &Path, label: &str) -> Result<()> {
    if !path.is_file() {
        bail!("{label} not found at {}", path.display());
    }
    Ok(())
}

fn validate_directory(path: &Path, label: &str) -> Result<()> {
    if !path.is_dir() {
        bail!("{label} directory not found at {}", path.display());
    }
    Ok(())
}

fn encode_header(sample_rate: u32, sample_count: u32) -> [u8; 16] {
    let mut header = [0_u8; 16];
    header[..4].copy_from_slice(&PROTOCOL_MAGIC);
    header[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&sample_rate.to_le_bytes());
    header[12..16].copy_from_slice(&sample_count.to_le_bytes());
    header
}

fn validate_request(sample_count: usize, sample_rate: u32) -> Result<u32> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        bail!(
            "sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz; got {sample_rate}"
        );
    }
    let sample_count = u32::try_from(sample_count).context("audio sample count exceeds u32")?;
    if sample_count == 0 {
        bail!("Core ML worker received empty audio");
    }
    let maximum_sample_count = u64::from(sample_rate) * MAX_AUDIO_SECONDS;
    if u64::from(sample_count) > maximum_sample_count {
        bail!("audio exceeds the {MAX_AUDIO_SECONDS}-second Core ML worker limit");
    }
    Ok(sample_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_header_is_versioned_little_endian() {
        let header = encode_header(48_000, 240_000);
        assert_eq!(&header[..4], b"PRKT");
        assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(header[8..12].try_into().unwrap()),
            48_000
        );
        assert_eq!(
            u32::from_le_bytes(header[12..16].try_into().unwrap()),
            240_000
        );
    }

    #[test]
    fn failed_response_preserves_worker_error() {
        let response = WorkerResponse {
            kind: "result".to_string(),
            ok: false,
            text: None,
            error: Some("model rejected input".to_string()),
            load_seconds: None,
            decode_seconds: None,
            resample_seconds: None,
        };
        let error = response
            .require_success("result")
            .expect_err("failure response must not pass");
        assert!(error.to_string().contains("model rejected input"));
    }

    #[test]
    fn request_validation_matches_worker_sample_rate_range() {
        assert!(validate_request(16_000, MIN_SAMPLE_RATE).is_ok());
        assert!(validate_request(16_000, MAX_SAMPLE_RATE).is_ok());
        assert!(validate_request(16_000, MIN_SAMPLE_RATE - 1).is_err());
        assert!(validate_request(16_000, MAX_SAMPLE_RATE + 1).is_err());
    }

    #[test]
    fn request_duration_limit_uses_the_input_sample_rate() {
        let maximum_48khz_samples = 48_000_usize * MAX_AUDIO_SECONDS as usize;
        assert!(validate_request(maximum_48khz_samples, 48_000).is_ok());
        assert!(validate_request(maximum_48khz_samples + 1, 48_000).is_err());
    }
}
