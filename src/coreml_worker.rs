//! Resident native Core ML ASR worker backend.
//!
//! The Swift worker owns FluidAudio's Parakeet Unified models for the process
//! lifetime. Rust sends mono Float32 samples at 16 kHz over a framed pipe
//! protocol, avoiding per-utterance process startup, model loading, WAV
//! encoding, and temporary files.
//!
//! The 16 kHz part matters: FluidAudio's `AudioConverter` builds a fresh
//! `AVAudioConverter` for every call and cost 4.6 ms per audio-second of 48 kHz
//! input, a third of the worker's time at 5 s. Its `resample` returns the input
//! untouched when the rate already matches, so converting on this side with the
//! project's own resampler retires that stage instead of duplicating it. See
//! ADR-0030.

use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::asr::{Asr, AsrBackend, AsrBackendMetadata, Decoded, StageReport};
use crate::resample::{to_target_rate, TARGET_SAMPLE_RATE};

const PROTOCOL_MAGIC: [u8; 4] = *b"PRKT";
const PROTOCOL_VERSION: u32 = 1;
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 384_000;
const MAX_AUDIO_SECONDS: u64 = 30 * 60;
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
pub const DEFAULT_LONG_REGIME_SECONDS: u32 = 8;
pub const MAX_LONG_REGIME_SECONDS: u32 = 60;
/// Matches the worker's own `--tdt-chunk-concurrency` bound.
pub const MAX_TDT_CHUNK_CONCURRENCY: u32 = 16;
const WORKER_NAME: &str = "parakeet-coreml-worker";
pub const COREML_MODEL_FOLDER: &str = "parakeet-unified-en-0.6b";
pub const COREML_TDT_V3_MODEL_FOLDER: &str = "parakeet-tdt-0.6b-v3";

/// Which Parakeet graph set the worker loads.
///
/// `Unified` is the shipping default (ADR-0022). `TdtV3` is the multilingual
/// TDT 0.6B v3 conversion, present so it can be measured against Unified on
/// the same corpus and stage profiler (kata f0zg). It is not a production
/// path: Rust has no pinned download and integrity gate for it, so it can only
/// be pointed at a directory that is already on disk.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoreMlModelVariant {
    #[default]
    Unified,
    TdtV3,
}

impl CoreMlModelVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unified => "unified",
            Self::TdtV3 => "tdt-v3",
        }
    }

    /// Directory name FluidAudio derives from its repository enum. The worker
    /// re-appends it to the parent of the directory it is given, so a model
    /// directory under any other name fails to load.
    pub const fn folder_name(self) -> &'static str {
        match self {
            Self::Unified => COREML_MODEL_FOLDER,
            Self::TdtV3 => COREML_TDT_V3_MODEL_FOLDER,
        }
    }

    pub const fn model_description(self) -> &'static str {
        match self {
            Self::Unified => "Parakeet Unified EN 0.6B offline 15s",
            Self::TdtV3 => "Parakeet TDT 0.6B v3 multilingual offline 15s",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "unified" => Ok(Self::Unified),
            "tdt-v3" => Ok(Self::TdtV3),
            _ => bail!("unknown Core ML model variant {value:?}; expected unified or tdt-v3"),
        }
    }
}

/// Paths needed to start the native Core ML worker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoreMlWorkerConfig {
    pub worker_path: PathBuf,
    pub model_source: CoreMlModelSource,
    pub model_variant: CoreMlModelVariant,
    /// Long-form chunks the TDT path may decode at once. The worker defaults
    /// to 1; raising it measures FluidAudio's parallel arm, whose overlapping
    /// dispatch the stage profiler reports as un-partitionable.
    pub tdt_chunk_concurrency: u32,
    /// Where TDT's decoder and joint run. `None` keeps FluidAudio's own
    /// placement; `Some` pins them, which is how the K=64 top-K cost is
    /// separated from the placement difference against Unified. The encoder
    /// stays on the regime's units either way.
    pub tdt_decode_compute_units: Option<CoreMlComputeUnits>,
    pub short_compute_units: CoreMlComputeUnits,
    pub long_compute_units: CoreMlComputeUnits,
    pub long_regime_seconds: u32,
    /// Ask the worker to report a per-stage breakdown with every result. The
    /// bench turns this on; the dictation path leaves the worker's Core ML
    /// dispatch path untouched.
    pub emit_stage_timings: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreMlModelSource {
    ExistingDirectory(PathBuf),
    DownloadRoot(PathBuf),
}

/// Core ML placement request passed to the native worker.
///
/// These are candidates, not capability claims: a tuner may select one only
/// after the worker loads it and the measured result clears quality.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoreMlComputeUnits {
    All,
    CpuAndGpu,
    #[default]
    CpuAndNeuralEngine,
    CpuOnly,
}

impl CoreMlComputeUnits {
    pub const CANDIDATES: [Self; 4] = [
        Self::CpuAndNeuralEngine,
        Self::All,
        Self::CpuAndGpu,
        Self::CpuOnly,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::CpuAndGpu => "cpu-and-gpu",
            Self::CpuAndNeuralEngine => "cpu-and-neural-engine",
            Self::CpuOnly => "cpu-only",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "all" => Ok(Self::All),
            "cpu-and-gpu" => Ok(Self::CpuAndGpu),
            "cpu-and-neural-engine" => Ok(Self::CpuAndNeuralEngine),
            "cpu-only" => Ok(Self::CpuOnly),
            _ => bail!(
                "unknown Core ML compute units {value:?}; expected all, cpu-and-gpu, \
                 cpu-and-neural-engine, or cpu-only"
            ),
        }
    }
}

impl CoreMlWorkerConfig {
    pub fn new(worker_path: impl Into<PathBuf>, model_directory: impl Into<PathBuf>) -> Self {
        Self {
            worker_path: worker_path.into(),
            model_source: CoreMlModelSource::ExistingDirectory(model_directory.into()),
            model_variant: CoreMlModelVariant::Unified,
            tdt_chunk_concurrency: 1,
            tdt_decode_compute_units: None,
            short_compute_units: CoreMlComputeUnits::default(),
            long_compute_units: CoreMlComputeUnits::default(),
            long_regime_seconds: DEFAULT_LONG_REGIME_SECONDS,
            emit_stage_timings: false,
        }
    }

    pub fn download_to(worker_path: impl Into<PathBuf>, model_root: impl Into<PathBuf>) -> Self {
        Self {
            worker_path: worker_path.into(),
            model_source: CoreMlModelSource::DownloadRoot(model_root.into()),
            model_variant: CoreMlModelVariant::Unified,
            tdt_chunk_concurrency: 1,
            tdt_decode_compute_units: None,
            short_compute_units: CoreMlComputeUnits::default(),
            long_compute_units: CoreMlComputeUnits::default(),
            long_regime_seconds: DEFAULT_LONG_REGIME_SECONDS,
            emit_stage_timings: false,
        }
    }

    pub fn set_existing_model_directory(&mut self, model_directory: impl Into<PathBuf>) {
        self.model_source = CoreMlModelSource::ExistingDirectory(model_directory.into());
    }

    pub fn set_download_root(&mut self, model_root: impl Into<PathBuf>) {
        self.model_source = CoreMlModelSource::DownloadRoot(model_root.into());
    }

    /// Select the graph set. `TdtV3` is an evaluation path with no
    /// Rust-managed download, so it requires an existing model directory.
    pub fn set_model_variant(&mut self, variant: CoreMlModelVariant) -> Result<()> {
        if variant == CoreMlModelVariant::TdtV3
            && matches!(self.model_source, CoreMlModelSource::DownloadRoot(_))
        {
            bail!(
                "the tdt-v3 model variant has no integrity-gated download; point it at an \
                 existing model directory instead of a download root"
            );
        }
        self.model_variant = variant;
        Ok(())
    }

    /// Raise TDT's long-form chunk concurrency above the worker's serial
    /// default. Anything but 1 makes the per-stage columns overlap, which the
    /// bench then refuses, so this is for wall-clock arms only.
    pub fn set_tdt_chunk_concurrency(&mut self, chunks: u32) -> Result<()> {
        if !(1..=MAX_TDT_CHUNK_CONCURRENCY).contains(&chunks) {
            bail!("TDT chunk concurrency must be between 1 and {MAX_TDT_CHUNK_CONCURRENCY}");
        }
        self.tdt_chunk_concurrency = chunks;
        Ok(())
    }

    pub fn set_tdt_decode_compute_units(&mut self, units: Option<CoreMlComputeUnits>) {
        self.tdt_decode_compute_units = units;
    }

    pub fn set_emit_stage_timings(&mut self, emit: bool) {
        self.emit_stage_timings = emit;
    }

    pub fn set_compute_units(&mut self, compute_units: CoreMlComputeUnits) {
        self.short_compute_units = compute_units;
        self.long_compute_units = compute_units;
    }

    pub fn set_regime_compute_units(
        &mut self,
        short: CoreMlComputeUnits,
        long: CoreMlComputeUnits,
        long_regime_seconds: u32,
    ) -> Result<()> {
        if !(1..=MAX_LONG_REGIME_SECONDS).contains(&long_regime_seconds) {
            bail!(
                "long-regime threshold must be between 1 and \
                 {MAX_LONG_REGIME_SECONDS} seconds"
            );
        }
        self.short_compute_units = short;
        self.long_compute_units = long;
        self.long_regime_seconds = long_regime_seconds;
        Ok(())
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
        let model_variant = match std::env::var("PARAKEET_COREML_MODEL_VARIANT") {
            Ok(value) => CoreMlModelVariant::parse(&value)?,
            Err(std::env::VarError::NotPresent) => CoreMlModelVariant::default(),
            Err(error) => return Err(error).context("reading PARAKEET_COREML_MODEL_VARIANT"),
        };
        // The variant-specific override wins so both packs can be configured
        // at once: a shell that exports `PARAKEET_COREML_MODEL_DIR` for the
        // shipping pack would otherwise silently hand that directory to TDT,
        // which fails at load having named the wrong folder.
        let variant_directory = match model_variant {
            CoreMlModelVariant::Unified => None,
            CoreMlModelVariant::TdtV3 => std::env::var_os("PARAKEET_COREML_TDT_V3_MODEL_DIR"),
        };
        let model_directory = match variant_directory
            .or_else(|| std::env::var_os("PARAKEET_COREML_MODEL_DIR"))
        {
            Some(path) => PathBuf::from(path),
            None => dirs::data_dir()
                .ok_or_else(|| anyhow!("macOS application-support directory is unavailable"))?
                .join("FluidAudio")
                .join("Models")
                .join(model_variant.folder_name()),
        };
        let mut config = Self::new(worker_path, model_directory);
        config.set_model_variant(model_variant)?;
        Ok(config)
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
    /// Stage breakdown from the most recent result, when the worker reports one.
    last_stages: Mutex<Option<StageReport>>,
}

impl CoreMlWorkerBackend {
    fn spawn(config: &CoreMlWorkerConfig) -> Result<Self> {
        validate_file(&config.worker_path, "Core ML worker")?;
        let (model_flag, model_path) = match &config.model_source {
            CoreMlModelSource::ExistingDirectory(path) => {
                validate_directory(path, "Core ML model")?;
                ("--model-dir", path)
            }
            CoreMlModelSource::DownloadRoot(path) => {
                if config.model_variant != CoreMlModelVariant::Unified {
                    bail!(
                        "the {} model variant has no integrity-gated download and cannot be \
                         started from a download root",
                        config.model_variant.as_str()
                    );
                }
                ("--model-root", path)
            }
        };

        let mut command = Command::new(&config.worker_path);
        command
            .arg(model_flag)
            .arg(model_path)
            .arg("--short-compute-units")
            .arg(config.short_compute_units.as_str())
            .arg("--long-compute-units")
            .arg(config.long_compute_units.as_str())
            .arg("--long-regime-seconds")
            .arg(config.long_regime_seconds.to_string());
        // Only when it differs from the default, so the shipping worker's
        // command line is exactly what ADR-0022 measured.
        if config.model_variant != CoreMlModelVariant::default() {
            command
                .arg("--model-variant")
                .arg(config.model_variant.as_str());
        }
        if config.tdt_chunk_concurrency != 1 {
            command
                .arg("--tdt-chunk-concurrency")
                .arg(config.tdt_chunk_concurrency.to_string());
        }
        if let Some(units) = config.tdt_decode_compute_units {
            command
                .arg("--tdt-decode-compute-units")
                .arg(units.as_str());
        }
        if config.emit_stage_timings {
            command.arg("--emit-stage-timings");
        }
        let mut child = command
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
                model: config.model_variant.model_description().to_string(),
                quantization: "int8 encoder".to_string(),
                execution_provider: format!(
                    "Core ML short={} long={} threshold={}s",
                    config.short_compute_units.as_str(),
                    config.long_compute_units.as_str(),
                    config.long_regime_seconds
                ),
            },
            load_seconds,
            last_stages: Mutex::new(None),
        })
    }
}

impl AsrBackend for CoreMlWorkerBackend {
    fn metadata(&self) -> &AsrBackendMetadata {
        &self.metadata
    }

    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
        validate_request(samples.len(), sample_rate)?;
        // Production capture already delivers 16 kHz, so this borrows. File-fed
        // callers (the gold corpus is 48 kHz) convert here rather than leaving
        // it to the worker's per-call AVAudioConverter.
        let model_samples = to_target_rate(samples, sample_rate)
            .context("converting audio to the Core ML worker's 16 kHz input rate")?;
        let sample_count = validate_request(model_samples.len(), TARGET_SAMPLE_RATE)?;

        let mut process = self.process.lock();
        process
            .write_request(&model_samples, TARGET_SAMPLE_RATE, sample_count)
            .context("sending audio to Core ML worker")?;
        let response = process
            .read_response()
            .context("reading Core ML worker result")?;
        response.require_success("result")?;
        *self.last_stages.lock() = response.stages.clone();
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

    fn last_stage_report(&self) -> Option<StageReport> {
        self.last_stages.lock().clone()
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
    stages: Option<StageReport>,
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
            stages: None,
        };
        let error = response
            .require_success("result")
            .expect_err("failure response must not pass");
        assert!(error.to_string().contains("model rejected input"));
    }

    #[test]
    fn result_response_carries_the_worker_stage_breakdown() {
        // Field names are the worker's `convertToSnakeCase` encoding of
        // StageProfiler.Report, so a rename on either side must break here
        // rather than silently deserialize `stages` as absent.
        let payload = br#"{
            "kind": "result", "ok": true, "text": "hello",
            "decode_seconds": 0.044, "resample_seconds": 0.023,
            "stages": {
                "resample_ms": 22.976, "windows": 1, "encoder_calls": 1,
                "decoder_calls": 35, "joint_calls": 96, "other_calls": 0,
                "mel_ms": 3.08, "encoder_ms": 25.959, "decode_loop_ms": 15.84,
                "decode_loop_dispatch_ms": 15.135, "decoder_dispatch_ms": 5.359,
                "joint_dispatch_ms": 9.776, "post_ms": 0.066, "total_ms": 44.944,
                "compute_units": "encoder=cpu-and-neural-engine decoder=cpu-only joint=cpu-only"
            }
        }"#;
        let response: WorkerResponse = serde_json::from_slice(payload).expect("valid result");
        response.require_success("result").expect("ok result");
        let stages = response.stages.expect("stages present");
        assert_eq!(stages.windows, 1);
        assert_eq!(stages.joint_calls, 96);
        assert_eq!(stages.encoder_calls, 1);
        assert_eq!(stages.other_calls, 0);
        assert!(stages.compute_units.contains("decoder=cpu-only"));
        // This is a wire-format test over a captured payload, so it pins the
        // field names and types only. Whether a live pipeline still satisfies
        // the frame identity is checked at runtime by
        // `bench_asr::validate_stage_report`, which is where a moved Core ML
        // entry point gets caught.
    }

    #[test]
    fn tdt_result_response_carries_the_mel_front_end_stage() {
        // TDT's mel front end is a Core ML graph, so it appears as its own
        // stage. Pins the two field names the Unified payload above cannot.
        let payload = br#"{
            "kind": "result", "ok": true, "text": "hello",
            "decode_seconds": 0.044, "resample_seconds": 0.0,
            "stages": {
                "resample_ms": 0.0, "windows": 1, "preprocessor_calls": 1,
                "encoder_calls": 1, "decoder_calls": 12, "joint_calls": 40,
                "other_calls": 0, "mel_ms": 0.42, "preprocessor_ms": 6.1,
                "encoder_ms": 25.9, "decode_loop_ms": 9.2,
                "decode_loop_dispatch_ms": 8.8, "decoder_dispatch_ms": 3.1,
                "joint_dispatch_ms": 5.7, "post_ms": 0.05, "total_ms": 41.67,
                "compute_units": "preprocessor=cpu-only encoder=cpu-and-neural-engine decoder=cpu-only joint=cpu-only"
            }
        }"#;
        let response: WorkerResponse = serde_json::from_slice(payload).expect("valid result");
        let stages = response.stages.expect("stages present");
        assert_eq!(stages.preprocessor_calls, 1);
        assert!((stages.preprocessor_ms - 6.1).abs() < 1e-9);
    }

    #[test]
    fn model_variant_names_round_trip_and_keep_unified_as_the_default() {
        assert_eq!(CoreMlModelVariant::default(), CoreMlModelVariant::Unified);
        for candidate in [CoreMlModelVariant::Unified, CoreMlModelVariant::TdtV3] {
            assert_eq!(
                CoreMlModelVariant::parse(candidate.as_str()).unwrap(),
                candidate
            );
        }
        assert!(CoreMlModelVariant::parse("tdt").is_err());
        assert_eq!(
            CoreMlModelVariant::TdtV3.folder_name(),
            COREML_TDT_V3_MODEL_FOLDER
        );
    }

    #[test]
    fn tdt_chunk_concurrency_defaults_to_serial_and_is_bounded() {
        let mut config = CoreMlWorkerConfig::new("worker", "model");
        assert_eq!(config.tdt_chunk_concurrency, 1);
        config.set_tdt_chunk_concurrency(4).unwrap();
        assert_eq!(config.tdt_chunk_concurrency, 4);
        assert!(config.set_tdt_chunk_concurrency(0).is_err());
        assert!(config
            .set_tdt_chunk_concurrency(MAX_TDT_CHUNK_CONCURRENCY + 1)
            .is_err());
    }

    #[test]
    fn tdt_refuses_a_download_root_because_nothing_verifies_it() {
        let mut config = CoreMlWorkerConfig::download_to("worker", "root");
        let error = config
            .set_model_variant(CoreMlModelVariant::TdtV3)
            .expect_err("an unverified download root must be refused");
        assert!(error.to_string().contains("integrity-gated download"));
        assert_eq!(config.model_variant, CoreMlModelVariant::Unified);

        let mut config = CoreMlWorkerConfig::new("worker", "model");
        config
            .set_model_variant(CoreMlModelVariant::TdtV3)
            .expect("an existing directory is the supported source");
        assert_eq!(config.model_variant, CoreMlModelVariant::TdtV3);
    }

    #[test]
    fn absent_stage_breakdown_is_not_an_error() {
        let payload = br#"{"kind": "result", "ok": true, "text": "hi", "decode_seconds": 0.04}"#;
        let response: WorkerResponse = serde_json::from_slice(payload).expect("valid result");
        assert!(response.stages.is_none());
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

    #[test]
    fn compute_unit_names_round_trip_and_keep_the_safe_default() {
        assert_eq!(
            CoreMlComputeUnits::default(),
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
        for candidate in CoreMlComputeUnits::CANDIDATES {
            assert_eq!(
                CoreMlComputeUnits::parse(candidate.as_str()).unwrap(),
                candidate
            );
        }
        assert!(CoreMlComputeUnits::parse("ane").is_err());

        let mut config = CoreMlWorkerConfig::new("worker", "model");
        config
            .set_regime_compute_units(CoreMlComputeUnits::All, CoreMlComputeUnits::CpuOnly, 12)
            .unwrap();
        assert_eq!(config.short_compute_units, CoreMlComputeUnits::All);
        assert_eq!(config.long_compute_units, CoreMlComputeUnits::CpuOnly);
        assert_eq!(config.long_regime_seconds, 12);
        assert!(config
            .set_regime_compute_units(CoreMlComputeUnits::All, CoreMlComputeUnits::CpuOnly, 0,)
            .is_err());
        assert!(config
            .set_regime_compute_units(
                CoreMlComputeUnits::All,
                CoreMlComputeUnits::CpuOnly,
                MAX_LONG_REGIME_SECONDS + 1,
            )
            .is_err());
    }
}
