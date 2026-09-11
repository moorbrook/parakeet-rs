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

use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::asr::{
    Asr, AsrBackend, AsrBackendMetadata, Decoded, EncodedTerm, StageReport, VocabularyStatus,
};
use crate::resample::{to_target_rate, TARGET_SAMPLE_RATE};
use crate::windows::TokenSpan;

const PROTOCOL_MAGIC: [u8; 4] = *b"PRKT";
/// Control frame carrying the custom vocabulary. A distinct magic rather than a
/// protocol version bump: the audio frame is unchanged, and a worker that
/// predates this frame refuses it by magic instead of reading its length as a
/// sample rate.
const VOCABULARY_MAGIC: [u8; 4] = *b"PRKV";
/// Structural cap on the vocabulary payload, matching the worker's.
const MAX_VOCABULARY_BYTES: usize = 1 << 20;
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
///
/// Not `Eq`: `vocabulary_score` is an `f32`, and the app compares configured
/// biasing through `Biasing`, which owns that comparison deliberately.
#[derive(Clone, Debug, PartialEq)]
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
    /// Which implementation of the greedy RNNT loop the worker runs.
    pub rnnt_engine: CoreMlRnntEngine,
    /// Terms to bias recognition toward, sent once at spawn so the worker
    /// builds its context trie a single time rather than per utterance. Empty
    /// is the default and costs the decode loop one nil check per window.
    pub vocabulary: Vec<String>,
    /// Per-token log-probability boost applied to the biased tokens. Read only
    /// when `vocabulary` is non-empty; sherpa's `hotwords_score` is the
    /// reference for what the number means.
    pub vocabulary_score: f32,
    /// Model download and first compilation can take substantially longer than inference.
    pub startup_timeout: Duration,
    /// Total budget for sending a request and receiving its complete response.
    pub request_timeout: Duration,
}

/// The two implementations of the transducer decode loop the worker can run.
///
/// They decode the same weights. The Core ML one pays a dispatch per
/// prediction-network step and per joint evaluation; the native one reads the
/// weights out of the same bundles and runs the arithmetic in process. Both
/// stay reachable so a measurement can name which produced it.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoreMlRnntEngine {
    #[default]
    Native,
    CoreMl,
}

impl CoreMlRnntEngine {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::CoreMl => "coreml",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "native" => Ok(Self::Native),
            "coreml" => Ok(Self::CoreMl),
            other => bail!("unknown RNNT engine: {other} (expected native or coreml)"),
        }
    }
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
            rnnt_engine: CoreMlRnntEngine::default(),
            vocabulary: Vec::new(),
            vocabulary_score: 0.0,
            startup_timeout: Duration::from_secs(30 * 60),
            request_timeout: Duration::from_secs(5 * 60),
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
            rnnt_engine: CoreMlRnntEngine::default(),
            vocabulary: Vec::new(),
            vocabulary_score: 0.0,
            startup_timeout: Duration::from_secs(30 * 60),
            request_timeout: Duration::from_secs(5 * 60),
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

    pub fn set_rnnt_engine(&mut self, engine: CoreMlRnntEngine) {
        self.rnnt_engine = engine;
    }

    /// Bias recognition toward `terms` with a per-token boost of `score`.
    ///
    /// The terms go over the wire as written: the tokenizer that turns them
    /// into the model's pieces lives in the model bundle, which only the worker
    /// has open.
    pub fn set_vocabulary(&mut self, terms: Vec<String>, score: f32) {
        self.vocabulary = terms;
        self.vocabulary_score = score;
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
        let model_directory =
            match variant_directory.or_else(|| std::env::var_os("PARAKEET_COREML_MODEL_DIR")) {
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
    config: CoreMlWorkerConfig,
    metadata: AsrBackendMetadata,
    load_seconds: f64,
    /// Stage breakdown from the most recent result, when the worker reports one.
    last_stages: Mutex<Option<StageReport>>,
    /// What the worker made of the vocabulary, or `None` when none was sent.
    vocabulary: Option<VocabularyStatus>,
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
            .arg(config.long_regime_seconds.to_string())
            .arg("--rnnt-engine")
            .arg(config.rnnt_engine.as_str());
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
        let mut process = WorkerProcess::new(child, stdin, stdout)?;
        let ready = process
            .transaction(config.startup_timeout, |process, deadline| {
                process.read_response(deadline)
            })
            .context("waiting for Core ML worker readiness")?;
        ready.require_success("ready")?;
        let load_seconds = ready
            .load_seconds
            .ok_or_else(|| anyhow!("Core ML ready response omitted load_seconds"))?;

        // A vocabulary the worker cannot apply is fatal here rather than
        // logged: the caller asked for biasing, and a worker that silently
        // decoded without it would look like a quality regression with no
        // trace of its cause.
        let vocabulary = if config.vocabulary.is_empty() {
            None
        } else {
            Some(
                process
                    .transaction(config.request_timeout, |process, deadline| {
                        process.set_vocabulary(
                            &config.vocabulary,
                            config.vocabulary_score,
                            deadline,
                        )
                    })
                    .context("sending the custom vocabulary to the Core ML worker")?,
            )
        };
        if let Some(status) = &vocabulary {
            for term in &status.rejected {
                log::warn!(
                    "vocabulary: {term:?} can't be represented by this model's token \
                     inventory — skipped"
                );
            }
            // Logged rather than summarized: a term whose split disagrees with
            // the model's own segmentation is accepted and boosts nothing, and
            // this line is the only place that shows it.
            for entry in &status.encoded {
                log::info!(
                    "vocabulary: {:?} biased as {:?}",
                    entry.term,
                    entry.pieces.join(" ")
                );
            }
            log::info!(
                "contextual biasing ON (score {}): {} of {} terms active",
                config.vocabulary_score,
                status.accepted,
                config.vocabulary.len()
            );
        }

        Ok(Self {
            process: Mutex::new(process),
            config: config.clone(),
            metadata: AsrBackendMetadata {
                backend: "fluid-audio-worker".to_string(),
                model: config.model_variant.model_description().to_string(),
                quantization: "int8 encoder".to_string(),
                execution_provider: format!(
                    "Core ML short={} long={} threshold={}s rnnt={}",
                    config.short_compute_units.as_str(),
                    config.long_compute_units.as_str(),
                    config.long_regime_seconds,
                    config.rnnt_engine.as_str()
                ),
            },
            load_seconds,
            last_stages: Mutex::new(None),
            vocabulary,
        })
    }
}

impl CoreMlWorkerBackend {
    /// One request/response round trip. Both trait entry points share it so the
    /// text a caller gets can never disagree with the spans beside it.
    fn decode(
        &self,
        samples: &[f32],
        sample_rate: u32,
        require_spans: bool,
    ) -> Result<(Decoded, Vec<TokenSpan>)> {
        validate_request(samples.len(), sample_rate)?;
        // Production capture already delivers 16 kHz, so this borrows. File-fed
        // callers (the gold corpus is 48 kHz) convert here rather than leaving
        // it to the worker's per-call AVAudioConverter.
        let model_samples = to_target_rate(samples, sample_rate)
            .context("converting audio to the Core ML worker's 16 kHz input rate")?;
        let sample_count = validate_request(model_samples.len(), TARGET_SAMPLE_RATE)?;

        let mut process = self.process.lock();
        if process.failed {
            // Retry on a new request only: replaying audio after an uncertain
            // response could hide a failure. Spawn also reapplies vocabulary.
            *process = Self::spawn(&self.config)
                .context("restarting Core ML worker after an IPC failure")?
                .process
                .into_inner();
        }
        let response = process.transaction(self.config.request_timeout, |process, deadline| {
            process
                .write_request(&model_samples, TARGET_SAMPLE_RATE, sample_count, deadline)
                .context("sending audio to Core ML worker")?;
            process
                .read_response(deadline)
                .context("reading Core ML worker result")
        })?;
        response.require_success("result")?;
        *self.last_stages.lock() = response.stages.clone();
        drop(process);
        let decode_seconds = response
            .decode_seconds
            .ok_or_else(|| anyhow!("Core ML result omitted decode_seconds"))?
            + response.resample_seconds.unwrap_or(0.0);
        // Only the caller that asked for spans may fail for their absence. The
        // plain path does not read them, and it is the fallback every Hold
        // window failure lands on — making it depend on a field it ignores
        // would defeat the fallback for exactly the worker that needs it.
        let spans = match (&response.token_spans, require_spans) {
            (Some(spans), _) => spans.iter().map(TokenSpan::from).collect(),
            (None, false) => Vec::new(),
            (None, true) => bail!(
                "Core ML result omitted token_spans; this worker predates the \
                 Hold window merge and must be rebuilt"
            ),
        };

        Ok((
            Decoded {
                text: response
                    .text
                    .ok_or_else(|| anyhow!("Core ML result omitted text"))?,
                audio_seconds: samples.len() as f32 / sample_rate as f32,
                decode_seconds: decode_seconds as f32,
            },
            spans,
        ))
    }
}

impl AsrBackend for CoreMlWorkerBackend {
    fn metadata(&self) -> &AsrBackendMetadata {
        &self.metadata
    }

    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
        Ok(self
            .decode(samples, sample_rate, /* require_spans = */ false)?
            .0)
    }

    fn reports_token_spans(&self) -> bool {
        true
    }

    fn transcribe_with_token_spans(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<(Decoded, Vec<TokenSpan>)> {
        self.decode(samples, sample_rate, /* require_spans = */ true)
    }

    fn auxiliary_resident_bytes(&self) -> Result<u64> {
        let pid = self.process.lock().child.id();
        crate::performance::resident_bytes(pid)
            .with_context(|| format!("reading Core ML worker {pid} resident set"))
    }

    fn last_stage_report(&self) -> Option<StageReport> {
        self.last_stages.lock().clone()
    }

    fn contextual_vocabulary(&self) -> Option<VocabularyStatus> {
        self.vocabulary.clone()
    }
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    failed: bool,
}

impl WorkerProcess {
    fn new(child: Child, stdin: ChildStdin, stdout: ChildStdout) -> Result<Self> {
        let process = Self {
            child,
            stdin,
            stdout,
            failed: false,
        };
        set_nonblocking(process.stdin.as_raw_fd())?;
        set_nonblocking(process.stdout.as_raw_fd())?;
        Ok(process)
    }

    fn transaction<T>(
        &mut self,
        timeout: Duration,
        operation: impl FnOnce(&mut Self, Instant) -> Result<T>,
    ) -> Result<T> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| anyhow!("Core ML IPC timeout is too large"))?;
        let result = operation(self, deadline);
        if result.is_err() {
            self.failed = true;
            self.terminate();
        }
        result
    }

    fn terminate(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn write_request(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        sample_count: u32,
        deadline: Instant,
    ) -> Result<()> {
        let header = encode_header(sample_rate, sample_count);
        write_before(&mut self.stdin, &header, deadline)?;

        // The worker protocol is explicitly little-endian. Encoding without a
        // raw-slice cast keeps this safe and portable; IPC cost is measured by
        // the outer benchmark and can be optimized only if it is material.
        let mut payload = Vec::with_capacity(std::mem::size_of_val(samples));
        for sample in samples {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        write_before(&mut self.stdin, &payload, deadline)?;
        Ok(())
    }

    /// Send the `PRKV` control frame and read the worker's verdict on it.
    fn set_vocabulary(
        &mut self,
        terms: &[String],
        score: f32,
        deadline: Instant,
    ) -> Result<VocabularyStatus> {
        let payload = serde_json::to_vec(&VocabularyRequest { terms, score })
            .context("encoding the vocabulary request")?;
        if payload.len() > MAX_VOCABULARY_BYTES {
            bail!(
                "vocabulary payload is {} bytes, over the {MAX_VOCABULARY_BYTES}-byte limit",
                payload.len()
            );
        }
        let length = u32::try_from(payload.len()).context("vocabulary payload exceeds u32")?;
        write_before(&mut self.stdin, &encode_vocabulary_header(length), deadline)?;
        write_before(&mut self.stdin, &payload, deadline)?;
        let response = self.read_response(deadline)?;
        response.require_success("vocabulary")?;
        Ok(VocabularyStatus {
            accepted: response
                .vocabulary_accepted
                .ok_or_else(|| anyhow!("Core ML vocabulary response omitted the accepted count"))?,
            encoded: response.vocabulary_encoded.unwrap_or_default(),
            rejected: response.vocabulary_rejected.unwrap_or_default(),
        })
    }

    fn read_response(&mut self, deadline: Instant) -> Result<WorkerResponse> {
        let mut length = [0_u8; 4];
        read_before(&mut self.stdout, &mut length, deadline)?;
        let length = u32::from_le_bytes(length) as usize;
        if length > MAX_RESPONSE_BYTES {
            bail!("Core ML worker response is too large: {length} bytes");
        }
        let mut payload = vec![0_u8; length];
        read_before(&mut self.stdout, &mut payload, deadline)?;
        serde_json::from_slice(&payload).context("decoding Core ML worker response")
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.terminate();
    }
}

// These descriptors are exclusively owned by WorkerProcess. Nonblocking mode
// is essential: poll readiness alone does not bound a large pipe write.
fn set_nonblocking(fd: RawFd) -> io::Result<()> {
    // SAFETY: fd is a live pipe descriptor; F_GETFL takes no third argument.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: F_SETFL accepts these flags and does not retain any pointers.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn wait_before(fd: RawFd, events: libc::c_short, deadline: Instant) -> io::Result<()> {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(io::ErrorKind::TimedOut,
                "Core ML worker IPC deadline expired; worker stopped; retry dictation to restart it"));
        }
        let millis = i32::try_from(remaining.as_millis().saturating_add(1)).unwrap_or(i32::MAX);
        let mut pollfd = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        // SAFETY: pollfd is valid for one descriptor and lives across this call.
        let ready = unsafe { libc::poll(&mut pollfd, 1, millis) };
        if ready > 0 && Instant::now() < deadline {
            return Ok(());
        } // EOF/error is reported by read/write.
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}

fn read_before(pipe: &mut ChildStdout, mut bytes: &mut [u8], deadline: Instant) -> io::Result<()> {
    while !bytes.is_empty() {
        wait_before(pipe.as_raw_fd(), libc::POLLIN, deadline)?;
        match pipe.read(bytes) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Core ML worker closed its response pipe",
                ))
            }
            Ok(count) => bytes = &mut bytes[count..],
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                ) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn write_before(pipe: &mut ChildStdin, mut bytes: &[u8], deadline: Instant) -> io::Result<()> {
    while !bytes.is_empty() {
        wait_before(pipe.as_raw_fd(), libc::POLLOUT, deadline)?;
        match pipe.write(bytes) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "Core ML worker accepted no request bytes",
                ))
            }
            Ok(count) => bytes = &bytes[count..],
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                ) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// The `PRKV` frame's JSON body. Borrowed, so a large vocabulary is not cloned
/// on its way to the worker.
#[derive(Serialize)]
struct VocabularyRequest<'a> {
    terms: &'a [String],
    score: f32,
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
    vocabulary_accepted: Option<u32>,
    vocabulary_rejected: Option<Vec<String>>,
    vocabulary_encoded: Option<Vec<EncodedTerm>>,
    /// One entry per RNNT emission, on the request's own timeline. Absent from
    /// `ready` and failure frames, and from any worker built before the Hold
    /// window merge needed them.
    token_spans: Option<Vec<WireTokenSpan>>,
}

/// Wire form of one emission. Separate from [`TokenSpan`] so the worker's
/// `convertToSnakeCase` field names are pinned here rather than in the module
/// the merge logic lives in.
#[derive(Clone, Debug, Deserialize)]
struct WireTokenSpan {
    text: String,
    start_s: f64,
    end_s: f64,
}

impl From<&WireTokenSpan> for TokenSpan {
    fn from(wire: &WireTokenSpan) -> Self {
        Self {
            text: wire.text.clone(),
            start_s: wire.start_s as f32,
            end_s: wire.end_s as f32,
        }
    }
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

/// magic, protocol version, JSON payload length, reserved.
fn encode_vocabulary_header(payload_bytes: u32) -> [u8; 16] {
    let mut header = [0_u8; 16];
    header[..4].copy_from_slice(&VOCABULARY_MAGIC);
    header[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&payload_bytes.to_le_bytes());
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
    fn vocabulary_frame_is_distinguishable_from_an_audio_frame() {
        // The worker dispatches on the magic, so an audio header must never be
        // readable as a vocabulary one: a length read as a sample rate would
        // fail the range check rather than transcribe garbage, but the frames
        // still have to be told apart before that.
        let vocabulary = encode_vocabulary_header(1234);
        assert_eq!(&vocabulary[..4], b"PRKV");
        assert_ne!(&vocabulary[..4], &encode_header(16_000, 1)[..4]);
        assert_eq!(u32::from_le_bytes(vocabulary[4..8].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(vocabulary[8..12].try_into().unwrap()),
            1234
        );
        assert_eq!(&vocabulary[12..], &[0, 0, 0, 0]);
    }

    #[test]
    fn vocabulary_request_sends_the_terms_verbatim() {
        // The worker tokenizes them against the model bundle, so anything this
        // side does to a term (case folding, the sherpa word-start marker)
        // would bias a word the user did not write.
        let terms = vec!["IBM".to_string(), "New York".to_string()];
        let payload = serde_json::to_string(&VocabularyRequest {
            terms: &terms,
            score: 2.0,
        })
        .unwrap();
        assert_eq!(payload, r#"{"terms":["IBM","New York"],"score":2.0}"#);
    }

    #[test]
    fn vocabulary_response_reports_accepted_and_rejected_terms() {
        // Rejected terms are the diagnostic the sherpa path gets from its own
        // token validation. Losing them means a user's word silently boosts
        // nothing with nothing anywhere to explain it.
        let payload = br#"{
            "kind": "vocabulary", "ok": true,
            "vocabulary_accepted": 2, "vocabulary_rejected": ["Zzz"],
            "vocabulary_encoded": [
                {"term": "IBM", "pieces": ["\u2581I", "BM"]},
                {"term": "New York", "pieces": ["\u2581New", "\u2581York"]}
            ]
        }"#;
        let response: WorkerResponse = serde_json::from_slice(payload).expect("valid response");
        response.require_success("vocabulary").expect("ok");
        assert_eq!(response.vocabulary_accepted, Some(2));
        assert_eq!(
            response.vocabulary_rejected.as_deref(),
            Some(&["Zzz".to_string()][..])
        );
        // The segmentation is the only signal that separates a term which is
        // biasing the model's own token path from one that is accepted and
        // boosting a path the joint never walks.
        let encoded = response.vocabulary_encoded.expect("segmentation present");
        assert_eq!(encoded[0].term, "IBM");
        assert_eq!(encoded[0].pieces, vec!["▁I", "BM"]);
        assert_eq!(encoded[1].pieces, vec!["▁New", "▁York"]);
    }

    #[test]
    fn a_result_response_carries_no_vocabulary_fields() {
        let payload = br#"{"kind": "result", "ok": true, "text": "hi", "decode_seconds": 0.04}"#;
        let response: WorkerResponse = serde_json::from_slice(payload).expect("valid result");
        assert!(response.vocabulary_accepted.is_none());
        assert!(response.vocabulary_rejected.is_none());
    }

    #[test]
    fn an_empty_vocabulary_leaves_the_config_unbiased() {
        let mut config = CoreMlWorkerConfig::new("worker", "model");
        assert!(config.vocabulary.is_empty());
        config.set_vocabulary(vec!["IBM".to_string()], 2.0);
        assert_eq!(config.vocabulary, vec!["IBM".to_string()]);
        assert_eq!(config.vocabulary_score, 2.0);
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
            vocabulary_accepted: None,
            vocabulary_rejected: None,
            vocabulary_encoded: None,
            token_spans: None,
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
                "decoder_calls": 35, "joint_calls": 96,
                "native_decoder_steps": 0, "native_joint_steps": 0, "other_calls": 0,
                "mel_ms": 3.08, "encoder_ms": 25.959, "decode_loop_ms": 15.84,
                "decode_loop_dispatch_ms": 15.135, "decoder_dispatch_ms": 5.359,
                "joint_dispatch_ms": 9.776, "decode_loop_native_ms": 0.0,
                "post_ms": 0.066, "total_ms": 44.944,
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
                "native_decoder_steps": 0, "native_joint_steps": 0,
                "other_calls": 0, "mel_ms": 0.42, "preprocessor_ms": 6.1,
                "encoder_ms": 25.9, "decode_loop_ms": 9.2,
                "decode_loop_dispatch_ms": 8.8, "decoder_dispatch_ms": 3.1,
                "joint_dispatch_ms": 5.7, "decode_loop_native_ms": 0.0,
                "post_ms": 0.05, "overlapped_dispatch_ms": 0.0, "total_ms": 41.67,
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
    fn result_response_carries_the_worker_token_spans() {
        // Field names are the worker's `convertToSnakeCase` encoding of its
        // `TokenSpan`, and the piece keeps the leading space the tokenizer's
        // word-start marker became — that space is what groups tokens into
        // words on this side.
        let payload = br#"{
            "kind": "result", "ok": true, "text": "Hi there.", "decode_seconds": 0.04,
            "token_spans": [
                {"text": " Hi", "start_s": 0.08, "end_s": 0.16},
                {"text": " there", "start_s": 0.24, "end_s": 0.32},
                {"text": ".", "start_s": 0.32, "end_s": 0.40}
            ]
        }"#;
        let response: WorkerResponse = serde_json::from_slice(payload).expect("valid result");
        let wire = response.token_spans.expect("token spans present");
        let spans: Vec<TokenSpan> = wire.iter().map(TokenSpan::from).collect();
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, " Hi");
        assert!((spans[2].end_s - 0.40).abs() < 1e-6);
        let words = crate::windows::words_from_tokens(&spans, 0.0);
        assert_eq!(crate::windows::words_to_text(&words), "Hi there.");
    }

    #[test]
    fn a_worker_without_token_spans_still_serves_the_plain_decode() {
        // The plain decode is where every Hold window failure falls back to,
        // so it must not depend on a field it never reads. Only the span-aware
        // entry point may refuse a worker that predates them.
        let payload = br#"{"kind": "result", "ok": true, "text": "hi", "decode_seconds": 0.04}"#;
        let response: WorkerResponse = serde_json::from_slice(payload).expect("valid result");
        assert!(response.token_spans.is_none());
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

#[cfg(test)]
mod ipc_tests;
