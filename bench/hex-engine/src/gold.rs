//! The gold-corpus arm, for both engines. Follows `asr_diff --gold` step for
//! step: the same 16 kHz buffers from `resample::to_target_rate`, the same
//! `warmup::dummy_decode`, the same RSS sampling after every decode (worker
//! included), and `asr_eval::evaluate` for the report.
//!
//! Every arm is timed by one wall clock on the caller's side of one request:
//! what the app waits for between handing over 16 kHz audio and holding the
//! text. For our Core ML engine that includes the worker IPC and its JSON;
//! for Hex's it includes the padding, copies and the session rotation Hex
//! performs before it returns. `asr_diff` reports the worker's own inference
//! time instead, which leaves the IPC out, so its timings are not comparable
//! with Hex's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use parakeet_dictation::asr::{Asr, Decoded};
use parakeet_dictation::asr_eval::{self, DecodeMetadata, GoldManifest, RunMetadata};
use parakeet_dictation::coreml_worker::{
    load_coreml_worker, CoreMlModelVariant, CoreMlWorkerConfig,
};
use parakeet_dictation::performance;
use parakeet_dictation::resample::{to_target_rate, TARGET_SAMPLE_RATE};
use parakeet_dictation::warmup;
use parakeet_dictation::wav::read_wav_mono;

use crate::engine::{HexEngine, HexOfflineBackend};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    /// Hex's whole-clip run: its CPU path, and its fallback on Metal.
    Offline,
    /// Hex's Metal dictation path: the clip fed to a live stream in
    /// capture-sized blocks, unpaced, then finalized.
    Stream,
}

impl DecodePath {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "offline" => Ok(Self::Offline),
            "stream" => Ok(Self::Stream),
            other => Err(anyhow!(
                "unknown path {other:?}; expected offline or stream"
            )),
        }
    }

    fn method(self) -> &'static str {
        match self {
            Self::Offline => "transcribe_cpp_greedy_offline",
            Self::Stream => "transcribe_cpp_greedy_buffered_stream",
        }
    }
}

/// Which engine a gold run drives.
pub enum GoldEngine {
    /// Hex's transcribe.cpp engine on a GGUF.
    Hex { model: PathBuf, path: DecodePath },
    /// Our Core ML Unified engine through its worker, as `asr_diff
    /// --backend coreml-unified` loads it.
    CoreMl { worker: PathBuf, model_dir: PathBuf },
}

impl GoldEngine {
    fn method(&self) -> &'static str {
        match self {
            Self::Hex { path, .. } => path.method(),
            Self::CoreMl { .. } => "coreml_unified_greedy",
        }
    }
}

pub struct GoldArgs {
    pub engine: GoldEngine,
    pub gold: PathBuf,
    pub audio_dir: PathBuf,
    pub repetitions: usize,
    pub json_out: PathBuf,
}

/// The benchmark's resident set plus any worker process behind the facade.
fn resident(asr: &Asr) -> Result<u64> {
    let own = performance::resident_bytes(std::process::id())
        .context("reading benchmark resident set")?;
    Ok(own + asr.auxiliary_resident_bytes()?)
}

/// Loads the engine. The Hex backend comes back too, for its stream path,
/// which the facade cannot express.
fn load(engine: &GoldEngine) -> Result<(Asr, Option<Arc<HexOfflineBackend>>)> {
    match engine {
        GoldEngine::Hex { model, .. } => {
            let backend = Arc::new(HexOfflineBackend::new(HexEngine::load(model)?));
            Ok((Asr::from_backend(backend.clone()), Some(backend)))
        }
        GoldEngine::CoreMl { worker, model_dir } => {
            let mut config = CoreMlWorkerConfig::discover()?;
            config.worker_path.clone_from(worker);
            config.set_existing_model_directory(model_dir);
            config.set_model_variant(CoreMlModelVariant::Unified)?;
            let (asr, worker_load_seconds) = load_coreml_worker(&config)?;
            eprintln!("Core ML worker ready in {worker_load_seconds:.3}s");
            Ok((asr, None))
        }
    }
}

pub fn run(args: &GoldArgs) -> Result<bool> {
    let raw = std::fs::read_to_string(&args.gold)
        .with_context(|| format!("reading gold manifest {}", args.gold.display()))?;
    let manifest: GoldManifest = serde_json::from_str(&raw)
        .with_context(|| format!("parsing gold manifest {}", args.gold.display()))?;
    manifest.validate()?;

    let run_started = Instant::now();
    let load_started = Instant::now();
    let (asr, hex) = load(&args.engine)?;
    let stream = match (&args.engine, &hex) {
        (GoldEngine::Hex { path: DecodePath::Stream, .. }, Some(backend)) => Some(backend),
        _ => None,
    };
    let backend_metadata = asr.backend_metadata().clone();
    let model_load_seconds = load_started.elapsed().as_secs_f64();
    let mut peak = resident(&asr)?;

    let warmup_started = Instant::now();
    warmup::dummy_decode(&asr)?;
    if let Some(backend) = stream {
        backend.with_engine(HexEngine::prewarm_stream)?;
    }
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();
    peak = peak.max(resident(&asr)?);

    let inputs = manifest
        .fixtures
        .iter()
        .map(|fixture| {
            let path = args.audio_dir.join(&fixture.file);
            let (samples, rate) = read_wav_mono(&path)?;
            Ok((
                fixture.file.clone(),
                to_target_rate(&samples, rate)?.into_owned(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut decoded: BTreeMap<String, Vec<Decoded>> = inputs
        .iter()
        .map(|(name, _)| (name.clone(), Vec::with_capacity(args.repetitions)))
        .collect();
    let mut first_result_seconds = None;
    for repetition in 0..args.repetitions {
        eprintln!("repetition {}/{}", repetition + 1, args.repetitions);
        for (name, samples) in &inputs {
            // The one timer every arm shares; see the module comment.
            let started = Instant::now();
            let text = match stream {
                Some(backend) => backend.with_engine(|engine| engine.transcribe_streamed(samples))?,
                None => asr.recognize_with_metrics(samples, TARGET_SAMPLE_RATE)?.text,
            };
            let result = Decoded {
                text,
                audio_seconds: samples.len() as f32 / TARGET_SAMPLE_RATE as f32,
                decode_seconds: started.elapsed().as_secs_f32(),
            };
            peak = peak.max(resident(&asr)?);
            first_result_seconds.get_or_insert_with(|| run_started.elapsed().as_secs_f64());
            if repetition == 0 {
                eprintln!(
                    "  {name}: {:?} ({:.3}s, {:.1}x real time)",
                    result.text,
                    result.decode_seconds,
                    result.rtfx()
                );
            }
            decoded
                .get_mut(name)
                .expect("decoded map was built from the same fixture list")
                .push(result);
        }
    }

    let metadata = run_metadata(
        backend_metadata,
        args.engine.method(),
        model_load_seconds,
        warmup_seconds,
        first_result_seconds.ok_or_else(|| anyhow!("no fixture produced a result"))?,
        peak.max(performance::peak_resident_bytes()?),
    )?;
    let report = asr_eval::evaluate(&manifest, &decoded, metadata)?;
    write_json(&args.json_out, &report)?;
    let overall = &report.overall;
    let repeat = &report.repeatability;
    println!(
        "{}: WER {:.4}% CER {:.4}%, corpus decode p50 {:.4}s / p95 {:.4}s, peak RSS {:.3} GiB, gate {}",
        args.engine.method(),
        overall.wer_percent.unwrap_or(f64::NAN),
        overall.cer_percent.unwrap_or(f64::NAN),
        repeat.corpus_decode_seconds_p50,
        repeat.corpus_decode_seconds_p95,
        report.metadata.peak_resident_bytes as f64 / f64::from(1_u32 << 30),
        if report.passed { "pass" } else { "fail" }
    );
    println!("wrote {}", args.json_out.display());
    Ok(report.passed)
}

fn run_metadata(
    backend: parakeet_dictation::asr::AsrBackendMetadata,
    method: &str,
    model_load_seconds: f64,
    warmup_seconds: f64,
    first_result_seconds: f64,
    peak_resident_bytes: u64,
) -> Result<RunMetadata> {
    let os_name = command_output("sw_vers", &["-productName"])
        .unwrap_or_else(|| std::env::consts::OS.to_string());
    let operating_system = command_output("sw_vers", &["-productVersion"])
        .map_or(os_name.clone(), |version| format!("{os_name} {version}"));
    Ok(RunMetadata {
        application_version: env!("CARGO_PKG_VERSION").to_string(),
        backend,
        decoding: DecodeMetadata {
            method: method.to_string(),
            contextual_vocabulary_requested: false,
            contextual_vocabulary_active: false,
            hotword_score: None,
            vocabulary_terms_requested: 0,
            vocabulary_sha256: None,
            generated_hotwords_sha256: None,
        },
        operating_system,
        architecture: std::env::consts::ARCH.to_string(),
        chip: performance::sysctl_string("machdep.cpu.brand_string")
            .context("reading chip name")?,
        memory_bytes: performance::sysctl_u64("hw.memsize").context("reading memory size")?,
        logical_cpus: std::thread::available_parallelism()?.get(),
        model_load_seconds,
        warmup_seconds,
        first_result_seconds,
        peak_resident_bytes,
    })
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn write_json(path: &Path, report: &asr_eval::QualityReport) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, format!("{}\n", serde_json::to_string_pretty(report)?))
        .with_context(|| format!("writing {}", path.display()))
}
