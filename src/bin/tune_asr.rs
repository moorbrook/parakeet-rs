//! Explicit full Core ML runtime-plan tuner.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use parakeet_rs::asr::{Asr, Decoded};
use parakeet_rs::asr_eval::{self, GoldManifest, RunMetadata};
use parakeet_rs::asr_tuning::{
    self, CandidateMeasurement, CandidateStatus, CategoryQuality, HardwareFingerprint,
    QualityMeasurement, RegimeMeasurement, TuningProfile,
};
use parakeet_rs::coreml_worker::{
    load_coreml_worker, CoreMlComputeUnits, CoreMlWorkerConfig, DEFAULT_LONG_REGIME_SECONDS,
};
use parakeet_rs::model_fetch;
use parakeet_rs::performance;
use parakeet_rs::settings::SettingsStore;
use parakeet_rs::warmup;
use parakeet_rs::wav::read_wav_mono;

const DEFAULT_GOLD: &str = "bench/gold/manifest.json";
const DEFAULT_AUDIO_DIR: &str = "bench/gold/audio";
const DEFAULT_REPETITIONS: usize = 5;
const DEFAULT_LOAD_REPETITIONS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Tune,
    Show,
    Remove,
}

struct Args {
    action: Action,
    gold: PathBuf,
    audio_dir: PathBuf,
    worker: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    profile: Option<PathBuf>,
    repetitions: usize,
    load_repetitions: usize,
}

struct AudioInput {
    file: String,
    samples: Vec<f32>,
    sample_rate: u32,
    audio_seconds: f64,
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    match parse_args().and_then(|args| run(&args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("tune_asr failed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args() -> Result<Args> {
    let mut action = Action::Tune;
    let mut gold = PathBuf::from(DEFAULT_GOLD);
    let mut audio_dir = PathBuf::from(DEFAULT_AUDIO_DIR);
    let mut worker = None;
    let mut model_dir = None;
    let mut profile = None;
    let mut repetitions = DEFAULT_REPETITIONS;
    let mut load_repetitions = DEFAULT_LOAD_REPETITIONS;
    let mut iterator = std::env::args().skip(1);
    while let Some(argument) = iterator.next() {
        match argument.as_str() {
            "--gold" => gold = next_path(&mut iterator, "--gold")?,
            "--audio-dir" => audio_dir = next_path(&mut iterator, "--audio-dir")?,
            "--worker" => worker = Some(next_path(&mut iterator, "--worker")?),
            "--model-dir" => model_dir = Some(next_path(&mut iterator, "--model-dir")?),
            "--profile" => profile = Some(next_path(&mut iterator, "--profile")?),
            "--repetitions" => {
                repetitions = next_usize(&mut iterator, "--repetitions")?;
            }
            "--load-repetitions" => {
                load_repetitions = next_usize(&mut iterator, "--load-repetitions")?;
            }
            "--show-profile" => action = exclusive_action(action, Action::Show)?,
            "--remove-profile" => action = exclusive_action(action, Action::Remove)?,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    if repetitions == 0 || load_repetitions == 0 {
        bail!("repetition counts must be at least one");
    }
    Ok(Args {
        action,
        gold,
        audio_dir,
        worker,
        model_dir,
        profile,
        repetitions,
        load_repetitions,
    })
}

fn next_path(iterator: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    iterator
        .next()
        .map(PathBuf::from)
        .with_context(|| format!("{flag} needs a path"))
}

fn next_usize(iterator: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize> {
    iterator
        .next()
        .with_context(|| format!("{flag} needs a positive integer"))?
        .parse()
        .with_context(|| format!("parsing {flag}"))
}

fn exclusive_action(current: Action, requested: Action) -> Result<Action> {
    if current != Action::Tune {
        bail!("--show-profile and --remove-profile are mutually exclusive");
    }
    Ok(requested)
}

fn print_usage() {
    eprintln!(
        "usage: tune_asr [--gold JSON] [--audio-dir DIR] [--worker PATH]\n\
         \x20               [--model-dir DIR] [--profile JSON]\n\
         \x20               [--repetitions N] [--load-repetitions N]\n\
         \x20      tune_asr [--profile JSON] --show-profile\n\
         \x20      tune_asr [--profile JSON] --remove-profile\n\
         \n\
         Benchmarks every bounded Core ML compute-unit candidate on the\n\
         checked-in human-speech gold corpus, refuses quality/category/memory\n\
         regressions, selects short and long regimes deterministically, and\n\
         atomically caches the complete evidence profile."
    );
}

fn run(args: &Args) -> Result<()> {
    let settings = SettingsStore::new()?;
    let profile_path = args
        .profile
        .clone()
        .unwrap_or_else(|| settings.asr_tuning_profile_path());
    match args.action {
        Action::Show => return show_profile(&profile_path),
        Action::Remove => {
            let removed = asr_tuning::remove(&profile_path)?;
            println!(
                "{} {}",
                if removed { "removed" } else { "not found" },
                profile_path.display()
            );
            return Ok(());
        }
        Action::Tune => {}
    }

    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(|| settings.coreml_model_dir());
    verify_model(&model_dir)?;
    let manifest = read_manifest(&args.gold)?;
    let inputs = read_inputs(&manifest, &args.audio_dir)?;
    let hardware = HardwareFingerprint::current()?;
    println!(
        "tuning {} on {} / macOS {} / {} GiB / {:?}",
        asr_tuning::BACKEND_ID,
        hardware.chip,
        hardware.macos_version,
        hardware.memory_bytes / 1024 / 1024 / 1024,
        hardware.performance_levels
    );

    let mut base_config = CoreMlWorkerConfig::discover()?;
    if let Some(worker) = &args.worker {
        base_config.worker_path.clone_from(worker);
    }
    base_config.set_existing_model_directory(&model_dir);

    let mut candidates = Vec::with_capacity(CoreMlComputeUnits::CANDIDATES.len());
    for compute_units in CoreMlComputeUnits::CANDIDATES {
        eprintln!("tuning {}…", compute_units.as_str());
        let mut config = base_config.clone();
        config.set_compute_units(compute_units);
        let candidate = tune_candidate(
            compute_units,
            &config,
            &manifest,
            &inputs,
            &hardware,
            args.repetitions,
            args.load_repetitions,
        )
        .unwrap_or_else(|error| {
            eprintln!("  failed: {error:#}");
            CandidateMeasurement::failed(compute_units, format!("{error:#}"))
        });
        candidates.push(candidate);
    }

    let artifact_digest = model_fetch::coreml_artifact_digest();
    let profile = TuningProfile::new(hardware, artifact_digest, args.repetitions, candidates)?;
    asr_tuning::save(&profile_path, &profile)?;
    println!("{}", serde_json::to_string_pretty(&profile)?);
    eprintln!("saved {}", profile_path.display());
    Ok(())
}

fn verify_model(model_dir: &Path) -> Result<()> {
    let progress: model_fetch::ProgressFn = Arc::new(|_| {});
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(model_fetch::ensure_coreml_model(model_dir, progress))
}

fn show_profile(path: &Path) -> Result<()> {
    let profile = asr_tuning::load(path)?
        .with_context(|| format!("tuning profile not found at {}", path.display()))?;
    println!("{}", serde_json::to_string_pretty(&profile)?);
    Ok(())
}

fn read_manifest(path: &Path) -> Result<GoldManifest> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let manifest: GoldManifest =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    manifest.validate()?;
    Ok(manifest)
}

fn read_inputs(manifest: &GoldManifest, audio_dir: &Path) -> Result<Vec<AudioInput>> {
    manifest
        .fixtures
        .iter()
        .map(|fixture| {
            let path = audio_dir.join(&fixture.file);
            let (samples, sample_rate) = read_wav_mono(&path)?;
            let audio_seconds = samples.len() as f64 / f64::from(sample_rate);
            Ok(AudioInput {
                file: fixture.file.clone(),
                samples,
                sample_rate,
                audio_seconds,
            })
        })
        .collect()
}

fn tune_candidate(
    compute_units: CoreMlComputeUnits,
    config: &CoreMlWorkerConfig,
    manifest: &GoldManifest,
    inputs: &[AudioInput],
    hardware: &HardwareFingerprint,
    repetitions: usize,
    load_repetitions: usize,
) -> Result<CandidateMeasurement> {
    let mut load_ms = Vec::with_capacity(load_repetitions);
    let mut loaded = None;
    for load_index in 0..load_repetitions {
        let started = Instant::now();
        let (asr, _) = load_coreml_worker(config)?;
        load_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        if load_index + 1 == load_repetitions {
            loaded = Some(asr);
        }
    }
    let asr = loaded.context("load loop did not retain a recognizer")?;
    let first_observed_load_ms = load_ms[0];
    let load_p50_ms = percentile(&mut load_ms, 50);

    let warmup_started = Instant::now();
    warmup::dummy_decode(&asr)?;
    let warmup_ms = warmup_started.elapsed().as_secs_f64() * 1_000.0;
    let mut peak_resident_bytes = process_tree_resident_bytes(&asr)?;
    let run_started = Instant::now();
    let mut first_result_seconds = None;
    let mut decoded: BTreeMap<String, Vec<Decoded>> = inputs
        .iter()
        .map(|input| (input.file.clone(), Vec::with_capacity(repetitions)))
        .collect();
    let mut short_wall_ms = vec![0.0; repetitions];
    let mut long_wall_ms = vec![0.0; repetitions];
    let mut short_audio_seconds = 0.0;
    let mut long_audio_seconds = 0.0;
    let mut short_fixture_count = 0;
    let mut long_fixture_count = 0;

    for repetition in 0..repetitions {
        for input in inputs {
            let started = Instant::now();
            let result = asr.recognize_with_metrics(&input.samples, input.sample_rate)?;
            let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
            first_result_seconds.get_or_insert_with(|| run_started.elapsed().as_secs_f64());
            if input.audio_seconds >= f64::from(DEFAULT_LONG_REGIME_SECONDS) {
                long_wall_ms[repetition] += wall_ms;
                if repetition == 0 {
                    long_audio_seconds += input.audio_seconds;
                    long_fixture_count += 1;
                }
            } else {
                short_wall_ms[repetition] += wall_ms;
                if repetition == 0 {
                    short_audio_seconds += input.audio_seconds;
                    short_fixture_count += 1;
                }
            }
            decoded
                .get_mut(&input.file)
                .expect("decoded map was created from these inputs")
                .push(result);
            peak_resident_bytes = peak_resident_bytes.max(process_tree_resident_bytes(&asr)?);
        }
    }
    if short_fixture_count == 0 || long_fixture_count == 0 {
        bail!("gold corpus must exercise both short and long tuning regimes");
    }

    let report = asr_eval::evaluate(
        manifest,
        &decoded,
        RunMetadata {
            application_version: env!("CARGO_PKG_VERSION").to_string(),
            backend: asr.backend_metadata().clone(),
            operating_system: format!("macOS {}", hardware.macos_version),
            architecture: hardware.architecture.clone(),
            chip: hardware.chip.clone(),
            memory_bytes: hardware.memory_bytes,
            logical_cpus: hardware.logical_cpus,
            model_load_seconds: load_p50_ms / 1_000.0,
            warmup_seconds: warmup_ms / 1_000.0,
            first_result_seconds: first_result_seconds.context("candidate produced no result")?,
            peak_resident_bytes,
        },
    )?;
    let quality = QualityMeasurement {
        passed: report.passed,
        wer_percent: report.repeatability.wer_percent_max,
        cer_percent: report.repeatability.cer_percent_max,
        nondeterministic_outputs: report.repeatability.nondeterministic_outputs,
        categories: report
            .categories
            .iter()
            .map(|(name, score)| {
                (
                    name.clone(),
                    CategoryQuality {
                        wer_percent: score.wer_percent,
                        cer_percent: score.cer_percent,
                    },
                )
            })
            .collect(),
    };
    let short = regime_measurement(short_fixture_count, short_audio_seconds, &mut short_wall_ms);
    let long = regime_measurement(long_fixture_count, long_audio_seconds, &mut long_wall_ms);
    eprintln!(
        "  load p50 {:.1} ms, short p50 {:.1} ms, long p50 {:.1} ms, WER {:?}, CER {:?}, RSS {:.2} GiB",
        load_p50_ms,
        short.wall_p50_ms,
        long.wall_p50_ms,
        quality.wer_percent,
        quality.cer_percent,
        peak_resident_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    Ok(CandidateMeasurement {
        compute_units,
        status: CandidateStatus::Completed,
        first_observed_load_ms: Some(first_observed_load_ms),
        load_p50_ms: Some(load_p50_ms),
        warmup_ms: Some(warmup_ms),
        peak_resident_bytes: Some(peak_resident_bytes),
        quality: Some(quality),
        short: Some(short),
        long: Some(long),
    })
}

fn regime_measurement(
    fixture_count: usize,
    audio_seconds: f64,
    wall_ms: &mut [f64],
) -> RegimeMeasurement {
    let wall_p50_ms = percentile(wall_ms, 50);
    let wall_p95_ms = percentile(wall_ms, 95);
    RegimeMeasurement {
        fixture_count,
        audio_seconds,
        wall_p50_ms,
        wall_p95_ms,
        rtfx_p50: audio_seconds * 1_000.0 / wall_p50_ms,
    }
}

fn percentile(values: &mut [f64], percentile: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() * percentile).div_ceil(100)).saturating_sub(1);
    values[index]
}

fn process_tree_resident_bytes(asr: &Asr) -> Result<u64> {
    let parent = performance::resident_bytes(std::process::id())?;
    Ok(parent + asr.auxiliary_resident_bytes()?)
}
