//! Explicit full Core ML runtime-plan tuner.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use parakeet_dictation::asr::{Asr, Decoded};
use parakeet_dictation::asr_eval::{self, DecodeMetadata, GoldManifest, RunMetadata};
use parakeet_dictation::asr_tuning::{
    self, CandidateMeasurement, CandidateStatus, CategoryQuality, EncoderCost, GateOutcome,
    GateVerdict, HardwareFingerprint, QualityMeasurement, RegimeMeasurement, SwitchCost,
    TuningProfile, WindowEvidence, WindowPlan, WindowTuning, STOCK_ENCODER_WINDOW_SECONDS,
};
use parakeet_dictation::coreml_worker::{
    load_coreml_worker, CoreMlComputeUnits, CoreMlWorkerConfig, DEFAULT_LONG_REGIME_SECONDS,
};
use parakeet_dictation::model_fetch;
use parakeet_dictation::performance;
use parakeet_dictation::resample::{to_target_rate, TARGET_SAMPLE_RATE};
use parakeet_dictation::settings::SettingsStore;
use parakeet_dictation::streamer::decode_hold_windowed;
use parakeet_dictation::warmup;
use parakeet_dictation::wav::read_wav_mono;
use parakeet_dictation::windows::HoldWindowConfig;
use sha2::{Digest, Sha256};

const DEFAULT_GOLD: &str = "bench/gold/manifest.json";
const DEFAULT_AUDIO_DIR: &str = "bench/gold/audio";
const DEFAULT_REPETITIONS: usize = 5;
const DEFAULT_LOAD_REPETITIONS: usize = 3;
const DEFAULT_PROBE_REPETITIONS: usize = 20;
/// Room for all three hosted buckets (about 1.7 GB installed), so by default
/// the tuner keeps a bucket only for speed and the budget binds only when set.
const DEFAULT_BUCKET_BUDGET_MB: u64 = 1_800;
/// Encoder probes: one short enough for the 2 s bucket, and one longer, run on
/// the stock encoder only, for the per-second cost of the audio itself.
const SHORT_PROBE_SECONDS: f64 = 1.5;
const LONG_PROBE_SECONDS: f64 = 7.5;
/// How long an encoder sits idle, with the engine kept busy on another, before
/// its switch cost is timed: past the knee of the idle re-wake sweep
/// (`bench/README.md`, "Cool-down knee"), as for an encoder last used a
/// recording ago.
const SWITCH_IDLE: std::time::Duration = std::time::Duration::from_secs(8);
/// Repetitions of each switch timing; each costs `SWITCH_IDLE`.
const SWITCH_REPETITIONS: usize = 3;

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
    probe_repetitions: usize,
    bucket_budget_bytes: u64,
    bucket_fetch: model_fetch::BucketFetch,
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
    let mut probe_repetitions = DEFAULT_PROBE_REPETITIONS;
    let mut bucket_budget_mb = DEFAULT_BUCKET_BUDGET_MB;
    let mut bucket_fetch = model_fetch::BucketFetch::Download;
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
            "--probe-repetitions" => {
                probe_repetitions = next_usize(&mut iterator, "--probe-repetitions")?;
            }
            "--bucket-budget-mb" => {
                bucket_budget_mb = next_usize(&mut iterator, "--bucket-budget-mb")? as u64;
            }
            "--no-bucket-download" => bucket_fetch = model_fetch::BucketFetch::VerifyOnly,
            "--show-profile" => action = exclusive_action(action, Action::Show)?,
            "--remove-profile" => action = exclusive_action(action, Action::Remove)?,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    if repetitions == 0 || load_repetitions == 0 || probe_repetitions == 0 {
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
        probe_repetitions,
        bucket_budget_bytes: bucket_budget_mb * 1_000_000,
        bucket_fetch,
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
         \x20               [--probe-repetitions N] [--bucket-budget-mb MB]\n\
         \x20               [--no-bucket-download]\n\
         \x20      tune_asr [--profile JSON] --show-profile\n\
         \x20      tune_asr [--profile JSON] --remove-profile\n\
         \n\
         Benchmarks every bounded Core ML compute-unit candidate on the\n\
         checked-in human-speech gold corpus, refuses quality/category/memory\n\
         regressions, and selects short and long regimes deterministically.\n\
         Then, under the chosen units, measures each encoder bucket's cost and\n\
         picks the bucket set (within the disk budget, default 1800 MB) and\n\
         Hold window whose gold output is byte-identical to the 15 s encoder\n\
         at 6 s and whose predicted release cost, averaged over utterances\n\
         of 0.5-20 s, is lowest. Every hosted\n\
         bucket must be installed or downloadable. Atomically caches the\n\
         complete evidence profile."
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
    // Compute units are tuned on the 15 s encoder alone, which the artifact
    // digest covers; the buckets are measured afterwards under the chosen
    // units (`tune_windows`).
    base_config.set_encoder_buckets(Vec::new());

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
    let mut profile = TuningProfile::new(hardware, artifact_digest, args.repetitions, candidates)?;

    let window_result = tune_windows(
        args,
        &base_config,
        &profile,
        &model_dir,
        &settings.vad_path(),
        &inputs,
    )
    .and_then(|tuning| profile.set_window_tuning(tuning));
    asr_tuning::save(&profile_path, &profile)?;
    println!("{}", serde_json::to_string_pretty(&profile)?);
    eprintln!("saved {}", profile_path.display());
    print_plan(&profile);
    window_result
        .context("bucket and Hold-window tuning failed; the profile keeps every bucket at 6 s")
}

fn print_plan(profile: &TuningProfile) {
    let plan = WindowPlan::for_profile(profile, model_fetch::ENCODER_BUCKETS);
    eprintln!(
        "the app will use compute units short={} long={}, encoder buckets {:?}, Hold window {} s",
        profile.selection.short.compute_units.as_str(),
        profile.selection.long.compute_units.as_str(),
        plan.encoder_buckets,
        plan.hold_window_seconds
    );
    if let Some(tuning) = &profile.window_tuning {
        eprintln!(
            "predicted release ASR {:.1} ms against {:.1} ms for the 15 s encoder at 6 s",
            tuning.selection.predicted_release_ms, tuning.selection.baseline_predicted_release_ms
        );
    }
}

/// Measure every encoder under the profile's compute units, then gate bucket
/// subsets and Hold windows on the gold corpus, fastest predicted first.
fn tune_windows(
    args: &Args,
    base_config: &CoreMlWorkerConfig,
    profile: &TuningProfile,
    model_dir: &Path,
    vad_path: &Path,
    inputs: &[AudioInput],
) -> Result<WindowTuning> {
    if !vad_path.is_file() {
        bail!(
            "Hold windowing needs the Silero model at {}; launch the app once to fetch it",
            vad_path.display()
        );
    }
    let hosted = model_fetch::all_encoder_bucket_windows();
    let installed = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(model_fetch::ensure_encoder_buckets(
            model_dir,
            &hosted,
            args.bucket_fetch,
            Arc::new(|_| {}),
        ));
    if installed != hosted {
        bail!(
            "every hosted bucket {hosted:?} must be measured, but only {installed:?} verified in {}",
            model_dir.display()
        );
    }

    // The gate runs the app's own runtime plan, so its outputs are the ones
    // the app will produce.
    let mut config = base_config.clone();
    config.set_regime_compute_units(
        profile.selection.short.compute_units,
        profile.selection.long.compute_units,
        profile.long_regime_seconds,
    )?;
    // Costs are measured under the short-regime units alone, which every Hold
    // request at every candidate window runs under (checked here). Under a
    // split plan the 9 s stock switch probe would otherwise reach the
    // long-regime manager, which no Hold request uses.
    if !asr_tuning::hold_requests_use_short_units(
        profile.selection.short.compute_units,
        profile.selection.long.compute_units,
        profile.long_regime_seconds,
    ) {
        bail!(
            "compute units short={} long={} with a {} s long regime would run some Hold \
             windows on the long-regime units, which the window tuner does not measure",
            profile.selection.short.compute_units.as_str(),
            profile.selection.long.compute_units.as_str(),
            profile.long_regime_seconds
        );
    }
    let mut measure_config = base_config.clone();
    measure_config.set_compute_units(profile.selection.short.compute_units);
    let probe_source = inputs
        .iter()
        .max_by(|a, b| a.audio_seconds.total_cmp(&b.audio_seconds))
        .context("gold corpus is empty")?;
    // The stock switch probe must outlast the widest bucket, or it would run
    // on that bucket and be recorded as the stock encoder.
    let widest_probe_seconds = hosted
        .iter()
        .map(|window| f64::from(*window) + 1.0)
        .fold(LONG_PROBE_SECONDS, f64::max);
    if probe_source.audio_seconds < widest_probe_seconds {
        bail!(
            "the longest gold fixture is {:.2} s; the encoder probes need {widest_probe_seconds} s",
            probe_source.audio_seconds
        );
    }
    let short_probe = probe_slice(probe_source, SHORT_PROBE_SECONDS);
    let long_probe = probe_slice(probe_source, LONG_PROBE_SECONDS);

    let mut encoders = Vec::with_capacity(hosted.len() + 1);
    eprintln!("measuring the {STOCK_ENCODER_WINDOW_SECONDS} s encoder…");
    let stock = measure_encoder(
        &measure_config,
        &[],
        &[&short_probe, &long_probe],
        args.probe_repetitions,
    )?;
    let decode_ms_per_audio_second = ((stock[1].other_ms - stock[0].other_ms)
        / (LONG_PROBE_SECONDS - SHORT_PROBE_SECONDS))
        .max(0.0);
    eprintln!(
        "  encoder p50 {:.2} ms, audio cost {decode_ms_per_audio_second:.2} ms/s",
        stock[0].encoder_ms
    );
    encoders.push(EncoderCost {
        window_seconds: STOCK_ENCODER_WINDOW_SECONDS,
        encoder_ms_p50: stock[0].encoder_ms,
        archive_sha256: None,
        installed_bytes: 0,
    });
    for bucket in model_fetch::ENCODER_BUCKETS {
        eprintln!("measuring the {} s bucket…", bucket.seconds);
        let measured = measure_encoder(
            &measure_config,
            &[bucket.seconds],
            &[&short_probe],
            args.probe_repetitions,
        )?;
        eprintln!("  encoder p50 {:.2} ms", measured[0].encoder_ms);
        encoders.push(EncoderCost {
            window_seconds: bucket.seconds,
            encoder_ms_p50: measured[0].encoder_ms,
            archive_sha256: Some(bucket.archive_sha256().to_string()),
            installed_bytes: bucket.installed_bytes(),
        });
    }

    eprintln!("measuring encoder switches…");
    let switch_costs =
        measure_switches(&measure_config, &hosted, probe_source, SWITCH_REPETITIONS)?;
    for cost in &switch_costs {
        eprintln!(
            "  {} s -> {} s: +{:.2} ms",
            cost.from_window, cost.to_window, cost.extra_ms_p50
        );
    }

    let mut gate = Gate::new(config, inputs, vad_path, args.repetitions);
    eprintln!(
        "gold gate: no bucket at {} s…",
        asr_tuning::BASELINE_HOLD_WINDOW_SECONDS
    );
    let baseline_output_sha256 = match gate.outcome(&[], asr_tuning::BASELINE_HOLD_WINDOW_SECONDS) {
        GateOutcome::Output { sha256 } => sha256,
        other => bail!("the baseline itself is not a stable reference: {other:?}"),
    };
    let mut evidence = WindowEvidence {
        disk_budget_bytes: args.bucket_budget_bytes,
        hold_window_candidates: asr_tuning::HOLD_WINDOW_CANDIDATES.to_vec(),
        workload_seconds: asr_tuning::release_workload_seconds(),
        encoders,
        decode_ms_per_audio_second,
        switch_costs,
        prime_seconds: warmup::PRIME_SECONDS,
        baseline_output_sha256,
        gates: Vec::new(),
    };
    for candidate in asr_tuning::rank_window_candidates(&evidence)? {
        if candidate.is_baseline() {
            break;
        }
        eprintln!(
            "gold gate: buckets {:?} at {} s, predicted {:.1} ms…",
            candidate.encoder_buckets,
            candidate.hold_window_seconds,
            candidate.predicted_release_ms
        );
        let outcome = gate.outcome(&candidate.encoder_buckets, candidate.hold_window_seconds);
        let passed = matches!(
            &outcome,
            GateOutcome::Output { sha256 } if *sha256 == evidence.baseline_output_sha256
        );
        eprintln!(
            "  {}",
            if passed {
                "identical".to_string()
            } else {
                format!("{outcome:?}")
            }
        );
        evidence.gates.push(GateVerdict {
            encoder_buckets: candidate.encoder_buckets,
            hold_window_seconds: candidate.hold_window_seconds,
            outcome,
        });
        if passed {
            break;
        }
    }
    WindowTuning::new(evidence)
}

/// The first `seconds` of a fixture, at the rate the app hands the worker.
fn probe_slice(input: &AudioInput, seconds: f64) -> Vec<f32> {
    let length = (seconds * f64::from(TARGET_SAMPLE_RATE)).round() as usize;
    input.samples[..length.min(input.samples.len())].to_vec()
}

struct ProbeCost {
    encoder_ms: f64,
    /// Wall time outside the encoder: mel, decode and the request itself.
    other_ms: f64,
}

/// Extra wall time of a request on an encoder that sat idle while the engine
/// kept running another, for every ordered pair of the stock encoder and the
/// hosted buckets, all loaded together as the app loads them. That is a Hold
/// tail whose encoder is not the one the keydown primer or the background
/// windows ran: the engine is awake, but this encoder's program last ran a
/// recording ago. Measured as the target after `SWITCH_IDLE` of the other
/// encoder every half second, against the target after the same of itself.
/// Back to back, with no idle time, a switch measured within 1 ms.
fn measure_switches(
    config: &CoreMlWorkerConfig,
    buckets: &[u32],
    source: &AudioInput,
    repetitions: usize,
) -> Result<Vec<SwitchCost>> {
    let widest = buckets.iter().copied().max().unwrap_or(0);
    let mut windows = buckets.to_vec();
    windows.push(STOCK_ENCODER_WINDOW_SECONDS);
    let probes: Vec<(u32, Vec<f32>)> = windows
        .iter()
        .map(|&window| {
            // Half a second inside a bucket; a second past the widest bucket
            // for the stock encoder, as the warm-up does.
            let seconds = if window == STOCK_ENCODER_WINDOW_SECONDS {
                f64::from(widest) + 1.0
            } else {
                f64::from(window) - 0.5
            };
            (window, probe_slice(source, seconds))
        })
        .collect();
    let mut config = config.clone();
    config.set_encoder_buckets(buckets.to_vec());
    let (asr, _) = load_coreml_worker(&config)?;
    warmup::dummy_decode(&asr)?;
    warmup::warm_encoder_buckets(&asr, buckets)?;
    // The target after SWITCH_IDLE of `keeper` every half second.
    let after = |keeper: &[f32], target: &[f32]| -> Result<f64> {
        asr.recognize_with_metrics(target, TARGET_SAMPLE_RATE)?;
        let idle_started = Instant::now();
        while idle_started.elapsed() < SWITCH_IDLE {
            std::thread::sleep(std::time::Duration::from_millis(500));
            asr.recognize_with_metrics(keeper, TARGET_SAMPLE_RATE)?;
        }
        let started = Instant::now();
        asr.recognize_with_metrics(target, TARGET_SAMPLE_RATE)?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0)
    };
    let mut costs = Vec::new();
    for (to_window, to_probe) in &probes {
        let mut same = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            same.push(after(to_probe, to_probe)?);
        }
        let same_ms = percentile(&mut same, 50);
        for (from_window, from_probe) in &probes {
            if from_window == to_window {
                continue;
            }
            let mut switched = Vec::with_capacity(repetitions);
            for _ in 0..repetitions {
                switched.push(after(from_probe, to_probe)?);
            }
            costs.push(SwitchCost {
                from_window: *from_window,
                to_window: *to_window,
                extra_ms_p50: (percentile(&mut switched, 50) - same_ms).max(0.0),
            });
        }
    }
    Ok(costs)
}

/// p50 encoder and non-encoder cost of each probe on a worker loaded with
/// exactly `buckets`, so a probe the bucket holds runs on it.
fn measure_encoder(
    config: &CoreMlWorkerConfig,
    buckets: &[u32],
    probes: &[&Vec<f32>],
    repetitions: usize,
) -> Result<Vec<ProbeCost>> {
    let mut config = config.clone();
    config.set_encoder_buckets(buckets.to_vec());
    config.set_emit_stage_timings(true);
    let (asr, _) = load_coreml_worker(&config)?;
    warmup::dummy_decode(&asr)?;
    warmup::warm_encoder_buckets(&asr, buckets)?;
    probes
        .iter()
        .map(|probe| {
            let mut encoder_ms = Vec::with_capacity(repetitions);
            let mut other_ms = Vec::with_capacity(repetitions);
            // Two unrecorded runs settle the engine on this shape.
            for repetition in 0..repetitions + 2 {
                let started = Instant::now();
                asr.recognize_with_metrics(probe, TARGET_SAMPLE_RATE)?;
                let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
                let stages = asr
                    .last_stage_report()
                    .context("the worker returned no stage timings")?;
                if stages.encoder_calls != 1 {
                    bail!(
                        "a {:.1} s probe ran {} encoder calls; expected one",
                        probe.len() as f64 / f64::from(TARGET_SAMPLE_RATE),
                        stages.encoder_calls
                    );
                }
                if repetition >= 2 {
                    encoder_ms.push(stages.encoder_ms);
                    other_ms.push(wall_ms - stages.encoder_ms);
                }
            }
            Ok(ProbeCost {
                encoder_ms: percentile(&mut encoder_ms, 50),
                other_ms: percentile(&mut other_ms, 50),
            })
        })
        .collect()
}

/// Each fixture's output, or the outcome when repetitions disagreed.
type StableOutputs = Result<BTreeMap<String, String>, GateOutcome>;

/// The worker for one bucket set and its whole-utterance outputs, which no
/// Hold window changes.
struct LoadedSet {
    buckets: Vec<u32>,
    asr: Asr,
    whole: StableOutputs,
}

/// Gold outputs per bucket set, keeping one worker loaded at a time.
struct Gate<'a> {
    config: CoreMlWorkerConfig,
    inputs: &'a [AudioInput],
    vad_path: &'a Path,
    repetitions: usize,
    loaded: Option<LoadedSet>,
}

impl<'a> Gate<'a> {
    fn new(
        config: CoreMlWorkerConfig,
        inputs: &'a [AudioInput],
        vad_path: &'a Path,
        repetitions: usize,
    ) -> Self {
        Self {
            config,
            inputs,
            vad_path,
            repetitions,
            loaded: None,
        }
    }

    fn outcome(&mut self, buckets: &[u32], hold_window_seconds: f32) -> GateOutcome {
        match self.try_outcome(buckets, hold_window_seconds) {
            Ok(outcome) => outcome,
            Err(error) => GateOutcome::Failed {
                error: format!("{error:#}"),
            },
        }
    }

    fn try_outcome(&mut self, buckets: &[u32], hold_window_seconds: f32) -> Result<GateOutcome> {
        if self.loaded.as_ref().map(|set| set.buckets.as_slice()) != Some(buckets) {
            self.loaded = None;
            let mut config = self.config.clone();
            config.set_encoder_buckets(buckets.to_vec());
            let (asr, _) = load_coreml_worker(&config)?;
            warmup::dummy_decode(&asr)?;
            warmup::warm_encoder_buckets(&asr, buckets)?;
            let whole = stable_outputs(self.inputs, self.repetitions, |input| {
                asr.recognize(&input.samples, TARGET_SAMPLE_RATE)
            })?;
            self.loaded = Some(LoadedSet {
                buckets: buckets.to_vec(),
                asr,
                whole,
            });
        }
        let LoadedSet { asr, whole, .. } = self.loaded.as_ref().expect("loaded above");
        let whole = match whole {
            Ok(outputs) => outputs.clone(),
            Err(outcome) => return Ok(outcome.clone()),
        };
        let config = HoldWindowConfig {
            enabled: true,
            min_seconds: hold_window_seconds,
            max_seconds: hold_window_seconds,
        };
        let hold = match stable_outputs(self.inputs, self.repetitions, |input| {
            decode_hold_windowed(asr, self.vad_path, &input.samples, config)
        })? {
            Ok(outputs) => outputs,
            Err(outcome) => return Ok(outcome),
        };
        Ok(GateOutcome::Output {
            sha256: output_digest(whole, hold)?,
        })
    }
}

/// SHA-256 of one configuration's gold outputs, whole-utterance and
/// Hold-windowed, as canonical JSON.
fn output_digest(
    whole: BTreeMap<String, String>,
    hold: BTreeMap<String, String>,
) -> Result<String> {
    let bytes = serde_json::to_vec(&BTreeMap::from([("hold", hold), ("whole", whole)]))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Each fixture's output when every repetition agrees, else the outcome that
/// says they did not.
fn stable_outputs(
    inputs: &[AudioInput],
    repetitions: usize,
    decode: impl Fn(&AudioInput) -> Result<String>,
) -> Result<StableOutputs> {
    let mut outputs = BTreeMap::new();
    for repetition in 0..repetitions {
        for input in inputs {
            let text = decode(input)?;
            if repetition == 0 {
                outputs.insert(input.file.clone(), text);
            } else if outputs.get(&input.file) != Some(&text) {
                return Ok(Err(GateOutcome::Nondeterministic));
            }
        }
    }
    Ok(Ok(outputs))
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
    // The app's own check, so this says whether the app will use the profile.
    match profile.validate_for(
        &HardwareFingerprint::current()?,
        &model_fetch::coreml_artifact_digest(),
    ) {
        Ok(()) => print_plan(&profile),
        Err(error) => eprintln!("the app will ignore this profile and run untuned: {error:#}"),
    }
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
            let (fixture_samples, fixture_rate) = read_wav_mono(&path)?;
            // What production hands the recogniser: 16 kHz mono from the
            // project's resampler (ADR-0030), not the corpus's 48 kHz.
            let samples = to_target_rate(&fixture_samples, fixture_rate)?.into_owned();
            let sample_rate = TARGET_SAMPLE_RATE;
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
            decoding: DecodeMetadata {
                method: "coreml_unified_greedy".to_string(),
                contextual_vocabulary_requested: false,
                contextual_vocabulary_active: false,
                hotword_score: None,
                vocabulary_terms_requested: 0,
                vocabulary_sha256: None,
                generated_hotwords_sha256: None,
            },
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn fixture(file: &str) -> AudioInput {
        AudioInput {
            file: file.into(),
            samples: Vec::new(),
            sample_rate: TARGET_SAMPLE_RATE,
            audio_seconds: 1.0,
        }
    }

    #[test]
    fn a_change_on_a_later_repetition_is_nondeterministic() {
        let inputs = [fixture("a.wav"), fixture("b.wav")];
        let calls = Cell::new(0);
        let outputs = stable_outputs(&inputs, 3, |input| {
            calls.set(calls.get() + 1);
            // The fifth decode is b.wav on the third repetition.
            Ok(if calls.get() == 5 {
                "drift".into()
            } else {
                input.file.clone()
            })
        })
        .unwrap();
        assert_eq!(outputs, Err(GateOutcome::Nondeterministic));

        let outputs = stable_outputs(&inputs, 3, |input| Ok(input.file.clone())).unwrap();
        assert_eq!(
            outputs,
            Ok(BTreeMap::from([
                ("a.wav".to_string(), "a.wav".to_string()),
                ("b.wav".to_string(), "b.wav".to_string()),
            ]))
        );
    }

    #[test]
    fn a_decode_failure_is_an_error_not_an_output() {
        let inputs = [fixture("a.wav")];
        let calls = Cell::new(0);
        let result = stable_outputs(&inputs, 2, |_| {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                bail!("worker exited")
            }
            Ok("text".into())
        });
        assert!(result.is_err());
    }

    #[test]
    fn the_digest_covers_both_whole_and_hold_outputs() {
        let map = |text: &str| BTreeMap::from([("a.wav".to_string(), text.to_string())]);
        // printf '%s' '{"hold":{"a.wav":"hold text"},"whole":{"a.wav":"whole text"}}' | shasum -a 256
        assert_eq!(
            output_digest(map("whole text"), map("hold text")).unwrap(),
            "cbfac7369418b7afa95d671b2c7367f949fb8b2bc18d8947552258e90f5cfc8b"
        );
        let reference = output_digest(map("whole text"), map("hold text")).unwrap();
        assert_ne!(
            output_digest(map("whole text!"), map("hold text")).unwrap(),
            reference
        );
        assert_ne!(
            output_digest(map("whole text"), map("hold text!")).unwrap(),
            reference
        );
    }
}
