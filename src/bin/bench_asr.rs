//! Headless ASR bench harness.
//!
//! Loads a WAV file, runs it through `parakeet_dictation::asr::Asr` N times, and
//! emits one `phase_timer` line per iteration to stderr. `scripts/bench-latency.sh`
//! drives it across {1, 3, 5, 10, 20} s fixtures and `scripts/bench-aggregate.py`
//! reduces the log into p50/p95/p99 per length.
//!
//! Uses the same `SettingsStore` paths as the menu-bar app, so the model
//! must already be downloaded (launch Parakeet.app once and let the
//! first-run fetch finish). This binary does NOT request mic permissions,
//! touch the clipboard, or synthesize keystrokes — it isolates the ASR
//! decode cost so the bench number is comparable across runs.
//!
//! Usage:
//!   bench_asr --wav bench/audio/5s.wav --reps 30 [--warmup-reps 3]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use parakeet_dictation::asr::{Asr, AsrConfig, StageReport};
use parakeet_dictation::coreml_worker::{
    load_coreml_worker, CoreMlComputeUnits, CoreMlRnntEngine, CoreMlWorkerConfig,
};
use parakeet_dictation::performance::{self, next_session_id, PhaseTimer, PhaseTimerMode};
use parakeet_dictation::resample::{to_target_rate, TARGET_SAMPLE_RATE};
use parakeet_dictation::settings::SettingsStore;
use parakeet_dictation::warmup;
use parakeet_dictation::wav::read_wav_mono;

const DEFAULT_REPS: usize = 30;
const DEFAULT_WARMUP_REPS: usize = 3;

struct Args {
    wav: PathBuf,
    reps: usize,
    warmup_reps: usize,
    backend: Backend,
    worker: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    compute_units: CoreMlComputeUnits,
    stage_timings: bool,
    rnnt_engine: CoreMlRnntEngine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    Sherpa,
    CoreMlUnified,
}

impl Backend {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "sherpa" => Ok(Self::Sherpa),
            "coreml-unified" => Ok(Self::CoreMlUnified),
            _ => anyhow::bail!("unknown backend {value:?}; expected sherpa or coreml-unified"),
        }
    }
}

fn parse_args() -> anyhow::Result<Args> {
    use anyhow::{anyhow, bail, Context};
    let mut wav: Option<PathBuf> = None;
    let mut reps: usize = DEFAULT_REPS;
    let mut warmup_reps: usize = DEFAULT_WARMUP_REPS;
    let mut backend = Backend::Sherpa;
    let mut worker = None;
    let mut model_dir = None;
    let mut compute_units = CoreMlComputeUnits::default();
    let mut stage_timings = false;
    let mut rnnt_engine = CoreMlRnntEngine::default();

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--wav" => {
                wav = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--wav needs a path"))?,
                ));
            }
            "--reps" => {
                reps = it
                    .next()
                    .ok_or_else(|| anyhow!("--reps needs a number"))?
                    .parse()
                    .context("--reps")?;
            }
            "--warmup-reps" => {
                warmup_reps = it
                    .next()
                    .ok_or_else(|| anyhow!("--warmup-reps needs a number"))?
                    .parse()
                    .context("--warmup-reps")?;
            }
            "--backend" => {
                backend =
                    Backend::parse(&it.next().ok_or_else(|| anyhow!("--backend needs a name"))?)?;
            }
            "--worker" => {
                worker = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--worker needs a path"))?,
                ));
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--model-dir needs a path"))?,
                ));
            }
            "--compute-units" => {
                compute_units = CoreMlComputeUnits::parse(
                    &it.next()
                        .ok_or_else(|| anyhow!("--compute-units needs a name"))?,
                )?;
            }
            "--stage-timings" => {
                stage_timings = true;
            }
            "--rnnt-engine" => {
                rnnt_engine = CoreMlRnntEngine::parse(
                    &it.next()
                        .ok_or_else(|| anyhow!("--rnnt-engine needs a name"))?,
                )?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    let wav = wav.ok_or_else(|| anyhow!("--wav is required"))?;
    Ok(Args {
        wav,
        reps,
        warmup_reps,
        backend,
        worker,
        model_dir,
        compute_units,
        stage_timings,
        rnnt_engine,
    })
}

fn print_usage() {
    eprintln!(
        "usage: bench_asr --wav PATH [--reps N] [--warmup-reps N]\n\
         \x20                [--backend sherpa|coreml-unified]\n\
         \x20                [--worker PATH] [--model-dir DIR]\n\
         \x20                [--compute-units all|cpu-and-gpu|cpu-and-neural-engine|cpu-only]\n\
         \x20                [--stage-timings] [--rnnt-engine native|coreml]\n\
         \n\
         Runs the loaded Parakeet recognizer over WAV PATH `--reps` times,\n\
         emitting one `phase_timer` log line per iteration on stderr.\n\
         `--warmup-reps` decodes are run first and not recorded — they pay\n\
         the CoreML graph-compile cost so steady-state numbers are clean.\n\
         \n\
         The model must already be downloaded (launch Parakeet.app once\n\
         to trigger the first-run fetch)."
    );
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e:#}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    if let Err(e) = run(&args) {
        eprintln!("bench_asr failed: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: &Args) -> anyhow::Result<()> {
    let store = SettingsStore::new()?;
    let asr = load_backend(args, &store)?;

    // CoreML graph compile happens on first inference. The aggregator
    // ignores the warmup reps so steady-state numbers aren't contaminated.
    log::info!("warming recognizer (silent decode)");
    if args.backend == Backend::Sherpa {
        warmup::page_touch(&store.encoder_path())?;
    }
    warmup::dummy_decode(&asr)?;

    let (fixture, fixture_rate) = read_wav_mono(&args.wav)?;
    let audio_s = fixture.len() as f32 / fixture_rate as f32;
    // Convert once, outside the measured loop, because that is what production
    // does: `AudioCapture` resamples inside its capture callbacks, so
    // `Asr::recognize` is always handed 16 kHz mono. Timing the conversion here
    // would measure a step the endpoint path no longer has. ADR-0030.
    let samples = to_target_rate(&fixture, fixture_rate)?.into_owned();
    let sample_rate = TARGET_SAMPLE_RATE;
    log::info!(
        "loaded {} ({audio_s:.3}s mono @ {fixture_rate} Hz, {} samples); \
         converted to {sample_rate} Hz ({} samples) before the measured loop",
        args.wav.display(),
        fixture.len(),
        samples.len()
    );

    let stem = args
        .wav
        .file_stem()
        .map_or_else(|| "unknown".into(), |s| s.to_string_lossy().to_string());

    // Warmup reps: emit phase_timer lines but tagged so the aggregator
    // can drop them. Using session_id with a `warmup-` prefix keeps the
    // log file self-describing.
    for i in 0..args.warmup_reps {
        run_one(
            &asr,
            &samples,
            sample_rate,
            audio_s,
            &format!("warmup-{stem}-r{i:03}"),
        )?;
    }
    if let Some(stages) = asr.last_stage_report() {
        log::info!(
            "stage profiler active: compute units {}",
            stages.compute_units
        );
    }
    // Measured reps. session_id has no `warmup-` prefix → aggregator counts it.
    for i in 0..args.reps {
        run_one(
            &asr,
            &samples,
            sample_rate,
            audio_s,
            &format!("bench-{stem}-r{i:03}"),
        )?;
    }
    Ok(())
}

fn load_backend(args: &Args, store: &SettingsStore) -> anyhow::Result<Asr> {
    match args.backend {
        Backend::Sherpa => {
            if !store.model_present() {
                anyhow::bail!(
                    "ASR model not present at {}. Launch Parakeet.app once so it can \
                     download the first-run model bundle.",
                    store.encoder_path().display()
                );
            }
            let threads = performance::performance_core_count();
            log::info!("loading sherpa Asr (threads={threads}, provider=coreml)");
            // Latency bench: no contextual biasing, so the number stays
            // comparable to every previously-recorded run. `asr_diff
            // --vocabulary` is where the biased path gets exercised.
            Asr::load(&AsrConfig {
                encoder: &store.encoder_path(),
                decoder: &store.decoder_path(),
                joiner: &store.joiner_path(),
                tokens: &store.tokens_path(),
                num_threads: threads,
                hotwords: None,
                hotwords_score: 0.0,
            })
        }
        Backend::CoreMlUnified => {
            let mut config = CoreMlWorkerConfig::discover()?;
            if let Some(worker) = &args.worker {
                config.worker_path.clone_from(worker);
            }
            if let Some(model_dir) = &args.model_dir {
                config.set_existing_model_directory(model_dir);
            }
            config.set_compute_units(args.compute_units);
            config.set_emit_stage_timings(args.stage_timings);
            config.set_rnnt_engine(args.rnnt_engine);
            log::info!(
                "loading Core ML worker {} with {:?}",
                config.worker_path.display(),
                config.model_source
            );
            let (asr, load_seconds) = load_coreml_worker(&config)?;
            log::info!("Core ML worker ready in {load_seconds:.3}s");
            Ok(asr)
        }
    }
}

/// Reject a stage report that cannot describe the Parakeet Unified pipeline.
///
/// Installation failure is already loud, but per-stage failure is not: if a
/// future Core ML release moves an entry point the way the encoder's already
/// differs from the decoder's, that stage's dispatches stop being intercepted
/// and its cost silently reads as zero while every other number stays
/// plausible. The encoder is the exposed one, because losing it also drives
/// `windows` to zero, and the frame identity keeps holding.
///
/// A broken profile invalidates the run it belongs to, so this fails the bench
/// rather than warning into a log nobody reads.
fn validate_stage_report(stages: &StageReport) -> anyhow::Result<()> {
    if stages.encoder_calls == 0 {
        anyhow::bail!(
            "stage profiler recorded no encoder dispatches: its Core ML entry point was not \
             intercepted, so encoder and mel time are missing from every stage row. See \
             StageProfiler.install()."
        );
    }
    if stages.windows != stages.encoder_calls {
        anyhow::bail!(
            "stage profiler recorded {} windows against {} encoder calls; the offline path \
             runs exactly one encoder prediction per window",
            stages.windows,
            stages.encoder_calls
        );
    }
    // The decode loop runs either through Core ML or natively, never both, so
    // the frame identity is checked against whichever engine reported steps.
    if stages.decoder_calls != 0 && stages.native_decoder_steps != 0 {
        anyhow::bail!(
            "stage profiler recorded {} Core ML decoder calls and {} native decoder steps in \
             one utterance; the decode loop cannot have run on both engines",
            stages.decoder_calls,
            stages.native_decoder_steps
        );
    }
    let decoder_steps = stages.decoder_calls + stages.native_decoder_steps;
    let joint_steps = stages.joint_calls + stages.native_joint_steps;
    if decoder_steps < stages.windows {
        anyhow::bail!(
            "stage profiler recorded {decoder_steps} decoder steps against {} windows; the \
             greedy loop runs at least one prediction-network step per window",
            stages.windows
        );
    }
    if joint_steps < decoder_steps - stages.windows {
        anyhow::bail!(
            "stage profiler recorded {joint_steps} joint steps against {decoder_steps} decoder \
             steps over {} windows, which implies a negative decoded-frame count",
            stages.windows
        );
    }
    if stages.other_calls != 0 {
        anyhow::bail!(
            "stage profiler saw {} Core ML predictions it could not attribute to the encoder, \
             decoder, or joint; the pipeline's inputs changed and the stage split is no longer \
             trustworthy",
            stages.other_calls
        );
    }
    Ok(())
}

fn run_one(
    asr: &Asr,
    samples: &[f32],
    sample_rate: u32,
    audio_s: f32,
    session_label: &str,
) -> anyhow::Result<()> {
    // Combine the label with a unique counter so the timer log can be
    // grouped without collisions across runs.
    let sid = format!("{session_label}-{}", next_session_id());
    let mut t = PhaseTimer::start(PhaseTimerMode::Bench, sid.clone());
    // The WAV is already in hand; capture and VAD collapsed into t0.
    t.mark_capture_end(audio_s);
    t.mark_vad_endpoint();
    t.mark_asr_start();
    let wall_started = Instant::now();
    let decoded = asr.recognize_with_metrics(samples, sample_rate)?;
    let wall_seconds = wall_started.elapsed().as_secs_f64();
    let internal_seconds = f64::from(decoded.decode_seconds);
    let boundary_seconds = (wall_seconds - internal_seconds).max(0.0);
    log::info!(
        "asr_boundary session_id={sid} internal_ms={:.3} wall_ms={:.3} boundary_ms={:.3}",
        internal_seconds * 1_000.0,
        wall_seconds * 1_000.0,
        boundary_seconds * 1_000.0
    );
    t.mark_asr_done();
    // No paste in bench mode — mark it equal to asr_done so the
    // `dur_post_endpoint_ms` field cleanly reads as "ASR-only latency".
    t.mark_paste_done();
    t.emit();
    // Emitted after the clock stops: formatting this line is real work and
    // must not land inside the latency the bench publishes.
    if let Some(stages) = asr.last_stage_report() {
        validate_stage_report(&stages)?;
        log::info!(
            "asr_stages session_id={sid} audio_s={audio_s:.3} resample_ms={:.3} windows={} \
             encoder_calls={} decoder_calls={} joint_calls={} \
             native_decoder_steps={} native_joint_steps={} other_calls={} \
             mel_ms={:.3} encoder_ms={:.3} decode_loop_ms={:.3} \
             decode_loop_dispatch_ms={:.3} decoder_dispatch_ms={:.3} \
             joint_dispatch_ms={:.3} decode_loop_native_ms={:.3} \
             post_ms={:.3} total_ms={:.3} \
             boundary_ms={:.3} compute_units={}",
            stages.resample_ms,
            stages.windows,
            stages.encoder_calls,
            stages.decoder_calls,
            stages.joint_calls,
            stages.native_decoder_steps,
            stages.native_joint_steps,
            stages.other_calls,
            stages.mel_ms,
            stages.encoder_ms,
            stages.decode_loop_ms,
            stages.decode_loop_dispatch_ms,
            stages.decoder_dispatch_ms,
            stages.joint_dispatch_ms,
            stages.decode_loop_native_ms,
            stages.post_ms,
            stages.total_ms,
            boundary_seconds * 1_000.0,
            stages.compute_units.replace(' ', ","),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy() -> StageReport {
        StageReport {
            resample_ms: 22.9,
            windows: 1,
            encoder_calls: 1,
            decoder_calls: 35,
            joint_calls: 96,
            native_decoder_steps: 0,
            native_joint_steps: 0,
            other_calls: 0,
            mel_ms: 3.0,
            encoder_ms: 25.5,
            decode_loop_ms: 15.2,
            decode_loop_dispatch_ms: 14.5,
            decoder_dispatch_ms: 5.0,
            joint_dispatch_ms: 9.5,
            decode_loop_native_ms: 0.0,
            post_ms: 0.06,
            total_ms: 43.8,
            compute_units: "encoder=cpu-and-neural-engine decoder=cpu-only joint=cpu-only"
                .to_string(),
        }
    }

    #[test]
    fn a_healthy_report_passes() {
        validate_stage_report(&healthy()).expect("a complete pipeline must validate");
    }

    #[test]
    fn a_silently_unintercepted_encoder_fails() {
        // The exact shape a moved Core ML entry point produces: the encoder
        // vanishes, windows follows it to zero, and every other field stays
        // plausible. This is what the check exists for.
        let mut stages = healthy();
        stages.encoder_calls = 0;
        stages.windows = 0;
        stages.encoder_ms = 0.0;
        stages.mel_ms = 0.0;
        let error = validate_stage_report(&stages).expect_err("a missing encoder must fail");
        assert!(error.to_string().contains("no encoder dispatches"));
    }

    #[test]
    fn windows_must_match_encoder_calls() {
        let mut stages = healthy();
        stages.windows = 2;
        let error = validate_stage_report(&stages).expect_err("mismatched windows must fail");
        assert!(error.to_string().contains("2 windows"));
    }

    #[test]
    fn unattributed_predictions_fail() {
        let mut stages = healthy();
        stages.other_calls = 3;
        let error = validate_stage_report(&stages).expect_err("unattributed calls must fail");
        assert!(error.to_string().contains("could not attribute"));
    }

    #[test]
    fn fewer_decoder_calls_than_windows_fails() {
        let mut stages = healthy();
        stages.windows = 2;
        stages.encoder_calls = 2;
        stages.decoder_calls = 1;
        let error = validate_stage_report(&stages).expect_err("too few decoder calls must fail");
        assert!(error.to_string().contains("decoder steps"));
    }

    /// The native decode loop reports the same counts under the native fields
    /// and leaves the Core ML ones at zero.
    fn healthy_native() -> StageReport {
        let mut stages = healthy();
        stages.native_decoder_steps = stages.decoder_calls;
        stages.native_joint_steps = stages.joint_calls;
        stages.decoder_calls = 0;
        stages.joint_calls = 0;
        stages.decode_loop_native_ms = stages.decode_loop_dispatch_ms;
        stages.decode_loop_dispatch_ms = 0.0;
        stages.decoder_dispatch_ms = 0.0;
        stages.joint_dispatch_ms = 0.0;
        stages.compute_units =
            "encoder=cpu-and-neural-engine decoder=native joint=native".to_string();
        stages
    }

    #[test]
    fn a_natively_decoded_report_passes() {
        validate_stage_report(&healthy_native())
            .expect("a loop with no Core ML dispatches must validate");
    }

    #[test]
    fn a_report_claiming_both_engines_fails() {
        let mut stages = healthy_native();
        stages.decoder_calls = 35;
        let error =
            validate_stage_report(&stages).expect_err("both engines at once must fail");
        assert!(error.to_string().contains("cannot have run on both engines"));
    }

    #[test]
    fn fewer_native_steps_than_windows_fails() {
        let mut stages = healthy_native();
        stages.windows = 2;
        stages.encoder_calls = 2;
        stages.native_decoder_steps = 1;
        let error =
            validate_stage_report(&stages).expect_err("too few native steps must fail");
        assert!(error.to_string().contains("decoder steps"));
    }
}
