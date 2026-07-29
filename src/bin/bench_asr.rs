//! Headless ASR bench harness.
//!
//! Loads a WAV file, runs it through `parakeet_rs::asr::Asr` N times, and
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

use parakeet_rs::asr::{Asr, AsrConfig};
use parakeet_rs::performance::{self, next_session_id, PhaseTimer, PhaseTimerMode};
use parakeet_rs::settings::SettingsStore;
use parakeet_rs::warmup;
use parakeet_rs::wav::read_wav_mono;

const DEFAULT_REPS: usize = 30;
const DEFAULT_WARMUP_REPS: usize = 3;

struct Args {
    wav: PathBuf,
    reps: usize,
    warmup_reps: usize,
}

fn parse_args() -> anyhow::Result<Args> {
    use anyhow::{anyhow, bail, Context};
    let mut wav: Option<PathBuf> = None;
    let mut reps: usize = DEFAULT_REPS;
    let mut warmup_reps: usize = DEFAULT_WARMUP_REPS;

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
    })
}

fn print_usage() {
    eprintln!(
        "usage: bench_asr --wav PATH [--reps N] [--warmup-reps N]\n\
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
    if !store.model_present() {
        anyhow::bail!(
            "ASR model not present at {}. Launch Parakeet.app once so it can \
             download the first-run model bundle.",
            store.encoder_path().display()
        );
    }

    let threads = performance::performance_core_count();
    log::info!("loading Asr (threads={threads}, provider=coreml)");
    // Latency bench: no contextual biasing, so the number stays
    // comparable to every previously-recorded run. `asr_diff
    // --vocabulary` is where the biased path gets exercised.
    let asr = Asr::load(&AsrConfig {
        encoder: &store.encoder_path(),
        decoder: &store.decoder_path(),
        joiner: &store.joiner_path(),
        tokens: &store.tokens_path(),
        num_threads: threads,
        hotwords: None,
        hotwords_score: 0.0,
    })?;

    // CoreML graph compile happens on first inference. The aggregator
    // ignores the warmup reps so steady-state numbers aren't contaminated.
    log::info!("warming recognizer (page-touch + silent decode)");
    warmup::page_touch(&store.encoder_path())?;
    warmup::dummy_decode(&asr)?;

    let (samples, sample_rate) = read_wav_mono(&args.wav)?;
    let audio_s = samples.len() as f32 / sample_rate as f32;
    log::info!(
        "loaded {} ({audio_s:.3}s mono @ {sample_rate} Hz, {} samples)",
        args.wav.display(),
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
    let mut t = PhaseTimer::start(PhaseTimerMode::Bench, sid);
    // The WAV is already in hand; capture and VAD collapsed into t0.
    t.mark_capture_end(audio_s);
    t.mark_vad_endpoint();
    t.mark_asr_start();
    let _decoded = asr.recognize(samples, sample_rate)?;
    t.mark_asr_done();
    // No paste in bench mode — mark it equal to asr_done so the
    // `dur_post_endpoint_ms` field cleanly reads as "ASR-only latency".
    t.mark_paste_done();
    t.emit();
    Ok(())
}
