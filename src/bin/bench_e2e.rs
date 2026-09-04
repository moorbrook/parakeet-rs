//! Matched end-of-speech-to-transcript benchmark through the production
//! capture, VAD, endpoint, and ASR path.
//!
//! A WAV fixture is played to a named duplex Core Audio device (normally
//! BlackHole 2ch) while `streamer` captures that same device. The output
//! stream emits silence after the fixture, so both endpoint strategies see
//! an identical acoustic boundary. No system audio defaults are changed.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use parakeet_dictation::asr::{Asr, AsrConfig};
use parakeet_dictation::asr_eval::normalize_lexical;
use parakeet_dictation::coreml_worker::{load_coreml_worker, CoreMlWorkerConfig};
use parakeet_dictation::endpointing::EndpointPolicy;
use parakeet_dictation::performance;
use parakeet_dictation::settings::SettingsStore;
use parakeet_dictation::streamer::{self, EndpointStrategy, Mode, Outcome};
use parakeet_dictation::warmup;
use parakeet_dictation::wav::read_wav_mono;

const DEFAULT_REPS: usize = 30;
const DEFAULT_WARMUP_REPS: usize = 2;
const DEFAULT_DEVICE: &str = "BlackHole 2ch";
const SILENCE_AMPLITUDE: f32 = 0.0001;

struct Args {
    wav: PathBuf,
    reps: usize,
    warmup_reps: usize,
    backend: Backend,
    strategy: EndpointStrategy,
    endpoint_policy: EndpointPolicy,
    /// Sweep override for the confirmation window.
    confirmation_ms: Option<u32>,
    /// Count false cuts and transcript mismatches instead of aborting on the
    /// first one. A rate needs every repetition, and restarting the process
    /// per repetition would reload and re-warm the Core ML worker.
    tolerate_false_cuts: bool,
    device: String,
    expected: Option<String>,
    worker: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    mode: BenchMode,
    arm: Arm,
    idle_gap_ms: u64,
    keepalive_ms: u64,
}

/// Which dictation UX the harness reproduces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchMode {
    /// Tap: Silero VAD owns the endpoint.
    VadAutoStop,
    /// Hold: the hotkey release owns the endpoint. The harness releases at the
    /// fixture's measured acoustic end, which is the earliest a user could.
    Hold,
}

/// Idle/keep-warm treatment applied before each repetition.
///
/// Unlike the `bench_asr` variant, the recording interval here is real: the
/// fixture plays through the loopback in the time it actually takes. So the
/// arms differ only in the idle gap before the press and in what runs between
/// the press and the release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    /// Repetitions back to back, as the existing Hold baseline runs them.
    Warm,
    /// `--idle-gap-ms` of silence before the press, and nothing after it.
    Cold,
    /// Idle gap, then one prime dispatch at the press edge.
    Prime,
    /// Idle gap, then a keep-alive dispatch every `--keepalive-ms` from the
    /// press until just before the release.
    Cadence,
}

impl Arm {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "warm" => Ok(Self::Warm),
            "cold" => Ok(Self::Cold),
            "prime" => Ok(Self::Prime),
            "cadence" => Ok(Self::Cadence),
            _ => bail!("unknown arm {value:?}; expected warm, cold, prime, or cadence"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Cold => "cold",
            Self::Prime => "prime",
            Self::Cadence => "cadence",
        }
    }
}

fn parse_mode(value: &str) -> anyhow::Result<BenchMode> {
    match value {
        "vad" => Ok(BenchMode::VadAutoStop),
        "hold" => Ok(BenchMode::Hold),
        _ => bail!("unknown mode {value:?}; expected vad or hold"),
    }
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
            _ => bail!("unknown backend {value:?}; expected sherpa or coreml-unified"),
        }
    }
}

fn parse_strategy(value: &str) -> anyhow::Result<EndpointStrategy> {
    match value {
        "serial" => Ok(EndpointStrategy::Serial),
        "speculative" => Ok(EndpointStrategy::Speculative),
        _ => bail!("unknown strategy {value:?}; expected serial or speculative"),
    }
}

fn parse_endpoint_policy(value: &str) -> anyhow::Result<EndpointPolicy> {
    match value {
        "fast" => Ok(EndpointPolicy::Fast),
        "long-form" => Ok(EndpointPolicy::LongForm),
        _ => bail!("unknown endpoint policy {value:?}; expected fast or long-form"),
    }
}

fn parse_args() -> anyhow::Result<Args> {
    let mut wav = None;
    let mut reps = DEFAULT_REPS;
    let mut warmup_reps = DEFAULT_WARMUP_REPS;
    let mut backend = Backend::Sherpa;
    let mut strategy = EndpointStrategy::Serial;
    let mut endpoint_policy = EndpointPolicy::LongForm;
    let mut confirmation_ms = None;
    let mut tolerate_false_cuts = false;
    let mut device = DEFAULT_DEVICE.to_string();
    let mut expected = None;
    let mut worker = None;
    let mut model_dir = None;
    let mut mode = BenchMode::VadAutoStop;
    let mut arm = Arm::Warm;
    let mut idle_gap_ms: u64 = 0;
    let mut keepalive_ms: u64 = 250;

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
            "--arm" => {
                arm = Arm::parse(&it.next().ok_or_else(|| anyhow!("--arm needs a name"))?)?;
            }
            "--idle-gap-ms" => {
                idle_gap_ms = it
                    .next()
                    .ok_or_else(|| anyhow!("--idle-gap-ms needs a number"))?
                    .parse()
                    .context("--idle-gap-ms")?;
            }
            "--keepalive-ms" => {
                keepalive_ms = it
                    .next()
                    .ok_or_else(|| anyhow!("--keepalive-ms needs a number"))?
                    .parse()
                    .context("--keepalive-ms")?;
            }
            "--strategy" => {
                strategy = parse_strategy(
                    &it.next()
                        .ok_or_else(|| anyhow!("--strategy needs a name"))?,
                )?;
            }
            "--endpoint-policy" => {
                endpoint_policy = parse_endpoint_policy(
                    &it.next()
                        .ok_or_else(|| anyhow!("--endpoint-policy needs a name"))?,
                )?;
            }
            "--confirmation-ms" => {
                confirmation_ms = Some(
                    it.next()
                        .ok_or_else(|| anyhow!("--confirmation-ms needs a number"))?
                        .parse()
                        .context("--confirmation-ms")?,
                );
            }
            "--tolerate-false-cuts" => {
                tolerate_false_cuts = true;
            }
            "--device" => {
                device = it.next().ok_or_else(|| anyhow!("--device needs a name"))?;
            }
            "--expected" => {
                expected = Some(
                    it.next()
                        .ok_or_else(|| anyhow!("--expected needs transcript text"))?,
                );
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
            "--mode" => {
                mode = parse_mode(&it.next().ok_or_else(|| anyhow!("--mode needs a name"))?)?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }

    Ok(Args {
        wav: wav.ok_or_else(|| anyhow!("--wav is required"))?,
        reps,
        warmup_reps,
        backend,
        strategy,
        endpoint_policy,
        confirmation_ms,
        tolerate_false_cuts,
        device,
        expected,
        worker,
        model_dir,
        mode,
        arm,
        idle_gap_ms,
        keepalive_ms,
    })
}

fn print_usage() {
    eprintln!(
        "usage: bench_e2e --wav PATH [--reps N] [--warmup-reps N]\n\
         \x20                [--backend sherpa|coreml-unified]\n\
         \x20                [--strategy serial|speculative]\n\
         \x20                [--endpoint-policy fast|long-form]\n\
         \x20                [--confirmation-ms N]\n\
         \x20                [--tolerate-false-cuts]\n\
         \x20                [--device 'BlackHole 2ch']\n\
         \x20                [--expected 'reference transcript']\n\
         \x20                [--worker PATH] [--model-dir DIR]\n\
         \x20                [--mode vad|hold]\n\
         \x20                [--arm warm|cold|prime|cadence]\n\
         \x20                [--idle-gap-ms N] [--keepalive-ms N]\n\n\
         Plays WAV through the named loopback device and measures the\n\
         production capture -> VAD -> ASR path. The device must expose\n\
         both input and output at the WAV sample rate.\n\
         \n\
         `--arm` selects the idle treatment. Every arm but `warm` sleeps\n\
         `--idle-gap-ms` before the press so the Neural Engine gates off;\n\
         `prime` then fires one dispatch at the press edge and `cadence`\n\
         re-primes every `--keepalive-ms` until just before the release."
    );
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error:#}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("bench_e2e failed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

impl Args {
    /// The session's silence window: the named policy, unless the sweep
    /// overrode it with a value no policy names.
    fn confirmation_ms(&self) -> u32 {
        self.confirmation_ms
            .unwrap_or_else(|| self.endpoint_policy.confirmation_ms())
    }
}

/// What one repetition produced. A false cut is a commit that landed before
/// playback reached the fixture's last audible sample.
#[derive(Clone, Copy, Debug, Default)]
struct RepResult {
    false_cut: bool,
    mismatch: bool,
}

fn run(args: &Args) -> anyhow::Result<()> {
    if args.arm == Arm::Cadence && args.keepalive_ms == 0 {
        bail!("--arm cadence needs a nonzero --keepalive-ms");
    }
    if args.arm == Arm::Cadence && args.mode != BenchMode::Hold {
        // VAD mode has no host-visible endpoint edge to stop the cadence at,
        // so a keep-alive would still hold the worker's pipe when the decode
        // starts and the measurement would include waiting for it. The Tap
        // cadence arm is `bench_asr --arm cadence`, which owns both edges.
        bail!("--arm cadence is only defined for --mode hold");
    }
    let store = SettingsStore::new()?;
    let asr = Arc::new(load_backend(args, &store)?);

    if args.backend == Backend::Sherpa {
        warmup::page_touch(&store.encoder_path())?;
    }
    warmup::dummy_decode(asr.as_ref())?;

    let (samples, sample_rate) = read_wav_mono(&args.wav)?;
    let last_non_silent = samples
        .iter()
        .rposition(|sample| sample.abs() >= SILENCE_AMPLITUDE)
        .ok_or_else(|| anyhow!("fixture contains no audible samples"))?;
    let replay_samples: Arc<[f32]> = Arc::from(&samples[..=last_non_silent]);
    let trimmed_samples = samples.len().saturating_sub(replay_samples.len());
    let trimmed_ms = trimmed_samples as f32 / sample_rate as f32 * 1_000.0;
    let audio_s = samples.len() as f32 / sample_rate as f32;
    log::info!(
        "loaded {} ({audio_s:.3}s mono @ {sample_rate} Hz); trimmed {trimmed_ms:.1}ms trailing silence; loopback={}",
        args.wav.display(),
        args.device
    );

    log::info!("endpoint config: confirmation_ms={}", args.confirmation_ms());

    for rep in 0..args.warmup_reps {
        run_one(
            args,
            &store,
            asr.clone(),
            replay_samples.clone(),
            sample_rate,
            rep,
            false,
        )?;
    }
    let mut false_cuts = 0_usize;
    let mut mismatches = 0_usize;
    // `scripts/bench-idle.py` attributes every phase_timer line that follows
    // this marker to the arm it names, which keeps the arm out of the shared
    // PhaseTimer format and out of `streamer.rs`.
    log::info!(
        "idle_arm arm={} idle_gap_ms={} record_gap_ms=audio keepalive_ms={}",
        args.arm.as_str(),
        args.idle_gap_ms,
        args.keepalive_ms
    );
    for rep in 0..args.reps {
        let result = run_one(
            args,
            &store,
            asr.clone(),
            replay_samples.clone(),
            sample_rate,
            rep,
            true,
        )?;
        false_cuts += usize::from(result.false_cut);
        mismatches += usize::from(result.mismatch);
    }
    log::info!(
        "bench_e2e_summary reps={} false_cuts={false_cuts} mismatches={mismatches} \
         confirmation_ms={}",
        args.reps,
        args.confirmation_ms()
    );
    Ok(())
}

fn load_backend(args: &Args, store: &SettingsStore) -> anyhow::Result<Asr> {
    match args.backend {
        Backend::Sherpa => {
            if !store.model_present() {
                bail!(
                    "ASR model not present at {}",
                    store.encoder_path().display()
                );
            }
            Asr::load(&AsrConfig {
                encoder: &store.encoder_path(),
                decoder: &store.decoder_path(),
                joiner: &store.joiner_path(),
                tokens: &store.tokens_path(),
                num_threads: performance::performance_core_count(),
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
            let (asr, load_seconds) = load_coreml_worker(&config)?;
            log::info!("Core ML worker ready in {load_seconds:.3}s");
            Ok(asr)
        }
    }
}

fn run_one(
    args: &Args,
    store: &SettingsStore,
    asr: Arc<Asr>,
    samples: Arc<[f32]>,
    sample_rate: u32,
    rep: usize,
    emit: bool,
) -> anyhow::Result<RepResult> {
    let streamer_mode = match args.mode {
        BenchMode::VadAutoStop => Mode::VadAutoStop,
        BenchMode::Hold => Mode::Manual,
    };
    // Only measured repetitions pay the idle gap: warmup reps exist to compile
    // the Core ML graph, and sleeping a minute in front of each one buys
    // nothing.
    if emit && args.arm != Arm::Warm {
        std::thread::sleep(Duration::from_millis(args.idle_gap_ms));
    }
    let (session, outcome_rx) = streamer::start_with_strategy_on_device(
        &store.vad_path(),
        streamer_mode,
        asr.clone(),
        args.strategy,
        args.confirmation_ms(),
        Some(&args.device),
    )?;
    // The press edge. The mic is open and the fixture has not started playing,
    // which is where `App::on_hotkey_press` would fire the prime.
    let mut keep_alive = None;
    if emit {
        match args.arm {
            Arm::Prime => warmup::prime_engine(&asr)?,
            Arm::Cadence => {
                keep_alive = Some(warmup::KeepAlive::start(
                    asr.clone(),
                    Duration::from_millis(args.keepalive_ms),
                ));
            }
            Arm::Warm | Arm::Cold => {}
        }
    }
    let playback = start_playback(&args.device, samples.clone(), sample_rate)?;
    let audio_s = samples.len() as f32 / sample_rate as f32;
    let timeout = Duration::from_secs_f32(audio_s + 15.0);

    // Hold mode has no VAD: the harness plays the fixture, waits for the
    // predicted instant of its last audible sample, and releases there. That
    // release is the endpoint, so `dur_end_to_end_ms` is exactly the
    // release-to-transcript-ready latency the user feels.
    let hold_release = if args.mode == BenchMode::Hold {
        let acoustic_end = wait_for_acoustic_end(&playback, timeout)
            .with_context(|| format!("waiting for playback end on repetition {rep}"))?;
        // Stop the cadence here rather than at the release: `stop` joins, and
        // the worker serializes on one pipe, so an in-flight keep-alive would
        // otherwise sit in front of the decode this repetition is measuring.
        // Joining during the remaining wait keeps it off the release edge.
        let keep_alive_dispatches = keep_alive.take().map(warmup::KeepAlive::stop);
        if let Some(remaining) = acoustic_end.checked_duration_since(Instant::now()) {
            std::thread::sleep(remaining);
        }
        if let Some(dispatches) = keep_alive_dispatches {
            log::info!(
                "idle_keepalive arm={} idle_gap_ms={} dispatches={dispatches}",
                args.arm.as_str(),
                args.idle_gap_ms
            );
        }
        let release = Instant::now();
        session.finalize();
        Some(release)
    } else {
        None
    };

    let outcome = outcome_rx
        .0
        .recv_timeout(timeout)
        .with_context(|| format!("waiting for endpoint on repetition {rep}"))?;
    let Outcome::Speech {
        samples,
        sample_rate,
        early_transcript,
        mut timer,
    } = outcome
    else {
        return match outcome {
            Outcome::Cancelled => bail!("repetition {rep} was cancelled"),
            Outcome::NoSpeech => bail!("repetition {rep} detected no speech"),
            Outcome::Error(error) => Err(error).context(format!("repetition {rep}")),
            Outcome::Speech { .. } => unreachable!("matched above"),
        };
    };

    // In Tap the acoustic-end marker only exists once playback has rendered
    // the fixture's last audible sample, so its absence *is* the false cut. A
    // marker read that fails outright is a harness fault and still aborts.
    let marker = match hold_release {
        Some(release) => Some(release),
        None => playback.acoustic_end_marker()?,
    };
    let acoustic_end = match marker {
        Some(end) => end,
        None if args.tolerate_false_cuts => {
            // The provisional transcript is the evidence for *why* it cut:
            // it shows how much of the utterance the decoder had when the
            // window elapsed, and whether the cut point was a sentence end.
            log::info!(
                "bench_e2e false_cut rep={rep} provisional={:?}",
                early_transcript.unwrap_or_default()
            );
            drop(session);
            // Let the fixture finish rendering so the next repetition starts
            // from silence rather than mid-utterance.
            wait_for_acoustic_end(&playback, timeout)
                .with_context(|| format!("draining playback after a false cut on rep {rep}"))?;
            drop(playback);
            return Ok(RepResult {
                false_cut: true,
                mismatch: false,
            });
        }
        None => {
            return Err(anyhow!(
                "playback ended before emitting the acoustic-end marker"
            ))
            .context(format!("repetition {rep}"))
        }
    };
    drop(playback);
    drop(session);

    let transcript = match early_transcript {
        Some(text) => text,
        None => {
            timer.mark_asr_start();
            let text = asr.recognize(&samples, sample_rate)?;
            timer.mark_asr_done();
            text
        }
    };

    // The app's default no-polish delivery is one synchronous CGEvent post.
    // The benchmark deliberately does not type into the user's focused app;
    // this marker is transcript-ready and excludes only that sub-ms OS post.
    std::hint::black_box(&transcript);
    timer.mark_speech_end_at_instant(acoustic_end);
    timer.mark_paste_done();
    if emit {
        timer.emit();
    }
    let mut mismatch = false;
    if let Some(expected) = &args.expected {
        let wanted = normalize_lexical(expected);
        let got = normalize_lexical(&transcript);
        if wanted != got {
            if !args.tolerate_false_cuts {
                bail!(
                    "repetition {rep} transcript mismatch: expected {expected:?}, got {transcript:?}"
                );
            }
            mismatch = true;
            log::info!("bench_e2e mismatch rep={rep} transcript={transcript:?}");
        }
    }
    log::info!(
        "bench_e2e rep={rep} measured={emit} strategy={:?} transcript={transcript:?}",
        args.strategy
    );
    Ok(RepResult {
        false_cut: false,
        mismatch,
    })
}

/// Block until the output stream reports the predicted instant of the
/// fixture's last audible sample. Polled rather than signalled because the
/// marker is written from the Core Audio render callback.
fn wait_for_acoustic_end(playback: &Playback, timeout: Duration) -> anyhow::Result<Instant> {
    const POLL: Duration = Duration::from_millis(2);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(end) = playback.acoustic_end() {
            return Ok(end);
        }
        std::thread::sleep(POLL);
    }
    bail!("playback never reported an acoustic-end marker")
}

struct Playback {
    _stream: cpal::Stream,
    acoustic_end: Arc<Mutex<Option<Instant>>>,
}

impl Playback {
    /// `None` means playback has not yet rendered the fixture's last audible
    /// sample. An `Err` is a harness fault, never a statement about the
    /// recording — keeping the two apart is what stops a poisoned mutex from
    /// being counted as a false cut.
    fn acoustic_end_marker(&self) -> anyhow::Result<Option<Instant>> {
        Ok(self
            .acoustic_end
            .lock()
            .map_err(|_| anyhow!("acoustic-end marker mutex poisoned"))?
            .as_ref()
            .copied())
    }

    fn acoustic_end(&self) -> anyhow::Result<Instant> {
        self.acoustic_end_marker()?
            .ok_or_else(|| anyhow!("playback ended before emitting the acoustic-end marker"))
    }
}

fn start_playback(
    device_name: &str,
    samples: Arc<[f32]>,
    sample_rate: u32,
) -> anyhow::Result<Playback> {
    let host = cpal::default_host();
    let device = host
        .output_devices()
        .context("enumerating output devices")?
        .find(|device| {
            device
                .name()
                .is_ok_and(|candidate| candidate == device_name)
        })
        .ok_or_else(|| anyhow!("output device not found: {device_name}"))?;
    let supported = device
        .default_output_config()
        .with_context(|| format!("default output config for {device_name}"))?;
    if supported.sample_rate().0 != sample_rate {
        bail!(
            "{device_name} output is {} Hz but WAV is {sample_rate} Hz",
            supported.sample_rate().0
        );
    }
    let format = supported.sample_format();
    let config = supported.config();
    let acoustic_end = Arc::new(Mutex::new(None));
    let stream = match format {
        SampleFormat::F32 => {
            build_output_stream::<f32>(&device, &config, samples, acoustic_end.clone())?
        }
        SampleFormat::I16 => {
            build_output_stream::<i16>(&device, &config, samples, acoustic_end.clone())?
        }
        SampleFormat::U16 => {
            build_output_stream::<u16>(&device, &config, samples, acoustic_end.clone())?
        }
        other => bail!("unsupported output sample format: {other:?}"),
    };
    stream.play().context("starting loopback playback")?;
    Ok(Playback {
        _stream: stream,
        acoustic_end,
    })
}

fn build_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<[f32]>,
    acoustic_end: Arc<Mutex<Option<Instant>>>,
) -> anyhow::Result<cpal::Stream>
where
    T: SizedSample + Sample + FromSample<f32>,
{
    let channels = usize::from(config.channels);
    let sample_rate = f64::from(config.sample_rate.0);
    let mut cursor = 0_usize;
    let error_callback = |error| log::error!("loopback output error: {error}");
    device
        .build_output_stream(
            config,
            move |output: &mut [T], info| {
                let callback_started = Instant::now();
                let timestamp = info.timestamp();
                let playback_delay = timestamp
                    .playback
                    .duration_since(&timestamp.callback)
                    .unwrap_or_default();
                let start_cursor = cursor;
                for frame in output.chunks_mut(channels) {
                    let value = samples.get(cursor).copied().unwrap_or(0.0);
                    cursor = cursor.saturating_add(1);
                    let value = T::from_sample(value);
                    for sample in frame {
                        *sample = value;
                    }
                }
                if start_cursor < samples.len() && cursor >= samples.len() {
                    let frames_to_end = samples.len().saturating_sub(start_cursor);
                    let within_buffer = Duration::from_secs_f64(frames_to_end as f64 / sample_rate);
                    if let Ok(mut marker) = acoustic_end.lock() {
                        *marker = callback_started.checked_add(playback_delay + within_buffer);
                    }
                }
            },
            error_callback,
            None,
        )
        .context("building loopback output stream")
}
