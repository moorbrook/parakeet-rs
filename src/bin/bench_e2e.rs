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
use parakeet_rs::asr::{Asr, AsrConfig};
use parakeet_rs::asr_eval::normalize_lexical;
use parakeet_rs::coreml_worker::{load_coreml_worker, CoreMlWorkerConfig};
use parakeet_rs::performance;
use parakeet_rs::settings::SettingsStore;
use parakeet_rs::streamer::{self, EndpointStrategy, Mode, Outcome};
use parakeet_rs::warmup;
use parakeet_rs::wav::read_wav_mono;

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
    device: String,
    expected: Option<String>,
    worker: Option<PathBuf>,
    model_dir: Option<PathBuf>,
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

fn parse_args() -> anyhow::Result<Args> {
    let mut wav = None;
    let mut reps = DEFAULT_REPS;
    let mut warmup_reps = DEFAULT_WARMUP_REPS;
    let mut backend = Backend::Sherpa;
    let mut strategy = EndpointStrategy::Serial;
    let mut device = DEFAULT_DEVICE.to_string();
    let mut expected = None;
    let mut worker = None;
    let mut model_dir = None;

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
            "--strategy" => {
                strategy = parse_strategy(
                    &it.next()
                        .ok_or_else(|| anyhow!("--strategy needs a name"))?,
                )?;
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
        device,
        expected,
        worker,
        model_dir,
    })
}

fn print_usage() {
    eprintln!(
        "usage: bench_e2e --wav PATH [--reps N] [--warmup-reps N]\n\
         \x20                [--backend sherpa|coreml-unified]\n\
         \x20                [--strategy serial|speculative]\n\
         \x20                [--device 'BlackHole 2ch']\n\
         \x20                [--expected 'reference transcript']\n\
         \x20                [--worker PATH] [--model-dir DIR]\n\n\
         Plays WAV through the named loopback device and measures the\n\
         production capture -> VAD -> ASR path. The device must expose\n\
         both input and output at the WAV sample rate."
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

fn run(args: &Args) -> anyhow::Result<()> {
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
    for rep in 0..args.reps {
        run_one(
            args,
            &store,
            asr.clone(),
            replay_samples.clone(),
            sample_rate,
            rep,
            true,
        )?;
    }
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
) -> anyhow::Result<()> {
    let (session, outcome_rx) = streamer::start_with_strategy_on_device(
        &store.vad_path(),
        Mode::VadAutoStop,
        asr.clone(),
        args.strategy,
        Some(&args.device),
    )?;
    let playback = start_playback(&args.device, samples.clone(), sample_rate)?;
    let audio_s = samples.len() as f32 / sample_rate as f32;
    let timeout = Duration::from_secs_f32(audio_s + 15.0);
    let outcome = outcome_rx
        .0
        .recv_timeout(timeout)
        .with_context(|| format!("waiting for endpoint on repetition {rep}"))?;
    let acoustic_end = playback.acoustic_end()?;
    drop(playback);
    drop(session);

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
    if let Some(expected) = &args.expected {
        let wanted = normalize_lexical(expected);
        let got = normalize_lexical(&transcript);
        if wanted != got {
            bail!(
                "repetition {rep} transcript mismatch: expected {expected:?}, got {transcript:?}"
            );
        }
    }
    log::info!(
        "bench_e2e rep={rep} measured={emit} strategy={:?} transcript={transcript:?}",
        args.strategy
    );
    Ok(())
}

struct Playback {
    _stream: cpal::Stream,
    acoustic_end: Arc<Mutex<Option<Instant>>>,
}

impl Playback {
    fn acoustic_end(&self) -> anyhow::Result<Instant> {
        self.acoustic_end
            .lock()
            .map_err(|_| anyhow!("acoustic-end marker mutex poisoned"))?
            .as_ref()
            .copied()
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
