//! The Hold loopback arm, with `bench_e2e --mode hold`'s release definition:
//! play the fixture (trailing silence trimmed) through the loopback device,
//! release at Core Audio's predicted instant of its last audible sample, and
//! stop the clock when the transcript is ready. Lines go out through
//! `PhaseTimer::emit`, so `scripts/bench-aggregate.py` reads both arms alike.
//!
//! `offline` drives the Hex engine through the app's own streamer, capture
//! and release (`Mode::Manual`); with no token spans the streamer decodes the
//! whole recording once at release, like our serial arm. `stream` is Hex's
//! Metal dictation path: the capture feeds a live stream on an engine thread
//! during the hold, and the release feeds the rest and finalizes. Its capture
//! is a plain cpal input on the same device, downmixed and resampled by the
//! app's `resample::Resampler` exactly as `AudioCapture` does.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use parakeet_dictation::asr::Asr;
use parakeet_dictation::endpointing::EndpointPolicy;
use parakeet_dictation::performance::{self, PhaseTimer, PhaseTimerMode};
use parakeet_dictation::resample::Resampler;
use parakeet_dictation::settings::SettingsStore;
use parakeet_dictation::streamer::{self, EndpointStrategy, Mode, Outcome};
use parakeet_dictation::warmup;
use parakeet_dictation::wav::read_wav_mono;
use parakeet_dictation::windows::HoldWindowConfig;

use crate::engine::{HexEngine, HexOfflineBackend, SAMPLE_RATE};
use crate::gold::DecodePath;

/// `bench_e2e`'s threshold for the fixture's last audible sample.
const SILENCE_AMPLITUDE: f32 = 0.0001;

pub struct HoldArgs {
    pub model: PathBuf,
    pub wav: PathBuf,
    pub device: String,
    pub reps: usize,
    pub warmup_reps: usize,
    pub path: DecodePath,
}

pub fn run(args: &HoldArgs) -> Result<()> {
    let engine = HexEngine::load(&args.model)?;
    let backend = Arc::new(HexOfflineBackend::new(engine));
    let asr = Arc::new(Asr::from_backend(backend.clone()));
    // bench_e2e's warm-up, then Hex's own stream probe for the stream path.
    warmup::dummy_decode(&asr)?;
    if args.path == DecodePath::Stream {
        backend.with_engine(HexEngine::prewarm_stream)?;
    }

    let (samples, sample_rate) = read_wav_mono(&args.wav)?;
    let last = samples
        .iter()
        .rposition(|sample| sample.abs() >= SILENCE_AMPLITUDE)
        .ok_or_else(|| anyhow!("fixture contains no audible samples"))?;
    let replay: Arc<[f32]> = Arc::from(&samples[..=last]);
    log::info!(
        "loaded {} ({:.3}s mono @ {sample_rate} Hz); loopback={}; path={}",
        args.wav.display(),
        samples.len() as f32 / sample_rate as f32,
        args.device,
        match args.path {
            DecodePath::Offline => "offline",
            DecodePath::Stream => "stream",
        }
    );

    let store = SettingsStore::new()?;
    let mut peak = 0_u64;
    for rep in 0..args.warmup_reps + args.reps {
        let emit = rep >= args.warmup_reps;
        let transcript = match args.path {
            DecodePath::Offline => {
                run_offline(args, &store, asr.clone(), replay.clone(), sample_rate, emit)?
            }
            DecodePath::Stream => backend.with_engine(|engine| {
                run_stream(args, engine, replay.clone(), sample_rate, emit)
            })?,
        };
        peak = peak.max(performance::resident_bytes(std::process::id())?);
        log::info!("bench_hex rep={rep} measured={emit} transcript={transcript:?}");
    }
    let peak = peak.max(performance::peak_resident_bytes()?);
    log::info!(
        "bench_hex_summary reps={} peak_resident_bytes={peak} peak_resident_gib={:.3}",
        args.reps,
        peak as f64 / f64::from(1_u32 << 30)
    );
    Ok(())
}

fn run_offline(
    args: &HoldArgs,
    store: &SettingsStore,
    asr: Arc<Asr>,
    samples: Arc<[f32]>,
    sample_rate: u32,
    emit: bool,
) -> Result<String> {
    let (session, outcome_rx) = streamer::start_with_strategy_on_device(
        &store.vad_path(),
        Mode::Manual,
        asr.clone(),
        EndpointStrategy::Serial,
        EndpointPolicy::LongForm.confirmation_ms(),
        HoldWindowConfig {
            enabled: false,
            ..HoldWindowConfig::default()
        },
        Some(&args.device),
    )?;
    let playback = start_playback(&args.device, samples.clone(), sample_rate)?;
    let timeout = Duration::from_secs_f32(samples.len() as f32 / sample_rate as f32 + 15.0);
    let acoustic_end = wait_for_acoustic_end(&playback, timeout)?;
    if let Some(remaining) = acoustic_end.checked_duration_since(Instant::now()) {
        std::thread::sleep(remaining);
    }
    let release = Instant::now();
    session.finalize();
    let outcome = outcome_rx
        .0
        .recv_timeout(timeout)
        .context("waiting for the Hold outcome")?;
    let Outcome::Speech {
        samples: recorded,
        sample_rate: recorded_rate,
        early_transcript,
        mut timer,
    } = outcome
    else {
        bail!("the Hold session ended without speech");
    };
    drop(playback);
    drop(session);
    let transcript = match early_transcript {
        Some(text) => text,
        None => {
            timer.mark_asr_start();
            let text = asr.recognize(&recorded, recorded_rate)?;
            timer.mark_asr_done();
            text
        }
    };
    std::hint::black_box(&transcript);
    timer.mark_speech_end_at_instant(release);
    timer.mark_paste_done();
    if emit {
        timer.emit();
    }
    Ok(transcript)
}

enum Feed {
    Audio(Vec<f32>),
    Finish(Vec<f32>),
}

fn run_stream(
    args: &HoldArgs,
    engine: &mut HexEngine,
    samples: Arc<[f32]>,
    sample_rate: u32,
    emit: bool,
) -> Result<String> {
    let mut timer = PhaseTimer::start(PhaseTimerMode::Bench, performance::next_session_id());
    let (feed_tx, feed_rx) = channel::<Feed>();
    let (done_tx, done_rx) = channel::<Result<(String, usize)>>();
    std::thread::scope(move |scope| -> Result<String> {
        scope.spawn(move || {
            let result = engine.with_stream(|feed| drive_stream(feed, &feed_rx));
            let _ = done_tx.send(result);
        });
        let resampler = Arc::new(Mutex::new(Resampler::new(sample_rate)?));
        let captured = Arc::new(AtomicUsize::new(0));
        let capture = start_capture(
            &args.device,
            sample_rate,
            resampler.clone(),
            captured.clone(),
            feed_tx.clone(),
        )?;
        let playback = start_playback(&args.device, samples.clone(), sample_rate)?;
        let timeout = Duration::from_secs_f32(samples.len() as f32 / sample_rate as f32 + 15.0);
        let acoustic_end = wait_for_acoustic_end(&playback, timeout)?;
        if let Some(remaining) = acoustic_end.checked_duration_since(Instant::now()) {
            std::thread::sleep(remaining);
        }
        let release = Instant::now();
        timer.mark_vad_endpoint();
        // Stopping the input joins its callback, so nothing is sent after
        // the flushed tail.
        drop(capture);
        let tail = resampler
            .lock()
            .map_err(|_| anyhow!("resampler mutex poisoned"))?
            .flush();
        let captured = captured.load(Ordering::Acquire) + tail.len();
        timer.mark_capture_end(captured as f32 / SAMPLE_RATE as f32);
        timer.mark_asr_start();
        feed_tx
            .send(Feed::Finish(tail))
            .map_err(|_| anyhow!("the stream thread stopped"))?;
        let (transcript, fed) = done_rx
            .recv_timeout(timeout)
            .context("waiting for the stream to finish")??;
        timer.mark_asr_done();
        std::hint::black_box(&transcript);
        if fed != captured {
            bail!("the stream heard {fed} samples of the {captured} captured");
        }
        timer.mark_speech_end_at_instant(release);
        timer.mark_paste_done();
        drop(playback);
        if emit {
            timer.emit();
        }
        Ok(transcript)
    })
}

fn drive_stream(
    feed: &mut crate::engine::LiveFeed<'_, '_>,
    rx: &Receiver<Feed>,
) -> Result<(String, usize)> {
    loop {
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(Feed::Audio(samples)) => feed.push(&samples)?,
            Ok(Feed::Finish(tail)) => {
                // Audio the callback queued before the stop is ahead of the
                // finish in the channel; only the resampler tail rides here.
                feed.push(&tail)?;
                let text = feed.finish()?;
                return Ok((text, feed.fed));
            }
            Err(RecvTimeoutError::Timeout) => bail!("no audio or finish for 60 s"),
            Err(RecvTimeoutError::Disconnected) => bail!("the capture side hung up"),
        }
    }
}

fn find_device(input: bool, name: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();
    let mut devices = if input {
        host.input_devices().context("enumerating input devices")?
    } else {
        host.output_devices()
            .context("enumerating output devices")?
    };
    devices
        .find(|device| device.name().is_ok_and(|candidate| candidate == name))
        .ok_or_else(|| anyhow!("device not found: {name}"))
}

fn start_capture(
    device_name: &str,
    sample_rate: u32,
    resampler: Arc<Mutex<Resampler>>,
    captured: Arc<AtomicUsize>,
    tx: Sender<Feed>,
) -> Result<cpal::Stream> {
    let device = find_device(true, device_name)?;
    let supported = device.default_input_config()?;
    if supported.sample_rate().0 != sample_rate {
        bail!(
            "{device_name} input is {} Hz but the fixture is {sample_rate} Hz",
            supported.sample_rate().0
        );
    }
    if supported.sample_format() != SampleFormat::F32 {
        bail!(
            "{device_name} input is {:?}, expected f32",
            supported.sample_format()
        );
    }
    let config = supported.config();
    let channels = usize::from(config.channels);
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            // `audio::to_mono`: average the channels of each frame.
            let mono: Vec<f32> = if channels <= 1 {
                data.to_vec()
            } else {
                data.chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                    .collect()
            };
            if let Ok(mut resampler) = resampler.lock() {
                let out = resampler.push(&mono).into_owned();
                if !out.is_empty() {
                    captured.fetch_add(out.len(), Ordering::Release);
                    let _ = tx.send(Feed::Audio(out));
                }
            }
        },
        |error| log::error!("loopback input error: {error}"),
        None,
    )?;
    stream.play().context("starting loopback capture")?;
    Ok(stream)
}

// The playback half is `bench_e2e`'s, unchanged.

fn wait_for_acoustic_end(playback: &Playback, timeout: Duration) -> Result<Instant> {
    const POLL: Duration = Duration::from_millis(2);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(end) = playback.marker()? {
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
    fn marker(&self) -> Result<Option<Instant>> {
        Ok(*self
            .acoustic_end
            .lock()
            .map_err(|_| anyhow!("acoustic-end marker mutex poisoned"))?)
    }
}

fn start_playback(device_name: &str, samples: Arc<[f32]>, sample_rate: u32) -> Result<Playback> {
    let device = find_device(false, device_name)?;
    let supported = device.default_output_config()?;
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
        SampleFormat::F32 => output_stream::<f32>(&device, &config, samples, acoustic_end.clone())?,
        SampleFormat::I16 => output_stream::<i16>(&device, &config, samples, acoustic_end.clone())?,
        SampleFormat::U16 => output_stream::<u16>(&device, &config, samples, acoustic_end.clone())?,
        other => bail!("unsupported output sample format: {other:?}"),
    };
    stream.play().context("starting loopback playback")?;
    Ok(Playback {
        _stream: stream,
        acoustic_end,
    })
}

fn output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<[f32]>,
    acoustic_end: Arc<Mutex<Option<Instant>>>,
) -> Result<cpal::Stream>
where
    T: SizedSample + Sample + FromSample<f32>,
{
    let channels = usize::from(config.channels);
    let rate = f64::from(config.sample_rate.0);
    let mut cursor = 0_usize;
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
                    let value = T::from_sample(samples.get(cursor).copied().unwrap_or(0.0));
                    cursor = cursor.saturating_add(1);
                    for sample in frame {
                        *sample = value;
                    }
                }
                if start_cursor < samples.len() && cursor >= samples.len() {
                    let frames_to_end = samples.len().saturating_sub(start_cursor);
                    let within = Duration::from_secs_f64(frames_to_end as f64 / rate);
                    if let Ok(mut marker) = acoustic_end.lock() {
                        *marker = callback_started.checked_add(playback_delay + within);
                    }
                }
            },
            |error| log::error!("loopback output error: {error}"),
            None,
        )
        .context("building loopback output stream")
}
