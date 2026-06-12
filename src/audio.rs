//! Mic capture via cpal.
//!
//! Two consumers per session:
//! 1. a `Vec<f32>` buffer that accumulates the full raw recording, returned on
//!    `stop()` for the ASR pass;
//! 2. a `mpsc::Sender<Vec<f32>>` "tap" that hands a copy of every cpal callback
//!    chunk to the VAD watcher in `streamer.rs` so it can react in real time.

use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use parking_lot::Mutex;

use crate::qos;

pub struct Recording {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

enum Cmd {
    Stop(Sender<Result<Recording>>),
}

pub struct AudioCapture {
    tx: Sender<Cmd>,
    sample_rate: u32,
    join: Option<JoinHandle<()>>,
}

impl AudioCapture {
    /// Start a capture with a streaming tap. Every cpal callback chunk is also
    /// forwarded over `tap`; if the receiver is dropped the send fails silently
    /// and capture continues for the buffered-recording consumer.
    pub fn start_with_tap(tap: Sender<Vec<f32>>) -> Result<Self> {
        let (tx, rx) = channel::<Cmd>();
        let (ready_tx, ready_rx) = channel::<Result<(u32, u16)>>();

        let join = std::thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || {
                qos::set_user_interactive();
                // Allocate zero-capacity here; we don't yet know the
                // device's native sample rate. Hardcoding `16_000 * 30`
                // is wrong on every Mac (built-in mics are 44.1 / 48
                // kHz mono or stereo) and forces a reallocation
                // mid-recording, blocking the realtime cpal callback.
                let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
                let (sample_rate, channels, stream) =
                    match build_stream(buffer.clone(), tap.clone()) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                // `buffer` stores INTERLEAVED native-channel samples
                // (the F32/I16/U16 callbacks at `build_stream` push
                // `data` verbatim, not the down-mixed mono). 30 s at
                // 48 kHz stereo is 2.88 M samples ≈ 11 MB; 96 kHz
                // stereo is ~23 MB. Worst case is generous but
                // single-shot (per dictation session) so the
                // allocation amortises away.
                buffer
                    .lock()
                    .reserve(sample_rate as usize * channels as usize * 30);
                if let Err(e) = stream.play().context("starting stream") {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
                let _ = ready_tx.send(Ok((sample_rate, channels)));

                // The thread only handles one command (Stop), so receive
                // it inline rather than looping — clippy::never_loop.
                if let Ok(Cmd::Stop(reply)) = rx.recv() {
                    drop(stream);
                    let samples = std::mem::take(&mut *buffer.lock());
                    let _ = reply.send(Ok(Recording {
                        samples,
                        sample_rate,
                        channels,
                    }));
                }
            })
            .context("spawning audio thread")?;

        let (sample_rate, _channels) = ready_rx
            .recv()
            .map_err(|_| anyhow!("audio thread exited before ready"))??;

        Ok(Self {
            tx,
            sample_rate,
            join: Some(join),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn stop(mut self) -> Result<Recording> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .send(Cmd::Stop(reply_tx))
            .map_err(|_| anyhow!("audio thread is gone"))?;
        let rec = reply_rx
            .recv()
            .map_err(|_| anyhow!("audio thread closed without replying"))??;
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
        Ok(rec)
    }
}

fn build_stream(
    buffer: Arc<Mutex<Vec<f32>>>,
    tap: Sender<Vec<f32>>,
) -> Result<(u32, u16, cpal::Stream)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let config = device
        .default_input_config()
        .context("default input config")?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    let err_fn = |err| log::error!("audio stream error: {err}");

    // Fold multi-channel input down to mono inline so the tap is mono. Keeps
    // the VAD math 1:1 and saves the resampler from doing it after the fact.
    let to_mono = move |data: &[f32], channels: u16| -> Vec<f32> {
        if channels <= 1 {
            return data.to_vec();
        }
        let ch = channels as usize;
        let n = data.len() / ch;
        let mut out = Vec::with_capacity(n);
        for frame in data.chunks_exact(ch) {
            let sum: f32 = frame.iter().sum();
            out.push(sum / ch as f32);
        }
        out
    };

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let cfg: cpal::StreamConfig = config.into();
            device.build_input_stream(
                &cfg,
                {
                    let buffer = buffer.clone();
                    let tap = tap.clone();
                    move |data: &[f32], _| {
                        buffer.lock().extend_from_slice(data);
                        let mono = to_mono(data, channels);
                        crate::hud::set_audio_level(peak_amplitude(&mono));
                        let _ = tap.send(mono);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let cfg: cpal::StreamConfig = config.into();
            device.build_input_stream(
                &cfg,
                {
                    let buffer = buffer.clone();
                    let tap = tap.clone();
                    move |data: &[i16], _| {
                        let floats: Vec<f32> = data
                            .iter()
                            .map(|&s| f32::from(s) / f32::from(i16::MAX))
                            .collect();
                        buffer.lock().extend_from_slice(&floats);
                        let mono = to_mono(&floats, channels);
                        crate::hud::set_audio_level(peak_amplitude(&mono));
                        let _ = tap.send(mono);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let cfg: cpal::StreamConfig = config.into();
            device.build_input_stream(
                &cfg,
                {
                    let buffer = buffer.clone();
                    let tap = tap.clone();
                    move |data: &[u16], _| {
                        let floats: Vec<f32> = data
                            .iter()
                            .map(|&s| {
                                let centered = f32::from(s) - f32::from(i16::MAX) - 1.0;
                                centered / (f32::from(i16::MAX) + 1.0)
                            })
                            .collect();
                        buffer.lock().extend_from_slice(&floats);
                        let mono = to_mono(&floats, channels);
                        crate::hud::set_audio_level(peak_amplitude(&mono));
                        let _ = tap.send(mono);
                    }
                },
                err_fn,
                None,
            )?
        }
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    };
    Ok((sample_rate, channels, stream))
}

/// Peak absolute amplitude across a chunk, clamped to [0, 1]. Cheap
/// (one fold, no allocation) so the cpal realtime callback can call
/// it per chunk without risking xruns. Used to drive the HUD's
/// listening-state waveform bars.
fn peak_amplitude(samples: &[f32]) -> f32 {
    samples
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max)
        .min(1.0)
}
