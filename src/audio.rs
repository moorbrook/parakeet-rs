//! Mic capture via cpal, converted to 16 kHz mono inside the capture callback.
//!
//! Core Audio hands us the device's native format — 48 kHz on the built-in
//! microphone, sometimes stereo — and every consumer downstream wants 16 kHz
//! mono. Folding and resampling here, per callback, means the conversion is
//! finished by the time the user stops speaking instead of sitting on the
//! endpoint path. See ADR-0030.
//!
//! Two consumers per session:
//! 1. a `Vec<f32>` buffer that accumulates the full 16 kHz mono recording,
//!    returned on `stop()` for the ASR pass;
//! 2. a `mpsc::Sender<Vec<f32>>` "tap" that hands the same 16 kHz chunks to the
//!    VAD watcher in `streamer.rs` so it can react in real time.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use parking_lot::Mutex;

use crate::qos;
use crate::resample::{Resampler, TARGET_SAMPLE_RATE};

/// Mono Float32 audio at [`TARGET_SAMPLE_RATE`], ready for the recognizer.
pub struct Recording {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

enum Cmd {
    Stop(Sender<Result<Recording>>),
}

pub struct AudioCapture {
    tx: Sender<Cmd>,
    join: Option<JoinHandle<()>>,
}

impl AudioCapture {
    /// Start a capture with a streaming tap. Every cpal callback chunk is
    /// folded to mono, resampled, and forwarded over `tap`; if the receiver is
    /// dropped the send fails silently and capture continues for the buffered
    /// recording consumer.
    pub fn start_with_tap(tap: Sender<Vec<f32>>) -> Result<Self> {
        Self::start_with_tap_on_device(tap, None)
    }

    /// Start capture on a specific Core Audio input device. Production passes
    /// `None` and uses the system default; the end-to-end benchmark selects the
    /// BlackHole loopback explicitly so it never mutates the user's defaults.
    pub fn start_with_tap_on_device(
        tap: Sender<Vec<f32>>,
        device_name: Option<&str>,
    ) -> Result<Self> {
        let (tx, rx) = channel::<Cmd>();
        let (ready_tx, ready_rx) = channel::<Result<u32>>();
        let device_name = device_name.map(str::to_owned);

        let join = std::thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || {
                qos::set_user_interactive();
                let (device, config) = match open_device(device_name.as_deref()) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let device_sample_rate = config.sample_rate().0;
                let resampler = match Resampler::new(device_sample_rate) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // The buffer now holds MONO samples at TARGET_SAMPLE_RATE, not
                // interleaved device-rate frames, so 30 s is 480 k samples
                // (~1.9 MB) regardless of device format. Reserving it up front
                // keeps a reallocation out of the realtime cpal callback.
                let sinks = Arc::new(CaptureSinks {
                    buffer: Mutex::new(Vec::with_capacity(TARGET_SAMPLE_RATE as usize * 30)),
                    resampler: Mutex::new(resampler),
                    tap,
                    channels: config.channels(),
                    callback: CallbackStats::new(),
                });
                let stream = match build_stream(&device, config, sinks.clone()) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if let Err(e) = stream.play().context("starting stream") {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
                let _ = ready_tx.send(Ok(device_sample_rate));

                // The thread only handles one command (Stop), so receive
                // it inline rather than looping — clippy::never_loop.
                if let Ok(Cmd::Stop(reply)) = rx.recv() {
                    // Dropping the stream first guarantees no callback is
                    // running, so the flush below cannot race one and the
                    // filter tail lands after the last callback's output.
                    drop(stream);
                    let tail = sinks.resampler.lock().flush();
                    let mut buffer = sinks.buffer.lock();
                    buffer.extend_from_slice(&tail);
                    let samples = std::mem::take(&mut *buffer);
                    drop(buffer);
                    // Read after the stream is dropped: no callback can be
                    // running, so the histogram is a consistent snapshot.
                    let callback = sinks.callback.summary();
                    log::info!(
                        "capture_callback device_hz={device_sample_rate} chunks={} \
                         mean_us={:.1} max_us={} p99_us={}{}",
                        callback.count,
                        callback.mean_micros,
                        callback.max_micros,
                        callback.p99_micros,
                        if callback.p99_saturated { "+" } else { "" }
                    );
                    let _ = reply.send(Ok(Recording {
                        samples,
                        sample_rate: TARGET_SAMPLE_RATE,
                    }));
                }
            })
            .context("spawning audio thread")?;

        let device_sample_rate = ready_rx
            .recv()
            .map_err(|_| anyhow!("audio thread exited before ready"))??;
        log::info!(
            "capture device runs at {device_sample_rate} Hz; delivering {TARGET_SAMPLE_RATE} Hz mono"
        );

        Ok(Self {
            tx,
            join: Some(join),
        })
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

/// One microsecond per bucket. The last bucket collects everything at or above
/// its index, so a callback that runs long is counted but not mis-binned.
const CALLBACK_BUCKETS: usize = 512;

/// Duration histogram for the realtime capture callback.
///
/// The callback now folds to mono, filters, and copies twice, and the cost of
/// that has to be a measurement rather than an estimate: a cpal callback that
/// overruns its buffer period drops audio. Recording is three relaxed atomic
/// operations, so it does not add a lock or a syscall to the path it measures.
struct CallbackStats {
    buckets: Vec<AtomicU32>,
    max_micros: AtomicU64,
    total_micros: AtomicU64,
    count: AtomicU64,
}

/// What [`CallbackStats`] reports once capture stops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackSummary {
    pub count: u64,
    pub max_micros: u64,
    pub mean_micros: f64,
    pub p99_micros: u64,
    /// True when p99 fell in the final bucket, so the reported value is a
    /// lower bound rather than the figure itself.
    pub p99_saturated: bool,
}

impl CallbackStats {
    fn new() -> Self {
        Self {
            buckets: (0..CALLBACK_BUCKETS).map(|_| AtomicU32::new(0)).collect(),
            max_micros: AtomicU64::new(0),
            total_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    fn record(&self, micros: u64) {
        let index = usize::try_from(micros).unwrap_or(usize::MAX);
        let index = index.min(CALLBACK_BUCKETS - 1);
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
        self.max_micros.fetch_max(micros, Ordering::Relaxed);
        self.total_micros.fetch_add(micros, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the histogram after the stream is dropped, so no callback is
    /// concurrently writing and the counts are a consistent snapshot.
    fn summary(&self) -> CallbackSummary {
        let count = self.count.load(Ordering::Relaxed);
        let max_micros = self.max_micros.load(Ordering::Relaxed);
        if count == 0 {
            return CallbackSummary {
                count: 0,
                max_micros: 0,
                mean_micros: 0.0,
                p99_micros: 0,
                p99_saturated: false,
            };
        }
        let mean_micros = self.total_micros.load(Ordering::Relaxed) as f64 / count as f64;
        // Rank of the p99 sample, counting from one: the smallest value with at
        // least 99% of the samples at or below it.
        let target = count.saturating_mul(99).div_ceil(100).max(1);
        let mut seen = 0_u64;
        for (index, bucket) in self.buckets.iter().enumerate() {
            seen = seen.saturating_add(u64::from(bucket.load(Ordering::Relaxed)));
            if seen >= target {
                return CallbackSummary {
                    count,
                    max_micros,
                    mean_micros,
                    p99_micros: index as u64,
                    p99_saturated: index == CALLBACK_BUCKETS - 1,
                };
            }
        }
        CallbackSummary {
            count,
            max_micros,
            mean_micros,
            p99_micros: (CALLBACK_BUCKETS - 1) as u64,
            p99_saturated: true,
        }
    }
}

/// Everything the realtime callback writes into, shared with the capture
/// thread so it can flush the resampler and take the buffer after `stop()`.
struct CaptureSinks {
    buffer: Mutex<Vec<f32>>,
    /// Streaming filter state. `Resampler::push` takes `&mut self` because the
    /// underlying Kaldi object carries a remainder between calls; the lock is
    /// what makes that exclusive. It is never contended — only the callback
    /// takes it until the stream is dropped, and only the capture thread takes
    /// it afterwards.
    resampler: Mutex<Resampler>,
    tap: Sender<Vec<f32>>,
    channels: u16,
    callback: CallbackStats,
}

/// Fold multi-channel input down to mono so the tap is mono. Keeps the
/// VAD math 1:1 and saves the resampler from doing it after the fact.
fn to_mono(data: &[f32], channels: u16) -> Vec<f32> {
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
}

/// Shared tail of every sample-format callback: fold to mono, drive the HUD
/// level meter, resample to 16 kHz, then append to the capture buffer and feed
/// the VAD tap. The three `build_input_stream` closures differ only in how they
/// convert their native sample type to f32.
///
/// The level meter reads the pre-resample mono chunk so the HUD keeps showing
/// the device's own peak, independent of the filter.
fn forward_samples(sinks: &CaptureSinks, floats: &[f32]) {
    let started = Instant::now();
    let mono = to_mono(floats, sinks.channels);
    crate::hud::set_audio_level(peak_amplitude(&mono));

    // A device already at the target rate hands `mono` straight through rather
    // than copying it, which is what `Resampler::push` borrowing its argument
    // is for. Taking that branch before the borrow starts is what lets the
    // owned `mono` move into the tap.
    let mut resampler = sinks.resampler.lock();
    let converted = if resampler.is_pass_through() {
        drop(resampler);
        mono
    } else {
        let out = resampler.push(&mono).into_owned();
        drop(resampler);
        out
    };
    if converted.is_empty() {
        // Normal for a chunk shorter than the filter's window; the samples are
        // held inside the resampler and come out with the next push.
        sinks.callback.record(started.elapsed().as_micros() as u64);
        return;
    }
    sinks.buffer.lock().extend_from_slice(&converted);
    let _ = sinks.tap.send(converted);
    sinks.callback.record(started.elapsed().as_micros() as u64);
}

fn open_device(device_name: Option<&str>) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();
    let device = match device_name {
        Some(name) => host
            .input_devices()
            .context("enumerating input devices")?
            .find(|device| device.name().is_ok_and(|candidate| candidate == name))
            .ok_or_else(|| anyhow!("input device not found: {name}"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?,
    };
    // Deliberately the device's DEFAULT config, not a 16 kHz request. Asking
    // cpal for a rate the device does not list makes its Core Audio host set
    // the device's nominal sample rate, a system-wide change other apps see,
    // and the built-in microphone does not offer 16 kHz at all. ADR-0030.
    let config = device
        .default_input_config()
        .context("default input config")?;
    Ok((device, config))
}

fn build_stream(
    device: &cpal::Device,
    config: cpal::SupportedStreamConfig,
    sinks: Arc<CaptureSinks>,
) -> Result<cpal::Stream> {
    let err_fn = |err| log::error!("audio stream error: {err}");
    let sample_format = config.sample_format();
    let cfg: cpal::StreamConfig = config.into();

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &cfg,
            move |data: &[f32], _| forward_samples(&sinks, data),
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &cfg,
            move |data: &[i16], _| {
                let floats: Vec<f32> = data
                    .iter()
                    .map(|&s| f32::from(s) / f32::from(i16::MAX))
                    .collect();
                forward_samples(&sinks, &floats);
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &cfg,
            move |data: &[u16], _| {
                let floats: Vec<f32> = data
                    .iter()
                    .map(|&s| {
                        let centered = f32::from(s) - f32::from(i16::MAX) - 1.0;
                        centered / (f32::from(i16::MAX) + 1.0)
                    })
                    .collect();
                forward_samples(&sinks, &floats);
            },
            err_fn,
            None,
        )?,
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    };
    Ok(stream)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_frames_average_to_mono() {
        let interleaved = [1.0, 0.0, 0.5, -0.5, -1.0, 1.0];
        assert_eq!(to_mono(&interleaved, 2), vec![0.5, 0.0, 0.0]);
    }

    #[test]
    fn mono_input_is_unchanged() {
        let samples = [0.25, -0.5, 1.0];
        assert_eq!(to_mono(&samples, 1), samples.to_vec());
    }

    #[test]
    fn callback_histogram_reports_max_and_p99() {
        let stats = CallbackStats::new();
        // 99 samples at 10 µs and one at 300 µs: p99 is the 99th-ranked
        // sample, which is still 10 µs, while max is the outlier.
        for _ in 0..99 {
            stats.record(10);
        }
        stats.record(300);
        let summary = stats.summary();
        assert_eq!(summary.count, 100);
        assert_eq!(summary.max_micros, 300);
        assert!(
            (summary.mean_micros - 12.9).abs() < 0.05,
            "{}",
            summary.mean_micros
        );
        assert_eq!(summary.p99_micros, 10);
        assert!(!summary.p99_saturated);
    }

    #[test]
    fn callback_histogram_flags_a_saturated_tail() {
        let stats = CallbackStats::new();
        for _ in 0..10 {
            stats.record(u64::from(u32::MAX));
        }
        let summary = stats.summary();
        assert_eq!(summary.max_micros, u64::from(u32::MAX));
        assert_eq!(summary.p99_micros, (CALLBACK_BUCKETS - 1) as u64);
        assert!(
            summary.p99_saturated,
            "a value past the last bucket must be reported as a lower bound"
        );
    }

    #[test]
    fn callback_histogram_is_empty_before_any_callback() {
        let summary = CallbackStats::new().summary();
        assert_eq!(summary.count, 0);
        assert_eq!(summary.max_micros, 0);
        assert_eq!(summary.mean_micros, 0.0);
        assert_eq!(summary.p99_micros, 0);
        assert!(!summary.p99_saturated);
    }

    /// The capture buffer must contain exactly what a single batch resample of
    /// the whole recording would contain, including the flushed tail. This is
    /// the callback-level version of the property `resample` proves for the
    /// filter itself: it also covers the mono fold and the "empty push" early
    /// return in `forward_samples`.
    #[test]
    fn callback_sequence_reproduces_a_batch_resample() {
        let device_rate = 48_000_u32;
        let frames: Vec<f32> = (0..device_rate as usize / 2)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / device_rate as f32).sin() * 0.5)
            .collect();
        // Stereo interleave with a second channel that cancels to the same
        // mono signal, so the expected batch input is `frames` itself.
        let interleaved: Vec<f32> = frames.iter().flat_map(|&s| [s, s]).collect();

        let (tap, tap_rx) = channel::<Vec<f32>>();
        let sinks = CaptureSinks {
            buffer: Mutex::new(Vec::new()),
            resampler: Mutex::new(Resampler::new(device_rate).expect("48k resampler")),
            tap,
            channels: 2,
            callback: CallbackStats::new(),
        };

        let mut offset = 0;
        let mut sizes = [64_usize, 2, 1024, 7, 512].iter().copied().cycle();
        while offset < interleaved.len() {
            let take = sizes.next().unwrap_or(512).min(interleaved.len() - offset);
            // Keep the slice frame-aligned the way Core Audio delivers it.
            let take = take - take % 2;
            let take = if take == 0 { 2 } else { take };
            let take = take.min(interleaved.len() - offset);
            forward_samples(&sinks, &interleaved[offset..offset + take]);
            offset += take;
        }
        let tail = sinks.resampler.lock().flush();
        let mut captured = std::mem::take(&mut *sinks.buffer.lock());
        captured.extend_from_slice(&tail);

        let expected = crate::resample::to_target_rate(&frames, device_rate)
            .expect("batch resample")
            .into_owned();
        assert_eq!(captured.len(), expected.len());
        for (index, (a, b)) in captured.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "sample {index}: {a} vs {b}");
        }

        // The tap carries the same samples, minus the post-stop flush tail.
        let tapped: Vec<f32> = tap_rx.try_iter().flatten().collect();
        assert_eq!(tapped.len(), captured.len() - tail.len());
        assert_eq!(tapped.as_slice(), &captured[..tapped.len()]);
    }
}
