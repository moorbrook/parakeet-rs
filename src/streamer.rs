//! Press-once / VAD-auto-stop dictation driver.
//!
//! Owns the cpal capture, a Silero VAD, and a small state machine that decides
//! when an utterance is done. The hotkey handler in `lib.rs` calls `start`,
//! then awaits an `Outcome` on the returned receiver. A second hotkey press
//! before the VAD fires sends `cancel` and produces `Outcome::Cancelled`.

use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::LinearResampler;

use crate::audio::{AudioCapture, Recording};
use crate::vad::{VAD_SAMPLE_RATE, Vad, WINDOW_SIZE};

/// If the user starts dictation and says nothing within this window, give up.
const NO_SPEECH_TIMEOUT: Duration = Duration::from_secs(5);

pub enum Outcome {
    /// End of speech reached. Carries the raw mono samples at the native
    /// capture rate so the ASR can do its own resample / decode.
    Speech { samples: Vec<f32>, sample_rate: u32 },
    /// User hit the hotkey again before VAD fired.
    Cancelled,
    /// VAD never saw speech in the timeout window.
    NoSpeech,
    Error(anyhow::Error),
}

pub struct Session {
    cancel_tx: Sender<()>,
    pub outcome_rx: Receiver<Outcome>,
    join: Option<JoinHandle<()>>,
}

impl Session {
    pub fn cancel(&self) {
        let _ = self.cancel_tx.send(());
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.cancel_tx.send(());
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

/// Start a new dictation session. The session keeps running until the VAD
/// detects end-of-speech, a cancel signal arrives, or `NO_SPEECH_TIMEOUT`
/// elapses without any detected speech.
pub fn start(vad_model: &Path) -> Result<Session> {
    let (tap_tx, tap_rx) = channel::<Vec<f32>>();
    let (cancel_tx, cancel_rx) = channel::<()>();
    let (outcome_tx, outcome_rx) = channel::<Outcome>();

    let capture = AudioCapture::start_with_tap(tap_tx).context("starting capture")?;
    let sample_rate = capture.sample_rate();

    // Silero is a small RNN — single thread is plenty and avoids contention
    // with the ASR's CoreML threads. Vad::load reuses sherpa-onnx's pooled
    // ORT runtime so the cost is just creating the session.
    let vad = Vad::load(vad_model, 1).context("loading Silero VAD")?;

    let join = std::thread::Builder::new()
        .name("vad-watcher".into())
        .spawn(move || {
            let outcome = run(capture, vad, sample_rate, tap_rx, cancel_rx);
            let _ = outcome_tx.send(outcome);
        })
        .context("spawning VAD watcher")?;

    Ok(Session {
        cancel_tx,
        outcome_rx,
        join: Some(join),
    })
}

fn run(
    capture: AudioCapture,
    vad: Vad,
    sample_rate: u32,
    tap_rx: Receiver<Vec<f32>>,
    cancel_rx: Receiver<()>,
) -> Outcome {
    let resampler = match LinearResampler::create(sample_rate as i32, VAD_SAMPLE_RATE) {
        Some(r) => r,
        None => {
            let _ = capture.stop();
            return Outcome::Error(anyhow!(
                "could not build {sample_rate}->{VAD_SAMPLE_RATE} resampler"
            ));
        }
    };

    let mut window_buf: Vec<f32> = Vec::with_capacity(WINDOW_SIZE as usize * 4);
    let session_start = Instant::now();
    let mut saw_speech = false;
    let mut speech_started_at: Option<Instant> = None;

    loop {
        if cancel_rx.try_recv().is_ok() {
            return finish(capture, Outcome::Cancelled);
        }

        let chunk = match tap_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(c) => c,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !saw_speech && session_start.elapsed() > NO_SPEECH_TIMEOUT {
                    return finish(capture, Outcome::NoSpeech);
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Outcome::Error(anyhow!("audio tap closed before VAD finished"));
            }
        };

        let chunk16 = resampler.resample(&chunk, false);
        if chunk16.is_empty() {
            continue;
        }
        window_buf.extend_from_slice(&chunk16);

        // Silero expects 512-sample windows at 16 kHz. Drain in window-sized
        // bites; anything left over stays in `window_buf` for next tick.
        while window_buf.len() >= WINDOW_SIZE as usize {
            let head: Vec<f32> = window_buf.drain(..WINDOW_SIZE as usize).collect();
            vad.accept_waveform(&head);
            vad.drain_segments();

            let detected_now = vad.detected();
            if detected_now {
                if !saw_speech {
                    saw_speech = true;
                    speech_started_at = Some(Instant::now());
                }
            } else if saw_speech {
                // Silero only flips back to !detected after its own
                // min_silence_duration grace period (we configured 150 ms),
                // so a single false here is already end-of-speech.
                return finish_with_recording(capture);
            }

            // Safety net: respect max_speech_duration. If Silero hasn't fired
            // an end-of-speech by then, force a cut.
            if let Some(t) = speech_started_at {
                if t.elapsed() > Duration::from_secs(crate::vad::MAX_SPEECH_S as u64) {
                    return finish_with_recording(capture);
                }
            }
        }
    }
}

fn finish_with_recording(capture: AudioCapture) -> Outcome {
    match capture.stop() {
        Ok(rec) => {
            let Recording {
                samples,
                sample_rate,
                channels,
            } = rec;
            let mono = if channels <= 1 {
                samples
            } else {
                let ch = channels as usize;
                let n = samples.len() / ch;
                let mut out = Vec::with_capacity(n);
                for frame in samples.chunks_exact(ch) {
                    let sum: f32 = frame.iter().sum();
                    out.push(sum / ch as f32);
                }
                out
            };
            Outcome::Speech {
                samples: mono,
                sample_rate,
            }
        }
        Err(e) => Outcome::Error(e),
    }
}

fn finish(capture: AudioCapture, outcome: Outcome) -> Outcome {
    let _ = capture.stop();
    outcome
}
