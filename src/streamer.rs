//! Dictation-session driver. Two modes:
//!
//! - **`Mode::VadAutoStop`** (tap-once UX): runs Silero VAD on the capture
//!   stream and finishes the session when it detects end-of-speech.
//! - **`Mode::Manual`** (press-and-hold UX): no VAD — the caller decides
//!   when to stop by calling `Session::finalize()`. Used when the hotkey
//!   itself defines the speech window.

use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use sherpa_onnx::LinearResampler;

use crate::audio::{AudioCapture, Recording};
use crate::performance::{next_session_id, PhaseTimer, PhaseTimerMode};
use crate::vad::{Vad, VAD_SAMPLE_RATE, WINDOW_SIZE};

/// If the user starts dictation and says nothing within this window, give up.
const NO_SPEECH_TIMEOUT: Duration = Duration::from_secs(5);

/// Hold-mode safety cap: refuse to record longer than this even if the user
/// keeps the key held. Matches the VAD's `max_speech_duration` so both modes
/// have the same upper bound on a single utterance.
const MANUAL_MAX_RECORDING: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Silero VAD watches the stream and ends the session when speech stops.
    VadAutoStop,
    /// Caller drives stop explicitly via `Session::finalize()`.
    Manual,
}

pub enum Outcome {
    /// End of speech reached. Carries the raw mono samples at the native
    /// capture rate so the ASR can do its own resample / decode.
    /// `timer` already has `mark_capture_end` (and `mark_vad_endpoint` in
    /// VadAutoStop mode) populated; the consumer is responsible for the
    /// remaining `mark_asr_*` / `mark_paste_done` calls and the final
    /// `emit()`. See `docs/latency-plan.md` §1.
    Speech {
        samples: Vec<f32>,
        sample_rate: u32,
        timer: PhaseTimer,
    },
    /// User aborted before any audio was eligible to commit.
    Cancelled,
    /// VAD never saw speech in the timeout window (VadAutoStop mode only).
    NoSpeech,
    Error(anyhow::Error),
}

enum Signal {
    Cancel,
    Finalize,
}

/// Command half of a dictation session. Lives in `App::session` for the
/// whole life of the recording so that hotkey press/release edges can
/// always reach the active session — the bug this split fixes was the
/// watcher thread `take()`ing the session out of `App::session` right
/// after start, leaving Hold-mode release with no way to call `finalize`.
pub struct Session {
    signal_tx: Sender<Signal>,
    join: Option<JoinHandle<()>>,
}

/// Outcome half — owned by the watcher thread. Cannot be `Send`-cloned
/// because `Receiver<T>` is single-consumer, so we split it out at
/// construction time and pass it directly to the watcher.
pub struct OutcomeRx(pub Receiver<Outcome>);

impl Session {
    /// Discard the in-flight recording. Produces `Outcome::Cancelled`.
    pub fn cancel(&self) {
        let _ = self.signal_tx.send(Signal::Cancel);
    }

    /// Stop capture immediately and commit whatever audio we've collected.
    /// Used by Hold-mode hotkey release. Produces `Outcome::Speech` (or
    /// `Outcome::Cancelled` if the buffer is empty).
    pub fn finalize(&self) {
        let _ = self.signal_tx.send(Signal::Finalize);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.signal_tx.send(Signal::Cancel);
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

/// Start a new dictation session in the given mode. `vad_model` is only
/// loaded for `Mode::VadAutoStop`; in `Mode::Manual` the path is ignored.
///
/// Returns the command half (kept by `App` so hotkey edges can reach it)
/// and the outcome half (passed directly to the watcher thread that
/// waits for the session to finish).
pub fn start(vad_model: &Path, mode: Mode) -> Result<(Session, OutcomeRx)> {
    let (tap_tx, tap_rx) = channel::<Vec<f32>>();
    let (signal_tx, signal_rx) = channel::<Signal>();
    let (outcome_tx, outcome_rx) = channel::<Outcome>();

    let capture = AudioCapture::start_with_tap(tap_tx).context("starting capture")?;
    let sample_rate = capture.sample_rate();

    let vad = if matches!(mode, Mode::VadAutoStop) {
        // Silero is a small RNN — single thread is plenty.
        Some(Vad::load(vad_model, 1).context("loading Silero VAD")?)
    } else {
        None
    };

    // The latency clock starts here — mic is hot, the user is about to
    // speak. The downstream `Outcome::Speech` carries this timer all the
    // way through ASR and paste; cancellations / no-speech drop it.
    let timer = PhaseTimer::start(PhaseTimerMode::Real, next_session_id());

    let join = std::thread::Builder::new()
        .name(match mode {
            Mode::VadAutoStop => "vad-watcher".into(),
            Mode::Manual => "hold-watcher".into(),
        })
        .spawn(move || {
            let outcome = match (mode, vad) {
                (Mode::VadAutoStop, Some(vad)) => {
                    run_vad(capture, vad, sample_rate, tap_rx, signal_rx, timer)
                }
                (Mode::Manual, _) => run_manual(capture, sample_rate, tap_rx, signal_rx, timer),
                _ => Outcome::Error(anyhow!("invalid mode/vad combination")),
            };
            let _ = outcome_tx.send(outcome);
        })
        .context("spawning session watcher")?;

    Ok((
        Session {
            signal_tx,
            join: Some(join),
        },
        OutcomeRx(outcome_rx),
    ))
}

fn run_vad(
    capture: AudioCapture,
    vad: Vad,
    sample_rate: u32,
    tap_rx: Receiver<Vec<f32>>,
    signal_rx: Receiver<Signal>,
    timer: PhaseTimer,
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
    let mut window: Vec<f32> = Vec::with_capacity(WINDOW_SIZE as usize);
    let session_start = Instant::now();
    let mut saw_speech = false;
    let mut speech_started_at: Option<Instant> = None;

    loop {
        match signal_rx.try_recv() {
            Ok(Signal::Cancel) => return finish(capture, Outcome::Cancelled),
            // VAD mode treats an explicit finalize the same as VAD-end-of-speech.
            Ok(Signal::Finalize) => return finish_at_vad_endpoint(capture, timer),
            Err(_) => {}
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

        while window_buf.len() >= WINDOW_SIZE as usize {
            window.clear();
            window.extend(window_buf.drain(..WINDOW_SIZE as usize));
            vad.accept_waveform(&window);
            vad.drain_segments();

            let detected_now = vad.detected();
            if detected_now {
                if !saw_speech {
                    saw_speech = true;
                    speech_started_at = Some(Instant::now());
                }
            } else if saw_speech {
                return finish_at_vad_endpoint(capture, timer);
            }

            if let Some(t) = speech_started_at {
                if t.elapsed() > Duration::from_secs(crate::vad::MAX_SPEECH_S as u64) {
                    return finish_at_vad_endpoint(capture, timer);
                }
            }
        }
    }
}

fn finish_at_vad_endpoint(capture: AudioCapture, mut timer: PhaseTimer) -> Outcome {
    timer.mark_vad_endpoint();
    finish_with_recording(capture, timer)
}

fn run_manual(
    capture: AudioCapture,
    _sample_rate: u32,
    tap_rx: Receiver<Vec<f32>>,
    signal_rx: Receiver<Signal>,
    timer: PhaseTimer,
) -> Outcome {
    let session_start = Instant::now();
    loop {
        // Check controller signals every tick.
        match signal_rx.try_recv() {
            Ok(Signal::Cancel) => return finish(capture, Outcome::Cancelled),
            // Hold-mode endpoint = hotkey release. There's no VAD here,
            // so `t_vad_endpoint` stays absent — `mark_capture_end` (in
            // `finish_with_recording`) becomes the endpoint anchor.
            Ok(Signal::Finalize) => return finish_with_recording(capture, timer),
            Err(_) => {}
        }
        // Drain the tap so the capture thread doesn't back up its channel.
        // We don't need the chunks for anything in Manual mode — the audio
        // is also being accumulated into AudioCapture's internal buffer,
        // which is what `capture.stop()` returns.
        while tap_rx.try_recv().is_ok() {}

        if session_start.elapsed() > MANUAL_MAX_RECORDING {
            return finish_with_recording(capture, timer);
        }
        std::thread::sleep(Duration::from_millis(15));
    }
}

fn finish_with_recording(capture: AudioCapture, mut timer: PhaseTimer) -> Outcome {
    match capture.stop() {
        Ok(rec) => {
            let Recording {
                samples,
                sample_rate,
                channels,
            } = rec;
            if samples.is_empty() {
                return Outcome::Cancelled;
            }
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
            let audio_s = mono.len() as f32 / sample_rate as f32;
            timer.mark_capture_end(audio_s);
            Outcome::Speech {
                samples: mono,
                sample_rate,
                timer,
            }
        }
        Err(e) => Outcome::Error(e),
    }
}

fn finish(capture: AudioCapture, outcome: Outcome) -> Outcome {
    let _ = capture.stop();
    outcome
}
