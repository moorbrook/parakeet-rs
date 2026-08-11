//! Dictation-session driver. Two modes:
//!
//! - **`Mode::VadAutoStop`** (tap-once UX): runs Silero VAD on the capture
//!   stream and finishes the session when it detects end-of-speech.
//! - **`Mode::Manual`** (press-and-hold UX): no VAD — the caller decides
//!   when to stop by calling `Session::finalize()`. Used when the hotkey
//!   itself defines the speech window.

use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use sherpa_onnx::LinearResampler;

use crate::asr::Asr;
use crate::audio::{AudioCapture, Recording};
use crate::endpointing::{
    EndpointEvent, EndpointPolicy, EndpointTracker, SAMPLE_RATE, WINDOW_SAMPLES,
};
use crate::performance::{next_session_id, PhaseTimer, PhaseTimerMode};
use crate::vad::Vad;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointStrategy {
    /// Previous behavior: wait for the full VAD confirmation, then decode.
    Serial,
    /// Decode after the first silent frame, but retain the full confirmation
    /// window before committing the recording.
    Speculative,
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
        /// Transcript decoded during the VAD confirmation window. `None`
        /// means the app must run the normal post-capture decode.
        early_transcript: Option<String>,
        timer: PhaseTimer,
    },
    /// User aborted before any audio was eligible to commit.
    Cancelled,
    /// VAD never saw speech in the timeout window (VadAutoStop mode only).
    NoSpeech,
    Error(anyhow::Error),
}

struct VadSet {
    /// The policy-configured detector is the only authority allowed to end a
    /// recording. Fast and Long-form modes share the same commit path.
    confirming: Vad,
    /// A second low-latency detector may start provisional ASR, but can never
    /// commit a recording by itself.
    candidate: Vad,
}

struct VadRun {
    capture: AudioCapture,
    vad: VadSet,
    asr: Arc<Asr>,
    endpoint_strategy: EndpointStrategy,
    endpoint_policy: EndpointPolicy,
    sample_rate: u32,
    tap_rx: Receiver<Vec<f32>>,
    signal_rx: Receiver<Signal>,
    timer: PhaseTimer,
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
pub fn start(
    vad_model: &Path,
    mode: Mode,
    asr: Arc<Asr>,
    endpoint_policy: EndpointPolicy,
) -> Result<(Session, OutcomeRx)> {
    start_with_strategy(
        vad_model,
        mode,
        asr,
        EndpointStrategy::Speculative,
        endpoint_policy,
    )
}

/// Benchmark seam for comparing the frozen serial pipeline against the
/// production speculative path through identical capture/session code.
pub fn start_with_strategy(
    vad_model: &Path,
    mode: Mode,
    asr: Arc<Asr>,
    endpoint_strategy: EndpointStrategy,
    endpoint_policy: EndpointPolicy,
) -> Result<(Session, OutcomeRx)> {
    start_with_strategy_on_device(
        vad_model,
        mode,
        asr,
        endpoint_strategy,
        endpoint_policy,
        None,
    )
}

/// Identical to [`start_with_strategy`], with an explicit capture device for
/// deterministic loopback benchmarks. Production always passes `None`.
pub fn start_with_strategy_on_device(
    vad_model: &Path,
    mode: Mode,
    asr: Arc<Asr>,
    endpoint_strategy: EndpointStrategy,
    endpoint_policy: EndpointPolicy,
    input_device: Option<&str>,
) -> Result<(Session, OutcomeRx)> {
    let (tap_tx, tap_rx) = channel::<Vec<f32>>();
    let (signal_tx, signal_rx) = channel::<Signal>();
    let (outcome_tx, outcome_rx) = channel::<Outcome>();

    let capture = match input_device {
        Some(name) => AudioCapture::start_with_tap_on_device(tap_tx, Some(name)),
        None => AudioCapture::start_with_tap(tap_tx),
    }
    .context("starting capture")?;
    let sample_rate = capture.sample_rate();

    // Anchor the audio timeline immediately after capture becomes live. VAD
    // construction happens while the microphone records leading silence, so
    // starting this clock after VAD load would shift every sample-derived
    // endpoint earlier than its matching wall-clock instant.
    let timer_mode = if input_device.is_some() {
        PhaseTimerMode::Bench
    } else {
        PhaseTimerMode::Real
    };
    let timer = PhaseTimer::start(timer_mode, next_session_id());

    let vad = if matches!(mode, Mode::VadAutoStop) {
        // Silero is a small RNN; two single-threaded states cost far less than
        // the ASR pass they allow us to hide behind endpoint confirmation.
        Some(VadSet {
            confirming: Vad::load_confirming(vad_model, 1, endpoint_policy)
                .context("loading confirming Silero VAD")?,
            // Also run the early detector in serial benchmark mode so old and
            // new measurements share the exact same acoustic-end anchor. Only
            // the speculative production path acts on its candidate early.
            candidate: Vad::load_candidate(vad_model, 1).context("loading candidate Silero VAD")?,
        })
    } else {
        None
    };

    let join = std::thread::Builder::new()
        .name(match mode {
            Mode::VadAutoStop => "vad-watcher".into(),
            Mode::Manual => "hold-watcher".into(),
        })
        .spawn(move || {
            // VAD-mode-requires-Some-vad and Manual-mode-ignores-vad
            // are both invariant by construction above. The two-arm
            // match here is exhaustive on `mode`; the prior "invalid
            // combination" arm was unreachable.
            let outcome = match mode {
                Mode::VadAutoStop => match vad {
                    Some(vad) => run_vad(VadRun {
                        capture,
                        vad,
                        asr,
                        endpoint_strategy,
                        endpoint_policy,
                        sample_rate,
                        tap_rx,
                        signal_rx,
                        timer,
                    }),
                    None => Outcome::Error(anyhow!("VadAutoStop spawned without a VAD model")),
                },
                Mode::Manual => run_manual(capture, tap_rx, signal_rx, timer),
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

fn run_vad(run: VadRun) -> Outcome {
    let VadRun {
        capture,
        vad,
        asr,
        endpoint_strategy,
        endpoint_policy,
        sample_rate,
        tap_rx,
        signal_rx,
        mut timer,
    } = run;
    let Some(resampler) = LinearResampler::create(sample_rate as i32, SAMPLE_RATE as i32) else {
        let _ = capture.stop();
        return Outcome::Error(anyhow!(
            "could not build {sample_rate}->{SAMPLE_RATE} resampler"
        ));
    };

    let mut window_buf: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize * 4);
    let mut window: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize);
    let mut mono_audio: Vec<f32> = Vec::with_capacity(sample_rate as usize * 5);
    let mut endpoint = EndpointTracker::new(endpoint_policy);
    let mut candidate_speech_end: Option<u64> = None;
    let mut processed_vad_samples: u64 = 0;
    let mut early_transcript: Option<String> = None;
    let session_start = Instant::now();
    let mut saw_speech = false;
    let mut speech_started_at: Option<Instant> = None;

    loop {
        match signal_rx.try_recv() {
            Ok(Signal::Cancel) => return finish(capture, Outcome::Cancelled),
            // VAD mode treats an explicit finalize the same as VAD-end-of-speech.
            Ok(Signal::Finalize) => return finish_at_vad_endpoint(capture, timer, None),
            Err(_) => {}
        }

        // Check the no-speech timeout on EVERY iteration, not only in
        // the `Timeout` arm below. A silent (but live) mic still
        // delivers zero-filled chunks at the device's framerate, so
        // `recv_timeout` returns `Ok` every tick and the Timeout arm
        // never runs. Without this check the capture buffer grows
        // unbounded until `MAX_SPEECH_S` kicks in (which only starts
        // after `saw_speech` flips), i.e. forever if the user never
        // speaks.
        if !saw_speech && session_start.elapsed() > NO_SPEECH_TIMEOUT {
            return finish(capture, Outcome::NoSpeech);
        }

        let chunk = match tap_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(c) => c,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Outcome::Error(anyhow!("audio tap closed before VAD finished"));
            }
        };

        mono_audio.extend_from_slice(&chunk);
        let chunk16 = resampler.resample(&chunk, false);
        if chunk16.is_empty() {
            continue;
        }
        window_buf.extend_from_slice(&chunk16);

        while window_buf.len() >= WINDOW_SAMPLES as usize {
            window.clear();
            window.extend(window_buf.drain(..WINDOW_SAMPLES as usize));
            vad.confirming.accept_waveform(&window);
            vad.confirming.drain_segments();
            processed_vad_samples = processed_vad_samples.saturating_add(u64::from(WINDOW_SAMPLES));

            let detected_now = vad.confirming.detected();
            vad.candidate.accept_waveform(&window);
            vad.candidate.drain_segments();
            let candidate_detected = vad.candidate.detected();
            if detected_now && !saw_speech {
                saw_speech = true;
                speech_started_at = Some(Instant::now());
            }

            if endpoint_strategy == EndpointStrategy::Serial && !detected_now && saw_speech {
                let speech_end_sample = candidate_speech_end.unwrap_or_else(|| {
                    let confirmed_silence_samples =
                        u64::from(endpoint_policy.confirmation_windows())
                            * u64::from(WINDOW_SAMPLES);
                    processed_vad_samples.saturating_sub(confirmed_silence_samples)
                });
                timer
                    .mark_speech_end_at_audio_offset(speech_end_sample as f32 / SAMPLE_RATE as f32);
                return finish_at_vad_endpoint(capture, timer, None);
            }

            let endpoint_event = endpoint.observe(candidate_detected);
            match endpoint_event {
                EndpointEvent::Candidate { speech_end_sample } => {
                    candidate_speech_end = Some(speech_end_sample);
                    log::debug!(
                        "endpoint candidate at {:.3}s",
                        speech_end_sample as f32 / SAMPLE_RATE as f32
                    );
                    if endpoint_strategy == EndpointStrategy::Speculative {
                        timer.mark_asr_start();
                        early_transcript = match asr.recognize(&mono_audio, sample_rate) {
                            Ok(text) if !text.trim().is_empty() => Some(text),
                            Ok(_) => None,
                            Err(error) => {
                                log::warn!(
                                    "speculative ASR failed; final decode will retry: {error:#}"
                                );
                                None
                            }
                        };
                        timer.mark_asr_done();
                    }
                }
                EndpointEvent::SpeechResumed => {
                    log::debug!(
                        "speech resumed at {:.3}s; invalidating endpoint candidate",
                        processed_vad_samples as f32 / SAMPLE_RATE as f32
                    );
                    candidate_speech_end = None;
                    if early_transcript.take().is_some() {
                        log::debug!("speech resumed; discarded speculative transcript");
                    }
                }
                // Local confirmation prevents repeated candidates while the
                // early detector remains silent. The policy-configured
                // confirming detector below still owns the actual stop.
                EndpointEvent::Confirmed { .. } => {}
                EndpointEvent::None => {}
            }

            if endpoint_strategy == EndpointStrategy::Speculative && !detected_now && saw_speech {
                let speech_end_sample = candidate_speech_end.unwrap_or_else(|| {
                    let confirmed_silence_samples =
                        u64::from(endpoint_policy.confirmation_windows())
                            * u64::from(WINDOW_SAMPLES);
                    processed_vad_samples.saturating_sub(confirmed_silence_samples)
                });
                timer
                    .mark_speech_end_at_audio_offset(speech_end_sample as f32 / SAMPLE_RATE as f32);
                return finish_at_vad_endpoint(capture, timer, early_transcript);
            }

            if let Some(t) = speech_started_at {
                if t.elapsed() > Duration::from_secs(crate::vad::MAX_SPEECH_S as u64) {
                    return finish_at_vad_endpoint(capture, timer, None);
                }
            }
        }
    }
}

fn finish_at_vad_endpoint(
    capture: AudioCapture,
    mut timer: PhaseTimer,
    early_transcript: Option<String>,
) -> Outcome {
    timer.mark_vad_endpoint();
    finish_with_recording(capture, timer, early_transcript)
}

fn run_manual(
    capture: AudioCapture,
    tap_rx: Receiver<Vec<f32>>,
    signal_rx: Receiver<Signal>,
    mut timer: PhaseTimer,
) -> Outcome {
    let session_start = Instant::now();
    loop {
        // Check controller signals every tick.
        match signal_rx.try_recv() {
            Ok(Signal::Cancel) => return finish(capture, Outcome::Cancelled),
            // Hold-mode endpoint = hotkey release. Mark the endpoint
            // HERE (release entry), NOT inside `finish_with_recording`.
            // `finish_with_recording` runs `capture.stop()` which
            // joins the audio thread — that gap (a few ms) is the
            // user's actual release-to-paste latency and shouldn't
            // be excluded from `dur_post_endpoint_ms`.
            Ok(Signal::Finalize) => {
                timer.mark_vad_endpoint();
                return finish_with_recording(capture, timer, None);
            }
            Err(_) => {}
        }
        // Drain the tap so the capture thread doesn't back up its channel.
        // We don't need the chunks for anything in Manual mode — the audio
        // is also being accumulated into AudioCapture's internal buffer,
        // which is what `capture.stop()` returns.
        while tap_rx.try_recv().is_ok() {}

        if session_start.elapsed() > MANUAL_MAX_RECORDING {
            // Auto-cap fallback for "user forgot to release". Mark
            // the cap-hit moment as the endpoint so latency math
            // doesn't include the post-cap stop+join overhead either.
            timer.mark_vad_endpoint();
            return finish_with_recording(capture, timer, None);
        }
        std::thread::sleep(Duration::from_millis(15));
    }
}

fn finish_with_recording(
    capture: AudioCapture,
    mut timer: PhaseTimer,
    early_transcript: Option<String>,
) -> Outcome {
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
                early_transcript,
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
