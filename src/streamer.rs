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

use anyhow::{anyhow, bail, Context, Result};

use crate::asr::Asr;
use crate::audio::{AudioCapture, Recording};
use crate::endpointing::{
    EndpointEvent, EndpointPolicy, EndpointTracker, SAMPLE_RATE, WINDOW_SAMPLES,
};
use crate::performance::{next_session_id, PhaseTimer, PhaseTimerMode};
use crate::vad::Vad;
use crate::windows::{
    merge_words, words_to_text, HoldWindowConfig, SeamOutcome, WindowPlanner, Word,
};

/// If the user starts dictation and says nothing within this window, give up.
const NO_SPEECH_TIMEOUT: Duration = Duration::from_secs(5);

/// Hold-mode safety cap: refuse to record longer than this even if the user
/// keeps the key held. Matches the VAD's `max_speech_duration` so both modes
/// have the same upper bound on a single utterance.
const MANUAL_MAX_RECORDING: Duration = Duration::from_secs(30);

/// How long the Hold loop blocks on the audio tap before re-checking the
/// hotkey signal. This is the floor on release-to-observed latency, so it is
/// deliberately shorter than a capture callback period rather than the 15 ms
/// sleep that used to cost 12-14 ms of the measured baseline.
const MANUAL_POLL: Duration = Duration::from_millis(3);

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
    /// End of speech reached. Carries mono samples already at
    /// [`SAMPLE_RATE`]; `AudioCapture` resampled them during the capture
    /// callbacks, so no conversion work remains on this path.
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
    hold_windows: HoldWindowConfig,
) -> Result<(Session, OutcomeRx)> {
    start_with_strategy(
        vad_model,
        mode,
        asr,
        EndpointStrategy::Speculative,
        endpoint_policy,
        hold_windows,
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
    hold_windows: HoldWindowConfig,
) -> Result<(Session, OutcomeRx)> {
    start_with_strategy_on_device(
        vad_model,
        mode,
        asr,
        endpoint_strategy,
        endpoint_policy,
        hold_windows,
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
    hold_windows: HoldWindowConfig,
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

    // Hold mode cuts windows at pauses so most of the recording is already
    // decoded by the time the key is released. It needs its own Silero state
    // (Tap's two belong to the endpoint authority) and a recognizer that
    // reports word boundaries, without which two windows cannot be joined.
    // Anything missing leaves Hold on the original single-decode path.
    let windowing = if matches!(mode, Mode::Manual) {
        build_hold_windowing(vad_model, &asr, hold_windows)
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
                        tap_rx,
                        signal_rx,
                        timer,
                    }),
                    None => Outcome::Error(anyhow!("VadAutoStop spawned without a VAD model")),
                },
                Mode::Manual => run_manual(capture, tap_rx, signal_rx, timer, windowing),
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

/// Decode a recording exactly as a Hold session would have, from a buffer
/// rather than a live microphone.
///
/// Quality harnesses need to compare a windowed transcript against the plain
/// single-pass one on the same audio, which the live path cannot give them
/// without a loopback device and a real hold. The VAD, the planner, and the
/// merge are the same code; only the audio source and the threading differ,
/// and the merge is order-deterministic so decoding the windows in sequence
/// here yields the transcript the session would have produced.
pub fn decode_hold_windowed(
    asr: &Asr,
    vad_model: &Path,
    samples: &[f32],
    config: HoldWindowConfig,
) -> Result<String> {
    config.validate().map_err(|reason| anyhow!(reason))?;
    if !asr.reports_token_spans() {
        bail!("this recognizer reports no word boundaries, so windows cannot be joined");
    }
    let vad = Vad::load_candidate(vad_model, 1).context("loading Silero for windowed decode")?;
    let mut planner = WindowPlanner::new(config, u64::from(WINDOW_SAMPLES));
    let mut merged: Vec<Word> = Vec::new();

    let frames = samples.len() / WINDOW_SAMPLES as usize;
    for index in 0..frames {
        let frame =
            &samples[index * WINDOW_SAMPLES as usize..(index + 1) * WINDOW_SAMPLES as usize];
        vad.accept_waveform(frame);
        vad.drain_segments();
        let Some(cut) = planner.observe_frame(vad.detected()) else {
            continue;
        };
        let from = cut.start as usize;
        let to = (cut.end as usize).min(samples.len());
        if from >= to {
            continue;
        }
        let words = asr.recognize_window(&samples[from..to], SAMPLE_RATE, cut.start_seconds())?;
        merged = merge_words(&merged, &words, cut.start_seconds()).0;
    }

    let tail_start = (planner.open_window_start() as usize).min(samples.len());
    let tail_start_s = tail_start as f32 / SAMPLE_RATE as f32;
    let tail = asr.recognize_window(&samples[tail_start..], SAMPLE_RATE, tail_start_s)?;
    merged = merge_words(&merged, &tail, tail_start_s).0;
    Ok(words_to_text(&merged))
}

/// Build the Hold-mode window state, or `None` with a reason logged. Never an
/// error: dictation must still work when windowing cannot.
fn build_hold_windowing(
    vad_model: &Path,
    asr: &Arc<Asr>,
    config: HoldWindowConfig,
) -> Option<HoldWindowing> {
    if !config.enabled {
        log::debug!("hold windowing off by configuration");
        return None;
    }
    if !asr.reports_token_spans() {
        log::info!("hold windowing off: this recognizer reports no word boundaries");
        return None;
    }
    if let Err(reason) = config.validate() {
        log::warn!("hold windowing off: {reason}");
        return None;
    }
    let vad = match Vad::load_candidate(vad_model, 1) {
        Ok(vad) => vad,
        Err(error) => {
            log::warn!("hold windowing off: loading Silero failed: {error:#}");
            return None;
        }
    };
    let decoder = match WindowDecoder::spawn(asr.clone()) {
        Ok(decoder) => decoder,
        Err(error) => {
            log::warn!("hold windowing off: {error:#}");
            return None;
        }
    };
    Some(HoldWindowing {
        vad,
        planner: WindowPlanner::new(config, u64::from(WINDOW_SAMPLES)),
        decoder,
    })
}

fn run_vad(run: VadRun) -> Outcome {
    let VadRun {
        capture,
        vad,
        asr,
        endpoint_strategy,
        endpoint_policy,
        tap_rx,
        signal_rx,
        mut timer,
    } = run;

    // The tap already delivers SAMPLE_RATE mono: `AudioCapture` resamples in
    // its capture callbacks, so both the VAD and the speculative ASR read the
    // same samples with no conversion left to do here. ADR-0030.
    let mut window_buf: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize * 4);
    let mut window: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize);
    let mut mono_audio: Vec<f32> = Vec::with_capacity(SAMPLE_RATE as usize * 5);
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

        window_buf.extend_from_slice(&chunk);
        mono_audio.extend(chunk);

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
                        early_transcript = match asr.recognize(&mono_audio, SAMPLE_RATE) {
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

/// One window handed to the background decoder. `start_s` is where the window
/// begins in the recording, which is also where its overlap with the previous
/// window begins.
struct WindowJob {
    samples: Vec<f32>,
    start_s: f32,
    queued_at: Instant,
    is_tail: bool,
}

/// What the background decoder produced for a whole held recording.
#[derive(Default)]
struct WindowSummary {
    words: Vec<Word>,
    windows: usize,
    seams: Vec<SeamOutcome>,
    /// How long the tail window sat behind an already-running window decode.
    /// Inherent: the worker serves one request at a time.
    tail_queue_wait: Duration,
    tail_decode: Duration,
    /// First failure. Any failure abandons the whole windowed transcript and
    /// the session falls back to decoding the recording in one pass, so a seam
    /// bug can never lose what the user said.
    error: Option<String>,
}

/// Background decoder for Hold-mode windows.
///
/// One thread, one job queue, results merged as they arrive. A single thread is
/// what the worker protocol wants anyway — it serves one request at a time —
/// and joining it before the session ends keeps a stray decode from holding the
/// worker while the next session starts.
struct WindowDecoder {
    job_tx: Option<Sender<WindowJob>>,
    join: JoinHandle<WindowSummary>,
}

impl WindowDecoder {
    fn spawn(asr: Arc<Asr>) -> Result<Self> {
        let (job_tx, job_rx) = channel::<WindowJob>();
        let join = std::thread::Builder::new()
            .name("hold-windows".into())
            .spawn(move || {
                let mut summary = WindowSummary::default();
                for job in job_rx {
                    let queued_for = job.queued_at.elapsed();
                    let started = Instant::now();
                    if summary.error.is_none() {
                        match asr.recognize_window(&job.samples, SAMPLE_RATE, job.start_s) {
                            Ok(next) => {
                                let (merged, seam) =
                                    merge_words(&summary.words, &next, job.start_s);
                                summary.words = merged;
                                summary.seams.push(seam);
                            }
                            Err(error) => summary.error = Some(format!("{error:#}")),
                        }
                        summary.windows += 1;
                    }
                    if job.is_tail {
                        summary.tail_queue_wait = queued_for;
                        summary.tail_decode = started.elapsed();
                    }
                }
                summary
            })
            .context("spawning the Hold window decoder")?;
        Ok(Self {
            job_tx: Some(job_tx),
            join,
        })
    }

    /// Queue a window. A send failure means the decoder thread died, which is
    /// reported when the session joins it.
    fn submit(&self, samples: Vec<f32>, start_s: f32, is_tail: bool) {
        if let Some(tx) = &self.job_tx {
            let _ = tx.send(WindowJob {
                samples,
                start_s,
                queued_at: Instant::now(),
                is_tail,
            });
        }
    }

    /// Close the queue and wait for every submitted window.
    fn finish(mut self) -> WindowSummary {
        self.job_tx = None;
        self.join.join().unwrap_or_else(|_| WindowSummary {
            error: Some("Hold window decoder panicked".to_string()),
            ..WindowSummary::default()
        })
    }
}

/// Silero state and the cut planner for one held recording.
struct HoldWindowing {
    vad: Vad,
    planner: WindowPlanner,
    decoder: WindowDecoder,
}

fn run_manual(
    capture: AudioCapture,
    tap_rx: Receiver<Vec<f32>>,
    signal_rx: Receiver<Signal>,
    mut timer: PhaseTimer,
    mut windowing: Option<HoldWindowing>,
) -> Outcome {
    let session_start = Instant::now();
    // Only needed while windows are being cut: the recording itself is
    // accumulated by `AudioCapture`, and `capture.stop()` is the authority for
    // what a window contains. The tap is read for VAD and to know how far the
    // recording has got.
    let mut window_buf: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize * 4);
    let mut frame: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize);
    let mut cut_audio: Vec<f32> = Vec::with_capacity(SAMPLE_RATE as usize * 8);
    let mut cut_audio_start: u64 = 0;

    loop {
        // Check controller signals every tick.
        match signal_rx.try_recv() {
            Ok(Signal::Cancel) => {
                if let Some(windowing) = windowing {
                    // Join before returning so no decode outlives the session.
                    let _ = windowing.decoder.finish();
                }
                return finish(capture, Outcome::Cancelled);
            }
            // Hold-mode endpoint = hotkey release. Mark the endpoint
            // HERE (release entry), NOT inside `finish_manual`.
            // `finish_manual` runs `capture.stop()` which joins the
            // audio thread — that gap (a few ms) is the user's actual
            // release-to-paste latency and shouldn't be excluded from
            // `dur_post_endpoint_ms`.
            Ok(Signal::Finalize) => {
                timer.mark_vad_endpoint();
                return finish_manual(capture, timer, windowing);
            }
            Err(_) => {}
        }

        // Drain the tap so the capture thread doesn't back up its channel.
        // Blocking briefly on the first chunk replaces the old fixed sleep:
        // it wakes on audio rather than on a timer, so the release edge is
        // seen sooner.
        let mut chunk = match tap_rx.recv_timeout(MANUAL_POLL) {
            Ok(chunk) => Some(chunk),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // Capture ended under us. The recording is still whatever
                // `AudioCapture` accumulated, so give up on windowing rather
                // than on the session, and stop spinning on a dead channel.
                if windowing.take().is_some() {
                    log::warn!("audio tap closed during hold; decoding the recording in one pass");
                }
                std::thread::sleep(MANUAL_POLL);
                None
            }
        };
        while let Some(samples) = chunk {
            if let Some(state) = &mut windowing {
                window_buf.extend_from_slice(&samples);
                cut_audio.extend_from_slice(&samples);
                while window_buf.len() >= WINDOW_SAMPLES as usize {
                    frame.clear();
                    frame.extend(window_buf.drain(..WINDOW_SAMPLES as usize));
                    state.vad.accept_waveform(&frame);
                    state.vad.drain_segments();
                    let Some(cut) = state.planner.observe_frame(state.vad.detected()) else {
                        continue;
                    };
                    // `cut_audio` starts at `cut_audio_start`; both bounds are
                    // inside it by construction, since the planner never cuts
                    // past what it has been fed and never rewinds before the
                    // window it just closed.
                    let from = (cut.start.saturating_sub(cut_audio_start)) as usize;
                    let to = (cut.end.saturating_sub(cut_audio_start)) as usize;
                    let to = to.min(cut_audio.len());
                    if from >= to {
                        log::warn!("hold window {from}..{to} is empty; skipping the cut");
                        continue;
                    }
                    log::debug!(
                        "hold window {:.2}s..{:.2}s ({:?}), next starts at {:.2}s",
                        cut.start_seconds(),
                        cut.end as f32 / SAMPLE_RATE as f32,
                        cut.reason,
                        cut.overlap_start_seconds(),
                    );
                    state
                        .decoder
                        .submit(cut_audio[from..to].to_vec(), cut.start_seconds(), false);
                    // Keep only what the next window still needs.
                    let keep = (cut.next_start.saturating_sub(cut_audio_start)) as usize;
                    cut_audio.drain(..keep.min(cut_audio.len()));
                    cut_audio_start = cut.next_start;
                }
            }
            chunk = tap_rx.try_recv().ok();
        }

        if session_start.elapsed() > MANUAL_MAX_RECORDING {
            // Auto-cap fallback for "user forgot to release". Mark
            // the cap-hit moment as the endpoint so latency math
            // doesn't include the post-cap stop+join overhead either.
            timer.mark_vad_endpoint();
            return finish_manual(capture, timer, windowing);
        }
    }
}

/// Stop capture, decode whatever tail is left, and merge it onto the windows
/// already decoded during the hold.
fn finish_manual(
    capture: AudioCapture,
    mut timer: PhaseTimer,
    windowing: Option<HoldWindowing>,
) -> Outcome {
    let recording = match capture.stop() {
        Ok(recording) => recording,
        Err(error) => {
            if let Some(windowing) = windowing {
                let _ = windowing.decoder.finish();
            }
            return Outcome::Error(error);
        }
    };
    let Recording {
        samples,
        sample_rate,
    } = recording;
    if samples.is_empty() {
        if let Some(windowing) = windowing {
            let _ = windowing.decoder.finish();
        }
        return Outcome::Cancelled;
    }
    let audio_s = samples.len() as f32 / sample_rate as f32;
    timer.mark_capture_end(audio_s);

    let Some(HoldWindowing {
        planner, decoder, ..
    }) = windowing
    else {
        return Outcome::Speech {
            samples,
            sample_rate,
            early_transcript: None,
            timer,
        };
    };

    // The planner counts on the VAD's 16 kHz timeline. Since ADR-0030 capture
    // delivers exactly that, but check rather than slice on an assumption.
    if sample_rate != SAMPLE_RATE {
        log::warn!(
            "hold windowing expected {SAMPLE_RATE} Hz capture, got {sample_rate};              decoding the recording in one pass"
        );
        let _ = decoder.finish();
        return Outcome::Speech {
            samples,
            sample_rate,
            early_transcript: None,
            timer,
        };
    }

    timer.mark_asr_start();
    let tail_start = (planner.open_window_start() as usize).min(samples.len());
    let tail_s = (samples.len() - tail_start) as f32 / SAMPLE_RATE as f32;
    decoder.submit(
        samples[tail_start..].to_vec(),
        tail_start as f32 / SAMPLE_RATE as f32,
        true,
    );
    let summary = decoder.finish();
    timer.mark_asr_done();

    let early_transcript = match summary.error {
        Some(error) => {
            log::warn!("hold window decode failed ({error}); decoding the recording in one pass");
            None
        }
        None => {
            let text = words_to_text(&summary.words);
            log::info!(
                "hold windows={} seams={:?} tail={tail_s:.2}s queued={:.1}ms decode={:.1}ms",
                summary.windows,
                summary.seams,
                summary.tail_queue_wait.as_secs_f32() * 1_000.0,
                summary.tail_decode.as_secs_f32() * 1_000.0,
            );
            // An empty merge on non-empty audio is indistinguishable from a
            // seam that ate the transcript, so re-decode rather than paste
            // nothing.
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        }
    };
    Outcome::Speech {
        samples,
        sample_rate,
        early_transcript,
        timer,
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
            } = rec;
            if samples.is_empty() {
                return Outcome::Cancelled;
            }
            let audio_s = samples.len() as f32 / sample_rate as f32;
            timer.mark_capture_end(audio_s);
            Outcome::Speech {
                samples,
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
