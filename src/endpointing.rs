//! Early speech-end candidate tracking.
//!
//! Silero is configured to expose the first silent 32 ms frame so ASR can
//! begin speculatively. This tracker invalidates a provisional transcript if
//! speech resumes and locally suppresses duplicate candidates after five
//! silent frames. It never owns the production stop decision: an independent
//! Silero state with the original 150 ms configuration is the sole authority
//! allowed to commit the recording.

/// Silero's native sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// One Silero inference frame (32 ms at 16 kHz).
pub const WINDOW_SAMPLES: u32 = 512;
/// The existing cutoff-safety policy. It quantizes to five 32 ms frames and
/// configures the independent confirming detector in `vad.rs`.
pub const CONFIRM_SILENCE_MS: u32 = 150;
pub const CONFIRM_SILENCE_SAMPLES: u32 = SAMPLE_RATE * CONFIRM_SILENCE_MS / 1_000;
pub const CONFIRM_SILENCE_WINDOWS: u32 = CONFIRM_SILENCE_SAMPLES.div_ceil(WINDOW_SAMPLES);
/// Make the candidate detector expose an edge after one silent frame.
pub const SPECULATIVE_MIN_SILENCE_S: f32 = WINDOW_SAMPLES as f32 / SAMPLE_RATE as f32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndpointEvent {
    None,
    /// First silent frame after speech. `speech_end_sample` is the beginning
    /// of that frame on the 16 kHz VAD timeline.
    Candidate {
        speech_end_sample: u64,
    },
    /// Speech returned before the safety window elapsed.
    SpeechResumed,
    /// The candidate survived the tracker's local confirmation window. The
    /// production pipeline still waits for its independent confirming VAD.
    Confirmed {
        speech_end_sample: u64,
    },
}

#[derive(Debug, Default)]
pub struct EndpointTracker {
    processed_samples: u64,
    saw_speech: bool,
    candidate_speech_end: Option<u64>,
    consecutive_silent_windows: u32,
    confirmed: bool,
}

impl EndpointTracker {
    pub fn observe(&mut self, detected: bool) -> EndpointEvent {
        self.processed_samples = self
            .processed_samples
            .saturating_add(u64::from(WINDOW_SAMPLES));

        if detected {
            self.saw_speech = true;
            self.consecutive_silent_windows = 0;
            self.confirmed = false;
            return if self.candidate_speech_end.take().is_some() {
                EndpointEvent::SpeechResumed
            } else {
                EndpointEvent::None
            };
        }

        if !self.saw_speech || self.confirmed {
            return EndpointEvent::None;
        }

        self.consecutive_silent_windows = self.consecutive_silent_windows.saturating_add(1);
        let speech_end_sample = *self.candidate_speech_end.get_or_insert_with(|| {
            self.processed_samples
                .saturating_sub(u64::from(WINDOW_SAMPLES))
        });

        if self.consecutive_silent_windows >= CONFIRM_SILENCE_WINDOWS {
            self.confirmed = true;
            EndpointEvent::Confirmed { speech_end_sample }
        } else if self.consecutive_silent_windows == 1 {
            EndpointEvent::Candidate { speech_end_sample }
        } else {
            EndpointEvent::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirmation_preserves_the_150ms_safety_window() {
        let windows = std::hint::black_box(CONFIRM_SILENCE_WINDOWS);
        let window_samples = std::hint::black_box(WINDOW_SAMPLES);
        let confirmation_samples = std::hint::black_box(CONFIRM_SILENCE_SAMPLES);
        assert_eq!(windows, 5);
        assert!(windows * window_samples >= confirmation_samples);
        assert!((windows - 1) * window_samples < confirmation_samples);
    }

    #[test]
    fn candidate_is_early_but_commit_waits_for_five_silent_frames() {
        let mut tracker = EndpointTracker::default();
        assert_eq!(tracker.observe(true), EndpointEvent::None);
        assert_eq!(
            tracker.observe(false),
            EndpointEvent::Candidate {
                speech_end_sample: u64::from(WINDOW_SAMPLES)
            }
        );
        for _ in 2..CONFIRM_SILENCE_WINDOWS {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
        assert_eq!(
            tracker.observe(false),
            EndpointEvent::Confirmed {
                speech_end_sample: u64::from(WINDOW_SAMPLES)
            }
        );
    }

    #[test]
    fn resumed_speech_invalidates_the_candidate() {
        let mut tracker = EndpointTracker::default();
        tracker.observe(true);
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Candidate { .. }
        ));
        assert_eq!(tracker.observe(true), EndpointEvent::SpeechResumed);
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Candidate { .. }
        ));
    }

    #[test]
    fn silence_before_first_speech_never_ends_the_session() {
        let mut tracker = EndpointTracker::default();
        for _ in 0..(CONFIRM_SILENCE_WINDOWS * 3) {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
    }
}
