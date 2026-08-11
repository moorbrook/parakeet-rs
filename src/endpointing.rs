//! Early speech-end candidate tracking.
//!
//! Silero is configured to expose the first silent 32 ms frame so ASR can
//! begin speculatively. This tracker invalidates a provisional transcript if
//! speech resumes and locally suppresses duplicate candidates after the
//! selected confirmation window. It never owns the production stop decision:
//! an independent Silero state with the same policy is the sole authority
//! allowed to commit the recording.

/// Silero's native sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// One Silero inference frame (32 ms at 16 kHz).
pub const WINDOW_SAMPLES: u32 = 512;
/// Make the candidate detector expose an edge after one silent frame.
pub const SPECULATIVE_MIN_SILENCE_S: f32 = WINDOW_SAMPLES as f32 / SAMPLE_RATE as f32;

/// Product-level pause policy for tap-to-dictate sessions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EndpointPolicy {
    /// Preserve the original low-latency 150 ms behavior. Useful for short
    /// commands, but may split prose at an intra-sentence pause.
    Fast,
    /// Wait through a natural clause/sentence pause before committing. This
    /// is the default because a false stop loses speech; speculative ASR hides
    /// most of the longer confirmation window from inference latency.
    #[default]
    LongForm,
}

impl EndpointPolicy {
    pub const fn confirmation_ms(self) -> u32 {
        match self {
            Self::Fast => 150,
            Self::LongForm => 750,
        }
    }

    pub const fn confirmation_windows(self) -> u32 {
        let samples = SAMPLE_RATE * self.confirmation_ms() / 1_000;
        samples.div_ceil(WINDOW_SAMPLES)
    }
}

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

#[derive(Debug)]
pub struct EndpointTracker {
    confirmation_windows: u32,
    processed_samples: u64,
    saw_speech: bool,
    candidate_speech_end: Option<u64>,
    consecutive_silent_windows: u32,
    confirmed: bool,
}

impl Default for EndpointTracker {
    fn default() -> Self {
        Self::new(EndpointPolicy::default())
    }
}

impl EndpointTracker {
    pub fn new(policy: EndpointPolicy) -> Self {
        Self {
            confirmation_windows: policy.confirmation_windows(),
            processed_samples: 0,
            saw_speech: false,
            candidate_speech_end: None,
            consecutive_silent_windows: 0,
            confirmed: false,
        }
    }

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

        if self.consecutive_silent_windows >= self.confirmation_windows {
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
    fn confirmation_windows_quantize_each_policy_upward() {
        let window_samples = std::hint::black_box(WINDOW_SAMPLES);
        for policy in [EndpointPolicy::Fast, EndpointPolicy::LongForm] {
            let windows = std::hint::black_box(policy.confirmation_windows());
            let confirmation_samples =
                std::hint::black_box(SAMPLE_RATE * policy.confirmation_ms() / 1_000);
            assert!(windows * window_samples >= confirmation_samples);
            assert!((windows - 1) * window_samples < confirmation_samples);
        }
        assert_eq!(EndpointPolicy::Fast.confirmation_windows(), 5);
        assert_eq!(EndpointPolicy::LongForm.confirmation_windows(), 24);
    }

    #[test]
    fn candidate_is_early_but_fast_commit_waits_for_five_silent_frames() {
        let policy = EndpointPolicy::Fast;
        let mut tracker = EndpointTracker::new(policy);
        assert_eq!(tracker.observe(true), EndpointEvent::None);
        assert_eq!(
            tracker.observe(false),
            EndpointEvent::Candidate {
                speech_end_sample: u64::from(WINDOW_SAMPLES)
            }
        );
        for _ in 2..policy.confirmation_windows() {
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
        let mut tracker = EndpointTracker::new(EndpointPolicy::Fast);
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
        let policy = EndpointPolicy::LongForm;
        let mut tracker = EndpointTracker::new(policy);
        for _ in 0..(policy.confirmation_windows() * 3) {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
    }

    #[test]
    fn long_form_waits_through_a_640ms_pause() {
        let policy = EndpointPolicy::LongForm;
        let mut tracker = EndpointTracker::new(policy);
        tracker.observe(true);
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Candidate { .. }
        ));
        for _ in 1..20 {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
        assert_eq!(tracker.observe(true), EndpointEvent::SpeechResumed);
    }
}
