//! Early speech-end candidate tracking.
//!
//! Silero is configured to expose the first silent 32 ms frame so ASR can
//! begin speculatively. This tracker invalidates a provisional transcript if
//! speech resumes and locally suppresses duplicate candidates after the
//! selected confirmation window. It never owns the production stop decision:
//! an independent Silero state with the same window is the sole authority
//! allowed to commit the recording.

/// Silero's native sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// One Silero inference frame (32 ms at 16 kHz).
pub const WINDOW_SAMPLES: u32 = 512;
/// Make the candidate detector expose an edge after one silent frame.
pub const SPECULATIVE_MIN_SILENCE_S: f32 = WINDOW_SAMPLES as f32 / SAMPLE_RATE as f32;

/// Tap Fast confirmation window. Below roughly this value the synchronous
/// speculative decode, not the window, sets the floor; see ADR-0031.
pub const FAST_CONFIRMATION_MS: u32 = 90;
/// Pause-friendly Tap confirmation window.
pub const LONG_FORM_CONFIRMATION_MS: u32 = 750;

/// Product-level pause policy for tap-to-dictate sessions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EndpointPolicy {
    /// Low-latency mode for short commands. May split prose at an
    /// intra-sentence pause.
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
            Self::Fast => FAST_CONFIRMATION_MS,
            Self::LongForm => LONG_FORM_CONFIRMATION_MS,
        }
    }

    pub const fn confirmation_windows(self) -> u32 {
        confirmation_windows(self.confirmation_ms())
    }
}

/// Silero frames covering `confirmation_ms`, rounded up.
pub const fn confirmation_windows(confirmation_ms: u32) -> u32 {
    let samples = SAMPLE_RATE * confirmation_ms / 1_000;
    samples.div_ceil(WINDOW_SAMPLES)
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
        Self::new(EndpointPolicy::default().confirmation_ms())
    }
}

impl EndpointTracker {
    /// `confirmation_ms` is the session's resolved silence window, not a
    /// policy: the sweep benchmark drives values the two shipping policies do
    /// not name, and the confirming Silero state is loaded from the same value
    /// so speculation and capture shutdown cannot disagree.
    pub fn new(confirmation_ms: u32) -> Self {
        Self {
            confirmation_windows: confirmation_windows(confirmation_ms),
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
        assert_eq!(EndpointPolicy::Fast.confirmation_windows(), 3);
        assert_eq!(EndpointPolicy::LongForm.confirmation_windows(), 24);
    }

    #[test]
    fn candidate_is_early_but_fast_commit_waits_for_the_whole_window() {
        let policy = EndpointPolicy::Fast;
        let mut tracker = EndpointTracker::new(policy.confirmation_ms());
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
    fn the_sweep_window_is_honored_independently_of_the_shipping_policies() {
        // 150 ms is the pre-ADR-0031 Fast window and names no policy now.
        let mut tracker = EndpointTracker::new(150);
        tracker.observe(true);
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Candidate { .. }
        ));
        for _ in 2..5 {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Confirmed { .. }
        ));
    }

    #[test]
    fn resumed_speech_invalidates_the_candidate() {
        let mut tracker = EndpointTracker::new(EndpointPolicy::Fast.confirmation_ms());
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
        let mut tracker = EndpointTracker::new(policy.confirmation_ms());
        for _ in 0..(policy.confirmation_windows() * 3) {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
    }

    #[test]
    fn long_form_waits_through_a_640ms_pause() {
        let policy = EndpointPolicy::LongForm;
        let mut tracker = EndpointTracker::new(policy.confirmation_ms());
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
