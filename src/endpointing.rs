//! Early speech-end candidate tracking.
//!
//! Silero is configured to expose the first silent 32 ms frame so ASR can
//! begin speculatively. This tracker invalidates a provisional transcript if
//! speech resumes and locally suppresses duplicate candidates after the
//! selected confirmation window.
//!
//! Two commit authorities exist, and both are driven from silence:
//!
//! - The confirming Silero state, configured with
//!   [`EndpointConfig::confirmation_ms`], owns every ordinary stop.
//! - When the provisional transcript already ends a sentence, the tracker may
//!   commit at the shorter [`EndpointConfig::punctuated_confirmation_ms`]
//!   using the candidate detector's silent-frame count. The candidate detector
//!   re-arms on a single 32 ms speech frame, so it is strictly more
//!   resume-sensitive than the confirming state it short-circuits.

/// Silero's native sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// One Silero inference frame (32 ms at 16 kHz).
pub const WINDOW_SAMPLES: u32 = 512;
/// Make the candidate detector expose an edge after one silent frame.
pub const SPECULATIVE_MIN_SILENCE_S: f32 = WINDOW_SAMPLES as f32 / SAMPLE_RATE as f32;

/// Ordinary Tap Fast confirmation window.
pub const FAST_CONFIRMATION_MS: u32 = 150;
/// Tap Fast window when the provisional transcript ends a sentence.
pub const FAST_PUNCTUATED_CONFIRMATION_MS: u32 = 90;
/// Pause-friendly Tap confirmation window.
pub const LONG_FORM_CONFIRMATION_MS: u32 = 750;

/// Product-level pause policy for tap-to-dictate sessions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EndpointPolicy {
    /// Preserve the original low-latency behavior. Useful for short
    /// commands, but may split prose at an intra-sentence pause.
    Fast,
    /// Wait through a natural clause/sentence pause before committing. This
    /// is the default because a false stop loses speech; speculative ASR hides
    /// most of the longer confirmation window from inference latency.
    #[default]
    LongForm,
}

impl EndpointPolicy {
    pub const fn config(self) -> EndpointConfig {
        match self {
            Self::Fast => EndpointConfig {
                confirmation_ms: FAST_CONFIRMATION_MS,
                punctuated_confirmation_ms: Some(FAST_PUNCTUATED_CONFIRMATION_MS),
            },
            // Long-form exists to survive intra-utterance pauses, and a
            // sentence-final period is exactly what a speaker emits before
            // one. Punctuation is not a discriminator here, so this policy
            // keeps a single authority.
            Self::LongForm => EndpointConfig {
                confirmation_ms: LONG_FORM_CONFIRMATION_MS,
                punctuated_confirmation_ms: None,
            },
        }
    }
}

/// Resolved silence thresholds for one session. Production builds this from
/// [`EndpointPolicy`]; the endpoint sweep benchmark constructs it directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EndpointConfig {
    /// Silence required before an ordinary commit.
    pub confirmation_ms: u32,
    /// Silence required when the provisional transcript ends a sentence.
    /// `None` disables the punctuation-aware commit entirely.
    pub punctuated_confirmation_ms: Option<u32>,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        EndpointPolicy::default().config()
    }
}

const fn windows_for(ms: u32) -> u32 {
    let samples = SAMPLE_RATE * ms / 1_000;
    samples.div_ceil(WINDOW_SAMPLES)
}

impl EndpointConfig {
    pub const fn confirmation_windows(self) -> u32 {
        windows_for(self.confirmation_ms)
    }

    pub const fn punctuated_windows(self) -> Option<u32> {
        match self.punctuated_confirmation_ms {
            Some(ms) => Some(windows_for(ms)),
            None => None,
        }
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
    /// A sentence-terminated provisional transcript survived the shorter
    /// punctuated window. This event alone may end the recording.
    PunctuatedCommit {
        speech_end_sample: u64,
    },
}

#[derive(Debug)]
pub struct EndpointTracker {
    confirmation_windows: u32,
    punctuated_windows: Option<u32>,
    punctuated_armed: bool,
    processed_samples: u64,
    saw_speech: bool,
    candidate_speech_end: Option<u64>,
    consecutive_silent_windows: u32,
    confirmed: bool,
}

impl Default for EndpointTracker {
    fn default() -> Self {
        Self::new(EndpointConfig::default())
    }
}

impl EndpointTracker {
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            confirmation_windows: config.confirmation_windows(),
            punctuated_windows: config.punctuated_windows(),
            punctuated_armed: false,
            processed_samples: 0,
            saw_speech: false,
            candidate_speech_end: None,
            consecutive_silent_windows: 0,
            confirmed: false,
        }
    }

    /// Record whether the provisional transcript ends a sentence.
    ///
    /// The speculative decode blocks the VAD loop for tens of milliseconds, so
    /// by the time a transcript exists the silent-frame count may already have
    /// passed the punctuated threshold. Returning the commit event here rather
    /// than waiting for the next `observe` avoids paying an extra 32 ms frame
    /// for that catch-up.
    pub fn arm_punctuated(&mut self, punctuated: bool) -> Option<EndpointEvent> {
        self.punctuated_armed = punctuated;
        if !punctuated {
            return None;
        }
        self.punctuated_commit()
    }

    fn punctuated_commit(&mut self) -> Option<EndpointEvent> {
        if self.confirmed || !self.punctuated_armed {
            return None;
        }
        let threshold = self.punctuated_windows?;
        let speech_end_sample = self.candidate_speech_end?;
        if self.consecutive_silent_windows < threshold {
            return None;
        }
        self.confirmed = true;
        Some(EndpointEvent::PunctuatedCommit { speech_end_sample })
    }

    pub fn observe(&mut self, detected: bool) -> EndpointEvent {
        self.processed_samples = self
            .processed_samples
            .saturating_add(u64::from(WINDOW_SAMPLES));

        if detected {
            self.saw_speech = true;
            self.consecutive_silent_windows = 0;
            self.confirmed = false;
            self.punctuated_armed = false;
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

        if let Some(event) = self.punctuated_commit() {
            return event;
        }

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

/// Final tokens that carry a period without ending a sentence.
const NON_TERMINAL_ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "inc", "ltd", "co", "no",
    "fig", "approx", "dept", "est", "min", "max", "sec", "vol", "am", "pm", "e.g", "i.e",
];

/// Characters a speaker's sentence can legitimately end after the terminator.
const TRAILING_CLOSERS: &[char] = &['"', '\'', ')', ']', '}', '\u{201d}', '\u{2019}', '\u{bb}'];

/// True when `text` reads as a completed sentence.
///
/// Used only to shorten the confirmation window, so it is deliberately
/// conservative: an ambiguous trailing period keeps the ordinary window.
pub fn ends_sentence(text: &str) -> bool {
    let trimmed = text
        .trim_end()
        .trim_end_matches(|c: char| TRAILING_CLOSERS.contains(&c))
        .trim_end();
    let Some(terminator) = trimmed.chars().next_back() else {
        return false;
    };
    match terminator {
        '?' | '!' | '\u{2026}' => true,
        '.' => final_token_is_a_word(trimmed),
        _ => false,
    }
}

/// Reject a trailing period that belongs to an initial, a number, or a known
/// abbreviation rather than to a sentence.
fn final_token_is_a_word(trimmed: &str) -> bool {
    let Some(token) = trimmed.split_whitespace().next_back() else {
        return false;
    };
    let stem = token.trim_end_matches('.');
    if stem.is_empty() {
        // A bare "." or "..." carries no word to judge.
        return false;
    }
    if stem.chars().count() == 1 && stem.chars().all(char::is_alphabetic) {
        return false;
    }
    if stem.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    let lowered = stem.to_lowercase();
    !NON_TERMINAL_ABBREVIATIONS.contains(&lowered.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast() -> EndpointConfig {
        EndpointPolicy::Fast.config()
    }

    fn unpunctuated(confirmation_ms: u32) -> EndpointConfig {
        EndpointConfig {
            confirmation_ms,
            punctuated_confirmation_ms: None,
        }
    }

    #[test]
    fn confirmation_windows_quantize_each_policy_upward() {
        let window_samples = std::hint::black_box(WINDOW_SAMPLES);
        for policy in [EndpointPolicy::Fast, EndpointPolicy::LongForm] {
            let config = policy.config();
            let windows = std::hint::black_box(config.confirmation_windows());
            let confirmation_samples =
                std::hint::black_box(SAMPLE_RATE * config.confirmation_ms / 1_000);
            assert!(windows * window_samples >= confirmation_samples);
            assert!((windows - 1) * window_samples < confirmation_samples);
        }
        assert_eq!(EndpointPolicy::Fast.config().confirmation_windows(), 5);
        assert_eq!(EndpointPolicy::LongForm.config().confirmation_windows(), 24);
        assert_eq!(EndpointPolicy::Fast.config().punctuated_windows(), Some(3));
        assert_eq!(EndpointPolicy::LongForm.config().punctuated_windows(), None);
    }

    #[test]
    fn candidate_is_early_but_fast_commit_waits_for_five_silent_frames() {
        let config = unpunctuated(FAST_CONFIRMATION_MS);
        let mut tracker = EndpointTracker::new(config);
        assert_eq!(tracker.observe(true), EndpointEvent::None);
        assert_eq!(
            tracker.observe(false),
            EndpointEvent::Candidate {
                speech_end_sample: u64::from(WINDOW_SAMPLES)
            }
        );
        for _ in 2..config.confirmation_windows() {
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
        let mut tracker = EndpointTracker::new(unpunctuated(FAST_CONFIRMATION_MS));
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
        let config = EndpointPolicy::LongForm.config();
        let mut tracker = EndpointTracker::new(config);
        for _ in 0..(config.confirmation_windows() * 3) {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
    }

    #[test]
    fn long_form_waits_through_a_640ms_pause() {
        let config = EndpointPolicy::LongForm.config();
        let mut tracker = EndpointTracker::new(config);
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

    #[test]
    fn arming_before_the_punctuated_window_commits_on_the_reaching_frame() {
        let mut tracker = EndpointTracker::new(fast());
        tracker.observe(true);
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Candidate { .. }
        ));
        assert_eq!(tracker.arm_punctuated(true), None);
        assert_eq!(tracker.observe(false), EndpointEvent::None);
        assert_eq!(
            tracker.observe(false),
            EndpointEvent::PunctuatedCommit {
                speech_end_sample: u64::from(WINDOW_SAMPLES)
            }
        );
    }

    #[test]
    fn arming_after_the_window_already_elapsed_commits_immediately() {
        // The speculative decode blocks the VAD loop, so the tracker often
        // catches up past the punctuated threshold before a transcript exists.
        let mut tracker = EndpointTracker::new(fast());
        tracker.observe(true);
        for _ in 0..4 {
            tracker.observe(false);
        }
        assert_eq!(
            tracker.arm_punctuated(true),
            Some(EndpointEvent::PunctuatedCommit {
                speech_end_sample: u64::from(WINDOW_SAMPLES)
            })
        );
    }

    #[test]
    fn an_unpunctuated_transcript_still_waits_for_the_full_window() {
        let config = fast();
        let mut tracker = EndpointTracker::new(config);
        tracker.observe(true);
        tracker.observe(false);
        assert_eq!(tracker.arm_punctuated(false), None);
        for _ in 2..config.confirmation_windows() {
            assert_eq!(tracker.observe(false), EndpointEvent::None);
        }
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Confirmed { .. }
        ));
    }

    #[test]
    fn resumed_speech_disarms_the_punctuated_commit() {
        let config = fast();
        let mut tracker = EndpointTracker::new(config);
        tracker.observe(true);
        tracker.observe(false);
        assert_eq!(tracker.arm_punctuated(true), None);
        assert_eq!(tracker.observe(true), EndpointEvent::SpeechResumed);
        // Without re-arming, the shorter window must not fire again.
        for _ in 1..config.confirmation_windows() {
            assert!(!matches!(
                tracker.observe(false),
                EndpointEvent::PunctuatedCommit { .. }
            ));
        }
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Confirmed { .. }
        ));
    }

    #[test]
    fn a_disabled_punctuated_window_never_commits_early() {
        let config = EndpointPolicy::LongForm.config();
        let mut tracker = EndpointTracker::new(config);
        tracker.observe(true);
        tracker.observe(false);
        assert_eq!(tracker.arm_punctuated(true), None);
        for _ in 2..config.confirmation_windows() {
            assert!(!matches!(
                tracker.observe(false),
                EndpointEvent::PunctuatedCommit { .. }
            ));
        }
        assert!(matches!(
            tracker.observe(false),
            EndpointEvent::Confirmed { .. }
        ));
    }

    #[test]
    fn sentence_terminators_are_accepted_through_trailing_closers() {
        for text in [
            "Send the report.",
            "Are you ready?",
            "Stop!",
            "She said \"go home.\"",
            "Ship it (finally).",
            "That's all\u{2026}",
            "Call the front desk.   ",
        ] {
            assert!(ends_sentence(text), "expected a sentence end: {text:?}");
        }
    }

    #[test]
    fn ambiguous_or_missing_terminators_keep_the_full_window() {
        for text in [
            "",
            "   ",
            "open the",
            "open the door",
            "I emailed Dr.",
            "meet me on Main St.",
            "the total was 4.",
            "signed J.",
            "coffee, tea, etc.",
            "...",
            "half a comma,",
        ] {
            assert!(!ends_sentence(text), "expected no sentence end: {text:?}");
        }
    }
}
