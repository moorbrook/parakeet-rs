//! Incremental window cutting and seam merging for Hold-mode dictation.
//!
//! Hold mode used to be strictly serial: nothing ran until the hotkey was
//! released, so the user waited for the whole recording to be encoded. This
//! module lets the session close a window early — at a VAD-confirmed pause, or
//! at a hard length cap when the speaker never pauses — decode it in the
//! background while capture continues, and join the pieces on the words the
//! neighbouring windows agree on. On release only the tail window is left to
//! decode.
//!
//! This is the Voz long-audio recipe applied incrementally. It changes *when*
//! windows are submitted and *how* they are joined; the recognizer is still the
//! same offline full-attention encoder. It is not the streaming-model swap
//! rejected in ADR-0009. See ADR-0031.
//!
//! Everything here is pure: the planner consumes VAD verdicts one Silero frame
//! at a time and emits cuts, and the merge consumes word spans. Neither touches
//! audio, threads, or the recognizer, so both are unit-testable without a model.

use crate::endpointing::{EndpointEvent, EndpointPolicy, EndpointTracker, SAMPLE_RATE};

/// Overlap given to a forced (mid-speech) cut, so the word straddling the cut
/// is decoded whole by the following window and the merge has real words to
/// agree on. A pause cut carries the confirmation silence as its overlap
/// instead, which is usually wordless.
pub const FORCED_CUT_OVERLAP_S: f32 = 1.5;

/// Longest common run of words required before the merge splices on agreement,
/// when both sides of the overlap have at least this many words. A single
/// shared function word ("the") is far too easy to match by chance.
const MIN_AGREEMENT_WORDS: usize = 2;

/// One decoded word with its span on the recording's timeline, in seconds.
///
/// Built from the worker's RNNT token timings, which are *emission* times: a
/// token is timestamped at the encoder frame that produced it, so a span can
/// sit slightly after the word's acoustic onset. The merge therefore matches on
/// text and only uses the times to bound the search region.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start_s: f32,
    pub end_s: f32,
}

impl Word {
    pub fn new(text: impl Into<String>, start_s: f32, end_s: f32) -> Self {
        Self {
            text: text.into(),
            start_s,
            end_s,
        }
    }
}

/// Render a merged word sequence back to a transcript.
///
/// Word texts are built by [`words_from_tokens`], which keeps punctuation
/// attached to the word it follows, so a single space between them reproduces
/// the recognizer's own spacing.
pub fn words_to_text(words: &[Word]) -> String {
    words
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// One RNNT token as the worker reports it: the detokenized piece, with a
/// leading space where the tokenizer's word-start marker was, and the emission
/// span in seconds relative to the window's own start.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenSpan {
    pub text: String,
    pub start_s: f32,
    pub end_s: f32,
}

/// Group tokens into words and shift them onto the recording's timeline.
///
/// A token whose piece begins with a space starts a new word; everything else —
/// including punctuation, which the tokenizer emits without a leading space —
/// joins the word in progress. `offset_s` is where this window starts in the
/// recording.
pub fn words_from_tokens(tokens: &[TokenSpan], offset_s: f32) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    for token in tokens {
        let starts_word = token.text.starts_with(' ') || words.is_empty();
        let piece = token.text.trim();
        if piece.is_empty() {
            continue;
        }
        if starts_word {
            words.push(Word {
                text: piece.to_string(),
                start_s: token.start_s + offset_s,
                end_s: token.end_s + offset_s,
            });
        } else if let Some(last) = words.last_mut() {
            last.text.push_str(piece);
            last.end_s = token.end_s + offset_s;
        }
    }
    words
}

/// Where a window boundary came from. Only used for logging and the bench
/// tables, but the two cases have genuinely different merge behaviour so it is
/// worth naming them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutReason {
    /// Silero confirmed a pause and enough audio had accumulated to be worth a
    /// window. The overlap handed to the next window is the confirmation
    /// silence.
    Pause,
    /// No pause arrived before the length cap. The cut lands mid-speech, and
    /// the next window is backed up by [`FORCED_CUT_OVERLAP_S`].
    Forced,
}

/// A window the planner wants decoded, on the 16 kHz sample timeline of the
/// recording.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowCut {
    /// First sample of this window.
    pub start: u64,
    /// One past the last sample of this window.
    pub end: u64,
    /// First sample of the *next* window. Always `<= end`; the difference is
    /// the overlap the merge works on.
    pub next_start: u64,
    pub reason: CutReason,
}

impl WindowCut {
    pub fn start_seconds(&self) -> f32 {
        self.start as f32 / SAMPLE_RATE as f32
    }

    pub fn overlap_start_seconds(&self) -> f32 {
        self.next_start as f32 / SAMPLE_RATE as f32
    }
}

/// Window-cutting policy for Hold mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoldWindowConfig {
    /// Off leaves Hold on the original path: nothing is decoded until the key
    /// is released, and the whole recording is decoded at once. Kept as an
    /// explicit switch so the benchmark can measure both through one binary.
    pub enabled: bool,
    /// Never cut at a pause until the current window holds at least this much
    /// audio. Too small and a hesitant speaker generates a window per word.
    pub min_seconds: f32,
    /// Cut regardless once the current window reaches this length. This is what
    /// bounds the tail decode the user actually waits for on release, so it is
    /// the number that sets the release-to-text ceiling.
    pub max_seconds: f32,
}

impl Default for HoldWindowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_seconds: 3.0,
            max_seconds: 6.0,
        }
    }
}

impl HoldWindowConfig {
    /// Reject a configuration that cannot produce a sane cut sequence rather
    /// than silently clamping it: a max below the forced-cut overlap would make
    /// every window start before the previous one ended.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if !(self.min_seconds.is_finite() && self.min_seconds > 0.0) {
            return Err(format!(
                "hold window minimum must be a positive number of seconds, got {}",
                self.min_seconds
            ));
        }
        if !(self.max_seconds.is_finite() && self.max_seconds >= self.min_seconds) {
            return Err(format!(
                "hold window maximum ({}) must be at least the minimum ({})",
                self.max_seconds, self.min_seconds
            ));
        }
        if self.max_seconds <= FORCED_CUT_OVERLAP_S {
            return Err(format!(
                "hold window maximum ({}) must exceed the forced-cut overlap of \
                 {FORCED_CUT_OVERLAP_S}s",
                self.max_seconds
            ));
        }
        Ok(())
    }
}

fn seconds_to_samples(seconds: f32) -> u64 {
    (seconds * SAMPLE_RATE as f32).round().max(0.0) as u64
}

/// Decides where to cut the held recording, one Silero frame at a time.
///
/// Pause detection reuses [`EndpointTracker`] with the `Fast` policy: it fires
/// `Confirmed` once per pause after 160 ms of silence and reports the sample
/// where the silence began. The tracker re-arms itself when speech resumes, so
/// one instance covers the whole session. Tap's own endpoint policy is
/// untouched — this tracker only chooses window boundaries and can never stop a
/// recording.
pub struct WindowPlanner {
    config: HoldWindowConfig,
    tracker: EndpointTracker,
    frame_samples: u64,
    processed: u64,
    window_start: u64,
    min_samples: u64,
    max_samples: u64,
    overlap_samples: u64,
}

impl WindowPlanner {
    pub fn new(config: HoldWindowConfig, frame_samples: u64) -> Self {
        Self {
            config,
            tracker: EndpointTracker::new(EndpointPolicy::Fast),
            frame_samples,
            processed: 0,
            window_start: 0,
            min_samples: seconds_to_samples(config.min_seconds),
            max_samples: seconds_to_samples(config.max_seconds),
            overlap_samples: seconds_to_samples(FORCED_CUT_OVERLAP_S),
        }
    }

    pub fn config(&self) -> HoldWindowConfig {
        self.config
    }

    /// First sample of the window still being accumulated. On release this is
    /// exactly where the tail window starts.
    pub fn open_window_start(&self) -> u64 {
        self.window_start
    }

    /// Feed one Silero frame's verdict. Returns a cut when this frame closes a
    /// window.
    pub fn observe_frame(&mut self, speech_detected: bool) -> Option<WindowCut> {
        self.processed = self.processed.saturating_add(self.frame_samples);
        let event = self.tracker.observe(speech_detected);

        if let EndpointEvent::Confirmed { speech_end_sample } = event {
            // Cut only if the *speech* in this window is long enough to be
            // worth a decode; the trailing silence should not count toward the
            // minimum or a long pause would let a two-word window through.
            if speech_end_sample.saturating_sub(self.window_start) >= self.min_samples {
                let cut = WindowCut {
                    start: self.window_start,
                    end: self.processed,
                    // The next window starts where the silence started, so any
                    // word this window emitted late (RNNT emission lags the
                    // acoustics) is also inside the next window's audio and the
                    // merge can recognize it.
                    next_start: speech_end_sample.max(self.window_start),
                    reason: CutReason::Pause,
                };
                self.window_start = cut.next_start;
                return Some(cut);
            }
        }

        if self.processed.saturating_sub(self.window_start) >= self.max_samples {
            let next_start = self.processed.saturating_sub(self.overlap_samples);
            let cut = WindowCut {
                start: self.window_start,
                end: self.processed,
                next_start: next_start.max(self.window_start),
                reason: CutReason::Forced,
            };
            self.window_start = cut.next_start;
            return Some(cut);
        }

        None
    }
}

/// Normalized form used to decide whether two decoded words are the same word.
/// Case and edge punctuation move around freely at a seam because each window
/// sees different context; the letters do not.
fn normalize(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Longest contiguous run of equal normalized words between `tail` and `head`.
/// Returns `(tail_index, head_index, length)`.
fn longest_common_run(tail: &[String], head: &[String]) -> (usize, usize, usize) {
    let mut best = (0_usize, 0_usize, 0_usize);
    for (i, tail_word) in tail.iter().enumerate() {
        for (j, head_word) in head.iter().enumerate() {
            if tail_word != head_word || tail_word.is_empty() {
                continue;
            }
            let mut length = 0;
            while i + length < tail.len()
                && j + length < head.len()
                && tail[i + length] == head[j + length]
            {
                length += 1;
            }
            // Prefer the longest run; among equals prefer the one earliest in
            // the following window, which is the one nearest the seam.
            if length > best.2 || (length == best.2 && length > 0 && j < best.1) {
                best = (i, j, length);
            }
        }
    }
    best
}

/// How a seam was resolved. Logged per session and asserted in tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeamOutcome {
    /// One of the two sides had no words inside the overlap, so the windows
    /// concatenate with nothing to reconcile. This is the normal result of a
    /// pause cut.
    EmptyOverlap,
    /// The windows agreed on a run of words; the splice is that run.
    Agreed { words: usize },
    /// The windows overlap in time but share no word run. The seam falls back
    /// to the overlap midpoint.
    Disagreed,
}

/// Join `next` onto `prev`, where the two windows share the audio from
/// `overlap_start_s` to the end of `prev`.
///
/// The rule is Voz's: find the longest run of words the two windows agree on
/// inside the shared audio and splice there, so a word emitted by both appears
/// once. When they agree on nothing, the seam falls to the midpoint of the
/// overlap — `prev` is truncated at its right edge and `next` at its left, so
/// the midpoint is where each is least damaged, and taking a straddling word
/// from `next` keeps the copy that was decoded whole.
pub fn merge_words(prev: &[Word], next: &[Word], overlap_start_s: f32) -> (Vec<Word>, SeamOutcome) {
    if prev.is_empty() {
        return (next.to_vec(), SeamOutcome::EmptyOverlap);
    }
    if next.is_empty() {
        return (prev.to_vec(), SeamOutcome::EmptyOverlap);
    }

    let prev_end_s = prev
        .iter()
        .map(|word| word.end_s)
        .fold(f32::NEG_INFINITY, f32::max);

    // Words of `prev` that fall inside the shared audio, and words of `next`
    // that fall before `prev` ran out. Only these can be duplicates.
    let tail_start = prev
        .iter()
        .position(|word| word.end_s > overlap_start_s)
        .unwrap_or(prev.len());
    let head_len = next
        .iter()
        .position(|word| word.start_s >= prev_end_s)
        .unwrap_or(next.len());

    if tail_start == prev.len() || head_len == 0 {
        let mut merged = prev.to_vec();
        merged.extend_from_slice(next);
        return (merged, SeamOutcome::EmptyOverlap);
    }

    let tail: Vec<String> = prev[tail_start..]
        .iter()
        .map(|word| normalize(&word.text))
        .collect();
    let head: Vec<String> = next[..head_len]
        .iter()
        .map(|word| normalize(&word.text))
        .collect();
    let (i, j, length) = longest_common_run(&tail, &head);
    let required = MIN_AGREEMENT_WORDS.min(tail.len()).min(head.len());
    if length >= required && length > 0 {
        let mut merged = prev[..tail_start + i + length].to_vec();
        merged.extend_from_slice(&next[j + length..]);
        return (merged, SeamOutcome::Agreed { words: length });
    }

    let midpoint = (overlap_start_s + prev_end_s) / 2.0;
    let mut merged: Vec<Word> = prev
        .iter()
        .filter(|word| word.end_s <= midpoint)
        .cloned()
        .collect();
    let boundary = merged.last().map_or(f32::NEG_INFINITY, |word| word.end_s);
    merged.extend(next.iter().filter(|word| word.start_s >= boundary).cloned());
    (merged, SeamOutcome::Disagreed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpointing::WINDOW_SAMPLES;

    const FRAME: u64 = WINDOW_SAMPLES as u64;

    fn words(spec: &[(&str, f32, f32)]) -> Vec<Word> {
        spec.iter()
            .map(|(text, start, end)| Word::new(*text, *start, *end))
            .collect()
    }

    fn texts(words: &[Word]) -> Vec<&str> {
        words.iter().map(|word| word.text.as_str()).collect()
    }

    #[test]
    fn tokens_group_into_words_and_render_back_to_the_transcript() {
        // Pieces as the worker emits them: the tokenizer's word-start marker
        // has already become a leading space, and punctuation carries none.
        let tokens = [
            TokenSpan {
                text: " Hello".to_string(),
                start_s: 0.08,
                end_s: 0.16,
            },
            TokenSpan {
                text: " there".to_string(),
                start_s: 0.24,
                end_s: 0.32,
            },
            TokenSpan {
                text: ",".to_string(),
                start_s: 0.32,
                end_s: 0.40,
            },
            TokenSpan {
                text: " friend".to_string(),
                start_s: 0.48,
                end_s: 0.56,
            },
            TokenSpan {
                text: ".".to_string(),
                start_s: 0.56,
                end_s: 0.64,
            },
        ];
        let words = words_from_tokens(&tokens, 2.0);
        assert_eq!(texts(&words), ["Hello", "there,", "friend."]);
        assert_eq!(words_to_text(&words), "Hello there, friend.");
        // Offsets land on the recording's timeline, not the window's.
        assert!((words[0].start_s - 2.08).abs() < 1e-5);
        assert!((words[2].end_s - 2.64).abs() < 1e-5);
    }

    #[test]
    fn empty_overlap_concatenates() {
        // A pause cut: the shared audio is silence, so neither window put a
        // word in it and there is nothing to reconcile.
        let prev = words(&[("first", 0.0, 0.5), ("half", 0.6, 1.0)]);
        let next = words(&[("second", 2.0, 2.5), ("half", 2.6, 3.0)]);
        let (merged, outcome) = merge_words(&prev, &next, 1.8);
        assert_eq!(outcome, SeamOutcome::EmptyOverlap);
        assert_eq!(texts(&merged), ["first", "half", "second", "half"]);
    }

    #[test]
    fn agreement_splices_once_and_keeps_both_sides_whole() {
        // A forced cut with 1.5 s of overlap. Both windows decoded "over the
        // lazy"; the merge must emit it exactly once and lose nothing.
        let prev = words(&[
            ("quick", 3.0, 3.3),
            ("brown", 3.4, 3.7),
            ("fox", 3.8, 4.1),
            ("jumps", 4.2, 4.6),
            ("over", 4.7, 5.0),
            ("the", 5.1, 5.3),
            ("lazy", 5.4, 5.8),
        ]);
        let next = words(&[
            ("over", 4.75, 5.05),
            ("the", 5.15, 5.35),
            ("lazy", 5.45, 5.85),
            ("dog", 6.0, 6.4),
            ("today", 6.5, 6.9),
        ]);
        let (merged, outcome) = merge_words(&prev, &next, 4.3);
        assert_eq!(outcome, SeamOutcome::Agreed { words: 3 });
        assert_eq!(
            texts(&merged),
            ["quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today"]
        );
    }

    #[test]
    fn agreement_survives_case_punctuation_and_a_frame_of_timestamp_drift() {
        // The same word in two windows is emitted at different encoder frames
        // (80 ms resolution) and can pick up different sentence casing and
        // trailing punctuation. Matching is on letters only, never on time.
        let prev = words(&[("Ready", 3.0, 3.3), ("the", 3.4, 3.6), ("Report", 3.7, 4.1)]);
        let next = words(&[
            ("report,", 3.78, 4.18),
            ("please", 4.3, 4.7),
            ("send", 4.8, 5.1),
        ]);
        let (merged, outcome) = merge_words(&prev, &next, 3.5);
        assert_eq!(outcome, SeamOutcome::Agreed { words: 1 });
        assert_eq!(texts(&merged), ["Ready", "the", "Report", "please", "send"]);
    }

    #[test]
    fn disagreement_falls_back_to_the_overlap_midpoint_without_duplicating() {
        // Two windows overlap in time but share no word. Everything before the
        // midpoint comes from the earlier window, everything after from the
        // later one, and no word appears twice.
        let prev = words(&[
            ("alpha", 0.5, 1.0),
            ("bravo", 1.2, 1.7),
            ("charlie", 2.2, 2.7),
        ]);
        let next = words(&[("delta", 2.3, 2.8), ("echo", 3.0, 3.5)]);
        let (merged, outcome) = merge_words(&prev, &next, 2.0);
        assert_eq!(outcome, SeamOutcome::Disagreed);
        // Midpoint of [2.0, 2.7] is 2.35: "charlie" ends after it and is
        // dropped in favour of the copy the later window decoded whole.
        assert_eq!(texts(&merged), ["alpha", "bravo", "delta", "echo"]);
        for pair in merged.windows(2) {
            assert!(pair[0].start_s <= pair[1].start_s, "seam reordered words");
        }
    }

    #[test]
    fn a_lone_shared_function_word_is_not_treated_as_agreement() {
        // "the" appearing in both windows is chance, not agreement, so the
        // merge must not splice on it while longer evidence is available.
        let prev = words(&[("send", 3.0, 3.3), ("the", 3.4, 3.6), ("invoice", 3.7, 4.2)]);
        let next = words(&[
            ("returned", 3.75, 4.2),
            ("the", 4.3, 4.5),
            ("package", 4.6, 5.1),
        ]);
        let (_, outcome) = merge_words(&prev, &next, 3.5);
        assert_eq!(outcome, SeamOutcome::Disagreed);
    }

    #[test]
    fn an_empty_window_never_loses_the_other_side() {
        let spoken = words(&[("only", 0.0, 0.4), ("window", 0.5, 1.0)]);
        let (merged, outcome) = merge_words(&[], &spoken, 0.0);
        assert_eq!(outcome, SeamOutcome::EmptyOverlap);
        assert_eq!(texts(&merged), ["only", "window"]);
        let (merged, outcome) = merge_words(&spoken, &[], 0.5);
        assert_eq!(outcome, SeamOutcome::EmptyOverlap);
        assert_eq!(texts(&merged), ["only", "window"]);
    }

    #[test]
    fn a_pause_cuts_the_window_and_hands_the_silence_to_the_next_one() {
        let config = HoldWindowConfig {
            enabled: true,
            min_seconds: 1.0,
            max_seconds: 100.0,
        };
        let mut planner = WindowPlanner::new(config, FRAME);
        // 2 s of speech, then silence until the Fast policy confirms.
        let speech_frames = (2.0 * SAMPLE_RATE as f32 / FRAME as f32) as u64;
        for _ in 0..speech_frames {
            assert!(planner.observe_frame(true).is_none());
        }
        let mut cut = None;
        for _ in 0..EndpointPolicy::Fast.confirmation_windows() {
            if let Some(found) = planner.observe_frame(false) {
                cut = Some(found);
                break;
            }
        }
        let cut = cut.expect("a 2 s utterance followed by silence must close a window");
        assert_eq!(cut.reason, CutReason::Pause);
        assert_eq!(cut.start, 0);
        // The window keeps the confirmation silence; the next window rewinds to
        // where that silence began.
        assert_eq!(cut.next_start, speech_frames * FRAME);
        assert!(cut.next_start < cut.end);
        assert_eq!(planner.open_window_start(), cut.next_start);
    }

    #[test]
    fn a_pause_before_the_minimum_does_not_cut() {
        let config = HoldWindowConfig {
            enabled: true,
            min_seconds: 3.0,
            max_seconds: 100.0,
        };
        let mut planner = WindowPlanner::new(config, FRAME);
        for _ in 0..30 {
            assert!(planner.observe_frame(true).is_none());
        }
        for _ in 0..200 {
            assert!(
                planner.observe_frame(false).is_none(),
                "under a second of speech is not worth its own window"
            );
        }
    }

    #[test]
    fn unbroken_speech_is_cut_at_the_cap_with_an_overlap() {
        let config = HoldWindowConfig {
            enabled: true,
            min_seconds: 3.0,
            max_seconds: 6.0,
        };
        let mut planner = WindowPlanner::new(config, FRAME);
        let mut cut = None;
        for _ in 0..1_000 {
            if let Some(found) = planner.observe_frame(true) {
                cut = Some(found);
                break;
            }
        }
        let cut = cut.expect("speech with no pause must still be cut at the cap");
        assert_eq!(cut.reason, CutReason::Forced);
        assert_eq!(cut.start, 0);
        assert!(cut.end >= seconds_to_samples(6.0));
        assert!(cut.end - cut.next_start == seconds_to_samples(FORCED_CUT_OVERLAP_S));
        assert_eq!(planner.open_window_start(), cut.next_start);
    }

    #[test]
    fn windows_tile_the_recording_with_no_gap() {
        // Whatever the mix of pause and forced cuts, each window must start
        // where the previous one's overlap begins — a gap would silently drop
        // audio, which no merge could recover.
        let mut planner = WindowPlanner::new(HoldWindowConfig::default(), FRAME);
        let mut expected_start = 0_u64;
        let mut cuts = 0;
        for frame in 0..2_000_u64 {
            // Speak in bursts with pauses between them.
            let detected = (frame / 60) % 3 != 2;
            if let Some(cut) = planner.observe_frame(detected) {
                assert_eq!(cut.start, expected_start, "window {cuts} skipped audio");
                assert!(cut.next_start <= cut.end);
                expected_start = cut.next_start;
                cuts += 1;
            }
        }
        assert!(cuts > 0, "a minute of bursty speech must produce cuts");
        assert_eq!(planner.open_window_start(), expected_start);
    }

    #[test]
    fn a_configuration_that_cannot_tile_is_rejected() {
        assert!(HoldWindowConfig::default().validate().is_ok());
        assert!(HoldWindowConfig {
            max_seconds: 1.0,
            ..HoldWindowConfig::default()
        }
        .validate()
        .is_err());
        assert!(HoldWindowConfig {
            min_seconds: 0.5,
            max_seconds: FORCED_CUT_OVERLAP_S,
            ..HoldWindowConfig::default()
        }
        .validate()
        .is_err());
        assert!(HoldWindowConfig {
            min_seconds: 0.0,
            ..HoldWindowConfig::default()
        }
        .validate()
        .is_err());
        // Off is always a legal configuration; the numbers beside it are then
        // never read.
        assert!(HoldWindowConfig {
            enabled: false,
            min_seconds: 0.0,
            max_seconds: 0.0,
        }
        .validate()
        .is_ok());
    }
}
