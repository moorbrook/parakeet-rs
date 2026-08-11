//! Gold-reference scoring for ASR backend comparisons.
//!
//! The transcript-drift harness answers "did output change?". This module
//! answers the more useful question: "is the new output within an explicit
//! quality budget?" It deliberately keeps lexical accuracy (WER/CER) separate
//! from exact formatting so punctuation-only changes remain visible.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

use crate::asr::{AsrBackendMetadata, Decoded};

pub const GOLD_MANIFEST_VERSION: u32 = 1;
pub const QUALITY_REPORT_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityThresholds {
    pub max_wer_percent: f64,
    pub max_cer_percent: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GoldFixture {
    pub file: String,
    pub reference: String,
    #[serde(default)]
    pub categories: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GoldManifest {
    pub version: u32,
    pub thresholds: QualityThresholds,
    pub fixtures: Vec<GoldFixture>,
}

impl GoldManifest {
    pub fn validate(&self) -> Result<()> {
        if self.version != GOLD_MANIFEST_VERSION {
            bail!(
                "unsupported gold manifest version {}; expected {GOLD_MANIFEST_VERSION}",
                self.version
            );
        }
        validate_percent(
            "thresholds.max_wer_percent",
            self.thresholds.max_wer_percent,
        )?;
        validate_percent(
            "thresholds.max_cer_percent",
            self.thresholds.max_cer_percent,
        )?;
        if self.fixtures.is_empty() {
            bail!("gold manifest must contain at least one fixture");
        }

        let mut files = BTreeSet::new();
        let mut total_reference_words = 0usize;
        let mut total_reference_chars = 0usize;
        for fixture in &self.fixtures {
            let path = Path::new(&fixture.file);
            if fixture.file.is_empty() || path.is_absolute() || path.components().count() != 1 {
                bail!(
                    "fixture file {:?} must be one filename relative to --audio-dir",
                    fixture.file
                );
            }
            if !path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            {
                bail!("fixture file {:?} must end in .wav", fixture.file);
            }
            if !files.insert(&fixture.file) {
                bail!("duplicate fixture file {:?}", fixture.file);
            }

            let normalized = normalize_lexical(&fixture.reference);
            total_reference_words += normalized.split_whitespace().count();
            total_reference_chars += normalized.chars().count();

            let mut categories = BTreeSet::new();
            for category in &fixture.categories {
                if category.trim().is_empty() {
                    bail!("fixture {:?} has an empty category", fixture.file);
                }
                if !categories.insert(category) {
                    bail!("fixture {:?} repeats category {:?}", fixture.file, category);
                }
            }
        }
        if total_reference_words == 0 || total_reference_chars == 0 {
            bail!("gold manifest must contain non-empty lexical reference text");
        }
        Ok(())
    }
}

fn validate_percent(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("{label} must be a finite, non-negative percentage");
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunMetadata {
    pub application_version: String,
    pub backend: AsrBackendMetadata,
    pub operating_system: String,
    pub architecture: String,
    pub chip: Option<String>,
    pub memory_bytes: Option<u64>,
    pub logical_cpus: Option<usize>,
    pub model_load_seconds: f64,
    pub warmup_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FixtureMetrics {
    pub file: String,
    pub categories: Vec<String>,
    pub reference: String,
    pub hypothesis: String,
    pub normalized_reference: String,
    pub normalized_hypothesis: String,
    pub exact_match: bool,
    pub lexical_match: bool,
    pub reference_words: usize,
    pub word_edits: usize,
    pub wer_percent: Option<f64>,
    pub reference_chars: usize,
    pub char_edits: usize,
    pub cer_percent: Option<f64>,
    pub audio_seconds: f32,
    pub decode_seconds: f32,
    pub rtfx: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AggregateMetrics {
    pub fixtures: usize,
    pub exact_matches: usize,
    pub exact_match_percent: Option<f64>,
    pub lexical_matches: usize,
    pub lexical_match_percent: Option<f64>,
    pub reference_words: usize,
    pub word_edits: usize,
    pub wer_percent: Option<f64>,
    pub reference_chars: usize,
    pub char_edits: usize,
    pub cer_percent: Option<f64>,
    pub audio_seconds: f64,
    pub decode_seconds: f64,
    pub rtfx: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QualityReport {
    pub schema_version: u32,
    pub manifest_version: u32,
    pub passed: bool,
    pub thresholds: QualityThresholds,
    pub metadata: RunMetadata,
    pub overall: AggregateMetrics,
    pub categories: BTreeMap<String, AggregateMetrics>,
    pub fixtures: Vec<FixtureMetrics>,
}

pub fn evaluate(
    manifest: &GoldManifest,
    decoded: &BTreeMap<String, Decoded>,
    metadata: RunMetadata,
) -> Result<QualityReport> {
    manifest.validate()?;

    let fixtures: Vec<FixtureMetrics> = manifest
        .fixtures
        .iter()
        .map(|gold| {
            let hypothesis = decoded.get(&gold.file).ok_or_else(|| {
                anyhow::anyhow!("fixture {:?} did not produce a transcript", gold.file)
            })?;
            Ok(score_fixture(gold, hypothesis))
        })
        .collect::<Result<_>>()?;

    let overall = aggregate(fixtures.iter());
    let mut category_names = BTreeSet::new();
    for fixture in &fixtures {
        category_names.extend(fixture.categories.iter().cloned());
    }
    let categories = category_names
        .into_iter()
        .map(|category| {
            let metrics = aggregate(
                fixtures
                    .iter()
                    .filter(|fixture| fixture.categories.contains(&category)),
            );
            (category, metrics)
        })
        .collect();

    let passed = overall
        .wer_percent
        .is_some_and(|wer| wer <= manifest.thresholds.max_wer_percent)
        && overall
            .cer_percent
            .is_some_and(|cer| cer <= manifest.thresholds.max_cer_percent);

    Ok(QualityReport {
        schema_version: QUALITY_REPORT_VERSION,
        manifest_version: manifest.version,
        passed,
        thresholds: manifest.thresholds.clone(),
        metadata,
        overall,
        categories,
        fixtures,
    })
}

fn score_fixture(gold: &GoldFixture, decoded: &Decoded) -> FixtureMetrics {
    let normalized_reference = normalize_lexical(&gold.reference);
    let normalized_hypothesis = normalize_lexical(&decoded.text);
    let reference_words: Vec<&str> = normalized_reference.split_whitespace().collect();
    let hypothesis_words: Vec<&str> = normalized_hypothesis.split_whitespace().collect();
    let word_edits = edit_distance(&reference_words, &hypothesis_words);
    let reference_chars: Vec<char> = normalized_reference.chars().collect();
    let hypothesis_chars: Vec<char> = normalized_hypothesis.chars().collect();
    let char_edits = edit_distance(&reference_chars, &hypothesis_chars);
    let reference_word_count = reference_words.len();
    let reference_char_count = reference_chars.len();

    FixtureMetrics {
        file: gold.file.clone(),
        categories: gold.categories.clone(),
        reference: gold.reference.clone(),
        hypothesis: decoded.text.clone(),
        normalized_reference,
        normalized_hypothesis,
        exact_match: gold.reference == decoded.text,
        lexical_match: word_edits == 0,
        reference_words: reference_word_count,
        word_edits,
        wer_percent: ratio(word_edits, reference_word_count),
        reference_chars: reference_char_count,
        char_edits,
        cer_percent: ratio(char_edits, reference_char_count),
        audio_seconds: decoded.audio_seconds,
        decode_seconds: decoded.decode_seconds,
        rtfx: (decoded.decode_seconds > 0.0).then(|| decoded.rtfx()),
    }
}

#[derive(Default)]
struct Totals {
    fixtures: usize,
    exact_matches: usize,
    lexical_matches: usize,
    reference_words: usize,
    word_edits: usize,
    reference_chars: usize,
    char_edits: usize,
    audio_seconds: f64,
    decode_seconds: f64,
}

impl Totals {
    fn add(&mut self, fixture: &FixtureMetrics) {
        self.fixtures += 1;
        self.exact_matches += usize::from(fixture.exact_match);
        self.lexical_matches += usize::from(fixture.lexical_match);
        self.reference_words += fixture.reference_words;
        self.word_edits += fixture.word_edits;
        self.reference_chars += fixture.reference_chars;
        self.char_edits += fixture.char_edits;
        self.audio_seconds += f64::from(fixture.audio_seconds);
        self.decode_seconds += f64::from(fixture.decode_seconds);
    }

    fn finish(self) -> AggregateMetrics {
        AggregateMetrics {
            fixtures: self.fixtures,
            exact_matches: self.exact_matches,
            exact_match_percent: ratio(self.exact_matches, self.fixtures),
            lexical_matches: self.lexical_matches,
            lexical_match_percent: ratio(self.lexical_matches, self.fixtures),
            reference_words: self.reference_words,
            word_edits: self.word_edits,
            wer_percent: ratio(self.word_edits, self.reference_words),
            reference_chars: self.reference_chars,
            char_edits: self.char_edits,
            cer_percent: ratio(self.char_edits, self.reference_chars),
            audio_seconds: self.audio_seconds,
            decode_seconds: self.decode_seconds,
            rtfx: (self.decode_seconds > 0.0).then(|| self.audio_seconds / self.decode_seconds),
        }
    }
}

fn aggregate<'a>(fixtures: impl Iterator<Item = &'a FixtureMetrics>) -> AggregateMetrics {
    let mut totals = Totals::default();
    for fixture in fixtures {
        totals.add(fixture);
    }
    totals.finish()
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| 100.0 * numerator as f64 / denominator as f64)
}

/// NFC-normalize, lowercase, and remove formatting for lexical WER/CER.
///
/// Canonically equivalent text scores identically. Unicode letters, numbers,
/// and their combining marks are preserved and lowercased. Apostrophes are
/// removed without splitting a word (`don't` → `dont`); other punctuation and
/// whitespace become one separator. We intentionally do not transliterate,
/// apply compatibility folding, or strip accents. Exact reference/hypothesis
/// strings are retained separately so capitalization and punctuation
/// regressions remain visible.
pub fn normalize_lexical(input: &str) -> String {
    let mut normalized = String::new();
    let mut separator_pending = false;
    // Normalize before lowercasing so decomposed accents cannot be mistaken
    // for punctuation. Normalize again because Unicode lowercase mappings may
    // themselves emit decomposed sequences.
    for character in input.nfc().flat_map(char::to_lowercase).nfc() {
        if character.is_alphanumeric() {
            if separator_pending && !normalized.is_empty() {
                normalized.push(' ');
            }
            normalized.push(character);
            separator_pending = false;
        } else if is_combining_mark(character) && !normalized.is_empty() && !separator_pending {
            // Some canonically valid marks have no precomposed form. Keep them
            // attached to the current word rather than dropping the accent or
            // manufacturing a word boundary.
            normalized.push(character);
        } else if character == '\'' || character == '’' {
            // Apostrophes do not create a word boundary for lexical scoring.
        } else if !normalized.is_empty() {
            separator_pending = true;
        }
    }
    normalized
}

fn edit_distance<T: Eq>(reference: &[T], hypothesis: &[T]) -> usize {
    if reference.is_empty() {
        return hypothesis.len();
    }
    if hypothesis.is_empty() {
        return reference.len();
    }
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0; hypothesis.len() + 1];
    for (reference_index, reference_item) in reference.iter().enumerate() {
        current[0] = reference_index + 1;
        for (hypothesis_index, hypothesis_item) in hypothesis.iter().enumerate() {
            let substitution =
                previous[hypothesis_index] + usize::from(reference_item != hypothesis_item);
            let deletion = previous[hypothesis_index + 1] + 1;
            let insertion = current[hypothesis_index] + 1;
            current[hypothesis_index + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> RunMetadata {
        RunMetadata {
            application_version: "test".to_string(),
            backend: AsrBackendMetadata {
                backend: "fake".to_string(),
                model: "fake".to_string(),
                quantization: "none".to_string(),
                execution_provider: "cpu".to_string(),
            },
            operating_system: "test".to_string(),
            architecture: "test".to_string(),
            chip: None,
            memory_bytes: None,
            logical_cpus: None,
            model_load_seconds: 0.0,
            warmup_seconds: 0.0,
        }
    }

    fn manifest(reference: &str, max_wer_percent: f64, max_cer_percent: f64) -> GoldManifest {
        GoldManifest {
            version: GOLD_MANIFEST_VERSION,
            thresholds: QualityThresholds {
                max_wer_percent,
                max_cer_percent,
            },
            fixtures: vec![GoldFixture {
                file: "fixture.wav".to_string(),
                reference: reference.to_string(),
                categories: vec!["commands".to_string()],
            }],
        }
    }

    fn decoded(text: &str) -> BTreeMap<String, Decoded> {
        BTreeMap::from([(
            "fixture.wav".to_string(),
            Decoded {
                text: text.to_string(),
                audio_seconds: 2.0,
                decode_seconds: 0.5,
            },
        )])
    }

    #[test]
    fn normalization_policy_separates_words_but_not_apostrophes() {
        assert_eq!(
            normalize_lexical("  HéLLO—don't stop, CNC’s ready!  "),
            "héllo dont stop cncs ready"
        );
    }

    #[test]
    fn canonical_unicode_forms_normalize_to_the_same_lexical_text() {
        let composed = "café résumé";
        let decomposed = "cafe\u{301} re\u{301}sume\u{301}";

        assert_eq!(normalize_lexical(composed), "café résumé");
        assert_eq!(normalize_lexical(decomposed), "café résumé");
    }

    #[test]
    fn composed_reference_and_decomposed_hypothesis_score_zero_errors() {
        let report = evaluate(
            &manifest("café résumé", 0.0, 0.0),
            &decoded("cafe\u{301} re\u{301}sume\u{301}"),
            metadata(),
        )
        .unwrap();

        assert!(report.passed);
        assert_eq!(report.overall.word_edits, 0);
        assert_eq!(report.overall.char_edits, 0);
    }

    #[test]
    fn decomposed_reference_and_composed_hypothesis_score_zero_errors() {
        let report = evaluate(
            &manifest("cafe\u{301} re\u{301}sume\u{301}", 0.0, 0.0),
            &decoded("café résumé"),
            metadata(),
        )
        .unwrap();

        assert!(report.passed);
        assert_eq!(report.overall.word_edits, 0);
        assert_eq!(report.overall.char_edits, 0);
    }

    #[test]
    fn lowercasing_does_not_drop_uncomposed_marks() {
        assert_eq!(normalize_lexical("İSTANBUL"), "i\u{307}stanbul");
    }

    #[test]
    fn compatibility_characters_are_not_folded() {
        assert_eq!(normalize_lexical("①"), "①");
        assert_ne!(normalize_lexical("①"), normalize_lexical("1"));
    }

    #[test]
    fn formatting_only_changes_have_zero_wer_and_cer() {
        let report = evaluate(
            &manifest("Hello, world!", 0.0, 0.0),
            &decoded("hello world"),
            metadata(),
        )
        .unwrap();

        assert!(report.passed);
        assert_eq!(report.overall.wer_percent, Some(0.0));
        assert_eq!(report.overall.cer_percent, Some(0.0));
        assert_eq!(report.overall.exact_match_percent, Some(0.0));
        assert_eq!(report.overall.lexical_match_percent, Some(100.0));
    }

    #[test]
    fn insertion_deletion_and_substitution_contribute_to_wer() {
        let report = evaluate(
            &manifest("one two three", 100.0, 100.0),
            &decoded("one four three extra"),
            metadata(),
        )
        .unwrap();

        assert_eq!(report.overall.word_edits, 2);
        assert_eq!(report.overall.reference_words, 3);
        assert_eq!(report.overall.wer_percent, Some(200.0 / 3.0));
    }

    #[test]
    fn thresholds_determine_pass_or_fail() {
        let predictions = decoded("one three");
        assert!(
            evaluate(&manifest("one two", 50.0, 100.0), &predictions, metadata())
                .unwrap()
                .passed
        );
        assert!(
            !evaluate(&manifest("one two", 49.9, 100.0), &predictions, metadata())
                .unwrap()
                .passed
        );
    }

    #[test]
    fn category_totals_include_only_tagged_fixtures() {
        let report = evaluate(
            &manifest("open settings", 100.0, 100.0),
            &decoded("open setting"),
            metadata(),
        )
        .unwrap();

        let commands = report.categories.get("commands").unwrap();
        assert_eq!(commands.fixtures, 1);
        assert_eq!(commands.word_edits, 1);
    }

    #[test]
    fn validation_rejects_unsupported_versions_and_duplicate_files() {
        let mut gold = manifest("hello", 1.0, 1.0);
        gold.version += 1;
        assert!(gold.validate().is_err());

        let mut gold = manifest("hello", 1.0, 1.0);
        gold.fixtures.push(gold.fixtures[0].clone());
        assert!(gold.validate().is_err());
    }

    #[test]
    fn validation_rejects_paths_outside_the_audio_directory() {
        let mut gold = manifest("hello", 1.0, 1.0);
        gold.fixtures[0].file = "../fixture.wav".to_string();
        assert!(gold.validate().is_err());
    }

    #[test]
    fn missing_predictions_are_an_error() {
        assert!(evaluate(&manifest("hello", 1.0, 1.0), &BTreeMap::new(), metadata()).is_err());
    }
}
