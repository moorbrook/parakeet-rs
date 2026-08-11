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

pub const GOLD_MANIFEST_VERSION: u32 = 2;
pub const QUALITY_REPORT_VERSION: u32 = 3;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityThresholds {
    pub max_wer_percent: f64,
    pub max_cer_percent: f64,
    pub baseline_wer_percent: f64,
    pub baseline_cer_percent: f64,
    pub max_wer_regression_percent: f64,
    pub max_cer_regression_percent: f64,
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
        validate_percent(
            "thresholds.baseline_wer_percent",
            self.thresholds.baseline_wer_percent,
        )?;
        validate_percent(
            "thresholds.baseline_cer_percent",
            self.thresholds.baseline_cer_percent,
        )?;
        validate_percent(
            "thresholds.max_wer_regression_percent",
            self.thresholds.max_wer_regression_percent,
        )?;
        validate_percent(
            "thresholds.max_cer_regression_percent",
            self.thresholds.max_cer_regression_percent,
        )?;
        if self.thresholds.baseline_wer_percent > self.thresholds.max_wer_percent {
            bail!("baseline WER exceeds the absolute WER limit");
        }
        if self.thresholds.baseline_cer_percent > self.thresholds.max_cer_percent {
            bail!("baseline CER exceeds the absolute CER limit");
        }
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
pub struct DecodeMetadata {
    pub method: String,
    pub contextual_vocabulary_requested: bool,
    pub contextual_vocabulary_active: bool,
    pub hotword_score: Option<f32>,
    pub vocabulary_terms_requested: usize,
    pub vocabulary_sha256: Option<String>,
    pub generated_hotwords_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunMetadata {
    pub application_version: String,
    pub backend: AsrBackendMetadata,
    pub decoding: DecodeMetadata,
    pub operating_system: String,
    pub architecture: String,
    pub chip: String,
    pub memory_bytes: u64,
    pub logical_cpus: usize,
    pub model_load_seconds: f64,
    pub warmup_seconds: f64,
    /// Backend-load start through completion of the first real fixture decode.
    pub first_result_seconds: f64,
    /// Observed peak resident set for the benchmark process tree.
    pub peak_resident_bytes: u64,
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
    pub word_insertions: usize,
    pub word_deletions: usize,
    pub word_substitutions: usize,
    pub wer_percent: Option<f64>,
    pub reference_chars: usize,
    pub char_edits: usize,
    pub char_insertions: usize,
    pub char_deletions: usize,
    pub char_substitutions: usize,
    pub cer_percent: Option<f64>,
    pub audio_seconds: f32,
    pub decode_seconds: f32,
    pub rtfx: Option<f32>,
    pub repetitions: usize,
    pub unique_hypotheses: usize,
    pub decode_seconds_p50: f32,
    pub decode_seconds_p95: f32,
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
    pub word_insertions: usize,
    pub word_deletions: usize,
    pub word_substitutions: usize,
    pub wer_percent: Option<f64>,
    pub reference_chars: usize,
    pub char_edits: usize,
    pub char_insertions: usize,
    pub char_deletions: usize,
    pub char_substitutions: usize,
    pub cer_percent: Option<f64>,
    pub audio_seconds: f64,
    pub decode_seconds: f64,
    pub rtfx: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RepeatabilityMetrics {
    pub repetitions: usize,
    pub nondeterministic_fixtures: usize,
    pub nondeterministic_outputs: usize,
    pub wer_percent_min: Option<f64>,
    pub wer_percent_max: Option<f64>,
    pub wer_spread_percent: Option<f64>,
    pub cer_percent_min: Option<f64>,
    pub cer_percent_max: Option<f64>,
    pub cer_spread_percent: Option<f64>,
    pub corpus_decode_seconds_p50: f64,
    pub corpus_decode_seconds_p95: f64,
    pub corpus_rtfx_p50: Option<f64>,
    pub corpus_rtfx_p95: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QualityReport {
    pub schema_version: u32,
    pub manifest_version: u32,
    pub passed: bool,
    pub thresholds: QualityThresholds,
    pub metadata: RunMetadata,
    pub overall: AggregateMetrics,
    pub repeatability: RepeatabilityMetrics,
    pub categories: BTreeMap<String, AggregateMetrics>,
    pub fixtures: Vec<FixtureMetrics>,
}

pub fn evaluate(
    manifest: &GoldManifest,
    decoded: &BTreeMap<String, Vec<Decoded>>,
    metadata: RunMetadata,
) -> Result<QualityReport> {
    manifest.validate()?;

    let repetitions = decoded
        .values()
        .next()
        .map(Vec::len)
        .ok_or_else(|| anyhow::anyhow!("no decoded fixtures"))?;
    if repetitions == 0 {
        bail!("decoded fixtures must contain at least one repetition");
    }
    for (file, runs) in decoded {
        if runs.len() != repetitions {
            bail!(
                "fixture {file:?} has {} repetitions; expected {repetitions}",
                runs.len()
            );
        }
    }

    let fixtures: Vec<FixtureMetrics> = manifest
        .fixtures
        .iter()
        .map(|gold| {
            let hypotheses = decoded.get(&gold.file).ok_or_else(|| {
                anyhow::anyhow!("fixture {:?} did not produce a transcript", gold.file)
            })?;
            Ok(score_fixture(gold, hypotheses))
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

    let repeatability = repeatability(manifest, decoded, repetitions)?;
    let passed = repeatability.wer_percent_max.is_some_and(|wer| {
        wer <= manifest.thresholds.max_wer_percent
            && wer
                <= manifest.thresholds.baseline_wer_percent
                    + manifest.thresholds.max_wer_regression_percent
    }) && repeatability.cer_percent_max.is_some_and(|cer| {
        cer <= manifest.thresholds.max_cer_percent
            && cer
                <= manifest.thresholds.baseline_cer_percent
                    + manifest.thresholds.max_cer_regression_percent
    });

    Ok(QualityReport {
        schema_version: QUALITY_REPORT_VERSION,
        manifest_version: manifest.version,
        passed,
        thresholds: manifest.thresholds.clone(),
        metadata,
        overall,
        repeatability,
        categories,
        fixtures,
    })
}

fn score_fixture(gold: &GoldFixture, decoded: &[Decoded]) -> FixtureMetrics {
    let first = &decoded[0];
    let normalized_reference = normalize_lexical(&gold.reference);
    let normalized_hypothesis = normalize_lexical(&first.text);
    let reference_words: Vec<&str> = normalized_reference.split_whitespace().collect();
    let hypothesis_words: Vec<&str> = normalized_hypothesis.split_whitespace().collect();
    let word_alignment = align(&reference_words, &hypothesis_words);
    let reference_chars: Vec<char> = normalized_reference.chars().collect();
    let hypothesis_chars: Vec<char> = normalized_hypothesis.chars().collect();
    let char_alignment = align(&reference_chars, &hypothesis_chars);
    let reference_word_count = reference_words.len();
    let reference_char_count = reference_chars.len();
    let mut decode_seconds: Vec<f32> = decoded.iter().map(|run| run.decode_seconds).collect();
    let unique_hypotheses: BTreeSet<&str> = decoded.iter().map(|run| run.text.as_str()).collect();

    FixtureMetrics {
        file: gold.file.clone(),
        categories: gold.categories.clone(),
        reference: gold.reference.clone(),
        hypothesis: first.text.clone(),
        normalized_reference,
        normalized_hypothesis,
        exact_match: gold.reference == first.text,
        lexical_match: word_alignment.total() == 0,
        reference_words: reference_word_count,
        word_edits: word_alignment.total(),
        word_insertions: word_alignment.insertions,
        word_deletions: word_alignment.deletions,
        word_substitutions: word_alignment.substitutions,
        wer_percent: ratio(word_alignment.total(), reference_word_count),
        reference_chars: reference_char_count,
        char_edits: char_alignment.total(),
        char_insertions: char_alignment.insertions,
        char_deletions: char_alignment.deletions,
        char_substitutions: char_alignment.substitutions,
        cer_percent: ratio(char_alignment.total(), reference_char_count),
        audio_seconds: first.audio_seconds,
        decode_seconds: first.decode_seconds,
        rtfx: (first.decode_seconds > 0.0).then(|| first.rtfx()),
        repetitions: decoded.len(),
        unique_hypotheses: unique_hypotheses.len(),
        decode_seconds_p50: percentile_f32(&mut decode_seconds, 50),
        decode_seconds_p95: percentile_f32(&mut decode_seconds, 95),
    }
}

#[derive(Default)]
struct Totals {
    fixtures: usize,
    exact_matches: usize,
    lexical_matches: usize,
    reference_words: usize,
    word_edits: usize,
    word_insertions: usize,
    word_deletions: usize,
    word_substitutions: usize,
    reference_chars: usize,
    char_edits: usize,
    char_insertions: usize,
    char_deletions: usize,
    char_substitutions: usize,
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
        self.word_insertions += fixture.word_insertions;
        self.word_deletions += fixture.word_deletions;
        self.word_substitutions += fixture.word_substitutions;
        self.reference_chars += fixture.reference_chars;
        self.char_edits += fixture.char_edits;
        self.char_insertions += fixture.char_insertions;
        self.char_deletions += fixture.char_deletions;
        self.char_substitutions += fixture.char_substitutions;
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
            word_insertions: self.word_insertions,
            word_deletions: self.word_deletions,
            word_substitutions: self.word_substitutions,
            wer_percent: ratio(self.word_edits, self.reference_words),
            reference_chars: self.reference_chars,
            char_edits: self.char_edits,
            char_insertions: self.char_insertions,
            char_deletions: self.char_deletions,
            char_substitutions: self.char_substitutions,
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

fn repeatability(
    manifest: &GoldManifest,
    decoded: &BTreeMap<String, Vec<Decoded>>,
    repetitions: usize,
) -> Result<RepeatabilityMetrics> {
    let mut wer = Vec::with_capacity(repetitions);
    let mut cer = Vec::with_capacity(repetitions);
    let mut corpus_decode_seconds = Vec::with_capacity(repetitions);
    let mut corpus_audio_seconds = 0.0_f64;

    for repetition in 0..repetitions {
        let mut word_edits = 0usize;
        let mut reference_words = 0usize;
        let mut char_edits = 0usize;
        let mut reference_chars = 0usize;
        let mut decode_seconds = 0.0_f64;
        for gold in &manifest.fixtures {
            let run = &decoded
                .get(&gold.file)
                .ok_or_else(|| anyhow::anyhow!("fixture {:?} did not decode", gold.file))?
                [repetition];
            let normalized_reference = normalize_lexical(&gold.reference);
            let normalized_hypothesis = normalize_lexical(&run.text);
            let reference_word_tokens: Vec<&str> =
                normalized_reference.split_whitespace().collect();
            let hypothesis_word_tokens: Vec<&str> =
                normalized_hypothesis.split_whitespace().collect();
            let reference_char_tokens: Vec<char> = normalized_reference.chars().collect();
            let hypothesis_char_tokens: Vec<char> = normalized_hypothesis.chars().collect();
            word_edits += align(&reference_word_tokens, &hypothesis_word_tokens).total();
            reference_words += reference_word_tokens.len();
            char_edits += align(&reference_char_tokens, &hypothesis_char_tokens).total();
            reference_chars += reference_char_tokens.len();
            decode_seconds += f64::from(run.decode_seconds);
            if repetition == 0 {
                corpus_audio_seconds += f64::from(run.audio_seconds);
            }
        }
        wer.push(ratio(word_edits, reference_words));
        cer.push(ratio(char_edits, reference_chars));
        corpus_decode_seconds.push(decode_seconds);
    }

    let nondeterministic_fixtures = decoded
        .values()
        .filter(|runs| {
            runs.iter()
                .map(|run| run.text.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                > 1
        })
        .count();
    let nondeterministic_outputs = decoded
        .values()
        .map(|runs| {
            runs.iter()
                .skip(1)
                .filter(|run| run.text != runs[0].text)
                .count()
        })
        .sum();

    let wer_percent_min = option_min(&wer);
    let wer_percent_max = option_max(&wer);
    let cer_percent_min = option_min(&cer);
    let cer_percent_max = option_max(&cer);
    let corpus_decode_seconds_p50 = percentile_f64(&mut corpus_decode_seconds.clone(), 50);
    let corpus_decode_seconds_p95 = percentile_f64(&mut corpus_decode_seconds, 95);

    Ok(RepeatabilityMetrics {
        repetitions,
        nondeterministic_fixtures,
        nondeterministic_outputs,
        wer_percent_min,
        wer_percent_max,
        wer_spread_percent: spread(wer_percent_min, wer_percent_max),
        cer_percent_min,
        cer_percent_max,
        cer_spread_percent: spread(cer_percent_min, cer_percent_max),
        corpus_decode_seconds_p50,
        corpus_decode_seconds_p95,
        corpus_rtfx_p50: (corpus_decode_seconds_p50 > 0.0)
            .then(|| corpus_audio_seconds / corpus_decode_seconds_p50),
        corpus_rtfx_p95: (corpus_decode_seconds_p95 > 0.0)
            .then(|| corpus_audio_seconds / corpus_decode_seconds_p95),
    })
}

fn option_min(values: &[Option<f64>]) -> Option<f64> {
    values
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .reduce(f64::min)
}

fn option_max(values: &[Option<f64>]) -> Option<f64> {
    values
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .reduce(f64::max)
}

fn spread(minimum: Option<f64>, maximum: Option<f64>) -> Option<f64> {
    minimum.zip(maximum).map(|(min, max)| max - min)
}

fn percentile_index(len: usize, percentile: usize) -> usize {
    debug_assert!(len > 0);
    ((len * percentile).div_ceil(100)).saturating_sub(1)
}

fn percentile_f32(values: &mut [f32], percentile: usize) -> f32 {
    values.sort_by(f32::total_cmp);
    values[percentile_index(values.len(), percentile)]
}

fn percentile_f64(values: &mut [f64], percentile: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    values[percentile_index(values.len(), percentile)]
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EditCounts {
    insertions: usize,
    deletions: usize,
    substitutions: usize,
}

impl EditCounts {
    fn total(self) -> usize {
        self.insertions + self.deletions + self.substitutions
    }
}

/// Levenshtein alignment with a deterministic substitution/deletion/insertion
/// tie-break. The full matrix is intentional: reporting the edit kinds needs a
/// backtrace, while the old two-row implementation could return only a scalar.
fn align<T: Eq>(reference: &[T], hypothesis: &[T]) -> EditCounts {
    #[derive(Clone, Copy)]
    enum Step {
        Match,
        Substitute,
        Delete,
        Insert,
    }

    let columns = hypothesis.len() + 1;
    let mut costs = vec![0usize; (reference.len() + 1) * columns];
    let mut steps = vec![Step::Match; costs.len()];
    for reference_index in 1..=reference.len() {
        let index = reference_index * columns;
        costs[index] = reference_index;
        steps[index] = Step::Delete;
    }
    for hypothesis_index in 1..=hypothesis.len() {
        costs[hypothesis_index] = hypothesis_index;
        steps[hypothesis_index] = Step::Insert;
    }

    for reference_index in 1..=reference.len() {
        for hypothesis_index in 1..=hypothesis.len() {
            let index = reference_index * columns + hypothesis_index;
            let diagonal = (reference_index - 1) * columns + hypothesis_index - 1;
            if reference[reference_index - 1] == hypothesis[hypothesis_index - 1] {
                costs[index] = costs[diagonal];
                steps[index] = Step::Match;
                continue;
            }

            let substitution = costs[diagonal] + 1;
            let deletion = costs[(reference_index - 1) * columns + hypothesis_index] + 1;
            let insertion = costs[reference_index * columns + hypothesis_index - 1] + 1;
            costs[index] = substitution.min(deletion).min(insertion);
            steps[index] = if substitution == costs[index] {
                Step::Substitute
            } else if deletion == costs[index] {
                Step::Delete
            } else {
                Step::Insert
            };
        }
    }

    let mut counts = EditCounts::default();
    let mut reference_index = reference.len();
    let mut hypothesis_index = hypothesis.len();
    while reference_index > 0 || hypothesis_index > 0 {
        match steps[reference_index * columns + hypothesis_index] {
            Step::Match => {
                reference_index -= 1;
                hypothesis_index -= 1;
            }
            Step::Substitute => {
                counts.substitutions += 1;
                reference_index -= 1;
                hypothesis_index -= 1;
            }
            Step::Delete => {
                counts.deletions += 1;
                reference_index -= 1;
            }
            Step::Insert => {
                counts.insertions += 1;
                hypothesis_index -= 1;
            }
        }
    }
    counts
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
            decoding: DecodeMetadata {
                method: "greedy_search".to_string(),
                contextual_vocabulary_requested: false,
                contextual_vocabulary_active: false,
                hotword_score: None,
                vocabulary_terms_requested: 0,
                vocabulary_sha256: None,
                generated_hotwords_sha256: None,
            },
            operating_system: "test".to_string(),
            architecture: "test".to_string(),
            chip: "test".to_string(),
            memory_bytes: 1,
            logical_cpus: 1,
            model_load_seconds: 0.0,
            warmup_seconds: 0.0,
            first_result_seconds: 0.0,
            peak_resident_bytes: 1,
        }
    }

    fn manifest(reference: &str, max_wer_percent: f64, max_cer_percent: f64) -> GoldManifest {
        GoldManifest {
            version: GOLD_MANIFEST_VERSION,
            thresholds: QualityThresholds {
                max_wer_percent,
                max_cer_percent,
                baseline_wer_percent: max_wer_percent,
                baseline_cer_percent: max_cer_percent,
                max_wer_regression_percent: 0.0,
                max_cer_regression_percent: 0.0,
            },
            fixtures: vec![GoldFixture {
                file: "fixture.wav".to_string(),
                reference: reference.to_string(),
                categories: vec!["commands".to_string()],
            }],
        }
    }

    fn decoded(text: &str) -> BTreeMap<String, Vec<Decoded>> {
        BTreeMap::from([(
            "fixture.wav".to_string(),
            vec![Decoded {
                text: text.to_string(),
                audio_seconds: 2.0,
                decode_seconds: 0.5,
            }],
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
    fn quality_report_serializes_decode_provenance() {
        let report = evaluate(
            &manifest("hello world", 0.0, 0.0),
            &decoded("hello world"),
            metadata(),
        )
        .unwrap();
        let json = serde_json::to_value(report).unwrap();

        assert_eq!(json["schema_version"], QUALITY_REPORT_VERSION);
        assert_eq!(json["metadata"]["decoding"]["method"], "greedy_search");
        assert_eq!(
            json["metadata"]["decoding"]["contextual_vocabulary_active"],
            false
        );
        assert!(json["metadata"]["decoding"]["hotword_score"].is_null());
        assert!(json["metadata"]["decoding"]["vocabulary_sha256"].is_null());
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
        assert_eq!(report.overall.word_insertions, 1);
        assert_eq!(report.overall.word_deletions, 0);
        assert_eq!(report.overall.word_substitutions, 1);
        assert_eq!(report.overall.reference_words, 3);
        assert_eq!(report.overall.wer_percent, Some(200.0 / 3.0));
    }

    #[test]
    fn alignment_reports_each_edit_kind_and_empty_sequences() {
        assert_eq!(
            align(&["one", "two"], &["zero", "one", "three"]),
            EditCounts {
                insertions: 1,
                deletions: 0,
                substitutions: 1,
            }
        );
        assert_eq!(
            align::<&str>(&[], &["extra"]),
            EditCounts {
                insertions: 1,
                deletions: 0,
                substitutions: 0,
            }
        );
        assert_eq!(
            align::<&str>(&["missing"], &[]),
            EditCounts {
                insertions: 0,
                deletions: 1,
                substitutions: 0,
            }
        );
    }

    #[test]
    fn repeatability_reports_transcript_and_quality_spread() {
        let mut predictions = decoded("one two");
        predictions.get_mut("fixture.wav").unwrap().push(Decoded {
            text: "one three".to_string(),
            audio_seconds: 2.0,
            decode_seconds: 1.0,
        });

        let report =
            evaluate(&manifest("one two", 100.0, 100.0), &predictions, metadata()).unwrap();

        assert_eq!(report.repeatability.repetitions, 2);
        assert_eq!(report.repeatability.nondeterministic_fixtures, 1);
        assert_eq!(report.repeatability.nondeterministic_outputs, 1);
        assert_eq!(report.repeatability.wer_percent_min, Some(0.0));
        assert_eq!(report.repeatability.wer_percent_max, Some(50.0));
        assert_eq!(report.repeatability.wer_spread_percent, Some(50.0));
        assert_eq!(report.repeatability.corpus_decode_seconds_p50, 0.5);
        assert_eq!(report.repeatability.corpus_decode_seconds_p95, 1.0);
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
    fn regression_cap_is_independent_of_absolute_limit() {
        let mut gold = manifest("one two", 100.0, 100.0);
        gold.thresholds.baseline_wer_percent = 0.0;
        gold.thresholds.baseline_cer_percent = 0.0;
        gold.thresholds.max_wer_regression_percent = 49.9;
        gold.thresholds.max_cer_regression_percent = 100.0;

        assert!(
            !evaluate(&gold, &decoded("one three"), metadata())
                .unwrap()
                .passed
        );
        gold.thresholds.max_wer_regression_percent = 50.0;
        assert!(
            evaluate(&gold, &decoded("one three"), metadata())
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

    #[test]
    fn mismatched_repetition_counts_are_an_error() {
        let gold = GoldManifest {
            version: GOLD_MANIFEST_VERSION,
            thresholds: QualityThresholds {
                max_wer_percent: 100.0,
                max_cer_percent: 100.0,
                baseline_wer_percent: 100.0,
                baseline_cer_percent: 100.0,
                max_wer_regression_percent: 0.0,
                max_cer_regression_percent: 0.0,
            },
            fixtures: vec![
                GoldFixture {
                    file: "one.wav".to_string(),
                    reference: "one".to_string(),
                    categories: vec![],
                },
                GoldFixture {
                    file: "two.wav".to_string(),
                    reference: "two".to_string(),
                    categories: vec![],
                },
            ],
        };
        let run = Decoded {
            text: "one".to_string(),
            audio_seconds: 1.0,
            decode_seconds: 0.1,
        };
        let predictions = BTreeMap::from([
            ("one.wav".to_string(), vec![run.clone()]),
            ("two.wav".to_string(), vec![run.clone(), run]),
        ]);

        assert!(evaluate(&gold, &predictions, metadata()).is_err());
    }
}
