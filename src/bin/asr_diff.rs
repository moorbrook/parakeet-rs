//! Transcript-diff harness: does a change move what the recogniser
//! *says*, not just how fast it says it?
//!
//! `bench_asr` measures latency. Nothing measured accuracy, which meant
//! changes with real transcription consequences — int8 weights, the
//! CoreML provider silently falling back to CPU, contextual biasing,
//! a new hotword score — landed on the strength of "it still runs and
//! the numbers look fine".
//!
//! This binary decodes a directory of WAV fixtures, then either records
//! the transcripts as a baseline or compares against a recorded one and
//! reports the word-level differences.
//!
//! Usage:
//!   asr_diff --record                      # write bench/transcripts.json
//!   asr_diff                               # compare against it
//!   asr_diff --vocabulary path/to/vocab.txt   # compare WITH biasing on
//!
//! Exits non-zero when any transcript differs, so it can gate a change.
//! Like `bench_asr`, this needs the model already downloaded and does
//! not touch the mic, clipboard, or keyboard.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use parakeet_rs::asr::{Asr, AsrConfig};
use parakeet_rs::performance;
use parakeet_rs::settings::SettingsStore;
use parakeet_rs::wav::read_wav_mono;
use parakeet_rs::{vocabulary, warmup};

const DEFAULT_AUDIO_DIR: &str = "bench/audio";
const DEFAULT_BASELINE: &str = "bench/transcripts.json";

struct Args {
    audio_dir: PathBuf,
    baseline: PathBuf,
    record: bool,
    /// Vocabulary file to bias with. `None` = greedy, unbiased.
    vocabulary: Option<PathBuf>,
    hotword_score: f32,
}

fn parse_args() -> anyhow::Result<Args> {
    use anyhow::{anyhow, bail, Context};
    let mut audio_dir = PathBuf::from(DEFAULT_AUDIO_DIR);
    let mut baseline = PathBuf::from(DEFAULT_BASELINE);
    let mut record = false;
    let mut vocabulary = None;
    let mut hotword_score = 2.0_f32;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--audio-dir" => {
                audio_dir = PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--audio-dir needs a path"))?,
                );
            }
            "--baseline" => {
                baseline = PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--baseline needs a path"))?,
                );
            }
            "--vocabulary" => {
                vocabulary = Some(PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--vocabulary needs a path"))?,
                ));
            }
            "--hotword-score" => {
                hotword_score = it
                    .next()
                    .ok_or_else(|| anyhow!("--hotword-score needs a number"))?
                    .parse()
                    .context("--hotword-score")?;
            }
            "--record" => record = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    Ok(Args {
        audio_dir,
        baseline,
        record,
        vocabulary,
        hotword_score,
    })
}

fn print_usage() {
    eprintln!(
        "usage: asr_diff [--record] [--audio-dir DIR] [--baseline JSON]\n\
        \x20               [--vocabulary FILE] [--hotword-score N]\n\
         \n\
         Decodes every *.wav in DIR (default {DEFAULT_AUDIO_DIR}) and either\n\
         records the transcripts to JSON (--record) or diffs against a\n\
         previously recorded set (default {DEFAULT_BASELINE}).\n\
         \n\
         Exits 1 if any transcript changed, so it can gate a change that\n\
         was only ever checked for latency.\n\
         \n\
         The model must already be downloaded (launch Parakeet.app once)."
    );
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e:#}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("asr_diff failed: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// Deletes its path on drop. The generated hotwords file is a
/// tokenised copy of the user's vocabulary; leaving it in `/tmp` after
/// the run is both litter and a small disclosure.
struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Returns `Ok(false)` when transcripts differed from the baseline.
fn run(args: &Args) -> anyhow::Result<bool> {
    let store = SettingsStore::new()?;
    if !store.model_present() {
        anyhow::bail!(
            "ASR model not present at {}. Launch Parakeet.app once so it can \
             download the first-run model bundle.",
            store.encoder_path().display()
        );
    }

    // Translate the vocabulary the same way the app does, into a temp
    // file — the point of this harness is to measure the *shipping*
    // encoding, so it must not reimplement it.
    // Per-process filename: a fixed one let two concurrent runs
    // overwrite each other's hotwords while their recognisers loaded,
    // and left a tokenised copy of the user's vocabulary lying around
    // in the temp dir afterwards.
    let generated =
        std::env::temp_dir().join(format!("parakeet-asr-diff-hotwords.{}.txt", std::process::id()));
    let hotwords = match &args.vocabulary {
        Some(v) => vocabulary::prepare(v, &generated, Some(&store.tokens_path()))?,
        None => None,
    };
    // Removed on every exit path below via this guard.
    let _cleanup = TempFileGuard(generated.clone());
    match &hotwords {
        Some(p) => eprintln!(
            "biasing ON (score {}) from {}",
            args.hotword_score,
            p.display()
        ),
        None => eprintln!("biasing OFF (greedy decoding)"),
    }

    let asr = Asr::load(&AsrConfig {
        encoder: &store.encoder_path(),
        decoder: &store.decoder_path(),
        joiner: &store.joiner_path(),
        tokens: &store.tokens_path(),
        num_threads: performance::performance_core_count(),
        hotwords: hotwords.as_deref(),
        hotwords_score: args.hotword_score,
    })?;
    warmup::page_touch(&store.encoder_path())?;
    warmup::dummy_decode(&asr)?;

    let wavs = collect_wavs(&args.audio_dir)?;
    if wavs.is_empty() {
        anyhow::bail!(
            "no *.wav fixtures in {}. Generate them with scripts/bench-latency.sh",
            args.audio_dir.display()
        );
    }

    let mut transcripts: BTreeMap<String, String> = BTreeMap::new();
    for wav in &wavs {
        let (samples, sample_rate) = read_wav_mono(wav)?;
        let text = asr.recognize(&samples, sample_rate)?;
        let name = wav
            .file_name()
            .map_or_else(|| "unknown".into(), |n| n.to_string_lossy().to_string());
        eprintln!("  {name}: {text:?}");
        transcripts.insert(name, text);
    }

    if args.record {
        if let Some(parent) = args.baseline.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &args.baseline,
            format!("{}\n", serde_json::to_string_pretty(&transcripts)?),
        )?;
        println!(
            "recorded {} transcripts to {}",
            transcripts.len(),
            args.baseline.display()
        );
        return Ok(true);
    }

    let raw = std::fs::read_to_string(&args.baseline).map_err(|e| {
        anyhow::anyhow!(
            "reading baseline {}: {e}. Record one first with --record.",
            args.baseline.display()
        )
    })?;
    let baseline: BTreeMap<String, String> = serde_json::from_str(&raw)?;
    Ok(report(&baseline, &transcripts))
}

/// Print a per-fixture comparison. Returns true iff everything matched.
fn report(baseline: &BTreeMap<String, String>, current: &BTreeMap<String, String>) -> bool {
    let mut changed = 0usize;
    let mut total_ref_words = 0usize;
    let mut total_edits = 0usize;

    println!();
    for (name, want) in baseline {
        let want_words: Vec<&str> = want.split_whitespace().collect();
        total_ref_words += want_words.len();

        let Some(got) = current.get(name) else {
            // A fixture that didn't decode is a total loss of its words,
            // not a free pass. Counting it as zero edits let a run where
            // NOTHING decoded report "0.00% divergence".
            println!(
                "MISSING  {name}  (in baseline, not decoded this run — {} words lost)",
                want_words.len()
            );
            changed += 1;
            total_edits += want_words.len();
            continue;
        };
        let got_words: Vec<&str> = got.split_whitespace().collect();
        let edits = word_edit_distance(&want_words, &got_words);
        total_edits += edits;

        // Exact string comparison decides pass/fail; word distance only
        // sizes the change. Word-distance alone reported "ok" for
        // punctuation and spacing drift ("end-to-end" vs "end to end"
        // collapses differently), which is exactly the kind of output
        // change a decoder swap produces.
        if want == got {
            println!("ok       {name}");
        } else {
            changed += 1;
            let wer = ratio(edits, want_words.len());
            if edits == 0 {
                println!("CHANGED  {name}  (whitespace only)");
            } else {
                println!("CHANGED  {name}  ({edits} word edits, {wer:.1}% of baseline)");
            }
            println!("    baseline: {want:?}");
            println!("    current : {got:?}");
        }
    }
    for (name, got) in current {
        if !baseline.contains_key(name) {
            // Also a failure: an unrecorded fixture is unverified, and
            // silently passing meant an empty baseline made the gate a
            // no-op that always exited 0.
            println!("NEW      {name}  (decoded this run, not in baseline — re-record)");
            println!("    current : {got:?}");
            changed += 1;
        }
    }

    println!();
    match ratio_opt(total_edits, total_ref_words) {
        Some(overall) => println!(
            "{changed} fixture(s) differ; {total_edits} word edits over \
             {total_ref_words} baseline words ({overall:.2}% divergence)"
        ),
        // No baseline words at all: a percentage would read as 0.00%,
        // which looks like "nothing changed" rather than "nothing to
        // compare against".
        None => println!(
            "{changed} fixture(s) differ; no baseline words to compare against \
             (divergence undefined)"
        ),
    }
    if changed == 0 {
        println!("no transcript drift");
    }
    changed == 0
}

/// Percentage, or `None` when there is no denominator to divide by.
/// Kept distinct from "0%" so an empty baseline can't masquerade as a
/// clean run.
fn ratio_opt(num: usize, denom: usize) -> Option<f64> {
    (denom > 0).then(|| 100.0 * num as f64 / denom as f64)
}

fn ratio(num: usize, denom: usize) -> f64 {
    ratio_opt(num, denom).unwrap_or(0.0)
}

/// Levenshtein distance over whole words — the standard WER numerator.
///
/// Two rows rather than a full matrix: fixtures are short, but there's
/// no reason to allocate `n*m` for a value that only ever reads the
/// previous row.
fn word_edit_distance(a: &[&str], b: &[&str]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, aw) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bw) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(aw != bw);
            let deletion = prev[j + 1] + 1;
            let insertion = curr[j] + 1;
            curr[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Every `*.wav` directly inside `dir`, sorted so runs are comparable.
fn collect_wavs(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")))
        .collect();
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_word_sequences_have_no_edits() {
        assert_eq!(word_edit_distance(&["a", "b", "c"], &["a", "b", "c"]), 0);
    }

    #[test]
    fn substitution_insertion_and_deletion_each_cost_one() {
        assert_eq!(word_edit_distance(&["a", "b"], &["a", "x"]), 1);
        assert_eq!(word_edit_distance(&["a"], &["a", "b"]), 1);
        assert_eq!(word_edit_distance(&["a", "b"], &["a"]), 1);
    }

    #[test]
    fn empty_sides_cost_the_other_length() {
        assert_eq!(word_edit_distance(&[], &["a", "b"]), 2);
        assert_eq!(word_edit_distance(&["a", "b"], &[]), 2);
        assert_eq!(word_edit_distance(&[], &[]), 0);
    }

    #[test]
    fn distance_counts_words_not_characters() {
        // "recognise"/"recognize" is ONE edit, not two character edits —
        // otherwise a single spelling change would swamp the metric.
        assert_eq!(
            word_edit_distance(&["speech", "recognise"], &["speech", "recognize"]),
            1
        );
    }

    #[test]
    fn report_is_true_only_when_every_fixture_matches() {
        let mut base = BTreeMap::new();
        base.insert("1s.wav".to_string(), "hello world".to_string());
        let mut same = BTreeMap::new();
        same.insert("1s.wav".to_string(), "hello world".to_string());
        assert!(report(&base, &same));

        let mut drifted = BTreeMap::new();
        drifted.insert("1s.wav".to_string(), "hello word".to_string());
        assert!(!report(&base, &drifted));
    }

    #[test]
    fn a_fixture_missing_from_the_run_counts_as_a_failure() {
        // Silently passing because a fixture didn't decode would make
        // the gate worthless.
        let mut base = BTreeMap::new();
        base.insert("1s.wav".to_string(), "hello world".to_string());
        assert!(!report(&base, &BTreeMap::new()));
    }

    #[test]
    fn a_fixture_not_in_the_baseline_counts_as_a_failure() {
        // Otherwise an EMPTY baseline passes against any number of
        // decoded fixtures — the gate would exit 0 having verified
        // nothing at all.
        let mut current = BTreeMap::new();
        current.insert("new.wav".to_string(), "unverified".to_string());
        assert!(!report(&BTreeMap::new(), &current));
    }

    #[test]
    fn whitespace_only_drift_is_reported_as_changed() {
        // Word-distance alone called this "ok". Spacing changes are
        // exactly what a decoder or normalisation swap produces, so the
        // gate has to catch them.
        let mut base = BTreeMap::new();
        base.insert("1s.wav".to_string(), "hello world".to_string());
        let mut spaced = BTreeMap::new();
        spaced.insert("1s.wav".to_string(), "hello  world".to_string());
        assert!(!report(&base, &spaced));
    }

    #[test]
    fn an_empty_baseline_and_empty_run_is_vacuously_clean() {
        // No fixtures either side: nothing changed, but also nothing
        // verified. Passing is right; the "no baseline words" message
        // is what tells the user it was vacuous.
        assert!(report(&BTreeMap::new(), &BTreeMap::new()));
    }

    #[test]
    fn ratio_opt_distinguishes_no_denominator_from_zero_percent() {
        // A missing denominator rendered as "0.00%" reads as "nothing
        // changed" when it actually means "nothing to compare".
        assert_eq!(ratio_opt(0, 0), None);
        assert_eq!(ratio_opt(5, 0), None);
        assert_eq!(ratio_opt(1, 4), Some(25.0));
    }
}
