//! Transcript-diff harness: does a change move what the recogniser
//! *says*, not just how fast it says it?
//!
//! `bench_asr` measures latency. Nothing measured accuracy, which meant
//! changes with real transcription consequences — int8 weights, the
//! CoreML provider silently falling back to CPU, contextual biasing,
//! a new hotword score — landed on the strength of "it still runs and
//! the numbers look fine".
//!
//! This binary decodes WAV fixtures, then either records/compares an exact
//! machine-local transcript baseline or evaluates human-authored gold
//! references with WER, CER, formatting, category, and latency metrics.
//!
//! Usage:
//!   asr_diff --record                      # write bench/transcripts.json
//!   asr_diff                               # compare against it
//!   asr_diff --gold bench/gold.json         # quality gate + JSON report
//!   asr_diff --vocabulary path/to/vocab.txt   # compare WITH biasing on
//!
//! Exits non-zero when any transcript differs, so it can gate a change.
//! Like `bench_asr`, this needs the model already downloaded and does
//! not touch the mic, clipboard, or keyboard.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use parakeet_dictation::asr::{Asr, AsrBackendMetadata, AsrConfig, Decoded};
use parakeet_dictation::asr_eval::{
    self, DecodeMetadata, GoldManifest, QualityReport, RunMetadata,
};
use parakeet_dictation::coreml_worker::{load_coreml_worker, CoreMlWorkerConfig};
use parakeet_dictation::performance;
use parakeet_dictation::settings::SettingsStore;
use parakeet_dictation::wav::read_wav_mono;
use parakeet_dictation::{vocabulary, warmup};
use sha2::{Digest, Sha256};

const DEFAULT_AUDIO_DIR: &str = "bench/audio";
const DEFAULT_BASELINE: &str = "bench/transcripts.json";
const DEFAULT_QUALITY_REPORT: &str = "bench/asr-quality.json";

struct Args {
    audio_dir: PathBuf,
    baseline: PathBuf,
    record: bool,
    gold: Option<PathBuf>,
    json_out: PathBuf,
    /// Vocabulary file to bias with. `None` = greedy, unbiased.
    vocabulary: Option<PathBuf>,
    hotword_score: f32,
    backend: Backend,
    worker: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    repetitions: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    Sherpa,
    CoreMlUnified,
}

impl Backend {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "sherpa" => Ok(Self::Sherpa),
            "coreml-unified" => Ok(Self::CoreMlUnified),
            _ => anyhow::bail!("unknown backend {value:?}; expected sherpa or coreml-unified"),
        }
    }
}

fn parse_args() -> anyhow::Result<Args> {
    use anyhow::{anyhow, bail, Context};
    let mut audio_dir = PathBuf::from(DEFAULT_AUDIO_DIR);
    let mut baseline = PathBuf::from(DEFAULT_BASELINE);
    let mut record = false;
    let mut gold = None;
    let mut json_out = PathBuf::from(DEFAULT_QUALITY_REPORT);
    let mut vocabulary = None;
    let mut hotword_score = 2.0_f32;
    let mut backend = Backend::Sherpa;
    let mut worker = None;
    let mut model_dir = None;
    let mut repetitions = 1usize;

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
            "--gold" => {
                gold = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--gold needs a path"))?,
                ));
            }
            "--json-out" => {
                json_out = PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--json-out needs a path"))?,
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
            "--backend" => {
                backend =
                    Backend::parse(&it.next().ok_or_else(|| anyhow!("--backend needs a name"))?)?;
            }
            "--worker" => {
                worker = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--worker needs a path"))?,
                ));
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--model-dir needs a path"))?,
                ));
            }
            "--repetitions" => {
                repetitions = it
                    .next()
                    .ok_or_else(|| anyhow!("--repetitions needs a positive integer"))?
                    .parse()
                    .context("--repetitions")?;
            }
            "--record" => record = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    if record && gold.is_some() {
        bail!("--record and --gold are mutually exclusive");
    }
    if repetitions == 0 {
        bail!("--repetitions must be at least 1");
    }
    if repetitions != 1 && gold.is_none() {
        bail!("--repetitions is supported only with --gold");
    }
    validate_hotword_score(hotword_score)?;
    Ok(Args {
        audio_dir,
        baseline,
        record,
        gold,
        json_out,
        vocabulary,
        hotword_score,
        backend,
        worker,
        model_dir,
        repetitions,
    })
}

fn validate_hotword_score(score: f32) -> anyhow::Result<()> {
    if !score.is_finite() || score < 0.0 {
        anyhow::bail!("--hotword-score must be a finite, non-negative number");
    }
    Ok(())
}

fn print_usage() {
    eprintln!(
        "usage: asr_diff [--record | --gold JSON] [--audio-dir DIR]\n\
        \x20               [--baseline JSON] [--json-out JSON]\n\
        \x20               [--backend sherpa|coreml-unified]\n\
        \x20               [--worker PATH] [--model-dir DIR]\n\
        \x20               [--repetitions N]\n\
        \x20               [--vocabulary FILE] [--hotword-score N]\n\
         \n\
         Without --gold, decodes every *.wav in DIR (default\n\
         {DEFAULT_AUDIO_DIR}) and either records transcripts (--record) or\n\
         diffs against {DEFAULT_BASELINE}.\n\
         \n\
         With --gold, decodes the manifest's fixtures, prints WER/CER and\n\
         category summaries, and writes a machine report (default\n\
         {DEFAULT_QUALITY_REPORT}). Exits 1 when a manifest threshold fails.\n\
         Repetitions expose transcript/quality spread and p50/p95 latency.\n\
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
    let gold = args.gold.as_deref().map(read_gold_manifest).transpose()?;
    let store = SettingsStore::new()?;
    if args.backend == Backend::Sherpa && !store.model_present() {
        anyhow::bail!(
            "ASR model not present at {}. Launch Parakeet.app once so it can \
             download the first-run model bundle.",
            store.encoder_path().display()
        );
    }
    if args.backend == Backend::CoreMlUnified && args.vocabulary.is_some() {
        anyhow::bail!(
            "--vocabulary is not supported by the coreml-unified challenger; \
             run the unbiased quality/performance gate first"
        );
    }

    // Translate the vocabulary the same way the app does, into a temp
    // file — the point of this harness is to measure the *shipping*
    // encoding, so it must not reimplement it.
    // Per-process filename: a fixed one let two concurrent runs
    // overwrite each other's hotwords while their recognisers loaded,
    // and left a tokenised copy of the user's vocabulary lying around
    // in the temp dir afterwards.
    let generated = std::env::temp_dir().join(format!(
        "parakeet-asr-diff-hotwords.{}.txt",
        std::process::id()
    ));
    let hotwords = match &args.vocabulary {
        Some(v) => vocabulary::prepare(v, &generated, Some(&store.tokens_path()))?,
        None => None,
    };
    let decoding = decoding_metadata(args, hotwords.as_deref())?;
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

    let run_started = Instant::now();
    let model_load_started = Instant::now();
    let asr = match args.backend {
        Backend::Sherpa => Asr::load(&AsrConfig {
            encoder: &store.encoder_path(),
            decoder: &store.decoder_path(),
            joiner: &store.joiner_path(),
            tokens: &store.tokens_path(),
            num_threads: performance::performance_core_count(),
            hotwords: hotwords.as_deref(),
            hotwords_score: args.hotword_score,
        })?,
        Backend::CoreMlUnified => {
            let mut config = CoreMlWorkerConfig::discover()?;
            if let Some(worker) = &args.worker {
                config.worker_path.clone_from(worker);
            }
            if let Some(model_dir) = &args.model_dir {
                config.set_existing_model_directory(model_dir);
            }
            let (asr, worker_load_seconds) = load_coreml_worker(&config)?;
            eprintln!("Core ML worker ready in {worker_load_seconds:.3}s");
            asr
        }
    };
    let model_load_seconds = model_load_started.elapsed().as_secs_f64();
    let mut peak_resident_bytes = process_tree_resident_bytes(&asr)?;

    let warmup_started = Instant::now();
    if args.backend == Backend::Sherpa {
        warmup::page_touch(&store.encoder_path())?;
    }
    warmup::dummy_decode(&asr)?;
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();
    peak_resident_bytes = peak_resident_bytes.max(process_tree_resident_bytes(&asr)?);

    let wavs = match &gold {
        Some(manifest) => manifest
            .fixtures
            .iter()
            .map(|fixture| args.audio_dir.join(&fixture.file))
            .collect(),
        None => collect_wavs(&args.audio_dir)?,
    };
    if wavs.is_empty() {
        anyhow::bail!(
            "no *.wav fixtures in {}. Generate them with scripts/bench-latency.sh",
            args.audio_dir.display()
        );
    }

    let wav_inputs = wavs
        .iter()
        .map(|wav| {
            let (samples, sample_rate) = read_wav_mono(wav)?;
            let name = wav.file_name().map_or_else(
                || "unknown".into(),
                |name| name.to_string_lossy().to_string(),
            );
            Ok((name, samples, sample_rate))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut transcripts: BTreeMap<String, String> = BTreeMap::new();
    let mut decoded: BTreeMap<String, Vec<Decoded>> = wav_inputs
        .iter()
        .map(|(name, _, _)| (name.clone(), Vec::with_capacity(args.repetitions)))
        .collect();
    let mut first_result_seconds = None;
    for repetition in 0..args.repetitions {
        if args.repetitions > 1 {
            eprintln!("repetition {}/{}", repetition + 1, args.repetitions);
        }
        for (name, samples, sample_rate) in &wav_inputs {
            let result = asr.recognize_with_metrics(samples, *sample_rate)?;
            peak_resident_bytes = peak_resident_bytes.max(process_tree_resident_bytes(&asr)?);
            first_result_seconds.get_or_insert_with(|| run_started.elapsed().as_secs_f64());
            if repetition == 0 {
                eprintln!(
                    "  {name}: {:?} ({:.3}s, {:.1}x real time)",
                    result.text,
                    result.decode_seconds,
                    result.rtfx()
                );
                transcripts.insert(name.clone(), result.text.clone());
            }
            decoded
                .get_mut(name)
                .expect("decoded map was initialized from the same fixture list")
                .push(result);
        }
    }

    if let Some(manifest) = &gold {
        let quality = asr_eval::evaluate(
            manifest,
            &decoded,
            run_metadata(
                asr.backend_metadata().clone(),
                decoding,
                model_load_seconds,
                warmup_seconds,
                first_result_seconds
                    .ok_or_else(|| anyhow::anyhow!("no fixture produced a first result"))?,
                peak_resident_bytes
                    .max(performance::peak_resident_bytes()? + asr.auxiliary_resident_bytes()?),
            )?,
        )?;
        write_quality_report(&args.json_out, &quality)?;
        print_quality_report(&quality, &args.json_out);
        return Ok(quality.passed);
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

fn read_gold_manifest(path: &Path) -> anyhow::Result<GoldManifest> {
    use anyhow::Context;

    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading gold manifest {}", path.display()))?;
    let manifest: GoldManifest = serde_json::from_str(&raw)
        .with_context(|| format!("parsing gold manifest {}", path.display()))?;
    manifest.validate()?;
    Ok(manifest)
}

fn write_quality_report(path: &Path, report: &QualityReport) -> anyhow::Result<()> {
    use anyhow::Context;

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(path, format!("{}\n", serde_json::to_string_pretty(report)?))
        .with_context(|| format!("writing quality report {}", path.display()))
}

fn print_quality_report(report: &QualityReport, json_path: &Path) {
    println!();
    println!("gold-reference quality");
    for fixture in &report.fixtures {
        let match_kind = if fixture.exact_match {
            "exact"
        } else if fixture.lexical_match {
            "format-only"
        } else {
            "lexical-error"
        };
        println!(
            "  {:<14} WER {:>8}  CER {:>8}  {:>13}  {:>6.1}x  {}",
            fixture.file,
            format_percent(fixture.wer_percent),
            format_percent(fixture.cer_percent),
            match_kind,
            fixture.rtfx.unwrap_or(0.0),
            fixture.categories.join(",")
        );
        if !fixture.exact_match {
            println!("      reference : {:?}", fixture.reference);
            println!("      hypothesis: {:?}", fixture.hypothesis);
        }
        if fixture.word_edits > 0 {
            println!(
                "      word edits: {} insertion(s), {} deletion(s), {} substitution(s)",
                fixture.word_insertions, fixture.word_deletions, fixture.word_substitutions
            );
        }
        if fixture.repetitions > 1 {
            println!(
                "      repeats: {} output(s), {} unique; decode p50 {:.3}s, p95 {:.3}s",
                fixture.repetitions,
                fixture.unique_hypotheses,
                fixture.decode_seconds_p50,
                fixture.decode_seconds_p95
            );
        }
    }

    if !report.categories.is_empty() {
        println!();
        println!("categories");
        for (category, metrics) in &report.categories {
            println!(
                "  {category:<20} {:>2} fixture(s)  WER {:>8}  CER {:>8}  exact {:>8}",
                metrics.fixtures,
                format_percent(metrics.wer_percent),
                format_percent(metrics.cer_percent),
                format_percent(metrics.exact_match_percent)
            );
        }
    }

    println!();
    println!(
        "overall: worst WER {} (absolute {:.2}%, regression cap {:.2}%), worst CER {} (absolute {:.2}%, regression cap {:.2}%), exact {}, {:.1}x first-pass real time",
        format_percent(report.repeatability.wer_percent_max),
        report.thresholds.max_wer_percent,
        report.thresholds.baseline_wer_percent + report.thresholds.max_wer_regression_percent,
        format_percent(report.repeatability.cer_percent_max),
        report.thresholds.max_cer_percent,
        report.thresholds.baseline_cer_percent + report.thresholds.max_cer_regression_percent,
        format_percent(report.overall.exact_match_percent),
        report.overall.rtfx.unwrap_or(0.0)
    );
    println!(
        "repeatability: {} run(s), {} nondeterministic fixture(s), {} changed output(s), WER spread {}, CER spread {}",
        report.repeatability.repetitions,
        report.repeatability.nondeterministic_fixtures,
        report.repeatability.nondeterministic_outputs,
        format_percent(report.repeatability.wer_spread_percent),
        format_percent(report.repeatability.cer_spread_percent)
    );
    println!(
        "performance: load {:.3}s, warmup {:.3}s, first result {:.3}s, corpus decode p50 {:.3}s / p95 {:.3}s, peak RSS {:.2} GiB",
        report.metadata.model_load_seconds,
        report.metadata.warmup_seconds,
        report.metadata.first_result_seconds,
        report.repeatability.corpus_decode_seconds_p50,
        report.repeatability.corpus_decode_seconds_p95,
        report.metadata.peak_resident_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "hardware: {}, {} GiB, {} logical CPUs, {} ({})",
        report.metadata.chip,
        report.metadata.memory_bytes / 1024 / 1024 / 1024,
        report.metadata.logical_cpus,
        report.metadata.operating_system,
        report.metadata.architecture
    );
    println!(
        "{} — machine report: {}",
        if report.passed { "PASS" } else { "FAIL" },
        json_path.display()
    );
}

fn format_percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "undefined".to_string(),
        |percent| format!("{percent:.2}%"),
    )
}

fn run_metadata(
    backend: AsrBackendMetadata,
    decoding: DecodeMetadata,
    model_load_seconds: f64,
    warmup_seconds: f64,
    first_result_seconds: f64,
    peak_resident_bytes: u64,
) -> anyhow::Result<RunMetadata> {
    use anyhow::Context;

    let os_name = command_output("sw_vers", &["-productName"])
        .unwrap_or_else(|| std::env::consts::OS.to_string());
    let os_version = command_output("sw_vers", &["-productVersion"]);
    let operating_system =
        os_version.map_or(os_name.clone(), |version| format!("{os_name} {version}"));

    Ok(RunMetadata {
        application_version: env!("CARGO_PKG_VERSION").to_string(),
        backend,
        decoding,
        operating_system,
        architecture: std::env::consts::ARCH.to_string(),
        chip: performance::sysctl_string("machdep.cpu.brand_string")
            .context("reading chip name with sysctl")?,
        memory_bytes: performance::sysctl_u64("hw.memsize")
            .context("reading physical memory with sysctl")?,
        logical_cpus: std::thread::available_parallelism()
            .context("reading logical CPU count")?
            .get(),
        model_load_seconds,
        warmup_seconds,
        first_result_seconds,
        peak_resident_bytes,
    })
}

fn decoding_metadata(args: &Args, hotwords: Option<&Path>) -> anyhow::Result<DecodeMetadata> {
    let vocabulary_requested = args.vocabulary.is_some();
    let vocabulary_raw = args.vocabulary.as_deref().map(std::fs::read).transpose()?;
    let vocabulary_terms_requested = vocabulary_raw
        .as_deref()
        .map(|raw| String::from_utf8_lossy(raw))
        .map_or(0, |raw| vocabulary::parse_terms(&raw).len());
    Ok(DecodeMetadata {
        method: if hotwords.is_some() {
            "modified_beam_search".to_string()
        } else if args.backend == Backend::CoreMlUnified {
            "coreml_unified_greedy".to_string()
        } else {
            "greedy_search".to_string()
        },
        contextual_vocabulary_requested: vocabulary_requested,
        contextual_vocabulary_active: hotwords.is_some(),
        hotword_score: hotwords.map(|_| args.hotword_score),
        vocabulary_terms_requested,
        vocabulary_sha256: vocabulary_raw.as_deref().map(sha256_bytes),
        generated_hotwords_sha256: hotwords.map(sha256_file).transpose()?,
    })
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    Ok(sha256_bytes(&std::fs::read(path)?))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn process_tree_resident_bytes(asr: &Asr) -> anyhow::Result<u64> {
    use anyhow::Context;

    let parent = performance::resident_bytes(std::process::id())
        .context("reading benchmark resident set")?;
    Ok(parent + asr.auxiliary_resident_bytes()?)
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
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

    #[test]
    fn hotword_score_must_be_finite_and_non_negative() {
        assert!(validate_hotword_score(0.0).is_ok());
        assert!(validate_hotword_score(2.75).is_ok());
        assert!(validate_hotword_score(-0.01).is_err());
        assert!(validate_hotword_score(f32::NAN).is_err());
        assert!(validate_hotword_score(f32::INFINITY).is_err());
    }
}
