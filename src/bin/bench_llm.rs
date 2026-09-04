//! LLM polish bench: completion latency and output quality on a GGUF model.
//!
//! Loads a GGUF file via llama.cpp (Metal backend on Apple Silicon) and
//! runs polish iterations through the exact `PromptTemplate` and decode
//! loop from `src/polish.rs`, emitting one `llm_timer` log line per
//! iteration to stderr in this shape:
//!
//! ```text
//! llm_timer session_id=bench-qwen3.5-2b-r042-... model=qwen3.5-2b-q4_k_m \
//!   prompt_tokens=234 out_tokens=58 ttft_ms=183 gen_ms=472 \
//!   total_ms=655 tokens_per_s=122.9
//! ```
//!
//! Two modes:
//!
//! - **Fixed sample** (default). One hardcoded noisy transcript, N
//!   repetitions. This is the historical Phase-0 shape; the numbers in
//!   `bench/polish-backends.csv` and ADR-0018 come from it.
//! - **Eval set** (`--eval bench/polish/eval.json`). Every item in the
//!   eval set, N repetitions, scored against each item's `expected`
//!   text. Latency percentiles and the quality score then describe one
//!   population, which is what the <1000 ms p50 acceptance in Kata 0tpp
//!   needs — a p50 over a single worst-case transcript is not a p50
//!   over anything a user dictates.
//!
//! `--variant` selects the polish strategy under test (`full-text` or
//! `edits-only`), `--prompt-cache` keeps one context alive so the
//! system prompt's KV state is reused, and `--skip-min-words` applies
//! the short-utterance skip policy. Each combination is one row of the
//! table in `bench/README.md`.
//!
//! Sampler, batch sizing, and `GenerateConfig` come from
//! `polish::generate` + `polish::PROD_GENERATE_CONFIG`, so any change
//! to the production decode path shows up here on the next bench run
//! rather than silently invalidating the CSV.
//!
//! See `docs/latency-plan.md` §6.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use serde::Deserialize;

use parakeet_dictation::performance::next_session_id;
use parakeet_dictation::polish::{
    self, GenerateConfig, GenerateOutcome, PolishStrategy, PrefixCache, PromptTemplate, SkipPolicy,
    PROD_GENERATE_CONFIG,
};

/// A representative messy dictation transcript — fillers, no punctuation,
/// the kind of thing the polish pass is supposed to fix. Length picked
/// so the chat-formatted prompt lands around the 200-token mark, matching
/// the latency plan's "60-token transcript input" intent (the plan's
/// "60 tokens" referred to the *output*; this is the input).
///
/// Retained verbatim as the fixed-sample input so runs against
/// `bench/polish-backends.csv` and the ADR-0018 numbers stay
/// comparable. The eval set carries the same string as
/// `legacy-bench-sample`.
const SAMPLE_INPUT: &str = "um so I was thinking we could you know probably uh move the deadline back to next Friday I mean it's getting kind of tight and like the team's been a bit stretched with the migration work and all the on-call stuff so yeah let's just push it";

// ── Eval set ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct EvalSet {
    items: Vec<EvalItem>,
}

#[derive(Debug, Deserialize)]
struct EvalItem {
    id: String,
    category: String,
    input: String,
    expected: String,
}

/// One measured pass over one eval item.
struct ItemResult {
    id: String,
    category: String,
    /// Wall clock the user would wait: `ttft + gen`, or zero when the
    /// skip policy bypassed the model entirely.
    total: Duration,
    out_tokens: u32,
    /// Prompt tokens served from the prefix cache rather than
    /// prefilled. Stays zero when llama.cpp refuses partial KV removal,
    /// which is how a `--prompt-cache` run reports that the cache did
    /// nothing rather than quietly claiming a win.
    reused_prompt_tokens: usize,
    /// Polish bypassed by [`SkipPolicy`]; no decode happened.
    skipped: bool,
    /// `edits-only` produced a reply that [`polish::apply_edits`]
    /// refused. Production falls back to the raw transcript here, so
    /// the quality score does too.
    edit_fallback: bool,
    /// Word-error rate of the produced text against `expected`.
    wer: f64,
    exact: bool,
    /// What the pass actually delivered. Only populated under
    /// `--show-output`: a WER number tells you an item scored badly, but
    /// only the text tells you whether the model misbehaved or the
    /// item's `expected` asks for something the system prompt forbids.
    produced: Option<String>,
}

// ── Args ────────────────────────────────────────────────────────────

struct Args {
    model_path: PathBuf,
    eval_path: Option<PathBuf>,
    reps: usize,
    warmup_reps: usize,
    show_output: bool,
    strategy: PolishStrategy,
    prompt_cache: bool,
    skip: SkipPolicy,
    csv_path: Option<PathBuf>,
    /// Tag baked into the `llm_timer` log line. The aggregator
    /// (`scripts/bench-aggregate.py`) buckets rows by this string, so
    /// two runs against the same GGUF should share a tag, and a swap
    /// to a different quant should change it. Defaults to the model
    /// filename stem (lowercased) so it tracks `--model` honestly
    /// instead of lying when the user supplies a different GGUF;
    /// `--tag` overrides for cross-quant comparison runs.
    model_tag: Option<String>,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut model_path: Option<PathBuf> = None;
    let mut eval_path: Option<PathBuf> = None;
    let mut reps: usize = 100;
    let mut warmup_reps: usize = 3;
    let mut show_output = false;
    let mut model_tag: Option<String> = None;
    let mut strategy = PolishStrategy::FullText;
    let mut prompt_cache = false;
    let mut skip = SkipPolicy::DISABLED;
    let mut csv_path: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                model_path = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--model needs PATH"))?,
                ));
            }
            "--eval" => {
                eval_path = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--eval needs PATH"))?,
                ));
            }
            "--csv" => {
                csv_path = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--csv needs PATH"))?,
                ));
            }
            "--variant" => {
                let name = it.next().ok_or_else(|| anyhow!("--variant needs NAME"))?;
                strategy = match name.as_str() {
                    "full-text" => PolishStrategy::FullText,
                    "edits-only" => PolishStrategy::EditsOnly,
                    other => bail!("unknown --variant {other}; want full-text or edits-only"),
                };
            }
            "--prompt-cache" => prompt_cache = true,
            "--skip-min-words" => {
                skip = SkipPolicy {
                    min_words: it
                        .next()
                        .ok_or_else(|| anyhow!("--skip-min-words needs N"))?
                        .parse()
                        .context("--skip-min-words")?,
                };
            }
            "--reps" => {
                reps = it
                    .next()
                    .ok_or_else(|| anyhow!("--reps needs N"))?
                    .parse()
                    .context("--reps")?;
            }
            "--warmup-reps" => {
                warmup_reps = it
                    .next()
                    .ok_or_else(|| anyhow!("--warmup-reps needs N"))?
                    .parse()
                    .context("--warmup-reps")?;
            }
            "--show-output" => show_output = true,
            "--tag" => {
                model_tag = Some(it.next().ok_or_else(|| anyhow!("--tag needs STRING"))?);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    if reps == 0 {
        bail!("--reps must be at least 1");
    }
    Ok(Args {
        model_path: model_path.ok_or_else(|| anyhow!("--model is required"))?,
        eval_path,
        reps,
        warmup_reps,
        show_output,
        strategy,
        prompt_cache,
        skip,
        csv_path,
        model_tag,
    })
}

/// Derive a default model tag from the GGUF filename. `--tag` overrides.
fn default_tag(model_path: &Path) -> String {
    model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map_or_else(|| "unknown-model".to_string(), str::to_ascii_lowercase)
}

fn strategy_name(s: PolishStrategy) -> &'static str {
    match s {
        PolishStrategy::FullText => "full-text",
        PolishStrategy::EditsOnly => "edits-only",
    }
}

fn print_usage() {
    eprintln!(
        "usage: bench_llm --model PATH [--eval PATH] [--reps N] [--warmup-reps N]\n\
        \x20                [--variant full-text|edits-only] [--prompt-cache]\n\
        \x20                [--skip-min-words N] [--csv PATH] [--show-output] [--tag NAME]\n\
         \n\
         Without --eval: runs N polish iterations of one fixed sample transcript.\n\
         With --eval: runs every item of the eval set N times and scores the\n\
         output against each item's `expected` text.\n\
         \n\
         Both paths use the `PromptTemplate` and decode loop from src/polish.rs.\n\
         One `llm_timer` log line per iteration goes to stderr.\n\
         \n\
         --variant       polish strategy under test (default full-text).\n\
         --prompt-cache  keep one llama context alive so the system prompt's\n\
                         KV state is reused across iterations.\n\
         --skip-min-words N   bypass polish for transcripts under N words.\n\
         --csv PATH      append a summary row set in bench/polish-backends.csv shape.\n\
         --show-output   print the generated text for the first measured\n\
                         iteration only — useful for sanity-checking the\n\
                         prompt/template pipeline before trusting the numbers.\n\
         --tag NAME      sets the aggregator bucket; defaults to the model\n\
                         filename stem (lowercased)."
    );
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e:#}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bench_llm failed: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<()> {
    let t_load_start = Instant::now();
    let backend = LlamaBackend::init().context("init llama backend")?;
    // Default model params; gpu offload is handled by the metal feature
    // build, no per-call flag needed on macOS.
    let model_params = LlamaModelParams::default();
    let model_params = pin!(model_params);
    let model = LlamaModel::load_from_file(&backend, &args.model_path, &model_params)
        .with_context(|| format!("loading model {}", args.model_path.display()))?;
    let load_ms = t_load_start.elapsed().as_millis();
    let model_tag = args
        .model_tag
        .clone()
        .unwrap_or_else(|| default_tag(&args.model_path));
    log::info!(
        "llm_load model={model_tag} path={} load_ms={load_ms}",
        args.model_path.display()
    );
    log::info!(
        "llm_config variant={} prompt_cache={} skip_min_words={} reps={}",
        strategy_name(args.strategy),
        args.prompt_cache,
        args.skip.min_words,
        args.reps
    );

    match &args.eval_path {
        Some(path) => run_eval(args, &backend, &model, &model_tag, path),
        None => run_fixed_sample(args, &backend, &model, &model_tag),
    }
}

// ── Fixed-sample mode (historical Phase-0 shape) ────────────────────

fn run_fixed_sample(
    args: &Args,
    backend: &LlamaBackend,
    model: &LlamaModel,
    model_tag: &str,
) -> Result<()> {
    let chat_prompt = PromptTemplate::for_strategy(args.strategy).render(SAMPLE_INPUT);
    log::info!(
        "chat prompt: {} chars (sample input={} chars + template)",
        chat_prompt.len(),
        SAMPLE_INPUT.len()
    );

    let mut runner = Runner::new(backend, model, args)?;
    // Warmup reps prime the Metal kernel cache + tokenizer; tagged with
    // a `warmup-` session_id prefix so the aggregator drops them.
    for i in 0..args.warmup_reps {
        let sid = format!("warmup-r{i:03}-{}", next_session_id());
        let (_, outcome) = runner.generate(&chat_prompt)?;
        log_timer(&sid, model_tag, &outcome);
    }
    for i in 0..args.reps {
        let sid = format!("bench-{model_tag}-r{i:03}-{}", next_session_id());
        let (text, outcome) = runner.generate(&chat_prompt)?;
        log_timer(&sid, model_tag, &outcome);
        if args.show_output && i == 0 {
            log::info!("---- generated output ----");
            log::info!("{}", text.trim());
            log::info!("--------------------------");
        }
    }
    Ok(())
}

// ── Eval mode ───────────────────────────────────────────────────────

fn run_eval(
    args: &Args,
    backend: &LlamaBackend,
    model: &LlamaModel,
    model_tag: &str,
    eval_path: &Path,
) -> Result<()> {
    let raw = std::fs::read_to_string(eval_path)
        .with_context(|| format!("reading eval set {}", eval_path.display()))?;
    let set: EvalSet = serde_json::from_str(&raw)
        .with_context(|| format!("parsing eval set {}", eval_path.display()))?;
    if set.items.is_empty() {
        bail!("eval set {} has no items", eval_path.display());
    }
    log::info!(
        "llm_eval_set path={} items={}",
        eval_path.display(),
        set.items.len()
    );

    let mut runner = Runner::new(backend, model, args)?;

    // Warm on the first item so the Metal kernel cache is hot before
    // anything is measured.
    let warm_prompt = PromptTemplate::for_strategy(args.strategy).render(&set.items[0].input);
    for _ in 0..args.warmup_reps {
        runner.generate(&warm_prompt)?;
    }

    let mut results: Vec<ItemResult> = Vec::with_capacity(set.items.len() * args.reps);
    for rep in 0..args.reps {
        for item in &set.items {
            let sid = format!("bench-{model_tag}-{}-r{rep:03}", item.id);
            let show = args.show_output && rep == 0;
            let result = run_eval_item(&mut runner, args, item, &sid, model_tag, show)?;
            if show {
                log::info!(
                    "---- {} ({}) wer={:.3} exact={}",
                    item.id,
                    item.category,
                    result.wer,
                    result.exact
                );
                log::info!("  input    : {}", item.input);
                log::info!("  expected : {}", item.expected);
                log::info!("  produced : {}", result.produced.as_deref().unwrap_or(""));
            }
            results.push(result);
        }
    }

    report(args, model_tag, &set, &results)
}

fn run_eval_item(
    runner: &mut Runner<'_>,
    args: &Args,
    item: &EvalItem,
    sid: &str,
    model_tag: &str,
    show_output: bool,
) -> Result<ItemResult> {
    if args.skip.skips(&item.input) {
        log::info!(
            "llm_timer session_id={sid} model={model_tag} item={} prompt_tokens=0 \
             out_tokens=0 ttft_ms=0 gen_ms=0 total_ms=0 tokens_per_s=0.0 \
             truncated=false skipped=true",
            item.id
        );
        return Ok(ItemResult {
            id: item.id.clone(),
            category: item.category.clone(),
            total: Duration::ZERO,
            out_tokens: 0,
            reused_prompt_tokens: 0,
            skipped: true,
            edit_fallback: false,
            wer: word_error_rate(&item.expected, &item.input),
            exact: item.input == item.expected,
            produced: show_output.then(|| item.input.clone()),
        });
    }

    let prompt = PromptTemplate::for_strategy(args.strategy).render(&item.input);
    let (reply, outcome) = runner.generate(&prompt)?;
    log_timer_item(sid, model_tag, &item.id, &outcome);

    // Resolve the model's reply into the text the user would see, the
    // same way `LlamaPolish` does for each strategy.
    let (produced, edit_fallback) = match args.strategy {
        PolishStrategy::FullText => (polish_full_text_output(&reply), false),
        PolishStrategy::EditsOnly => match polish::apply_edits(&item.input, &reply) {
            Ok(text) => (text, false),
            Err(e) => {
                // Production pastes the raw transcript when polish
                // fails, so the eval scores the raw transcript. Counting
                // it as a fallback rather than a crash is what makes the
                // quality column honest about the strategy's real cost.
                log::warn!("[{}] edit list rejected: {e:#}", item.id);
                (item.input.clone(), true)
            }
        },
    };

    Ok(ItemResult {
        id: item.id.clone(),
        category: item.category.clone(),
        total: outcome.ttft + outcome.gen_time,
        out_tokens: outcome.out_tokens,
        reused_prompt_tokens: outcome.reused_prompt_tokens,
        skipped: false,
        edit_fallback,
        wer: word_error_rate(&item.expected, &produced),
        exact: produced == item.expected,
        produced: show_output.then(|| produced.clone()),
    })
}

/// What `LlamaPolish::stream_full_text` ends up delivering: the model's
/// output with any `/no_think` echo removed. The streaming path splits
/// this across chunks; the concatenation is identical.
fn polish_full_text_output(reply: &str) -> String {
    // `strip_no_think_tail` is private to polish.rs; `apply_edits`
    // exposes the same tail-strip for the NONE case. Reproduce the
    // trim here rather than widening polish.rs's public surface for a
    // bench-only need — the shapes it strips are pinned by
    // `strip_no_think_tail_handles_directive_variants`.
    let trimmed = reply.trim();
    for suffix in [
        "/no_think",
        "/no think",
        "no_think",
        "no think",
        "/nothink",
        "nothink",
    ] {
        let zone = trimmed.trim_end_matches([' ', '\t', '\n', '\r', '.', '!', '?', ',', ';', ':']);
        if zone.len() >= suffix.len() {
            let split = zone.len() - suffix.len();
            if zone.is_char_boundary(split) && zone[split..].eq_ignore_ascii_case(suffix) {
                return zone[..split].trim_end().to_string();
            }
        }
    }
    trimmed.to_string()
}

// ── Generation plumbing ─────────────────────────────────────────────

/// Owns whichever context strategy the run needs. With `--prompt-cache`
/// one context and one [`PrefixCache`] live for the whole run, so the
/// system prompt's KV state is prefilled once; without it, every call
/// allocates a fresh context exactly like `polish::generate` does in
/// production today.
struct Runner<'m> {
    backend: &'m LlamaBackend,
    model: &'m LlamaModel,
    /// Decode budget for the strategy under test. `edits-only` runs
    /// under a tighter output cap than `full-text` — see
    /// `polish::EDITS_GENERATE_CONFIG`.
    cfg: GenerateConfig,
    cached: Option<(llama_cpp_2::context::LlamaContext<'m>, PrefixCache)>,
}

impl<'m> Runner<'m> {
    fn new(backend: &'m LlamaBackend, model: &'m LlamaModel, args: &Args) -> Result<Self> {
        // Context size is the same across strategies; only the output
        // cap differs, so one context serves either.
        let cfg = GenerateConfig::for_strategy(args.strategy);
        let cached = if args.prompt_cache {
            let ctx = polish::new_context(backend, model, &PROD_GENERATE_CONFIG)?;
            Some((ctx, PrefixCache::new()))
        } else {
            None
        };
        Ok(Self {
            backend,
            model,
            cfg,
            cached,
        })
    }

    fn generate(&mut self, prompt: &str) -> Result<(String, GenerateOutcome)> {
        let mut buf = String::new();
        let outcome = match &mut self.cached {
            Some((ctx, cache)) => {
                polish::generate_in(ctx, self.model, prompt, &self.cfg, cache, |piece| {
                    buf.push_str(piece);
                    Ok(())
                })?
            }
            None => polish::generate(self.backend, self.model, prompt, &self.cfg, |piece| {
                buf.push_str(piece);
                Ok(())
            })?,
        };
        Ok((buf, outcome))
    }
}

fn log_timer(sid: &str, model_tag: &str, outcome: &GenerateOutcome) {
    let (ttft_ms, gen_ms, total_ms, tokens_per_s) = timer_fields(outcome);
    log::info!(
        "llm_timer session_id={sid} model={model_tag} prompt_tokens={} \
         out_tokens={} ttft_ms={ttft_ms} gen_ms={gen_ms} \
         total_ms={total_ms} tokens_per_s={tokens_per_s:.1} truncated={}",
        outcome.prompt_tokens,
        outcome.out_tokens,
        outcome.truncated
    );
}

fn log_timer_item(sid: &str, model_tag: &str, item_id: &str, outcome: &GenerateOutcome) {
    let (ttft_ms, gen_ms, total_ms, tokens_per_s) = timer_fields(outcome);
    log::info!(
        "llm_timer session_id={sid} model={model_tag} item={item_id} prompt_tokens={} \
         reused_prompt_tokens={} out_tokens={} ttft_ms={ttft_ms} gen_ms={gen_ms} \
         total_ms={total_ms} tokens_per_s={tokens_per_s:.1} truncated={} skipped=false",
        outcome.prompt_tokens,
        outcome.reused_prompt_tokens,
        outcome.out_tokens,
        outcome.truncated
    );
}

/// `Duration::as_millis()` returns `u128`. Bench reps are sub-second on
/// every platform we care about — a >49-day rep would mean the model is
/// wedged, which is a bug, not a measurement worth logging. Fail loudly
/// instead of silently logging `u32::MAX`.
fn timer_fields(outcome: &GenerateOutcome) -> (u32, u32, u32, f64) {
    let ttft_ms =
        u32::try_from(outcome.ttft.as_millis()).expect("ttft >49 days indicates wedged model");
    let gen_ms = u32::try_from(outcome.gen_time.as_millis())
        .expect("gen_time >49 days indicates wedged model");
    let total_ms = ttft_ms.saturating_add(gen_ms);
    let tokens_per_s = if gen_ms > 0 {
        f64::from(outcome.out_tokens) * 1000.0 / f64::from(gen_ms)
    } else {
        0.0
    };
    (ttft_ms, gen_ms, total_ms, tokens_per_s)
}

// ── Scoring ─────────────────────────────────────────────────────────

/// Word-level error rate of `hypothesis` against `reference`: edit
/// distance over whitespace-separated words, divided by the reference's
/// word count. 0.0 is a perfect match. Values above 1.0 are possible
/// (a hypothesis longer than the reference and wrong throughout).
///
/// Words are compared verbatim, punctuation and casing included —
/// punctuation and casing are two of the three things the polish pass
/// exists to fix, so normalising them away would score the pass on a
/// task it was not asked to do.
///
/// Line breaks are tokens, not whitespace. `new paragraph` and
/// `new line` are inline editing commands whose whole job is to produce
/// one, and a scorer that split on whitespace generally would treat a
/// space and a newline as the same thing — scoring zero errors for a
/// model that silently ignored the command.
fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r = tokenize_for_wer(reference);
    let h = tokenize_for_wer(hypothesis);
    let r: Vec<&str> = r.iter().map(String::as_str).collect();
    let h: Vec<&str> = h.iter().map(String::as_str).collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    // Row-by-row Levenshtein: O(|r| · |h|) time, O(|h|) space.
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0_usize; h.len() + 1];
    for (i, rw) in r.iter().enumerate() {
        cur[0] = i + 1;
        for (j, hw) in h.iter().enumerate() {
            let sub = prev[j] + usize::from(rw != hw);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()] as f64 / r.len() as f64
}

/// Split into scoring tokens: whitespace-separated words, with every
/// line break emitted as its own `\n` token so newline handling is
/// scored rather than silently normalised away.
fn tokenize_for_wer(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, line) in s.split('\n').enumerate() {
        if i > 0 {
            out.push("\\n".to_string());
        }
        out.extend(line.split_whitespace().map(str::to_string));
    }
    out
}

fn percentile(sorted_ms: &[u128], q: f64) -> u128 {
    if sorted_ms.is_empty() {
        return 0;
    }
    // Nearest-rank. Matches `scripts/bench-aggregate.py`, so a row
    // produced here and a row produced there mean the same thing.
    let rank = (q * sorted_ms.len() as f64).ceil().max(1.0) as usize;
    sorted_ms[rank.min(sorted_ms.len()) - 1]
}

fn report(args: &Args, model_tag: &str, set: &EvalSet, results: &[ItemResult]) -> Result<()> {
    let mut totals: Vec<u128> = results.iter().map(|r| r.total.as_millis()).collect();
    totals.sort_unstable();
    let mean_ms = totals.iter().sum::<u128>() as f64 / totals.len() as f64;
    let p50 = percentile(&totals, 0.50);
    let p95 = percentile(&totals, 0.95);
    let p99 = percentile(&totals, 0.99);

    // Quality is deterministic under greedy sampling, so score the
    // first repetition only; later reps would just re-add identical
    // numbers and make the mean look better-supported than it is.
    let first_rep = &results[..set.items.len()];
    let mean_wer = first_rep.iter().map(|r| r.wer).sum::<f64>() / first_rep.len() as f64;
    let exact = first_rep.iter().filter(|r| r.exact).count();
    let skipped = first_rep.iter().filter(|r| r.skipped).count();
    let fallbacks = first_rep.iter().filter(|r| r.edit_fallback).count();
    let mean_out_tokens =
        f64::from(first_rep.iter().map(|r| r.out_tokens).sum::<u32>()) / first_rep.len() as f64;
    // Measured over every rep: the first pass through the set can never
    // reuse anything (the cache starts empty), so a first-rep-only mean
    // would understate a working cache and hide a broken one.
    let mean_reused = results
        .iter()
        .map(|r| r.reused_prompt_tokens)
        .sum::<usize>() as f64
        / results.len() as f64;

    log::info!(
        "llm_eval_summary model={model_tag} variant={} prompt_cache={} skip_min_words={} \
         n={} items={} mean_ms={mean_ms:.1} p50_ms={p50} p95_ms={p95} p99_ms={p99} \
         mean_wer={mean_wer:.4} exact={exact}/{} skipped={skipped}/{} edit_fallbacks={fallbacks}/{} \
         mean_out_tokens={mean_out_tokens:.1} mean_reused_prompt_tokens={mean_reused:.1}",
        strategy_name(args.strategy),
        args.prompt_cache,
        args.skip.min_words,
        totals.len(),
        set.items.len(),
        first_rep.len(),
        first_rep.len(),
        first_rep.len(),
    );

    // Per-category quality, so a regression that only hits one shape of
    // input (technical terms, inline commands) is visible instead of
    // being averaged away.
    let mut cats: Vec<&str> = first_rep.iter().map(|r| r.category.as_str()).collect();
    cats.sort_unstable();
    cats.dedup();
    for cat in cats {
        let rows: Vec<&ItemResult> = first_rep.iter().filter(|r| r.category == cat).collect();
        let wer = rows.iter().map(|r| r.wer).sum::<f64>() / rows.len() as f64;
        let ex = rows.iter().filter(|r| r.exact).count();
        // Latency over EVERY rep of this category, not just the scored
        // one. The whole-set p50 is a function of how many items of
        // each category the eval set happens to contain, and that
        // composition was a judgement call — so publish the per-category
        // numbers and let the reader weigh them against their own
        // dictation mix instead of trusting one blended figure.
        let mut cat_ms: Vec<u128> = results
            .iter()
            .filter(|r| r.category == cat)
            .map(|r| r.total.as_millis())
            .collect();
        cat_ms.sort_unstable();
        let cat_p50 = percentile(&cat_ms, 0.50);
        let cat_max = cat_ms.last().copied().unwrap_or(0);
        log::info!(
            "llm_eval_category model={model_tag} variant={} category={cat} n={} \
             mean_wer={wer:.4} exact={ex}/{} p50_ms={cat_p50} max_ms={cat_max}",
            strategy_name(args.strategy),
            rows.len(),
            rows.len()
        );
    }

    // Per-item worst offenders, so the report can name what broke.
    let mut worst: Vec<&ItemResult> = first_rep.iter().collect();
    worst.sort_by(|a, b| b.wer.total_cmp(&a.wer));
    for r in worst.iter().take(5).filter(|r| r.wer > 0.0) {
        log::info!(
            "llm_eval_worst model={model_tag} variant={} item={} wer={:.3} fallback={}",
            strategy_name(args.strategy),
            r.id,
            r.wer,
            r.edit_fallback
        );
    }

    // The item the 1225 ms on record was measured from. Reported on its
    // own line because it is the structural bound — 55 output tokens at
    // the model's decode rate — and no amount of eval-set composition
    // changes it.
    let mut legacy_ms: Vec<u128> = results
        .iter()
        .filter(|r| r.id == "legacy-bench-sample")
        .map(|r| r.total.as_millis())
        .collect();
    legacy_ms.sort_unstable();
    if !legacy_ms.is_empty() {
        log::info!(
            "llm_eval_legacy model={model_tag} variant={} n={} p50_ms={} max_ms={}",
            strategy_name(args.strategy),
            legacy_ms.len(),
            percentile(&legacy_ms, 0.50),
            legacy_ms.last().copied().unwrap_or(0)
        );
    }

    if let Some(path) = &args.csv_path {
        write_csv(path, args, model_tag, mean_ms, p50, p95, p99, mean_wer)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_csv(
    path: &Path,
    args: &Args,
    model_tag: &str,
    mean_ms: f64,
    p50: u128,
    p95: u128,
    p99: u128,
    mean_wer: f64,
) -> Result<()> {
    let variant = format!(
        "{}{}{}",
        strategy_name(args.strategy),
        if args.prompt_cache { "+cache" } else { "" },
        if args.skip.min_words > 0 {
            format!("+skip{}", args.skip.min_words)
        } else {
            String::new()
        }
    );
    let header = "model,variant,metric,mean,p50,p95,p99\n";
    let mut rows = String::new();
    let _ = write!(
        rows,
        "{model_tag},{variant},total_ms,{mean_ms:.1},{p50},{p95},{p99}\n\
         {model_tag},{variant},mean_wer,{mean_wer:.4},,,\n"
    );
    let existed = path.exists();
    let mut out = if existed {
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
    } else {
        header.to_string()
    };
    out.push_str(&rows);
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    log::info!("llm_eval_csv path={}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_error_rate_scores_exact_match_as_zero() {
        assert_eq!(
            word_error_rate("hello there world", "hello there world"),
            0.0
        );
    }

    #[test]
    fn word_error_rate_counts_one_substitution_per_reference_word() {
        // 3 reference words, 1 wrong.
        let wer = word_error_rate("hello there world", "hello here world");
        assert!((wer - 1.0 / 3.0).abs() < 1e-9, "got {wer}");
    }

    #[test]
    fn word_error_rate_counts_insertions_and_deletions() {
        // Deletion: reference has 3, hypothesis 2.
        let del = word_error_rate("a b c", "a c");
        assert!((del - 1.0 / 3.0).abs() < 1e-9, "got {del}");
        // Insertion: hypothesis has an extra word.
        let ins = word_error_rate("a b c", "a b x c");
        assert!((ins - 1.0 / 3.0).abs() < 1e-9, "got {ins}");
    }

    #[test]
    fn word_error_rate_scores_a_missing_line_break() {
        // `new line` / `new paragraph` exist to produce a break. A
        // scorer that split on whitespace generally gave this a free
        // pass, which is why the command category read 0.000.
        let reference = "Add two spare belts.\nAdd one drive coupling.";
        let ignored_command = "Add two spare belts. Add one drive coupling.";
        assert!(
            word_error_rate(reference, ignored_command) > 0.0,
            "a model that ignored the line-break command must not score zero"
        );
        assert_eq!(word_error_rate(reference, reference), 0.0);
    }

    #[test]
    fn tokenize_for_wer_emits_breaks_as_their_own_token() {
        assert_eq!(tokenize_for_wer("a b"), vec!["a", "b"]);
        assert_eq!(tokenize_for_wer("a\nb"), vec!["a", "\\n", "b"]);
        // A blank line is still exactly one break token per newline.
        assert_eq!(tokenize_for_wer("a\n\nb"), vec!["a", "\\n", "\\n", "b"]);
        assert_eq!(tokenize_for_wer(""), Vec::<String>::new());
    }

    #[test]
    fn word_error_rate_is_case_and_punctuation_sensitive() {
        // Casing and punctuation are what the polish pass fixes, so a
        // scorer blind to them would report success for output that
        // fixed nothing.
        assert!(word_error_rate("Hello world.", "hello world") > 0.0);
    }

    #[test]
    fn word_error_rate_handles_empty_sides() {
        assert_eq!(word_error_rate("", ""), 0.0);
        assert_eq!(word_error_rate("", "anything"), 1.0);
        assert_eq!(word_error_rate("a b", ""), 1.0);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let v: Vec<u128> = (1..=100).collect();
        assert_eq!(percentile(&v, 0.50), 50);
        assert_eq!(percentile(&v, 0.95), 95);
        assert_eq!(percentile(&v, 0.99), 99);
        assert_eq!(percentile(&[], 0.5), 0);
        assert_eq!(percentile(&[7], 0.5), 7);
    }

    #[test]
    fn full_text_output_strips_the_no_think_echo() {
        assert_eq!(
            polish_full_text_output("Hello, world. /no_think"),
            "Hello, world."
        );
        assert_eq!(polish_full_text_output("  Hello.\n"), "Hello.");
        assert_eq!(
            polish_full_text_output("Don't think about it."),
            "Don't think about it."
        );
    }

    #[test]
    fn shipped_eval_set_parses_and_is_well_formed() {
        // The eval set is the referent for both the latency percentiles
        // and the quality gate in Kata 0tpp. A malformed or duplicated
        // item silently changes what "p50" means.
        let raw = include_str!("../../bench/polish/eval.json");
        let set: EvalSet = serde_json::from_str(raw).expect("eval.json must parse");
        assert!(set.items.len() >= 20, "eval set too small to be a p50");
        let mut ids: Vec<&str> = set.items.iter().map(|i| i.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate item ids in eval.json");
        for item in &set.items {
            assert!(!item.input.trim().is_empty(), "{} has empty input", item.id);
            assert!(
                !item.expected.trim().is_empty(),
                "{} has empty expected",
                item.id
            );
            assert!(!item.category.is_empty(), "{} has no category", item.id);
        }
        // The `clean` and `short` categories exist to prove the pass
        // leaves already-good text alone; if their expected text ever
        // differs from their input, they have stopped testing that.
        for item in set
            .items
            .iter()
            .filter(|i| i.category == "clean" || i.category == "short")
        {
            assert_eq!(
                item.input, item.expected,
                "{} is a no-change item; input and expected must match",
                item.id
            );
        }
    }

    #[test]
    fn eval_set_keeps_the_historical_bench_sample() {
        // `bench/README.md` §6 cites 1225 ms p50 from SAMPLE_INPUT. The
        // eval set carries the same string so the old number and the new
        // table are anchored to a shared item.
        let raw = include_str!("../../bench/polish/eval.json");
        let set: EvalSet = serde_json::from_str(raw).unwrap();
        let item = set
            .items
            .iter()
            .find(|i| i.id == "legacy-bench-sample")
            .expect("eval set must retain legacy-bench-sample");
        assert_eq!(item.input, SAMPLE_INPUT);
    }
}
