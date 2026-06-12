//! Phase-0 LLM bench: polish latency on a GGUF model.
//!
//! Loads a GGUF file via llama.cpp (Metal backend on Apple Silicon),
//! runs N polish iterations against a fixed sample transcript using
//! the exact `PromptTemplate::prod()` and decode loop from
//! `src/polish.rs`, and emits one `llm_timer` log line per iteration
//! to stderr in this shape:
//!
//! ```text
//! llm_timer session_id=bench-qwen3.5-2b-r042-... model=qwen3.5-2b-q4_k_m \
//!   prompt_tokens=234 out_tokens=58 ttft_ms=183 gen_ms=472 \
//!   total_ms=655 tokens_per_s=122.9
//! ```
//!
//! Sampler, batch sizing, and `GenerateConfig` come from
//! `polish::generate` + `polish::PROD_GENERATE_CONFIG`, so any change
//! to the production decode path shows up here on the next bench run
//! rather than silently invalidating the CSV.
//!
//! `scripts/bench-llm.sh` orchestrates download + run + aggregate into
//! `bench/polish-backends.csv`. See `docs/latency-plan.md` §6.

use std::path::PathBuf;
use std::pin::pin;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

use parakeet_rs::polish::{
    self, GenerateOutcome, PromptTemplate, PROD_GENERATE_CONFIG,
};
use parakeet_rs::performance::next_session_id;

/// A representative messy dictation transcript — fillers, no punctuation,
/// the kind of thing the polish pass is supposed to fix. Length picked
/// so the chat-formatted prompt lands around the 200-token mark, matching
/// the latency plan's "60-token transcript input" intent (the plan's
/// "60 tokens" referred to the *output*; this is the input).
const SAMPLE_INPUT: &str = "um so I was thinking we could you know probably uh move the deadline back to next Friday I mean it's getting kind of tight and like the team's been a bit stretched with the migration work and all the on-call stuff so yeah let's just push it";

struct Args {
    model_path: PathBuf,
    reps: usize,
    warmup_reps: usize,
    show_output: bool,
    /// Tag baked into the `llm_timer` log line. The aggregator
    /// (`scripts/bench-aggregate.py`) buckets rows by this string, so
    /// two runs against the same GGUF should share a tag, and a swap
    /// to a different quant should change it. Defaults to the model
    /// filename stem (lowercased) so it tracks `--model` honestly
    /// instead of lying when the user supplies a different GGUF;
    /// `--tag` overrides for cross-quant comparison runs.
    model_tag: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model_path: Option<PathBuf> = None;
    let mut reps: usize = 100;
    let mut warmup_reps: usize = 3;
    let mut show_output = false;
    let mut model_tag: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model_path = Some(PathBuf::from(it.next().ok_or("--model needs PATH")?)),
            "--reps" => {
                reps = it
                    .next()
                    .ok_or("--reps needs N")?
                    .parse()
                    .map_err(|e| format!("--reps: {e}"))?
            }
            "--warmup-reps" => {
                warmup_reps = it
                    .next()
                    .ok_or("--warmup-reps needs N")?
                    .parse()
                    .map_err(|e| format!("--warmup-reps: {e}"))?
            }
            "--show-output" => show_output = true,
            "--tag" => model_tag = Some(it.next().ok_or("--tag needs STRING")?),
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        model_path: model_path.ok_or("--model is required")?,
        reps,
        warmup_reps,
        show_output,
        model_tag,
    })
}

/// Derive a default model tag from the GGUF filename. `--tag` overrides.
fn default_tag(model_path: &std::path::Path) -> String {
    model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "unknown-model".to_string())
}

fn print_usage() {
    eprintln!(
        "usage: bench_llm --model PATH [--reps N] [--warmup-reps N] [--show-output] [--tag NAME]\n\
         \n\
         Runs N polish iterations of a fixed sample transcript through the GGUF\n\
         at PATH, using the exact `SYSTEM_PROMPT` and decode loop from\n\
         src/polish.rs. Emits one `llm_timer` log line per iteration on stderr.\n\
         \n\
         `--show-output` prints the generated text for the first measured\n\
         iteration only — useful for sanity-checking the prompt/template\n\
         pipeline before trusting the numbers.\n\
         \n\
         `--tag NAME` sets the aggregator bucket; defaults to the model\n\
         filename stem (lowercased)."
    );
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
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

    let chat_prompt = PromptTemplate::prod().render(SAMPLE_INPUT);
    log::info!(
        "chat prompt: {} chars (sample input={} chars + template)",
        chat_prompt.len(),
        SAMPLE_INPUT.len()
    );

    // Warmup reps prime the Metal kernel cache + tokenizer; tagged with
    // a `warmup-` session_id prefix so the aggregator drops them.
    for i in 0..args.warmup_reps {
        let sid = format!("warmup-r{i:03}-{}", next_session_id());
        run_one(&backend, &model, &chat_prompt, &sid, &model_tag, false)?;
    }
    // Measured reps.
    for i in 0..args.reps {
        let sid = format!("bench-{model_tag}-r{i:03}-{}", next_session_id());
        let show = args.show_output && i == 0;
        run_one(&backend, &model, &chat_prompt, &sid, &model_tag, show)?;
    }
    Ok(())
}

/// One polish iteration. Drives `polish::generate` (same function the
/// production polish path uses) and reports timing in the shape the
/// aggregator expects.
fn run_one(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    sid: &str,
    model_tag: &str,
    show_output: bool,
) -> Result<()> {
    let mut output_buf = String::new();
    let GenerateOutcome {
        prompt_tokens,
        out_tokens,
        ttft,
        gen_time,
        truncated,
    } = polish::generate(backend, model, prompt, &PROD_GENERATE_CONFIG, |piece| {
        if show_output {
            output_buf.push_str(piece);
        }
        Ok(())
    })?;

    // `Duration::as_millis()` returns `u128`. Bench reps are sub-second
    // on every platform we care about — a >49-day rep would mean the
    // model is wedged, which is a bug, not a measurement worth logging.
    // Fail loudly instead of silently logging `u32::MAX`.
    let ttft_ms = u32::try_from(ttft.as_millis()).expect("ttft >49 days indicates wedged model");
    let gen_ms =
        u32::try_from(gen_time.as_millis()).expect("gen_time >49 days indicates wedged model");
    let total_ms = ttft_ms.saturating_add(gen_ms);
    let tokens_per_s = if gen_ms > 0 {
        f64::from(out_tokens) * 1000.0 / f64::from(gen_ms)
    } else {
        0.0
    };

    log::info!(
        "llm_timer session_id={sid} model={model_tag} prompt_tokens={prompt_tokens} \
         out_tokens={out_tokens} ttft_ms={ttft_ms} gen_ms={gen_ms} \
         total_ms={total_ms} tokens_per_s={tokens_per_s:.1} truncated={truncated}"
    );

    if show_output {
        log::info!("---- generated output ----");
        log::info!("{}", output_buf.trim());
        log::info!("--------------------------");
    }
    Ok(())
}
