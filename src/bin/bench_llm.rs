//! Phase-0 LLM bench: cleanup latency on a GGUF model.
//!
//! Loads a GGUF file via llama.cpp (Metal backend on Apple Silicon),
//! runs N polish iterations against a fixed sample transcript using
//! the exact `SYSTEM_PROMPT` from `src/cleanup.rs`, and emits one
//! `llm_timer` log line per iteration to stderr in this shape:
//!
//! ```text
//! llm_timer session_id=bench-qwen3.5-2b-r042-... model=qwen3.5-2b-q4_k_m \
//!   prompt_tokens=234 out_tokens=58 ttft_ms=183 gen_ms=472 \
//!   total_ms=655 tokens_per_s=122.9
//! ```
//!
//! `scripts/bench-llm.sh` orchestrates download + run + aggregate into
//! `bench/cleanup-backends.csv`. See `docs/latency-plan.md` §6.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use parakeet_rs::cleanup::{format_chat, SYSTEM_PROMPT};
use parakeet_rs::performance::next_session_id;

/// A representative messy dictation transcript — fillers, no punctuation,
/// the kind of thing the cleanup pass is supposed to fix. Length picked
/// so the chat-formatted prompt lands around the 200-token mark, matching
/// the latency plan's "60-token transcript input" intent (the plan's
/// "60 tokens" referred to the *output*; this is the input).
const SAMPLE_INPUT: &str = "um so I was thinking we could you know probably uh move the deadline back to next Friday I mean it's getting kind of tight and like the team's been a bit stretched with the migration work and all the on-call stuff so yeah let's just push it";

/// Hard cap on generated tokens. Sized to comfortably cover the cleanup
/// output for the sample input (the cleaned form is a sentence or two,
/// well under 80 tokens). Acts as a safety brake against runaway gen.
const MAX_OUTPUT_TOKENS: i32 = 80;

/// KV-cache context size. Prompt + max output must fit. The system
/// prompt + sample input + 80 output tokens lands well under 1024.
const CTX_SIZE: u32 = 1024;

/// Model tag baked into the `llm_timer` log line. The aggregator
/// (`scripts/bench-aggregate.py`) buckets rows by this string, so two
/// runs against the same GGUF should share a tag, and a swap to a
/// different quant should change it.
const MODEL_TAG: &str = "qwen3.5-2b-q4_k_m";

struct Args {
    model_path: PathBuf,
    reps: usize,
    warmup_reps: usize,
    show_output: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model_path: Option<PathBuf> = None;
    let mut reps: usize = 100;
    let mut warmup_reps: usize = 3;
    let mut show_output = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                model_path = Some(PathBuf::from(it.next().ok_or("--model needs PATH")?))
            }
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
    })
}

fn print_usage() {
    eprintln!(
        "usage: bench_llm --model PATH [--reps N] [--warmup-reps N] [--show-output]\n\
         \n\
         Runs N polish iterations of a fixed sample transcript through the GGUF\n\
         at PATH, using the exact `SYSTEM_PROMPT` from src/cleanup.rs. Emits\n\
         one `llm_timer` log line per iteration on stderr.\n\
         \n\
         `--show-output` prints the generated text for the first measured\n\
         iteration only — useful for sanity-checking the prompt/template\n\
         pipeline before trusting the numbers."
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
    log::info!(
        "llm_load model={MODEL_TAG} path={} load_ms={load_ms}",
        args.model_path.display()
    );

    let chat_prompt = format_chat(SYSTEM_PROMPT, SAMPLE_INPUT);
    log::info!(
        "chat prompt: {} chars (system={} chars + sample input={} chars + template)",
        chat_prompt.len(),
        SYSTEM_PROMPT.len(),
        SAMPLE_INPUT.len()
    );

    // Warmup reps prime the Metal kernel cache + tokenizer; tagged with
    // a `warmup-` session_id prefix so the aggregator drops them.
    for i in 0..args.warmup_reps {
        let sid = format!("warmup-r{i:03}-{}", next_session_id());
        let _ = run_one(&backend, &model, &chat_prompt, &sid, false)?;
    }
    // Measured reps.
    for i in 0..args.reps {
        let sid = format!("bench-{MODEL_TAG}-r{i:03}-{}", next_session_id());
        let show = args.show_output && i == 0;
        let _ = run_one(&backend, &model, &chat_prompt, &sid, show)?;
    }
    Ok(())
}

/// One polish iteration. Builds a fresh context per call so the KV cache
/// doesn't carry state between iterations (production cleanup is one
/// shot per dictation; warming a shared KV cache across reps would
/// over-state real-world performance).
fn run_one(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    sid: &str,
    show_output: bool,
) -> Result<()> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CTX_SIZE));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("create llama context")?;

    let t_iter_start = Instant::now();

    // Tokenize. ChatML's `<|im_start|>` already implies sequence boundaries,
    // so AddBos::Never avoids a duplicate <bos> token.
    let tokens_list = model
        .str_to_token(prompt, AddBos::Never)
        .context("tokenize prompt")?;
    let prompt_tokens = tokens_list.len();

    let mut batch = LlamaBatch::new(512, 1);
    let last_index = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        batch.add(token, i, &[0], i == last_index)?;
    }

    // Prompt-processing (prefill) step — TTFT clock ends after this returns.
    ctx.decode(&mut batch).context("prefill decode")?;
    let ttft_ms = t_iter_start.elapsed().as_millis() as u32;

    // Greedy sampling: deterministic, repeatable. The cleanup task wants
    // exact output, not creative variation; greedy also gives the cleanest
    // tokens/sec measurement (no sampler temperature overhead).
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(1234),
        LlamaSampler::greedy(),
    ]);

    let mut n_cur = batch.n_tokens();
    let mut n_decode: u32 = 0;
    let max_total = prompt_tokens as i32 + MAX_OUTPUT_TOKENS;

    let mut decoder = if show_output {
        Some(encoding_rs::UTF_8.new_decoder())
    } else {
        None
    };
    let mut output_buf = String::new();

    let t_gen_start = Instant::now();
    while n_cur <= max_total {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        if let Some(dec) = decoder.as_mut() {
            let piece = model.token_to_piece(token, dec, true, None)?;
            output_buf.push_str(&piece);
        }
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch).context("gen decode")?;
        n_decode += 1;
    }
    let gen_ms = t_gen_start.elapsed().as_millis() as u32;
    let total_ms = t_iter_start.elapsed().as_millis() as u32;
    let tokens_per_s = if gen_ms > 0 {
        (n_decode as f32 * 1000.0) / gen_ms as f32
    } else {
        0.0
    };

    log::info!(
        "llm_timer session_id={sid} model={MODEL_TAG} prompt_tokens={prompt_tokens} \
         out_tokens={n_decode} ttft_ms={ttft_ms} gen_ms={gen_ms} \
         total_ms={total_ms} tokens_per_s={tokens_per_s:.1}"
    );

    if show_output {
        log::info!("---- generated output ----");
        log::info!("{}", output_buf.trim());
        log::info!("--------------------------");
    }
    Ok(())
}
