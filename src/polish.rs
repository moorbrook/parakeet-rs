//! In-process LLM polish pass.
//!
//! Sits between `Asr::recognize()` and `paste::deliver()` in the dictation
//! pipeline. Takes the raw ASR transcript and (when enabled) cleans it
//! through a local Qwen 3.5 4B Q6_K model running on llama.cpp's Metal
//! backend to:
//!
//! - strip filler words (`um`, `uh`, `you know`, `like`),
//! - fix punctuation and capitalisation,
//! - honour inline editing commands (`new paragraph`, `scratch that`).
//!
//! **Transport: in-process via `llama-cpp-2` FFI**, replacing the previous
//! `claude -p` subprocess path. See [ADR-0018](../docs/ADR.md#0018--polish-backend-llamacpp--qwen-35-2b-q4_k_m)
//! for the library-selection rationale, measured Phase-0 numbers, and
//! rejected alternatives.
//!
//! Public surface:
//!
//! - `trait PolishBackend` — the seam between [`App`] and the in-process
//!   inference engine. Lets unit tests swap in a fake backend without
//!   needing a real GGUF on disk.
//! - `fn polish_streaming(...)` — front-door function that handles
//!   empty input, the `PolishMode::Off` short-circuit, and otherwise
//!   delegates to the backend.
//! - `fn generate(...)` — shared decode loop used by [`LlamaPolish`] in
//!   production AND by `bin/bench_llm`. Pinning these together is what
//!   makes the bench numbers in `bench/polish-backends.csv` actually
//!   measure the path users hit.
//!
//! [`App`]: crate::app::App

use std::num::NonZeroU32;
use std::path::Path;
use std::pin::pin;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use parking_lot::Mutex;

use crate::settings::{PolishMode, Settings};

/// Polish-pass system prompt. Private; assemble production-ready
/// prompts via [`PromptTemplate::prod`] so callers (bench + production)
/// can't drift.
const SYSTEM_PROMPT: &str = "You clean up raw speech-to-text transcriptions for direct insertion into the user's document. Output only the cleaned text. No preamble, no commentary, no quotes around the output, no Markdown formatting.\n\
\n\
Rules:\n\
1. Remove filler words: um, uh, er, ah, like, you know, sort of, kind of, I mean (when used as filler).\n\
2. Fix punctuation and capitalisation. Add commas, periods, question marks.\n\
3. Honour inline editing commands: 'new paragraph' or 'new line' becomes a literal newline; 'scratch that', 'delete that', or 'strike that' removes the immediately preceding sentence; 'period' / 'question mark' / 'comma' become the literal punctuation.\n\
4. Preserve the speaker's meaning, tone, and vocabulary. Do NOT paraphrase, summarise, expand, or 'improve' the content.\n\
5. Do NOT add information the speaker did not say.\n\
6. If the input is empty, single-word, or unintelligible, return it unchanged.\n\
7. Preserve technical terms, names, and code-like fragments exactly as transcribed.\n\
8. Do not call any tools. Output text only.";

/// Knob set for [`generate`]. Keep two production-facing instances:
/// [`PROD_GENERATE_CONFIG`] for the real polish path, and (implicitly)
/// the same instance reused by `bin/bench_llm` so bench numbers track
/// production behaviour rather than a bench-only sandbox.
#[derive(Clone, Copy, Debug)]
pub struct GenerateConfig {
    /// KV-cache context size in tokens. Prompt + max output must fit.
    /// System prompt (~250 tokens) + up to ~1 minute of dictation
    /// (~1000 tokens) + `max_output_tokens` ≤ 2048.
    pub ctx_size: u32,
    /// Hard cap on generated tokens. Bounds worst-case latency and acts
    /// as a safety brake against runaway generation. Polish output is
    /// roughly the same length as its input, so this must comfortably
    /// exceed the longest supported transcript — a cap below the input
    /// size silently truncates the user's dictation mid-sentence.
    pub max_output_tokens: i32,
}

/// Production knobs. Bench code imports this directly so the two paths
/// can't drift; a divergent bench would silently invalidate the
/// `bench/polish-backends.csv` numbers cited in ADR-0018.
pub const PROD_GENERATE_CONFIG: GenerateConfig = GenerateConfig {
    ctx_size: 2048,
    // 768 covers ~45 s of dictation output (output ≈ input length).
    // The old 256 cap truncated anything past ~20 s of speech with no
    // error — the generate loop just stopped and the truncated text
    // pasted as if complete.
    max_output_tokens: 768,
};

/// Timing + token counts from one [`generate`] call. Bench code emits
/// these as one `llm_timer` log line per iteration; production code
/// ignores them (the dictation pipeline has its own [`PhaseTimer`]).
///
/// [`PhaseTimer`]: crate::performance::PhaseTimer
#[derive(Clone, Copy, Debug)]
pub struct GenerateOutcome {
    pub prompt_tokens: usize,
    pub out_tokens: u32,
    /// Wall-clock from start-of-call to end-of-prefill. Includes
    /// `LlamaContext::new` + tokenize + prefill decode.
    pub ttft: Duration,
    /// Wall-clock spent in the sampler/decode loop (post-prefill).
    pub gen_time: Duration,
    /// Generation stopped because it hit `max_output_tokens` rather
    /// than the model's end-of-sequence token. The emitted text is
    /// almost certainly cut off mid-sentence; callers must surface
    /// this rather than present the output as complete.
    pub truncated: bool,
}

/// Seam between [`crate::app::App`] and the in-process inference engine.
///
/// Two implementors:
/// - [`LlamaPolish`] — production. Holds the loaded GGUF + Metal
///   backend; one instance per process.
/// - Test-only fakes (see `polish::tests`) — let `app::deliver_cleaned`
///   be exercised without a real model.
pub trait PolishBackend: Send + Sync {
    /// Run the polish transform on `text`. The caller (`polish_streaming`)
    /// has already filtered out empty input and the `PolishMode::Off`
    /// short-circuit, so the implementation only handles the "real work"
    /// case.
    fn polish_into(&self, text: &str, on_chunk: &mut dyn FnMut(&str) -> Result<()>) -> Result<()>;

    /// Throwaway run to JIT compile kernels and warm caches. Called once
    /// at boot from [`crate::app::App::spawn_llm_setup`]; cost is paid
    /// off the user's first real dictation.
    fn warmup(&self) -> Result<()>;
}

/// llama.cpp's static `LlamaBackend` plus the loaded model weights.
/// One per process; sharable across threads (`LlamaModel` is `Send +
/// Sync`). Held inside `App::llm` as `Arc<dyn PolishBackend>`.
pub struct LlamaPolish {
    backend: LlamaBackend,
    model: LlamaModel,
    /// Serialises polish calls. llama.cpp contexts themselves aren't
    /// safe to call concurrently against the same model on Metal —
    /// dispatch queue contention shows up as garbled output. Real
    /// dictation is one-polish-at-a-time anyway, so the mutex never
    /// contends.
    polish_lock: Mutex<()>,
}

impl LlamaPolish {
    /// Load weights + initialise the Metal backend. Expensive (~250 ms
    /// page-touched, plus model file mmap). Call once at app boot.
    pub fn load(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!(
                "polish model not present at {}",
                model_path.display()
            ));
        }
        // Silence llama.cpp's per-load log spew (MTL0 buffer sizes,
        // graph reservations, etc.). Useful when bench_llm prints it,
        // noise inside the menu-bar app.
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        let backend = LlamaBackend::init().context("init llama backend")?;
        let model_params = LlamaModelParams::default();
        let model_params = pin!(model_params);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("loading polish model {}", model_path.display()))?;
        Ok(Self {
            backend,
            model,
            polish_lock: Mutex::new(()),
        })
    }
}

impl PolishBackend for LlamaPolish {
    /// Caller invariant: `on_chunk` must not call back into this
    /// `LlamaPolish` (or any other code that needs `polish_lock`).
    /// The lock is held across the entire generation loop including
    /// every `on_chunk` invocation; a re-entrant callback would
    /// deadlock. The only production caller is `paste::Streamer::push`,
    /// which posts CGEvent keystrokes (ADR-0019) but never re-enters
    /// polish. The lock CAN be released around `on_chunk`, but
    /// doing so would let two polish calls interleave Metal kernel
    /// invocations, which produces garbled output (see the field's
    /// doc comment). Holding it is the lesser evil.
    fn polish_into(&self, text: &str, on_chunk: &mut dyn FnMut(&str) -> Result<()>) -> Result<()> {
        let _guard = self.polish_lock.lock();
        let prompt = PromptTemplate::prod().render(text);
        // Look-back buffer: Qwen 3.5 sometimes echoes the `/no_think`
        // directive at the very end of its output — and the model
        // even "cleans" it on the way out, so we've seen both the
        // literal `/no_think` and natural-language variants like
        // `No think`, `no_think`, etc. 16 chars is enough headroom
        // for " No think." plus a leading char or two.
        let mut pending = String::new();
        let outcome = generate(
            &self.backend,
            &self.model,
            &prompt,
            &PROD_GENERATE_CONFIG,
            |piece| {
                pending.push_str(piece);
                flush_safe_prefix(&mut pending, 16, on_chunk)
            },
        )?;
        let final_str = strip_no_think_tail(&pending);
        if !final_str.is_empty() {
            on_chunk(final_str)?;
        }
        if outcome.out_tokens == 0 {
            return Err(anyhow!("polish model produced no output"));
        }
        // Flush first (the tail is still valid text), THEN report the
        // truncation. `deliver_cleaned` keeps already-streamed output
        // on error and tells the user via status text — far better
        // than pasting a mid-sentence cutoff as if it were complete.
        if outcome.truncated {
            return Err(anyhow!(
                "polish output truncated at {} tokens (no end-of-sequence); \
                 transcript may be longer than the polish output cap",
                outcome.out_tokens
            ));
        }
        Ok(())
    }

    fn warmup(&self) -> Result<()> {
        // Throwaway "hi" polish to JIT the Metal kernels. One iteration
        // is enough — the kernel cache persists for the life of the
        // backend.
        let mut sink = |_chunk: &str| Ok(());
        self.polish_into("hi", &mut sink)
    }
}

/// Run the polish pass, invoking `on_chunk` for each generated text
/// chunk. Returns once generation hits the model's end-of-sequence
/// token or `MAX_OUTPUT_TOKENS`.
///
/// `on_chunk` is called from the polish thread (`transcribe` thread in
/// production). It should not block — slow chunk handlers stretch
/// wall-clock polish latency.
///
/// Returns `Ok(())` even when polish is disabled in settings; in that
/// case `on_chunk` is invoked exactly once with the original `text` so
/// streaming-paste callers stay symmetric.
pub fn polish_streaming<F>(
    backend: &dyn PolishBackend,
    text: &str,
    settings: &Settings,
    mut on_chunk: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
{
    if text.trim().is_empty() {
        on_chunk(text)?;
        return Ok(());
    }
    match settings.polish_mode {
        PolishMode::Off => {
            on_chunk(text)?;
            Ok(())
        }
        PolishMode::On => backend.polish_into(text, &mut on_chunk),
    }
}

/// Prompt assembly for the polish task. Hides the Qwen-specific
/// ChatML / `/no_think` / pre-filled `<think></think>` convention
/// behind a single render method so production
/// ([`LlamaPolish::polish_into`]) and bench (`bin/bench_llm`) can't
/// reassemble the prompt with subtly different shapes — the bench
/// claim "measures the production path" depends on both routes going
/// through the same template.
#[derive(Clone, Copy, Debug)]
pub struct PromptTemplate {
    system_prompt: &'static str,
}

impl PromptTemplate {
    /// Template used by the production polish path. The system prompt
    /// is fixed in `polish.rs`; there's no per-user customisation in
    /// v1.
    pub fn prod() -> Self {
        Self {
            system_prompt: SYSTEM_PROMPT,
        }
    }

    /// Render the polish request as a Qwen 3.5 ChatML prompt with two
    /// tweaks: append `/no_think` to disable the reasoning mode, and
    /// pre-fill an empty `<think></think>` block on the assistant side
    /// so the model jumps straight to the answer. Without these, Qwen
    /// 3.5 emits `<think>` reflection that blows past
    /// `max_output_tokens` and produces no usable polish output.
    pub fn render(&self, user_input: &str) -> String {
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user_input} /no_think<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
            system = self.system_prompt,
        )
    }
}

/// Shared llama.cpp decode loop. Owns context creation, tokenisation,
/// prefill, greedy sampling, and the gen loop; emits each detokenised
/// piece through `on_piece` and returns timing+count metadata in
/// [`GenerateOutcome`].
///
/// Both [`LlamaPolish::polish_into`] (production) and
/// `bin/bench_llm::run_one` (bench) go through this function. Pinning
/// them to the same path means a change to sampling strategy, batch
/// sizing, or context params immediately shows up in the bench numbers
/// rather than silently invalidating them.
pub fn generate<F>(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    cfg: &GenerateConfig,
    mut on_piece: F,
) -> Result<GenerateOutcome>
where
    F: FnMut(&str) -> Result<()>,
{
    let t_start = Instant::now();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(cfg.ctx_size));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("create llama context")?;

    // Tokenise. ChatML's `<|im_start|>` already implies sequence
    // boundaries, so `AddBos::Never` avoids a duplicate <bos>.
    let tokens_list = model
        .str_to_token(prompt, AddBos::Never)
        .context("tokenize prompt")?;
    let prompt_tokens = tokens_list.len();
    if prompt_tokens == 0 {
        return Err(anyhow!("empty prompt after tokenization"));
    }
    // Enforce the context budget BEFORE allocating the batch. Without
    // this, a prompt at the full `ctx_size` would still pass the
    // `< max_total` loop guard but write the first generated token at
    // KV position == ctx_size — outside the KV cache, undefined
    // behaviour in llama.cpp.
    if cfg.max_output_tokens <= 0 {
        return Err(anyhow!(
            "GenerateConfig.max_output_tokens must be > 0; got {}",
            cfg.max_output_tokens
        ));
    }
    let ctx_size_usize = cfg.ctx_size as usize;
    let max_output_usize = cfg.max_output_tokens as usize;
    if prompt_tokens + max_output_usize > ctx_size_usize {
        return Err(anyhow!(
            "prompt ({prompt_tokens} tokens) + max_output ({max_output_usize}) \
             exceeds ctx_size ({ctx_size_usize}); shorten the input or raise ctx_size"
        ));
    }

    // Size the prefill batch to the full context. A hardcoded 512 was
    // smaller than the prompt-token budget documented in `GenerateConfig`
    // (~750 prompt tokens for a 30 s dictation), so long transcripts
    // would fail at `batch.add` with no useful error and silently fall
    // back to raw paste.
    let mut batch = LlamaBatch::new(cfg.ctx_size as usize, 1);
    // `LlamaBatch::add` takes positions as `i32`. ctx_size is capped at
    // 2048 in `PROD_GENERATE_CONFIG`, so this never overflows in
    // practice — but a fallible conversion documents the bound and
    // turns a silent wrap into a clean error if a future caller raises
    // the context.
    let last_index = i32::try_from(prompt_tokens - 1)
        .context("prompt too long: token count exceeds i32 batch index")?;
    for (i, token) in (0_i32..).zip(tokens_list) {
        batch.add(token, i, &[0], i == last_index)?;
    }
    ctx.decode(&mut batch).context("prefill decode")?;
    let ttft = t_start.elapsed();

    // Greedy sampling: deterministic, repeatable. The polish task
    // wants exact output, not creative variation; greedy also gives
    // the cleanest tokens/sec since there's no temperature overhead.
    // No `dist` in the chain — a trailing greedy selector overrides
    // whatever an earlier `dist` picked, so chaining both is just a
    // misleading no-op.
    let mut sampler = LlamaSampler::greedy();

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = batch.n_tokens();
    let mut n_decode: u32 = 0;
    // Use `checked_add` so a misconfigured `GenerateConfig` (large
    // prompt + max_output_tokens near `i32::MAX`) fails cleanly instead
    // of wrapping. `GenerateConfig` is `pub`, so a future caller can
    // construct one directly.
    let prompt_tokens_i32 = i32::try_from(prompt_tokens)
        .context("prompt too long: token count exceeds i32 batch index")?;
    let max_total = prompt_tokens_i32
        .checked_add(cfg.max_output_tokens)
        .ok_or_else(|| anyhow!("GenerateConfig overflow: prompt + max_output_tokens > i32::MAX"))?;

    let t_gen_start = Instant::now();
    // `<`, not `<=`. With `<=` the loop runs `max_output_tokens + 1`
    // iterations and the last `batch.add(token, n_cur, ...)` writes at
    // `n_cur == prompt_tokens + max_output_tokens` — at the maximum
    // config that's `ctx_size`, one past the last valid KV slot.
    let mut hit_eog = false;
    while n_cur < max_total {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            hit_eog = true;
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .context("token_to_piece")?;
        on_piece(&piece)?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch).context("gen decode")?;
        n_decode += 1;
    }
    let gen_time = t_gen_start.elapsed();

    Ok(GenerateOutcome {
        prompt_tokens,
        out_tokens: n_decode,
        ttft,
        gen_time,
        truncated: !hit_eog,
    })
}

/// Strip any of the `/no_think` directive's echoes from the END of
/// the model's output. The directive disables Qwen 3.5's reasoning
/// trace; pre-filling `<think></think>` empty on the assistant side
/// usually suffices, but the model occasionally echoes the directive
/// — sometimes literally as `/no_think`, sometimes "cleaned" into
/// natural-language variants like `No think.` or `no think`. Matching
/// is case-insensitive and looks past trailing punctuation, but the
/// punctuation the SPEAKER's sentence ends with is preserved: only
/// the directive echo and its own surrounding separators are removed.
/// Without a directive match, the output passes through with just
/// trailing whitespace trimmed — eagerly eating terminal punctuation
/// here used to delete the final period of every single dictation.
fn strip_no_think_tail(s: &str) -> &str {
    const TERMINAL: &[char] = &[' ', '\t', '\n', '\r', '.', '!', '?', ',', ';', ':'];
    const SUFFIXES: &[&str] = &[
        "/no_think",
        "/no think",
        "no_think",
        "no think",
        "/nothink",
        "nothink",
    ];
    let match_zone = s.trim_end_matches(TERMINAL);
    for suffix in SUFFIXES {
        if let Some(stripped) = strip_suffix_ascii_ci(match_zone, suffix) {
            // Trim only whitespace before the matched directive — the
            // char preceding it may be the sentence's legitimate
            // terminal punctuation ("Hello, world. /no_think").
            return stripped.trim_end();
        }
    }
    s.trim_end()
}

/// ASCII case-insensitive suffix strip. Returns `Some(prefix)` if
/// `haystack` ends with `needle` (compared case-insensitively, ASCII
/// only — the needles we feed it are pure ASCII directive variants),
/// `None` otherwise. Guards against splitting `haystack` inside a
/// multi-byte UTF-8 codepoint.
fn strip_suffix_ascii_ci<'a>(haystack: &'a str, needle: &str) -> Option<&'a str> {
    if haystack.len() < needle.len() {
        return None;
    }
    let split = haystack.len() - needle.len();
    if !haystack.is_char_boundary(split) {
        return None;
    }
    let tail = &haystack[split..];
    if tail.eq_ignore_ascii_case(needle) {
        Some(&haystack[..split])
    } else {
        None
    }
}

/// Emit everything except the last `hold` chars of `pending` to
/// `on_chunk`, then truncate `pending` to keep only the tail. Lets us
/// look back at the most recent characters in case they're the start
/// of a `/no_think` marker we want to strip on the final flush.
fn flush_safe_prefix(
    pending: &mut String,
    hold: usize,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<()> {
    if pending.len() <= hold {
        return Ok(());
    }
    // Split on a char boundary so we don't bisect a multi-byte UTF-8
    // sequence. Walk backwards from len-hold until we hit a boundary.
    // `is_char_boundary(0)` is universally `true`, so the loop is
    // guaranteed to terminate without underflow — no need for a
    // `split_at > 0` guard.
    let mut split_at = pending.len() - hold;
    while !pending.is_char_boundary(split_at) {
        split_at -= 1;
    }
    if split_at == 0 {
        return Ok(());
    }
    // `Drain::as_str()` exposes the slice still inside `pending` without
    // copying. Holding the drain alive until after `on_chunk` keeps that
    // slice valid; dropping it finalises the removal.
    let drain = pending.drain(..split_at);
    let result = on_chunk(drain.as_str());
    drop(drain);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{PolishMode, Settings};
    use std::sync::Mutex as StdMutex;

    /// Test backend that prefixes input with "[clean] " — distinct
    /// enough that `polish_streaming` either uses it or doesn't.
    struct FakeBackend;
    impl PolishBackend for FakeBackend {
        fn polish_into(
            &self,
            text: &str,
            on_chunk: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<()> {
            on_chunk(&format!("[clean] {text}"))
        }
        fn warmup(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Test backend that records the inputs it was asked to polish.
    /// Lets a test assert "polish_streaming did NOT call the backend".
    #[derive(Default)]
    struct RecordingBackend {
        seen: StdMutex<Vec<String>>,
    }
    impl PolishBackend for RecordingBackend {
        fn polish_into(
            &self,
            text: &str,
            on_chunk: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<()> {
            self.seen.lock().unwrap().push(text.to_string());
            on_chunk(text)
        }
        fn warmup(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn strip_no_think_tail_handles_directive_variants() {
        // The Qwen `/no_think` directive bleeds into the model's
        // output in several shapes. All must be stripped from the
        // tail; the prefix — INCLUDING its terminal punctuation —
        // must survive intact.
        assert_eq!(
            strip_no_think_tail("Hello, world. /no_think"),
            "Hello, world."
        );
        assert_eq!(strip_no_think_tail("Hello. /no_think."), "Hello.");
        assert_eq!(strip_no_think_tail("Hello no_think"), "Hello");
        assert_eq!(strip_no_think_tail("Hello. No think."), "Hello.");
        assert_eq!(strip_no_think_tail("Hello no think"), "Hello");
        assert_eq!(strip_no_think_tail("Hello nothink"), "Hello");
        assert_eq!(strip_no_think_tail("Hello /nothink"), "Hello");
        // Case-insensitive
        assert_eq!(strip_no_think_tail("Hello NO_THINK"), "Hello");
        assert_eq!(strip_no_think_tail("Hello /No_Think."), "Hello");
        // Stripping doesn't consume legitimate content
        assert_eq!(
            strip_no_think_tail("Don't think about it."),
            "Don't think about it."
        );
        // Multi-byte chars in the prefix don't trip char-boundary checks
        assert_eq!(strip_no_think_tail("héllo /no_think"), "héllo");
        // No directive: output passes through untouched except trailing
        // whitespace. Eating the final period here was a real bug —
        // every dictation lost its terminal punctuation.
        assert_eq!(strip_no_think_tail(""), "");
        assert_eq!(strip_no_think_tail("Hello."), "Hello.");
        assert_eq!(strip_no_think_tail("Did it work?"), "Did it work?");
        assert_eq!(strip_no_think_tail("Hello.\n"), "Hello.");
    }

    #[test]
    fn flush_safe_prefix_holds_tail_exactly() {
        // 22-byte input, hold 12 → emit first 10 bytes verbatim, keep
        // the trailing 12 bytes in `pending`. Mutation-survivable:
        // any off-by-one in `flush_safe_prefix` fails one of these.
        let mut s = String::from("Hello, world!/no_think");
        let mut emitted = String::new();
        flush_safe_prefix(&mut s, 12, &mut |c| {
            emitted.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted, "Hello, wor");
        assert_eq!(s, "ld!/no_think");
    }

    #[test]
    fn flush_safe_prefix_noop_when_below_hold() {
        let mut s = String::from("short");
        let mut emitted = String::new();
        flush_safe_prefix(&mut s, 12, &mut |c| {
            emitted.push_str(c);
            Ok(())
        })
        .unwrap();
        assert!(emitted.is_empty());
        assert_eq!(s, "short");
    }

    #[test]
    fn flush_safe_prefix_backs_off_to_char_boundary() {
        // 'é' is two bytes (0xC3 0xA9). "hellé!" is 7 bytes total:
        //   h(1) e(1) l(1) l(1) é(2) !(1) = 7 bytes
        // hold=5 → naive split at index 2 lands ON a boundary
        // (between 'e' and 'l'), so emit "he", keep "llé!" (5 bytes).
        // Mutation-survivable: bytes asserted exactly.
        let mut s = String::from("hellé!");
        let mut emitted = String::new();
        flush_safe_prefix(&mut s, 5, &mut |c| {
            emitted.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted, "he");
        assert_eq!(s, "llé!");
        assert_eq!(emitted.len() + s.len(), "hellé!".len());
    }

    #[test]
    fn flush_safe_prefix_walks_back_when_split_lands_mid_codepoint() {
        // "ab🦀!" = a(1) b(1) 🦀(4) !(1) = 7 bytes; '🦀' occupies
        // indices 2..6, so any of 3, 4, 5 land mid-codepoint.
        // hold=4 → naive split_at = 3 (inside '🦀'); walker steps
        // back: 3→2. Index 2 IS a boundary (start of '🦀'), so emit
        // "ab" and keep "🦀!" (5 bytes). Without the backoff this
        // panics in `String::drain` for a non-boundary index.
        let mut s = String::from("ab🦀!");
        let mut emitted = String::new();
        flush_safe_prefix(&mut s, 4, &mut |c| {
            emitted.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted, "ab");
        assert_eq!(s, "🦀!");
    }

    #[test]
    fn flush_safe_prefix_walks_all_the_way_to_zero_bails_out() {
        // "🦀!" = 5 bytes; '🦀' occupies indices 0..4.
        // hold=4 → naive split_at = 1 (inside '🦀'); walker steps
        // back to 0, which is the start — function returns Ok(())
        // without emitting. Holding the whole string is correct: the
        // tail is too "fat" to flush anything without bisecting a
        // codepoint.
        let mut s = String::from("🦀!");
        let mut emitted = String::new();
        flush_safe_prefix(&mut s, 4, &mut |c| {
            emitted.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted, "");
        assert_eq!(s, "🦀!");
    }

    #[test]
    fn polish_streaming_empty_text_emits_raw_without_touching_backend() {
        let backend = RecordingBackend::default();
        let settings = Settings {
            polish_mode: PolishMode::On,
            ..Settings::default()
        };
        let mut captured = String::new();
        polish_streaming(&backend, "", &settings, |c| {
            captured.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(captured, "");
        assert!(
            backend.seen.lock().unwrap().is_empty(),
            "empty input must not invoke the backend"
        );
    }

    #[test]
    fn polish_streaming_whitespace_text_emits_raw_without_touching_backend() {
        let backend = RecordingBackend::default();
        let settings = Settings {
            polish_mode: PolishMode::On,
            ..Settings::default()
        };
        let mut captured = String::new();
        polish_streaming(&backend, "   \n  ", &settings, |c| {
            captured.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(captured, "   \n  ");
        assert!(backend.seen.lock().unwrap().is_empty());
    }

    #[test]
    fn polish_streaming_off_mode_bypasses_backend() {
        // PolishMode::Off short-circuits — the FakeBackend prefix
        // should NOT appear in the output.
        let backend = FakeBackend;
        let settings = Settings {
            polish_mode: PolishMode::Off,
            ..Settings::default()
        };
        let mut captured = String::new();
        polish_streaming(&backend, "hello world", &settings, |c| {
            captured.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(captured, "hello world");
    }

    #[test]
    fn polish_streaming_on_mode_delegates_to_backend() {
        let backend = FakeBackend;
        let settings = Settings {
            polish_mode: PolishMode::On,
            ..Settings::default()
        };
        let mut captured = String::new();
        polish_streaming(&backend, "hello world", &settings, |c| {
            captured.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(captured, "[clean] hello world");
    }

    #[test]
    fn prompt_template_prod_renders_canonical_chatml_with_no_think() {
        // Mutation-survivable: every load-bearing token of the
        // production template appears exactly once. A regression that
        // drops `/no_think` or the pre-filled `<think></think>` block
        // makes Qwen 3.5 spew reflection past `max_output_tokens` and
        // silently produce no usable output — the kind of bug that
        // only surfaces in `bench/polish-backends.csv` weeks later.
        let rendered = PromptTemplate::prod().render("hello world");
        assert_eq!(
            rendered.matches("<|im_start|>system").count(),
            1,
            "system role boundary should appear exactly once"
        );
        assert_eq!(
            rendered.matches("<|im_start|>user").count(),
            1,
            "user role boundary should appear exactly once"
        );
        assert_eq!(
            rendered.matches("<|im_start|>assistant").count(),
            1,
            "assistant role boundary should appear exactly once"
        );
        assert_eq!(
            rendered.matches("/no_think").count(),
            1,
            "/no_think directive should appear exactly once"
        );
        assert!(
            rendered.contains("<think>\n\n</think>"),
            "assistant turn must pre-fill an empty <think></think> block"
        );
        assert!(
            rendered.contains("hello world"),
            "user input must round-trip into the prompt"
        );
        // System prompt content survives — sample a load-bearing phrase
        // so a future edit that accidentally strips it fails loudly.
        assert!(
            rendered.contains("Remove filler words"),
            "system prompt must reach the rendered output"
        );
    }

    #[test]
    fn prod_generate_config_constants_match_documented_budget() {
        // ctx_size 2048 matches the latency-plan §6 / ADR-0018 bench
        // setup. max_output_tokens was raised 256 → 768 after the cap
        // was found to silently truncate dictations past ~20 s (polish
        // output ≈ input length, and a 30 s transcript alone is ~500
        // tokens).
        assert_eq!(PROD_GENERATE_CONFIG.ctx_size, 2048);
        assert_eq!(PROD_GENERATE_CONFIG.max_output_tokens, 768);
    }
}
