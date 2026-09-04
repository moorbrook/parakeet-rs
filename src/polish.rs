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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
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

/// Separator between the old and new halves of one edit line. Three
/// characters, no natural-language collision, and tokenises to a single
/// piece in the Qwen 3.5 vocabulary — the separator is paid once per
/// edit, so a chatty one would eat the savings the strategy exists for.
const EDIT_SEP: &str = "==>";

/// System prompt for [`PolishStrategy::EditsOnly`]. Same editorial
/// rules as [`SYSTEM_PROMPT`]; the output contract is what differs.
const EDITS_SYSTEM_PROMPT: &str = "You proofread raw speech-to-text transcriptions. You do not rewrite them. You output a minimal list of literal text replacements.\n\
\n\
Output format. Output only this, with no preamble, no commentary and no Markdown:\n\
- If the transcription needs no change at all, output exactly: NONE\n\
- Otherwise output one replacement per line, in the form:\n\
  OLD ==> NEW\n\
  OLD is text copied character-for-character from the transcription. NEW is what it should be replaced with. To delete text, leave NEW empty. Copy just enough surrounding words into OLD that it appears exactly once in the transcription. Write a line break inside NEW as the two characters \\n.\n\
\n\
What to change:\n\
1. Remove filler words: um, uh, er, ah, like, you know, sort of, kind of, I mean (only where they are filler).\n\
2. Fix punctuation and capitalisation.\n\
3. Apply inline editing commands: 'new paragraph' or 'new line' becomes a line break; 'scratch that', 'delete that', or 'strike that' removes the immediately preceding sentence; spoken 'period' / 'comma' / 'question mark' become the literal punctuation.\n\
4. Do NOT paraphrase, summarise, expand, reorder, or improve the wording, and do not add information the speaker did not say.\n\
5. Leave technical terms, names, and code-like fragments exactly as transcribed.\n\
6. If the input is empty, a single word, or unintelligible, output NONE.\n\
7. Do not call any tools.\n\
\n\
Never output the corrected transcription itself. Never repeat these instructions.\n\
\n\
Example. Transcription:\n\
um so the fixture is like not repeating and I want to check the clamp pressure new line also order two belts\n\
Your entire output:\n\
um so the ==> So the\n\
is like not ==> is not\n\
pressure new line also ==> pressure.\\nAlso\n\
two belts ==> two belts.";

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

/// Output cap for [`PolishStrategy::EditsOnly`].
///
/// An edit list is proportional to the number of *fixes*, not to the
/// length of the dictation, so it does not need the full-text budget.
/// The 768-token cap is actively harmful here: a model that ignores the
/// format and starts reciting the instructions back runs to the cap and
/// costs 4 s before the reply is rejected. 256 bounds that worst case at
/// roughly 6 s of decode on the 4B while still fitting about 40 edits —
/// far more than a dictation ever needs.
///
/// Truncation is safe in this strategy: [`apply_edits`] never applies a
/// partial list, so a cut-off reply becomes a raw-transcript fallback
/// rather than a half-polished paste.
pub const EDITS_GENERATE_CONFIG: GenerateConfig = GenerateConfig {
    ctx_size: 2048,
    max_output_tokens: 256,
};

// The edits cap only earns its keep while it is materially tighter than
// the full-text budget. A future edit that raises it back toward 768
// re-opens the runaway-reply cost this constant exists to bound, so
// fail the build rather than a bench run weeks later.
const _: () =
    assert!(EDITS_GENERATE_CONFIG.max_output_tokens * 2 <= PROD_GENERATE_CONFIG.max_output_tokens);

impl GenerateConfig {
    /// The decode budget a strategy runs under.
    pub fn for_strategy(strategy: PolishStrategy) -> Self {
        match strategy {
            PolishStrategy::FullText => PROD_GENERATE_CONFIG,
            PolishStrategy::EditsOnly => EDITS_GENERATE_CONFIG,
        }
    }
}

/// Which shape of output the polish pass asks the model for.
///
/// Decode is the whole cost: at 43 tok/s on Qwen 3.5 4B Q6_K, a
/// 1000 ms budget buys about 43 output tokens, and re-emitting a
/// typical dictation costs more than that. [`PolishStrategy::EditsOnly`]
/// exists to break that bound by making output length proportional to
/// the number of *fixes* rather than to the length of the transcript.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolishStrategy {
    /// The model re-emits the whole cleaned transcript. Streams to the
    /// paste target token by token.
    #[default]
    FullText,
    /// The model emits a list of literal `OLD ==> NEW` replacements,
    /// which [`apply_edits`] applies to the raw transcript. Cannot
    /// stream — the transcript is not known to be correct until the
    /// last edit line has been parsed — so the whole result arrives in
    /// one chunk.
    EditsOnly,
}

/// When to not run the model at all.
///
/// A transcript short enough to be a whole utterance on its own ("Yes.",
/// "On my way.") is exactly the case the system prompt's rule 6 already
/// tells the model to pass through unchanged, so paying a full decode
/// for it buys nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkipPolicy {
    /// Skip polish for transcripts with fewer than this many
    /// whitespace-separated words. `0` disables the skip.
    pub min_words: usize,
}

impl SkipPolicy {
    /// No skipping — every non-empty transcript reaches the model.
    pub const DISABLED: Self = Self { min_words: 0 };

    /// True when `text` is short enough that polish should be bypassed.
    pub fn skips(self, text: &str) -> bool {
        self.min_words > 0 && text.split_whitespace().count() < self.min_words
    }
}

impl Default for SkipPolicy {
    fn default() -> Self {
        Self::DISABLED
    }
}

/// Latency knobs for [`LlamaPolish`] that are independent of the decode
/// loop itself. Split from [`GenerateConfig`] because these change
/// *what we ask for*, not *how many tokens we allow*.
#[derive(Clone, Copy, Debug, Default)]
pub struct PolishTuning {
    pub strategy: PolishStrategy,
    pub skip: SkipPolicy,
}

/// Timing + token counts from one [`generate`] call. Bench code emits
/// these as one `llm_timer` log line per iteration; production code
/// ignores them (the dictation pipeline has its own [`PhaseTimer`]).
///
/// [`PhaseTimer`]: crate::performance::PhaseTimer
#[derive(Clone, Copy, Debug)]
pub struct GenerateOutcome {
    pub prompt_tokens: usize,
    /// Prompt tokens served from a [`PrefixCache`] rather than
    /// prefilled. Zero for every call through [`generate`], which
    /// starts from an empty context. Reported so a run cannot claim a
    /// prompt-cache benefit it did not receive — llama.cpp refuses
    /// partial KV removal on some architectures and this silently
    /// drops to zero when it does.
    pub reused_prompt_tokens: usize,
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

    /// When this backend wants polish bypassed entirely.
    /// [`polish_streaming`] consults it before calling
    /// [`PolishBackend::polish_into`]. Backends that always want to run
    /// (the test fakes) inherit the default of never skipping.
    fn skip_policy(&self) -> SkipPolicy {
        SkipPolicy::DISABLED
    }
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
    tuning: PolishTuning,
}

impl LlamaPolish {
    /// Load weights + initialise the Metal backend. Expensive (~250 ms
    /// page-touched, plus model file mmap). Call once at app boot.
    pub fn load(model_path: &Path) -> Result<Self> {
        Self::with_tuning(model_path, PolishTuning::default())
    }

    /// [`LlamaPolish::load`] with non-default latency knobs. Separate
    /// constructor so the bench can sweep strategies without the app
    /// having to thread a config it does not yet expose.
    pub fn with_tuning(model_path: &Path, tuning: PolishTuning) -> Result<Self> {
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
            tuning,
        })
    }

    /// [`PolishStrategy::FullText`]: stream the model's own output
    /// straight through, holding back a 16-char look-back window so a
    /// `/no_think` echo can be stripped before it reaches the document.
    fn stream_full_text(
        &self,
        text: &str,
        on_chunk: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()> {
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
        Self::check_outcome(&outcome)
    }

    /// [`PolishStrategy::EditsOnly`]: buffer the model's whole reply,
    /// apply it to `text`, emit the result in one chunk.
    ///
    /// Nothing can be streamed here. An edit list is only meaningful
    /// once it is complete — a half-read list would paste a transcript
    /// with some fixes applied and some not. The strategy therefore
    /// trades time-to-first-word for time-to-last-word, which is the
    /// trade the 1000 ms completion target asks for.
    fn apply_edit_list(
        &self,
        text: &str,
        on_chunk: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        let prompt = PromptTemplate::edits_only().render(text);
        let mut reply = String::new();
        let outcome = generate(
            &self.backend,
            &self.model,
            &prompt,
            &EDITS_GENERATE_CONFIG,
            |piece| {
                reply.push_str(piece);
                Ok(())
            },
        )?;
        // Truncation first: a cut-off edit list is missing edits, and
        // applying the surviving prefix would silently half-polish.
        // There is no partial output on screen to preserve here (the
        // strategy does not stream), so failing before `on_chunk` is
        // clean.
        Self::check_outcome(&outcome)?;
        let polished = apply_edits(text, &reply)?;
        on_chunk(&polished)
    }

    /// Shared post-generation checks: the model has to have said
    /// something, and it has to have stopped on its own.
    fn check_outcome(outcome: &GenerateOutcome) -> Result<()> {
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
        match self.tuning.strategy {
            PolishStrategy::FullText => self.stream_full_text(text, on_chunk),
            PolishStrategy::EditsOnly => self.apply_edit_list(text, on_chunk),
        }
    }

    fn skip_policy(&self) -> SkipPolicy {
        self.tuning.skip
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
        PolishMode::On if backend.skip_policy().skips(text) => {
            // Too short to be worth a decode. The system prompt's rule
            // 6 already says to return input this short unchanged, so
            // the model's own answer here is the identity — we just
            // reach it without paying for it.
            on_chunk(text)?;
            Ok(())
        }
        PolishMode::On => backend.polish_into(text, &mut on_chunk),
    }
}

/// A polish pass started on a transcript that is not final yet.
///
/// The ASR path already speculates: at the VAD endpoint *candidate* it
/// decodes a provisional transcript, then waits out the confirmation
/// window (150 ms Fast, 750 ms LongForm) to see whether the speaker
/// resumes. Polish can run inside that same window. If the confirmed
/// transcript matches the provisional one, the polish is already done
/// and the window paid for it; if not, the speculative run is discarded
/// and a fresh one starts, costing nothing but power.
///
/// This shortens the *wait*, not the polish. It hides at most one
/// confirmation window — up to 750 ms in LongForm, 150 ms in Fast — so
/// it is reported as a perceived-latency row, never as a completion p50.
///
/// Output is buffered rather than streamed: a speculative transcript
/// can still be wrong, and text pasted into the user's document cannot
/// be taken back.
pub struct Speculation {
    provisional: String,
    cancel: Arc<AtomicBool>,
    /// `None` only after [`Speculation::confirm`] or
    /// [`Speculation::cancel`] has taken it. Optional so [`Drop`] can
    /// reap a speculation the caller abandoned without moving out of a
    /// type that implements `Drop`.
    worker: Option<JoinHandle<Result<String>>>,
}

/// Sentinel error the cancel flag raises inside the decode loop. Never
/// reaches a caller — [`Speculation::confirm`] and
/// [`Speculation::cancel`] both swallow it — but it has to be
/// distinguishable from a real failure in the log.
const SPECULATION_CANCELLED: &str = "speculative polish cancelled";

impl Speculation {
    /// Start polishing `provisional` on a worker thread.
    ///
    /// The worker holds `backend`'s polish lock for the duration, and
    /// cancellation is only noticed where the polish path calls back:
    ///
    /// - [`PolishStrategy::FullText`] flushes about every 16 characters,
    ///   so the flag lands within roughly four or five tokens (~100 ms
    ///   on the 4B).
    /// - [`PolishStrategy::EditsOnly`] calls back exactly once, after
    ///   the whole reply is decoded, so the flag is never seen early. A
    ///   mispredicted speculation there makes the confirmed polish wait
    ///   out the entire speculative decode — **slower than not
    ///   speculating at all**.
    ///
    /// That asymmetry is why speculation ships coupled to the full-text
    /// strategy. Making it safe for edits-only needs a cancellation
    /// token threaded through [`PolishBackend::polish_into`] into the
    /// decode loop, which is a follow-up, not something the callback
    /// signature can express today.
    pub fn start(backend: Arc<dyn PolishBackend>, settings: Settings, provisional: String) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = {
            let cancel = Arc::clone(&cancel);
            let text = provisional.clone();
            std::thread::spawn(move || {
                let mut buf = String::new();
                polish_streaming(backend.as_ref(), &text, &settings, |chunk| {
                    if cancel.load(Ordering::Relaxed) {
                        // Aborting the decode loop is the only way out:
                        // `generate` has no cancellation of its own, and
                        // an error return from `on_piece` unwinds it
                        // immediately, releasing the polish lock.
                        return Err(anyhow!(SPECULATION_CANCELLED));
                    }
                    buf.push_str(chunk);
                    Ok(())
                })?;
                Ok(buf)
            })
        };
        Self {
            provisional,
            cancel,
            worker: Some(worker),
        }
    }

    /// The transcript this speculation was started on.
    pub fn provisional(&self) -> &str {
        &self.provisional
    }

    /// Resolve against the confirmed transcript.
    ///
    /// `Some` when the speculation was for this exact text: the polished
    /// result (or the error the polish failed with, which the caller
    /// handles exactly as it handles a synchronous polish failure).
    /// `None` when the speaker kept talking and the transcript changed —
    /// the caller must run a fresh polish, and this call has already
    /// cancelled the stale one.
    ///
    /// The comparison is **byte-identical**, deliberately: anything
    /// looser risks pasting a polish of text the speaker did not
    /// finally say. The cost is hit rate. A provisional and a confirmed
    /// transcript that differ only by a trailing period or a
    /// capitalisation the recognizer revised still miss, so the
    /// real-world hit rate will sit below the rate at which speakers
    /// actually stop talking. That rate is unmeasured — it needs the
    /// streamer hook and a capture corpus.
    pub fn confirm(mut self, confirmed: &str) -> Option<Result<String>> {
        if confirmed != self.provisional {
            self.abandon();
            return None;
        }
        self.worker.take().map(Self::reap)
    }

    /// Abandon the speculation. Signals the worker and waits for it to
    /// notice, so the polish lock is free before the caller starts
    /// anything else with the same backend.
    pub fn cancel(mut self) {
        self.abandon();
    }

    /// Signal the worker and wait for it. Idempotent: a second call
    /// finds `worker` already taken and does nothing, which is what
    /// makes `cancel()` followed by [`Drop`] safe.
    fn abandon(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = Self::reap(worker);
        }
    }

    fn reap(worker: JoinHandle<Result<String>>) -> Result<String> {
        match worker.join() {
            Ok(result) => result,
            // A panic inside polish is already handled for the
            // synchronous path by `app::run_polish_isolated`'s
            // `catch_unwind`. Here the thread boundary catches it
            // instead; either way the caller falls back to raw text.
            Err(payload) => Err(anyhow!(
                "speculative polish panicked: {}",
                panic_payload_message(&payload)
            )),
        }
    }
}

/// Dropping a `Speculation` without resolving it must not detach the
/// worker. The worker holds the backend's polish lock for its whole
/// decode, so a leaked one blocks the next real polish for up to a full
/// generation — the exact latency this type exists to remove. `Drop`
/// therefore cancels and joins, making an early return or a `?` in the
/// caller safe.
impl Drop for Speculation {
    fn drop(&mut self) {
        self.abandon();
    }
}

/// Best-effort text out of a `Box<dyn Any>` panic payload.
fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&'static str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_string())
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

    /// Template for [`PolishStrategy::EditsOnly`]. The model is asked
    /// for `OLD ==> NEW` lines; [`apply_edits`] turns them back into
    /// text.
    pub fn edits_only() -> Self {
        Self {
            system_prompt: EDITS_SYSTEM_PROMPT,
        }
    }

    /// The template a given strategy uses.
    pub fn for_strategy(strategy: PolishStrategy) -> Self {
        match strategy {
            PolishStrategy::FullText => Self::prod(),
            PolishStrategy::EditsOnly => Self::edits_only(),
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
    on_piece: F,
) -> Result<GenerateOutcome>
where
    F: FnMut(&str) -> Result<()>,
{
    let t_start = Instant::now();
    let mut ctx = new_context(backend, model, cfg)?;
    let ctx_setup = t_start.elapsed();
    let mut cache = PrefixCache::new();
    let mut outcome = generate_in(&mut ctx, model, prompt, cfg, &mut cache, on_piece)?;
    // Historical TTFT (`bench/polish-backends.csv`, ADR-0018) is
    // measured from start-of-call, context creation included. Fold the
    // setup cost back in so a run through `generate` stays comparable
    // with every number already on record.
    outcome.ttft += ctx_setup;
    Ok(outcome)
}

/// Build a context sized to `cfg`. Split out so a caller that wants to
/// keep one alive across calls (the prompt-cache path) allocates it the
/// same way `generate` does.
pub fn new_context<'m>(
    backend: &LlamaBackend,
    model: &'m LlamaModel,
    cfg: &GenerateConfig,
) -> Result<LlamaContext<'m>> {
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(cfg.ctx_size));
    model
        .new_context(backend, ctx_params)
        .context("create llama context")
}

/// Prompt tokens currently resident in sequence 0 of a context's KV
/// cache, so a later call can skip re-prefilling the shared prefix
/// (in practice: the ~250-token system prompt, identical every call).
///
/// The cache stores actual tokens rather than a length, and reuse is
/// decided by longest-common-prefix against the new prompt's tokens.
/// Comparing tokens rather than trusting a string prefix is what makes
/// this safe: tokenising `system + user` does not necessarily produce
/// `tokenize(system) ++ tokenize(user)`, so a length-based cache would
/// silently reuse KV state for a boundary token that changed.
#[derive(Default)]
pub struct PrefixCache {
    tokens: Vec<LlamaToken>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the cached prefix. Call after anything that may have left
    /// the context's KV cache in an unknown state (a caught panic, a
    /// failed decode).
    pub fn reset(&mut self) {
        self.tokens.clear();
    }

    /// How many leading tokens of `next` are already in the KV cache.
    fn reusable(&self, next: &[LlamaToken]) -> usize {
        self.tokens
            .iter()
            .zip(next)
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// [`generate`] against a caller-owned context, reusing whatever prompt
/// prefix `cache` says is already in its KV cache.
///
/// The returned `ttft` excludes context creation (the caller already
/// paid it); [`generate`] adds its own setup cost back in.
pub fn generate_in<F>(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    prompt: &str,
    cfg: &GenerateConfig,
    cache: &mut PrefixCache,
    mut on_piece: F,
) -> Result<GenerateOutcome>
where
    F: FnMut(&str) -> Result<()>,
{
    let t_start = Instant::now();
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

    // Reuse whatever leading tokens are already in seq 0's KV cache.
    // Capped at `prompt_tokens - 1`: llama.cpp needs at least one token
    // in the prefill batch to produce the logits the first sample reads,
    // so a fully-cached prompt still re-decodes its final token.
    let reuse = cache.reusable(&tokens_list).min(prompt_tokens - 1);
    // Drop every KV entry from `reuse` onward — the divergent tail of
    // the previous prompt AND that call's generated tokens, which sat
    // at positions `prompt_tokens..`. Anything left behind would be
    // attended to at the wrong position.
    let reuse_u32 = u32::try_from(reuse).context("prefix cache length exceeds u32")?;
    // `llama_memory_seq_rm` returns false when the memory module cannot
    // remove a *partial* sequence. Qwen 3.5 is a hybrid Gated-DeltaNet
    // architecture (ADR-0018): its recurrent state carries no per-token
    // position to roll back to, so llama.cpp refuses to truncate it and
    // leaves the cache untouched. Reusing the prefix anyway makes the
    // next decode fail with "inconsistent sequence positions" — the
    // cache's last position is still the previous call's last generated
    // token. Fall back to a full clear and a complete prefill.
    let partial_removed = ctx
        .clear_kv_cache_seq(Some(0), Some(reuse_u32), None)
        .context("trim kv cache to reusable prefix")?;
    let reuse = if partial_removed {
        reuse
    } else {
        ctx.clear_kv_cache_seq(Some(0), None, None)
            .context("clear kv cache after refused partial removal")?;
        0
    };
    // Committed to a reuse point; if the prefill below fails, the cache
    // no longer describes the context. Record the new prompt only after
    // the decode succeeds, and clear now so an early return can't leave
    // a stale claim behind.
    cache.reset();

    let reuse_i32 = i32::try_from(reuse).context("prefix cache length exceeds i32")?;
    for (i, token) in (reuse_i32..).zip(&tokens_list[reuse..]) {
        batch.add(*token, i, &[0], i == last_index)?;
    }
    ctx.decode(&mut batch).context("prefill decode")?;
    let ttft = t_start.elapsed();
    cache.tokens = tokens_list;

    // Greedy sampling: deterministic, repeatable. The polish task
    // wants exact output, not creative variation; greedy also gives
    // the cleanest tokens/sec since there's no temperature overhead.
    // No `dist` in the chain — a trailing greedy selector overrides
    // whatever an earlier `dist` picked, so chaining both is just a
    // misleading no-op.
    let mut sampler = LlamaSampler::greedy();

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    // KV position the first generated token occupies. This is the whole
    // prompt length, NOT `batch.n_tokens()` — with a reused prefix the
    // batch holds only the uncached tail, and writing generated tokens
    // at `tail_len` would overwrite live prefix KV entries.
    let mut n_cur = i32::try_from(prompt_tokens)
        .context("prompt too long: token count exceeds i32 batch index")?;
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
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
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
        reused_prompt_tokens: reuse,
        out_tokens: n_decode,
        ttft,
        gen_time,
        truncated: !hit_eog,
    })
}

/// Apply an [`PolishStrategy::EditsOnly`] edit list to the raw
/// transcript.
///
/// `model_output` is the model's whole reply: either the literal marker
/// `NONE`, or one `OLD ==> NEW` replacement per line.
///
/// Every failure mode is an error, never a partial apply. A half-applied
/// edit list is worse than no polish at all — the user gets a sentence
/// that is neither what they said nor what they meant, with nothing on
/// screen to tell them so. `deliver_cleaned` already treats a polish
/// error as "paste the raw transcript and say why", which is the right
/// outcome here.
///
/// The specific rejections:
/// - a line with no separator, or more than one;
/// - an empty `OLD`;
/// - an `OLD` that does not occur in the working text (the model
///   paraphrased instead of copying, so we cannot trust the `NEW`);
/// - an `OLD` that occurs more than once (ambiguous target — applying
///   it to the first match is a coin flip);
/// - a reply that parses to zero edits without being `NONE` (the model
///   answered in some other shape entirely).
pub fn apply_edits(input: &str, model_output: &str) -> Result<String> {
    let reply = strip_no_think_tail(model_output).trim();
    if reply.is_empty() {
        return Err(anyhow!("edit list was empty"));
    }
    if is_none_marker(reply) {
        return Ok(input.to_string());
    }

    let mut working = input.to_string();
    let mut applied = 0_usize;
    for line in reply.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((old_raw, new_raw)) = line.split_once(EDIT_SEP) else {
            return Err(anyhow!("edit line has no `{EDIT_SEP}` separator: {line:?}"));
        };
        if new_raw.contains(EDIT_SEP) {
            return Err(anyhow!(
                "edit line has more than one `{EDIT_SEP}` separator: {line:?}"
            ));
        }
        let old = old_raw.trim();
        if old.is_empty() {
            return Err(anyhow!("edit line has an empty search text: {line:?}"));
        }
        let new = unescape_newlines(new_raw.trim());

        let mut matches = working.match_indices(old);
        let Some((at, _)) = matches.next() else {
            return Err(anyhow!("edit search text not found in transcript: {old:?}"));
        };
        if matches.next().is_some() {
            return Err(anyhow!(
                "edit search text occurs more than once in transcript: {old:?}"
            ));
        }
        working.replace_range(at..at + old.len(), &new);
        applied += 1;
    }
    if applied == 0 {
        return Err(anyhow!("reply parsed to zero edits and was not NONE"));
    }
    Ok(collapse_spaces(&working))
}

/// True for the "nothing to change" reply. The model reaches it by
/// several routes — bare `NONE`, `NONE.`, lowercase — and all of them
/// mean the same thing.
fn is_none_marker(reply: &str) -> bool {
    reply
        .trim_end_matches(['.', '!', ' ', '\t'])
        .eq_ignore_ascii_case("none")
}

/// Turn the literal two-character sequence `\n` into a real line break.
/// The edit-line format is line-oriented, so a replacement that inserts
/// a paragraph break (`new paragraph`) has no other way to say so. `\\`
/// escapes a literal backslash; every other backslash passes through
/// unchanged, which keeps Windows-style paths and regex fragments in a
/// dictated code snippet intact.
fn unescape_newlines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Collapse runs of spaces and tabs left behind by deletions, and trim
/// each line's edges. Deleting `uh, ` out of `was, uh, pinched` is the
/// common case and leaves a double space the model never sees. Line
/// breaks survive — they are the one whitespace character an edit can
/// deliberately insert.
fn collapse_spaces(s: &str) -> String {
    s.lines()
        .map(|line| {
            let mut out = String::with_capacity(line.len());
            let mut prev_blank = false;
            for c in line.chars() {
                let blank = c == ' ' || c == '\t';
                if blank && prev_blank {
                    continue;
                }
                out.push(if blank { ' ' } else { c });
                prev_blank = blank;
            }
            out.trim().to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
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

    #[test]
    fn edits_config_caps_output_well_below_the_full_text_budget() {
        // The point of the edits-only strategy is a short reply. If its
        // cap ever creeps back up to the full-text budget, a model that
        // ignores the format silently costs the full worst-case decode
        // before the reply is rejected.
        assert_eq!(
            EDITS_GENERATE_CONFIG.ctx_size,
            PROD_GENERATE_CONFIG.ctx_size
        );
        assert_eq!(EDITS_GENERATE_CONFIG.max_output_tokens, 256);
    }

    #[test]
    fn generate_config_for_strategy_matches_the_constants() {
        assert_eq!(
            GenerateConfig::for_strategy(PolishStrategy::FullText).max_output_tokens,
            PROD_GENERATE_CONFIG.max_output_tokens
        );
        assert_eq!(
            GenerateConfig::for_strategy(PolishStrategy::EditsOnly).max_output_tokens,
            EDITS_GENERATE_CONFIG.max_output_tokens
        );
    }

    #[test]
    fn edits_prompt_carries_a_worked_example_in_the_output_format() {
        // Without a worked example a small model answers with the
        // corrected transcript, or recites the instructions back. The
        // example is what makes the format stick, so a future prompt
        // edit that drops it must fail here rather than in a bench run
        // weeks later.
        let p = PromptTemplate::edits_only().render("x");
        assert!(p.contains("Example. Transcription:"));
        assert!(p.contains("um so the ==> So the"));
        assert!(
            p.contains(r"pressure.\nAlso"),
            "escape example must survive"
        );
        assert!(p.contains("Never output the corrected transcription itself."));
    }

    // ── Edits-only strategy ─────────────────────────────────────────

    #[test]
    fn apply_edits_none_marker_returns_input_untouched() {
        let input = "The build finished in about four minutes.";
        for reply in ["NONE", "none", "NONE.", " NONE \n", "NONE /no_think"] {
            assert_eq!(apply_edits(input, reply).unwrap(), input, "reply {reply:?}");
        }
    }

    #[test]
    fn apply_edits_applies_replacements_in_order() {
        let input = "Um, I think we should, uh, ship it on Tuesday.";
        let reply = "Um, I ==> I\n, uh, ==> ";
        assert_eq!(
            apply_edits(input, reply).unwrap(),
            "I think we should ship it on Tuesday."
        );
    }

    #[test]
    fn apply_edits_expands_escaped_newline() {
        let input = "Ship the parts Monday new paragraph invoice follows.";
        let reply = "Monday new paragraph invoice ==> Monday.\\nInvoice";
        assert_eq!(
            apply_edits(input, reply).unwrap(),
            "Ship the parts Monday.\nInvoice follows."
        );
    }

    #[test]
    fn apply_edits_rejects_target_that_is_not_in_the_transcript() {
        // The model paraphrased the search text instead of copying it.
        // Applying the NEW half anyway would be guesswork.
        let err = apply_edits("I think we should ship it.", "we ought to ==> we should")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"), "unexpected error: {err}");
    }

    #[test]
    fn apply_edits_rejects_ambiguous_target() {
        let err = apply_edits("the cat sat on the mat", "the ==> a")
            .unwrap_err()
            .to_string();
        assert!(err.contains("more than once"), "unexpected error: {err}");
    }

    #[test]
    fn apply_edits_rejects_line_without_separator() {
        let err = apply_edits("hello there", "just remove the filler")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no `==>` separator"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn apply_edits_rejects_line_with_two_separators() {
        let err = apply_edits("hello there", "hello ==> hi ==> hey")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("more than one `==>`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn apply_edits_rejects_empty_search_text() {
        let err = apply_edits("hello there", "  ==> hi")
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty search text"), "unexpected error: {err}");
    }

    #[test]
    fn apply_edits_rejects_empty_reply() {
        assert!(apply_edits("hello there", "   ").is_err());
    }

    #[test]
    fn apply_edits_failure_leaves_no_partial_result() {
        // First edit is applicable, second is not. The whole call must
        // fail — a transcript with edit 1 applied and edit 2 missing is
        // neither what the speaker said nor what they meant, and the
        // caller has no way to tell it apart from a clean polish.
        let input = "Um, the fixture is not repeating.";
        let reply = "Um, the ==> The\nnot repeating ==> not repeating well";
        assert!(apply_edits(input, reply).is_ok());
        let bad = "Um, the ==> The\nthe clamp ==> a clamp";
        assert!(apply_edits(input, bad).is_err());
    }

    #[test]
    fn apply_edits_collapses_whitespace_left_by_a_deletion() {
        // Deleting a mid-sentence filler leaves a double space that the
        // model never sees, so nothing else would clean it up.
        let input = "So the encoder cable was uh pinched.";
        assert_eq!(
            apply_edits(input, "uh ==> ").unwrap(),
            "So the encoder cable was pinched."
        );
    }

    #[test]
    fn apply_edits_preserves_inserted_line_breaks_while_collapsing_spaces() {
        let input = "Add two spare belts new line add one coupling";
        let out = apply_edits(
            input,
            "belts new line add ==> belts.\\nAdd\ncoupling ==> coupling.",
        )
        .unwrap();
        assert_eq!(out, "Add two spare belts.\nAdd one coupling.");
    }

    #[test]
    fn unescape_newlines_leaves_unknown_escapes_alone() {
        // A dictated Windows path or regex fragment must survive.
        assert_eq!(unescape_newlines(r"C:\temp\data"), r"C:\temp\data");
        assert_eq!(unescape_newlines(r"a\nb"), "a\nb");
        assert_eq!(unescape_newlines(r"a\\nb"), r"a\nb");
        assert_eq!(unescape_newlines(r"trailing\"), r"trailing\");
    }

    #[test]
    fn edits_prompt_states_the_output_contract() {
        let rendered = PromptTemplate::edits_only().render("hello world");
        assert!(rendered.contains("OLD ==> NEW"));
        assert!(rendered.contains("output exactly: NONE"));
        assert!(rendered.contains("hello world"));
        assert_eq!(rendered.matches("/no_think").count(), 1);
        // The two strategies must not share a system prompt — that was
        // the whole point of the split.
        assert_ne!(rendered, PromptTemplate::prod().render("hello world"));
    }

    #[test]
    fn for_strategy_selects_the_matching_template() {
        let user = "hello world";
        assert_eq!(
            PromptTemplate::for_strategy(PolishStrategy::FullText).render(user),
            PromptTemplate::prod().render(user)
        );
        assert_eq!(
            PromptTemplate::for_strategy(PolishStrategy::EditsOnly).render(user),
            PromptTemplate::edits_only().render(user)
        );
    }

    // ── Skip policy ─────────────────────────────────────────────────

    #[test]
    fn skip_policy_counts_words_not_characters() {
        let p = SkipPolicy { min_words: 4 };
        assert!(p.skips("Yes."));
        assert!(p.skips("Sounds good."));
        assert!(p.skips("On my way."));
        assert!(!p.skips("I'll be there shortly."));
        // A single very long word is still one word.
        assert!(p.skips("supercalifragilisticexpialidocious"));
    }

    #[test]
    fn skip_policy_disabled_never_skips() {
        assert!(!SkipPolicy::DISABLED.skips("Yes."));
        assert!(!SkipPolicy::default().skips(""));
    }

    #[test]
    fn polish_streaming_skips_short_input_without_touching_backend() {
        struct ShortSkipping(RecordingBackend);
        impl PolishBackend for ShortSkipping {
            fn polish_into(
                &self,
                text: &str,
                on_chunk: &mut dyn FnMut(&str) -> Result<()>,
            ) -> Result<()> {
                self.0.polish_into(text, on_chunk)
            }
            fn warmup(&self) -> Result<()> {
                Ok(())
            }
            fn skip_policy(&self) -> SkipPolicy {
                SkipPolicy { min_words: 4 }
            }
        }
        let backend = ShortSkipping(RecordingBackend::default());
        let settings = Settings {
            polish_mode: PolishMode::On,
            ..Settings::default()
        };
        let mut captured = String::new();
        polish_streaming(&backend, "On my way.", &settings, |c| {
            captured.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(captured, "On my way.");
        assert!(
            backend.0.seen.lock().unwrap().is_empty(),
            "3-word input must not reach the model"
        );

        // One word over the threshold and the backend does run.
        let mut captured = String::new();
        polish_streaming(&backend, "I am on my way.", &settings, |c| {
            captured.push_str(c);
            Ok(())
        })
        .unwrap();
        assert_eq!(backend.0.seen.lock().unwrap().len(), 1);
        assert_eq!(captured, "I am on my way.");
    }

    #[test]
    fn default_tuning_is_todays_shipped_behaviour() {
        // `PolishTuning::default()` is what `LlamaPolish::load` uses, so
        // a change of default silently changes production. Pin it.
        let t = PolishTuning::default();
        assert_eq!(t.strategy, PolishStrategy::FullText);
        assert_eq!(t.skip, SkipPolicy::DISABLED);
    }

    // ── Speculative polish ──────────────────────────────────────────

    fn on_settings() -> Settings {
        Settings {
            polish_mode: PolishMode::On,
            ..Settings::default()
        }
    }

    #[test]
    fn speculation_confirmed_returns_the_buffered_polish() {
        let backend: Arc<dyn PolishBackend> = Arc::new(FakeBackend);
        let spec = Speculation::start(
            Arc::clone(&backend),
            on_settings(),
            "hello world".to_string(),
        );
        assert_eq!(spec.provisional(), "hello world");
        let out = spec.confirm("hello world").expect("transcript matched");
        assert_eq!(out.unwrap(), "[clean] hello world");
    }

    #[test]
    fn speculation_on_changed_transcript_returns_none() {
        // The speaker resumed and the confirmed transcript grew. The
        // speculative result is for text the user never finished saying,
        // so handing it back would paste a truncated sentence.
        let recorder = Arc::new(RecordingBackend::default());
        let backend: Arc<dyn PolishBackend> = Arc::clone(&recorder) as Arc<dyn PolishBackend>;
        let spec = Speculation::start(backend, on_settings(), "hello".to_string());
        assert!(spec.confirm("hello world").is_none());
        // It still ran — the cost of a wrong guess is power, not
        // correctness — but nothing reached the caller.
        assert_eq!(recorder.seen.lock().unwrap().as_slice(), ["hello"]);
    }

    #[test]
    fn speculation_cancel_joins_the_worker() {
        // `cancel` must not return while the worker still holds the
        // backend's polish lock, or the fresh polish that follows would
        // race it on the Metal queue.
        let recorder = Arc::new(RecordingBackend::default());
        let backend: Arc<dyn PolishBackend> = Arc::clone(&recorder) as Arc<dyn PolishBackend>;
        Speculation::start(backend, on_settings(), "hello".to_string()).cancel();
        assert_eq!(recorder.seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn dropping_a_speculation_does_not_detach_the_worker() {
        // A dropped speculation must not leave a worker holding the
        // backend's polish lock — that would block the next real polish
        // for a whole decode, which is worse than never speculating.
        // The lock is the thing under test: if `Drop` failed to join,
        // this `lock()` would still be contended when we reach it.
        let recorder = Arc::new(RecordingBackend::default());
        let backend: Arc<dyn PolishBackend> = Arc::clone(&recorder) as Arc<dyn PolishBackend>;
        drop(Speculation::start(
            backend,
            on_settings(),
            "hello".to_string(),
        ));
        assert_eq!(
            recorder.seen.lock().unwrap().len(),
            1,
            "Drop must join the worker, not detach it"
        );
    }

    #[test]
    fn cancel_then_drop_is_not_a_double_join() {
        // `cancel` takes the handle; the subsequent `Drop` must find it
        // already gone rather than panicking on a second join.
        let recorder = Arc::new(RecordingBackend::default());
        let backend: Arc<dyn PolishBackend> = Arc::clone(&recorder) as Arc<dyn PolishBackend>;
        Speculation::start(backend, on_settings(), "hello".to_string()).cancel();
        assert_eq!(recorder.seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn confirm_then_drop_is_not_a_double_join() {
        let backend: Arc<dyn PolishBackend> = Arc::new(FakeBackend);
        let out = Speculation::start(backend, on_settings(), "hi".to_string())
            .confirm("hi")
            .expect("transcript matched")
            .unwrap();
        assert_eq!(out, "[clean] hi");
    }

    #[test]
    fn speculation_reports_a_backend_failure_to_the_caller() {
        struct FailingBackend;
        impl PolishBackend for FailingBackend {
            fn polish_into(
                &self,
                _text: &str,
                _on_chunk: &mut dyn FnMut(&str) -> Result<()>,
            ) -> Result<()> {
                Err(anyhow!("model exploded"))
            }
            fn warmup(&self) -> Result<()> {
                Ok(())
            }
        }
        let backend: Arc<dyn PolishBackend> = Arc::new(FailingBackend);
        let spec = Speculation::start(backend, on_settings(), "hello".to_string());
        let err = spec
            .confirm("hello")
            .expect("transcript matched")
            .unwrap_err();
        assert!(err.to_string().contains("model exploded"));
    }

    #[test]
    fn speculation_survives_a_panicking_backend() {
        // A polish panic is isolated by `catch_unwind` on the
        // synchronous path; on the speculative path the thread boundary
        // does it. Either way the app must not die.
        struct PanickingBackend;
        impl PolishBackend for PanickingBackend {
            fn polish_into(
                &self,
                _text: &str,
                _on_chunk: &mut dyn FnMut(&str) -> Result<()>,
            ) -> Result<()> {
                panic!("metal kernel fault");
            }
            fn warmup(&self) -> Result<()> {
                Ok(())
            }
        }
        let backend: Arc<dyn PolishBackend> = Arc::new(PanickingBackend);
        let spec = Speculation::start(backend, on_settings(), "hello".to_string());
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = spec.confirm("hello").expect("transcript matched");
        std::panic::set_hook(prev);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("panicked"), "unexpected error: {err}");
        assert!(
            err.contains("metal kernel fault"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn speculation_respects_polish_mode_off() {
        // Speculating while polish is disabled would burn the GPU for a
        // result nobody asked for.
        let recorder = Arc::new(RecordingBackend::default());
        let backend: Arc<dyn PolishBackend> = Arc::clone(&recorder) as Arc<dyn PolishBackend>;
        let settings = Settings {
            polish_mode: PolishMode::Off,
            ..Settings::default()
        };
        let out = Speculation::start(backend, settings, "hello world".to_string())
            .confirm("hello world")
            .expect("transcript matched")
            .unwrap();
        assert_eq!(out, "hello world");
        assert!(recorder.seen.lock().unwrap().is_empty());
    }

    // ── Prefix cache ────────────────────────────────────────────────

    #[test]
    fn prefix_cache_reports_longest_common_token_prefix() {
        let mut cache = PrefixCache::new();
        let a: Vec<LlamaToken> = [1, 2, 3, 4].iter().map(|t| LlamaToken(*t)).collect();
        let b: Vec<LlamaToken> = [1, 2, 9, 4].iter().map(|t| LlamaToken(*t)).collect();
        assert_eq!(cache.reusable(&a), 0, "empty cache reuses nothing");
        cache.tokens = a.clone();
        assert_eq!(cache.reusable(&a), 4);
        assert_eq!(cache.reusable(&b), 2);
        assert_eq!(cache.reusable(&[]), 0);
        cache.reset();
        assert_eq!(cache.reusable(&a), 0, "reset must forget the prefix");
    }

    #[test]
    fn prefix_cache_reuse_stops_at_the_shorter_sequence() {
        let mut cache = PrefixCache::new();
        cache.tokens = [1, 2, 3].iter().map(|t| LlamaToken(*t)).collect();
        let longer: Vec<LlamaToken> = [1, 2, 3, 4, 5].iter().map(|t| LlamaToken(*t)).collect();
        assert_eq!(cache.reusable(&longer), 3);
    }

    // ── Property tests ──────────────────────────────────────────────
    //
    // The example tests above each pin ONE instance of a char-boundary
    // or tail-strip bug; these properties pin the whole class across
    // arbitrary unicode input. `\PC*` = any sequence of non-control
    // chars (multi-byte included).
    use proptest::prelude::*;

    proptest! {
        /// Splitting never loses or duplicates bytes: what was emitted
        /// plus what's still pending is exactly the original string —
        /// and the function never panics on a mid-codepoint split.
        #[test]
        fn flush_safe_prefix_is_lossless(s in "\\PC*", hold in 0usize..32) {
            let mut pending = s.clone();
            let mut emitted = String::new();
            flush_safe_prefix(&mut pending, hold, &mut |c| {
                emitted.push_str(c);
                Ok(())
            })
            .unwrap();
            prop_assert_eq!(format!("{emitted}{pending}"), s);
        }

        /// The look-back contract: after a flush the tail keeps at
        /// least `hold` bytes (or the whole string when it's shorter),
        /// so a directive marker arriving in pieces can't slip out.
        #[test]
        fn flush_safe_prefix_retains_hold(s in "\\PC*", hold in 0usize..32) {
            let mut pending = s.clone();
            flush_safe_prefix(&mut pending, hold, &mut |_| Ok(())).unwrap();
            prop_assert!(pending.len() >= hold.min(s.len()));
        }

        /// Tail-stripping only ever removes from the END: the output is
        /// always a prefix of the input, on a char boundary, no panics.
        #[test]
        fn strip_no_think_tail_returns_prefix(s in "\\PC*") {
            let out = strip_no_think_tail(&s);
            prop_assert!(s.starts_with(out));
        }

        /// Inputs that cannot end in a directive variant (forced 'x'
        /// terminator, no trailing whitespace/punctuation) pass through
        /// completely untouched — the eager-trim regression fixed in
        /// 2026-06 stays fixed for every input, not just "Hello.".
        #[test]
        fn strip_no_think_tail_no_directive_is_identity(s in "[a-zA-Z0-9 .,!?']*") {
            let input = format!("{s}x");
            prop_assert_eq!(strip_no_think_tail(&input), input.as_str());
        }
    }
}
