# Latency plan: closing the gap to Wispr Flow

**Status:** measurement-first. Budgets are targets, not promises, until baselines exist.

**Minimum hardware:** Apple Silicon **M5 Pro, 24 GB unified memory**. This is the only supported tier. M1/M2/M3 and 8/16 GB configurations are explicitly out of scope for this plan — RAM and latency budgets below assume the minimum spec, not a lineup.

## Reality check (what the repo actually does today)

Before any plan, the load-bearing facts from this checkout:

- **ASR is offline, not streaming.** `src/asr.rs:72` constructs an `OfflineRecognizer`; `src/asr.rs:122` creates a fresh stream per utterance, feeds the whole captured waveform, then calls `decode` once. There is no partial-transcript path today.
- **Decode is RTFx-bound, not constant.** [ADR.md](./ADR.md) line 32 measures **7.8× real-time on M5 Pro** (the minimum-spec target), so a 5 s utterance = ~640 ms of ASR decode alone.
- **VAD hangover is already 150 ms.** `src/vad.rs:15` sets `MIN_SILENCE_S = 0.150`. There is no 500–800 ms threshold to "tighten."
- **Current pipeline on main is ASR → `claude -p` subprocess → paste.** Main has `src/cleanup.rs` tracked since commit `c046668`, switched in `5824e20` to spawn `claude -p` per polish request. Its own source comment puts subprocess startup at ~1–3 s on a warm cache (`src/cleanup.rs:34-36`), which is a structural latency floor we cannot fit under a <1 s p50 target. This plan **replaces that backend** with in-process Rust inference via HuggingFace Candle (see §6). The `claude -p` path is removed. No Python, no subprocess, no HTTP — pure Rust.
- **`src/performance.rs` is CPU topology only**, not per-stage latency tracing. There is nothing measuring the budget today.
- **CoreML cache is not configured.** `src/asr.rs:72` requests `provider="coreml"` but no `ModelCacheDirectory` is set, so the EP recompiles subgraphs on cold start.
- **The ADR already publishes the latency story.** ADR.md:250 reports `~840 ms` on a 5 s utterance (150 ms VAD + 640 ms ASR + ~50 ms finalize). Current target is `<1 s p50 with WER ≤ 2%`. ADR-0009 documents that the streaming-Parakeet swap was evaluated and rejected.

Any plan that contradicts these facts (e.g. claiming "today's VAD is 500–800 ms" or "polish runs after endpoint" or "ASR decodes in 40–80 ms") is wrong.

## What "Wispr Flow latency" actually is

Wispr Flow's published <700 ms is cloud, GPU-backed, on a tuned Whisper-class encoder plus a fine-tuned Llama polish served by Baseten + TensorRT-LLM. Their <250 ms polish budget is achievable because they pay for the GPU. Their ASR likely runs on short audio segments via streaming. We are local-first on Apple Silicon. Fair targets on the minimum spec (M5 Pro 24 GB):

- **<700 ms p50 / <1.0 s p95** on 5 s utterances, no cleanup. Matches Wispr Flow on-device.
- **<1.0 s p50 / <1.5 s p95** on 5 s utterances, with local LLM polish. Cleanup is additive, not hidden.
- **<1.5 s p50** on 10 s utterances. Long-form dictation will lag Wispr Flow because their cloud GPU eats the encoder cost we pay on-device.

Going below the no-cleanup target requires either a streaming recognizer (already rejected in ADR-0009 — worth re-surveying, not assuming) or moving to cloud ASR.

## Where the time actually goes (5 s utterance, M5 Pro, today)

```
hotkey release / EoS
  → VAD hangover     ~150 ms     (src/vad.rs:15 MIN_SILENCE_S)
  → ASR decode       ~640 ms     (7.8× RTFx, ADR.md:32)
  → finalize + paste  ~50 ms
  ────────────────────────────
  total              ~840 ms     (ADR.md:250)
```

This ~840 ms is the baseline to beat. The ~140 ms gap to Wispr Flow's 700 ms is the entire budget for this plan — most of it has to come from either trimming the 640 ms ASR decode (CoreML cache, warmup, possibly a streaming recognizer if ADR-0009 can be re-opened) or trimming the 150 ms VAD hangover, or both.

## Levers, ranked by what we can actually move

### 1. Build the measurement infrastructure (prerequisite for everything else)

Nothing in this plan is defensible without per-stage timing on real hardware. Before any optimization:

- Extend `src/performance.rs` (currently CPU topology only) with a `PhaseTimer` that records `t_capture_end`, `t_vad_endpoint`, `t_asr_start`, `t_asr_done`, `t_paste_done`. Emit one structured log line per dictation.
- Run a benchmark harness on the **minimum-spec target (M5 Pro 24 GB)** at utterance lengths {1 s, 3 s, 5 s, 10 s, 20 s}, 30 reps each. Record p50 / p95 / p99.
- Land this **before** any other change in this plan. Without baselines, all later numbers are guessed.

Files: `src/performance.rs`, new `scripts/bench-latency.sh`.

### 2. Configure CoreML model cache (free win, no risk)

`src/asr.rs:72` builds the `OfflineRecognizer` with `provider="coreml"` but no `ModelCacheDirectory`. ONNX Runtime's CoreML EP recompiles MLProgram subgraphs on every cold start by default. Set the cache dir to `~/Library/Caches/parakeet-rs/coreml/` and verify with a cold-start benchmark.

Expected win: seconds off **first-dictation-after-launch** latency (size TBD by benchmark — ONNX Runtime CoreML EP recompile cost varies). Does not move warm p50.

Files: `src/asr.rs`.

### 3. Push CoreML warmup further into app startup

`src/warmup.rs` already runs a warmup decode at boot. Verify it actually exercises the CoreML graph (compiles + caches subgraphs) by measuring first-real-utterance latency before and after. If the warmup doesn't touch the same code path as a real decode, fix it.

Files: `src/warmup.rs`.

### 4. Re-survey streaming Parakeet (research, not commitment)

ADR-0009 rejected the streaming-model swap on prior survey. The landscape moves fast — sherpa-onnx and the Parakeet family have shipped multiple revisions since. Worth a one-day research spike to check:

- sherpa-onnx streaming Parakeet TDT variants on Apple Silicon
- WhisperKit's streaming path
- Parakeet-MLX (if it exists as a real, maintained project)

The goal of streaming ASR is not just lower endpoint→text time — it's overlapping the ASR work with the user still speaking, so the final-decode tail is short. If the survey turns up a viable option that wasn't viable in ADR-0009, write a new ADR.

Until a streaming recognizer ships, **speculative cleanup is impossible** — there's no partial transcript to speculate on.

### 5. VAD tuning (probably no headroom)

Current `MIN_SILENCE_S = 0.150`. Going lower will cut at breath pauses; a single misfire that splits a sentence into two dictations costs the user far more than 50 ms of trimmed silence saves. Do not change without a real measurement showing false-cut rate stays acceptable. Hysteresis on voiced/silent frame counts is fine to add but won't move p50.

Files: `src/vad.rs` (only if benchmark says so).

### 6. Cleanup tier — in-process Rust inference (replaces `claude -p`)

Cleanup is **fully local, fully in-process, pure Rust**. No Python, no subprocess, no HTTP. Architecture:

```
Rust app
  └── src/cleanup.rs ── direct call ──▶  Gemma 4 E2B-it, 4-bit quant
                                          (loaded into Metal via Rust ML framework)
```

**Why in-process, not a sidecar:** every layer of indirection costs latency. `claude -p` pays 1–3 s for subprocess startup. An `mlx_lm.server` Python sidecar pays 1–3 ms per HTTP round-trip plus Python interpreter overhead in the bundle. In-process pays 0 ms for IPC. The ~140 ms gap to Wispr Flow is too tight to spend on transport tax we can avoid.

**Two Rust-native candidates — pick by benchmark, not opinion:**

- **HuggingFace Candle** (`candle-core`, `candle-transformers`, `candle-nn`): minimalist Rust ML framework. Metal backend on Apple Silicon. Reported "day-0 Gemma 4 support across all modalities" (third-party summaries; **verify against the upstream `candle-transformers` crate before committing**). Mature crate, active maintainer, large user base.
- **OminiX-MLX**: Rust bindings to Apple's MLX with "dedicated crates for each model family, built for production use with zero Python." Closer to Apple's stack, likely faster on the same quantization since MLX's Metal kernels are hand-tuned by Apple's ML team. Smaller user base, less battle-tested.

Both are pure-Rust callers. Both leave the user-visible architecture identical: one in-process function call from `src/cleanup.rs`. The tradeoff is performance vs maturity, and it's settled by measurement.

**Phase-0 benchmark (gates the whole cleanup tier):**

Before any cleanup wiring lands in `cleanup.rs`, run a head-to-head on M5 Pro 24 GB:

- Same model weights (Gemma 4 E2B-it 4-bit, both libraries' supported quant format).
- Same prompt: the existing `SYSTEM_PROMPT` from `src/cleanup.rs` on main, plus a 60-token transcript input.
- Measure: time-to-first-token (TTFT), tokens/sec sustained, peak resident RAM, cold-load time, p50/p95/p99 over 100 polish requests.
- Stretch: include candle + Qwen 3.5 2B-Instruct Q4 as a third row, since Candle's Qwen support is older and stabler than its Gemma 4 support.

Output: a CSV in `bench/cleanup-backends.csv` and a one-page ADR with the winning library, model, and quant. The plan locks in only after this lands.

**Tiers, in order of latency cost:**

- **Raw mode (default for short utterances): no cleanup.** Paste raw Parakeet output. Parakeet TDT 0.6B v3 produces capitalised, punctuated output (per NVIDIA model card), though the published WER does not evaluate formatting quality. For short single-sentence dictations the raw output is usable as-is — and it's the only path that hits Wispr Flow's <700 ms.
- **In-process cleanup (default for normal dictations):** Candle or OminiX-MLX, whichever wins the Phase-0 benchmark. Candidate model: Gemma 4 E2B-it 4-bit (with Qwen 3.5 2B Q4 as the documented fallback if Gemma 4 support in the winning crate is too fresh to ship on).

  Note: "E" in Gemma cards denotes *effective* parameters. The 4-bit quant is realistically in the **2.5–3.5 GB on-disk / RAM range** including KV cache, not the 1.3 GB earlier drafts claimed.

  Expected polish latency on M5 Pro for a 60-token output: **TBD — established by the Phase-0 benchmark.** A defensible upper bound is **≤300 ms p50** to keep total perceived latency under 1.0 s. The benchmark either confirms or invalidates this; the acceptance criteria use whatever number it produces, not a guess.

- **No cloud fallback.** Project policy disallows direct Anthropic API. `claude -p` startup cost makes it incompatible with the latency target. If the in-process model fails to load (corrupted weights, OOM on a non-minimum-spec machine) → fall back to raw paste with a one-line user notice. Don't degrade silently to a slow path.

**Crash isolation tradeoff:** in-process inference means a panic / FFI segfault in the model runtime kills the dictation app. Candle is pure Rust, so the risk surface is Rust panics only — wrap polish calls in `std::panic::catch_unwind` and the failure mode is "this dictation pastes raw, the app keeps running." OminiX-MLX wraps Apple's MLX C++ via FFI; a hard segfault there cannot be caught by `catch_unwind` and would crash the whole app. This is another input to the Phase-0 ADR — if MLX wins on speed but introduces real crash risk, Candle's slightly slower path may still be the right ship.

**Model lifecycle:**

- Load weights at app boot, in a background thread (mirror the existing `src/warmup.rs` pattern for Parakeet).
- Keep the model resident for the app's lifetime. No unload/reload between dictations.
- First-launch model download: pull from Hugging Face on first cleanup-enabled launch, cached in `~/Library/Application Support/parakeet-rs/llm/`. Same first-run download pattern as Parakeet today.
- Warm a dummy 1-token inference before declaring cleanup ready, so the first real polish doesn't pay the lazy-init cost.

**RAM budget on minimum spec (M5 Pro 24 GB):** Parakeet mmap ~640 MB + ORT arenas + in-process LLM ~3 GB (weights + KV cache + scratch buffers) + app + OS overhead ≈ 4–5 GB resident. Comfortably under a quarter of 24 GB. Memory is not the constraint; latency is.

Files: `src/cleanup.rs` (rewrite — drop `Command::new("claude")`, replace with direct call into the winning inference crate), new `src/llm_warmup.rs` paralleling `src/warmup.rs`, `Cargo.toml` (drop subprocess deps, add the winning inference crate + its tokenizer dep).

### 7. Things explicitly NOT in this plan (and why)

- **Speculative cleanup on partial transcripts.** Requires streaming ASR. Two consecutive 150 ms frames with identical text is not a stability criterion supported by the streaming-ASR literature — partial E2E hypotheses can be revised right up to finalization.
- **Draft-paste-then-overwrite.** `src/paste.rs` writes the clipboard and synthesizes ⌘V via enigo. It has no idea what the focused app inserted, whether the user typed in between, or how to select and replace text. Reliable selection/replace across terminals, password fields, browser editors, Electron apps, and native rich-text editors is an accessibility-API project, not a paste tweak. Out of scope until/unless someone signs up for the AX work.
- **Supporting M1/M2/M3 or 8/16 GB configs.** Minimum spec is M5 Pro 24 GB. Quoting numbers for lower tiers is misleading; benchmark on the target or stay silent.
- **Generic "M5 Max benchmark" numbers from third-party blog posts.** This app runs on M5 Pro. Benchmark there, not on a one-off M5 Max review.

## Acceptance criteria

For this plan to be considered complete, all numbers measured on the minimum spec (M5 Pro 24 GB):

1. `src/performance.rs` emits per-dictation timing logs, and `scripts/bench-latency.sh` produces a CSV of p50/p95/p99 by utterance length.
2. CoreML model cache directory is configured and a benchmark shows first-dictation cold-start improves measurably.
3. Warm p50 for a 5 s utterance, **no cleanup**, is **≤ 700 ms** — matches Wispr Flow's published cloud number, on-device. (Current baseline is ~840 ms; the ~140 ms comes from CoreML cache + verified warmup + any free VAD/finalize trims the benchmark surfaces.)
4. Warm p50 for a 5 s utterance, **with in-process cleanup**, is **≤ 1.0 s**. The LLM must be warm (model loaded + dummy inference done) before this measurement.
5. No regression in `tests/` — settings round-trip and model-fetch URL stability tests pass (commit `70c46c3`).
6. Phase-0 benchmark (`bench/cleanup-backends.csv`) compares Candle vs OminiX-MLX on the same Gemma 4 E2B 4-bit weights and prompt; the ADR cites it as the basis for the chosen library.
7. In-process inference is wrapped in `std::panic::catch_unwind`; a smoke test that injects a panic from the polish call confirms the next dictation pastes raw with a user-visible notice and the app keeps running.

## Constraints

- Stay on Rust + native AppKit (no Tauri/WebKit additions; the `objc2-app-kit` bindings in `Cargo.toml` are the current direction).
- **Fully local, fully Rust, in-process.** No cloud transport for cleanup. No `claude -p`. No Anthropic API. No Python sidecar. No HTTP. No subprocess of any kind for the polish path.
- Phase-0 benchmark (Candle vs OminiX-MLX on Gemma 4 E2B 4-bit) gates the cleanup-tier ADR. Do not pick a library before the numbers exist.
- Minimum hardware is M5 Pro 24 GB; do not add code paths or settings UI for lower tiers.
- New dependencies need a one-paragraph justification in an ADR.
- Use `parking_lot::Mutex` (already in tree) for shared state.
- Do not break `tests/` — see commit `70c46c3` for the existing guard-rail tests.

## References

- [ADR.md](./ADR.md) — especially ADR-0009 (streaming-model swap rejected), ADR-0012 (CoreML EP shared linkage, 7.8× RTFx measurement).
- `src/asr.rs:72` — `OfflineRecognizer` construction, CoreML provider.
- `src/asr.rs:122` — fresh-stream-per-utterance decode path.
- `src/vad.rs:15` — current 150 ms hangover.
- `src/app.rs:155` — ASR → paste pipeline.
- `src/performance.rs` — current CPU-topology-only scope.
- NVIDIA Parakeet TDT 0.6B v3 model card — punctuation/capitalization claim.
- Google Gemma 4 model card — "E" = effective parameters, not total.
- [HuggingFace Candle](https://github.com/huggingface/candle) — minimalist Rust ML framework with Metal backend. Day-0 Gemma 4 support per third-party summaries; verify in `candle-transformers` before committing.
- [OminiX-MLX](https://github.com/OminiX-ai/OminiX-MLX) — Rust bindings to Apple's MLX with per-model-family crates. Closer to Apple's stack; smaller user base.
- [mlx-rs](https://github.com/oxiglade/mlx-rs) — lower-level alternative if neither of the above ships Gemma 4 cleanly. Lacks tokenizer / chat-template helpers; you'd build them.
- Hugging Face `tokenizers` crate for the Gemma 4 tokenizer.json regardless of which inference library wins.
- `src/cleanup.rs` on main (commits `c046668`, `5824e20`) — current `claude -p` subprocess implementation, being replaced by this plan.
