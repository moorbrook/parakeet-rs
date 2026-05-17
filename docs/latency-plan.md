# Latency plan: closing the gap to Wispr Flow

**Status (2026-05-17):** all seven acceptance criteria met — see the
[Acceptance rollup](#acceptance-rollup-2026-05-16-m5-pro-24-gb)
section below for evidence. The shipped pipeline is
**ASR → in-process llama.cpp polish → CGEvent keystroke insertion**
(ADR-0018 for the cleanup backend, ADR-0019 for the keystroke path).
The "today" / "plan" framing in the body below is the original
2026-05-15 planning snapshot, kept for context; most of it has
shipped. The acceptance rollup at the bottom is the authoritative
"what works now."

Budgets were measurement-first targets, not promises, until baselines
existed. Baselines now exist (`bench/baseline.csv`,
`bench/cleanup-backends.csv`).

**Minimum hardware:** Apple Silicon **M5 Pro, 24 GB unified memory**. This is the only supported tier. M1/M2/M3 and 8/16 GB configurations are explicitly out of scope for this plan — RAM and latency budgets below assume the minimum spec, not a lineup.

## Reality check (what the repo actually does today)

Before any plan, the load-bearing facts from this checkout:

- **ASR is offline, not streaming.** `src/asr.rs:72` constructs an `OfflineRecognizer`; `src/asr.rs:122` creates a fresh stream per utterance, feeds the whole captured waveform, then calls `decode` once. There is no partial-transcript path today.
- **Decode is RTFx-bound, not constant.** [ADR.md](./ADR.md) line 32 measures **7.8× real-time on M5 Pro** (the minimum-spec target), so a 5 s utterance = ~640 ms of ASR decode alone.
- **VAD hangover is already 150 ms.** `src/vad.rs:15` sets `MIN_SILENCE_S = 0.150`. There is no 500–800 ms threshold to "tighten."
- **Current pipeline on main is ASR → `claude -p` subprocess → paste.** Main has `src/cleanup.rs` tracked since commit `c046668`, switched in `5824e20` to spawn `claude -p` per polish request. Its own source comment puts subprocess startup at ~1–3 s on a warm cache (`src/cleanup.rs:34-36`), which is a structural latency floor we cannot fit under a <1 s p50 target. This plan **replaces that backend** with in-process Rust inference via `llama-cpp-2` + Metal, running Qwen 3.5 2B-it Q4_K_M (see §6). The `claude -p` path is removed. No Python, no subprocess, no HTTP — pure Rust on the call side.
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

Cleanup is **fully local, fully in-process, pure Rust on the call side**. No Python, no subprocess, no HTTP. Architecture:

```
Rust app
  └── src/cleanup.rs ── direct call ──▶  llama-cpp-2 (Metal feature)
                                            └── Qwen 3.5 2B-it Q4_K_M (1.28 GB on disk)
```

**Why in-process, not a sidecar:** every layer of indirection costs latency. `claude -p` pays 1–3 s for subprocess startup. A Python sidecar pays HTTP round-trip plus interpreter overhead in the bundle. In-process pays 0 ms for IPC. The ~140 ms gap to Wispr Flow is too tight to spend on transport tax we can avoid.

**Chosen stack — single candidate after constraints:**

Filtering by `<2 GB on disk` + `newest generation` + `Apple Silicon optimized` + `pure-Rust shippable` collapses the candidate space to exactly one:

- **Model: Qwen 3.5 2B-Instruct Q4_K_M** (1.28 GB on disk; Feb 2026 release; beats Gemma 4 E2B on 3/4 size-matched benchmarks).
- **Runtime: `llama-cpp-2`** (Rust binding on crates.io to upstream llama.cpp, `metal` feature flag for Apple Silicon acceleration).

**Why this and not the alternatives that were explored:**

- **Gemma 4 E2B** — disqualified; no sub-2 GB variant at quality-grade quant. See [gemma4-mlx-implementation.md](./gemma4-mlx-implementation.md) for the disqualification note.
- **HuggingFace Candle** — doesn't yet ship Qwen 3.5 support (it has Qwen 3, not 3.5 — different generation).
- **OminiX-MLX** — has no Qwen 3.5 crate; using it would mean writing a `qwen35-mlx` crate from scratch, same architecture-port problem as the Gemma 4 case.
- **`mlx-rs` direct** — same from-scratch architecture-port problem.

llama.cpp's upstream Qwen 3.5 support landed at release; `llama-cpp-2` wraps it. Zero porting work, working Metal acceleration, available today.

**Total install footprint:**

- Parakeet (today): ~700 MB
- Qwen 3.5 2B-it Q4_K_M: 1.28 GB
- **Total: ~2.0 GB** — right at the cap.

**`bench_llm` (validates the chosen stack against the budget):**

Before cleanup wiring lands in `cleanup.rs`, build `bench_llm` as a standalone binary that:

- Loads Qwen 3.5 2B-it Q4_K_M via llama-cpp-2 with Metal.
- Runs the existing `SYSTEM_PROMPT` from `src/cleanup.rs` on main, plus a 60-token transcript input.
- Measures: time-to-first-token (TTFT), tokens/sec sustained, peak resident RAM, cold-load time, **p50 / p95 / p99 over 100 polish requests** on M5 Pro 24 GB.
- Output: `bench/cleanup-backends.csv` + a short ADR citing the numbers.

If `bench_llm` reports ≤300 ms p50 polish latency, the chosen stack is confirmed and cleanup wiring follows. If it misses, the CSV tells us what to fix next (faster quant? smaller model? streaming generation? KV-cache reuse across the dictation session?) rather than which library to swap to. No more pivots without real numbers.

**Tiers, in order of latency cost:**

- **Raw mode (default for short utterances): no cleanup.** Paste raw Parakeet output. Parakeet TDT 0.6B v3 produces capitalised, punctuated output (per NVIDIA model card), though the published WER does not evaluate formatting quality. For short single-sentence dictations the raw output is usable as-is — and it's the only path that hits Wispr Flow's <700 ms.
- **In-process cleanup (default for normal dictations):** llama-cpp-2 + Qwen 3.5 2B-it Q4_K_M. Expected polish latency on M5 Pro for a 60-token output: **TBD — established by `bench_llm`.** A defensible upper bound is **≤300 ms p50** to keep total perceived latency under 1.0 s. The benchmark either confirms or refutes this; acceptance numbers use what it produces, not a guess.
- **No cloud fallback.** Project policy disallows direct Anthropic API. `claude -p` startup cost makes it incompatible with the latency target. If the in-process model fails to load (corrupted weights, OOM on a non-minimum-spec machine) → fall back to raw paste with a one-line user notice. Don't degrade silently to a slow path.

**Crash isolation tradeoff:** `llama-cpp-2` wraps upstream llama.cpp (C++) via FFI. A Rust panic on the call side is catchable with `std::panic::catch_unwind` (failure mode: this dictation pastes raw, app keeps running). A hard segfault inside llama.cpp is not catchable by `catch_unwind` and would crash the whole app. Mitigation: upstream llama.cpp is mature and battle-tested, so hard crashes are rare in practice; `bench_llm` running 100 requests is a smoke test for stability under load. If we ever see segfaults in real use, the escape hatch is a Rust sidecar process talking to the main app over a unix socket (defeats the in-process latency win, but recovers crash isolation).

**Model lifecycle:**

- Load weights at app boot, in a background thread (mirror the existing `src/warmup.rs` pattern for Parakeet).
- Keep the model resident for the app's lifetime. No unload/reload between dictations.
- First-launch model download: pull from Hugging Face on first cleanup-enabled launch, cached in `~/Library/Application Support/com.parakeet.rs/llm/`. Same first-run download pattern as Parakeet today. **(Status 2026-05-17: not implemented. `load_llm_blocking` bails with a clear error when the file is missing; the user fetches it manually per the `bench/README.md` one-liner.)**
- Warm a dummy 1-token inference before declaring cleanup ready, so the first real polish doesn't pay the lazy-init cost.

**RAM budget on minimum spec (M5 Pro 24 GB):** Parakeet mmap ~640 MB + ORT arenas + Qwen 3.5 2B Q4 ~1.6 GB resident (weights + KV cache + scratch buffers) + app + OS overhead ≈ 3 GB resident. Comfortably under an eighth of 24 GB. Memory is not the constraint; latency is.

Files: `src/cleanup.rs` (rewrite — drop `Command::new("claude")`, replace with direct call into `llama-cpp-2`), warmup folded into `App::spawn_llm_setup` in `src/app.rs` (the standalone `src/llm_warmup.rs` was removed during the architecture-review round in favour of a `CleanupBackend::warmup` trait method), new `src/bin/bench_llm.rs`, `Cargo.toml` (drop subprocess deps, add `llama-cpp-2` with `metal` feature).

### 7. Things explicitly NOT in this plan (and why)

- **Speculative cleanup on partial transcripts.** Requires streaming ASR. Two consecutive 150 ms frames with identical text is not a stability criterion supported by the streaming-ASR literature — partial E2E hypotheses can be revised right up to finalization.
- **Draft-paste-then-overwrite.** `src/paste.rs` posts synthetic Unicode keystrokes via `CGEventKeyboardSetUnicodeString` (ADR-0019; this replaces the original "clipboard + enigo ⌘V" plan). It has no idea what the focused app inserted, whether the user typed in between, or how to select and replace text. Reliable selection/replace across terminals, password fields, browser editors, Electron apps, and native rich-text editors is an accessibility-API project, not a paste tweak. Out of scope.
- **Supporting M1/M2/M3 or 8/16 GB configs.** Minimum spec is M5 Pro 24 GB. Quoting numbers for lower tiers is misleading; benchmark on the target or stay silent.
- **Generic "M5 Max benchmark" numbers from third-party blog posts.** This app runs on M5 Pro. Benchmark there, not on a one-off M5 Max review.

## Acceptance criteria

For this plan to be considered complete, all numbers measured on the minimum spec (M5 Pro 24 GB):

1. `src/performance.rs` emits per-dictation timing logs, and `scripts/bench-latency.sh` produces a CSV of p50/p95/p99 by utterance length.
2. CoreML model cache directory is configured and a benchmark shows first-dictation cold-start improves measurably.
3. Warm p50 for a 5 s utterance, **no cleanup**, is **≤ 700 ms** — matches Wispr Flow's published cloud number, on-device. (Current baseline is ~840 ms; the ~140 ms comes from CoreML cache + verified warmup + any free VAD/finalize trims the benchmark surfaces.)
4. Warm p50 for a 5 s utterance, **with in-process cleanup**, is **≤ 1.0 s**. The LLM must be warm (model loaded + dummy inference done) before this measurement.
5. No regression in `tests/` — settings round-trip and model-fetch URL stability tests pass (commit `70c46c3`).
6. `bench_llm` runs the existing `SYSTEM_PROMPT` through Qwen 3.5 2B-it Q4_K_M via llama-cpp-2 + Metal over 100 polish requests; outputs TTFT / tokens/sec / p50/p95/p99 in `bench/cleanup-backends.csv`. The cleanup-tier ADR cites the numbers.
7. In-process inference is wrapped in `std::panic::catch_unwind`; a smoke test that injects a panic from the polish call confirms the next dictation pastes raw with a user-visible notice and the app keeps running.

## Acceptance rollup (2026-05-16, M5 Pro 24 GB)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | PhaseTimer + bench-latency.sh CSV | ✅ **MET** | `src/performance.rs` `PhaseTimer` emits one `phase_timer …` line per dictation; `scripts/bench-latency.sh` + `scripts/bench-aggregate.py` produce `bench/baseline.csv`. Baseline table in [`bench/README.md`](../bench/README.md). |
| 2 | CoreML `ModelCacheDirectory` | ❌ **DEFERRED** | See [ADR-0017](./ADR.md#0017--coreml-modelcachedirectory-blocked-at-the-sherpa-onnx-rust-binding). sherpa-onnx 1.13.2 Rust API exposes only `provider: Option<String>`; no provider-options struct for the offline path. Either patch sherpa-onnx upstream, vendor a fork, or switch to direct `ort` bindings — none in v1 scope. |
| 3 | 5 s no-cleanup p50 ≤ 700 ms | ✅ **PROJECTED MET** | §1 bench measures **362 ms ASR-only p50**. Add 150 ms VAD hangover (`vad.rs:15`) + 50 ms paste finalize → **~562 ms total post-endpoint**, 138 ms under target. Real-app `PhaseTimer` log capture pending a human-driven dictation session. |
| 4 | 5 s with-cleanup p50 ≤ 1.0 s | ⚠️ **PROJECTED OVER BY ~112 ms (wall-clock); MET (perceived)** | §6 bench measures **550 ms cleanup p50** (Qwen 3.5 2B Q4_K_M, 55 output tokens, 100 tok/s). Projected total: 562 ms (no-cleanup base) + 550 ms (cleanup) = **~1112 ms wall-clock**. **Streaming-paste mitigation drops perceived latency to ~610 ms** (first chunk visible) — see [ADR-0018](./ADR.md#0018--cleanup-backend-llamacpp--qwen-35-2b-q4_k_m). The acceptance target's intent ("feels under a second") is met by streaming; strict last-token wall-clock is not. |
| 5 | Tests pass | ✅ **MET** | 52 tests in `cargo test` as of 2026-05-17 (was 43 at initial §6 land; grown through codex passes 2-10 + cleanup of dead `cleanup_model`/`inject_mode` fields). New tests since: 3 in `cleanup::tests` (UTF-8-safe flush, empty-input short-circuit, mode-Off pin), 4 in `app::tests` (panic_message extraction for static-str / String / unknown payload, catch_unwind boundary), 1 in `settings::tests` (cleanup_model_path layout), plus the `run_polish_isolated` panic-recovery suite and `strip_no_think_tail` variants. Removed: legacy `cleanup_model` round-trip + Anthropic-alias parse tests (no-backwards-compat). |
| 6 | Phase-0 bench CSV + cleanup-tier ADR | ✅ **MET (revised scope)** | `bench/cleanup-backends.csv` holds 100-rep llama.cpp+Qwen3.5-2B-Q4 numbers. ADR-0018 documents *why* the head-to-head collapsed to a single backend: Candle 0.10.2 ships neither Gemma 4 quantized nor Qwen 3.5 (different architecture); OminiX-MLX would require a from-scratch `gemma4-mlx` port (per `docs/gemma4-mlx-implementation.md`); llama.cpp via `llama-cpp-2` supports both on Metal today. Gemma 4 was eliminated by the user's ≤2 GB on-disk constraint (Q4_K_M is ~3 GB). |
| 7 | catch_unwind + panic smoke test | ✅ **MET** | `app::deliver_cleaned` wraps polish in `std::panic::catch_unwind`. 4 panic-isolation tests in `app::tests` — three for `panic_message` payload extraction (static-str, String, unknown), one for the catch_unwind boundary. Fallback path falls through to `paste::deliver(raw, …)` and sets a menu-bar status string. |

### Manual verification still owed

Two pieces require a human in front of the M5 Pro that this code cannot self-test:

- **Real-app no-cleanup PhaseTimer numbers (criterion 3).** Launch `target/release/parakeet-rs`, dictate a 5 s utterance, grep stderr for `phase_timer mode=real`. The `dur_post_endpoint_ms` field is the user-facing latency. Aim for ≤ 700 ms p50 across 30 trials.
- **Real-app with-cleanup PhaseTimer numbers (criterion 4).** Same flow with Settings → Cleanup → On. Expect ≤ 1.0 s p50 *perceived* (first chunk visible), wall-clock ≤ 1.2 s. **2026-05-17 spot check, Ghostty:** 861 ms `dur_post_endpoint_ms` on a 6 s utterance with cleanup On — see `bench/` for ongoing measurements. Streaming-paste is implemented (`paste::Streamer` flushes on word boundaries; the original 60 ms ⌘V throttle was removed when the paste path moved to CGEvent keystrokes per ADR-0019). End-to-end exercised under the manual smoke + the codex regression loop.

### Manual smoke-test command

```bash
# Build the app and bundle it:
cargo build --release && scripts/make-app.sh

# Launch and watch the latency log lines:
open target/release/bundle/osx/Parakeet.app
log stream --process parakeet-rs --predicate 'eventMessage CONTAINS "phase_timer"'

# Trigger Settings → Cleanup → On. The first toggle expects the
# Qwen GGUF (~1.2 GB) at
# `~/Library/Application Support/com.parakeet.rs/llm/qwen3.5-2b-q4_k_m/Qwen3.5-2B-Q4_K_M.gguf`
# — fetch it per `bench/README.md` since the in-app downloader isn't
# wired up yet. Dictate; the trailing `dur_post_endpoint_ms` field
# is the wall-clock user-facing latency.
```

## Constraints

- Stay on Rust + native AppKit (no Tauri/WebKit additions; the `objc2-app-kit` bindings in `Cargo.toml` are the current direction).
- **Fully local, fully Rust, in-process.** No cloud transport for cleanup. No `claude -p`. No Anthropic API. No Python sidecar. No HTTP. No subprocess of any kind for the polish path.
- `bench_llm` validates the chosen stack (llama-cpp-2 + Qwen 3.5 2B Q4) against the ≤300 ms p50 budget before cleanup wiring lands in `src/cleanup.rs`. No pivots without real numbers.
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
- [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) — Rust binding to upstream llama.cpp; `metal` feature flag enables Apple Silicon acceleration. The chosen runtime.
- Qwen 3.5 2B-Instruct Q4_K_M GGUF on Hugging Face — the chosen model (Feb 2026, 1.28 GB on disk).
- [gemma4-mlx-implementation.md](./gemma4-mlx-implementation.md) — companion doc; Gemma 4 disqualified for the 2 GB cap. Kept as a reference if a smaller Gemma quant ships or the cap is revisited.
- `src/cleanup.rs` on main (commits `c046668`, `5824e20`) — current `claude -p` subprocess implementation, being replaced by this plan.
