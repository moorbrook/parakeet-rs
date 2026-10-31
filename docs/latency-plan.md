# Latency plan: closing the gap to Wispr Flow

> **Status: archived planning snapshot.** This document records the May 2026
> route for recognition, endpointing, and paste latency on the sherpa pipeline.
> It is not the current architecture guide. The authoritative current state is
> [`ADR.md`](./ADR.md), while commands and measurements are maintained in
> [`../bench/README.md`](../bench/README.md).

**Current outcome:** the shipping pipeline is resident Parakeet Unified Core ML
recognition with speculative decode, an optional rewrite through an
already-running OpenCode service (ADR-0036), and synthetic Unicode keystroke
delivery. Tap Fast measured 148.5 ms p50 from acoustic endpoint to
transcript-ready after ADR-0031 moved its confirmation window from 150 to
90 ms; pause-friendly Tap stayed below one second p95 on the endpoint corpus.

The body below preserves the dated plan and its assumptions for decision
history. Statements labelled “original,” “snapshot,” or “historical” are not
claims about the current checkout.

**Benchmark target for this archived plan:** Apple Silicon **M5 Pro, 24 GB
unified memory**. Other Apple Silicon tiers were out of scope for these
particular budgets; this is not the application's compatibility floor.

## Original baseline snapshot (2026-05-15)

Before the work landed, the load-bearing facts were:

- **ASR is offline, not streaming.** `src/asr.rs:72` constructs an `OfflineRecognizer`; `src/asr.rs:122` creates a fresh stream per utterance, feeds the whole captured waveform, then calls `decode` once. There is no partial-transcript path today.
- **Decode is RTFx-bound, not constant.** [ADR.md](./ADR.md) line 32 measures **7.8× real-time on M5 Pro** (the minimum-spec target), so a 5 s utterance = ~640 ms of ASR decode alone.
- **VAD hangover was 150 ms.** At this planning snapshot, `src/vad.rs` used a
  single 150 ms confirmation window. ADR-0025 later split that into a
  pause-friendly 750 ms Tap default and an explicit 150 ms Tap Fast mode.
- **`src/performance.rs` is CPU topology only**, not per-stage latency tracing. There is nothing measuring the budget today.
- **CoreML cache is not configured.** `src/asr.rs:72` requests `provider="coreml"` but no `ModelCacheDirectory` is set, so the EP recompiles subgraphs on cold start.
- **The ADR already publishes the latency story.** ADR.md:250 reports `~840 ms` on a 5 s utterance (150 ms VAD + 640 ms ASR + ~50 ms finalize). Current target is `<1 s p50 with WER ≤ 2%`. ADR-0009 documents that the streaming-Parakeet swap was evaluated and rejected.

These facts explain the plan that followed; later ADRs intentionally supersede
several of them.

## What "Wispr Flow latency" actually is

Wispr Flow's published <700 ms is cloud, GPU-backed, on a tuned Whisper-class encoder. Their ASR likely runs on short audio segments via streaming. We are local-first on Apple Silicon. Fair targets on the minimum spec (M5 Pro 24 GB):

- **<700 ms p50 / <1.0 s p95** on 5 s utterances. Matches Wispr Flow on-device.
- **<1.5 s p50** on 10 s utterances. Long-form dictation will lag Wispr Flow because their cloud GPU eats the encoder cost we pay on-device.

Going below the 5 s target requires either a streaming recognizer (already rejected in ADR-0009 — worth re-surveying, not assuming) or moving to cloud ASR.

## Original latency breakdown (5 s utterance, M5 Pro)

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

### 5. VAD tuning (probably no headroom)

At this planning snapshot, `MIN_SILENCE_S = 0.150`. Going lower would cut at
breath pauses; a single misfire that splits a sentence into two dictations
costs the user far more than 50 ms of trimmed silence saves. ADR-0025 resolved
the broader long-pause problem by making 750 ms the normal Tap policy, keeping
150 ms as Tap Fast, and adding a real-speech false-stop/latency gate.

Files: `src/vad.rs` (only if benchmark says so).

### 6. Things explicitly NOT in this plan (and why)

- **Draft-paste-then-overwrite.** `src/paste.rs` posts synthetic Unicode keystrokes via `CGEventKeyboardSetUnicodeString` (ADR-0019; this replaces the original "clipboard + enigo ⌘V" plan). It has no idea what the focused app inserted, whether the user typed in between, or how to select and replace text. Reliable selection/replace across terminals, password fields, browser editors, Electron apps, and native rich-text editors is an accessibility-API project, not a paste tweak. Out of scope.
- **Supporting M1/M2/M3 or 8/16 GB configs.** Minimum spec is M5 Pro 24 GB. Quoting numbers for lower tiers is misleading; benchmark on the target or stay silent.
- **Generic "M5 Max benchmark" numbers from third-party blog posts.** This app runs on M5 Pro. Benchmark there, not on a one-off M5 Max review.

## Original acceptance criteria

For this plan to be considered complete, all numbers measured on the minimum spec (M5 Pro 24 GB):

1. `src/performance.rs` emits per-dictation timing logs, and `scripts/bench-latency.sh` produces a CSV of p50/p95/p99 by utterance length.
2. CoreML model cache directory is configured and a benchmark shows first-dictation cold-start improves measurably.
3. Warm p50 for a 5 s utterance is **≤ 700 ms** — matches Wispr Flow's published cloud number, on-device. (Current baseline is ~840 ms; the ~140 ms comes from CoreML cache + verified warmup + any free VAD/finalize trims the benchmark surfaces.)
4. No regression in `tests/` — settings round-trip and model-fetch URL stability tests pass (commit `70c46c3`).

## Outcome rollup (amended through 2026-08-11, M5 Pro 24 GB)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | PhaseTimer + bench-latency.sh CSV | ✅ **MET** | `src/performance.rs` `PhaseTimer` emits one `phase_timer …` line per dictation; `scripts/bench-latency.sh` + `scripts/bench-aggregate.py` produce `bench/baseline.csv`. Baseline table in [`bench/README.md`](../bench/README.md). |
| 2 | CoreML `ModelCacheDirectory` | **SUPERSEDED** | The sherpa binding still lacks this seam ([ADR-0017](./ADR.md#0017--coreml-modelcachedirectory-blocked-at-the-sherpa-onnx-rust-binding)), but the shipping backend moved to a resident native Core ML worker in ADR-0022. Startup warmup and Core ML's compiled plans keep initialization outside dictation. |
| 3 | 5 s p50 ≤ 700 ms | ✅ **MEASURED: 182.0 ms** | ADR-0023's matched production-capture replay measures **182.0 ms p50 / 203.0 ms p95** from Core Audio's final non-silent playback instant to transcript-ready, versus 613.0 / 652.5 ms for the serial baseline (**3.37× / 3.21×**). Synthetic Unicode delivery is the only excluded sub-ms step. |
| 4 | Tests pass | **MET** | The current suite covers settings compatibility, model integrity, endpoint policy, tuning evidence, permission policy, rewrite fallback, and delivery fallback. `cargo test --locked` is the current verification command; this document no longer pins a test count that immediately goes stale. |

### Current replay and smoke-test commands

```bash
# Build the worker and complete application bundle:
scripts/make-app.sh

# Launch and watch the latency log lines:
open target/release/bundle/osx/Parakeet.app
log stream --process parakeet-rs --predicate 'eventMessage CONTAINS "phase_timer"'

# Dictate; `dur_post_endpoint_ms` is the full post-endpoint wall clock.

# Reproduce the automated end-to-end and endpoint-policy gates:
REPS=30 WARMUP_REPS=2 scripts/bench-end-to-end.sh
REPS=30 WARMUP_REPS=2 scripts/bench-endpoint-policy.sh
```

## Historical constraints

- Stay on Rust + native AppKit (no Tauri/WebKit additions; the `objc2-app-kit` bindings in `Cargo.toml` are the current direction).
- Minimum hardware is M5 Pro 24 GB; do not add code paths or settings UI for lower tiers.
- New dependencies need a one-paragraph justification in an ADR.
- Use `parking_lot::Mutex` (already in tree) for shared state.
- Do not break `tests/` — see commit `70c46c3` for the existing guard-rail tests.

## References

- [ADR.md](./ADR.md) — especially ADR-0009 (streaming-model swap rejected), ADR-0012 (CoreML EP shared linkage, 7.8× RTFx measurement).
- `src/coreml_worker.rs` — resident native optimized backend and runtime-plan seam.
- `src/asr.rs` — common recognizer interface and sherpa fallback.
- `src/endpointing.rs` — current pause-friendly Tap and Tap Fast policies.
- `src/app.rs` — capture, speculative ASR, optional rewrite, and delivery orchestration.
- `src/performance.rs` — phase timing and Apple Silicon topology metadata.
- NVIDIA Parakeet TDT 0.6B v3 model card — punctuation/capitalization claim.
