# Latency bench

`scripts/bench-latency.sh` drives `bench_asr` over generated TTS WAVs at
{1, 3, 5, 10, 20}s, 30 reps each, and emits `phase_timer` log lines that
`scripts/bench-aggregate.py` reduces into `baseline.csv` (or `$OUT_CSV`).

See `docs/latency-plan.md` §1 for design and acceptance criteria.

## Quick start

```bash
# First time only: launch Parakeet.app once so the model bundle downloads
# into ~/Library/Application Support/com.parakeet.rs/models/.
open target/release/bundle/osx/Parakeet.app   # or however you launch it

# Then:
scripts/bench-latency.sh                           # → bench/baseline.csv
OUT_CSV=bench/post-coreml-cache.csv \
    scripts/bench-latency.sh                       # § 2 re-bench
```

## What is and isn't measured

The bench loads pre-recorded WAVs and runs `Asr::recognize()` directly.
It **does not** exercise:

- `cpal` mic-capture callback latency
- the Silero VAD endpoint hangover (~150 ms per `src/vad.rs:15`)
- the `CGEventKeyboardSetUnicodeString` keystroke insertion step
  (sub-ms per chord — see ADR-0019)

So the bench number is **ASR-only**. Real end-to-end is the bench number
plus ~150 ms (VAD hangover; keystroke insertion is negligible), captured
separately by the in-app PhaseTimer that emits the same `phase_timer`
log line during live dictation.

## Baseline: M5 Pro 24 GB (2026-05-16, pre-§2 CoreML cache)

| length | n  | mean ms | p50 ms | p95 ms | p99 ms |
|--------|----|---------|--------|--------|--------|
| 1 s    | 30 | 121     | 121    | 136    | 146    |
| 3 s    | 30 | 229     | 227    | 237    | 263    |
| 5 s    | 30 | 364     | **362**| 376    | 405    |
| 10 s   | 30 | 573     | 572    | 589    | 591    |
| 20 s   | 30 | 1120    | 1114   | 1162   | 1185   |

**Steady-state RTFx** ≈ 13–14× real time on the 5 s bucket. This is
materially better than ADR-0012's 7.8× figure — likely due to OS / driver
updates and/or that the bench uses clean TTS speech. Worth folding into
the §6 ADR once §2 numbers land.

**Implied total post-endpoint latency on 5 s (pre-cache):**
362 ms ASR + 150 ms VAD ≈ **512 ms** — under the 700 ms acceptance
target before any optimization. §2 should still cut
**first-dictation-after-launch** cold-start, which is what the user
actually feels on app open; warm steady-state may not budge much.

## §6 Phase-0 polish-backend bench: Qwen 3.5 2B Q4_K_M (2026-05-16, M5 Pro 24 GB)

Driven by `src/bin/bench_llm.rs`. 100 polish iterations of a fixed
240-char noisy transcript through `llama-cpp-2` (Metal feature)
loading `unsloth/Qwen3.5-2B-Q4_K_M.gguf`. Output: 55 tokens cleaned.

| Metric | Mean | p50 | p95 | p99 |
|--------|------|-----|-----|-----|
| TTFT (ms) | 2.0 | 2.0 | 2.0 | 2.0 |
| Generation (ms) | 548 | 548 | 558 | 567 |
| Total per polish (ms) | 551 | **550** | 560 | 570 |
| Decode (tokens/sec) | 100.3 | 100.4 | 101.7 | 101.9 |

Cold model load: 229 ms. p99/p50 = 1.04 (variance negligible).

Replay:

```bash
./target/release/bench_llm \
    --model ~/Library/Application\ Support/com.parakeet.rs/llm/qwen3.5-2b-q4_k_m/Qwen3.5-2B-Q4_K_M.gguf \
    --reps 100 --warmup-reps 3 2> bench/llm-raw.log
# then aggregate inline — see ADR-0018 for the one-shot Python snippet
```

Background and library-selection rationale: [ADR-0018](../docs/ADR.md#0018--polish-backend-llamacpp--qwen-35-2b-q4_k_m).

## §6 follow-up: Qwen 3.5 4B Q6_K (2026-06-11, M5 Pro 24 GB) — shipped

The 2B's instruction-following misses (paraphrasing, over-deleted
"like", fumbled `scratch that`) motivated a bump to **Qwen3.5-4B at
Q6_K** (3.53 GB) — same family, so the ChatML + `/no_think` template
carries over unchanged. See the ADR-0018 amendment.

Fetch one-liner:

```bash
mkdir -p ~/Library/Application\ Support/com.parakeet.rs/llm/qwen3.5-4b-q6_k && \
curl -L -o ~/Library/Application\ Support/com.parakeet.rs/llm/qwen3.5-4b-q6_k/Qwen3.5-4B-Q6_K.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q6_K.gguf
```

30 reps, same 240-char sample transcript, same `bench_llm` harness:

| Metric | Mean | p50 | p95 |
|--------|------|-----|-----|
| TTFT (ms) | 40 | 29 | 33 |
| Generation (ms) | 1200 | 1197 | 1231 |
| Total per polish (ms) | 1240 | **1225** | 1262 |
| Decode (tokens/sec) | 43.3 | 43.4 | 43.8 |

vs the 2B: total p50 550 ms → 1225 ms (2.2×), decode 100 → 43 tok/s.
Streaming paste (ADR-0019) absorbs the difference — perceived latency
is time-to-first-words (TTFT 29 ms + first chunks), not last-token.
No truncations at the 768-token output cap across the run.

Replay:

```bash
./target/release/bench_llm \
    --model ~/Library/Application\ Support/com.parakeet.rs/llm/qwen3.5-4b-q6_k/Qwen3.5-4B-Q6_K.gguf \
    --reps 30 --warmup-reps 3 2> bench/llm-4b-raw.log
```

## Files

| Path                         | Purpose                                          |
|------------------------------|--------------------------------------------------|
| `audio/{1,3,5,10,20}s_*.wav` | Generated fixtures (gitignored). Filename includes sample rate (e.g. `5s_16000.wav`). |
| `raw.log`                    | All `phase_timer` lines from the last ASR run.   |
| `llm-raw.log`                | All `llm_timer` lines from the last LLM run.     |
| `baseline.csv`               | Aggregated ASR baseline (pre-CoreML-cache).      |
| `post-coreml-cache.csv`      | Aggregated post-§2 (deferred — see ADR-0017).    |
| `polish-backends.csv`       | §6 Phase-0 polish backend numbers (this run).   |
