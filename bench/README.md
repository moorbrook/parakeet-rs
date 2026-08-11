# Latency bench

`scripts/bench-latency.sh` drives `bench_asr` over generated TTS WAVs at
{1, 3, 5, 10, 20}s, 30 reps each, and emits `phase_timer` log lines that
`scripts/bench-aggregate.py` reduces into `baseline.csv` (or `$OUT_CSV`). It
also writes the matching `*-boundary.csv` through the PEP 723
`scripts/bench-boundary.py` aggregator.

See `docs/latency-plan.md` §1 for design and acceptance criteria.

## Gold-reference quality gate

The checked-in real-speech corpus gates model, quantization, backend, and
hotword changes against human-authored transcripts:

```bash
REPETITIONS=10 scripts/bench-gold.sh
```

`bench/gold/manifest.json` contains independent absolute and baseline-regression
WER/CER limits. `asr_diff` applies them to the worst repeated result, prints
per-fixture and per-category summaries, exits non-zero when either threshold
fails, and writes schema-v3 machine reports with:

- normalized lexical WER/CER, exact formatting matches, and separate insertion,
  deletion, and substitution counts;
- categories such as names, commands, numbers, punctuation, and vocabulary;
- repeat-run transcript/WER/CER spread, p50/p95 decode time, and real-time factor;
- backend/model/quantization/provider labels;
- decoder method plus requested/active vocabulary state, score, term count,
  and source/generated-hotword SHA-256 identities;
- app version, macOS version, chip, memory, CPU count, model-load time, and
  warmup, first-result, and full-process-tree peak-resident memory.

Lexical normalization preserves Unicode letters/numbers and accents,
lowercases them, removes apostrophes without splitting words, and treats other
punctuation as a separator. Raw reference and hypothesis strings remain in the
report so capitalization and punctuation changes are still auditable.

The representative corpus contains seven real human recordings from pinned,
licensed LibriSpeech and SLURP revisions, including proper nouns, commands,
numbers, distant/noisy audio, and custom-vocabulary cases. Sources, hashes, and
reproduction instructions are under `bench/gold/`. `gold.example.json` remains
only as the older macOS `say` format/smoke example.

Measured M5 Pro baselines, fairness rules, threshold derivation, known errors,
and rejected variants live in `docs/asr/{PERF,DISCREPANCIES,NEGATIVE_EVIDENCE}.md`.

The developer-only Qwen3-ASR challenger uses pinned PEP 723/`uv` oracle scripts
and does not affect the shipping dependency graph. Its immutable artifact
summary and no-go evidence are under `bench/qwen3-asr/`; the full interpretation
is in `docs/asr/QWEN3_ASR_EVALUATION.md`.

The measured vocabulary-score sweep and no-training decision are under
`bench/domain-adaptation/`; interpretation and future adapter/data gates are in
`docs/asr/DOMAIN_ADAPTATION.md`.

## Quick start

```bash
# First time only: launch Parakeet.app once so the model bundle downloads
# into ~/Library/Application Support/com.parakeet.rs/models/.
open target/release/bundle/osx/Parakeet.app   # or however you launch it

# Then:
scripts/bench-latency.sh                           # → bench/baseline.csv
OUT_CSV=bench/post-coreml-cache.csv \
    scripts/bench-latency.sh                       # § 2 re-bench

# Native int8 Parakeet Unified challenger (builds the pinned Swift worker):
BACKEND=coreml-unified OUT_CSV=bench/coreml-unified.csv \
    scripts/bench-latency.sh

# Production capture + VAD + ASR, frozen serial baseline vs optimized path.
# Requires a duplex Core Audio loopback named "BlackHole 2ch".
scripts/bench-end-to-end.sh
```

## What is and isn't measured

The bench loads pre-recorded WAVs and runs `Asr::recognize()` directly.
Each repetition also emits `asr_boundary` with worker-internal
resample-plus-inference time, outer Rust wall time, and their difference. For
the Core ML worker that difference prices Float32 pipe transfer, scheduling,
framing, response JSON, and Rust-side handling instead of attributing the full
short-utterance floor to IPC.
It **does not** exercise:

- `cpal` mic-capture callback latency
- the Silero VAD endpoint policy (750 ms for Tap; 150 ms for Tap Fast)
- the `CGEventKeyboardSetUnicodeString` keystroke insertion step
  (sub-ms per chord — see ADR-0019)

So `scripts/bench-latency.sh` is **ASR-only**. Use
`scripts/bench-end-to-end.sh` for the production capture, resampling, dual-VAD,
endpoint, session-shutdown, and ASR path. That harness stops at
transcript-ready rather than typing into the user's focused app; the only
excluded production step is the synchronous synthetic-Unicode event post
(sub-ms per chord; see ADR-0019).

## End-to-end 3x result: M5 Pro 24 GB (2026-08-10)

The end-to-end harness selects `BlackHole 2ch` directly for both input and
output without changing macOS defaults. It trims only sub-threshold trailing
fixture silence, uses Core Audio's predicted playback timestamp for the last
non-silent sample, and feeds the production `cpal` capture and `streamer` path.
Both variants receive the same measurement-only acoustic endpoint marker.

The representative fixture is `5s_48000.wav` (4.854 s measured). Release
builds used two warmups and 30 measured repetitions per variant. Every
measured transcript had an exact lexical match to the reviewed reference.

| pipeline | n | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| sherpa + serial endpoint | 30 | 613.6 ms | 613.0 ms | 652.5 ms | 657.3 ms |
| Core ML + speculative decode | 30 | 189.7 ms | 182.0 ms | 203.0 ms | 203.0 ms |
| speedup | | **3.23×** | **3.37×** | **3.21×** | **3.24×** |

The optimized path starts ASR from the early 32 ms detector, discards the
provisional result whenever speech resumes, and lets an independent Silero
state remain the sole stop authority. This frozen comparison explicitly uses
Tap Fast's original 150 ms policy so the historical 3× result stays
like-for-like. The gate fails unless both p50 and p95 are at least 3.0× and
every transcript matches:

```bash
REPS=30 WARMUP_REPS=2 scripts/bench-end-to-end.sh
```

## Long-pause endpoint gate

Normal Tap now uses a 750 ms confirmation policy; Tap Fast retains 150 ms for
short commands. The separate endpoint gate replays versioned human LibriSpeech
audio through production capture, VAD, speculative Core ML inference, and
session shutdown. Its 14.225 s fixture includes a reviewed 544 ms natural
pause that the former policy cut. A pass requires zero early stops and p95
final-pause latency below one second for both the single- and multi-sentence
fixtures:

```bash
REPS=30 WARMUP_REPS=2 scripts/bench-endpoint-policy.sh
```

M5 Pro 24 GB release results (2026-08-11):

| fixture | repetitions | false stops | p50 | p95 |
|---|---:|---:|---:|---:|
| 3.505 s single sentence | 30 | **0** | 668.0 ms | 668.0 ms |
| 14.225 s multi sentence | 30 | **0** | 637.0 ms | 658.1 ms |

The unchanged Tap Fast comparison was also re-run for 30 repetitions after
this policy split. It retained **3.24× p50 / 3.18× p95** speedups (589.5 →
182.0 ms p50; 644.8 → 203.0 ms p95), so the representative no-polish gate
remains above its accepted 3× target.

The fixture manifest, source revision, hashes, references, and license are in
[`bench/endpointing/`](endpointing/). This gate isolates endpoint behavior;
transcript WER/CER remains the responsibility of `asr_diff`.

## Native Core ML result: M5 Pro 24 GB (2026-08-10)

Matched release runs used the same 48 kHz WAVs, resident recognizer, three
warmups, 30 measured repetitions, outer `Asr::recognize()` timer, and CSV
aggregator. `sherpa` is the frozen previous backend; `coreml-unified` is the
resident FluidAudio worker with the int8 offline encoder on CPU+ANE.

| bucket | sherpa p50 | unified p50 | p50 speedup | sherpa p95 | unified p95 | p95 speedup |
|--------|-----------:|------------:|------------:|-----------:|------------:|------------:|
| 1 s    | 112.0 ms   | 35.0 ms     | **3.20×**   | 116.5 ms   | 36.5 ms     | **3.19×**   |
| 3 s    | 226.0 ms   | 50.0 ms     | **4.52×**   | 239.6 ms   | 51.0 ms     | **4.70×**   |
| 5 s    | 361.5 ms   | 66.0 ms     | **5.48×**   | 384.3 ms   | 67.0 ms     | **5.74×**   |
| 10 s   | 580.0 ms   | 90.0 ms     | **6.44×**   | 597.8 ms   | 91.5 ms     | **6.53×**   |
| 20 s   | 1195.0 ms  | 188.0 ms    | **6.36×**   | 1245.3 ms  | 192.6 ms    | **6.47×**   |

The companion gold run passed at **2.38% WER / 2.22% CER** against limits of
4% / 3%, with 40% exact formatting and 74.4× aggregate model-reported RTFx.
This is a five-item macOS `say` smoke corpus, not a claim about real-user WER;
the representative-speech gate described above still applies before a release.

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
| `gold/manifest.json`       | Shipping real-speech quality manifest and thresholds. |
| `gold/sources.json`        | Immutable source provenance, licenses, and hashes. |
| `gold.example.json`        | Historical `say` smoke-example schema. |
| `asr-quality.json`          | Generated machine-readable quality/latency report (gitignored). |
| `llm-raw.log`                | All `llm_timer` lines from the last LLM run.     |
| `baseline.csv`               | Aggregated ASR baseline (pre-CoreML-cache).      |
| `post-coreml-cache.csv`      | Aggregated post-§2 (deferred — see ADR-0017).    |
| `polish-backends.csv`       | §6 Phase-0 polish backend numbers (this run).   |
