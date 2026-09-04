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
# Shipping native int8 Parakeet Unified backend (builds the pinned worker):
BACKEND=coreml-unified OUT_CSV=bench/coreml-unified.csv \
    scripts/bench-latency.sh

# Production capture + VAD + ASR, frozen serial baseline vs optimized path.
# Requires a duplex Core Audio loopback named "BlackHole 2ch".
scripts/bench-end-to-end.sh

# Hold mode: release-to-transcript through the same production path.
# Also requires the "BlackHole 2ch" loopback.
scripts/bench-hold.sh
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
endpoint, session-shutdown, and ASR path, and `scripts/bench-hold.sh` for the
Hold path, where the hotkey release replaces the VAD endpoint. Both harnesses
stop at transcript-ready rather than typing into the user's focused app; the
only excluded production step is the synchronous synthetic-Unicode event post
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

## Per-stage breakdown: M5 Pro 24 GB (2026-09-04)

`bench_asr --stage-timings` starts the worker with `--emit-stage-timings`, and
the worker then reports where each decode went. `scripts/bench-latency.sh`
passes the flag automatically for `BACKEND=coreml-unified` and reduces the
`asr_stages` log lines into `*-stages.csv` through `scripts/bench-stages.py`.

The decode pipeline lives in FluidAudio's `UnifiedAsrManager` and
`UnifiedRnntDecoder`, which are a pinned dependency this project depends on
rather than vendors. The worker therefore measures from outside: at startup it
replaces the prediction implementations of `MLModel` and its registered
subclasses with timing wrappers that call straight through, and attributes each
dispatch to a stage by the input feature names FluidAudio's providers declare
(`mel` for the encoder, `targets` for the decoder, `encoder_step` for the
joint). Stage boundaries come from the resulting dispatch timeline: mel is the
gap before an encoder dispatch, the RNNT loop is everything from an encoder
dispatch to the last dispatch of that window. Nothing in the pinned package is
patched, and the shipping dictation path never installs the wrappers.

Medians over 30 measured repetitions per bucket, three warmups, 48 kHz
fixtures, release build (`bench/coreml-unified-stages.csv`):

| fixture | resample | mel | encoder | RNNT loop | post | worker total | IPC | ASR p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.740 s | 3.41 ms | 3.07 ms | **25.55 ms** | 2.98 ms | 0.04 ms | 35.2 ms | 0.16 ms | 36.0 ms |
| 2.507 s | 11.46 ms | 3.08 ms | **25.48 ms** | 8.40 ms | 0.05 ms | 48.5 ms | 0.26 ms | 49.0 ms |
| 4.967 s | 22.71 ms | 3.06 ms | **25.49 ms** | 15.21 ms | 0.06 ms | 66.5 ms | 0.42 ms | 67.0 ms |
| 8.150 s | 37.77 ms | 3.32 ms | **26.03 ms** | 25.40 ms | 0.08 ms | 92.6 ms | 0.60 ms | 94.0 ms |
| 15.691 s | 72.57 ms | 6.13 ms | **51.59 ms** | 54.40 ms | 0.13 ms | 184.9 ms | 1.11 ms | 187.0 ms |

Worker total is resample plus the profiled transcribe interval; it sits 0.7 to
1.0 ms under the `asr_boundary` internal time, which is the response encode and
the Swift work outside the profiled window. IPC is unchanged from the
2026-08-11 boundary measurement and remains under 0.6% of the call.

Enabling the profiler costs nothing measurable. Matched 30-repetition runs with
and without `--stage-timings` measured 35.51 against 35.51 ms at 1 s and 67.09
against 66.92 ms at 5 s. The 10 s bucket, which has the most dispatches of any
single-window fixture, was measured with eight interleaved on/off blocks of 15
repetitions so that machine drift hits both arms equally: the per-block delta
has a median of -0.17 ms and a mean of -0.38 ms, and its sign flips across
blocks, so the effect is below the noise. The interception is two clock reads
and a lock per dispatch; the class sweep that finds the entry points stops once
all three stages have dispatched, and the model placement is read once per stage
rather than per call.

These absolute numbers need a quiet machine, which matters more than the
profiler does. Against the 2026-08-11 published baseline the 1, 3, 5, and 20 s
buckets land within about 2 ms, and 10 s is roughly 5 ms higher; the interleaved
A/B above rules the profiler out as the cause. A separate repeat under a
competing job reproduced the dispatch counts exactly and kept the encoder flat
at 26.4 to 28.5 ms, with resample and the decode loop 10 to 20% higher. The
shape of the breakdown is stable; the millisecond values are not, so compare
against the ASR-only p50 column from the same run.

The encoder cost does not depend on utterance length. `UnifiedAsrManager`
zero-pads every window to a fixed 15 s buffer (240,000 samples, 1,501 mel
frames) and runs the full offline encoder graph on it, so a 0.74 s utterance
pays the same 25.5 ms as an 8.15 s one. The 15.691 s fixture exceeds the 15 s
window and needs a second one, which doubles both mel and encoder.
`chunkStarts` adds that second window only past 240,000 samples, so a 14 s
utterance still runs a single encoder pass.

### Core ML dispatch counts per utterance

| fixture | windows | encoder calls | decoder calls | joint calls | decoded frames | per joint | per decoder |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.740 s | 1 | 1 | 7 | 16 | 10 | 107 µs | 158 µs |
| 2.507 s | 1 | 1 | 20 | 51 | 32 | 100 µs | 143 µs |
| 4.967 s | 1 | 1 | 35 | 96 | 62 | 99 µs | 142 µs |
| 8.150 s | 1 | 1 | 57 | 158 | 102 | 101 µs | 147 µs |
| 15.691 s | 2 | 2 | 122 | 342 | 222 | 101 µs | 145 µs |

Decoded frames are `joint_calls - (decoder_calls - windows)`, since the greedy
loop issues one joint per frame plus one per emitted token, and one decoder call
per window plus one per emitted token. Every bucket matches its 80 ms frame
arithmetic: 4.967 s of audio decodes 62 frames. That agreement is what
validates the counts.

The agreement is also checked at runtime, because a stage whose Core ML entry
point stops being intercepted reports zero cost while every other number stays
plausible. `bench_asr` fails the run when a report has no encoder dispatches,
when `windows` and `encoder_calls` disagree, or when any prediction goes
unattributed, and `scripts/bench-stages.py` repeats those checks before writing
a CSV. The encoder is the stage worth naming here: losing it also drives
`windows` to zero, which leaves the frame identity intact and the row
believable.

Per-frame loop cost is flat at roughly 0.25 ms across all lengths. The dispatch
floor of about 100 µs per joint call is the dominant term inside the loop:
`decode_loop_dispatch_ms` accounts for 95% of `decode_loop_ms` at every bucket,
so the Swift-side loop work (encoder-step extraction, `MLMultiArray`
allocation) is not where the loop time goes.

### Which compute unit runs each stage

The encoder runs on `cpuAndNeuralEngine`; the decoder and the joint-decision
model both run `cpuOnly`. The `compute_units` column reports
`MLModelConfiguration.computeUnits` read off each live model object at dispatch
time, so it records the placement Core ML actually used rather than the one the
worker asked for. It matches the pinned FluidAudio
source, where `UnifiedAsrManager.loadModels` builds a separate `cpuOnly`
configuration for the decoder and joint and comments that only the encoder uses
ANE or GPU.

The ANE is genuinely carrying the encoder. Eight repetitions on the 4.967 s
fixture at each placement:

| encoder placement | encoder median |
|---|---:|
| `cpu-and-neural-engine` | 26.05 ms |
| `cpu-only` | 86.01 ms |

`cpu-and-gpu` produces no result at all. The worker dies during warmup with
`MPSGraphExecutable.mm:5070: failed assertion 'Error: MLIR pass manager failed'`,
the int8-on-MPSGraph failure FluidAudio's own loader comments on and coerces
away from for `.all`.

```bash
BACKEND=coreml-unified OUT_CSV=bench/coreml-unified.csv scripts/bench-latency.sh
./target/release/bench_asr --backend coreml-unified --wav bench/audio/5s_48000.wav \
    --reps 8 --warmup-reps 3 --stage-timings --compute-units cpu-only
```

### Where the 66 ms at 5 s goes

No single stage owns it. On the 4.967 s fixture the 66.5 ms of worker-internal
time is encoder 25.5 ms (38%), resample 22.7 ms (34%), RNNT decode loop 15.2 ms
(23%), and mel 3.1 ms (5%), with IPC at 0.42 ms and post-processing under
0.1 ms. The 35 ms floor at 1 s is owned by the encoder: the offline path pads
every utterance to the fixed 15 s window, so 25.5 ms of encoder plus 3.1 ms of
mel is length-independent work that a one-word utterance pays in full, and that
28.6 ms is 80% of the 1 s result. The growth from 35 ms to 66 ms is split nearly
evenly between resample, which costs a linear 4.6 ms per second of 48 kHz input
and adds 19.3 ms, and the decode loop, which costs 0.25 ms per 80 ms frame and
adds 12.2 ms. The prior hypothesis that the per-frame RNNT loop accounts for the
gap over encoder arithmetic is half right: the loop is real, it is entirely on
the CPU, and roughly 100 µs of each 0.25 ms frame is a single joint dispatch,
but at every measured length the 48 kHz to 16 kHz resample costs more than the
loop does.

## Hold-mode baseline: M5 Pro 24 GB (2026-09-04)

Hold (press-and-hold) had no measured release-to-text number; the tables above
and the end-to-end gate both cover Tap, where Silero VAD owns the endpoint. In
Hold the hotkey release is the endpoint, so `scripts/bench-hold.sh` runs
`bench_e2e --mode hold`: it plays each fixture through the loopback device,
waits for Core Audio's predicted instant of the last audible sample, releases
there, and stops the clock at transcript-ready. Releasing at the acoustic end
is the earliest a user could, so these are floor numbers for the mode.

30 measured repetitions per bucket, two warmups, `BlackHole 2ch` loopback,
resident Core ML worker (`bench/hold.csv`):

| bucket | captured audio | n | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 1 s | 0.800 s | 30 | 58.0 ms | **54.0 ms** | 79.5 ms | 80.7 ms |
| 3 s | 2.571 s | 30 | 71.5 ms | **67.0 ms** | 97.7 ms | 108.5 ms |
| 5 s | 5.035 s | 30 | 117.9 ms | **106.5 ms** | 158.6 ms | 160.4 ms |
| 10 s | 8.213 s | 30 | 144.1 ms | **137.5 ms** | 184.5 ms | 190.4 ms |
| 20 s | 15.755 s | 30 | 238.5 ms | **231.5 ms** | 280.6 ms | 288.8 ms |

Bucket labels are nominal. The captured-audio column is the median measured
duration, and it is what these latencies belong to: the "20 s" row is a 15.755 s
utterance, which is only just over the 15 s encoder window, and the "10 s" row is
8.2 s. Budget against the captured column, not the label.

Medians of the parts, from the same `phase_timer` lines:

| bucket | captured audio | release to observed | capture stop and join | ASR | total p50 |
|---|---:|---:|---:|---:|---:|
| 1 s | 0.800 s | 8.0 ms | 0.0 ms | 44.5 ms | 54.0 ms |
| 3 s | 2.571 s | 8.0 ms | 0.0 ms | 53.5 ms | 67.0 ms |
| 5 s | 5.035 s | 9.5 ms | 1.0 ms | 95.0 ms | 106.5 ms |
| 10 s | 8.213 s | 9.5 ms | 1.0 ms | 122.0 ms | 137.5 ms |
| 20 s | 15.755 s | 12.0 ms | 1.0 ms | 219.5 ms | 231.5 ms |

`run_manual` polls its signal channel every 15 ms, which is the 8 to 12 ms
median seen in the first column and up to 15 ms in the tail. Capture shutdown
and the mono fold cost about 1 ms. Everything else is ASR, which runs 9 to 42%
slower here than in the isolated bench, not monotonically in length, because the
capture stream is still live in the same process. Hold also never sets `early_transcript`, so unlike Tap it
cannot overlap any decode with the tail of the utterance.

```bash
REPS=30 WARMUP_REPS=2 BACKEND=coreml-unified scripts/bench-hold.sh
```

## Baseline: M5 Pro 24 GB (2026-05-16, pre-§2 CoreML cache)

| length | n  | mean ms | p50 ms | p95 ms | p99 ms |
|--------|----|---------|--------|--------|--------|
| 1 s    | 30 | 121     | 121    | 136    | 146    |
| 3 s    | 30 | 229     | 227    | 237    | 263    |
| 5 s    | 30 | 364     | **362**| 376    | 405    |
| 10 s   | 30 | 573     | 572    | 589    | 591    |
| 20 s   | 30 | 1120    | 1114   | 1162   | 1185   |

**Steady-state RTFx** ≈ 13–14× real time on the 5 s bucket. This historical
sherpa baseline is retained for comparison; the shipping native Core ML and
end-to-end measurements above supersede it as the current performance record.

**Implied total post-endpoint latency on 5 s (pre-cache):**
362 ms ASR + 150 ms VAD ≈ **512 ms** — under the former 700 ms acceptance
target before the native worker and speculative-decode work landed.

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

Enable Polish once in the app to download and SHA-256 verify the pinned GGUF;
the replay command below uses that standard application path.

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

## §6 follow-up: polish latency and quality on an eval set (2026-09-04, M5 Pro 24 GB)

Kata 0tpp. Everything above measures **one** transcript — the hardcoded
`SAMPLE_INPUT`, 55 output tokens at the model's decode rate. That is the
structural bound, not a p50 over anything a user dictates, and it is
what the "1000 ms NOT MET" verdict was recorded against.

`bench/polish/eval.json` is a 26-item eval set with the polished text
each item should produce. `bench_llm --eval` runs it, scores word-level
error rate against `expected` (casing and punctuation included — those
are two of the three things polish exists to fix), and reports latency
and quality over the same population.

Machine load before each run: `Load Avg` 1.0–2.3, CPU ≥ 87 % idle.
3 repetitions × 26 items = 78 measured decodes per row. Repetitions are
low deliberately: within-item variance is negligible (§6's p99/p50 =
1.04), so the percentiles are driven by *which item* it is, not by
run-to-run noise. `p99` over 78 samples is a single item and is not
quoted below.

| Variant | p50 | p95 | `legacy-bench-sample` | mean WER | exact | mean out tokens |
|---|---|---|---|---|---|---|
| **4B Q6_K full-text (shipping)** | **444 ms** | 1219 ms | 1209 ms | **0.139** | 17/26 | 18.7 |
| 4B full-text + skip < 4 words | 446 ms | 1222 ms | 1221 ms | 0.139 | 17/26 | 18.3 |
| 4B full-text + context reuse | 424 ms | 1220 ms | 1198 ms | 0.139 | 17/26 | 18.7 |
| 4B edits-only | 600 ms | 2438 ms | 2421 ms | 0.321 | 9/26 | 26.3 |
| 2B Q6_K full-text | 230 ms | 572 ms | 571 ms | 0.185 | 13/26 | 20.5 |
| 0.8B Q6_K full-text | 141 ms | 338 ms | 330 ms | 0.249 | 11/26 | 21.6 |

**The shipping configuration meets the <1000 ms p50 target on the eval
set.** The `legacy-bench-sample` column shows why the two verdicts
differ: that one item costs 1209 ms in the same run whose p50 is 444 ms.

### Composition caveat — read this before quoting the p50

The eval set's composition was a judgement call, and composition
determines the blended p50. Nine of 26 items are zero-change by
construction. Per-category numbers are published so the blended figure
can be re-weighted against a different view of what real dictation looks
like:

| Category | n | 4B p50 | 4B max | 4B WER | 2B p50 | 2B WER |
|---|---|---|---|---|---|---|
| clean | 6 | 429 ms | 514 ms | 0.000 | 197 ms | 0.000 |
| short | 3 | 299 ms | 326 ms | 0.000 | 147 ms | 0.000 |
| filler-light | 6 | 446 ms | 492 ms | 0.042 | 212 ms | 0.173 |
| command | 4 | 439 ms | 503 ms | 0.000 | 251 ms | 0.211 |
| technical | 3 | 609 ms | 652 ms | 0.847 | 298 ms | 0.681 |
| filler-heavy | 3 | 857 ms | 1219 ms | 0.257 | 391 ms | 0.258 |
| long | 1 | 2812 ms | 2812 ms | 0.060 | 1400 ms | 0.103 |

Latency tracks output length almost exactly; quality does not. The
`technical` category is the worst on both models — spoken version
numbers and identifiers ("one point zero point two one nine",
`PolishMode colon colon On`) are where polish does real damage, and that
is a quality bug worth its own issue, not a latency one.

### Findings per avenue

- **Edits-only is worse on both axes.** The model answers with the
  corrected transcript instead of an edit list in 17 of 26 items, on the
  4B, with an explicit prohibition and a worked example in the prompt.
  The arithmetic does not favour it even when it works: full-text
  averages 18.7 output tokens, an `OLD ==> NEW` line costs 5–8 tokens
  per fix because `OLD` must carry enough context to be unique, and the
  variant measured 26.3. A GBNF grammar (`LlamaSampler::grammar`, and
  the `sampler` feature is already on) would force the `==>` structure
  but cannot stop `<whole input> ==> <corrected>`, which parses cleanly
  and doubles the tokens.
- **Prompt caching is unavailable on this model family.** Qwen 3.5 is a
  hybrid Gated-DeltaNet architecture (ADR-0018); its recurrent state
  carries no per-token position, so `llama_memory_seq_rm` refuses
  partial sequence removal. `mean_reused_prompt_tokens=0.0` on every
  run. The same refused operation is what draft-model speculative
  decoding would need for rollback, so that avenue is blocked too. What
  the `--prompt-cache` row actually measures is **context reuse** — not
  re-allocating a `LlamaContext` per call — worth 20 ms (444 → 424),
  consistent with the 29 ms TTFT bounding it.
- **Skipping short utterances helps the mean, not the p50.** It fires on
  3/26 items with identical quality, and moves the mean 591 → 556 ms.
  The p50 is unchanged (444 → 446) because the items it removes were
  already the fastest ones; taking three items off the bottom does not
  move the middle.
- **Neither smaller model is a free swap.** The 2B cuts p95 from 1219 to
  572 ms and the structural-bound item from 1209 to 571 ms, at WER 0.185
  vs 0.139 and 13/26 vs 17/26 exact. It is a tail fix bought with
  quality. The 0.8B is faster still and clearly worse.

Fixed-sample replays for cross-checking against §6 above (15 reps):
4B Q6_K **p50 1202 ms** (§6 recorded 1225), 2B Q6_K p50 565 ms.

Replay:

```bash
LLM="$HOME/Library/Application Support/com.parakeet.rs/llm"
./target/release/bench_llm \
    --model "$LLM/qwen3.5-4b-q6_k/Qwen3.5-4B-Q6_K.gguf" \
    --eval bench/polish/eval.json --variant full-text \
    --reps 3 --tag qwen3.5-4b-q6_k --csv bench/polish/variants.csv \
    2> bench/llm-4b-fulltext-raw.log
```

Swap `--variant edits-only`, `--skip-min-words 4`, `--prompt-cache`, or
`--model` for the other rows.

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
| `coreml-unified.csv`         | Generated shipping-backend ASR percentiles.     |
| `*-boundary.csv`             | Generated Rust/worker boundary measurements.    |
| `*-stages.csv`               | Generated per-stage breakdown and Core ML dispatch counts. |
| `hold.{log,csv}`             | Generated Hold-mode release-to-transcript runs. |
| `e2e-*.{log,csv}`            | Generated serial/speculative production-path runs. |
| `endpoint-*.{log,csv}`       | Generated pause-friendly endpoint gate runs.   |
| `polish-backends.csv`        | Historical §6 Phase-0 2B polish measurements.  |
| `polish/eval.json`           | Polish quality eval set: 26 transcripts with expected output. |
| `polish/variants.csv`        | Generated per-variant polish latency/quality summary. |
| `llm-*-raw.log`              | Generated `llm_timer` / `llm_eval_*` lines per polish variant run. |
