# ASR quality and performance ledger

## Benchmark contract

`scripts/bench-gold.sh` evaluates the same checked-in 34.05 seconds of real
human speech with each backend. Runs use release builds, the shipping model
precision and thread policy, the default allocator, one page-touch/dummy-decode
warmup, and ten measured corpus repetitions. Rows where a backend loses remain
in this ledger. A vocabulary-assisted row is separate because modified beam
search is not execution-equivalent to greedy decoding.

The machine report records model identity, quantization/provider, application
and report schema versions, OS, architecture, chip, memory, logical CPUs, model
load, warmup, first-result latency, corpus decode p50/p95, RTFx, and the observed
peak resident set of the Rust process plus any resident worker.

## M5 Pro baseline — 2026-08-11

Hardware: Apple M5 Pro, 24 GiB, 15 logical CPUs; macOS 26.5.1, arm64. Corpus:
92 reference words / 476 reference characters across seven fixtures.
The optimized row uses FluidAudio commit
`00a9aa771900ea09c485659663be31019e293e47` and Core ML model revision
`4252711f6f060f9a2f91e5f081a806d7f45eebd8`; the full artifact manifest is in
[`COREML_MODEL.md`](COREML_MODEL.md).

| backend | decoding | WER | CER | load | first result | decode p50 | decode p95 | p50 RTFx | peak RSS | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FluidAudio Core ML | greedy, int8 CPU+ANE | **5.43%** | **3.57%** | **0.168 s** | **0.321 s** | **0.452 s** | **0.470 s** | **75.4×** | **0.10 GiB** | pass |
| sherpa-onnx | greedy, int8, Core ML requested | 10.87% | 5.46% | 3.647 s | 4.582 s | 2.764 s | 2.949 s | 12.3× | 4.25 GiB | fail |
| sherpa-onnx | vocabulary, modified beam, score 2 | 10.87% | 5.46% | 3.717 s | 4.716 s | 3.126 s | 3.148 s | 10.9× | 4.26 GiB | fail |

Core ML is 6.12× faster at corpus-decode p50 and 6.28× at p95 than the unbiased
sherpa baseline. It reduces model load by 21.8×, first result by 14.3×, and
observed peak RSS by about 41.5×. The vocabulary row did not change any gold
transcript and was 13.1% slower than sherpa greedy at p50.

All three configurations produced one unique transcript per fixture across ten
runs: WER spread 0.00 points, CER spread 0.00 points, and zero changed outputs.
That measured floor sets both regression tolerances to 0.00 points. The checked
baseline is 5/92 word edits (5.4347826%) and 17/476 character edits (3.5714286%);
any increase fails even while the independent absolute product limits remain
8% WER and 5% CER. Those absolute limits require at least 92% word accuracy and
95% character accuracy on this corpus and permit at most seven word edits and
23 character edits.

## Domain-adaptation decision — 2026-08-11

A 27-point `hotword_score` exploration was narrowed to six authoritative
ten-repeat rows. The first transcript effect appears at score 2.75: overall WER
falls from 10.87% to 8.70% and custom-vocabulary WER from 54.55% to 27.27%, but
CER rises from 5.46% to 6.09%, noisy WER doubles from 9.09% to 18.18%, numbers
WER rises from 13.33% to 16.67%, and corpus p50 is 14.2% slower than greedy.
At score 4.5 the decoder begins producing `Olly` while injecting `IBM` into an
unrelated noisy command; score 6 reaches 42.39% WER.

No score clears the frozen shipping Core ML overall or per-category baseline,
and all six repeated rows are deterministic. Training, distillation, QAT, and
a global score change are therefore rejected on current evidence. The generic
Core ML model remains default. See [`DOMAIN_ADAPTATION.md`](DOMAIN_ADAPTATION.md)
for every category delta, data-separation rules, adapter ownership, and reopen
gates; machine-verifiable evidence is under `bench/domain-adaptation/`.

Replay:

```bash
REPETITIONS=10 scripts/bench-gold.sh
```

## Core ML worker boundary — 2026-08-11

The worker reports its own resample-plus-inference time. `bench_asr` measures
the same call from the Rust side and records the non-negative difference as
`boundary_ms`, covering Float32 pipe transfer, framing, process wakeup,
response JSON, and Rust-side handling. Release build, three warmups and 30
measured repetitions per generated 48 kHz bucket:

| bucket | internal p50 | outer p50 | boundary p50 | boundary p95 |
|---|---:|---:|---:|---:|
| 1 s | 35.112 ms | 35.242 ms | **0.128 ms** | 0.143 ms |
| 3 s | 49.965 ms | 50.231 ms | **0.260 ms** | 0.290 ms |
| 5 s | 65.664 ms | 66.044 ms | **0.380 ms** | 0.397 ms |
| 10 s | 88.856 ms | 89.480 ms | **0.581 ms** | 0.618 ms |
| 20 s | 185.029 ms | 186.142 ms | **1.111 ms** | 1.209 ms |

The earlier inference that roughly 30 ms of the one-second result was IPC was
wrong: 99.6% of that outer p50 is worker-internal. Boundary time is 0.36% of
the one-second call and 0.60% of the 20-second call. Shared memory, Mach ports,
or moving the Core ML runtime in-process cannot materially improve current
latency unless a future profile shows this balance has changed.

Replay:

```bash
PARAKEET_COREML_MODEL_DIR="$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-unified-en-0.6b" \
  BACKEND=coreml-unified REPS=30 WARMUP_REPS=3 \
  OUT_CSV=bench/coreml-unified.csv scripts/bench-latency.sh
```

## Per-stage attribution and Hold baseline — 2026-09-04

`bench_asr --stage-timings` makes the worker report where a decode went, by
wrapping `MLModel`'s prediction implementations at runtime and attributing each
dispatch by its input feature names. The profiler patches nothing: it wraps the
prediction entry points at runtime. Medians of 30 repetitions on the 4.967 s
fixture: encoder 25.5 ms, resample 22.7 ms, RNNT decode loop 15.2 ms, mel
3.1 ms, IPC 0.42 ms.

Two of those stages have since moved and this section is the record of the
measurement, not of current cost. ADR-0030 retired the resample, and short-window
encoder buckets cut encoder and mel on utterances under 8 s; the two sections
below carry the current numbers.

Two structural facts came out of it. The offline path zero-pads every utterance
to the fixed 15 s encoder window, so encoder time was 25.5 ms whether the audio
was 0.74 s or 8.15 s; with mel that was 28.6 ms of length-independent work and
80% of the 1 s result. And the 48 kHz to 16 kHz resample cost a linear 4.6 ms
per second of input, which exceeded the decode loop at every measured length.
Both are what the two sections below went on to remove.

The decoder and joint-decision models run `cpuOnly` and the encoder runs
`cpuAndNeuralEngine`, read off the live models rather than inferred from the
request. Forcing the encoder to `cpu-only` moves it from 26.0 ms to 86.0 ms,
which is the evidence that the ANE is engaged.

Hold mode had no measured release-to-text number until now.
`scripts/bench-hold.sh` releases at the fixture's predicted acoustic end and
stops at transcript-ready: 54.0 ms p50 at 1 s, 106.5 ms at 5 s, 231.5 ms at 20 s,
with p95 of 79.5 / 158.6 / 280.6 ms. The 15 ms `run_manual` poll contributes a
median 8 to 12 ms, capture shutdown about 1 ms, and the remainder is ASR, which
runs 9 to 42% slower than the isolated bench, not monotonically in length,
because capture is still live in the same process.

Enabling the profiler costs nothing measurable: matched 30-repetition runs at
1 s and 5 s are within 0.2 ms, and at 10 s an interleaved eight-block on/off A/B
puts the per-block delta at -0.17 ms median with the sign flipping between
blocks. Absolute stage times need a quiet machine, which is the larger effect: a
repeat under a competing job reproduced the dispatch counts exactly and kept the
encoder flat, with the CPU-side stages 10 to 20% higher.

The stage split is validated at runtime, not just by construction. `bench_asr`
fails a run whose report has no encoder dispatches, whose `windows` and
`encoder_calls` disagree, or that contains an unattributed prediction, and
`scripts/bench-stages.py` repeats those checks before writing a CSV. Without
that, a moved Core ML entry point would report the affected stage as free while
leaving every other number plausible.

Full tables, dispatch counts, and method are in
[`bench/README.md`](../../bench/README.md).

## Short-utterance encoder cost — 2026-09-04

With the resample retired by ADR-0030, the encoder is the whole of the
short-utterance floor: it is compiled at a fixed 15 s mel window and every
utterance is zero-padded to it, so a one-second utterance paid 26.0 ms of
encoder and 3.2 ms of mel out of a 32.7 ms result. Compiling the same NVIDIA
checkpoint at shorter windows and dispatching each utterance to the narrowest
one that holds it removes most of that. **A one-second utterance now costs
7.7 ms of encoder and 0.7 ms of mel instead of 26.0 and 3.2, taking the
worker-internal call from 32.4 to 11.3 ms.** At 2.8 s the encoder is 9.7 ms
against 25.3, at 4.9 s 9.98 against 25.4, and at 7.0 s 12.5 against 25.6;
utterances past the longest bucket are unchanged. Buckets of 2, 5 and 8 seconds
cost 1.77 GB of disk and take peak RSS from 0.10 to 0.19 GiB.

The bucket runs predate the ADR-0030 merge, so both arms still paid the
worker-side resample, which is why the numbers above are worker totals excluding
it rather than end-to-end p50. The unchanged arm's encoder and mel reproduce the
post-ADR-0030 per-stage table within 0.7 ms and its worker totals within 1.1 ms
at every fixture, which is what makes the two comparable.

Quality is unchanged and checked at the transcript, not the score: matched
ten-repetition gold runs differing only in model directory both give 5.434783%
WER and 3.571429% CER with zero spread, and every hypothesis is byte-identical
between the two arms. Corpus decode p50 falls from 0.4534 s to 0.3397 s
(75.1× to 100.2× RTFx).

Encoder cost is close to linear in the compiled window but not exactly: it rises
7.5 µs per mel frame from 201 to 801 frames, 30.7 µs per frame from 801 to 1201,
and 5.4 µs again to 1501. About 6 ms is fixed cost that no shorter window
removes. That shape is why 8 s is worth compiling and 12 s is not, and it is
unexplained. The compute plan behind it, and the per-operation device
assignments for all three models, are in [`COMPUTE_PLAN.md`](COMPUTE_PLAN.md);
the full tables are in [`bench/README.md`](../../bench/README.md).

This needed a three-file change to FluidAudio, which hardcodes the 15 s window.
The package is a local path override reconstituted by
`scripts/reconstitute-fluidaudio.sh` from the pinned upstream revision plus
`native/ParakeetCoreMLWorker/patches/fluidaudio.patch`; nothing
of FluidAudio is checked in but the patch, and the change is written to be
offered upstream.

## Mel and resample cost — 2026-09-04

Both sub-checks the encoder-bucketing work carried are now answered, and neither
leaves a follow-up.

The native Swift mel cost 3.1 ms at every utterance length, for the same reason
the encoder was flat: `UnifiedMelExtractor` is built at the batch layout's
window and computed 1501 frames whether or not the audio filled them. Bucketing
fixed it as a side effect — a 2 s bucket computes 201 frames and mel drops to
0.71 ms, a 5 s bucket to 1.34 ms. At those numbers mel is 6% of the remaining
one-second call and moving log-mel into the Core ML graph, which Voz does, would
be arguing over half a millisecond. Not worth doing on this evidence.

The double resample is gone, retired by ADR-0030 rather than by this work: Rust
already converted the same audio to 16 kHz for Silero VAD, so converting again
in the worker was the redundant copy and it was on the critical path after the
endpoint. Converting once in the capture callback took the worker's resample
stage from 4.6 ms per second of 48 kHz input to 0.001 ms and halved measured IPC
by shrinking the payload. There is nothing left to file: the stage no longer
exists.

## Tap confirmation window — 2026-09-04

Tap Fast was endpoint-bound: the speculative decode finished about 80 ms before
the 150 ms confirming Silero state released, and post-endpoint work was 0–1 ms.
The window, not the decode, set the number. `scripts/bench-endpoint-sweep.sh`
sweeps it against false early cuts on three fixtures.

Two oracles are reported per row, because neither alone is sound. `false_cuts`
counts commits that landed before Core Audio's predicted instant for the
fixture's last sample above -80 dBFS; the LibriSpeech fixtures carry room tone
above that floor, so a short window can miss that marker with every word
intact. `mismatches` compares the transcript against the fixture reference and
is the oracle for lost speech.

Fast curve on the 4.854 s synthesized fixture, 15 repetitions, speculative
Core ML:

| window | false cuts | mismatches | mean | p50 | p95 |
|---:|---:|---:|---:|---:|---:|
| 150 ms | 0/15 | 0 | 186.4 ms | 181.0 ms | 209.9 ms |
| 120 ms | 0/15 | 0 | 155.5 ms | 150.0 ms | 171.9 ms |
| **90 ms** | 0/15 | 0 | **134.9 ms** | **125.0 ms** | 159.5 ms |
| 60 ms | 0/15 | 0 | 137.1 ms | 147.0 ms | 156.1 ms |

The curve stops improving at 90 ms and reverses at 60. Every 60 ms repetition
reports `t_asr_done == t_vad_endpoint`: the synchronous speculative decode
blocks the VAD watcher, so below about 90 ms the decode is the floor and a
shorter window buys nothing. Decode time is bimodal at 54 ms and 94 ms, which
puts the 5 s p50 on a cluster boundary and makes its mean the steadier reading.

The same curve on the 3.505 s human fixture separates the candidates that the
synthesized one cannot, 15 repetitions:

| window | false cuts | mismatches | mean | p50 | p95 |
|---:|---:|---:|---:|---:|---:|
| 150 ms | 0/15 | 0 | 57.3 ms | 59.0 ms | 63.0 ms |
| **90 ms** | 0/15 | 2 | **1.5 ms** | **0.0 ms** | 6.8 ms |

The mismatches are `Concorde` for `Concord`: a decoder spelling variant, not
truncation. Those transcripts read `Concorde returned to its place amidst the
tents,` and carry every reference word. The variant is not window-dependent —
it appears in 12 of the 153 transcripts across every run on this fixture,
warmups included, at both 90 ms and 150 ms confirming windows.

The absolute numbers on this fixture are offset by the marker. Silero calls
silence inside the LibriSpeech room tone that keeps the -80 dBFS acoustic-end
marker alive, so the 0 ms reading is an artifact of where that marker sits, not
a commit before the speech ended. The 59 ms delta is the real saving.

Long-form, 14.225 s fixture with its reviewed 544 ms intra-utterance pause:

| window | false cuts | mean | p50 | p95 |
|---:|---:|---:|---:|---:|
| **750 ms** | 0/15 | — | — | — |
| 500 ms | 0/15 | 382.0 ms | 379.0 ms | 401.0 ms |
| 300 ms | 15/15 | — | — | — |

500 ms survived this pause in all 15 repetitions, which is one fixture and does
not justify moving a pause-safety policy. The 750 ms row reports no latency at
all: another agent's fixture played through the shared BlackHole device during
that run and its words appear in two of its transcripts, so every duration it
produced is discarded. Its zero false cuts stand, because a commit's timing
relative to playback does not depend on what else was audible. The shipping
750 ms latency comes from the clean 30-repetition gate run below.

### Decision

**Tap Fast moves from 150 ms to 90 ms. Long-form stays at 750 ms.** Confirmed at
30 repetitions:

| fixture | 150 ms | 90 ms | delta p50 |
|---|---:|---:|---:|
| 4.854 s synthesized | 182.0 ms p50 / 183.5 mean | 148.5 ms p50 / 141.8 mean | **-33.5 ms** |
| 3.505 s human | 59.0 ms p50 / 58.1 mean | 13.0 ms p50 / 10.1 mean | **-46.0 ms** |

False cuts 0/30 everywhere; the single `Concorde` mismatch is the same lexical
variant. The 40 ms target is met on human speech and missed by 6.5 ms on the
synthesized fixture, where the bimodal decode pins p50 to a cluster boundary —
that fixture's mean improves by 41.7 ms. The decode's two modes are the next
lever and are not addressed here.

Both gates were re-run at 30 repetitions and pass unchanged: the long-pause
endpoint gate at 0/30 false stops (single 667.0 ms p50, multi 635.0 ms p50 /
647.5 ms p95), and the frozen 3× end-to-end gate, now pinned to
`--confirmation-ms 150` so it stays like-for-like, at 594.5 → 182.0 ms p50
(3.27×) and 635.1 → 203.6 ms p95 (3.12×).

### Punctuation-aware early commit: rejected

A shorter window gated on the provisional transcript ending in sentence-final
punctuation was implemented, measured, and removed. Three results killed it.

It does not fire on real speech. The model ends the human single-sentence
fixture `...amidst the tents,` with a comma in all 30 repetitions, so the gate
never opened and the row is identical to the control: 59.0 ms p50 against
59.0 ms.

Where it does fire it is worth nothing over a plain shorter window. On the
synthesized fixture, 150 ms gated at 90 ms and an ungated 90 ms produce the same
distribution at 30 repetitions: 148.5 ms p50 both, means 139.4 and 141.8 ms.
The 15-repetition rows for those two configurations read 145.0 and 125.0 ms p50,
which is the bimodal decode landing a 15-sample median on either side of the
cluster boundary; their means differ by 4.4 ms. Read the 30-repetition figures.

Its premise is false. Long-form at 750 ms with a 90 ms punctuated window cut the
multi-sentence fixture 15/15, and the provisional transcript at the 544 ms
intra-utterance pause reads `...to greet the arrival of the young princess.` The
model emits a sentence-final period mid-utterance, which is exactly where the
policy must hold. Punctuation is not an end-of-utterance signal.

Replay:

```bash
REPS=15 WARMUP_REPS=2 scripts/bench-endpoint-sweep.sh
```

## Native RNNT decode loop — 2026-09-04 (kata 2564)

With bucketing and the resample gone, the greedy transducer loop was the largest
stage left on a short utterance. It ran through Core ML one dispatch at a time:
one prediction-network step per emitted token, one joint evaluation per frame and
per token, each against a floor near 100 µs. Neither model has a Neural Engine
path, so the round trip bought nothing. The worker now reads the weights out of
`parakeet_unified_decoder.mlmodelc` and
`parakeet_unified_joint_decision_single_step.mlmodelc` and runs both programs in
process, and the whole decode issues one Core ML dispatch: the encoder.

The loop is 2.4x faster and the utterance 1.4x: at 4.967 s the loop drops from
20.8 to 8.6 ms and the worker total from 42.4 to 30.2 ms, with 183 Core ML calls
becoming none. The full per-length table is in
[`bench/README.md`](../../bench/README.md). The 5 ms target the issue set is
missed, and the reason is not dispatch: one prediction step reads 13.1 MB of
fp16 weights and a 5 s utterance takes 49 of them. The next lever, unimplemented,
is precomputing `W_ih · embed[token]` for all 1025 tokens, which removes a
quarter of that traffic and changes only the order the fp32 sum accumulates in.

Three things the loop can now do that the compiled graph could not, and they are
most of the 2.4x: the encoder projection is one matrix product over the whole
window instead of a matrix-vector product per joint call, the decoder-side
projection is computed once per emitted token instead of once per call, and the
softmax runs only when a token is actually emitted.

The arithmetic is not bit-identical to Core ML and cannot be made so. Core ML's
CPU `lstm` accumulates its recurrent matrix product in fp16: with a zero `h_in`,
which drops that term, the two agree to about one fp16 ulp, and with a nonzero
one they differ by up to 0.02. The native loop accumulates in fp32 over the same
fp16 weights, which is the more accurate of the two, and reproduces every fp16
rounding the exported program performs at an operation boundary. Over a 32-step
trajectory the decoder output the joint consumes agrees to 0.014, and the
divergence does not compound. `scripts/capture-rnnt-parity.py` measures all of
this and captures the fixture the Swift tests replay; it also settled the one
thing the MIL text does not say, the LSTM gate order, at `i, f, o, g` by a 70x
margin over the next candidate. The empirical answer is the gold corpus: all
seven fixtures produce byte-identical hypotheses over 10 repetitions, at the
baseline 5.43% WER / 3.57% CER.

The inner matrix-vector product is in C. Swift compiles
`SIMD8<Float>(SIMD8<Float16>)` to an outlined runtime call with a register spill
around it, which measured 356 ms against 15 ms for the same loop written with
NEON intrinsics.

This needed a second change to FluidAudio, in the same checked-in patch as the
encoder window: a protocol for the decode loop and a factory the manager takes
it from. `--rnnt-engine coreml` keeps the original path, which is how the two
arms above were measured on one build.

## Core ML runtime-plan tuner — 2026-08-11

Release worker, ten corpus repetitions and three model-load repetitions on the
same M5 Pro 24 GiB / macOS 26.5.1 machine. `short` is the combined six-fixture
19.825 s bucket; `long` is the 14.225 s human fixture. Peak RSS includes the
Rust process and resident worker.

| plan | load p50 | warmup | short p50 / p95 | long p50 / p95 | short / long RTFx | peak RSS | WER / CER | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CPU+ANE | 90.6 ms | 77.4 ms | 316.6 / 324.6 ms | 132.9 / 135.6 ms | 62.6× / 107.1× | **0.10 GiB** | 5.43% / 3.57% | **selected both regimes** |
| All | 89.8 ms | **72.3 ms** | **315.5 / 318.8 ms** | **132.1 / 133.5 ms** | **62.8× / 107.7×** | **0.10 GiB** | 5.43% / 3.57% | safe, below 5% win floor |
| CPU+GPU | — | — | — | — | — | — | — | MPSGraph MLIR compile failure |
| CPU-only | **80.4 ms** | 262.2 ms | 680.3 / 705.1 ms | 194.8 / 217.1 ms | 29.1× / 73.0× | 1.26 GiB | 4.35% / 3.36% | memory and latency loss |

CPU+ANE and `all` differ by less than 0.8 ms at long p50 and 1.2 ms over the
entire short corpus. Normalizing each bucket to a per-utterance p50 and
including median load and first-decode warmup in a representative 20-utterance
session leaves `all` below the 5% minimum win, so deterministic policy retains
the baseline. CPU-only is 2.15× slower short and 1.46× slower long, with roughly
12.5× the observed memory.
Every completed candidate passed the absolute gold limits with zero
within-candidate output spread; category scores are preserved in the JSON
profile and checked against the baseline.

Independent ten-pass reruns selected the same two CPU+ANE regimes and repeated
the same three-completed/one-failed candidate pattern. Core ML's persistent
plan cache made CPU-only's process-first warmup vary from 262 ms to 3.35 s;
steady-state latency and the memory gate still rejected it in every run.

Replay:

```bash
scripts/build-coreml-worker.sh
cargo run --release --locked --bin tune_asr -- \
  --repetitions 10 --load-repetitions 3
```

## Qwen3-ASR 0.6B challenger — 2026-08-11

Developer-only MLX oracle at commit
`d1a035514e1d6ac31da7658b273482656eacba61`, cached immutable weights, one
warmup, and ten measured runs over the same gold corpus:

| precision | WER / CER | corpus p50 / p95 | RTFx | peak RSS | weights | decision |
|---|---:|---:|---:|---:|---:|---|
| q8 | **4.35% / 3.15%** | 0.840 / 0.843 s | 40.5× | 1.12 GiB | 960 MiB | offline aggregate win; production fail |
| q4 | 6.52% / 3.36% | **0.717 / 0.722 s** | **47.5×** | **0.84 GiB** | **676 MiB** | quality loss vs q8 |
| fp16 | 5.43% / 3.57% | 2.961 / 2.974 s | 11.5× | 1.94 GiB | 1.75 GiB | dominated by q8 |

The Qwen WAVs are preloaded/resampled before timing, a conservative advantage
against shipping Core ML's resample-plus-inference row. Even so, q8 is 1.86×
slower at corpus p50 and uses about 11.0× its measured resident memory. It also
regresses the shipping `noisy` and `numbers` category rows, so the better
aggregate offline WER does not clear the existing no-category-regression gate.

Actual q8 streaming over exact 2 s, unpaced 100 ms segmented, and jittered
transport writes was repeated ten times. Every schedule produced the same
final output at 12 model boundaries per repetition, with zero nondeterministic
outputs. These synchronous schedules test segmentation, not real-time queueing.
All fixtures ended mid-model-chunk. Every schedule measured **23.91% WER /
21.01% CER**;
the 14.225 s fixture rose from 0% offline WER to 41.86% streaming WER. Qwen is
therefore rejected as a production backend. Full artifact identities,
per-category rows, native-build evidence, replay commands, and packaging
analysis are in [`QWEN3_ASR_EVALUATION.md`](QWEN3_ASR_EVALUATION.md). The raw
reports and machine-verifiable summary are under `bench/qwen3-asr/`.

## Hold incremental windows — 2026-09-04 (kata ktfa, ADR-0032)

Hold had nothing overlapping its decode: the first model call happened after
the hotkey came up, so release-to-text grew with the recording. ADR-0032 cuts
the held recording into windows while the key is still down, decodes each in
the background, and joins them on the words neighbouring windows agree on, so
what the user waits for is the tail window.

Release-to-text p50, 30 repetitions per bucket, both arms back to back on the
same quiet machine (`bench/hold.csv` against `bench/hold-serial.csv`):

| captured audio | serial | windowed | p95 serial | p95 windowed |
|---:|---:|---:|---:|---:|
| 0.875 s | 48.5 ms | **36.0 ms** | 57.5 ms | 38.0 ms |
| 2.891 s | 59.5 ms | **47.0 ms** | 69.8 ms | 65.5 ms |
| 4.917 s | 64.5 ms | **58.5 ms** | 96.3 ms | 88.2 ms |
| 8.128 s | 104.5 ms | **66.0 ms** | 146.6 ms | 76.0 ms |
| 16.576 s | 180.5 ms | **65.0 ms** | 231.8 ms | 84.0 ms |
| 16.139 s (multipause) | 177.5 ms | **62.0 ms** | 197.9 ms | 90.0 ms |

Release-to-text stops growing with the recording past five seconds, which is the
structural change: the encoder still costs what it costs, but all of it except
the tail now runs while the user is still talking. The tail never waited behind
an in-flight window at p95 in any bucket.

Across the two multi-window buckets, 240 seams were merged over 30 repetitions
each, 180 of them resolved by word agreement rather than by a silent overlap,
and none duplicated or dropped a word.

The three shortest buckets never cut a window — the per-session log shows
`windows=1` on every repetition — so their 6 to 13 ms comes from replacing
`run_manual`'s 15 ms sleep with a 3 ms blocking read on the audio tap, which
recovers what the Hold baseline section attributes to that sleep.

**Windows are cut at a fixed 6 s cap, not at pauses, and that is the measured
decision.** Cutting at pauses moved gold WER from 5.43% to 6.52% and CER from
3.57% to 4.62%, failing the manifest's zero-regression gate. The damage was in
`commands` (12.20% to 14.63%) and `numbers` (6.67% to 10.00%) — fixtures two to
four seconds long, where a pause inside a short utterance split it and each half
decoded worse than the whole. Cutting only at the cap reproduced the plain
transcript on every category: 5.43% WER, 3.57% CER, no change anywhere.

FluidAudio had already measured the same effect and left the note in
`UnifiedAsrManager.decodedTokens`: silence-aligned starts cost about 1 WER point
against a fixed stride on the 15 s offline encoder, with no artifact benefit.
Two independent measurements agreeing is enough to ship against pause alignment.
The pause path is kept and tested but off by default, disabled by setting
`hold_window_min_seconds` equal to the cap.

The `6,6` PASS deserves one caveat: six of the seven gold fixtures are under
4.3 s and decode as a single window in that arm, so they are identical to plain
by construction. The one fixture long enough to cut, `librispeech-multi` at
14.2 s, was cut and scored 0.00% WER. The multi-window evidence comes from there
plus the 20 s and multipause loopback buckets.

The join needs word boundaries, which the worker now reports as `token_spans`
from FluidAudio's `transcribeWithTimings`. It runs the same decode as
`transcribe` and reads emission frames the greedy RNNT decoder already recorded,
so the spans cost only a frame-to-seconds conversion.

Full tables, per-session window and seam counts, and the three gold arms are in
[`bench/README.md`](../../bench/README.md).

## Neural Engine idle re-wake — 2026-09-04 (kata snx0)

Measured before fajz moved resampling into the capture callbacks and before
fgzt's bucketed encoders landed, so the absolute milliseconds below describe
the pre-merge path. Every arm paid the same conditions and the arm-to-arm
comparisons the decision rests on are unaffected.

The Neural Engine power-gates when idle, and between dictations this app is
idle for seconds to minutes. Measured on an M5 Pro at `bench/idle-*.csv`, a
fully cold decode of the 1 s fixture costs 26.5 ms more than a back-to-back one
at p50, and the encoder — the only stage on the engine — accounts for 25.4 ms
of it. Warmth decays gradually: nothing measurable at a 100 ms gap, about 3 ms
by 2 s, half the total by 5 s, plateau by 10 s.

Three treatments were measured against cold at 1 s and 5 s in both Tap Fast and
Hold. Deltas are the arm against `cold`, positive meaning faster:

| mode | fixture | prime p50 | prime p95 | cadence p50 | cadence p95 |
|---|---|---:|---:|---:|---:|
| Tap Fast | 1 s | +24.0 ms | **+60.9 ms** | +26.0 ms | +37.3 ms |
| Tap Fast | 5 s | +22.0 ms | **+27.4 ms** | +12.0 ms | +6.5 ms |
| Hold | 1 s | +73.0 ms | **+70.6 ms** | +76.0 ms | +71.0 ms |

**Decision: ship the hotkey-down prime, reject the keep-alive cadence.**

The prime clears the 10 ms p95 bar by a wide margin everywhere it was measured
cleanly, and it costs one silent 0.5 s dispatch per hotkey press. It ships on
by default behind `Settings::prime_engine_on_keydown`, fired from
`App::on_hotkey_press` through `warmup::EnginePrimer`, which spawns so the
event-tap callback never blocks and drops a second request while the first is
in flight.

The cadence is rejected on two counts. It buys the warmest encoder of the three
arms and still loses to the prime on total latency, badly in the tail — 65.2
against 41.6 ms at p95 on the 1 s Tap Fast fixture — because its dispatches
contend with the decode that follows them for the worker's single pipe. And it
costs 2.41 s of worker CPU per minute against a measured zero when idle, which
is the "measurable battery cost" the issue set as its rejection condition. The
engine's own draw is on top of that and was not measured: `powermetrics
--samplers ane_power` needs root and no interactive sudo was available.

The prime recovers less at 5 s than at 1 s, and that is inherent rather than a
tuning problem: five seconds of talking is already half the cool-down, so a
dispatch fired at the press has partly decayed by the endpoint. Closing that
remainder needs a dispatch nearer the endpoint, which is what the cadence was,
and the cadence costs more than it returns.

The 5 s Hold rows are omitted from the table above because they do not separate
at n=12. Hold's `warm` arm still plays the fixture, so it sits about one
utterance from its own previous dispatch rather than back to back, which at 5 s
leaves an expected cold-to-warm separation of roughly 10 ms - inside the noise
at that sample size. `bench/README.md` records the rows and the reasoning.

## Parakeet TDT 0.6B v3 challenger — 2026-09-04

**No-go.** Measured at ten repetitions with zero WER and CER spread, TDT 0.6B
v3 records 7.608696% WER / 3.991597% CER against the frozen Unified baseline of
5.434783% / 3.571429%. That is seven word edits of 92 against five. The
manifest sets `max_wer_regression_percent` to 0.00, so the bar is WER ≤
5.434783% exactly and TDT is 2.17 points over. Its absolute 7.61% is still
under the 8.00% ceiling; the gate that fails is the regression one. Keep
Unified as the default and do not open a switch issue.

The quality margin is thinner than the aggregate reads. Four of seven fixtures
are clean in both arms and two more fail identically in both; the whole 5 → 7
difference is one four-word utterance, "Is IBM up today?", which Unified renders
"Is IPM up today?" (1 edit) and TDT renders "It's I PM up today." (3 edits).
Every per-category difference traces to that fixture. So the corpus supports
"TDT is not better and fails a gate Unified passes", not "TDT is worse at
English"; a wider corpus would be needed for the stronger claim. The latency
result below is what makes building one pointless.

| arm | WER | CER | corpus p50 | RTFx p50 | peak RSS | load | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Unified | **5.434783%** | **3.571429%** | **0.3048 s** | **111.7×** | **0.10 GiB** | 0.135 s | pass |
| TDT v3 | 7.608696% | 3.991597% | 0.4161 s | 81.8× | 0.12 GiB | **0.100 s** | **fail** |

The latency case that motivated the trial does not survive measurement either.
At 30 repetitions TDT is slower at every length: 45.0 against 31.0 ms at one
second, 51.0/36.0 at three, 58.5/44.5 at five, 68.0/53.0 at ten, and
182.0/110.5 at twenty. The published 155.6× versus 123.3× RTFx does not appear
here at any length.

TDT's duration head does what it claims — 92 joint predictions against
Unified's 261 on a 14.225 s fixture, 2.84× fewer — and returns nothing, because
each TDT joint call costs 3.06× what a Unified one does, 0.300 ms against
0.098 ms. Pinning the decoder and joint CPU-only, where the Unified loader pins
them, moved joint dispatch from 27.58 ms to 28.90 ms: slightly worse, so
placement is not the cause. What is left is the graph. `JointDecisionv3`
computes `top_k_ids` and `top_k_logits` at K=64 on every call for script-aware
language filtering, over an 8,192-entry vocabulary against Unified's 1,024.

Separately, and larger, TDT carries an **unexplained** post-dispatch tail of
about 15 ms against Unified's 0.03 to 0.12 ms. It is close to constant where a
per-token cost could not be — 14.72 to 15.16 ms while decoder calls go from 5
to 52 — so tokenizer decode and token-timing assembly are ruled out as the bulk
of it; it rises to 19.58 ms on the two-window fixture, so some is per-window.
Overlap-merge in `ChunkProcessor`, the per-utterance progress-emitter session,
and decoder-state teardown are candidates, none measured: the profiler's
timeline ends at the last dispatch. That tail alone is the whole of the 14 ms
deficit from 1 to 10 seconds, and it erases a real 1.6 to 2.2 ms encoder
advantage and 1.8 ms mel advantage. It is the first thing to measure if TDT is
ever revisited.

TDT's published Core ML encoder takes a fixed `[1, 128, 1501]` mel, the same
15 s window the Unified offline encoder takes, so the bucketed short-window
encoders would need a separate TDT re-conversion at each window before TDT
could pay the same short-utterance saving.

The evaluation path stays in the tree so the numbers can be re-checked against
a future conversion: `--model-variant tdt-v3` on the worker,
`PARAKEET_COREML_MODEL_VARIANT` / `--backend coreml-tdt-v3` in the bench, and
`scripts/fetch-tdt-v3-model.py` for the pinned artifact. It is deliberately not
reachable from the shipping download path: TDT has no Rust integrity gate, so
the worker refuses `--model-root` for it and forbids FluidAudio's downloader.
Reopen only for a conversion that is EN-competitive on this corpus and drops
the top-K joint outputs. Full tables, the artifact manifest and the replay
commands are in [`../../bench/README.md`](../../bench/README.md); raw reports
are under `bench/f0zg/`.
