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
