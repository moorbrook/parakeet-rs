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

| backend | decoding | WER | CER | load | first result | decode p50 | decode p95 | p50 RTFx | peak RSS | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FluidAudio Core ML | greedy, int8 CPU+ANE | **5.43%** | **3.57%** | **0.130 s** | **0.278 s** | **0.445 s** | **0.462 s** | **76.5×** | **0.10 GiB** | pass |
| sherpa-onnx | greedy, int8, Core ML requested | 10.87% | 5.46% | 3.647 s | 4.582 s | 2.764 s | 2.949 s | 12.3× | 4.25 GiB | fail |
| sherpa-onnx | vocabulary, modified beam, score 2 | 10.87% | 5.46% | 3.717 s | 4.716 s | 3.126 s | 3.148 s | 10.9× | 4.26 GiB | fail |

Core ML is 6.21× faster at corpus-decode p50 and 6.39× at p95 than the unbiased
sherpa baseline. It reduces model load by 28.0×, first result by 16.5×, and
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
