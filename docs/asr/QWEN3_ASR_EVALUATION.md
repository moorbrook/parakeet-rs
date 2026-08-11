# Qwen3-ASR 0.6B Apple Silicon evaluation

## Decision

**No-go for the production backend.** Keep native Core ML Parakeet Unified as
the default and sherpa-onnx as the contextual-vocabulary/load-failure fallback.
Qwen3-ASR q8 is an interesting offline multilingual model, but it does not
clear this app's streaming-quality, latency, memory, packaging, or native-build
gates on the target Mac. No Qwen or Python dependency enters Parakeet.app.

This is a measured rejection, not a permanent claim about the model family.
Re-open the decision if a maintained native Apple Silicon runtime appears and
its streaming path passes the same per-category corpus gate.

## Reproducibility contract

Measurements used an Apple M5 Pro with 24 GiB, 15 logical CPUs, arm64 macOS
26.5.1, and the checked-in 34.0495625 seconds of human speech: 92 words / 476
characters across seven fixtures. Each offline row is a process-cold,
file-cached model load, explicit parameter evaluation, one warmup, and ten
corpus repetitions. The immutable Hub snapshot is resolved, every model asset
is verified, every WAV must match `bench/gold/sources.json`, and the exact
ffmpeg 8.1.2 resampler binary is pinned by SHA-256. WAV decode/resampling to
16 kHz occurs before the timers start. This gives Qwen the conservative
inference-only boundary while the
shipping Core ML row includes its internal resampling. Peak RSS is the Python
process high-water mark. Network download and artifact hashing are not timed.

The developer-only oracle is the independent Apache-2.0
[`mlx-qwen3-asr`](https://github.com/moona3k/mlx-qwen3-asr) implementation at
commit `d1a035514e1d6ac31da7658b273482656eacba61`. PEP 723 pins that commit in:

- `scripts/eval-qwen3-asr-offline.py`
- `scripts/eval-qwen3-asr-streaming.py`

PEP 723 requires the measured Python 3.14.2 and direct package versions; the
adjacent `uv` lockfiles freeze the full transitive graph and wheel hashes. The
commands below require the recorded model snapshot, weight digest, corpus
digests, resampler digest, and a
composite digest over every weight, configuration, tokenizer, prompt, and
preprocessing asset. A moved or partially changed snapshot therefore fails
before warmup instead of silently becoming a new benchmark row.

Run them with `uv`; neither dependency becomes part of Cargo or the app:

```bash
uv run scripts/eval-qwen3-asr-offline.py \
  --model mlx-community/Qwen3-ASR-0.6B-8bit \
  --manifest bench/gold/manifest.json --audio-dir bench/gold/audio \
  --sources bench/gold/sources.json \
  --baseline bench/qwen3-asr/shipping-coreml-baseline.json \
  --expected-snapshot 89e96d92ba34aca20b3e29fb10cc284097d1219f \
  --expected-weight-sha256 b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8 \
  --expected-artifact-sha256 901975e84de875144dabd3a64655ed8f7335562626f816e68409bd61c50f278a \
  --expected-ffmpeg-sha256 1332dc2de372bade9a8a63da0d6cdfab9de97fcefbae707bcc0b0506e1203327 \
  --repetitions 10 --json-out /tmp/qwen-q8.json

uv run scripts/eval-qwen3-asr-streaming.py \
  --model mlx-community/Qwen3-ASR-0.6B-8bit \
  --manifest bench/gold/manifest.json --audio-dir bench/gold/audio \
  --sources bench/gold/sources.json \
  --baseline bench/qwen3-asr/shipping-coreml-baseline.json \
  --expected-snapshot 89e96d92ba34aca20b3e29fb10cc284097d1219f \
  --expected-weight-sha256 b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8 \
  --expected-artifact-sha256 901975e84de875144dabd3a64655ed8f7335562626f816e68409bd61c50f278a \
  --expected-ffmpeg-sha256 1332dc2de372bade9a8a63da0d6cdfab9de97fcefbae707bcc0b0506e1203327 \
  --repetitions 10 --json-out /tmp/qwen-q8-streaming.json
```

Each evaluator writes its report and exits 2 when the candidate fails the
aggregate, repeatability, or per-category baseline gate; that exit is expected
for these rejected rows. The four Qwen reports and a fresh shipping Core ML
gold report are checked in beside the summary. `uv run
scripts/verify-qwen3-asr-evidence.py` verifies their
SHA-256 values and every summarized metric, so accidental drift or a one-sided
edit fails deterministically. The reports are local benchmark
evidence, not externally signed attestations.

## Offline result

| backend | WER | CER | load | warmup | corpus p50 / p95 | RTFx | peak RSS | weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shipping Core ML Parakeet | 5.43% | 3.57% | 0.168 s | **0.321 s first result** | **0.452 / 0.470 s** | **75.4×** | **0.10 GiB** | 586 MiB graph pack |
| Qwen3-ASR q8 | **4.35%** | **3.15%** | 0.119 s | 0.110 s | 0.840 / 0.843 s | 40.5× | 1.12 GiB | 960 MiB |
| Qwen3-ASR q4 | 6.52% | 3.36% | **0.096 s** | **0.093 s** | **0.717 / 0.722 s** | **47.5×** | **0.84 GiB** | **676 MiB** |
| Qwen3-ASR fp16 | 5.43% | 3.57% | 0.205 s | 0.453 s | 2.961 / 2.974 s | 11.5× | 1.94 GiB | 1.75 GiB |

q8 is the correct Qwen candidate. It is 3.52× faster than fp16 at corpus p50,
uses 42% less peak memory, and happened to remove one fp16 greedy-decoding
error on this small corpus. q4 has only 17.2% more throughput and saves 25% peak
memory, while adding two word errors: WER rises by 2.17 points to 6.52%.
That is not a justified default-quantization trade.

Against shipping Parakeet, q8 is 1.86× slower at corpus p50, has 1.86× lower
throughput, uses about 11.0× the observed memory, and has a 1.64× larger weight
artifact. Its better aggregate offline WER is therefore insufficient by
itself.

### Category gate

The runtime-plan policy requires a challenger to be no worse in every
category, not merely better in aggregate.

| category | shipping WER / CER | q8 WER / CER | result |
|---|---:|---:|---|
| commands | 12.20 / 8.50 | **9.76 / 7.50** | improve |
| custom vocabulary | 27.27 / 8.93 | **9.09 / 3.57** | improve |
| general | 0 / 0 | 0 / 0 | tie |
| long | 0 / 0 | 0 / 0 | tie offline |
| noisy | **0 / 0** | 9.09 / 3.70 | **fail** |
| numbers | **6.67 / 8.33** | 10.00 / 9.03 | **fail** |
| proper nouns | 3.57 / 1.14 | **2.38 / 0.91** | improve |

All three Qwen precisions produced one transcript per fixture across ten runs,
so the comparison is deterministic. q8 still fails because the distant Amy
fixture becomes “Ali” and spoken “seven thirty” changes form in a way that the
lexical policy scores worse.

## Actual streaming boundaries

The official project reports unified offline/streaming behavior, but its
[published implementation currently exposes streaming only through the vLLM
backend](https://github.com/QwenLM/Qwen3-ASR#streaming-inference). The local
test therefore exercises the independent MLX implementation's experimental
incremental cache path. It feeds the real 16 kHz waveform and records every
model-visible 2-second boundary plus finalization; it does not infer streaming
quality from offline files.

Three transport schedules were used:

- exact 2-second writes;
- unpaced 100 ms segmented writes;
- repeating jittered writes of 0.37, 0.83, 1.41, 0.29, 2.17, and 0.53 seconds.

All seven fixtures end mid-model-chunk. Across ten repetitions, each schedule
emitted 12 model boundaries per repetition, produced identical final text, and
had zero nondeterministic outputs. Outer segmentation is therefore not the
source of the errors. These synchronous runs do not claim to model a paced
producer, queueing, or capture backlog.

The model and tokenizer are bound through one verified runtime session. The
pinned library's tail-refinement helper otherwise falls back to its default
tokenizer when handed a preloaded model, so the evaluator explicitly routes
that synchronous helper through the same verified session. The binding is
recorded in the raw report and tail refinement remains enabled.

| mode | WER | CER | offline-divergent fixtures | corpus p50 / p95 | finalize p50 / p95 |
|---|---:|---:|---:|---:|---:|
| q8 offline | **4.35%** | **3.15%** | — | 0.840 / 0.843 s | — |
| q8 streaming, any schedule | **23.91%** | **21.01%** | 2 / 7 | 1.854–1.860 / 1.858–1.866 s | 0.985 / 0.987–0.991 s |

The 14.225-second fixture is exact offline but reaches 41.86% WER in streaming:
middle spans disappear and an earlier phrase repeats. The 3.968-second alarm
changes from offline “Please wake me up at 7:30 AM.” to streaming “Please wake
me up at seven. hundred.” The long-form result alone rejects this streaming
path, and the corpus regression is far beyond any product threshold.

## Native Rust route

### `mlx-rs`

`mlx-rs` 0.25.3 is the most direct no-Python bridge. It is MIT OR Apache-2.0,
wraps MLX-C/MLX 0.25.1, and binds the needed primitives: conv2d, quantized
matmul, scaled-dot-product attention, RoPE, RMSNorm, and safetensors. It does
not implement Qwen3-ASR. A port still owns the audio encoder, decoder, mRoPE,
tokenizer/prompt rules, greedy generation, KV cache, chunk merging, and tail
finalization.

The exact release spike did not produce a binary:

- The crate requires Rust 1.82 while this repository declares 1.77.
- `mlx-sys` builds MLX-C and MLX from source with CMake and bindgen.
- On Xcode 26.6 with first-launch setup complete, the fresh default target
  failed after 15.11 seconds because `xcodebuild -showComponent
  MetalToolchain -json` reports `uninstalled`; `xcrun metal` is only a stub.
- Downstream `default-features = false` did not provide a CPU-only escape:
  `mlx-rs` enables `mlx-sys` defaults at its own dependency edge, so Cargo
  still turned Metal on. That second fresh target failed after 46.57 seconds.
- Each failed target had already grown to about 372 MiB. A minimal Rust release
  control built in 0.14 seconds to a 431,216-byte binary.

Installing an Xcode component merely to force a successful size number was not
justified after quality, latency, and memory had already rejected the backend.
The missing completed-binary delta is treated as a failed gate, not estimated
away.

The wrapper statically links MLX and MLX-C, but Metal kernels remain a separate
`mlx.metallib`. MLX 0.25.1 searches beside the executable and under an
executable-relative `Resources/` directory. A future Parakeet worker would
therefore need the metallib colocated in `Contents/MacOS` (or explicit runtime
path wiring), included before signing, and verified in a signed app. The
worker binary is nested code; the metallib and license notices are sealed app
resources. Downloaded weights remain outside the app signature and use the
existing immutable SHA-256 fetch/activation contract.

The independent Python runtime contains 8,982 lines and 10,189 test lines.
Even restricting the port to model, encoder/decoder, tokenizer, generation,
audio, cache, and streaming behavior leaves several thousand reference lines.
A native MLX port is a multi-week implementation plus tensor truth-pack work,
not a small FFI adapter. The Python oracle should capture seam activations only
if that port is funded; there is no Rust implementation to compare today.

### Candle and Core ML

Hugging Face Candle has a real Metal backend but no Qwen3-ASR model in its
official model inventory. Choosing it removes the MLX build dependency but not
the architecture port. The same several-thousand-line encoder/decoder/cache/
streaming implementation would be required, with `tokenizers` reading the
upstream `tokenizer.json`. A responsible initial parity estimate is several
engineer-weeks, followed by Apple Silicon optimization and signing QA. It is
not a comparable ready-made alternative.

There is likewise no reviewed Core ML Qwen3-ASR artifact or reproducible
exporter. Converting this autoregressive, cache-bearing architecture would add
graph splitting, dynamic-cache contracts, quantization calibration, and a
native decoder/tokenizer around the exported graphs. Since q8 already loses
the production gates, that conversion is not justified.

## License, artifact, and distribution boundary

The [official Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR), the
official 0.6B checkpoint, and the evaluated MLX conversions identify as
Apache-2.0. `mlx-qwen3-asr` is Apache-2.0; `mlx-rs` is MIT OR Apache-2.0.
The evaluated Hugging Face snapshots do not themselves contain a `LICENSE`
file, so redistributing them requires Parakeet to add the upstream Apache-2.0
text and notices explicitly rather than assuming the snapshot is
self-documenting.

The q8 weight is 1,006,229,426 bytes and has SHA-256
`b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8`.
The evaluated conversion already contains its tokenizer and configuration
assets; the full 1.75 GiB source checkpoint in the benchmark cache was the
separately measured fp16 candidate, not a q8 runtime dependency. A production
artifact must package the exact tokenizer/config files beside q8 and preserve
source/conversion attribution. The recorded composite artifact digest already
pins every file used by this evaluation.

If reconsidered, the only acceptable integration is a dedicated native worker
behind the existing ASR backend seam, with sherpa fallback, immutable
first-use download, atomic activation, offline startup after installation, and
no Python. It must pass the current transcript/category, signed-package,
user-perceived latency, memory, and fallback gates before it can become a
selectable backend.

## Deferred scope

Multilingual fixtures were not downloaded. The issue explicitly says to test
English first and add languages if the product needs them. The current product
baseline is English, and q8 already fails English streaming, latency, memory,
and native-build gates. Multilingual quality cannot reverse that production
decision and would add corpus licensing/review work without changing the
outcome.
