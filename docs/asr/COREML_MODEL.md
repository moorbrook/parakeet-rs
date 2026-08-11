# Core ML model contract

Parakeet's optimized backend is the offline int8 Parakeet Unified EN 0.6B
graph set published by FluidInference. The shipping worker and graph pack have
independent, immutable identities:

- FluidAudio runtime commit:
  [`00a9aa771900ea09c485659663be31019e293e47`](https://github.com/FluidInference/FluidAudio/tree/00a9aa771900ea09c485659663be31019e293e47)
- Core ML model revision:
  [`4252711f6f060f9a2f91e5f081a806d7f45eebd8`](https://huggingface.co/FluidInference/parakeet-unified-en-0.6b-coreml/tree/4252711f6f060f9a2f91e5f081a806d7f45eebd8)
- Source checkpoint ID: `nvidia/parakeet-unified-en-0.6b`
- Architecture: FastConformer-RNNT, greedy decode, blank ID 1024 and at most
  ten symbols per encoder frame
- Tokenizer: pinned 1,024-entry `vocab.json`; punctuation and case are model
  outputs rather than a postprocessing restoration pass
- Frontend: FluidAudio native 16 kHz, 128-bin mel extraction with NeMo
  per-feature normalization; fixed 15-second full-attention windows, 8x
  subsampling, and a two-second overlap for longer audio
- Execution: Core ML, int8 encoder, CPU and Apple Neural Engine
- Text contract: the model's punctuation and capitalization are retained
- License: model CC-BY-4.0; FluidAudio Apache-2.0

The Rust first-use path fetches every required file from the pinned model
revision, streams it through SHA-256 verification, and atomically publishes it
only after its byte length and digest match. Existing files pass the same gate
before use. Later launches use the metadata-bound verification cache described
in [ADR-0024](../ADR.md#0024--immutable-identities-for-rust-managed-model-artifacts).
The Swift worker receives only `--model-dir` and therefore cannot silently
fall back to FluidAudio's mutable-main downloader.

## Reviewed truth pack

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 1,355 | `6cbe6c76445410c5c6debf3d44c8c3b75e9966bf09bba5cd138c2378c62120f6` |
| `metadata.json` | 1,046 | `2b26a96b76fe1f7a04d3e867f50c75d6ce5dd1650d0dbcd4c35b591b22305f0e` |
| `vocab.json` | 15,088 | `e1a7bff4f5df133c0f4ad47b8e43c96f6bf1865d99126a4c4725ef51d0108bec` |
| `parakeet_unified_decoder.mlmodelc/analytics/coremldata.bin` | 243 | `9ae70f6559989f88b856b326e59315798f9f0d08207a19fcc2dd3287a30088a5` |
| `parakeet_unified_decoder.mlmodelc/coremldata.bin` | 560 | `ce99c4488840fc463d59f8d4d6d2a9e8ceae8138ead51e3c265dde4d2ba4a0e9` |
| `parakeet_unified_decoder.mlmodelc/model.mil` | 13,102 | `6e60965b89c93943aa2be2d991c2461108145851fde05e1d048223a32d4cb20d` |
| `parakeet_unified_decoder.mlmodelc/weights/weight.bin` | 14,429,952 | `96f990461a5986d5e7309ad1a0f36084fbf0f4b28aec35948f8b8d0dcbf8599e` |
| `parakeet_unified_encoder_int8.mlmodelc/analytics/coremldata.bin` | 243 | `57e116a9d5765e39c0cdf754137ab744ddae34d9c6d68a5fdcad6600ae3a7b6b` |
| `parakeet_unified_encoder_int8.mlmodelc/coremldata.bin` | 492 | `54f533d30343d5e62b324a0691e4c262a6768b07b6e88e7aa14c617a2baba8a3` |
| `parakeet_unified_encoder_int8.mlmodelc/model.mil` | 1,110,902 | `c1c5d71c6cbf4d35bba08458746bde3640da7b1b444e1229a269393a58222c10` |
| `parakeet_unified_encoder_int8.mlmodelc/weights/weight.bin` | 595,051,904 | `f984b81590a4deae041ae20fbab8981c2d2a5b528b2ac81fae81c432633535c6` |
| `parakeet_unified_joint_decision_single_step.mlmodelc/analytics/coremldata.bin` | 243 | `163877ad14af97ec4107cd854fd1c6d336ee5d40ad25a657cc764fb763f452f5` |
| `parakeet_unified_joint_decision_single_step.mlmodelc/coremldata.bin` | 556 | `68a081570a48b52ec9379e153bd56748a5408a50be16767601563f231eaeff03` |
| `parakeet_unified_joint_decision_single_step.mlmodelc/model.mil` | 9,611 | `03c21096090bcd0b71c896c5ae0eb815db31a91c6676f572a7868eee4299abe3` |
| `parakeet_unified_joint_decision_single_step.mlmodelc/weights/weight.bin` | 3,446,978 | `06831afa6d1beb0c0b10350ebf7886bc37638e951d14e738d7e06fbd2a05012f` |

The compiled `.mlmodelc` graph set is a reviewed upstream artifact, not an
in-repository conversion result. The pinned FluidAudio source tree does not
contain a reproducible exporter for this published pack, so Parakeet does not
claim conversion reproducibility. Any replacement must include its conversion
code and toolchain lock, preserve the frontend/tokenizer/decoder seam, and pass
the gold corpus before its manifest can replace this one.

## Offline versus streaming

FluidAudio's [pinned benchmark](https://github.com/FluidInference/FluidAudio/blob/00a9aa771900ea09c485659663be31019e293e47/Documentation/Benchmarks.md#parakeet-unified-english-batch--streaming)
documents 2.15% average WER and 123.3x RTFx for batch versus 2.21% and
29.1x for streaming. The default streaming graph has a 70/13/13 chunk
configuration and 2.08 seconds of baked right-context latency; the offline
graph uses 15-second full attention, while streaming uses a 7.68-second
window. These are upstream comparisons rather than Parakeet's seven-file gold
results.

Parakeet therefore keeps offline inference with speculative decoding. In this
app that combination measured 182 ms p50 endpoint-to-transcript latency while
retaining the slightly more accurate and roughly 4.2x higher-throughput batch
graph. A streaming backend can be reconsidered when its user-visible latency,
quality, and resource use beat that full production pipeline.

## Compatibility and fallback

The native worker requires Apple Silicon and macOS 14 or later. Worker spawn,
model verification, Core ML load, or runtime failure returns to the existing
sherpa-onnx backend. A non-empty custom vocabulary also selects sherpa because
it owns contextual biasing. For diagnosis or emergency rollback, launch with
`PARAKEET_ASR_BACKEND=sherpa`; unset it or use `auto` to restore normal policy.
Unknown values are rejected instead of silently selecting a backend.

Power is not reported as a one-shot benchmark. `powermetrics` requires elevated
access, and its sampling interval is longer than these sub-second decodes, so
an automated single-utterance number would not be representative. A future
energy gate needs a controlled repeated workload and an explicit measurement
protocol.
