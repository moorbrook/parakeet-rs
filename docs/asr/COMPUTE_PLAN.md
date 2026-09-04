# Core ML compute plan for the Parakeet Unified models

What Core ML actually schedules where, per operation, for the three models the
shipping backend loads. Measured on M5 Pro 24 GiB, macOS 26.5.1, model revision
`4252711f6f060f9a2f91e5f081a806d7f45eebd8` (see [`COREML_MODEL.md`](COREML_MODEL.md)).

`MLComputePlan` reports, for every operation in a compiled ML Program, the
device Core ML prefers and the devices the operation could run on, plus an
estimated relative cost. `native/ParakeetCoreMLWorker` builds
`parakeet-compute-plan` for this; it is a diagnostic and links no FluidAudio
code, so it cannot perturb the shipping path.

```bash
swift build --package-path native/ParakeetCoreMLWorker -c release \
    --product parakeet-compute-plan
M="$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-unified-en-0.6b"
B=native/ParakeetCoreMLWorker/.build/arm64-apple-macosx/release
./$B/parakeet-compute-plan --compute-units cpu-and-neural-engine \
    "$M/parakeet_unified_encoder_int8.mlmodelc"
./$B/parakeet-compute-plan --compute-units cpu-only "$M/parakeet_unified_decoder.mlmodelc"
./$B/parakeet-compute-plan --compute-units cpu-only \
    "$M/parakeet_unified_joint_decision_single_step.mlmodelc"
```

`--format csv` emits one row per operation instead of the summary.

## Summary

| model | requested units | scheduled ops | ANE | CPU | GPU | unplaced | ANE cost share |
|---|---|---:|---:|---:|---:|---:|---:|
| `parakeet_unified_encoder_int8` | `cpu-and-neural-engine` | 1722 | **1400** | 28 | 0 | 294 | 99.8% |
| `parakeet_unified_decoder` | `cpu-only` | 25 | 0 | 24 | 0 | 1 | — |
| `parakeet_unified_joint_decision_single_step` | `cpu-only` | 21 | 0 | 21 | 0 | 0 | — |

"Unplaced" operations are `ios16.constexpr_affine_dequantize` (and one
`identity`): int8 weight decompression that Core ML folds into the consuming
operation rather than scheduling, so the plan reports no device for them.
Cost shares are Core ML's own estimated weights, renormalized over the
operations that carry a nonzero weight.

The decoder and joint were also planned under `cpu-and-neural-engine`. Every
operation still came back CPU, so FluidAudio's decision to build them a
`cpuOnly` configuration costs nothing — Core ML would make the same choice.

## Encoder: no fallback inside the network

Requested `cpu-and-neural-engine`, int8 offline encoder, mel `[1, 128, 1501]` →
encoder `[1, 1024, 188]`.

| operator | ANE | CPU |
|---|---:|---:|
| `ios17.linear` | 193 | 0 |
| `ios17.add` | 174 | 3 |
| `ios17.transpose` | 170 | 2 |
| `ios17.reshape` | 145 | 0 |
| `ios17.layer_norm` | 120 | 0 |
| `ios17.mul` | 106 | 0 |
| `ios17.conv` | 77 | 0 |
| `ios16.silu` | 72 | 0 |
| `ios17.matmul` | 72 | 0 |
| `select` | 72 | 0 |
| `ios17.slice_by_index` | 48 | 0 |
| `pad` | 48 | 0 |
| `ios16.sigmoid` | 24 | 0 |
| `ios16.softmax` | 24 | 0 |
| `split` | 24 | 0 |
| `ios17.expand_dims` | 10 | 7 |
| `tile` | 7 | 1 |
| `ios17.cast` | 5 | 6 |
| `ios16.relu` | 3 | 0 |
| `ios17.floor_div` | 2 | 1 |
| `ios17.logical_not` | 2 | 0 |
| `ios17.sub` | 2 | 1 |
| `ios17.less` | 0 | 5 |
| `ios17.logical_and` | 0 | 2 |

All 28 CPU operations sit between `main.4` and `main.173`, ahead of the first
conformer layer (layer 0's first weight is `main.181`). They are the padding
mask chain the encoder derives from its `mel_length` input, plus the input cast
and transpose:

| op index | operator | output | preferred | supported | estimated cost |
|---|---|---|---|---|---:|
| main.4 | `ios17.cast` | `mel_to_fp16` | CPU | ANE+CPU | 0.00046 |
| main.5 | `ios17.transpose` | `x_1_cast_fp16` | CPU | ANE+CPU | 0.00031 |
| main.6 | `ios17.expand_dims` | `tensor_1_cast_fp16` | CPU | ANE+CPU | 0.00031 |
| main.9 | `ios17.expand_dims` | `var_101` | CPU | CPU | 0.00000 |
| main.10 | `ios17.less` | `time_mask_1` | CPU | CPU | 0.00000 |
| main.30 | `ios17.cast` | `mel_length_to_fp16` | CPU | CPU | 0.00000 |
| main.31 | `ios17.add` | `var_123_cast_fp16` | CPU | ANE+CPU | 0.00000 |
| main.33 | `ios17.add` | `var_125_cast_fp16` | CPU | ANE+CPU | 0.00000 |
| main.35 | `ios17.sub` | `var_127_cast_fp16` | CPU | ANE+CPU | 0.00000 |
| main.37 | `ios17.floor_div` | `floor_div_0_cast_fp16` | CPU | ANE+CPU | 0.00000 |
| main.39 | `ios17.add` | `current_lengths_3_cast_fp16` | CPU | ANE+CPU | 0.00000 |
| main.43 | `ios17.cast` | `current_lengths_3_cast_fp16_to_int32` | CPU | CPU | 0.00000 |
| main.44 | `ios17.expand_dims` | `var_138` | CPU | CPU | 0.00000 |
| main.45 | `ios17.less` | `time_mask_3` | CPU | CPU | 0.00000 |
| main.80 | `ios17.cast` | `current_lengths_5_cast_fp16_to_int32` | CPU | CPU | 0.00000 |
| main.81 | `ios17.expand_dims` | `var_184` | CPU | CPU | 0.00000 |
| main.82 | `ios17.less` | `time_mask_5` | CPU | CPU | 0.00000 |
| main.126 | `ios17.cast` | `current_lengths_cast_fp16_to_int32` | CPU | CPU | 0.00000 |
| main.127 | `ios17.expand_dims` | `var_245` | CPU | CPU | 0.00000 |
| main.128 | `ios17.less` | `time_mask` | CPU | CPU | 0.00000 |
| main.162 | `ios17.cast` | `encoder_length` | CPU | CPU | 0.00000 |
| main.163 | `ios17.expand_dims` | `var_330` | CPU | CPU | 0.00000 |
| main.164 | `ios17.less` | `pad_mask_1` | CPU | CPU | 0.00000 |
| main.166 | `ios17.expand_dims` | `var_332` | CPU | ANE+CPU | 0.00000 |
| main.168 | `tile` | `pad_mask_for_att_mask_1` | CPU | ANE+CPU | 0.00001 |
| main.170 | `ios17.transpose` | `var_335` | CPU | ANE+CPU | 0.00003 |
| main.171 | `ios17.logical_and` | `pad_mask_for_att_mask` | CPU | CPU | 0.00004 |
| main.173 | `ios17.logical_and` | `att_mask` | CPU | CPU | 0.00004 |

Eight of the 28 are CPU-only because no engine path exists for that operator
shape (`less`, `logical_and`, and the int32 casts feeding them). The other 20
could run on the engine and Core ML chose not to, which is the expected
treatment for a scalar-derived mask: moving three-element bookkeeping onto the
engine would cost a dispatch each. Together they are 0.12% of the encoder's
estimated cost against 99.8% for the ANE operations.

So the claim the issue set out to test holds for this graph. Under
`cpuAndNeuralEngine` the encoder has no per-op CPU fallback anywhere inside its
24 conformer layers — every `conv`, `linear`, `matmul`, `layer_norm`, `softmax`
and `silu` is on the engine. The measured 26 ms versus 86 ms when the encoder is
forced to `cpu-only` (bench/README.md) is the behavioural confirmation.

## Decoder: CPU by necessity, not by configuration

`parakeet_unified_decoder` is the RNNT prediction network: an embedding gather
and a two-layer LSTM.

| operator | CPU |
|---|---:|
| `ios17.cast` | 8 |
| `ios17.squeeze` | 4 |
| `ios17.lstm` | 2 |
| `split` | 2 |
| `stack` | 2 |
| `ios17.transpose` | 2 |
| `ios17.add` | 1 |
| `ios17.gather` | 1 |
| `ios17.greater_equal` | 1 |
| `select` | 1 |

The two `ios17.lstm` operations are 98.9% of the estimated cost (0.49453 each)
and the embedding `gather` is 1.0%. Every operation reports `supported: CPU`
alone even when the plan is built with `cpu-and-neural-engine`, which matches
arXiv 2606.22283 Table A.12: `lstm` has no engine path on any Apple silicon
family through M5, note "unroll on host". This is the fact that
[`reports/ane-direct/ASSESSMENT.md`](../../reports/ane-direct/ASSESSMENT.md) §4
predicted, now read off the plan rather than inferred.

## Joint decision: CPU, three small matrix multiplies

`parakeet_unified_joint_decision_single_step` projects one encoder frame and one
decoder state to the vocabulary and returns the argmax.

| operator | CPU |
|---|---:|
| `ios17.cast` | 6 |
| `ios17.expand_dims` | 3 |
| `ios17.linear` | 3 |
| `ios17.transpose` | 2 |
| `ios16.relu` | 1 |
| `ios16.softmax` | 1 |
| `ios17.add` | 1 |
| `ios17.gather_along_axis` | 1 |
| `ios17.reduce_argmax` | 1 |
| `ios17.squeeze` | 1 |
| `ios17.topk` | 1 |

The three `linear` operations are 98.9% of the estimated cost: the encoder
projection (0.37646), the decoder projection (0.23542), and the vocabulary
projection (0.37683). Every operation is CPU-only supported. That is consistent
with the measured 100 µs per joint dispatch: at these shapes
(`[1, 1024, 1]` and `[1, 640, 1]` in, 1025 logits out) the arithmetic is small
next to a dispatch, and this model is called once per decoded frame plus once
per emitted token.

## Encoder cost against compiled mel length

The encoder is compiled at one fixed shape and `UnifiedAsrManager` zero-pads
every utterance to it, so a one-word utterance runs the whole 1501-frame graph.
The question bucketing turns on is whether that cost is arithmetic, which a
shorter compiled window would cut, or weight streaming, which it would not: the
int8 encoder's weight blob is 595 MB.

`parakeet-encoder-probe` loads a compiled encoder, reads its declared shapes,
and times predictions on zero-filled inputs with `mel_length` set to the full
window. Comparing the 15 s offline encoder against the `70_13_13` streaming
export, which carries the same int8 weights at a 769-frame window:

| encoder | mel frames | encoder frames | weights | load | predict p50 |
|---|---:|---:|---:|---:|---:|
| `parakeet_unified_encoder_int8` | 1501 | 188 | 595 MB | 11.4 s | **26.11 ms** |
| `parakeet_unified_encoder_streaming_70_13_13_int8` | 769 | 97 | 591 MB | 8.9 s | **12.84 ms** |

Ten measured predictions after three warmups, `cpu-and-neural-engine`, min-max
spread under 0.3 ms in both rows. Load times are a cold Core ML plan compile.

A 1.95× frame ratio gives a 2.03× time ratio. The cost is arithmetic and close
to linear in the compiled window; there is no weight-streaming floor visible at
769 frames, which puts an upper bound of roughly 1 ms on the resident-weight
term. Extrapolating the line, a 4 s window (401 mel frames) would cost about
7 ms and a 2 s window about 3.5 ms, against 25.5 ms today.

Two caveats on that extrapolation. The streaming export's attention mask is
baked to a `[70 | 13 | 13]` chunk rather than full attention, so it does less
attention work than a full-attention encoder compiled at 769 frames would; a
real short-window offline encoder will land above this line, not below. And the
offline row was measured while the machine was at 79% user CPU, yet it still
reproduced the quiet-machine 25.5 ms from `bench/README.md`, which is evidence
the ANE path is not competing for CPU.

