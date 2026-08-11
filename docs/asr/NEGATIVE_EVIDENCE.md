# ASR negative-evidence ledger

## The short-utterance floor is not IPC — 2026-08-11

The 35 ms one-second Core ML result was initially read as roughly 30 ms of
fixed worker overhead because the five-second result was 66 ms. Direct paired
instrumentation rejects that interpretation. Over 30 release repetitions the
one-second call measured 35.112 ms p50 inside the worker and 35.242 ms at the
Rust boundary: only 0.128 ms p50 / 0.143 ms p95 was outside resampling and
inference. Even at 20 seconds the difference was 1.111 ms p50. Do not justify
shared memory, Mach ports, feature transfer, or an in-process Core ML rewrite
with the old 30 ms estimate.

## Sherpa token parity is not a valid Core ML gate — 2026-08-11

The proposed deepest-seam shortcut was to compare emitted token IDs against a
captured sherpa reference before comparing text. The two evaluated backends do
not share that seam: FluidAudio Unified uses the pinned 1,024-entry
`vocab.json`, while the sherpa fallback's Parakeet TDT v3 tokenizer includes a
different inventory and control-token scheme in `tokens.txt`. Equal speech can
therefore produce different, individually correct token streams. Treating
their IDs or pieces as equal would reject a valid independent checkpoint and
would not prove encoder/logit parity.

Token- or tensor-level parity remains required for a future conversion of the
*same* generic checkpoint, but its reference must come from that checkpoint's
own unfused runtime. The currently pinned publisher did not ship such an
exporter/reference route, so this closeout truthfully gates transcript WER/CER,
repeatability, exact artifact identity, and backend timing instead.

## Contextual vocabulary did not improve this corpus — 2026-08-11

Sherpa's shipping vocabulary preparation was evaluated with `IBM`, `Olly`, and
`Tactics`, hotword score 2, and modified beam search. Across ten repeats it
produced the same hypotheses as greedy sherpa: 10.87% WER and 5.46% CER. Corpus
decode p50 regressed from 2.764 s to 3.126 s (13.1%) and both rows failed the
8% / 5% absolute gate. Do not cite vocabulary support as the stronger quality
baseline for these fixtures unless the tokenization or scoring policy changes.

## Parent-only RSS is invalid for the Core ML backend — 2026-08-11

The first harness pass sampled only `getrusage(RUSAGE_SELF)`, which excluded the
resident FluidAudio worker and reported roughly 0.02 GiB. That result was
discarded. Report schema v2 samples the Rust process plus the backend's child
worker through `proc_pidinfo`; the repeated published row is 0.10 GiB.

## Core ML compute-unit labels are candidates, not speed claims — 2026-08-11

On Apple M5 Pro / macOS 26.5.1, `.all` did not beat explicit CPU+ANE: the
combined short-corpus p50 was 315.53 versus 316.64 ms, and the 14.225 s long
p50 was 132.10 versus 132.87 ms. After load and first-decode warmup, the session
score improvement is still below the 5% floor, so CPU+ANE wins the near-tie.

CPU-only loaded 10.2 ms faster at median but used 1.26 GiB instead of 0.10 GiB,
and was 2.15× / 1.46× slower in the short/long buckets. Its process-first
warmup varied from 262 ms to 3.35 s as Core ML's persistent plan cache changed.
CPU+GPU aborted on its first warm decode inside Apple's MPSGraph with
`MLIR pass manager failed`; the tuner isolated the crash to its child worker
and recorded the candidate as failed. Do not infer performance from a compute
unit name or retry CPU+GPU in production without a new macOS/Core ML result and
the full quality gate.
