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

## No global contextual-vocabulary score is safe — 2026-08-11

Sherpa's shipping vocabulary preparation was evaluated with `IBM`, `Olly`, and
`Tactics`. A 27-point exploration covered score 0 through 50, including 0.25
steps across the transition region; six boundary rows were then repeated ten
times. Scores through 2.5 were transcript-identical to greedy. Modified beam
search alone cost 13.2% at score 0, and the default score 2 cost 13.9%.

At the first effect, score 2.75, `IBM` improves but noisy `Amy` becomes `80`:
WER moves 10.87% → 8.70% while CER worsens 5.46% → 6.09%, noisy and numbers
both regress, and p50 costs 14.2%. Score 4.5 produces `Olly` but injects `IBM`
into unrelated speech. Score 6 is broadly destructive at 42.39% WER; scores
8–50 measured 96.74–119.57% WER. All repeated rows were deterministic.

Do not raise the global score, claim contextual biasing is a quality win, or
start adapter/QAT/distillation work from this corpus. The seven fixtures were
used to locate score transitions and are no longer a blind test of future
adaptation. Full results and reopen gates are in
[`DOMAIN_ADAPTATION.md`](DOMAIN_ADAPTATION.md).

## Parent-only RSS is invalid for the Core ML backend — 2026-08-11

The first harness pass sampled only `getrusage(RUSAGE_SELF)`, which excluded the
resident FluidAudio worker and reported roughly 0.02 GiB. That result was
discarded. Report schema v2 introduced sampling of the Rust process plus the
backend's child worker through `proc_pidinfo`; the repeated published row is
0.10 GiB. Schema v3 retains that accounting and adds exact decoding/vocabulary
provenance.

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

## Qwen3-ASR q8 is not a production streaming replacement — 2026-08-11

Qwen3-ASR 0.6B q8 improved aggregate offline gold quality from shipping
Parakeet's 5.43% / 3.57% WER/CER to 4.35% / 3.15%, but that row is not a backend
win. Even with audio preparation excluded, it was 1.86× slower at corpus p50,
used about 11.0× the observed resident memory, and regressed the `noisy` and
`numbers` category baselines.

More importantly, actual incremental q8 decoding at the model's 2-second
boundaries produced 23.91% WER / 21.01% CER under exact, unpaced 100 ms
segmented, and jittered transport schedules. The 14.225 s human fixture was
exact offline and 41.86% WER streaming because middle spans disappeared and an
earlier phrase repeated. The schedules' identical text rules out outer
segmentation as the cause; they do not model real-time queueing.

q4 does not rescue the candidate: it has only 17.2% more throughput than q8
while WER rises to 6.52%. A native `mlx-rs` 0.25.3 spike also failed before linking
because Xcode's separately downloaded Metal Toolchain was absent, and the
crate raises the repo MSRV from 1.77 to 1.82. Do not add Qwen, Python, MLX, a
Core ML conversion project, or a Candle architecture port unless a new native
runtime first changes these quality/resource facts. See
[`QWEN3_ASR_EVALUATION.md`](QWEN3_ASR_EVALUATION.md).
