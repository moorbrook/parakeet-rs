# ASR negative-evidence ledger

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
