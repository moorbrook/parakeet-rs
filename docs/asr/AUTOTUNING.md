# Apple Silicon ASR runtime-plan tuning

Parakeet ships a generic, reviewed Parakeet Unified Core ML model. Tuning does
not rewrite its weights: it measures which bounded Core ML execution plan is
best for the current Mac, macOS release, and exact model manifest, while the
checked-in human-speech gold set prevents a latency-only winner from shipping.

## Run the full tuner

```bash
scripts/build-coreml-worker.sh
cargo run --release --locked --bin tune_asr -- \
  --repetitions 10 --load-repetitions 3
```

The four allowed plans are `cpu-and-neural-engine` (the safe baseline), `all`,
`cpu-and-gpu`, and `cpu-only`. Each working candidate must:

- pass the manifest's absolute and baseline WER/CER limits;
- produce no nondeterministic outputs across repetitions;
- be no worse than CPU+ANE in every scored category;
- stay within 1.25 times the baseline peak resident set; and
- improve the median-load-plus-first-decode-warmup-plus-20-utterance score by
  at least 5% to replace the baseline.

Short and long utterances are selected independently at the fixed eight-second
boundary. The boundary is constrained to 1–60 seconds in both the Rust client
and Swift worker. A profile always contains the baseline, including its quality
and performance evidence; a failed challenger remains visible rather than
being omitted.

## Profile lifecycle

The atomic JSON profile lives at:

```text
~/Library/Application Support/com.parakeet.rs/asr-tuning-profile.json
```

Inspect or remove it with:

```bash
cargo run --release --locked --bin tune_asr -- --show-profile
cargo run --release --locked --bin tune_asr -- --remove-profile
```

Its cache key covers chip identity, architecture, physical memory, logical and
named performance levels, macOS version, backend identity, the complete pinned
Core ML artifact-manifest digest, and tuner version. App startup first verifies
the actual model files, then accepts the profile only if its cache key,
fingerprint, baseline, and selection recomputed from the saved evidence all
match. Missing, unreadable, stale, or invalid profiles use CPU+ANE.

`PARAKEET_ASR_TUNING=off` bypasses a valid profile for diagnosis without
deleting it. Unset it or use `auto` to restore automatic selection.

## M5 Pro result — 2026-08-11

Machine: Apple M5 Pro, 24 GiB, 15 logical CPUs (`Super`: 5,
`Performance`: 10), arm64, macOS 26.5.1. Ten measured passes over 34.05 seconds
of checked-in human speech selected CPU+ANE for both regimes. Full evidence is
recorded in [PERF.md](PERF.md); losing and failed routes remain in
[NEGATIVE_EVIDENCE.md](NEGATIVE_EVIDENCE.md).
