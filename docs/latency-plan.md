# Latency verification

Recognition runs in a resident Parakeet Unified Core ML worker. Capture
resamples once to 16 kHz. Tap endpointing uses Silero's 32 ms Core ML graph
in dedicated session-local processes. Candidate and confirming detectors keep
their independent pause windows; only the confirming detector can stop Tap.
Hold releases on the hotkey and can decode windows during capture.

The native VAD graph uses FluidAudio's explicit recurrent state. The 256 ms
graph is unsuitable for the unchanged short-pause latency policy. The pinned
32 ms graph accepts one 512-sample frame plus 64 context samples.

## Commands

Build the worker and benchmark executables before measuring. Check competing
CPU and Neural Engine activity, and run only one audio benchmark at a time.

```bash
scripts/build-coreml-worker.sh
cargo build --release --locked --bin bench_asr --bin bench_e2e --bin asr_diff
scripts/bench-latency.sh
scripts/bench-endpoint-policy.sh
scripts/bench-hold.sh
COREML_WORKER=target/release/parakeet-coreml-worker scripts/bench-gold.sh
```

Record the fixture, head SHA, load conditions, repetition count, failures, and
percentiles with each result. An endpoint run must preserve the complete
reviewed utterance and remain below the unchanged 1000 ms p95 limit. Hold
comparisons use release-to-transcript time, and quality comparisons use literal
transcripts as well as gold WER and CER. See [bench/README.md](../bench/README.md).
