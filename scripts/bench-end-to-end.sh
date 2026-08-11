#!/usr/bin/env bash
# Matched live-audio end-of-speech benchmark and 3x regression gate.
#
# Replays the representative five-second fixture through a named duplex
# Core Audio loopback. Both variants use the production capture/session path:
#   baseline:  sherpa recognizer after the unchanged serial VAD endpoint
#   optimized: resident Core ML recognizer overlapped with that same endpoint
#
# BlackHole is selected explicitly. The script never changes system defaults.

set -euo pipefail

cd "$(dirname "$0")/.."

REPS="${REPS:-30}"
WARMUP_REPS="${WARMUP_REPS:-2}"
DEVICE="${DEVICE:-BlackHole 2ch}"
WAV="${WAV:-bench/audio/5s_48000.wav}"
BASELINE_LOG="${BASELINE_LOG:-bench/e2e-serial.log}"
OPTIMIZED_LOG="${OPTIMIZED_LOG:-bench/e2e-speculative.log}"
BASELINE_CSV="${BASELINE_CSV:-bench/e2e-serial.csv}"
OPTIMIZED_CSV="${OPTIMIZED_CSV:-bench/e2e-speculative.csv}"
EXPECTED="The latency benchmark measures end to end speech recognition pipeline performance."

command -v uv >/dev/null || { echo "missing uv" >&2; exit 1; }
[[ -f "$WAV" ]] || { echo "missing fixture: $WAV" >&2; exit 1; }

scripts/build-coreml-worker.sh
cargo build --release --bin bench_e2e

echo "Baseline: sherpa + serial endpoint ($REPS measured repetitions)"
RUST_LOG=info ./target/release/bench_e2e \
    --backend sherpa \
    --strategy serial \
    --endpoint-policy fast \
    --device "$DEVICE" \
    --wav "$WAV" \
    --expected "$EXPECTED" \
    --warmup-reps "$WARMUP_REPS" \
    --reps "$REPS" \
    2>"$BASELINE_LOG"

echo "Optimized: Core ML + speculative decode ($REPS measured repetitions)"
RUST_LOG=info ./target/release/bench_e2e \
    --backend coreml-unified \
    --strategy speculative \
    --endpoint-policy fast \
    --device "$DEVICE" \
    --wav "$WAV" \
    --expected "$EXPECTED" \
    --warmup-reps "$WARMUP_REPS" \
    --reps "$REPS" \
    2>"$OPTIMIZED_LOG"

uv run --quiet scripts/bench-aggregate.py \
    --metric end-to-end --log "$BASELINE_LOG" --out "$BASELINE_CSV"
uv run --quiet scripts/bench-aggregate.py \
    --metric end-to-end --log "$OPTIMIZED_LOG" --out "$OPTIMIZED_CSV"

baseline_p50=$(awk -F, '$2 == 5 { print $5 }' "$BASELINE_CSV")
baseline_p95=$(awk -F, '$2 == 5 { print $6 }' "$BASELINE_CSV")
optimized_p50=$(awk -F, '$2 == 5 { print $5 }' "$OPTIMIZED_CSV")
optimized_p95=$(awk -F, '$2 == 5 { print $6 }' "$OPTIMIZED_CSV")

for value in "$baseline_p50" "$baseline_p95" "$optimized_p50" "$optimized_p95"; do
    [[ -n "$value" ]] || { echo "missing five-second benchmark row" >&2; exit 1; }
done

p50_speedup=$(awk -v old="$baseline_p50" -v new="$optimized_p50" \
    'BEGIN { printf "%.2f", old / new }')
p95_speedup=$(awk -v old="$baseline_p95" -v new="$optimized_p95" \
    'BEGIN { printf "%.2f", old / new }')

echo
echo "End-of-speech -> transcript-ready"
echo "  p50: ${baseline_p50} ms -> ${optimized_p50} ms (${p50_speedup}x)"
echo "  p95: ${baseline_p95} ms -> ${optimized_p95} ms (${p95_speedup}x)"

awk -v speedup="$p50_speedup" 'BEGIN { exit !(speedup >= 3.0) }' || {
    echo "FAIL: p50 speedup ${p50_speedup}x is below 3.0x" >&2
    exit 1
}
awk -v speedup="$p95_speedup" 'BEGIN { exit !(speedup >= 3.0) }' || {
    echo "FAIL: p95 speedup ${p95_speedup}x is below 3.0x" >&2
    exit 1
}

echo "PASS: matched p50 and p95 both exceed 3.0x"
