#!/usr/bin/env bash
# Real-human-speech false-stop + final-pause latency gate for Long-form Tap.

set -euo pipefail

cd "$(dirname "$0")/.."

REPS="${REPS:-30}"
WARMUP_REPS="${WARMUP_REPS:-2}"
DEVICE="${DEVICE:-BlackHole 2ch}"
FIXTURE_DIR="bench/endpointing"
OUT_DIR="${OUT_DIR:-bench}"

command -v uv >/dev/null || { echo "missing uv" >&2; exit 1; }
mkdir -p "$OUT_DIR"

scripts/build-coreml-worker.sh
cargo build --release --locked --bin bench_e2e

run_fixture() {
    local name="$1"
    local wav="$2"
    local log="$OUT_DIR/endpoint-${name}.log"
    local csv="$OUT_DIR/endpoint-${name}.csv"

    [[ -f "$wav" ]] || { echo "missing fixture: $wav" >&2; exit 1; }
    echo "Long-form endpoint: $name ($REPS measured repetitions)"
    RUST_LOG=info ./target/release/bench_e2e \
        --backend coreml-unified \
        --strategy speculative \
        --endpoint-policy long-form \
        --device "$DEVICE" \
        --wav "$wav" \
        --warmup-reps "$WARMUP_REPS" \
        --reps "$REPS" \
        2>"$log"

    uv run --quiet scripts/bench-aggregate.py \
        --metric end-to-end --log "$log" --out "$csv"

    local measured
    local p50
    local p95
    measured=$(awk -F, 'NR == 2 { print $3 }' "$csv")
    p50=$(awk -F, 'NR == 2 { print $5 }' "$csv")
    p95=$(awk -F, 'NR == 2 { print $6 }' "$csv")
    [[ "$measured" == "$REPS" ]] || {
        echo "FAIL: $name emitted $measured/$REPS measured endpoints" >&2
        exit 1
    }
    awk -v latency="$p95" 'BEGIN { exit !(latency < 1000.0) }' || {
        echo "FAIL: $name p95 ${p95}ms is not below 1000ms" >&2
        exit 1
    }
    printf '%-18s false_stops=0/%s  p50=%sms  p95=%sms\n' \
        "$name" "$REPS" "$p50" "$p95"
}

run_fixture \
    single_sentence \
    "$FIXTURE_DIR/librispeech-single-6930-75918-0000-48000.wav"
run_fixture \
    multi_sentence \
    "$FIXTURE_DIR/librispeech-multi-6930-75918-0001-48000.wav"

echo "PASS: zero false stops and p95 final-pause latency below 1000ms"
