#!/usr/bin/env bash
# scripts/bench-latency.sh — produce per-length p50/p95/p99 ASR-decode CSV.
#
# Drives `bench_asr` against generated WAV fixtures at {1, 3, 5, 10, 20}s,
# REPS reps each (default 30), captures `phase_timer` log lines, and
# reduces them via scripts/bench-aggregate.py.
#
# Required:
#   - macOS `say` + `afconvert` (preinstalled).
#   - Model downloaded into ~/Library/Application Support/com.parakeet.rs/
#     (launch Parakeet.app once to fetch it).
#   - `uv` on PATH for the aggregator (`brew install uv`).
#
# Outputs:
#   bench/audio/{1,3,5,10,20}s.wav     — synthesized once, kept on rerun.
#   bench/raw.log                       — every iteration's phase_timer line.
#   bench/baseline.csv (or $OUT_CSV)    — per-mode-per-length percentile table.
#
# Usage:
#   scripts/bench-latency.sh                   # defaults
#   REPS=50 scripts/bench-latency.sh           # more reps per length
#   OUT_CSV=bench/post-coreml-cache.csv \
#       scripts/bench-latency.sh               # name the output (for §2 re-bench)

set -euo pipefail

cd "$(dirname "$0")/.."

REPS="${REPS:-30}"
WARMUP_REPS="${WARMUP_REPS:-3}"
# Match the typical macOS default-input sample rate so the bench exercises
# the same sherpa-onnx resample path the live app pays.
SAMPLE_RATE="${SAMPLE_RATE:-48000}"
LENGTHS=(1 3 5 10 20)
WAV_DIR="bench/audio"
RAW_LOG="bench/raw.log"
OUT_CSV="${OUT_CSV:-bench/baseline.csv}"

# Texts sized to land near the target duration when fed through macOS `say`
# at its default rate (~200 wpm ≈ 3.3 wps). Exact durations don't matter —
# the aggregator buckets by *measured* audio_s (nearest of {1,3,5,10,20}).
#
# Function lookup rather than `declare -A` since macOS' system bash is 3.2
# and lacks associative arrays.
text_for_length() {
    case "$1" in
        1)  echo "Quick test." ;;
        3)  echo "The quick brown fox jumps over the lazy dog." ;;
        5)  echo "The latency benchmark measures end to end speech recognition pipeline performance." ;;
        10) echo "The latency benchmark drives the recognizer with audio of fixed lengths, recording timing for thirty iterations and aggregating the result." ;;
        20) echo "The latency benchmark drives the recognizer with audio of fixed lengths, recording timing for thirty iterations and aggregating the result into percentiles. Then it writes them to a CSV file for inclusion in the architecture decision record under the latency plan section." ;;
        *)  echo "" ;;
    esac
}

command -v say        >/dev/null || { echo "missing macOS \`say\`"; exit 1; }
command -v afconvert  >/dev/null || { echo "missing macOS \`afconvert\`"; exit 1; }
command -v uv         >/dev/null || { echo "missing \`uv\` (brew install uv)"; exit 1; }

mkdir -p "$WAV_DIR" "$(dirname "$RAW_LOG")"

# Step 1 — synthesize WAV fixtures (idempotent: regenerated only if missing).
for len in "${LENGTHS[@]}"; do
    wav="$WAV_DIR/${len}s_${SAMPLE_RATE}.wav"
    if [[ ! -f "$wav" ]]; then
        echo "Generating $wav (${SAMPLE_RATE} Hz mono PCM16)…"
        text="$(text_for_length "$len")"
        aiff="$WAV_DIR/${len}s.aiff"
        say -o "$aiff" "$text"
        afconvert -f WAVE -d "LEI16@${SAMPLE_RATE}" -c 1 "$aiff" "$wav"
        rm "$aiff"
    fi
done

# Step 2 — release build (debug numbers are useless for latency comparison).
echo "Building bench_asr (release)…"
cargo build --release --bin bench_asr 2>&1 | tail -3

# Step 3 — drive the harness per length, appending raw phase_timer lines.
: > "$RAW_LOG"
BENCH_BIN="./target/release/bench_asr"
for len in "${LENGTHS[@]}"; do
    wav="$WAV_DIR/${len}s_${SAMPLE_RATE}.wav"
    echo "Benching $wav (warmup=$WARMUP_REPS, reps=$REPS)…"
    "$BENCH_BIN" --wav "$wav" --reps "$REPS" --warmup-reps "$WARMUP_REPS" \
        2>>"$RAW_LOG" \
        || echo "  ↑ bench failed for $wav (see $RAW_LOG)"
done

# Step 4 — aggregate to CSV.
uv run --quiet scripts/bench-aggregate.py --log "$RAW_LOG" --out "$OUT_CSV"

echo
echo "Wrote $OUT_CSV"
cat "$OUT_CSV"
