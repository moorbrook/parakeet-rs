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
#   bench/*-stages.csv                  — per-stage breakdown (Core ML backends only).
#
# Usage:
#   scripts/bench-latency.sh                   # defaults
#   REPS=50 scripts/bench-latency.sh           # more reps per length
#   BACKEND=coreml-unified \
#       OUT_CSV=bench/coreml-unified.csv \
#       scripts/bench-latency.sh                # shipping native Core ML backend
#   BACKEND=coreml-unified RNNT_ENGINE=coreml \
#       MODEL_DIR=~/.cache/parakeet-bucket-encoders/model-dir-buckets-2-5-8 \
#       scripts/bench-latency.sh                # bucketed encoders, CoreML decode loop
#   BACKEND=coreml-tdt-v3 \
#       OUT_CSV=bench/coreml-tdt-v3.csv \
#       MODEL_DIR="$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-tdt-0.6b-v3" \
#       scripts/bench-latency.sh                # the TDT v3 challenger (kata f0zg)
#   OUT_CSV=bench/experiment.csv \
#       scripts/bench-latency.sh                # name an experiment output

set -euo pipefail

cd "$(dirname "$0")/.."

REPS="${REPS:-30}"
WARMUP_REPS="${WARMUP_REPS:-3}"
# Fixtures stay at the typical macOS default-input rate so these runs remain
# comparable with the published baselines. Since ADR-0030 `bench_asr` converts
# the fixture to 16 kHz once at load, outside the measured loop, which is what
# production capture now hands the recognizer.
SAMPLE_RATE="${SAMPLE_RATE:-48000}"
LENGTHS=(1 3 5 10 20)
WAV_DIR="bench/audio"
RAW_LOG="bench/raw.log"
OUT_CSV="${OUT_CSV:-bench/baseline.csv}"
BACKEND="${BACKEND:-sherpa}"
# The native worker can report where its decode time went (resample, mel,
# encoder, RNNT loop) plus its Core ML dispatch counts. sherpa has no
# equivalent seam, so the flag is only passed to the native backends.
STAGE_TIMINGS="${STAGE_TIMINGS:-1}"
# Model directory for the native backend. Unset means the worker's own default,
# which holds only the 15 s encoder; point this at a directory carrying bucket
# encoders to measure the bucketed path.
MODEL_DIR="${MODEL_DIR:-}"
# Which implementation of the RNNT decode loop the worker runs: `native` reads
# the decoder and joint weights and runs them in process, `coreml` keeps
# FluidAudio's dispatch per step. Unified only; the TDT path has no native
# decode loop.
RNNT_ENGINE="${RNNT_ENGINE:-native}"
# Extra flags appended verbatim to every bench_asr invocation, for arms that
# differ only by a backend knob (e.g. --tdt-decode-compute-units cpu-only).
# Word-split on purpose; keep the values shell-safe.
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$BACKEND" in
    sherpa|coreml-unified|coreml-tdt-v3) ;;
    *)
        echo "unknown BACKEND=$BACKEND (expected sherpa, coreml-unified, or coreml-tdt-v3)" >&2
        exit 2
        ;;
esac

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
if [[ "$BACKEND" != "sherpa" ]]; then
    echo "Building native Core ML worker…"
    scripts/build-coreml-worker.sh
fi
echo "Building bench_asr (release)…"
cargo build --release --bin bench_asr 2>&1 | tail -3

# Step 3 — drive the harness per length, appending raw phase_timer lines.
: > "$RAW_LOG"
BENCH_BIN="./target/release/bench_asr"
for len in "${LENGTHS[@]}"; do
    wav="$WAV_DIR/${len}s_${SAMPLE_RATE}.wav"
    echo "Benching $wav (backend=$BACKEND, warmup=$WARMUP_REPS, reps=$REPS)…"
    stage_args=()
    if [[ "$BACKEND" == "coreml-unified" ]]; then
        if [[ "$STAGE_TIMINGS" != "0" ]]; then
            stage_args=(--stage-timings)
        fi
        stage_args+=(--rnnt-engine "$RNNT_ENGINE")
        if [[ -n "$MODEL_DIR" ]]; then
            stage_args+=(--model-dir "$MODEL_DIR")
        fi
    elif [[ "$BACKEND" == "coreml-tdt-v3" ]]; then
        # No --rnnt-engine: the native decode loop is a Unified-only path.
        if [[ "$STAGE_TIMINGS" != "0" ]]; then
            stage_args=(--stage-timings)
        fi
        if [[ -n "$MODEL_DIR" ]]; then
            stage_args+=(--model-dir "$MODEL_DIR")
        fi
    fi
    "$BENCH_BIN" --backend "$BACKEND" --wav "$wav" \
        --reps "$REPS" --warmup-reps "$WARMUP_REPS" "${stage_args[@]+"${stage_args[@]}"}" \
        ${EXTRA_ARGS} \
        2>>"$RAW_LOG" \
        || echo "  ↑ bench failed for $wav (see $RAW_LOG)"
done

# Step 4 — aggregate to CSV.
uv run --quiet scripts/bench-aggregate.py --log "$RAW_LOG" --out "$OUT_CSV"
BOUNDARY_CSV="${OUT_CSV%.csv}-boundary.csv"
uv run --quiet scripts/bench-boundary.py --log "$RAW_LOG" --out "$BOUNDARY_CSV"

echo
echo "Wrote $OUT_CSV"
cat "$OUT_CSV"
echo
echo "Wrote $BOUNDARY_CSV"
cat "$BOUNDARY_CSV"

if [[ "$BACKEND" != "sherpa" && "$STAGE_TIMINGS" != "0" ]]; then
    STAGES_CSV="${OUT_CSV%.csv}-stages.csv"
    uv run --quiet scripts/bench-stages.py --log "$RAW_LOG" --out "$STAGES_CSV"
    echo
    echo "Wrote $STAGES_CSV"
    cat "$STAGES_CSV"
fi
