#!/usr/bin/env bash
# scripts/bench-idle.sh — ANE idle re-wake A/B (kata snx0).
#
# The Neural Engine hard power-gates when idle, so the first dispatch after a
# pause may pay a re-wake the user feels as endpoint latency. This harness
# measures four arms against each other at 1 s and 5 s:
#
#   warm     repetitions back to back — the floor
#   cold     an idle gap before each one, and nothing to keep the engine up
#   prime    one dispatch at the hotkey-down edge
#   cadence  a keep-alive dispatch every KEEPALIVE_MS until the endpoint
#
# Phases (first positional argument):
#   sweep   Find the cool-down knee: the 1 s fixture at gaps of 0.1 to 60 s.
#           Run this FIRST. If the knee sits well below 60 s, set IDLE_GAP_MS
#           to the smallest gap that reaches the cold plateau and the matrix
#           phases below take minutes instead of hours.
#   tap     Tap Fast matrix through bench_asr (ASR decode, no loopback).
#   hold    Hold matrix through bench_e2e (release-to-transcript). Needs the
#           "BlackHole 2ch" loopback and the fixtures from bench-latency.sh.
#   energy  Cost of the cadence arm: worker CPU time over a 60 s window with
#           the keep-alive running, against the same window idle. Adds ANE
#           milliwatts when `sudo powermetrics` is available.
#
# Outputs:
#   bench/idle-<phase>.log   raw phase_timer / idle_arm / asr_stages lines
#   bench/idle-<phase>.csv   per-arm percentile table
#
# Usage:
#   scripts/bench-idle.sh sweep
#   IDLE_GAP_MS=10000 scripts/bench-idle.sh tap
#   REPS=20 scripts/bench-idle.sh hold
#   scripts/bench-idle.sh energy

set -euo pipefail

cd "$(dirname "$0")/.."

PHASE="${1:-}"
REPS="${REPS:-20}"
WARMUP_REPS="${WARMUP_REPS:-3}"
SAMPLE_RATE="${SAMPLE_RATE:-48000}"
BACKEND="${BACKEND:-coreml-unified}"
DEVICE="${DEVICE:-BlackHole 2ch}"
IDLE_GAP_MS="${IDLE_GAP_MS:-60000}"
KEEPALIVE_MS="${KEEPALIVE_MS:-250}"
LENGTHS=(1 5)
SWEEP_GAPS_MS=(0 100 500 2000 5000 10000 60000)
SWEEP_REPS="${SWEEP_REPS:-8}"
WAV_DIR="bench/audio"
ENERGY_WINDOW_S="${ENERGY_WINDOW_S:-60}"

command -v uv >/dev/null || { echo "missing \`uv\` (brew install uv)"; exit 1; }

usage() {
    echo "usage: scripts/bench-idle.sh sweep|tap|hold|energy" >&2
    exit 2
}

[[ -n "$PHASE" ]] || usage

require_fixtures() {
    for len in "$@"; do
        wav="$WAV_DIR/${len}s_${SAMPLE_RATE}.wav"
        [[ -f "$wav" ]] || {
            echo "missing $wav — run scripts/bench-latency.sh first to generate fixtures" >&2
            exit 1
        }
    done
}

build() {
    if [[ "$BACKEND" == "coreml-unified" ]]; then
        echo "Building native Core ML worker…"
        scripts/build-coreml-worker.sh
    fi
    echo "Building $1 (release)…"
    cargo build --release --bin "$1" 2>&1 | tail -3
}

# The ANE and the CPU are shared with whatever else is running, so record the
# load the numbers were taken under rather than discovering later that they
# were not comparable.
note_load() {
    echo "=== machine load at $(date -u +%FT%TZ) ==="
    top -l 1 | head -12
    echo "=== end load ==="
}

case "$PHASE" in
sweep)
    require_fixtures 1
    build bench_asr
    LOG="bench/idle-sweep.log"
    : > "$LOG"
    note_load | tee -a "$LOG"
    wav="$WAV_DIR/1s_${SAMPLE_RATE}.wav"
    for gap in "${SWEEP_GAPS_MS[@]}"; do
        echo "Sweeping gap=${gap}ms (reps=$SWEEP_REPS)…"
        # record-gap-ms=0 isolates the gap: the decode follows the idle
        # interval directly, so the measured cost is the re-wake alone.
        ./target/release/bench_asr --backend "$BACKEND" --wav "$wav" \
            --reps "$SWEEP_REPS" --warmup-reps "$WARMUP_REPS" --stage-timings \
            --arm cold --idle-gap-ms "$gap" --record-gap-ms 0 \
            2>>"$LOG" || echo "  ↑ sweep failed at gap=$gap (see $LOG)"
    done
    uv run --quiet scripts/bench-idle.py --log "$LOG" --out bench/idle-sweep.csv
    echo; cat bench/idle-sweep.csv
    ;;

tap)
    require_fixtures "${LENGTHS[@]}"
    build bench_asr
    LOG="bench/idle-tap.log"
    : > "$LOG"
    note_load | tee -a "$LOG"
    for len in "${LENGTHS[@]}"; do
        wav="$WAV_DIR/${len}s_${SAMPLE_RATE}.wav"
        for arm in warm cold prime cadence; do
            gap="$IDLE_GAP_MS"
            [[ "$arm" == "warm" ]] && gap=0
            echo "Tap ${len}s arm=$arm gap=${gap}ms (reps=$REPS)…"
            ./target/release/bench_asr --backend "$BACKEND" --wav "$wav" \
                --reps "$REPS" --warmup-reps "$WARMUP_REPS" --stage-timings \
                --arm "$arm" --idle-gap-ms "$gap" --keepalive-ms "$KEEPALIVE_MS" \
                2>>"$LOG" || echo "  ↑ tap failed for $wav arm=$arm (see $LOG)"
        done
    done
    uv run --quiet scripts/bench-idle.py --log "$LOG" --out bench/idle-tap.csv
    echo; cat bench/idle-tap.csv
    ;;

hold)
    require_fixtures "${LENGTHS[@]}"
    build bench_e2e
    LOG="bench/idle-hold.log"
    : > "$LOG"
    note_load | tee -a "$LOG"
    for len in "${LENGTHS[@]}"; do
        wav="$WAV_DIR/${len}s_${SAMPLE_RATE}.wav"
        for arm in warm cold prime cadence; do
            gap="$IDLE_GAP_MS"
            [[ "$arm" == "warm" ]] && gap=0
            echo "Hold ${len}s arm=$arm gap=${gap}ms (reps=$REPS)…"
            ./target/release/bench_e2e --backend "$BACKEND" --mode hold \
                --wav "$wav" --device "$DEVICE" \
                --reps "$REPS" --warmup-reps "$WARMUP_REPS" \
                --arm "$arm" --idle-gap-ms "$gap" --keepalive-ms "$KEEPALIVE_MS" \
                2>>"$LOG" || echo "  ↑ hold failed for $wav arm=$arm (see $LOG)"
        done
    done
    uv run --quiet scripts/bench-idle.py --log "$LOG" --out bench/idle-hold.csv --metric end-to-end
    echo; cat bench/idle-hold.csv
    ;;

energy)
    require_fixtures 1
    build bench_asr
    LOG="bench/idle-energy.log"
    : > "$LOG"
    note_load | tee -a "$LOG"
    wav="$WAV_DIR/1s_${SAMPLE_RATE}.wav"

    # Each control and cadence run holds a single repetition open for
    # ENERGY_WINDOW_S with `--record-gap-ms`, so the two windows differ only
    # in whether the keep-alive is dispatching.
    measure() {
        local arm="$1" label="$2"
        echo "Measuring $label for ${ENERGY_WINDOW_S}s…"
        ./target/release/bench_asr --backend "$BACKEND" --wav "$wav" \
            --reps 1 --warmup-reps 1 \
            --arm "$arm" --idle-gap-ms 0 \
            --record-gap-ms "$((ENERGY_WINDOW_S * 1000))" \
            --keepalive-ms "$KEEPALIVE_MS" >>"$LOG" 2>&1 &
        local bench_pid=$!

        # Wait for the worker to exist, then sample its cumulative CPU time
        # across the window. This needs no root and captures the host-side
        # cost of the cadence; powermetrics below adds the engine's own draw.
        #
        # Match on the child of THIS bench process, not on the newest worker
        # named parakeet-coreml-worker: other worktrees on this machine run
        # the same binary, and a name match would silently sample a stranger.
        local worker_pid="" waited=0
        while [[ -z "$worker_pid" && $waited -lt 60 ]]; do
            worker_pid="$(pgrep -P "$bench_pid" -f parakeet-coreml-worker || true)"
            [[ -n "$worker_pid" ]] || { sleep 1; waited=$((waited + 1)); }
        done
        if [[ -z "$worker_pid" ]]; then
            echo "  worker never appeared; skipping $label" >&2
            wait "$bench_pid" || true
            return
        fi
        # Let the warmup rep finish before the window opens.
        sleep 5
        local before after
        before="$(ps -o cputime= -p "$worker_pid" | tr -d ' ')"
        # `grep` on an unexpected sampler name would yield nothing AND return
        # before the window elapsed, which would quietly turn a 60 s
        # measurement into a 0 s one. Sample into a file for the full window
        # and filter afterwards, so the window length never depends on what
        # powermetrics chose to print.
        local pm_out="bench/idle-energy-${label}.powermetrics"
        if command -v powermetrics >/dev/null && sudo -n true 2>/dev/null; then
            sudo -n powermetrics --samplers ane_power -i 5000 -n \
                "$((ENERGY_WINDOW_S / 5))" >"$pm_out" 2>/dev/null || true
            if grep -i -q "ANE" "$pm_out"; then
                grep -i "ANE" "$pm_out" | tee -a "$LOG"
            else
                echo "  powermetrics produced no ANE rows; check the sampler name" \
                    | tee -a "$LOG"
            fi
        else
            echo "  powermetrics needs an interactive sudo; reporting worker CPU only" \
                | tee -a "$LOG"
            sleep "$ENERGY_WINDOW_S"
        fi
        after="$(ps -o cputime= -p "$worker_pid" | tr -d ' ')"
        echo "energy_sample label=$label worker_pid=$worker_pid cputime_before=$before cputime_after=$after" \
            | tee -a "$LOG"
        wait "$bench_pid" || true
    }

    measure cold control-idle
    measure cadence keep-alive
    echo
    grep energy_sample "$LOG"
    ;;

*)
    usage
    ;;
esac
