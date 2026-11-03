#!/usr/bin/env bash
# Replay Hold capture with a live VAD, then with its worker killed mid-speech.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${BINARY:-$ROOT/target/release/bench_e2e}"
WORKER="${WORKER:-$ROOT/target/release/parakeet-coreml-worker}"
MODEL_DIR="${MODEL_DIR:-$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-unified-en-0.6b}"
OUT_DIR="${OUT_DIR:-$ROOT/target/vad-failure}"
FIXTURE="$ROOT/bench/audio/3s_48000.wav"
[[ -f "$FIXTURE" ]] || { echo 'Generate fixtures with scripts/bench-latency.sh first' >&2; exit 2; }
mkdir -p "$OUT_DIR"
bench_pid=""
trap 'if [[ -n "$bench_pid" ]]; then kill "$bench_pid" 2>/dev/null || true; fi' EXIT

for arm in control killed; do
    log="$OUT_DIR/$arm.log"
    RUST_LOG=info PARAKEET_COREML_WORKER="$WORKER" "$BINARY" \
        --backend coreml-unified --worker "$WORKER" --model-dir "$MODEL_DIR" \
        --mode hold --hold-windows 3,6 --device "${DEVICE:-BlackHole 2ch}" \
        --wav "$FIXTURE" \
        --expected 'The quick brown fox jumps over the lazy dog.' \
        --reps 1 --warmup-reps 0 >"$log" 2>&1 &
    bench_pid=$!
    if [[ "$arm" == killed ]]; then
        vad_pid=""
        for ((attempt = 0; attempt < 200; attempt++)); do
            vad_pid=$(ps -axo pid=,ppid=,command= | \
                awk -v parent="$bench_pid" '$2 == parent && /--vad-model/ && !found { print $1; found = 1 }')
            [[ -n "$vad_pid" ]] && break
            kill -0 "$bench_pid" 2>/dev/null || break
            sleep 0.05
        done
        [[ -n "$vad_pid" ]] || { cat "$log"; echo 'FAIL: no VAD child to kill' >&2; exit 1; }
        sleep 0.5
        [[ "$(ps -p "$vad_pid" -o ppid= | tr -d ' ')" == "$bench_pid" ]] || {
            echo 'FAIL: VAD child exited before fault injection' >&2
            exit 1
        }
        kill -KILL "$vad_pid"
    fi
    if ! wait "$bench_pid"; then
        bench_pid=""
        cat "$log"
        echo "FAIL: $arm Hold capture lost the transcript" >&2
        exit 1
    fi
    bench_pid=""
    if [[ "$arm" == killed ]]; then
        grep -q 'hold windowing off: VAD inference failed' "$log"
    elif grep -q 'hold windowing off' "$log"; then
        cat "$log"
        echo 'FAIL: control did not exercise live VAD windowing' >&2
        exit 1
    fi
    if [[ "$arm" == control ]]; then
        grep -q 'hold windows=' "$log"
    fi
    grep -q 'false_cuts=0 mismatches=0' "$log"
done
echo 'PASS: Hold preserves the complete transcript with a live or failed VAD worker'
