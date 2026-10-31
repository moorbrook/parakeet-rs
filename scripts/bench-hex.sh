#!/usr/bin/env bash
# scripts/bench-hex.sh — Hex's engine against ours, in one sitting (kata mspd).
#
# Hex (github.com/anomalyco/hex, MIT) runs the same Parakeet Unified EN 0.6B
# as a Q8_0 GGUF through transcribe.cpp on Metal. `bench/hex-engine` is a
# standalone package that mirrors Hex's engine path; the app never depends on
# it. This script runs both engines on the same audio with the same warm-up
# and the same release-to-text definition:
#
#   gold  `asr_diff --gold` (ours, Core ML) and `bench_hex gold` (Hex, whole
#         clip and live stream), ten repetitions each, same scorer.
#   hold  per loopback fixture, four arms back to back: ours windowed (the
#         shipping 6 s cap), ours serial, Hex whole clip at release through
#         our capture path, Hex live stream fed during the hold.
#
# Requires the Core ML model in Application Support, a duplex loopback named
# "BlackHole 2ch", and the Hold fixtures in bench/audio (scripts/bench-latency.sh
# generates them). The GGUF is fetched into $HEX_CACHE and checked against
# Hex's pinned SHA-256.
#
# BENCH_LOCK names a lock directory shared with other benchmark runners on the
# machine. Each timed group takes it with mkdir, retrying every 60 s for up to
# 30 minutes, records the load average, and releases it when the group ends.
#
# Outputs go to bench/hex-engine/results/.
#
# Usage:
#   scripts/bench-hex.sh
#   STAGES=gold BENCH_LOCK=/path/to/lock scripts/bench-hex.sh

set -euo pipefail

cd "$(dirname "$0")/.."

STAGES="${STAGES:-gold hold}"
REPETITIONS="${REPETITIONS:-10}"
REPS="${REPS:-30}"
WARMUP_REPS="${WARMUP_REPS:-2}"
DEVICE="${DEVICE:-BlackHole 2ch}"
SAMPLE_RATE=48000
HEX_CACHE="${HEX_CACHE:-$HOME/.cache/parakeet-hex}"
BENCH_LOCK="${BENCH_LOCK:-}"
OUT="bench/hex-engine/results"

GGUF_REPO="handy-computer/parakeet-unified-en-0.6b-gguf"
GGUF_REVISION="7e948f21b7bdbac698d3318db9d350f1096f3b6c"
GGUF_FILE="parakeet-unified-en-0.6b-Q8_0.gguf"
GGUF_SHA256="4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795"
GGUF="$HEX_CACHE/$GGUF_FILE"

WORKER="target/release/parakeet-coreml-worker"
MODEL_DIR="${COREML_MODEL_DIR:-$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-unified-en-0.6b}"
HEX="bench/hex-engine/target/release/bench_hex"
LENGTHS=(1 3 5 10 20)
MULTIPAUSE_WAV="bench/audio/multipause_${SAMPLE_RATE}.wav"

mkdir -p "$OUT" "$HEX_CACHE"

if [[ ! -f "$GGUF" ]] || [[ "$(shasum -a 256 "$GGUF" | cut -d' ' -f1)" != "$GGUF_SHA256" ]]; then
    echo "Fetching $GGUF_FILE at $GGUF_REVISION…"
    curl -fsSL -o "$GGUF.part" \
        "https://huggingface.co/$GGUF_REPO/resolve/$GGUF_REVISION/$GGUF_FILE"
    actual="$(shasum -a 256 "$GGUF.part" | cut -d' ' -f1)"
    if [[ "$actual" != "$GGUF_SHA256" ]]; then
        echo "GGUF SHA-256 mismatch: $actual" >&2
        exit 1
    fi
    mv "$GGUF.part" "$GGUF"
fi

echo "Building the Core ML worker, asr_diff, bench_e2e and bench_hex…"
scripts/build-coreml-worker.sh
cargo build --release --locked --bin asr_diff --bin bench_e2e
cargo build --release --locked --manifest-path bench/hex-engine/Cargo.toml

# A binary that cannot start (a missing dylib aborts it in dyld, status 134)
# would otherwise cost every arm it runs in, an hour into the sitting.
for binary in ./target/release/asr_diff ./target/release/bench_e2e "$HEX"; do
    status=0
    "$binary" --help >/dev/null 2>&1 || status=$?
    if (( status >= 128 )); then
        "$binary" --help 2>&1 | head -3 >&2 || true
        echo "bench-hex: $binary cannot start (status $status)" >&2
        exit 1
    fi
done

fail() {
    echo "bench-hex: $*" >&2
    exit 1
}
failures=0
held=0
release_lock() {
    if [[ -n "$BENCH_LOCK" && "$held" == 1 ]]; then
        rmdir "$BENCH_LOCK"
        held=0
    fi
}
trap release_lock EXIT
take_lock() {
    local label="$1"
    if [[ -n "$BENCH_LOCK" ]]; then
        local tries=0
        until mkdir "$BENCH_LOCK" 2>/dev/null; do
            tries=$((tries + 1))
            if (( tries > 30 )); then
                echo "gave up on $BENCH_LOCK after 30 minutes" >&2
                exit 3
            fi
            echo "waiting for $BENCH_LOCK ($tries/30)…"
            sleep 60
        done
        held=1
    fi
    echo "$(date -u +%FT%TZ) $label load:$(uptime | sed 's/.*load averages*: //')" | tee -a "$OUT/load.log"
}

{
    echo "date: $(date -u +%FT%TZ)"
    echo "commit: $(git rev-parse HEAD)"
    echo "os: $(sw_vers -productName) $(sw_vers -productVersion)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GiB, $(sysctl -n hw.ncpu) logical CPUs"
    echo "gguf: $GGUF_REPO@$GGUF_REVISION/$GGUF_FILE sha256=$GGUF_SHA256"
    echo "transcribe.cpp: $(grep -A2 'name = "transcribe-cpp-sys"' bench/hex-engine/Cargo.lock | grep source | sed 's/.*"\(.*\)"/\1/')"
} > "$OUT/identity.txt"

if [[ " $STAGES " == *" gold "* ]]; then
    take_lock gold
    # Both binaries exit 1 for a failed quality gate as well as for an error,
    # so the report's existence, not the exit status, says the arm ran.
    rm -f "$OUT/ours-gold.json" "$OUT/hex-offline-gold.json" "$OUT/hex-stream-gold.json"
    "./target/release/asr_diff" --gold bench/gold/manifest.json --audio-dir bench/gold/audio \
        --repetitions "$REPETITIONS" --backend coreml-unified \
        --worker "$WORKER" --model-dir "$MODEL_DIR" \
        --json-out "$OUT/ours-gold.json" || true
    for path in offline stream; do
        "$HEX" gold --path "$path" --model "$GGUF" --gold bench/gold/manifest.json \
            --audio-dir bench/gold/audio --repetitions "$REPETITIONS" \
            --json-out "$OUT/hex-$path-gold.json" || true
    done
    release_lock
    for report in ours-gold hex-offline-gold hex-stream-gold; do
        [[ -f "$OUT/$report.json" ]] || fail "$report.json was not written"
    done
fi

if [[ " $STAGES " == *" hold "* ]]; then
    for len in "${LENGTHS[@]}"; do
        [[ -f "bench/audio/${len}s_${SAMPLE_RATE}.wav" ]] || {
            echo "missing bench/audio/${len}s_${SAMPLE_RATE}.wav — run scripts/bench-latency.sh first" >&2
            exit 1
        }
    done
    if [[ ! -f "$MULTIPAUSE_WAV" ]]; then
        # The same text and gaps scripts/bench-hold.sh generates.
        aiff="bench/audio/multipause.aiff"
        say -o "$aiff" \
            "The latency benchmark drives the recognizer with audio of fixed lengths. \
[[slnc 800]] It records timing for thirty iterations and aggregates the result. \
[[slnc 800]] Then it writes the percentiles to a comma separated file. \
[[slnc 800]] That file is included in the architecture decision record."
        afconvert -f WAVE -d "LEI16@${SAMPLE_RATE}" -c 1 "$aiff" "$MULTIPAUSE_WAV"
        rm "$aiff"
    fi
    arms=(ours-windowed ours-serial hex-offline hex-stream)
    for arm in "${arms[@]}"; do
        : > "$OUT/hold-$arm.log"
        : > "$OUT/hold-$arm-multipause.log"
    done
    wavs=()
    for len in "${LENGTHS[@]}"; do wavs+=("bench/audio/${len}s_${SAMPLE_RATE}.wav"); done
    wavs+=("$MULTIPAUSE_WAV")
    for wav in "${wavs[@]}"; do
        suffix=""
        [[ "$wav" == "$MULTIPAUSE_WAV" ]] && suffix="-multipause"
        take_lock "hold $(basename "$wav")"
        for arm in "${arms[@]}"; do
            log="$OUT/hold-$arm$suffix.log"
            echo "Hold $arm on $wav…"
            before=$(grep -c phase_timer "$log" || true)
            case "$arm" in
                ours-windowed|ours-serial)
                    windows=$([[ "$arm" == ours-windowed ]] && echo 6,6 || echo off)
                    ./target/release/bench_e2e --backend coreml-unified --mode hold \
                        --worker "$WORKER" --model-dir "$MODEL_DIR" \
                        --wav "$wav" --device "$DEVICE" --hold-windows "$windows" \
                        --reps "$REPS" --warmup-reps "$WARMUP_REPS" 2>>"$log" || true
                    ;;
                hex-offline|hex-stream)
                    "$HEX" hold --path "${arm#hex-}" --model "$GGUF" --wav "$wav" \
                        --device "$DEVICE" --reps "$REPS" --warmup-reps "$WARMUP_REPS" \
                        2>>"$log" || true
                    ;;
            esac
            after=$(grep -c phase_timer "$log" || true)
            if (( after - before != REPS )); then
                echo "  ↑ $arm on $wav produced $((after - before)) of $REPS repetitions (see $log)" >&2
                failures=$((failures + 1))
            fi
        done
        release_lock
    done
    for arm in "${arms[@]}"; do
        uv run --quiet scripts/bench-aggregate.py --log "$OUT/hold-$arm.log" \
            --out "$OUT/hold-$arm.csv" --metric end-to-end
        uv run --quiet scripts/bench-aggregate.py --log "$OUT/hold-$arm-multipause.log" \
            --out "$OUT/hold-$arm-multipause.csv" --metric end-to-end
        echo "== $arm"
        cat "$OUT/hold-$arm.csv" "$OUT/hold-$arm-multipause.csv"
    done
fi

(( failures == 0 )) || fail "$failures Hold arm runs fell short of $REPS repetitions"
