#!/usr/bin/env bash
# Reproducible real-speech quality/performance A/B on the shipping Mac.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPETITIONS=${REPETITIONS:-10}
# The installed app's worker by default, so a gold run measures what shipped.
# Point this at target/release/parakeet-coreml-worker to measure the working
# tree — required whenever the worker protocol or decode loop has changed.
WORKER=${COREML_WORKER:-/Applications/Parakeet.app/Contents/MacOS/parakeet-coreml-worker}
MODEL_DIR=${COREML_MODEL_DIR:-$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-unified-en-0.6b}
# The TDT 0.6B v3 challenger (kata f0zg). Fetch it with
# scripts/fetch-tdt-v3-model.py; absent, the challenger row is skipped and the
# shipping gate still runs.
TDT_MODEL_DIR=${COREML_TDT_V3_MODEL_DIR:-$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-tdt-0.6b-v3}
MANIFEST="$ROOT/bench/gold/manifest.json"
AUDIO_DIR="$ROOT/bench/gold/audio"
VOCABULARY="$ROOT/bench/gold/vocabulary.txt"
BINARY="$ROOT/target/release/asr_diff"

if [[ ! -x "$WORKER" ]]; then
    echo "Core ML worker is not executable: $WORKER" >&2
    exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Core ML model directory is missing: $MODEL_DIR" >&2
    exit 2
fi

cargo build --manifest-path "$ROOT/Cargo.toml" --release --locked --bin asr_diff

common=(
    --gold "$MANIFEST"
    --audio-dir "$AUDIO_DIR"
    --repetitions "$REPETITIONS"
)

shipping_status=0
"$BINARY" "${common[@]}" \
    --backend coreml-unified \
    --worker "$WORKER" \
    --model-dir "$MODEL_DIR" \
    --json-out "$ROOT/bench/coreml-gold-quality.json" || shipping_status=$?

if [[ -d "$TDT_MODEL_DIR" ]]; then
    "$BINARY" "${common[@]}" \
        --backend coreml-tdt-v3 \
        --worker "$WORKER" \
        --model-dir "$TDT_MODEL_DIR" \
        --json-out "$ROOT/bench/coreml-tdt-v3-gold-quality.json" || true
else
    echo "skipping the TDT v3 challenger: $TDT_MODEL_DIR is missing" >&2
fi

"$BINARY" "${common[@]}" \
    --backend coreml-unified \
    --worker "$WORKER" \
    --model-dir "$MODEL_DIR" \
    --vocabulary "$VOCABULARY" \
    --hotword-score "${HOTWORD_SCORE:-2.0}" \
    --json-out "$ROOT/bench/coreml-vocabulary-gold-quality.json" || true

"$BINARY" "${common[@]}" \
    --backend sherpa \
    --json-out "$ROOT/bench/sherpa-gold-quality.json" || true

"$BINARY" "${common[@]}" \
    --backend sherpa \
    --vocabulary "$VOCABULARY" \
    --json-out "$ROOT/bench/sherpa-vocabulary-gold-quality.json" || true

exit "$shipping_status"
