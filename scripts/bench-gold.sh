#!/usr/bin/env bash
# Reproducible real-speech quality/performance A/B on the shipping Mac.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPETITIONS=${REPETITIONS:-10}
WORKER=${COREML_WORKER:-/Applications/Parakeet.app/Contents/MacOS/parakeet-coreml-worker}
MODEL_DIR=${COREML_MODEL_DIR:-$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-unified-en-0.6b}
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

"$BINARY" "${common[@]}" \
    --backend sherpa \
    --json-out "$ROOT/bench/sherpa-gold-quality.json" || true

"$BINARY" "${common[@]}" \
    --backend sherpa \
    --vocabulary "$VOCABULARY" \
    --json-out "$ROOT/bench/sherpa-vocabulary-gold-quality.json" || true

exit "$shipping_status"
