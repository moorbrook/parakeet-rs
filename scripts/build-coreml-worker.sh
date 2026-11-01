#!/usr/bin/env bash
# Build the pinned native Parakeet Unified worker and place it beside Cargo's
# release binaries. The Rust backend uses a resident process, so Swift/CoreML
# model load and ANE graph compilation are paid once per app launch.

set -euo pipefail

cd "$(dirname "$0")/.."

PACKAGE_DIR="native/ParakeetCoreMLWorker"
PRODUCT="parakeet-coreml-worker"
DESTINATION="target/release/$PRODUCT"

swift build --package-path "$PACKAGE_DIR" -c release --product "$PRODUCT"
BIN_DIR="$(swift build --package-path "$PACKAGE_DIR" -c release --show-bin-path)"
mkdir -p "$(dirname "$DESTINATION")"
cp "$BIN_DIR/$PRODUCT" "$DESTINATION"

echo "Built $DESTINATION"
