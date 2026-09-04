#!/usr/bin/env bash
# Build the pinned native Parakeet Unified worker and place it beside Cargo's
# release binaries. The Rust backend uses a resident process, so Swift/CoreML
# model load and ANE graph compilation are paid once per app launch.

set -euo pipefail

cd "$(dirname "$0")/.."

PACKAGE_DIR="native/ParakeetCoreMLWorker"
PRODUCT="parakeet-coreml-worker"
DESTINATION="target/release/$PRODUCT"

# FluidAudio hardcodes the offline encoder window at 15 s, so bucketed
# short-window encoders need a three-file change to it (kata fgzt). Package.swift
# depends on `.fluidaudio-local`, which this reconstitutes from the pinned
# upstream revision plus the checked-in patch, so the dependency is a local
# override rather than a vendored copy: nothing of FluidAudio is checked in
# except the patch. Drop the patch and restore the SCM dependency together if
# the change lands upstream.
FLUIDAUDIO_REVISION="00a9aa771900ea09c485659663be31019e293e47"
FLUIDAUDIO_LOCAL="$PACKAGE_DIR/.fluidaudio-local"
FLUIDAUDIO_PATCH="$PACKAGE_DIR/patches/fluidaudio-offline-window.patch"

if [ ! -d "$FLUIDAUDIO_LOCAL/.git" ]; then
    rm -rf "$FLUIDAUDIO_LOCAL"
    git clone --quiet https://github.com/FluidInference/FluidAudio.git "$FLUIDAUDIO_LOCAL"
fi
git -C "$FLUIDAUDIO_LOCAL" fetch --quiet origin "$FLUIDAUDIO_REVISION"
git -C "$FLUIDAUDIO_LOCAL" checkout --quiet --force "$FLUIDAUDIO_REVISION"
git -C "$FLUIDAUDIO_LOCAL" apply "$(cd "$(dirname "$FLUIDAUDIO_PATCH")" && pwd)/$(basename "$FLUIDAUDIO_PATCH")"

# `swift package edit --path` records the override in the workspace rather than
# in Package.swift, so the manifest keeps naming the upstream revision.
swift build --package-path "$PACKAGE_DIR" -c release --product "$PRODUCT"
BIN_DIR="$(swift build --package-path "$PACKAGE_DIR" -c release --show-bin-path)"
mkdir -p "$(dirname "$DESTINATION")"
cp "$BIN_DIR/$PRODUCT" "$DESTINATION"

echo "Built $DESTINATION"
