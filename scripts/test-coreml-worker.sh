#!/usr/bin/env bash
# Run the native worker's Swift tests. Reconstitutes the patched local
# FluidAudio checkout first, so this works from a clean clone where bare
# `swift test` cannot resolve Package.swift's path dependency.

set -euo pipefail

cd "$(dirname "$0")/.."

scripts/reconstitute-fluidaudio.sh
swift test --package-path native/ParakeetCoreMLWorker "$@"
