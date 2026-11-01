#!/usr/bin/env bash
# Run the native worker's Swift tests.

set -euo pipefail

cd "$(dirname "$0")/.."

swift test --package-path native/ParakeetCoreMLWorker "$@"
