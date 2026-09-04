#!/usr/bin/env bash
# Reconstitute the patched local FluidAudio checkout the Swift package depends on.
#
# FluidAudio hardcodes the offline encoder window at 15 s, which is the whole of
# what bucketed short-window encoders have to change (kata fgzt), so
# `native/ParakeetCoreMLWorker/Package.swift` takes it as a local path
# dependency instead of a source-control one. Nothing of FluidAudio is checked
# in except the patch: this clones the pinned upstream revision and applies it.
#
# Every entry point that builds or tests that package must run this first, and
# a clean clone has no checkout at all, so this is idempotent and safe to repeat.
# Drop the patch and restore the source-control dependency together if the
# change lands upstream.

set -euo pipefail

cd "$(dirname "$0")/.."

PACKAGE_DIR="native/ParakeetCoreMLWorker"
FLUIDAUDIO_REVISION="00a9aa771900ea09c485659663be31019e293e47"
FLUIDAUDIO_LOCAL="$PACKAGE_DIR/.fluidaudio-local"
FLUIDAUDIO_PATCH="$PACKAGE_DIR/patches/fluidaudio-offline-window.patch"

if [ ! -d "$FLUIDAUDIO_LOCAL/.git" ]; then
    rm -rf "$FLUIDAUDIO_LOCAL"
    git clone --quiet https://github.com/FluidInference/FluidAudio.git "$FLUIDAUDIO_LOCAL"
fi
git -C "$FLUIDAUDIO_LOCAL" fetch --quiet origin "$FLUIDAUDIO_REVISION"
# `checkout --force` resets tracked files but leaves untracked ones behind, and
# an untracked .swift file in Sources would silently join the build.
git -C "$FLUIDAUDIO_LOCAL" clean --quiet -fdx
git -C "$FLUIDAUDIO_LOCAL" checkout --quiet --force "$FLUIDAUDIO_REVISION"
git -C "$FLUIDAUDIO_LOCAL" apply "$(cd "$(dirname "$FLUIDAUDIO_PATCH")" && pwd)/$(basename "$FLUIDAUDIO_PATCH")"
