#!/usr/bin/env bash
# Produce a complete, runnable Parakeet.app:
#   1. `cargo bundle --release`           — scaffolds the .app skeleton
#   2. Merge our Info.plist keys          — LSUIElement, mic + apple-events
#                                           usage strings (cargo-bundle
#                                           can't merge plists cleanly)
#   3. Bundle dylibs                      — sherpa-onnx + onnxruntime into
#                                           Contents/Frameworks/
#   4. Rewrite rpath                      — point at @executable_path/../
#                                           Frameworks instead of the
#                                           dev-machine @loader_path
#   5. Ad-hoc code-sign                   — `codesign -s -` so Gatekeeper
#                                           lets it run locally
#
# Output: target/release/bundle/osx/Parakeet.app — drop into /Applications.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/target/release/bundle/osx/Parakeet.app"
BIN_SRC="$ROOT/target/release/parakeet-rs"
DYLIB_SRC="$ROOT/target/release"
EXTRA_PLIST="$ROOT/Info.plist"

echo "1. cargo bundle --release"
cd "$ROOT"
cargo bundle --release >/dev/null

# --- 2. merge Info.plist -------------------------------------------------
# cargo-bundle writes a clean Info.plist with the bundle-id, version, icon,
# category, etc. We need to add our own keys (mic permission, apple-events,
# LSUIElement). PlistBuddy reads/writes Apple's plist format natively.

echo "2. merge custom Info.plist keys"
PLIST="$APP/Contents/Info.plist"
add_key() {
  local key="$1" type="$2" value="$3"
  /usr/libexec/PlistBuddy -c "Delete :$key" "$PLIST" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c "Add :$key $type $value" "$PLIST"
}
add_key NSMicrophoneUsageDescription string \
  "Parakeet needs microphone access to transcribe your speech to text."
add_key NSAppleEventsUsageDescription string \
  "Parakeet uses Accessibility to paste transcribed text into the focused window."
add_key LSUIElement bool true

# --- 3. bundle dylibs ----------------------------------------------------
# sherpa-onnx-sys copied them to target/release/ during the cargo build.
# Put them in Contents/Frameworks/ where macOS expects bundled libraries.

echo "3. copy dylibs into Contents/Frameworks"
mkdir -p "$APP/Contents/Frameworks"
for lib in libsherpa-onnx-c-api.dylib libsherpa-onnx-cxx-api.dylib \
           libonnxruntime.dylib libonnxruntime.1.24.4.dylib; do
  if [ -f "$DYLIB_SRC/$lib" ]; then
    cp -f "$DYLIB_SRC/$lib" "$APP/Contents/Frameworks/$lib"
  else
    echo "  warn: missing $DYLIB_SRC/$lib" >&2
  fi
done

# --- 4. fix the binary's rpath -------------------------------------------
# The binary was linked with `-Wl,-rpath,@loader_path` (which in a .app
# resolves to Contents/MacOS/, no dylibs there) and a dev-machine absolute
# path to target/sherpa-onnx-prebuilt/.../lib (also gone in the bundle).
# Add an rpath that resolves correctly inside the .app, drop the stale ones.

echo "4. rewrite rpath"
EXE="$APP/Contents/MacOS/parakeet-rs"
# Drop any existing rpaths that won't resolve inside the .app.
for old in $(otool -l "$EXE" | awk '/cmd LC_RPATH/{getline;getline;print $2}'); do
  install_name_tool -delete_rpath "$old" "$EXE" 2>/dev/null || true
done
install_name_tool -add_rpath '@executable_path/../Frameworks' "$EXE"

# --- 5. ad-hoc code-sign -------------------------------------------------
# Gatekeeper on macOS 14+ rejects unsigned binaries by default. The `-`
# identity is the ad-hoc signature: enough to run on the dev machine, not
# enough for distribution. For real distribution you'd substitute a
# Developer ID Application certificate.

echo "5. ad-hoc code-sign"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || {
  echo "  warn: codesign failed; the .app may still run if SIP allows" >&2
}

echo
echo "Built $APP"
du -sh "$APP" | awk '{print "  size: " $1}'
echo "  drop into /Applications, or:  open $APP"
