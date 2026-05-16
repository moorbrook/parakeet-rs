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

# --- 0. arm64-only architecture gate -------------------------------------
# parakeet-rs is Apple Silicon only (ADR-0002). sherpa-onnx and the
# onnxruntime dylibs we bundle are arm64-only — silently producing a
# fat / Intel binary would yield a .app that crashes on launch. Pin the
# expected arch here so a wrong CI matrix entry fails loudly instead.
if [ -f "$BIN_SRC" ]; then
  if ! lipo -archs "$BIN_SRC" | grep -qx arm64; then
    echo "ERROR: $BIN_SRC is not arm64-only (got: $(lipo -archs "$BIN_SRC"))" >&2
    echo "       parakeet-rs ships Apple Silicon only — see docs/ADR.md ADR-0002." >&2
    exit 1
  fi
fi
# NOTE: there is no repo-root Info.plist; all custom keys are merged
# into cargo-bundle's generated plist via PlistBuddy below.

echo "1. cargo bundle --release"
cd "$ROOT"
cargo bundle --release >/dev/null

# --- 2. merge Info.plist -------------------------------------------------
# cargo-bundle writes a clean Info.plist with the bundle-id, version, icon,
# category, etc. We add our own keys (usage strings, LSUIElement, HiDPI,
# application category, copyright) here via PlistBuddy. There is NO
# repo-root Info.plist — every custom key lives in this script so the
# `target/release/bundle/osx/Parakeet.app/Contents/Info.plist` produced
# by cargo-bundle + this script is the single source of truth.

echo "2. merge custom Info.plist keys"
PLIST="$APP/Contents/Info.plist"
add_key() {
  local key="$1" type="$2" value="$3"
  /usr/libexec/PlistBuddy -c "Delete :$key" "$PLIST" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c "Add :$key $type $value" "$PLIST"
}
# TCC usage strings (required by macOS for the listed entitlements).
add_key NSMicrophoneUsageDescription string \
  "Parakeet needs microphone access to transcribe your speech to text."
add_key NSAppleEventsUsageDescription string \
  "Parakeet uses Accessibility to paste transcribed text into the focused window."
# Menu-bar agent app: no Dock icon, no main menu strip.
add_key LSUIElement bool true
# Default-true on modern macOS, but explicit is safer — avoids the
# rare case of a launcher (lipo'd or non-standard runtime) thinking
# the app wants 72-dpi rendering.
add_key NSHighResolutionCapable bool true
# App Store / Spotlight categorization. Productivity covers
# dictation, transcription, voice-to-text utilities.
add_key LSApplicationCategoryType string "public.app-category.productivity"
# Required for notarization + nice-to-have for the Get Info pane.
add_key NSHumanReadableCopyright string "Copyright © 2026 parakeet-rs"

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

# Verify the rpath actually took. `install_name_tool -add_rpath` can
# silently no-op if the binary's signature blocks it (e.g. some macOS
# versions reject load-command edits on previously-signed binaries);
# without an rpath the .app aborts at launch with
# "Library not loaded: @rpath/libsherpa-onnx-c-api.dylib ... no
# LC_RPATH's found", which we'd otherwise only learn about from a
# user crash report.
if ! otool -l "$EXE" | awk '/cmd LC_RPATH/{getline;getline;print $2}' \
     | grep -qx '@executable_path/../Frameworks'; then
  echo "ERROR: install_name_tool -add_rpath silently failed on $EXE" >&2
  echo "       Bundle would abort at launch with 'no LC_RPATH found'." >&2
  exit 1
fi

# --- 5. ad-hoc code-sign -------------------------------------------------
# Gatekeeper on macOS 14+ rejects unsigned binaries by default. We sign
# each piece individually rather than with `--deep`, because `--deep` is
# deprecated, silently glosses over nested-bundle problems, and produces
# signatures that don't survive notarisation. Order matters: dylibs
# first (leaves), executable last (root).
#
# `--options runtime` enables Hardened Runtime, which is mandatory for
# notarisation. `--entitlements` attaches the mic + apple-events strings.
# The `-` identity is the ad-hoc signature: enough to run locally on this
# Mac, NOT enough for distribution. For real shipping you'd swap `-` for
# a `Developer ID Application: <Name> (TEAMID)` identity, then run
# `xcrun notarytool submit ... --wait` and `xcrun stapler staple` —
# documented at the bottom of this script.

echo "5. code-sign"
SIGN_ID="${PARAKEET_SIGN_ID:--}"          # `-` = ad-hoc
ENTITLEMENTS="$ROOT/entitlements.plist"

# Hardened Runtime + the entitlements file are required for notarisation
# but BREAK ad-hoc signing: ad-hoc gives each artefact a different pseudo
# Team ID, and Hardened Runtime then refuses to load the dylibs because
# the bundle's Team ID doesn't match theirs. So both extras stay off for
# the ad-hoc path; the Developer-ID path turns them on.
if [ "$SIGN_ID" = "-" ]; then
  EXTRA_FLAGS=()
else
  EXTRA_FLAGS=(--options runtime --timestamp --entitlements "$ENTITLEMENTS")
fi

sign_one() {
  local target="$1"
  # `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` is the bash 3.x-compatible
  # idiom for expanding an array that may be empty under `set -u`.
  # Plain `${EXTRA_FLAGS[@]}` errors with "unbound variable" on the
  # macOS-stock bash when the ad-hoc branch leaves the array empty.
  codesign --force --sign "$SIGN_ID" \
           ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
           "$target" >/dev/null 2>&1 || {
    echo "  warn: codesign failed on $target" >&2
    return 1
  }
}

# Leaves first: every dylib in Contents/Frameworks/.
for lib in "$APP"/Contents/Frameworks/*.dylib; do
  [ -f "$lib" ] && sign_one "$lib"
done
# Then the main executable.
sign_one "$APP/Contents/MacOS/parakeet-rs"
# Then the bundle as a whole — codesign requires this last step to seal
# the Contents/_CodeSignature resource manifest.
sign_one "$APP"

# --- 6. verify -----------------------------------------------------------
# `--verify --deep --strict` walks every nested signature, the kind of
# check Apple's notary service performs. Surface any breakage early.

echo "6. verify signature"
codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | tail -3
# spctl --assess is Gatekeeper's view of the bundle. For an ad-hoc build
# (`SIGN_ID=-`, the default), rejection is EXPECTED — Gatekeeper won't
# trust an unnotarised signature regardless of how clean it is. For a
# real Developer-ID-signed build, rejection means the bundle is not
# distributable, which is a release blocker — fail the script so CI /
# release scripts see a non-zero exit.
if ! spctl --assess --type execute --verbose=2 "$APP" 2>&1 | tail -2; then
  if [ "$SIGN_ID" = "-" ]; then
    echo "  note: spctl rejected the ad-hoc signature — that's expected; it'd"
    echo "        accept a Developer-ID-signed + notarised build."
  else
    echo "ERROR: spctl rejected the bundle signed with '$SIGN_ID'. This build" >&2
    echo "       is not distributable. Common causes: identity missing from" >&2
    echo "       login keychain, intermediate cert expired, notarisation not" >&2
    echo "       yet stapled. Check 'codesign -dvv \"$APP\"' for details." >&2
    exit 1
  fi
fi

# --- Distribution notes (NOT run by default) -----------------------------
# To produce a build a user can download from the internet without
# Gatekeeper warnings:
#
#   1. Get a Developer ID Application cert into your login keychain.
#   2. PARAKEET_SIGN_ID="Developer ID Application: <Name> (TEAMID)" \
#        ./scripts/make-app.sh
#   3. ditto -c -k --keepParent "$APP" Parakeet.zip
#   4. xcrun notarytool submit Parakeet.zip \
#        --apple-id you@example.com --team-id TEAMID \
#        --password APP_SPECIFIC_PASSWORD --wait
#   5. xcrun stapler staple "$APP"
#   6. ditto -c -k --keepParent "$APP" Parakeet-notarised.zip
#
# Steps 3–6 only matter for distribution. Local installs work with
# `SIGN_ID=-` (the default), which is the ad-hoc identity.

echo
echo "Built $APP"
du -sh "$APP" | awk '{print "  size: " $1}'
echo "  drop into /Applications, or:  open $APP"
