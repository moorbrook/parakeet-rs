#!/usr/bin/env bash
# Regenerate the macOS .icns from the master source PNG.
# Run after each design change to `assets/icon-source.png`. Uses only
# macOS-built-in tools (`sips`, `iconutil`); no Homebrew install required.
#
# Output: assets/icon.icns (plus a discarded intermediate .iconset directory)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/assets/icon-source.png"
ICONSET="$ROOT/assets/parakeet.iconset"
OUT="$ROOT/assets/icon.icns"

if [ ! -f "$SRC" ]; then
  echo "missing $SRC" >&2
  exit 1
fi

rm -rf "$ICONSET"
mkdir -p "$ICONSET"

# Apple's full set: 16 / 32 / 128 / 256 / 512 in both 1x and 2x.
# We use `-z H W` to fit-into-bounds; the source is square so this just
# downscales. Last entry is the 1024 master itself.
sips -z 16   16   "$SRC" --out "$ICONSET/icon_16x16.png"      >/dev/null
sips -z 32   32   "$SRC" --out "$ICONSET/icon_16x16@2x.png"   >/dev/null
sips -z 32   32   "$SRC" --out "$ICONSET/icon_32x32.png"      >/dev/null
sips -z 64   64   "$SRC" --out "$ICONSET/icon_32x32@2x.png"   >/dev/null
sips -z 128  128  "$SRC" --out "$ICONSET/icon_128x128.png"    >/dev/null
sips -z 256  256  "$SRC" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256  256  "$SRC" --out "$ICONSET/icon_256x256.png"    >/dev/null
sips -z 512  512  "$SRC" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512  512  "$SRC" --out "$ICONSET/icon_512x512.png"    >/dev/null
sips -z 1024 1024 "$SRC" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

iconutil -c icns "$ICONSET" -o "$OUT"

# The .iconset is just a scratch directory; remove it once .icns is written.
rm -rf "$ICONSET"

echo "wrote $OUT ($(stat -f%z "$OUT") bytes)"
