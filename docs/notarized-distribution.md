# Notarized distribution (deferred — needs paid Apple Developer account)

Parakeet currently ships **ad-hoc signed**, so first launch requires the
Gatekeeper bypass (right-click → Open → Open Anyway). That is fine for a
build-from-source, single-user app. This doc records how to remove that
wart *if* the app ever becomes download-facing.

It is not wired up. Doing so requires a **paid Apple Developer account**
($99/yr) — without a `Developer ID Application` certificate none of the
below is reachable. Deferred deliberately: the app is personal-use only.

`scripts/make-app.sh` already does the hard parts (hardened runtime,
`entitlements.plist`, the `Developer ID Application:` codesign branch,
`spctl` assessment). The only missing piece is automating
notarize → staple → re-verify, which today lives as dead comments at the
tail of that script.

## One-time setup (GUI / interactive, once)

1. **Enroll** in the paid Apple Developer Program.
2. **Create a Developer ID Application certificate.** Xcode →
   Settings → Accounts → your Apple ID → Manage Certificates → + →
   *Developer ID Application*. This installs the cert **and its private
   key** into the login keychain. The private key does the signing and
   **cannot be re-downloaded** — back up the keychain. (This is separate
   from the *Apple Development* identity used for on-device debugging.)
3. **Store a notary credential once**, interactively (the prompt cannot
   be scripted away, and this keeps the app-specific password out of
   shell history):
   ```bash
   xcrun notarytool store-credentials Parakeet \
     --apple-id "you@example.com" --team-id YOURTEAMID
   # paste an app-specific password when prompted
   ```
   The app-specific password is **not** your Apple ID password —
   generate one at appleid.apple.com → Sign-In & Security. It goes stale
   silently whenever you change your Apple ID password; a `401 invalid
   credentials` from notarization means "make a fresh app-specific
   password," not "setup is broken." Confirm with:
   ```bash
   xcrun notarytool history --keychain-profile Parakeet
   ```

## Keep credentials out of the public repo

`github.com/moorbrook/parakeet-rs` is public. The team ID and Apple ID
must never be committed. Pass them via env vars or a gitignored file:

`.gitignore` already ignores `.env` / `.env.*` but **not**
`scripts/release.env`, so either name the file to match the existing rule
or add the path first:

```bash
echo 'scripts/release.env' >> .gitignore   # do this BEFORE creating the file
```

Then create `scripts/release.env` locally (never committed):

```bash
export PARAKEET_SIGN_ID="Developer ID Application: Your Name (YOURTEAMID)"
export PARAKEET_NOTARY_PROFILE="Parakeet"
```

## release.sh recipe (copy into scripts/release.sh when ready)

Wraps `make-app.sh` (which produces the signed bundle) and adds the
notarization chain. Fails loud, pre-flights before the slow build, and
re-verifies the *installed* bundle — a `cp -R` can silently mangle a
bundle, and you want the script to catch it, not Gatekeeper three days
later.

```bash
#!/usr/bin/env bash
# scripts/release.sh — Developer-ID-signed, notarized Parakeet.app → /Applications.
# Requires (one-time): paid Developer Program, a Developer ID Application cert
# in the login keychain, and a notarytool credential profile.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PARAKEET_SIGN_ID:?set PARAKEET_SIGN_ID to your 'Developer ID Application: ...' identity}"
NOTARY_PROFILE="${PARAKEET_NOTARY_PROFILE:-Parakeet}"
APP="target/release/bundle/osx/Parakeet.app"
INSTALL="/Applications/Parakeet.app"

step() { printf "\n\033[1;36m▸ %s\033[0m\n" "$*"; }
fail() { printf "\n\033[1;31m✗ %s\033[0m\n" "$*" >&2; exit 1; }

step "Pre-flight (before the slow build)"
xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1 \
  || fail "notary profile '$NOTARY_PROFILE' missing — run: xcrun notarytool store-credentials"

step "Building Developer-ID-signed bundle"
PARAKEET_SIGN_ID="$PARAKEET_SIGN_ID" scripts/make-app.sh   # hardened runtime + entitlements
[ -d "$APP" ] || fail "bundle not found at $APP"

step "Notarizing (uploads to Apple, may take a few minutes)"
ditto -c -k --keepParent "$APP" build/notarize.zip
xcrun notarytool submit build/notarize.zip --keychain-profile "$NOTARY_PROFILE" --wait

step "Stapling ticket"
xcrun stapler staple "$APP"

step "Verifying exported bundle"
spctl --assess --type execute --verbose=2 "$APP"

step "Installing to $INSTALL"
pkill -x Parakeet 2>/dev/null || true
rm -rf "$INSTALL"
cp -R "$APP" "$INSTALL"

step "Re-verifying the INSTALLED bundle (not just the exported one)"
xcrun stapler validate "$INSTALL"
spctl --assess --type execute --verbose=2 "$INSTALL"
printf "\n\033[1;32m✓ Parakeet notarized, stapled, installed.\033[0m\n"
```

## Why each step matters

- **Pre-flight before build.** A missing notary profile should fail in
  one second, not after a full release build.
- **`store-credentials`, not `--password` in argv.** The tail comment in
  `make-app.sh` currently shows `--password APP_SPECIFIC_PASSWORD` on the
  command line, which leaks into shell history. The keychain profile
  avoids that. Worth fixing that comment regardless of this doc.
- **`--options runtime --timestamp` + entitlements.** Already handled by
  `make-app.sh`'s `Developer ID Application:` branch. Notarization
  rejects bundles without hardened runtime.
- **Staple, then re-verify installed.** Stapling attaches the notary
  ticket so Gatekeeper trusts the app offline. For an `LSUIElement`
  menu-bar app, this is what keeps XProtect from flagging it.

## Optional: CLAUDE.md build-path hint

If added, a `CLAUDE.md` should tell the agent the two build paths so it
picks the right one without being re-told each session:

```markdown
## Build & release
- Fast local build (ad-hoc, Gatekeeper-bypass on first launch):
  `scripts/make-app.sh`
- Notarized release (needs Developer ID + notary profile):
  `scripts/release.sh` — archive → notarize → staple → install.
```
