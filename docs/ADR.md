# Architecture Decision Records — Parakeet Dictation

One file, one ADR per heading. Status legend: **Accepted** (in code today),
**Proposed** (next pass), **Rejected** (considered and dropped),
**Superseded** (replaced by a later ADR).

The overarching goal is in [ADR-0007](#0007--performance-targets-beat-wispr-flow). Every decision
below should be re-evaluated against it.

---

## Current state

Apple Silicon and macOS 14+ are required. Native Core ML workers own ASR and
Silero VAD. Rust owns capture, pause policy, speculative and Hold window
coordination, optional OpenCode rewrite, and Unicode text injection. Custom
vocabulary stays on the native recognition path. Load failures are explicit
setup errors. Runtime placement and short-window choices require the gold
quality and performance gates.

---

## 0001 — Tauri 2 + Rust shell (replacing Electron)

**Status:** **Superseded.** The Tauri shell was dropped in favour of
a single native AppKit binary (`objc2` + per-class `objc2-app-kit`
features) — see [ADR-0016](#0016--native-appkit-shell)
for the spike-was-revisited-and-flipped trail. ADR-0001's text below
records the original Electron → Tauri decision; the codebase no
longer ships any Tauri / WebView code.

**Context.** OpenWhispr ships ~100 MB of Electron + Node + Swift / C / C++
helpers per OS, with ten compiled native side-binaries (`globe-listener`,
`fast-paste`, `mic-listener`, `audio-tap`, etc.) duct-taped over an
Electron-managed JS surface. The dictation hot path bounces between V8 and
native via N-API. That is a lot of trust boundary and a lot of latency.

**Decision.** Tauri 2 with all logic in Rust. WebView only for the settings
window and the recording indicator. No Node, no npm.

**Alternatives.**
- *Native Cocoa / SwiftUI*: cleanest mac integration but locks us into Swift
  and loses the path to other OSes if we ever want them.
- *Electron (status quo)*: works, but ~250 MB RAM, slow cold start, and the
  IPC tax on every dictation.
- *Egui / Slint*: pure-Rust UI, beautiful in theory, but tray + always-on-top
  micro-windows + native dialogs are still rough; ships less than Tauri.

**Consequences.** ~30× smaller resident set than Electron OpenWhispr. JS
appears only on the cold path of opening the settings window. Loss: we don't
get React/Tailwind ergonomics, hand-write vanilla TS — fine for two windows.

---

## 0002 — macOS-only

**Status:** Accepted

**Context.** OpenWhispr ships on Mac, Win, Linux, AppImage, deb, rpm. Most of
its complexity is per-OS native helpers. The development hardware here is
M5 Pro and the target users are mac-first.

**Decision.** Apple Silicon macOS only (Intel deprecated). No `#[cfg]` arms
for other OSes, no cross-platform abstractions.

**Alternatives.**
- *Cross-platform via tauri-cross-platform-shortcut-y libs*: works, but every
  optimization in [ADR-0006](#0006--apple-silicon-execution-policy) becomes
  conditional, which doubles the maintenance burden for no user gain on day
  one.
- *Intel mac support*: M-series is now ubiquitous in our user base.

**Consequences.** Source code is straight-line Rust, no `cfg(target_os)`.
Optimizations target one hardware generation. If we ever ship to Windows or
Linux, this ADR gets explicitly superseded with a porting plan.

---

## 0003 — 100% local inference, no cloud APIs as defaults

**Status:** Accepted

**Context.** Three reasonable cloud paths exist: OpenAI Whisper API,
GPT-Realtime-Whisper ($0.017/min), Microsoft MAI-Transcribe-1 (3.9% FLEURS),
NVIDIA NIM Nemotron. All produce excellent WER. All require network and
external trust.

**Decision.** Speech recognition runs only on the Mac. No API keys, no BYOK
in the settings UI, and no internet call by Parakeet after the first-run
model download. The optional transcript rewrite
([ADR-0036](#0036--optional-transcript-rewrite-through-the-running-opencode-service))
hands the text to a loopback OpenCode service the user already runs, which
may call its own model provider; audio never leaves the Mac.

**Alternatives.**
- *Cloud-only* (Wispr Flow's choice): ~$30/mo at our user's usage profile
  (measured: 4,094 user messages / 30 days in `~/.claude/projects/`), ~$360/yr.
  Plus the cloud round-trip latency that the model latency cuts (200–500 ms
  per request) usually erases on flaky Wi-Fi.
- *Hybrid* (local default + cloud fallback toggle): nice in theory, adds two
  config knobs, an API-key store, and a privacy footnote. Defer until the
  local path is demonstrably insufficient.

**Consequences.** Privacy story is built in. Recurring cost is zero. We give
up the very latest cloud accuracy bumps until they trickle into open models.
Voice data never leaves the device — this is the differentiator vs. every
cloud competitor including Wispr Flow.

---

## 0004 — Parakeet Unified EN 0.6B as the model

**Status:** Accepted

The production recognizer uses the immutable int8 Parakeet Unified EN 0.6B Core ML bundle. The tokenizer, frontend, and greedy decode contract are pinned in `docs/asr/COREML_MODEL.md`. A replacement must clear the human-speech gold gate, preserve punctuation, and fit the local Apple Silicon runtime.

---

## 0005 — Native Core ML inference

**Status:** Accepted

Recognition and Silero VAD run in native worker processes. Rust owns capture, endpoint policy, transcript delivery, and bounded IPC. The app has one production ASR backend; a worker load failure is a setup error. See ADR-0040.

---

## 0006 — Apple Silicon execution policy

**Status:** Accepted

Models stay resident between dictations. Startup warms the ASR graphs. Capture converts audio to 16 kHz once using the local Kaldi resampler. CPU topology and thread QoS remain application concerns; model placement is selected by the evidence-gated tuner in ADR-0026.

---

## 0007 — Performance targets (beat Wispr Flow)

**Status:** **Accepted; the revised latency and representative-speech gates are
met.** See [Current state snapshot](#current-state-vs-target-snapshot).

**Context.** Wispr Flow is the de-facto premium AI dictation app: cloud-only,
$144/yr, claims **<500 ms** felt latency, **95%+ accuracy** in quiet
conditions, AI rewriting with context-aware tone matching, multi-platform.
On flaky Wi-Fi its latency degrades visibly.

We must clear all four of: latency, accuracy, formatting quality, privacy.
Tie the others is acceptable; **privacy is our differentiator**.

**Decision.** Quantitative targets (M5 Pro, after warm-up). Each row carries
its **current measured / asserted baseline** alongside the target so we
never confuse aspiration with engineering:

| Metric | Wispr Flow | Today (baseline) | Target | Path to closing |
|---|---|---|---|---|
| End-of-speech → text appears | <500 ms cloud | **182.0 ms p50 / 203.0 ms p95** in Tap Fast's matched 5 s production replay; pause-friendly Tap remains below 1 s p95 on the endpoint corpus | **<1 s p50** with the representative-speech quality gate intact | met by ADR-0022/0023/0025 |
| First word in indicator | <500 ms | **n/a — indicator removed in [ADR-0014]** | n/a | retired |
| Cold start (launch → first hotkey responsive) | ~2 s | native model load **0.130 s**, first-result path **0.278 s** on the measured M5 Pro; full AppKit launch is not separately claimed | **<3 s** with models present | component evidence passes; keep full-launch measurement honest |
| Representative-speech WER / CER | not published | **5.43% / 3.57%**, zero ten-repeat spread on the seven-file human corpus | frozen product gate **≤8% / ≤5%** with no added baseline error or category regression | met by ADR-0021/0022 |
| Privacy | cloud | zero network calls after installed model artifacts pass local verification | **zero internet calls by Parakeet after the model downloads**; with the opt-in rewrite on, transcripts go through the user's own OpenCode service | [ADR-0003](#0003--100-local-inference-no-cloud-apis-as-defaults) + [ADR-0036](#0036--optional-transcript-rewrite-through-the-running-opencode-service) |
| Smart formatting | yes, cloud LLM | optional rewrite through an already-running OpenCode service; the local transcript is pasted when it fails | **optional formatting pass** | met by [ADR-0036](#0036--optional-transcript-rewrite-through-the-running-opencode-service)/0019 |
| Resident set (steady state) | ~150 MB | ~800 MB asserted | **≤1.2 GB** | met by ADR-0016/0022 |
| Battery cost / 30 min dictation | n/a | unmeasured | **<2% on M5 Pro** | resident CPU+ANE worker and idle QoS policy exist; a controlled energy protocol is still required |

**Honesty notes.**
- The resident-set target was previously set at "<400 MB including model"
  which is arithmetically impossible (the model alone is 640 MB on disk and
  some of those pages will be resident under any access pattern). Corrected.
- The <200 ms p50 latency target was **retired** after the
  [ADR-0009](#0009--silero-endpointing-with-offline-recognition)
  streaming-model survey: no streaming Parakeet TDT 0.6B v3 ONNX exists,
  the realistic substitutes (NeMo FastConformer-streaming-large at 114 M
  params, Kroko Streaming Zipformer at ~50 M) all regress WER on test-other
  and lose native punctuation. Later ADR-0022/0023 retained offline quality but
  removed most of the finalize tail through a faster native model and
  speculative decode. The current acceptance limits are the representative
  corpus's 8% WER / 5% CER gates, not the former published-model proxy.

**Alternatives.** Lower bars (parity with Whisper.cpp dictation tools like
Superwhisper). Rejected — point of the exercise is to beat Wispr Flow.

**Consequences.** Every subsequent performance change is judged against the
representative quality, latency, repeatability, and memory gates recorded by
ADR-0021, ADR-0023, ADR-0025, and ADR-0026.

---

## 0008 — Hotkey press-to-toggle + clipboard paste

**Status:** **Both halves superseded.** Press-twice was replaced by
hotkey-press → talk → VAD-auto-stop per
[ADR-0009](#0009--silero-endpointing-with-offline-recognition). Clipboard +
⌘V was replaced by `CGEventKeyboardSetUnicodeString` synthetic
keystrokes per [ADR-0019](#0019--paste-delivery-synthetic-unicode-keystroke-annotatedsession).
Original v0.1 framing preserved below.

**Context (v0.1).** Press hotkey to start, press again to stop.
Transcript written to clipboard, ⌘V synthesized via enigo.

**Decision (v1, at the time).** Keep clipboard + ⌘V as the paste
path; the 15-50 ms cost and clipboard pollution were judged
tolerable in exchange for breadth of app compatibility. AX injection
([ADR-0011](#0011--direct-accessibility-text-injection-deferred))
deferred to v2.

The press-twice UX was replaced by hotkey-press → talk →
VAD-auto-stop in [ADR-0009](#0009--silero-endpointing-with-offline-recognition)
because press-twice adds a user-input delay on top of inference
latency.

---

## 0009 — Silero endpointing with offline recognition

**Status:** Accepted

Silero is the sole Tap endpoint authority. Speculative recognition can overlap a pause but cannot stop capture. Hold ends on hotkey release and can decode bounded windows during capture. The 32 ms Core ML VAD graph keeps the capture cadence. See ADR-0023, ADR-0025, ADR-0031, ADR-0032, and ADR-0040.

---

## 0011 — Direct Accessibility text injection (DEFERRED)

**Status:** **Superseded by [ADR-0019](#0019--paste-delivery-synthetic-unicode-keystroke-annotatedsession).**
The clipboard clobbering DID become a real user complaint
(2026-05-17), and so did the AX-silently-drops-the-write case in
terminals. The shipped path is synthetic Unicode keystrokes — see
ADR-0019. The text below is the original deferral note from v1, kept
verbatim for the v2 conversation it was written for.

**Context.** Current paste path: write to `NSPasteboard` → enigo synthesizes
⌘ down + V + ⌘ up → target app handles its paste event. Three costs:
~15–50 ms, clipboard pollution, fragile against apps that intercept ⌘V
oddly (Terminal, JetBrains IDEs, some web apps).

**Why deferred.** Codex correctly flagged that AX injection is more involved
than the ~150-line estimate: `kAXValueAttribute` replaces the entire field,
`kAXSelectedTextAttribute` support is inconsistent across Electron / browsers
/ JetBrains / Terminal / secure fields / custom editors, and a working
implementation needs a per-app-class fallback table plus user-permission UX.
Realistic scope is ~300 lines plus ongoing per-app-bug maintenance. The
incremental latency win (~15-50 ms) is not worth taking on that maintenance
surface in v1, especially since [ADR-0009](#0009--silero-endpointing-with-offline-recognition)
delivers larger latency gains for less code.

Kept in the ADR for the v2 conversation. Not on the v1 critical path.

**If/when we revisit:**
- Use AX API to write directly into the focused text element:
  `AXUIElementSetAttributeValue(focusedElement, kAXSelectedTextAttribute, text)`
  for caret-aware insertion, with fallback to `kAXValueAttribute` set on
  text-field-class elements, with fallback to clipboard+⌘V.
- Build a per-app behavior table (Electron, JetBrains, Terminal, web TipTap,
  native NSTextView, secure fields) and decide which path to use per-app via
  bundle ID detection.

---

## 0012 — Native runtime linkage

**Status:** Accepted

The app links its local Kaldi resampler statically. The native worker links the pinned FluidAudio package. The bundle contains the Rust executable, the native worker, and license resources; model weights download separately. No ONNX runtime is linked or bundled.

---

## 0015 — Core ML execution verification

**Status:** Accepted

Core ML stage reports identify execution placement and distinguish native RNNT work from graph dispatch. A runtime plan is accepted only against the quality, memory, and performance gates in ADR-0026. A load failure is surfaced to the user; a low observed RTFx is a performance diagnostic rather than proof of placement.

---

## 0013 — Custom dictionary support

**Status:** Accepted

Vocabulary is an editable plain-text list. Parsing removes comments and blank lines. The native worker tokenizes terms with its own inventory and reports accepted and rejected terms. See ADR-0033.

---

## 0014 — Tray-only headless UX

**Status:** **Superseded in implementation by [ADR-0016](#0016--native-appkit-shell).**
The intended tray-only outcome shipped in native AppKit without Tauri/WebKit;
the body below preserves the original Tauri-era proposal.

**Context.** Current `tauri.conf.json` opens the settings window at launch
(`visible: true`). WebKit init costs ~300–500 ms; for most launches the
user never looks at the settings.

**Decision.** Settings window `visible: false` by default;
lazy-instantiated when the tray menu's "Settings…" is clicked. Indicator
window also dropped — replaced with a recording-state-driven tray icon
variant (red dot when listening). One less WebView at runtime.

**Consequences.** Faster cold start, lower idle RAM, no UI surface unless
the user asks for it. Matches OpenWhispr's `LSUIElement` mode.

---

## 0016 — Native AppKit shell

**Status:** Accepted

The menu bar, HUD, settings, permissions, hotkey, and text injection use native macOS APIs through Rust. A separate Swift worker owns Core ML inference. Keeping model execution behind IPC isolates failures and gives the app and benchmarks the same recognition contract.

---

## 0017 — Core ML model cache ownership

**Status:** Accepted

The native Core ML runtime owns compiled model caching. Rust verifies each downloaded model file before loading and retains a metadata-bound verification cache. App code does not configure an ONNX execution-provider cache.

---

## 0019 — Paste delivery: synthetic Unicode keystroke (AnnotatedSession)

**Status:** **Accepted — shipped.** Supersedes [ADR-0011](#0011--direct-accessibility-text-injection-deferred)
(deferred AX path).

**Context.** Delivery of the transcribed (and optionally rewritten)
text into the focused app was the source of a long bug tail through
2026-05-15/16/17. Each round of fixes exposed the next layer:

1. **Clipboard + `enigo` ⌘V chord** (original): `Enigo::new()` calls
   `TSMGetInputSourceProperty` which asserts main-thread on macOS 26+.
   Our paste runs on the `transcribe` worker thread →
   `EXC_BREAKPOINT`/`SIGTRAP` on every dictation. Bucket A of the
   crash audit (3 reports in one day).
2. **Clipboard + raw `CGEvent` ⌘V chord:** TSM crash fixed, but
   exposed two paste-vs-clipboard races. (a) Write-to-read: the
   `pasteboardd` propagation of `copy_to_clipboard` hadn't reached
   the focused app before our `CGEventPost(⌘V)` did, so the app
   pasted the PREVIOUS dictation's clipboard contents. (b)
   Restore-before-read: `Streamer::commit`'s `restore_clipboard(saved)`
   overwrote the just-written chunk before the focused app dequeued
   the queued ⌘V, causing the same wrong-content-pasted symptom on
   the LAST chunk. Required tunable settle delays before AND after
   each chord (35 ms / 120 ms).
3. **Accessibility-first (`AXUIElementSetAttributeValue(AXSelectedText)`)**:
   the path Apple's own Voice Dictation uses. Worked cleanly for
   Safari, Chrome, Slack, Cursor, TextEdit, Notes. Discovered that
   Ghostty exposes an `AXTextArea` for its rendered scrollback view
   and accepts `AXSelectedText` writes WITHOUT ERROR — but silently
   never routes them to the PTY input pipe. AX success codes are
   therefore unreliable for terminals (and presumably anything else
   with a render-only AX surface). Without out-of-band verification,
   the AX → keystroke fallback chain never fires.

**Decision.** **Synthetic Unicode keystroke only**, posted at the
`AnnotatedSession` event-tap layer:

```rust
let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)?;
let keydown = CGEvent::new_keyboard_event(source.clone(), 0, true)?;
keydown.set_string(text);  // CGEventKeyboardSetUnicodeString
keydown.post(CGEventTapLocation::AnnotatedSession);
let keyup = CGEvent::new_keyboard_event(source, 0, false)?;
keyup.set_string(text);
keyup.post(CGEventTapLocation::AnnotatedSession);
```

The keycode (`0`) is irrelevant because the attached Unicode string
overrides it for text-aware apps. `AnnotatedSession` (rather than
`HID`) is the standard layer text-input frameworks /
`NSResponder` / WebView / PTY-bridge code consumes; HID-level
posting bypasses some terminals' input pipelines.

**Verified working on macOS 26.4.1:**

- Terminals: **Ghostty**, iTerm2, Terminal.app
- Browsers: Safari, Chrome (URL bar + page inputs)
- Native Cocoa: TextEdit, Notes, Mail, Messages
- Electron: Slack, Discord, VS Code, Cursor
- IDEs: Xcode, JetBrains family

**Rejected alternatives:**

- **Clipboard + ⌘V** — race-prone and `enigo`-dependent (TSM crash).
  Settle delays mitigated but didn't fully close the race; a single
  paste path costs ~155 ms of forced sleeps (35 ms ×N flushes plus
  120 ms restore).
- **AX-first with keystroke fallback** — Ghostty's silent-success
  case means the fallback never fires and the user sees nothing.
  Out-of-band verification (read-back `AXValue` or
  `AXNumberOfCharactersAttribute` after the set) is slow,
  race-prone, and many AX elements don't expose character counts.
- **Per-app allowlist (use keystroke for known terminals, AX
  otherwise)** — maintenance burden never ends; new terminals /
  Electron apps with custom input handlers keep appearing. Codex
  pass 8's "drop the fallback entirely" recommendation captured
  the right instinct.

**Out-of-scope failure modes:** password fields (intentionally
reject programmatic input), apps with aggressive input filtering
(some games, accessibility-blocking utilities). We surface
`recover_from_panic` status in the menubar rather than silently
dropping.

**Removed by this ADR:**

- `arboard` and `enigo` direct dependencies (~700 lines of paste
  machinery, settle delays, save/restore, clipboard-dirty
  bookkeeping).
- `Settings::inject_mode` field (was the unused "paste" / "type" /
  "clipboard" debug knob from the clipboard era; serde-unknown-field
  tolerance keeps legacy `settings.json` files parsing cleanly).
- The AX FFI machinery in `ax_paste.rs` (focused-element lookup,
  role/subrole telemetry).
- `Streamer::last_push_at` / `MIN_PASTE_INTERVAL` (the throttle
  was an artifact of ⌘V chord rate-limiting; `CGEventPost` of a
  Unicode keystroke doesn't need it).

**Dependent costs paid:** Accessibility permission preflight still
required (for `CGEventPost`), already checked at startup in
`permissions.rs`. No clipboard mutation means no `RESTORE_SETTLE_DELAY`,
no `PASTEBOARD_SETTLE_DELAY`, and the user's clipboard is left
exactly as they had it.

---

## 0020 — Vocabulary file ownership

**Status:** Accepted

The app owns `vocabulary.txt`, containing one natural-language term or phrase per line. It sends active terms directly to the native worker. No intermediate token file is generated. Vocabulary edits reload the resident recognizer after the current dictation. See ADR-0033.

---

## 0021 — `asr_diff`: transcript regression harness

**Status:** **Accepted — shipped.**

**Context.** `bench_asr` measures how *fast* the recognizer is. Nothing
measured what it *says*. Changes with direct transcription consequences
— int8 weights, CoreML silently falling back to CPU (which ADR-0015
detects only via an RTFx heuristic), contextual biasing, a hotword
score — were landing on the strength of "it still runs and the latency
looks fine". The comparison point is qwen-scribe, which ships a
`compare_models.py` for exactly this and warns to "validate names and
numbers on representative recordings before relying on a quantized
model"; we had int8 weights and no such check.

**Decision.** `src/bin/asr_diff.rs` decodes every `*.wav` in a fixture
directory and either records the transcripts (`--record`) or diffs
against a recorded set, reporting word-level Levenshtein distance per
fixture plus an aggregate divergence percentage. Non-zero exit on any
change, so it can gate.

- Reuses `crate::vocabulary::prepare` rather than reimplementing the
  encoding — a harness that validated a *different* encoding than the
  app ships would be worse than none.
- `read_wav_mono` moved from `bench_asr` into `crate::wav` so both
  harnesses agree on stereo folding; two subtly different parsers would
  manufacture phantom regressions.
- The baseline is gitignored. Fixtures come from macOS `say`, so the
  transcripts depend on the local TTS voice and OS version — a
  committed baseline would fail for everyone except its author.

Three gate holes were closed after adversarial review, all of the same
shape — a difference the harness saw but didn't count:

- **Fixtures not in the baseline** were printed and ignored. An empty
  baseline therefore passed against any number of decoded fixtures: the
  gate exited 0 having verified nothing.
- **Whitespace-only drift** passed, because the comparison was word
  edit distance over `split_whitespace`. Pass/fail is now exact string
  equality; edit distance only sizes the change.
- **Missing fixtures** counted as changed but contributed zero edits, so
  a run where nothing decoded reported "0.00% divergence". A missing
  fixture now costs its full baseline word count, and a zero
  denominator prints "divergence undefined" rather than `0.00%`.

**Consequences.** Accuracy-affecting changes now have a check that
fails loudly. The harness was validated against known-bad
configurations — ADR-0020 measurement 4, an added fixture, and an
unencodable vocabulary term — rather than only against the happy path.

**Unicode-normalization amendment (2026-08-11).** Gold WER/CER scoring now
canonicalizes both sides to NFC before lowercasing and preserves combining
marks that have no precomposed form. This makes visually identical NFC and NFD
transcripts score identically without the broader character changes of NFKC.
The `unicode-normalization` 0.1.25 crate is the narrow dependency for Unicode
Standard Annex #15 behavior; it is MIT OR Apache-2.0, supports Rust 1.36 (below
this crate's Rust 1.77 floor), and avoids adding ICU or an OS-dependent text
runtime.

---

## 0022 — Resident native Core ML Parakeet Unified backend

**Status:** Accepted

The native worker owns the int8 Parakeet Unified model and stays resident across hotkey presses. Rust supplies 16 kHz mono Float32 through a bounded framed protocol. Native RNNT decode and the encoder run against the same pinned graph pack. Vocabulary is applied inside native greedy selection. Model or worker load errors are user-visible setup errors. The app requires Apple Silicon and macOS 14+.

---

## 0023 — Speculative decode behind an unchanged endpoint authority

**Status:** **Accepted — implemented and measured.**

**Context.** ADR-0022 made five-second recognition 5.48× faster, but the live
pipeline remained serial: wait for Silero to confirm trailing silence, stop
capture, then start ASR. Adding the nominal 150 ms endpoint delay to both
backends reduced the projected end-to-end gain below the requested 3×. The
projection was also too weak to support a user-facing latency claim because it
excluded live Core Audio capture, resampling, session shutdown, and the actual
runtime interaction between VAD and inference.

**Decision.** Run two independent Silero states over each 32 ms frame:

1. A candidate state with 32 ms minimum speech/silence starts a provisional
   decode. Any resumed speech invalidates its transcript, including a resumed
   word as short as one VAD frame.
2. A confirming state retains the previous 150 ms configuration and is the
   only state allowed to stop capture. The candidate state cannot shorten the
   existing cutoff-safety policy.

Audio capture continues on its real-time thread during provisional inference.
The VAD watcher catches up from its channel after inference, commits the early
transcript only when the confirming state ends the session, and otherwise runs
the existing final decode fallback. Model load and Core ML compilation remain
outside the per-dictation path through ADR-0022's resident worker.

`PhaseTimer` now records a true `dur_end_to_end_ms` from the estimated acoustic
endpoint rather than calling confirmation-to-paste `dur_post_endpoint_ms`
user-facing. The deterministic harness selects the installed `BlackHole 2ch`
device directly, without mutating system defaults, and overrides that estimate
with Core Audio's predicted playback instant for the fixture's last non-silent
sample. It exercises production capture, resampling, VAD, endpoint, shutdown,
and ASR. It stops at transcript-ready so it cannot type into the user's focused
application; ADR-0019's synchronous Unicode event post is the only excluded
sub-ms step.

**Evidence.** M5 Pro, 24 GB, release build, `5s_48000.wav` (4.854 s), two
warmups, 30 measured repetitions per variant, exact lexical transcript checked
on every repetition:

| pipeline | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| Core ML + speculative ASR | 189.7 ms | 182.0 ms | 203.0 ms | 203.0 ms |
| speedup | **3.23×** | **3.37×** | **3.21×** | **3.24×** |

The optimized range was 181–203 ms versus 569–659 ms for the baseline. The
existing five-file offline quality gate also passed at 2.38% WER / 2.22% CER.
The endpoint tracker unit tests prove that silence before speech cannot stop a
session, a provisional candidate cannot locally confirm before five frames,
and resumed speech invalidates it.

**Consequence.** The requested end-of-speech gain is above 3× at both p50 and
p95 without relaxing the existing stop authority. The early detector may run
more than one decode during a long utterance with internal pauses; those
results are discarded and affect energy use, not visible text.

The 10 s and 20 s macOS `say` fixtures include long pauses that the unchanged
150 ms confirming detector treats as final. That pre-existing multi-sentence
endpoint-policy limitation was subsequently resolved by ADR-0025. The
representative five-second 3× gate remains frozen on the 150 ms Tap Fast policy
so its before/after comparison stays like-for-like.

---

## 0024 — Immutable identities for Rust-managed model artifacts

**Status:** Accepted

ASR graphs, short-window encoder archives, and the Silero Core ML VAD graph have immutable revisions, byte lengths, and SHA-256 digests in `model_fetch.rs`. Downloads stream into temporary files and publish only after verification. Existing files are hashed once, then a metadata-bound cache avoids unnecessary hashing. The Swift worker loads only the verified directory supplied by Rust.

---

## 0025 — Pause-friendly Tap with an explicit low-latency mode

**Status:** **Accepted — implemented and measured.**

**Context.** The original Tap mode committed after 150 ms of VAD silence. A
14.225 s human LibriSpeech fixture contains a reviewed natural 544 ms pause;
the old policy stopped during that pause and discarded the rest of the
utterance. The same failure occurs in the longer synthesized fixtures. Text or
punctuation cannot reliably resolve this ambiguity: a user may pause after a
grammatically complete sentence and continue with another.

**Decision.** Keep the early 32 ms candidate detector from ADR-0023, but make
the independent confirming detector a product policy:

- **Tap** is pause-friendly and waits 750 ms (24 32 ms Silero windows, 768 ms
  after frame quantization). Existing serialized `"tap"` settings migrate to
  this safer behavior without a settings-file rewrite.
- **Tap Fast** is an explicit 150 ms option (five windows, 160 ms after
  quantization) for short commands. It preserves ADR-0023's matched 3× latency
  gate exactly. *(Superseded by [ADR-0031](#0031--tap-fast-confirms-at-90-ms-punctuation-rejected-as-an-endpoint-signal), which measured the
  window down to 90 ms.)*
- **Hold** remains release-to-paste and does not depend on VAD to stop.

The threshold is centralized in `EndpointPolicy` and passed to both the local
candidate tracker and the authoritative Silero instance. This avoids a split
brain where speculation and capture shutdown use different confirmation
windows.

**Gate.** Two human-speech fixtures from LibriSpeech `test.clean` are versioned
under `bench/endpointing/`, with immutable source revision, SHA-256 hashes,
references, conversion method, and CC BY 4.0 attribution. The 3.505 s fixture
covers an ordinary single sentence; the 14.225 s fixture includes the reviewed
544 ms pause. `scripts/bench-endpoint-policy.sh` replays both through the
production Core Audio capture, dual VAD, speculative Core ML inference, and
shutdown path. It fails if playback does not reach the reviewed acoustic end
or if final-pause p95 is at least one second. Transcript quality is intentionally
kept in the separate `asr_diff` gate.

**Evidence.** On M5 Pro 24 GB, release builds with two warmups and 30 measured
repetitions per fixture produced:

| fixture | false stops | p50 | p95 |
|---|---:|---:|---:|
| 3.505 s single sentence | **0/30** | 668.0 ms | 668.0 ms |
| 14.225 s multi sentence | **0/30** | 637.0 ms | 658.1 ms |

The frozen Tap Fast comparison was re-run after the policy split and retained
the accepted 3× target: 589.5 → 182.0 ms p50 (**3.24×**) and 644.8 → 203.0 ms
p95 (**3.18×**), with exact lexical matches across 30 measured repetitions.

**Consequences.** Normal dictation no longer cuts at the representative natural
pause, while users who prefer the former minimum stop latency retain it as a
named mode. Speculative decode overlaps ASR with most of the added confirmation
window, keeping final-pause latency below one second. A pause longer than the
selected threshold is still indistinguishable from an intended endpoint; Hold
mode is the unambiguous choice for rehearsals or unusually long pauses.

---

## 0026 — Evidence-gated per-chip Core ML runtime plans

**Status:** **Accepted — implemented and measured.**

**Context.** Core ML's `MLComputeUnits` values are placement requests, not
portable performance guarantees. A generic model can behave differently across
Apple chip and macOS combinations, while a latency-only tuner could silently
trade away transcription quality, repeatability, or memory. Tuning model
weights locally is unnecessary for this layer: the safe adaptation seam is the
runtime plan around the same immutable model.

**Decision.** Keep CPU+ANE as an unconditional baseline and provide an explicit
full tuner over exactly four plans: CPU+ANE, `all`, CPU+GPU, and CPU-only. It
runs the ADR-0021 human-speech gold corpus, separates utterances at an
eight-second boundary, and records model load, warmup, p50/p95 wall latency,
RTFx, process-tree peak RSS, WER/CER, per-category quality, and output spread.
A challenger must pass every quality gate, be no worse in any category, stay
within 1.25× baseline memory, and improve a median-load-plus-first-decode-
warmup-plus-20-utterance score by at least 5%. Short and long plans may differ
only when each independently wins those gates.

The atomic JSON profile is keyed by chip, architecture, memory, logical and
named performance levels, macOS, backend, complete Core ML artifact-manifest
digest, and tuner version. Startup verifies model bytes first, rejects unknown
fields, recomputes the profile selection from its evidence, and falls back to
CPU+ANE on any missing, stale, unreadable, or invalid profile.
`PARAKEET_ASR_TUNING=off` is the non-destructive escape hatch. The worker
accepts only the four named plans and a 1–60 second regime boundary.

**Evidence.** On the M5 Pro 24 GiB / macOS 26.5.1 target, ten gold repetitions
selected CPU+ANE for both regimes. CPU+ANE versus `all` measured 316.64 versus
315.53 ms for the combined short corpus and 132.87 versus 132.10 ms for the
14.225 s long fixture: below the 5% win floor after startup costs. CPU-only
measured 680.35 / 194.83 ms and 1.26 GiB, versus CPU+ANE's 316.64 / 132.87 ms
and 0.10 GiB.
CPU+GPU failed in Apple's MPSGraph MLIR compiler and remained recorded as a
failed candidate. Shipping quality remained 5.43% WER / 3.57% CER with zero
output spread.

**Consequences.** The architecture can adapt a generic ASR model to later Mac
families without hard-coding marketing assumptions, but this M5 Pro correctly
receives no nominal plan change because none cleared the evidence threshold.
Profiles are inspectable and removable with `tune_asr`; ordinary startup never
runs a benchmark. A model, OS, hardware, backend, or tuner change invalidates
the old result and safely returns to CPU+ANE until the explicit full tuner is
run again.

---

## 0027 — Qwen3-ASR q8 Apple Silicon challenger

**Status:** Accepted

Qwen3-ASR is not a production dependency. The native Parakeet Unified backend remains the shipping recognizer. The evaluation, provenance, and requirements for reconsideration are in `docs/asr/QWEN3_ASR_EVALUATION.md`.

---

## 0028 — Generic ASR with evidence-gated adaptation

**Status:** Accepted

Keep the immutable generic Core ML model. Contextual vocabulary runs in the native worker at score 2.0. Training, distillation, and quantization-aware training require a repeated error class, consented data, separated development and blind-test sets, and the per-category gates in `docs/asr/DOMAIN_ADAPTATION.md`. Hardware runtime plans have a separate identity from semantic adapters.

---

## 0029 — Contextual macOS permission onboarding and recovery

**Status:** **Accepted — implemented.**

**Context.** Parakeet previously requested Microphone, Accessibility, and Input
Monitoring before AppKit installed its delegate, displayed a blocking alert,
and exited when any grant was absent. That front-loaded permissions the user
had not exercised, left no usable menu, duplicated the Input Monitoring
request in the hotkey module, and treated relaunch as the universal recovery
path.

Microsoft ZoomItForMac commit
[`e14bc9b97e784a5208addc8031f9f1c17c6f3a7f`](https://github.com/microsoft/ZoomitForMac/commit/e14bc9b97e784a5208addc8031f9f1c17c6f3a7f)
provides a better architectural pattern: a permission-service state seam,
tri-state AVFoundation authorization, explicit status/actions, direct System
Settings links, a persistent **Check Permissions** command, and refresh when
the app becomes active after a Settings trip.

**Decision.** Adopt that interaction model in native Rust while mapping it to
Parakeet's actual capabilities. Process launch never calls a TCC request API.
First-run onboarding explains and offers only Input Monitoring because the
global hotkey is the only capability needed then. Starting dictation gates and
offers only missing Microphone/Accessibility grants. Stop/cancel remains
available after revocation. The app always stays in the menu bar, and menu
dictation does not depend on Input Monitoring.

The status dashboard is also permanently available from the menu. Microphone
uses AVFoundation's granted/not-determined/denied/restricted states;
CoreGraphics and Accessibility expose only Boolean preflights, so their state
is honestly described as granted/not granted. Determined or granted states
open the service-specific System Settings pane, falling back to the generic
Privacy & Security pane. `applicationDidBecomeActive:` refreshes after a
Settings trip and surfaces granted-to-missing revocation.

Input Monitoring onboarding consumes the same preflight snapshot taken before
the detector threads start. A second `CGPreflightListenEventAccess` racing
`CGEventTapCreate` was observed to transiently report missing on the signed QA
build even though the registration-time preflight and System Settings both
reported granted; using one snapshot prevents false onboarding and relaunch
instructions.

**Differences from ZoomIt.** Parakeet never requests Screen Recording or
Camera; neither capability exists here. Input Monitoring is its launch-time
analog to ZoomIt's Screen Recording. Microphone and Accessibility are tied to
the user's first dictation request. A global event tap created before Input
Monitoring is granted cannot be made live safely with the process-global
detector guards, so the dashboard explicitly asks for one quit/reopen in that
single case and states that menu dictation remains usable meanwhile.

**Consequences.** The app no longer fails closed on missing TCC grants, system
prompts have explanatory user intent immediately before them, and each
published permission state has a recovery action. Pure policy tests cover
scope, every microphone state, Boolean permission actions, revocation, and
deep-link fallback. The signed clean/revoked manual matrix and implementation
map are maintained in [`docs/macos-permissions.md`](macos-permissions.md).

---

## 0030 — One 16 kHz resampler during capture

**Status:** Accepted

Every downstream consumer receives 16 kHz mono Float32. Capture owns one streaming Kaldi windowed-sinc resampler, now statically compiled from `native/resample`. The cutoff is 99% of the lower Nyquist rate, with six zero crossings. Target-rate input passes through. Chunked conversion preserves the filter carry; flush drains the tail. The local kernel retains source notices and preserves the released kernel's ordered 16-product loop and fused scalar tail.

---

## 0031 — Tap Fast confirms at 90 ms; punctuation rejected as an endpoint signal

**Status:** **Accepted — implemented and measured.**

**Context.** After ADR-0030 the Tap path stopped collecting decode savings.
The `phase_timer` line said why: `t_asr_start=4966`, `t_asr_done=5013`,
`t_vad_endpoint=5094`. The speculative decode finished about 80 ms before the
150 ms confirming Silero state released, and `dur_post_endpoint_ms` was 0 to
1 ms. Taking time out of the decode widened that margin instead of shortening
the result. Only the confirmation window moved Tap's number.

**Decision.** Tap Fast confirms after 90 ms of Silero silence instead of 150 ms.
`FAST_CONFIRMATION_MS` is 90; both the confirming Silero state and the candidate
tracker consume audio in 512-sample frames, so the earliest commit is the third
silent frame, 96 ms, exactly as ADR-0025's 150 ms became 160 ms. The shipped
constant is 90 ms and every table here reports measured latency, so 96 ms never
appears as a figure — it is only what the quantizer does with 90. Long-form keeps
750 ms. The resolved window is passed as a value rather than derived from the
policy inside each consumer, so the confirming Silero state, the local candidate
tracker, and the sweep benchmark cannot disagree.

90 ms is where the curve stops. Every 60 ms repetition reports
`t_asr_done == t_vad_endpoint`: `asr.recognize()` runs synchronously on the VAD
watcher thread and blocks frame intake, so below roughly 90 ms the decode is the
floor and a shorter window buys nothing. That decode is bimodal at 54 and 94 ms,
which is the next lever and is not addressed here.

**A punctuation-aware early commit was implemented, measured, and removed.** The
mechanism committed at a shorter window when the provisional transcript ended in
sentence-final punctuation, reopening on resumed speech. Three measurements
killed it:

1. It does not fire on real speech. The model ends the human single-sentence
   fixture `...amidst the tents,` with a comma in all 30 repetitions, leaving
   that row identical to its control at 59.0 ms p50.
2. Where it fires it is worth nothing over a plain shorter window. On the
   synthesized fixture, 150 ms gated at 90 ms and an ungated 90 ms give the same
   distribution over 30 repetitions: 148.5 ms p50 both, means 139.4 and 141.8 ms.
   Their 15-repetition p50s differ, 145.0 against 125.0 ms, because a 15-sample
   median lands on either side of the bimodal decode's cluster boundary; the
   means there differ by 4.4 ms.
3. Its premise is false. Long-form at 750 ms with a 90 ms punctuated window cut
   the multi-sentence fixture 15/15, and the provisional transcript at its
   reviewed 544 ms intra-utterance pause reads `...to greet the arrival of the
   young princess.` The model emits a sentence-final period mid-utterance, which
   is exactly where a pause policy must hold.

So punctuation is not an end-of-utterance signal, and the mechanism is gone
rather than left unreachable. Avenue 3 of the issue — a FluidAudio Parakeet EOU
streaming model, or a prosody feature on the last 300 ms — was not prototyped;
result 3 is the argument for why a real acoustic predictor, not a text cue,
would be the thing to try.

This supersedes ADR-0025's Tap Fast threshold. It does not change ADR-0023's
stop authority: the confirming Silero state remains the only state allowed to
end a recording, and the candidate state still only starts a provisional decode.

**Gate.** `scripts/bench-endpoint-sweep.sh` sweeps the window over three
fixtures and reports two oracles per row, because neither alone is sound.
`false_cuts` counts commits landing before Core Audio's predicted instant for
the fixture's last sample above -80 dBFS; the LibriSpeech fixtures carry room
tone above that floor, so a short window can miss that marker with every word
intact. `mismatches` compares the transcript against the fixture reference and
is the oracle for lost speech. `bench_e2e --tolerate-false-cuts` counts early
commits and continues instead of aborting on the first, which is what a rate
requires, and keeps the Core ML worker warm across repetitions.

**Evidence.** M5 Pro 24 GB, release builds, speculative Core ML, two warmups and
30 measured repetitions:

| fixture | 150 ms | 90 ms | delta p50 |
|---|---:|---:|---:|
| 4.854 s synthesized | 182.0 ms p50 / 183.5 mean | 148.5 ms p50 / 141.8 mean | **-33.5 ms** |
| 3.505 s human | 59.0 ms p50 / 58.1 mean | 13.0 ms p50 / 10.1 mean | **-46.0 ms** |

The human fixture's absolute figures are offset low: Silero calls silence inside
the LibriSpeech room tone that keeps the -80 dBFS acoustic-end marker alive, so
the marker sits later than the speech does. Compare the deltas, not the
absolutes.

False cuts were 0/30 in every row. The issue asked for at least 40 ms off p50:
human speech gives 46.0 ms, the synthesized fixture 33.5 ms, and that fixture's
mean improves by 41.7 ms. The shortfall is the bimodal decode pinning p50 to a
cluster boundary rather than a property of the window.

Both existing gates were re-run unchanged and pass. The long-pause endpoint gate
holds 0/30 false stops at 667.0 ms p50 single and 635.0 ms p50 / 647.5 ms p95
multi. The frozen end-to-end comparison is now pinned to `--confirmation-ms 150`
so it stays like-for-like against the historical numbers regardless of the
shipping window, and reads 594.5 → 182.0 ms p50 (**3.27×**) and 635.1 →
203.6 ms p95 (**3.12×**).

Full tables and the rejected mechanism's rows are in
[`docs/asr/PERF.md`](asr/PERF.md).

**Consequences.** Tap Fast tolerates a shorter hesitation before committing. That
is the mode's stated bargain — ADR-0025 made Long-form the default precisely
because a false stop loses speech — and neither corpus fixture lost a word at
90 ms. Below 90 ms nothing improves until the decode stops blocking the VAD
watcher, which is the work this ADR hands on.

The sweep harness can no longer reproduce the punctuation rows; they are
reachable at the commit named in the change log.

---

## 0032 — Hold decodes 6 s windows during the hold, joined on word agreement

**Status:** **Accepted — implemented and measured.** Release-to-text p50 on the
M5 Pro falls from 104.5 to 66.0 ms at an 8.1 s utterance and from 180.5 to
65.0 ms at 16.6 s, with gold-corpus WER and CER unchanged at 5.43% / 3.57%.

That result was re-measured after the seam-merge fixes and came back identical,
transcript for transcript, so it belongs to the shipped code rather than to the
version that was current when the arms were first run.

It is weaker than it reads, though, and should not be cited without the caveat: six of the seven gold fixtures are under 4.3 s, so at a 6 s cap they
decode as a single window and are identical to the plain path by construction.
Only `librispeech-multi` (14.2 s) is actually cut, and it scored 0.00% WER. The
rest of the multi-window evidence is loopback runs over `say`-generated audio,
not human speech. A corpus with several 15 to 30 s human utterances would settle
it; until one exists, treat the quality claim as "no regression detected on the
evidence available" rather than as a measured equality. Tables and method in
`bench/README.md`; the ledger entry is in `docs/asr/PERF.md`.

**This is not ADR-0009's rejected streaming swap.** The recognizer is
unchanged — the same offline full-attention Parakeet Unified encoder, the same
int8 weights, the same 15 s windows inside the worker; only *when* windows are
submitted and *how* their transcripts are joined changes, so none of the WER
cost that sank the chunked-attention candidates applies here.

**Context.** Hold (press-and-hold) had nothing overlapping its decode.
`run_manual` drained the audio tap and threw the chunks away, and the first
model call happened after the hotkey came up. The g38m instrumentation priced
that: 72.5 ms p50 at a 4.9 s utterance, 125.0 ms at 8.1 s, 192.5 ms at 15.8 s,
of which the ASR call alone was 61, 115.5, and 181 ms. The growth is structural,
not a bug — the encoder costs about 26 ms per 15 s window plus roughly 3 ms per
audio-second in the RNNT loop, and in Hold all of it lands after the endpoint.
Tap does not have this problem: ADR-0023's speculative decode already runs
during the endpoint confirmation window.

**Decision.** Cut the held recording into windows while the key is still down
and decode each one in the background, so that on release only the tail is left.
Voz's recipe for long audio — "cut into 15 s windows at pauses and join on the
words neighbouring windows agree on" — applied incrementally rather than to a
finished file.

- **Where to cut: at a fixed 6 s cap, not at pauses.** This is the one place the
  implementation departs from Voz's description, and it was decided by
  measurement. Cutting at pauses moved the gold corpus from 5.43% to **6.52%
  WER** and 3.57% to 4.62% CER, failing the manifest's zero-regression gate.
  The damage was not at the seams: it was in `commands` (12.20% to 14.63%) and
  `numbers` (6.67% to 10.00%), fixtures two to four seconds long, where a pause
  inside a short utterance split it in half and each half decoded worse than the
  whole. Cutting only at the cap scored **5.43% WER and 3.57% CER, identical to
  the plain single-pass decode** on every category.

  This reproduces a result FluidAudio had already recorded in
  `UnifiedAsrManager.decodedTokens`: silence-aligned window starts measured
  about 1 WER point worse than a fixed stride on the 15 s offline encoder
  (Earnings-22 long-form), with no artifact benefit, which is why their own
  offline path uses a fixed grid. Two independent measurements of the same
  effect is enough to ship against it.

  The pause path is kept and still tested, because it costs nothing to keep and
  the evidence against it is one small corpus plus one upstream note. It is
  turned off by setting `hold_window_min_seconds` equal to the cap, so the cap
  wins every race. Lowering it needs fresh evidence, not preference.

  The cap is what bounds the tail the user waits for, which makes it the
  release-to-text ceiling: 6 s of tail is roughly 26 ms of encoder plus 18 ms of
  RNNT loop, and the measured p50 at a 16.6 s utterance is 65.0 ms.

- **Pause detection, when enabled.** Hold runs one Silero state of its own, the
  low-latency candidate detector, through `EndpointTracker` under the `Fast`
  policy. That tracker fires `Confirmed` once per pause after 160 ms of silence
  and reports the sample where the silence began; it re-arms when speech
  resumes. Tap's endpoint authority is untouched — this tracker chooses window
  boundaries and can never stop a recording.
- **Overlap.** A pause cut hands the confirmation silence to the next window —
  RNNT emission lags the acoustics, so a word spoken just before the pause can
  be timestamped inside it, and the next window needs to contain that audio for
  the merge to see the duplicate. A forced cut lands mid-speech and hands over
  1.5 s, enough that the straddling word is decoded whole by the next window and
  the merge has several words to agree on.
- **How to join.** The worker now answers with `token_spans`: one entry per RNNT
  emission, carrying the detokenized piece and its span in seconds. FluidAudio's
  `transcribeWithTimings` runs the same decode as `transcribe` and reads
  emission frames the greedy decoder already recorded, so this costs only the
  frame-to-seconds conversion. Rust groups the pieces into words on the
  tokenizer's word-start marker and merges two windows by finding the longest
  run of words they agree on inside their shared audio, splicing there. Matching
  is on letters only, case- and punctuation-insensitive, and never on time:
  emission frames are 80 ms apart and the same word is routinely stamped a frame
  apart by two windows. A single shared function word is not accepted as
  agreement.
- **When they agree on nothing** both sides are kept whole, and only a leading
  run of the later window that exactly repeats what the earlier one ended with
  is removed. A disagreement is not evidence that either window is wrong, so
  nothing may be dropped on suspicion; an earlier rule that cut the earlier
  window at the midpoint of the overlap deleted real speech whenever the later
  window had not in fact re-decoded it. Timestamps cannot close this seam
  either, because emission lag lets a word only the later window heard carry an
  earlier stamp than the last word of the earlier one.

  Two costs are accepted here. A word the earlier window truncated
  mid-utterance survives beside the later window's complete copy, because a
  fragment does not match its whole form — a visible stutter, which a reader can
  correct, in place of a deleted clause, which looks like something the speaker
  never said. And that repeat check compares the whole trailing run of the
  earlier window against the whole leading run of the later one, rather than
  only the overlap region, and prefers the longest match, so a phrase the
  speaker genuinely said twice across a seam could be collapsed to one. It needs
  the repetition to fall exactly at the seam and to survive normalization on
  both sides; the probability is low and the alternative — bounding the check to
  the overlap — reintroduces the timestamp dependency the rest of this decision
  removes.

**Rejected: aligning on time instead of text.** Splitting the overlap at a fixed
instant needs no agreement search and is one line. It also cuts words in half
whenever the two windows' emission frames disagree, which they do by
construction. Text agreement is the only rule that can tell a duplicate from two
different words at the same moment.

**Rejected: submitting windows out of order.** The worker protocol serves one
request at a time behind a mutex, so a second concurrent submission would only
queue. One decoder thread with a job queue is what the protocol wants, and
joining it before the session ends keeps a stray decode from holding the worker
while the next session starts. The tail's wait behind an in-flight window is
inherent and shows at p95 rather than p50.

**Failure is always backwards.** Windowing turns itself off, with the reason
logged, when the recognizer reports no word boundaries, when the
configuration is out of range, when Silero will not load, when the audio tap
closes early, or when capture returns something other than 16 kHz. Any window
decode error, and any merge that comes back empty on non-empty audio, abandons
the windowed transcript and lets the app decode the recording in one pass. A
seam bug can cost latency; it must never cost what the user said.

**Configuration.** `hold_windows_enabled` (on), `hold_window_min_seconds` (6.0)
and `hold_window_max_seconds` (6.0) in `settings.json` — equal, so pause cutting
is off.
`bench_e2e --hold-windows off|MIN,MAX` measures both paths through one binary,
and `asr_diff --hold-windows MIN,MAX` decodes the gold corpus the windowed way
so the seam merge is held to the same WER as the plain decode.

## 0033 — Contextual biasing on the native Core ML path

**Status:** Accepted

The native RNNT decoder applies an Aho-Corasick context trie before greedy selection. The worker tokenizes natural-language terms with the model's piece inventory, reports rejected terms and accepted segmentations, and keeps completed entries' earned score out of later retraction. Empty vocabulary leaves the unbiased path intact. Score 2.0 is the shipped setting; changes must clear the gold corpus with no category regression.

---

## 0035 — Self-signed local identity so TCC grants survive rebuilds

**Status:** Accepted. Kata 0bb3.

**Context.** The Input Monitoring "Grant" button in the permission
dashboard silently did nothing on rebuilt installs: no TCC prompt, no
Settings pane, no dashboard refresh, no log line. `scripts/make-app.sh`
signed every bundle ad hoc (`codesign -s -`). An ad-hoc signature has
no stable identity, so TCC saw each rebuild as a brand-new app: the
stored ListenEvent decision for the previous build no longer matched,
`CGRequestListenEventAccess()` returned false without prompting, and
the app was often not even listed in System Settings. The only
recoveries were manual — `tccutil reset ListenEvent com.parakeet.rs` or
adding the app by hand — and they expired on the next rebuild.

**Decision.** Two changes, one per layer of the failure.

1. `scripts/make-app.sh` signs with the self-signed "Code Signing"
   certificate named `Parakeet Local Dev` in the login keychain
   whenever that identity exists, instead of defaulting to ad hoc. A
   self-signed identity yields a stable designated requirement across
   rebuilds, so TCC grants keep matching. `PARAKEET_SIGN_ID` still
   overrides (e.g. for Developer ID distribution builds). When the
   identity is missing the script falls back to ad hoc and prints a
   loud warning that TCC grants will not survive rebuilds, with a
   pointer to the one-time certificate creation steps documented in
   the script header (Keychain Access Certificate Assistant, or
   `openssl` + `security import`; no personal details required).
   Hardened Runtime (`--options runtime`) and entitlements stay
   Developer-ID-only: they are required for notarisation but break
   self-signed and ad-hoc bundles — under Hardened Runtime the dyld
   team-ID check rejects the bundled dylibs (observed as
   `dyld4::prepare` "different Team IDs" aborts on macOS 26.4.1).
2. `src/permissions.rs` distinguishes what a `CGRequestListenEventAccess()`
   false return means. The call reports the *current* access state, not
   whether it prompted: a first request shows the consent alert and still
   returns    false. The Grant path therefore re-checks on the main thread
   ~1.5 s later: the consent alert takes focus away from the app, so
   granted-at-re-check → refresh; app inactive, or an activation was
   observed during the window (the alert appeared and was answered) →
   the alert was live and the armed activation refresh handles the
   return; still not granted, app still active, and no activation
   observed → no prompt appeared (denied, or the stored
   decision went stale after a rebuild) — log at warn level and open
   System Settings → Privacy & Security → Input Monitoring through the
   existing per-permission deep-link with generic fallback, refreshing
   the dashboard only when that deep link fails — exactly mirroring the
   `OpenSettings` branch, so the modal never covers the pane. The re-check
   decision and warning text are pure functions, unit-tested without TCC
   calls. An earlier revision used a persisted requested-once marker file
   as the prompt oracle; review rejected it because the marker survives
   `tccutil reset` (Settings would open over a genuine re-prompt), a
   failed write silences every later click, and a pre-marker cached
   denial gets one silent click.

**Consequences.** A rebuilt-and-reinstalled Parakeet.app keeps its TCC
identity, so an existing Input Monitoring grant keeps working and
`codesign -dr -` stays byte-stable between builds; the recovery
matrix in [docs/macos-permissions.md](./macos-permissions.md) gains
the no-prompt fallback row. A self-signed certificate still cannot be
notarised or pass Gatekeeper on other Macs — distribution keeps
requiring Developer ID + notarisation, unchanged from the notes in
`scripts/make-app.sh`. Machines without the `Parakeet Local Dev`
identity get the old ad-hoc behaviour plus an explicit warning
explaining why their grants do not survive rebuilds.

**Rejected, with reasons.**

- Keeping ad-hoc as the default and documenting `PARAKEET_SIGN_ID`
  only. That is the status quo this ADR exists to end: defaults are
  what rebuilds actually run, and the workaround has been repeatedly
  rediscovered for months.
- Hardened Runtime for the self-signed identity, for "notarisation
  readiness". It breaks launch outright (team-ID mismatch aborts at
  `dyld4::prepare`); local-dev bundles are never notarised anyway.
- Making `tccutil reset` part of the build. It is a destructive,
  machine-wide TCC mutation; the correct fix is a stable identity so
  the reset is never needed.

---

## 0036 — Optional transcript rewrite through the running OpenCode service

**Status:** Accepted (2026-10-31, Kata ahg9).

**Decision.** Transcript cleanup is an optional rewrite done by an agent
harness that is already running on the Mac, not by a model Parakeet loads.
The only supported harness is an OpenCode service (`opencode serve`). The
rewrite is off by default (`rewrite_enabled`), and the model is chosen in
Settings as `provider/model` plus an optional variant (`rewrite_model`,
`rewrite_variant`); empty uses the service's default model.

**Contract** (`src/rewrite.rs`, adapted from Hex `src/opencode_service.rs` and
`src/dictation_processor.rs` at `07e0a05`, MIT):

- Discovery reads `$XDG_STATE_HOME/opencode/service.json` (else
  `~/.local/state/opencode/service.json`), then `service-<channel>.json`. The
  file must be a regular file owned by the user with no group or other
  permissions; its `url` must be loopback `http://`. `GET /api/info` must
  answer with the registered `pid` (and `version` when registered), otherwise
  the registration is stale and OpenCode is treated as not running.
- Requests are HTTP/1.1 over `TcpStream` to that loopback address, with the
  registration's password as HTTP Basic (`opencode:<password>`). That
  password is the service's own loopback credential; Parakeet reads no
  provider key and stores no secret.
- The rewrite is one `POST /api/experimental/generate` with
  `{"prompt", "model": {"providerID", "id", "variant"}}`, falling back to
  `/api/generate` only on HTTP 404. Only HTTP 200 with non-empty `data.text`
  is pasted. A refused connection, a stale registration, the deadline, any
  other status, a `{"message"}` error answer, an unparseable body or empty
  text pastes the local transcript unchanged.
- Parakeet never runs an OpenCode, Claude Code or Codex executable.

**Deadline: 5 s for discovery, `/api/info` and generation together.** The
hotkey is ignored while `DictationState::Rewriting` is shown, so this is the
longest a rewrite can delay the next dictation; it never loses one. Measured
on the M5 Pro on 2026-10-31 against OpenCode 2.0.32 with this prompt and a
25-word transcript, ten runs per row, no other benchmark running (load
average 3.5):

| Model | Variant | Latency |
|---|---|---|
| `zai-coding-plan/glm-5.3-flash` | `low` | 0.98–3.38 s, median 1.45 s |
| `zai-coding-plan/glm-5.3-flash` | default | 6.7–26.6 s, median 9.4 s |
| `opencode/flint-40b-free` (service default) | — | HTTP 503 "Free usage limit reached" |

A reasoning model therefore needs its low variant to fit; the variant setting
exists for that reason, and the model setting exists because the service's
default model on this Mac answers with an error. Hex's 30 s post-processing
default is sized for long prompts and is not used here.

**Other harnesses.** Claude Code 2.1.316 offers no local server a client can
call: non-interactive use is `claude -p`, which starts a process, and
`--remote-control` relays through claude.ai (`claude --help`). Codex 0.166.0
runs a shared app-server daemon on a unix socket
(`~/.codex/app-server-control/app-server-control.sock`, `codex app-server
--help`, marked experimental); its protocol creates persisted agent threads
and turns rather than answering a one-shot generation, so it is not used.

**Consequences.** With the rewrite on, the transcript and the vocabulary terms
leave the Mac through whichever provider OpenCode is configured for
(PRIVACY.md). Local recognition and contextual vocabulary are unchanged. An
old `settings.json` that still carries `polish_mode` loads normally: the
settings struct does not deny unknown keys, so the key is ignored.

---

## 0037 — Short-window encoder buckets 2/5/8 s as verified release assets

**Status:** Accepted (2026-10-31, Kata htdr; pre-warm from Kata hrs0).

**Context.** The offline encoder is compiled at one fixed 15 s window and
FluidAudio zero-pads every utterance to it, so a one-second dictation pays
the whole 26 ms encoder. Kata fgzt built the same NVIDIA checkpoint at
shorter windows and measured them on the M5 Pro (`bench/README.md`,
"Bucketed short-window encoders"): 7.7 ms at 2 s, 9.6 ms at 5 s and 12.2 ms
at 8 s against 26.1 ms at 15 s, with gold-corpus hypotheses byte-identical
to the 15 s arm. The window is a property of the compiled graph, not of the
chip, so one build of each bucket serves every Mac, and Core ML's own
first-load plan compile is the only per-device step.

**Decision.**

- Ship three buckets, 2, 5 and 8 s. 2 and 5 s cover the Hold medians
  (0.9, 2.9 and 4.9 s); 8 s covers 5 to 8 s for one more archive. 12 s is
  not shipped: it measured 24.5 ms, inside 2 ms of the 15 s encoder.
- Host each bucket as one uncompressed tar on this repository's release tag
  `encoder-buckets-v1` (zstd saves 11% on int8 weights, not worth a
  decompressor). `src/model_fetch.rs` pins each archive's SHA-256 and size
  and the SHA-256 and size of every file inside it. A download is hashed
  while it streams, unpacked into a hidden staging directory, refused if
  any entry is outside the manifest, verified file by file, and only then
  renamed into the model directory. A damaged installed bundle is removed.
- The app hands the worker an explicit `--encoder-buckets` list of the
  windows that verified, never filename discovery, so an unverified bundle
  is never loaded. With no bucket the list is `none` and every utterance
  uses the 15 s encoder, as before. If Core ML refuses a verified bucket the
  worker is reloaded without buckets while retaining the stock Core ML encoder.
- Startup loads the 15 s encoder alone. Core ML's compiled-plan cache is
  keyed by the worker executable's name and the model directory's path, not
  by the binary's contents, so it is cold after a first install, a moved
  model directory or a cleared cache, though not after an app update. A cold
  worker load measured 42.0 s with 2/5/8 against 12.0 s without, and warm
  loads 0.81 s and 0.32 s (M5 Pro; the cache-key measurements are in
  `bench/README.md`). Startup therefore never loads a bucket.
- After every boot a background task verifies the buckets, fetches missing
  ones (the first failed download stops that round, so an unreachable host
  costs one request per launch), and loads them in a throwaway worker with
  the resident recogniser's compute units. That load is the cold compile
  when the cache is cold and under a second otherwise, and it fills the plan
  cache. Only then are the buckets approved, the recogniser marked stale,
  and swapped when idle, so the swap is a warm load. Every load dispatches
  one silent utterance per bucket (w − 0.5 s, routing to w) and one past
  the widest bucket before the app reaches Idle.
- The bucket manifest is not part of `coreml_artifact_digest`, which keys
  the tuning profile; `tune_asr` runs with `--encoder-buckets none` until
  Kata 4s2q makes the profile choose buckets.

**Cost.** 1,769,503,176 bytes installed (587,968,514 + 589,810,292 +
591,724,370) on top of the 569 MB stock encoder, plus one archive (about
590 MB) of transient space while a bucket unpacks. Resident memory measured
0.10 to 0.19 GiB in fgzt because Core ML maps the weights.

**Licence.** The buckets are Derivative Models of
`nvidia/parakeet-unified-en-0.6b` under the NVIDIA Open Model License
Agreement. The release carries `NOTICE.txt` with the notice line and
`NVIDIA-Open-Model-License.txt` with the agreement; `THIRD_PARTY_NOTICES.md`
names both. The conversion pipeline (FluidInference `mobius`) and FluidAudio
are Apache-2.0.

**Not covered.** `EncoderBuckets.select` routes on `seconds × 16,000`
samples while a window decodes its frame-rounded part (79,360 samples at
5 s), so an utterance in that gap runs two windows (Kata qgwm). The window
patch is FluidAudio from the moorbrook fork, pinned by revision
([ADR-0038](#0038--fluidaudio-from-a-revision-pinned-moorbrook-fork)).

---

## 0038 — FluidAudio from a revision-pinned moorbrook fork

**Status:** Accepted (2026-11-01, Kata 3yrc).

**Context.** The worker needs two changes to FluidAudio: the offline encoder
window as a parameter instead of a hardcoded 15 s (the encoder buckets of
[ADR-0037](#0037--short-window-encoder-buckets-258-s-as-verified-release-assets)),
and the greedy RNNT decode loop behind a protocol and factory, which
`NativeRnntDecoder` implements. Neither is upstream. Until now every build
cloned upstream at `00a9aa771900ea09c485659663be31019e293e47` into a local
path dependency and applied a checked-in patch, so bare `swift build` and
`swift test` failed on a clean clone.

**Decision.** `native/ParakeetCoreMLWorker/Package.swift` depends on
<https://github.com/moorbrook/FluidAudio>, branch `parakeet-rs-window`,
pinned by revision `94702495056e0fd0ed61c16af31d42d20fccd7b7`. That commit's
parent is upstream `00a9aa77`, and its diff is exactly the former patch (four
files, +112/−40). The fetch-and-patch script, the patch and the local
checkout are gone; SwiftPM resolves the fork like any other dependency.

- Moving the pin, to a rebased fork commit or back to upstream if the change
  lands there, takes a new ADR.
- Licence: Apache-2.0, unchanged. `THIRD_PARTY_NOTICES.md` names the fork and
  the modifications by parakeet-rs.

---

## 0039 — Per-Mac choice of encoder buckets and Hold window on the gold gate

**Status:** Accepted (2026-11-01, Kata 4s2q).

**Context.** [ADR-0037](#0037--short-window-encoder-buckets-258-s-as-verified-release-assets)
ships 2, 5 and 8 s encoder buckets, and
[ADR-0032](#0032--hold-decodes-6-s-windows-during-the-hold-joined-on-word-agreement)
fixes the Hold window at 6 s. Whether a bucket pays for its disk, and which
window is fastest, depends on the chip's cost for each encoder window and
for switching between them. The window is also a quality choice, because it
decides where Hold cuts and merges. `tune_asr` already chooses compute units
per Mac ([ADR-0026](#0026--evidence-gated-per-chip-core-ml-runtime-plans)),
keyed by chip, memory, macOS and the artifact digest.

**Decision.**

- After the compute-unit pass, `tune_asr` measures every hosted bucket and
  the 15 s encoder under the short-regime units, which every Hold request
  runs on. Each encoder is loaded alone with the stage profiler and given a
  1.5 s probe; its encoder-stage p50 is its cost, which does not depend on
  audio length because the encoder pads to its window. The 15 s encoder also
  runs a 7.5 s probe, which gives the per-second cost of mel and decode. A
  plan whose regimes use different units, with a long-regime threshold that
  a candidate window reaches, is refused: those windows would run on units
  the tuner does not measure.
- It also measures, for every ordered pair of encoders, the extra cost of a
  request on one after 8 s of the other, against a control that idled on
  itself. On the ANE an encoder left idle while another runs is slower on
  its next request.
- From those numbers it predicts the release-time ASR cost of every bucket
  subset that fits the disk budget (`--bucket-budget-mb`, default 1800, room
  for all three) at every Hold window in {3, 4, 5, 6, 8} s, pause cutting
  off. The prediction averages, with equal weight, over utterances from
  0.5 to 20 s in 0.25 s steps. The tail and the last background window come
  from the session's own `WindowPlanner` and the encoder from the worker's
  routing rule. A tail in the 5 s bucket's frame-rounding gap is charged two
  passes (Kata qgwm), and a tail whose encoder differs from the previous
  one (the last background window's, else the 0.5 s keydown primer's) is
  charged the measured switch. The workload is fixed in the tuner and never
  learned from the user's dictations.
- The gold gate then runs those configurations, fastest predicted first.
  A configuration passes only when the SHA-256 of its gold outputs equals
  the baseline's, over every repetition: whole-utterance and Hold-windowed,
  against the 15 s encoder alone at 6 s. The first that passes is chosen.
  A configuration less than 5% faster than the baseline is never tried, and
  the baseline needs no gate.
- The profile gains an optional `window_tuning` section: the measured costs
  and switch costs, each bucket's archive digest and size, the budget, every
  gate verdict and the selection. Loading recomputes the selection from the
  evidence, as for compute units. The cache key adds the chosen buckets'
  archive digests and the chosen window only when the section exists, so a
  profile written before this ADR keeps its key and means what it meant:
  every hosted bucket at 6 s. A chosen bucket that has since been
  re-released drops only the window choice; the compute units stay.
- The app reads the profile once, at launch, and takes the compute units,
  buckets and window from that one read, because they passed the gate
  together; a profile written while the app runs applies at its next
  launch. It removes bundles of buckets the plan did not choose, fetches
  and pre-warms the chosen ones, and loads a tuned set whole or not at all. Hold sessions use the tuned window only while
  exactly the tuned set is loaded; otherwise they use 6 s, the configuration
  every gate compares against. `hold_window_min_seconds` and
  `hold_window_max_seconds` leave `settings.json`; the profile, else 6 s, is
  the only source.
- Profiles are read with serde_json's `float_roundtrip`. Without it a
  measured float could parse one ULP off, the recomputed selection would
  differ, and the app ignored the profile.

**Result on the M5 Pro (24 GB, macOS 26.6.2, 2026-11-01).** CPU+ANE for both
regimes, buckets 5 and 8 s (1.18 GB), Hold window 6 s: predicted release ASR
16.6 ms against 31.3 ms. Every faster configuration tried, all at 3, 4 or
5 s, changed the gold output.
The 2 s bucket loses because the keydown primer runs on it and every longer
tail then pays 12 to 25 ms to switch. Gold for the choice: 10 repetitions,
WER 5.43%, CER 3.57%, 0 changed outputs, Hold-windowed output byte-identical
to the 15 s encoder. Release to text in the app's primed Hold, 5+8 against the
15 s encoder alone: 36 against 53 ms p50 at 0.9 s of speech, 48 against
64 ms at 8.2 s, 67 against 65 ms at 16.6 s. Costs and the full table are in
`bench/README.md`, "Tuned buckets and Hold window".

**Not covered.**

- The gold corpus has one fixture longer than 6 s, so the window half of the
  gate rests on that file. A window other than 6 s can win only with
  byte-identical output; a WER-no-worse gate like the compute units' would
  admit more and is a separate decision.
- The primer warms only the 2 s bucket. Priming the encoder the tail will
  use is not tried here.

---

## 0040 — Core ML ASR and frame-sized Silero VAD

**Status:** Accepted

The app uses one native Core ML recognition backend and requires macOS 14+.
A missing worker, invalid model, or load failure produces a clear setup error.
No backend-selection environment variable, generated hotword file, ONNX model
download, or ONNX runtime library is part of the product. The ASR facade keeps
shared metrics, warmup, token-span handling, and evaluation adapters.

Silero VAD uses FluidInference's v6.0.0 32 ms Core ML graph at revision
`b419383c55c110e2c9271fa6ee0ea83d03c70d96`. Each detector owns a dedicated
worker with a fresh FluidAudio `VadState`. ASR work cannot block VAD IPC.
The adapter supplies 512 new samples plus 64 context samples directly;
FluidAudio's `VadManager` targets a fixed 4096-sample graph and cannot serve
the frame-sized endpoint contract. Both probability thresholds, minimum speech
duration, and candidate/confirming silence windows stay explicit in Rust.
Window-tuning keys include the VAD manifest digest, so a quality verdict from
another detector cannot select Hold windows. Compute-only keys stay independent
of endpointing.

The local Kaldi resampler is statically linked with its notices retained.
Its ordered product accumulation preserves the distributed kernel's rounding.
Packaging copies only the Rust executable, native worker, and notices.

---

## Target status

Production recognition, vocabulary, and endpointing use native Core ML.
Latency and quality commands are in the README and `bench/README.md`.
Historical evidence does not define a current latency guarantee.
