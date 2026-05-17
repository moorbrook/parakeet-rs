# parakeet-rs

A focused **native macOS / Apple Silicon** dictation menu-bar app built
around NVIDIA's **Parakeet TDT 0.6B v3** running locally via
sherpa-onnx + CoreML, with an optional in-process LLM cleanup pass
(**Qwen 3.5 2B Q4_K_M** via llama.cpp + Metal). Fully local — no API
keys, no network calls after the first-run model download.

Press a global hotkey, speak, get the transcript inserted at the
cursor.

Originally derived from the
[OpenWhispr](https://github.com/OpenWhispr/openwhispr) Electron app
(reimplementation, not a port — only the UX shape is shared).

## Stack

| Layer                      | parakeet-rs                                             |
| -------------------------- | ------------------------------------------------------- |
| Shell                      | Native AppKit single binary (no Tauri / Electron)       |
| Language                   | Rust (macOS-only)                                       |
| AppKit bindings            | `objc2` + per-class `objc2-app-kit` features            |
| ASR                        | NVIDIA Parakeet TDT 0.6B v3 int8 via sherpa-onnx        |
| ASR acceleration           | CoreML provider → ANE / GPU / CPU fallback              |
| VAD (auto-stop)            | Silero VAD via sherpa-onnx                              |
| Audio capture              | cpal (CoreAudio)                                        |
| LLM cleanup (optional)     | Qwen 3.5 2B Q4_K_M via llama-cpp-2 (Metal backend)      |
| Hotkey                     | `CGEventTap` (HID) + `NSEvent` global monitor for media keys |
| Text injection             | `CGEventKeyboardSetUnicodeString` keystroke (ADR-0019)  |
| Settings UI                | Native `NSWindow` + `NSTextField` / `NSPopUpButton`     |
| Build                      | `cargo bundle` + `scripts/make-app.sh`                  |

## Why Parakeet TDT 0.6B v3

This is the model that actually wins for English dictation on Apple
Silicon after pulling the latest benchmarks:

| Model | LibriSpeech-clean WER | Size on disk | Apple Silicon path | Punctuation |
|---|---|---|---|---|
| Canary-Qwen 2.5B | 1.6% (#1) | ~2.5 GB | ONNX exists, sherpa-onnx unsupported | yes |
| **Parakeet TDT 0.6B v3** | **1.93%** | **640 MB int8** | **sherpa-onnx, prebuilt int8** | **yes, native** |
| Whisper Large v3 | ~2% | ~1.5 GB | sherpa-onnx | no, post-process |
| Omnilingual ASR 1B int8 | not on English leaderboard | 1.0 GB | sherpa-onnx | no |

Wins on the four axes that matter for press-to-talk dictation: low
English WER, smallest size, native sherpa-onnx support, and **native
punctuation + capitalization**. Multilingual coverage drops from
~1,600 languages to ~25 European, fine for almost any practical
press-to-talk use case.

## What "really optimized for this Mac" means here

Borrowing the [ds4](https://github.com/antirez/ds4) playbook, applied
to the ASR path:

1. **CoreML execution provider** — `OfflineModelConfig.provider = "coreml"`,
   so ops route to ANE / Metal where supported, with CPU fallback per op.
2. **Page-touch warmup at startup** — `mmap` the .onnx file and read
   one byte per 16 KiB page (`warmup::page_touch`), exactly like ds4's
   `kernel_touch_u8_stride`. Pulls all 640 MB of weights into the
   page cache before the first hotkey press.
3. **Pre-warm the graph** — run one 0.5 s silent decode at startup
   (`warmup::dummy_decode`) so CoreML compiles & caches its compute
   graph before any real audio arrives.
4. **Single recognizer for the app's lifetime** — `Asr` lives in
   `App`, reused across every press. Zero re-init, zero graph
   recompile per dictation.
5. **Performance-core scheduling** — capture and recognition threads
   land at `QOS_CLASS_USER_INTERACTIVE` via
   `pthread_set_qos_class_self_np` (`src/qos.rs`). Apple's scheduler
   keeps them off the E-cores.
6. **`sherpa-onnx` num_threads = P-core count** — read via
   `sysctlbyname("hw.perflevel0.logicalcpu")`.
7. **Zero-copy audio path** — cpal f32 samples go straight to
   `OfflineStream::accept_waveform`. No temp WAV file, no `hound`
   round-trip, no `Vec<u8>` re-encode.
8. **int8 quantized weights** — 0.6B params, ~640 MB on disk vs
   ~2.4 GB fp32.

## Layout

```
parakeet-rs/
├── Cargo.toml              sherpa-onnx, llama-cpp-2, cpal, objc2-*, core-graphics
├── build.rs                CoreML EP symbol check via nm
├── entitlements.plist      Mic + Apple Events (for keystroke injection)
├── assets/icon.icns
├── scripts/
│   ├── make-app.sh         cargo bundle + dylib copy + rpath rewrite + codesign
│   ├── bench-latency.sh    bench_asr orchestration → bench/baseline.csv
│   └── bench-aggregate.py  log → CSV reducer
├── bench/
│   ├── README.md           latency-bench instructions
│   └── cleanup-backends.csv  Phase-0 cleanup-LLM numbers (ADR-0018)
├── docs/
│   ├── ADR.md              architectural decisions, 0001-0019
│   └── latency-plan.md     sub-1 s acceptance plan
└── src/
    ├── main.rs             entry: permissions preflight, NSApplication setup
    ├── lib.rs              module declarations + crate-wide lint policy
    ├── app.rs              App orchestration, supervised worker spawns,
    │                       deliver_cleaned pipeline, panic recovery
    ├── app_delegate.rs     NSApplicationDelegate (didFinishLaunching, reopen)
    ├── asr.rs              sherpa-onnx OfflineRecognizer wrapper
    ├── audio.rs            cpal capture, mono fold, audio-level publish
    ├── ax_paste.rs         CGEvent keystroke text insertion (ADR-0019)
    ├── cleanup.rs          CleanupBackend trait, LlamaCleanup,
    │                       PromptTemplate, generate loop
    ├── dictation_fsm.rs    atomic state machine for state/session/
    │                       pending_terminate transitions
    ├── hotkey.rs           CGEventTap + NSEvent media-key monitor
    ├── hud.rs              recording-state HUD panel + waveform bars
    ├── llm_manager.rs      single-mutex cleanup-LLM lifecycle
    │                       (Disabled / Loading / Ready)
    ├── menubar.rs          NSStatusItem + menu actions
    ├── model_fetch.rs      first-run HF model downloader
    ├── objc_util.rs        selector_guard panic boundary
    ├── paste.rs            TextSink trait + AxKeystrokeSink + Streamer
    │                       (word-boundary buffered delivery)
    ├── performance.rs      PhaseTimer, P-core count, session_id
    ├── permissions.rs      mic + accessibility + input-monitoring preflight
    ├── qos.rs              pthread_set_qos_class_self_np
    ├── settings.rs         JSON store (atomic save), model paths
    ├── settings_ui.rs      Settings window
    ├── sf_symbol.rs        SF Symbol → NSImage loader
    ├── streamer.rs         per-session VAD/manual capture orchestration
    ├── vad.rs              Silero VAD wrapper
    ├── warmup.rs           page_touch + dummy_decode
    └── bin/
        ├── bench_asr.rs    headless ASR latency bench
        └── bench_llm.rs    headless cleanup-LLM latency bench
```

## Running

Requires Rust 1.77+, Apple Silicon Mac, macOS 11.0+.

```bash
# Dev build (cargo run):
cargo run --release

# Production .app bundle:
scripts/make-app.sh
# Drop target/release/bundle/osx/Parakeet.app into /Applications.
```

For TCC-entry stability across rebuilds (so macOS doesn't treat each
ad-hoc-signed bundle as a different app), generate a self-signed
"Parakeet Local Dev" code-signing cert in Keychain Access and:

```bash
PARAKEET_SIGN_ID='Parakeet Local Dev' scripts/make-app.sh
```

First launch:

1. macOS prompts for Microphone, Accessibility, and Input Monitoring
   in System Settings. All three are required (Accessibility powers
   keystroke injection; Input Monitoring powers the global hotkey).
2. The Parakeet TDT v3 int8 triplet (encoder/decoder/joiner) +
   tokens.txt + Silero VAD download to
   `~/Library/Application Support/com.parakeet.rs/models/`. ~640 MB.
3. Page-touch warmup runs (~1 s). Dummy decode bakes the CoreML
   graph.
4. The menu bar shows the Parakeet icon. Hotkey is live.
5. Press `⌘⇧Space` (default), speak. With **Tap mode** (default), VAD
   auto-stops at end-of-speech and inserts the transcript at the
   cursor. With **Hold mode**, press-and-hold; release inserts.

Optional cleanup pass: in Settings, flip Cleanup to **On**. The
~1.2 GB Qwen 3.5 2B Q4_K_M GGUF needs to be present at
`~/Library/Application Support/com.parakeet.rs/llm/qwen3.5-2b-q4_k_m/Qwen3.5-2B-Q4_K_M.gguf`
— see `bench/README.md` for the fetch one-liner. The cleanup pass
adds ~550 ms wall-clock latency but streams output to the cursor on
word boundaries (perceived latency much lower).

## Honest caveats

- **Apple Silicon only (arm64)**: bundled `libsherpa-onnx-c-api.dylib`
  and `libonnxruntime.dylib` are arm64-only — Microsoft doesn't ship
  an x86_64 build of onnxruntime with CoreML EP enabled.
  `scripts/make-app.sh` runs `lipo -archs` on the release binary and
  refuses to bundle anything that isn't arm64-only. No plans for a
  universal binary. See [ADR-0002](docs/ADR.md#0002--macos-only).
- **CoreML EP linkage and engagement (verified)**: `sherpa-onnx-sys`
  uses the `shared` feature instead of the default `static` because
  the upstream static archive ships a CPU-only `libonnxruntime.a`
  (sherpa-onnx hardcodes `-DSHERPA_ONNX_DISABLE_COREML` in its
  static-onnxruntime cmake path). `build.rs` runs `nm -gU` on the
  linked dylib at every build and warns (or errors, with
  `PARAKEET_REQUIRE_COREML=1`) if the CoreML EP symbol is missing.
  **Runtime measurement on M5 Pro: 2.0 s of audio decodes in 0.258 s
  = 7.8× real time** — confirms ANE/GPU engagement.
- **Text injection works in most apps but not all**: keystroke-based
  via `CGEventKeyboardSetUnicodeString` at the `AnnotatedSession` tap
  layer (ADR-0019). Verified working in terminals (Ghostty, iTerm2,
  Terminal.app), browsers (Safari/Chrome), native Cocoa (TextEdit,
  Notes, Mail, Messages), Electron (Slack, Discord, VS Code, Cursor),
  JetBrains, Xcode. Doesn't reach password fields (intentional) or
  apps with aggressive input filtering.
- **First-decode latency**: the warmup pays a one-time CoreML
  graph-compile cost (you'll see ~13 `"Context leak detected,
  CoreAnalytics returned false"` lines from macOS — harmless
  lifecycle logs, not errors). After warmup, decodes run at the
  steady-state 7.8× RTFx rate.
- **Build size**: release binary ~7 MB; bundled dylibs add ~50 MB
  (mostly `libonnxruntime` at ~25 MB and `llama-cpp-2`-built libs).

## Verification

```bash
cargo build --release && scripts/make-app.sh    # ~60 s cold, then <5 s incremental
cargo test                                       # 81 unit tests
cargo clippy --all-targets --no-deps             # clean
```

End-to-end runtime test is up to you — needs a live mic and granted
permissions.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option. This matches the Rust ecosystem convention. Unless
you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project shall be dual-licensed as
above, without any additional terms or conditions.

The runtime-downloaded models ship under their own licenses:

- **Parakeet TDT 0.6B v3** (NVIDIA) — see
  [the model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Silero VAD** — MIT
- **Qwen 3.5 2B Instruct** (cleanup pass) — Apache-2.0
  ([model card](https://huggingface.co/Qwen/Qwen3.5-2B-Instruct))
