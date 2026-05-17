<p align="center">
  <img src="assets/icon-readme.png" width="160" alt="parakeet-rs icon" />
</p>

# parakeet-rs

Native macOS / Apple Silicon dictation menu-bar app. Press a global
hotkey, speak, transcript inserts at your cursor. Fully local — no API
keys, no network after the first-run model download.

- **ASR**: NVIDIA Parakeet TDT 0.6B v3 int8 via sherpa-onnx + CoreML
- **Cleanup (optional)**: Qwen 3.5 2B Q4_K_M via llama.cpp + Metal
- **Shell**: AppKit single binary (no Tauri / Electron)
- **Text injection**: `CGEventKeyboardSetUnicodeString` keystroke

## Running

Apple Silicon Mac, macOS 11.0+, Rust 1.77+.

```bash
cargo run --release                 # dev build
scripts/make-app.sh                 # production .app → /Applications
```

For stable TCC permissions across rebuilds, create a self-signed
"Parakeet Local Dev" cert in Keychain Access, then:

```bash
PARAKEET_SIGN_ID='Parakeet Local Dev' scripts/make-app.sh
```

First launch:

1. macOS prompts for **Microphone**, **Accessibility**, and **Input
   Monitoring**. All three are required.
2. ~640 MB of model files download to
   `~/Library/Application Support/com.parakeet.rs/models/`.
3. Press `⌘⇧Space` (default hotkey), speak. **Tap mode** auto-stops at
   end-of-speech; **Hold mode** stops on release.

### Optional cleanup pass

Flip Cleanup to On in Settings. Fetch the Qwen GGUF (see
`bench/README.md` for the one-liner) into
`~/Library/Application Support/com.parakeet.rs/llm/qwen3.5-2b-q4_k_m/`.
Cleanup strips fillers, fixes punctuation, honours inline commands
("new paragraph", "scratch that"); adds ~550 ms wall-clock but streams
to the cursor on word boundaries.

## Caveats

- **Apple Silicon only.** No plans for a universal binary
  ([ADR-0002](docs/ADR.md#0002--macos-only)).
- **Text injection** works in terminals (Ghostty, iTerm2, Terminal.app),
  browsers, native Cocoa, Electron (Slack/VS Code/etc.), JetBrains,
  Xcode. Doesn't reach password fields or apps with aggressive input
  filtering.
- **Build size**: ~7 MB binary + ~50 MB bundled dylibs (mostly
  onnxruntime and llama-cpp).

## Layout

App state lives behind two small state machines so the
session/cleanup-load races stay localised:

- `src/app.rs` — orchestration, supervised worker spawns, panic recovery
- `src/dictation_fsm.rs` — atomic (state, session, pending_terminate)
- `src/llm_manager.rs` — cleanup-LLM lifecycle (Disabled / Loading / Ready)
- `src/cleanup.rs` — `CleanupBackend` trait + `PromptTemplate` + decode loop
- `src/paste.rs` — `TextSink` trait + word-boundary `Streamer`
- `src/ax_paste.rs` — `CGEvent` keystroke implementation
- `src/streamer.rs` — per-session VAD/manual capture
- `src/{audio,asr,vad,hud,hotkey,menubar,settings,settings_ui,…}.rs`

Two headless benches under `src/bin/`: `bench_asr` and `bench_llm`.

Architectural rationale lives in [`docs/ADR.md`](docs/ADR.md) (decisions
0001-0019); latency targets and measurements in
[`docs/latency-plan.md`](docs/latency-plan.md).

## Verification

```bash
cargo build --release && scripts/make-app.sh
cargo test                                       # 81 unit tests
cargo clippy --all-targets --no-deps             # clean
```

## Roadmap

- Auto-download the cleanup GGUF on first toggle-on (currently manual).
- Wire keyboard shortcut customization into the Settings UI.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)
at your option (Rust ecosystem convention).

Runtime-downloaded models ship under their own licenses: Parakeet TDT
0.6B v3 (NVIDIA), Silero VAD (MIT), Qwen 3.5 2B Instruct (Apache-2.0).
