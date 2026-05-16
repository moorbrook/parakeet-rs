# parakeet-rs

A focused Tauri 2 + Rust dictation app for Apple Silicon, built around
NVIDIA's **Parakeet TDT 0.6B v3** running locally via sherpa-onnx + CoreML.
**macOS only**. 100% local — no API keys, no network calls after the
first-run model download.

Press a global hotkey, speak, get the transcript pasted at the cursor.

Originally derived from the
[OpenWhispr](https://github.com/OpenWhispr/openwhispr) Electron app
(reimplementation, not a port — only the UX shape is shared).

## Stack

| Layer        | OpenWhispr (upstream)                     | parakeet-rs                                     |
| ------------ | ----------------------------------------- | ----------------------------------------------- |
| Shell        | Electron 41                               | Tauri 2                                         |
| Backend lang | Node + Swift / C / C++ helpers per OS     | Rust (mac only)                                 |
| Inference    | whisper.cpp / Parakeet (sherpa-onnx)      | **NVIDIA Parakeet TDT 0.6B v3 int8** via sherpa-onnx |
| Acceleration | depends on engine                         | CoreML provider → ANE / GPU / CPU fallback     |
| UI           | React 19 + Tailwind v4                    | Vanilla TS + CSS                                |
| Audio        | Web Audio + native mic listeners          | cpal (CoreAudio)                                |
| Input inject | Native fast-paste binaries                | enigo + clipboard plugin (⌘V)                   |
| Tooling      | npm                                       | bun                                             |

## Why Parakeet TDT 0.6B v3

This is the model that actually wins for English dictation on M5 Pro after
pulling the latest benchmarks. We initially picked Meta Omnilingual ASR
(highest profile recent release, 1600 languages), then re-evaluated:

| Model | LibriSpeech-clean WER | Size on disk | Apple Silicon path | Punctuation |
|---|---|---|---|---|
| Canary-Qwen 2.5B | 1.6% (#1) | ~2.5 GB | ONNX exists, sherpa-onnx unsupported | yes |
| **Parakeet TDT 0.6B v3** | **1.93%** | **640 MB int8** | **sherpa-onnx, prebuilt int8** | **yes, native** |
| Whisper Large v3 | ~2% | ~1.5 GB | sherpa-onnx | no, post-process |
| Omnilingual ASR 1B int8 | not on English leaderboard | 1.0 GB | sherpa-onnx | no |

Parakeet TDT 0.6B v3 wins on the four axes that matter for press-to-talk
dictation: low English WER, smallest size, native sherpa-onnx support, and
**native punctuation + capitalization** (Omnilingual outputs raw character
sequences, so you'd need a post-processing step). Multilingual coverage drops
from 1,600 languages to 25 European, which is fine for almost any practical
use case.

## What "really optimized for this Mac" means here

Borrowing the [ds4](https://github.com/antirez/ds4) playbook, applied to the
ASR path:

1. **CoreML execution provider** — `OfflineModelConfig.provider = "coreml"`,
   so ops route to ANE / Metal where supported, with CPU fallback per op.
2. **Page-touch warmup at startup** — `mmap` the .onnx file and read one byte
   per 16 KiB page (`warmup::page_touch`), exactly like ds4's
   `kernel_touch_u8_stride`. Pulls all 1 GB of weights into the page cache
   before the first hotkey press.
3. **Pre-warm the graph** — run one 0.5 s silent decode at startup
   (`warmup::dummy_decode`) so CoreML compiles & caches its compute graph
   before any real audio shows up.
4. **Single recognizer for the app's lifetime** — `Asr` lives in `AppState`,
   reused across every press. Zero re-init, zero graph recompile per dictation.
5. **Performance-core scheduling** — capture thread (`audio.rs`) and
   recognition thread land at `QOS_CLASS_USER_INTERACTIVE` via
   `pthread_set_qos_class_self_np` (`qos.rs`). Apple's scheduler keeps them
   off the E-cores.
6. **`sherpa-onnx` num_threads = P-core count** — read via
   `sysctlbyname("hw.perflevel0.logicalcpu")`. On this M5 Pro that's 10 P-cores.
7. **Zero-copy audio path** — cpal f32 samples go straight to
   `OfflineStream::accept_waveform`. No temp WAV file, no `hound` round-trip,
   no `Vec<u8>` re-encode.
8. **int8 quantized weights** — 1B params, ~1 GB on disk vs ~3.7 GB fp32.

What's *not* transferable from ds4: that project's hand-written Metal kernels
(MoE, RoPE, FP8 KV cache) are DeepSeek-V4-specific. Wav2Vec2 + CTC needs a
different kernel set. A from-scratch MSL port of Wav2Vec2 is a multi-month
project; not in scope here. The list above is the realistic Apple-Silicon
overhead reduction without reimplementing the model.

## Layout

```
parakeet-rs/
├── package.json                Bun + Vite + TS frontend tooling
├── src/                        Frontend (no framework)
│   ├── index.html              Settings + model status panel
│   ├── indicator.html          Recording indicator
│   ├── main.ts                 Settings logic, listens for model-status events
│   └── styles.css
└── src-tauri/
    ├── Cargo.toml              sherpa-onnx 1.13, cpal, enigo, memmap2, libc
    ├── tauri.conf.json
    ├── entitlements.plist      Mic + Apple Events
    ├── Info.plist              NSMicrophoneUsageDescription, LSUIElement
    └── src/
        ├── main.rs             Thin bin entry
        ├── lib.rs              Tauri Builder, tray, hotkey, model setup
        ├── asr.rs              OfflineRecognizer + Parakeet TDT transducer config
        ├── audio.rs            cpal capture, mono fold, QoS-pinned thread
        ├── model_fetch.rs      First-run HF downloader with progress events
        ├── warmup.rs           Page-touch + dummy decode
        ├── qos.rs              pthread_set_qos_class_self_np wrapper
        ├── paste.rs            enigo + clipboard plugin
        └── settings.rs         JSON store, model paths
```

## Running

Requires Rust 1.77+, Bun 1.3+, Apple Silicon Mac.

```bash
bun install
bun run tauri:dev
```

First launch:

1. Settings window opens. Status reads "Downloading model (~1 GB, first run only)…"
2. The Parakeet TDT v3 int8 triplet (encoder/decoder/joiner) + tokens.txt
   download from `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` to
   `~/Library/Application Support/com.parakeet.rs/models/parakeet-tdt-0.6b-v3-int8/`.
   ~640 MB total.
3. Page-touch warmup runs (~1 s on SSD). Dummy decode bakes the CoreML graph.
4. Status flips to "Ready." Hotkey works.
5. Press `⌘⇧Space`, talk, press again. Transcript pastes at cursor.

macOS will prompt for Microphone and Accessibility permission on first use.
Both are required.

## Honest caveats

- **Apple Silicon only (arm64)**: the bundled `libsherpa-onnx-c-api.dylib`
  and `libonnxruntime.dylib` are arm64-only — Microsoft doesn't ship an
  x86_64 build of onnxruntime with CoreML EP enabled. `scripts/make-app.sh`
  runs `lipo -archs` on the release binary at the top of the script and
  refuses to bundle anything that isn't arm64-only. There is no plan to
  ship a universal binary. See [ADR-0002](docs/ADR.md#0002--macos-only).
- **CoreML EP linkage and engagement (verified)**: `sherpa-onnx-sys` is
  configured with the `shared` feature instead of the default `static`,
  because the upstream static archive ships a CPU-only `libonnxruntime.a`
  (sherpa-onnx hardcodes `-DSHERPA_ONNX_DISABLE_COREML` in its
  static-onnxruntime cmake path). The shared archive uses Microsoft's
  official `libonnxruntime.dylib`, which exports
  `_OrtSessionOptionsAppendExecutionProvider_CoreML`. `build.rs` runs
  `nm -gU` on the linked dylib at every build and warns (or errors, with
  `PARAKEET_REQUIRE_COREML=1`) if the symbol is missing. **Runtime
  measurement on M5 Pro: 2.0 s of audio decodes in 0.258 s = 7.8x real
  time** — well above the 2x CoreML floor; confirms ANE/GPU engagement.
- **First-decode latency**: the warmup pays a one-time CoreML graph-compile
  cost (you'll see ~13 `"Context leak detected, CoreAnalytics returned
  false"` lines from macOS — they're harmless lifecycle logs, not errors).
  After warmup, decodes run at the steady-state 7.8x RTFx rate.
- **Build size**: release binary is **~8.6 MB** (down from ~17 MB after
  enabling fat LTO + strip in `[profile.release]`); bundled dylibs add
  ~30 MB (mostly `libonnxruntime` at 25 MB).

## Verification

`cargo build --release` finishes in ~36 s on a cold cache, no warnings.
`bun run build` produces both HTML entry points cleanly.
End-to-end runtime test is up to you — needs a live mic and Accessibility grant.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option. This matches the Rust ecosystem convention. Unless you
explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project shall be dual-licensed as above, without
any additional terms or conditions.

The runtime-downloaded models ship under their own licenses:

- **Parakeet TDT 0.6B v3** (NVIDIA) — see
  [the model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Silero VAD** — MIT
- **Qwen 3.5 2B Instruct** (cleanup pass) — Apache-2.0
  ([model card](https://huggingface.co/Qwen/Qwen3.5-2B-Instruct))
