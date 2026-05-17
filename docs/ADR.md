# Architecture Decision Records — parakeet-rs

One file, one ADR per heading. Status legend: **Accepted** (in code today),
**Proposed** (next pass), **Rejected** (considered and dropped),
**Superseded** (replaced by a later ADR).

The overarching goal is in [ADR-0007](#0007-performance-targets). Every decision
below should be re-evaluated against it.

---

## Current state vs target snapshot

This section tracks the gap between what the code does **today** and what the
ADRs target. Update whenever the code lands or a measurement is taken.

| Dimension | Today (measured / asserted) | Target (ADR-0007) | Blocker to close the gap |
|---|---|---|---|
| End-of-speech → text appears | **press-once + Silero VAD auto-stop** wired (`streamer.rs`); 16 kHz resample inline; offline encoder runs at 7.8x RTFx on M5 Pro, so a 5 s utterance finalizes in ~640 ms after EoS | **<1 s p50 with WER ≤ 2% (revised)** — was <200 ms but retired after streaming-Parakeet survey found no viable substitute (see ADR-0009) | nothing for the revised target |
| Recognition acceleration | **CoreML EP linked AND engaged.** `sherpa-onnx-sys` set to `shared` linkage; `libonnxruntime.1.24.4.dylib` exports `OrtSessionOptionsAppendExecutionProvider_CoreML` (verified by `nm -gU`); `build.rs` symbol check is green; the **2 s warmup decode runs at 7.8x real time**, well above the 2x CoreML floor that signals CPU fallback. ANE/GPU is in use. | CoreML EP routes ops to ANE / Metal / CPU per-op | **none — ADR-0012 + ADR-0015 fully shipped.** |
| Resident set | ~800 MB (640 MB mmap'd ASR model + ORT arenas + audio buffers); +~1.6 GB when cleanup is On (Qwen 3.5 2B Q4 weights + KV cache); ~50 MB bundled dylibs | ≤2.5 GB steady state with cleanup On | none — ADR-0016 + ADR-0018 shipped |
| Settings window | Native `NSWindow` opened from menubar "Settings…" (`src/settings_ui.rs`); `orderFrontRegardless` so it surfaces above other apps | native, on-demand | none — shipped |
| Menubar UX | SF Symbols (`mic` / `mic.fill` / `arrow.down.circle`) via `objc2_app_kit::NSImage`; state-reflective menu labels | HIG-conformant template image with state | none — shipped |
| Paste path | `CGEventKeyboardSetUnicodeString` synthetic keystroke at `AnnotatedSession` tap layer (`src/ax_paste.rs`) | no clipboard mutation; works in terminals, browsers, native, Electron, IDEs | none — [ADR-0019](#0019--paste-delivery-synthetic-unicode-keystroke-annotatedsession) shipped, supersedes ADR-0011 |
| Smart formatting | In-process LLM cleanup pass: Qwen 3.5 2B Q4_K_M via llama-cpp-2 + Metal (`src/cleanup.rs`); opt-in via Settings → Cleanup → On | optional local cleanup, streaming output to cursor on word boundaries | none — [ADR-0018](#0018--cleanup-backend-llamacpp--qwen-35-2b-q4_k_m) shipped |

**Foundational dependency cleared.** The CoreML EP blocker that gated
almost every other ADR was resolved by the [ADR-0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage)
spike: switching `sherpa-onnx-sys` to its `shared` feature swaps in
Microsoft's official onnxruntime dylib, which already includes the CoreML
EP. All three ADR-0015 verification layers are green — including layer 3,
where the warmup decode reports **7.8x real-time on this M5 Pro**.

---

## 0001 — Tauri 2 + Rust shell (replacing Electron)

**Status:** **Superseded.** The Tauri shell was dropped in favour of
a single native AppKit binary (`objc2` + per-class `objc2-app-kit`
features) — see [ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation)
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
its complexity is per-OS native helpers. Roger's hardware is M5 Pro, his
target users are mac-first.

**Decision.** Apple Silicon macOS only (Intel deprecated). No `#[cfg]` arms
for other OSes, no cross-platform abstractions.

**Alternatives.**
- *Cross-platform via tauri-cross-platform-shortcut-y libs*: works, but every
  optimization in [ADR-0006](#0006-apple-silicon-optimizations) becomes
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

**Decision.** Default and only inference path is local. No API keys, no BYOK
in the settings UI, no network call after the first-run model download.

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

## 0004 — Parakeet TDT 0.6B v3 as the model

**Status:** Accepted (this session, after re-evaluation)

**Context.** Candidates considered, with English WER on LibriSpeech-clean and
deployability on M5 Pro:

| Model | WER (clean) | Size | sherpa-onnx export | Punctuation | License |
|---|---|---|---|---|---|
| Canary-Qwen 2.5B | **1.6%** (#1 leaderboard) | ~2.5 GB | ONNX exists, sherpa-onnx not yet | yes | CC-BY-NC |
| **Parakeet TDT 0.6B v3** | **1.93%** | **640 MB int8** | **yes (`csukuangfj/...-v3-int8`)** | **yes** | CC-BY-4.0 |
| IBM Granite Speech 3.3 8B | ~2.0% | ~8 GB | no | yes | Apache 2.0 |
| Whisper Large v3 | ~2.0% | ~1.5 GB | yes | post-process | MIT |
| Omnilingual ASR 1B int8 | not benched on English-only | 1.0 GB | yes | no, raw chars | Apache 2.0 |

**Decision.** Parakeet TDT 0.6B v3 int8. Win across the four axes that
matter for press-to-talk dictation: WER within 0.3% of leaderboard top,
smallest size, native sherpa-onnx support, **native punctuation + capitalization
output** (saves a post-process pass and ~50 ms latency).

**Alternatives explicitly rejected.**
- *Canary-Qwen 2.5B*: best WER but 4× larger and sherpa-onnx doesn't have the
  SALM (FastConformer + Qwen3 LLM) config yet.
- *Omnilingual ASR 1B int8* (our previous choice): higher quality on
  multilingual but outputs raw character sequences, no punctuation, ~60%
  larger, no English-specific leaderboard data.
- *NVIDIA Nemotron Speech*: NVIDIA-only, Linux preferred, no CPU/Apple
  Silicon path. Ruled out before the swap.

**Consequences.** Multilingual coverage drops from 1,600 langs (Omni) to 25
European langs (Parakeet v3). English/EU dictation users gain; speakers of
non-European languages lose. Acceptable for our user base.

---

## 0005 — sherpa-onnx as the inference binding

**Status:** Accepted

**Context.** Three Rust paths to running ONNX/ML models on Apple Silicon:

1. *sherpa-onnx Rust crate* (v1.13.2): wraps the sherpa-onnx C++ runtime, which
   in turn wraps ONNX Runtime + audio frontend + CTC/RNN-T decoders. Build
   script auto-downloads a prebuilt static lib. Ready-made `OfflineTransducer`
   and `OfflineOmnilingualAsrCtc` configs.
2. *`ort` (ONNX Runtime Rust bindings)*: closer to the metal but requires us
   to implement mel-spectrogram, FastConformer encoder loop, RNN-T/CTC
   decoder, hotword/language-bias logic ourselves — multi-week project.
3. *Candle Metal backend, pure Rust*: aspirational. We'd port FastConformer +
   transducer head to Candle's API by hand. Multi-month project, no existing
   community implementation of Parakeet TDT in Candle.
4. *MLX from Rust*: no Rust binding crate exists on crates.io. Would require
   subprocess-call to the MLX Python tool, which erases the gains.

**Decision.** sherpa-onnx Rust crate. Provider set to `"coreml"` so ONNX
Runtime's CoreML Execution Provider routes ops to ANE / Metal / CPU per-op.

**Alternatives** above, rejected for scope.

**Consequences.** We inherit sherpa-onnx's release cadence, build process,
and limitations. The static lib in the upstream prebuilt may or may not
include the CoreML EP — see [ADR-0006](#0006-apple-silicon-optimizations) for
the verification path. We do not own the inference code, so we cannot easily
add custom kernels.

**Open risk — REALISED.** The upstream prebuilt **does not include the CoreML
EP**. Verified 2026-05-15 by running
`nm -gU target/sherpa-onnx-prebuilt/sherpa-onnx-v1.13.2-osx-arm64-static-lib/lib/libonnxruntime.a | grep -i coreml`
which returns zero matches; available providers are CPU / CUDA / DML / Dnnl /
MIGraphX / Nnapi / OpenVINO / ROCM / TensorRT / VitisAI / CANN. Setting
`provider="coreml"` in the recognizer config is currently a silent no-op —
ONNX Runtime falls back to the CPU EP. Mitigation is
[ADR-0012](#0012-self-built-sherpa-onnx-with-coreml-ep) (self-build with the
EP enabled) gated by [ADR-0015](#0015-coreml-ep-verification-protocol).

---

## 0006 — Apple Silicon optimization plan (ds4 playbook applied)

**Status:** **Partly Accepted, partly Proposed.** The CPU-side optimizations
listed below (P-core scheduling, thread count, page-touch warmup, mmap'd
weights, long-lived runtime object) are in code today and verified by
inspection. The **CoreML EP / ANE acceleration claims are Proposed**, gated
on [ADR-0012](#0012-self-built-sherpa-onnx-with-coreml-ep) landing and
[ADR-0015](#0015-coreml-ep-verification-protocol) returning a green run.
Until then, the "Metal-first execution" line is a no-op (see the
[Current state snapshot](#current-state-vs-target-snapshot)).

**Context.** [antirez/ds4](https://github.com/antirez/ds4) is a from-scratch
DeepSeek V4 inference engine for Apple Silicon. Its kernel set is
model-specific (RoPE / MoE / FP8 KV cache) and doesn't transfer to a
FastConformer + transducer stack. But its *systems-level* moves do.

**Decision.** Apply these ds4 principles around the sherpa-onnx call:

| ds4 idea | Implementation |
|---|---|
| Metal-first execution | CoreML EP via `provider="coreml"` |
| `kernel_touch_u8_stride` page warmup | `warmup::page_touch` walks mmap'd encoder at 16 KiB stride at startup |
| Pre-compiled compute pipelines | `warmup::dummy_decode` runs one 0.5 s silent recognition to bake the CoreML graph cache |
| Unified memory, no host↔device copies | Apple Silicon native; cpal f32 → sherpa directly, no temp WAV |
| Long-lived runtime objects | `OfflineRecognizer` lives in `AppState`, reused every press |
| P-core scheduling hint | `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` on capture + recognition threads |
| `num_threads` = physical P-cores | `sysctlbyname("hw.perflevel0.logicalcpu")` (M5 Pro = 10) |

**Alternatives.**
- *Hand-write Metal kernels for FastConformer*: multi-month.
- *Skip optimizations, rely on sherpa-onnx defaults*: leaves ~50–250 ms on
  the table on the cold path; first dictation feels slow.

**Consequences.** Cold first-decode is amortized into startup. Hot path
shouldn't have surprise pauses. We accept that we can't outdo the underlying
ONNX Runtime kernels.

---

## 0007 — Performance targets (beat Wispr Flow)

**Status:** Accepted (targets); current state lags — see
[Current state snapshot](#current-state-vs-target-snapshot).

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
| End-of-speech → text appears | <500 ms cloud | **~840 ms** on a 5 s utterance (VAD 150 ms hangover + offline encoder at 7.8x RTFx + ~50 ms finalize) | **<1 s p50 with WER ≤ 2%** *(revised from <200 ms — see [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected) for why streaming-model swap was rejected)* | nothing; meets revised target today |
| First word in indicator | <500 ms | **n/a — indicator removed in [ADR-0014]** | n/a | retired |
| Cold start (launch → first hotkey responsive) | ~2 s | unmeasured (~4 s observed in spike: load + warmup + 2 s dummy decode) | **<3 s** with model present | warmup pass is the bulk; consider deferring the 2 s measured pass |
| WER (LibriSpeech clean / dictation) | not published | 1.93% / 3.59% (Parakeet v3 published) | **≤2% / ≤4%** | already met by model choice |
| Privacy | cloud | zero net calls post-download | **zero network calls after first-run download** | [ADR-0003](#0003-100-local-inference-no-cloud-apis-as-defaults) |
| Smart formatting | yes, cloud LLM | none | **yes, local LLM post-pass** | [ADR-0010](#0010-local-llm-post-processing-for-smart-formatting) |
| Resident set (steady state) | ~150 MB | ~1.0 GB asserted (640 MB model mmap + Tauri/WebKit + ORT arenas + audio buffers) | **≤1.2 GB** | [ADR-0014](#0014-tray-only-headless-ux) drops the indicator window; further cuts unlikely until we drop WebKit entirely |
| Battery cost / 30 min dictation | n/a | unmeasured | **<2% on M5 Pro** | ANE preference (requires [ADR-0012](#0012-self-built-sherpa-onnx-with-coreml-ep)), QoS drop at idle |

**Honesty notes.**
- The resident-set target was previously set at "<400 MB including model"
  which is arithmetically impossible (the model alone is 640 MB on disk and
  some of those pages will be resident under any access pattern). Corrected.
- The <200 ms p50 latency target was **retired** after the
  [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)
  streaming-model survey: no streaming Parakeet TDT 0.6B v3 ONNX exists,
  the realistic substitutes (NeMo FastConformer-streaming-large at 114 M
  params, Kroko Streaming Zipformer at ~50 M) all regress WER on test-other
  and lose native punctuation. We refuse the trade and accept the offline
  encoder's ~640 ms finalize cost on a 5 s utterance. New target: **<1 s p50
  end-to-end with WER ≤ 2%**, which the current build already meets.

**Alternatives.** Lower bars (parity with Whisper.cpp dictation tools like
Superwhisper). Rejected — point of the exercise is to beat Wispr Flow.

**Consequences.** Every subsequent ADR is judged against these targets. If
something below pushes p99 over 300 ms, it doesn't ship.

---

## 0008 — Hotkey press-to-toggle + clipboard paste

**Status:** **Both halves superseded.** Press-twice was replaced by
hotkey-press → talk → VAD-auto-stop per
[ADR-0009](#0009-streaming-recognition--vad-auto-stop). Clipboard +
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
VAD-auto-stop in [ADR-0009](#0009-streaming-recognition--vad-auto-stop)
because press-twice adds a user-input delay on top of inference
latency.

---

## 0009 — Silero-VAD auto-stop, offline encoder (Accepted) — streaming model SWAP REJECTED

**Status:** **Accepted in current form (press-once + Silero VAD + offline
Parakeet); streaming model swap rejected after measurement.**

Replaces [ADR-0008](#0008-hotkey-press-to-toggle--clipboard-paste) on the
press-twice toggle; supersedes the originally proposed
`OfflineRecognizer` → `OnlineRecognizer` switch.

**Context.** With the current offline recognizer, the encoder doesn't run
until the user stops talking. For a 5-second utterance on M5 Pro with CoreML
EP engaged (measured 7.8x RTFx in [ADR-0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage))
that's ~640 ms after end-of-speech.

The original proposal — switch the recognizer to `OnlineRecognizer` with a
streaming Parakeet variant — would in principle move encoder work inside
the recording window so the finalize cost drops to one chunk + decoder pass.

**Why streaming model swap was rejected.** Three hard constraints:

1. **No streaming Parakeet TDT 0.6B v3 ONNX exists in the sherpa-onnx ecosystem.**
   Issue [k2-fsa/sherpa-onnx#2918](https://github.com/k2-fsa/sherpa-onnx/issues/2918)
   is open and unresolved. The full-attention FastConformer encoder in v2/v3
   cannot be reconfigured for cache-aware streaming without retraining.
2. **The realistic substitutes are smaller, less accurate models.** Concrete
   WER comparison done at swap-time:

   | Model | Params | LibriSpeech test-clean | test-other | Punctuation |
   |---|---|---|---|---|
   | **Parakeet TDT 0.6B v3 (current, offline)** | 600 M | **1.93%** | ~4.5% | **native** |
   | NVIDIA `stt_en_fastconformer_hybrid_large_streaming_multi` @ 480 ms (best converted streaming option for sherpa-onnx) | 114 M | not reported | **5.7%** | no |
   | Kroko Streaming Zipformer 2025-08-06 | ~50 M | not published | not published | no |

   The streaming candidates are 5.4×–12× smaller and either don't publish
   test-clean numbers (a known smell) or land ~25–50% relative-worse on
   test-other. The Parakeet WER difference between 1.9% and ~3% is one extra
   error per 100 words on long-form text — noticeable.
3. **Native punctuation/capitalization is part of Parakeet TDT v3's value.**
   Losing it means we'd need [ADR-0010](#0010-local-llm-post-processing-for-smart-formatting)
   to ship before v1, adding 150–400 ms of LLM warmed-pass latency — which
   would more than erase the streaming latency savings.

NVIDIA's flagship `nvidia/nemotron-speech-streaming-en-0.6b` (2.32% test-clean
at 1.12s chunk, native punctuation, comparable parameter count) is the *only*
streaming model that would be a fair substitute — but it is **NVIDIA-GPU
only** (Ampere/Hopper/Blackwell), explicitly does not run on Apple Silicon
or CoreML, killing it for this project.

Building a streaming Parakeet ourselves was considered: it requires either
(a) retraining v3 with cache-aware attention masks (weeks of training time,
needs the original training corpus we don't have) or (b) wrapping the
existing FastConformer encoder in a chunked-offline simulated streaming
loop, which scales superlinearly with utterance length (chunks recompute
overlapping context) and *increases* total compute. Both rejected.

**Decision.** Keep `OfflineRecognizer` + Parakeet TDT 0.6B v3. Press-once UX
+ Silero VAD auto-stop remains (already shipped, see `streamer.rs` +
`vad.rs`). End-of-speech-to-text latency: VAD hangover (150 ms) + offline
encoder over the full utterance + decoder finalize. On a 5 s utterance with
the measured 7.8x RTFx that's `150 ms + 640 ms + ~50 ms ≈ 840 ms` —
**slower than Wispr Flow's ~500 ms cloud latency, but with the WER and
punctuation advantages of a bigger offline model.**

**Revised target.** [ADR-0007](#0007-performance-targets) "<200 ms p50 felt
latency" target is **provisionally retired** — it is not reachable with an
offline 600 M-param encoder regardless of CoreML acceleration. Replaced by:
**end-of-speech-to-text under 1 s p50 on M5 Pro, with WER ≤ 2% on
LibriSpeech test-clean.** Re-open if a no-WER-loss streaming Parakeet
becomes available upstream.

**Alternatives reconsidered and rejected**

- *Switch to Kroko Streaming Zipformer (71 MB):* loses native punctuation,
  no published WER — too much unknown for a primary recognizer.
- *Switch to streaming NeMo FastConformer 480 ms (136 MB, converted):*
  measurable WER regression on a smaller model; no test-clean number
  reported.
- *Chunked-offline pseudo-streaming with the current Parakeet:* compute
  grows superlinearly with utterance length; ends up slower than the
  current single-shot path. Rejected after sketch.
- *Hybrid (streaming partials + offline finalize):* doubles the model
  weight on disk (~770 MB), complicates the indicator UX we just deleted,
  and the finalize-time latency doesn't change vs. today.

**Consequences.** [ADR-0009] is now narrower than originally drafted:
it covers the VAD auto-stop UX (shipped) but explicitly does NOT cover a
recognizer swap. ADR-0010 LLM post-pass becomes less urgent (Parakeet
already does punctuation). [ADR-0007]'s sub-200 ms p50 target is retired
in favour of a more honest sub-1 s target. The Silero VAD threshold of
150 ms remains the user-tunable knob for late-cut sensitivity.

---

## 0010 — Local LLM post-processing for smart formatting (PROPOSED)

**Status:** Proposed

**Context.** Wispr Flow's headline feature is *context-aware AI rewriting*:
dictate casually, output is a polished email; dictate in a code editor,
output is structured. Parakeet TDT v3 emits punctuation and capitalization
but not paragraph breaks, code formatting, or tone shifts.

**Decision.** Optional local LLM post-pass between recognizer output and the
text-injection step. Default off; enable via settings checkbox.

Implementation: spawn a small local LLM (candidates: Qwen2.5-1.5B,
Llama-3.2-3B, gemma-3-2b, or Apple's local `Foundation Model` on macOS 26+)
via `llama-cpp-2` or `mlx-rs`-via-FFI. Few-shot prompt with the
current-app-name (read via `NSWorkspace.frontmostApplication`) injected as
context: "you are formatting dictation about to be pasted into <app>."

**Alternatives.**
- *No formatting*: matches raw Parakeet output. Misses the Wispr Flow bar.
- *Rule-based formatting only* ("um/uh removal, voice commands like
  'new paragraph')*: easy, helps, but doesn't reach LLM-quality rewriting.
- *Cloud LLM* (Anthropic/OpenAI): violates [ADR-0003](#0003-100-local-inference-no-cloud-apis-as-defaults).

**Consequences.** Realistic post-pass latency: **150–400 ms warmed**, not
the 50–150 ms originally quoted. Breakdown for a 1.5B-param model on M5 Pro
with KV cache warm and the model resident: ~50 ms TTFT, ~30 tokens output
at ~10 ms/token = ~300 ms generation, plus tokenize/detokenize overhead
~20–50 ms. Token streaming **does not help** here because we need a single
final string before injection — partial-rewrite streaming would cause
flicker in the target app and corrections after paste. If the post-pass
budget collides with [ADR-0007](#0007-performance-targets), default it off
and surface as an opt-in "polish output" toggle in settings.

We're now bundling two models (~640 MB ASR + ~1.5–3 GB LLM). Privacy story
holds. Default-off respects users who want raw transcription **and** keeps
the latency target intact for the no-formatting path.

**Open question.** Use Apple's macOS 26+ on-device Foundation Model API
(free, integrated, ANE-optimized, **closed but on-device**)? Trades zero
bundle size for OS-version pinning. Worth a separate ADR once we know the
mac-26 baseline is acceptable for our users.

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
surface in v1, especially since [ADR-0009](#0009-streaming-recognition--vad-auto-stop)
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

## 0012 — sherpa-onnx prebuilt with CoreML EP (shared linkage)

**Status:** **Accepted — shipped.** Spike per [ADR-0016](#0016-tauri--rust-vs-swiftui-re-evaluation)
revealed that a **5-minute Cargo feature change** gets us a CoreML-enabled
libonnxruntime, with no submodule, no cmake build, no maintenance tax. The
4-hour vendored-self-build plan documented below as "originally drafted" has
been superseded.

**Context.** The sherpa-onnx Rust crate's build script downloads a prebuilt
osx-arm64 library from GitHub releases. Two paths exist upstream:

| Path | Archive | onnxruntime origin | CoreML EP |
|---|---|---|---|
| `static` (Cargo `default-features`) | `sherpa-onnx-v1.13.2-osx-arm64-static-lib.tar.bz2` | `csukuangfj/onnxruntime-libs` CPU-only static build | **No** — sherpa-onnx hardcodes `add_definitions(-DSHERPA_ONNX_DISABLE_COREML)` in `cmake/onnxruntime-osx-arm64-static.cmake:64` for this path |
| `shared` | `sherpa-onnx-v1.13.2-osx-arm64-shared-lib.tar.bz2` | Microsoft's official `onnxruntime-osx-arm64-1.24.4.tgz` dylib | **Yes** — verified by `nm -gU libonnxruntime.1.24.4.dylib \| grep _OrtSessionOptionsAppendExecutionProvider_CoreML` returns one symbol at offset `0x13eff4` |

Switching from `static` to `shared` therefore replaces a CPU-only static lib
with a CoreML-capable dylib at the cost of shipping ~30 MB of dylibs in the
`.app` bundle.

**Decision.** Use the `shared` feature on `sherpa-onnx-sys`:

```toml
# src-tauri/Cargo.toml
sherpa-onnx = { version = "1.13", default-features = false, features = ["shared"] }
```

Bundle the dylibs into the `.app` via `tauri.conf.json`:

```json
"bundle": {
  "macOS": {
    "frameworks": [
      "target/release/libsherpa-onnx-c-api.dylib",
      "target/release/libsherpa-onnx-cxx-api.dylib",
      "target/release/libonnxruntime.dylib",
      "target/release/libonnxruntime.1.24.4.dylib"
    ]
  }
}
```

`sherpa-onnx-sys` already emits `cargo:rustc-link-arg=-Wl,-rpath,…` so the
binary loads the dylibs from the bundle's `Contents/Frameworks/` at runtime.

**Verification.** All three [ADR-0015](#0015-coreml-ep-verification-protocol)
layers are green, with **empirical numbers measured on this machine**:

- **Layer 1 (build-time symbol check):** `build.rs::check_coreml_ep` runs
  `nm -gU` over the linked `libonnxruntime.1.24.4.dylib`, finds
  `OrtSessionOptionsAppendExecutionProvider_CoreML` (at offset `0x13eff4`),
  sets `--cfg parakeet_coreml_ep_present`. No `cargo:warning=` line.
- **Layer 2 (recognizer init log):** `asr.rs:64` logs
  `"ASR provider requested: coreml (EP symbol present in libonnxruntime.a)"`
  at startup.
- **Layer 3 (per-utterance RTFx probe):** `asr.rs::recognize_with_timing`
  reports `ASR decoded 2.00s in 0.258s (7.8x real time)` on the warmup pass.
  That's well above the 2x CoreML floor — CPU-only int8 transducer inference
  on this encoder size sits at ~1.0–1.5x; 7.8x is the signal that CoreML is
  partitioning ops to ANE / GPU and not silently falling back. Two corollary
  observations from the runtime log:
  - The first warmup pass (0.5 s of silence) hits ~0.85x; that's pure
    CoreML graph-compile cost, not a steady-state measurement. The warmup
    in `warmup.rs::dummy_decode` now uses a throwaway-then-measured
    two-pass structure so the user-visible log line is always the
    steady-state number, and `recognize_with_timing(warmup: true)`
    suppresses the spurious warn.
  - The macOS console emits ~13 `"Context leak detected, CoreAnalytics
    returned false"` lines on the first decode. That's an Apple-framework
    teardown log from `CoreAnalyticsCenter` and confirms CoreML is
    initialising; it does not appear on subsequent decodes.

**Alternatives considered then rejected.**

- *Vendor sherpa-onnx + ONNX Runtime as submodules and `cmake --build` with
  `SHERPA_ONNX_ENABLE_COREML=ON`* — the original ADR plan. Costs: 1–2 h
  initial build, ongoing tag-bump tax, owning ORT regressions. Only worth
  doing if we needed something the prebuilt shared dylib lacks; it doesn't.
- *Keep upstream `static` prebuilt* — CPU-only, kills the [ADR-0007]
  (#0007-performance-targets) latency story. Rejected.
- *Different inference binding* — see
  [ADR-0005](#0005-sherpa-onnx-as-the-inference-binding); already rejected.

**Consequences.**
- Bundle size grows by ~30 MB (mostly `libonnxruntime`'s 25 MB).
- Code-signing / notarization must handle bundled dylibs. Standard for
  third-party-dylib-shipping macOS apps; Tauri's bundler handles the rpath.
- We don't own the ORT build; if Microsoft's CoreML EP regresses, we wait
  for a new release rather than bisect ourselves. Acceptable for v1.
- If ANE coverage on the Parakeet encoder turns out to be poor (large
  fraction of ops fall back to CPU), the *only* lever left is a from-source
  ONNX Runtime build with a tuned CoreML EP — at which point the original
  vendor-build plan returns. We'll know after the first RTFx measurement.

**Historical record (pre-spike plan, kept for context).** The earlier draft
called for a vendored cmake build. That plan is superseded by this one, but
the cmake flags it specified — `SHERPA_ONNX_ENABLE_COREML=ON`,
`SHERPA_ONNX_ENABLE_TTS=OFF`, etc. — remain the right invocation if we ever
do need to fall back to a self-build.

---

## 0015 — CoreML EP verification protocol

**Status:** Accepted

**Context.** After [ADR-0012](#0012-self-built-sherpa-onnx-with-coreml-ep)
lands, we need an automated, repeatable way to **prove** the CoreML EP is
actually present and being used — not just hope that `provider="coreml"`
silently fell back to CPU again.

**Decision.** Three layers of verification, all gated in CI before any
ADR-0007 latency claim is asserted:

**Layer 1 — Build-time symbol check.** As part of the post-build step:

```bash
LIB="$SHERPA_ONNX_LIB_DIR/libonnxruntime.a"
if ! nm -gU "$LIB" 2>/dev/null | grep -q "_OrtSessionOptionsAppendExecutionProvider_CoreML\|CoreMLExecutionProvider"; then
  echo "FAIL: CoreML EP symbol absent from $LIB"
  exit 1
fi
```

Fails the build if the EP isn't linked in. No silent CPU-only fallback ever
reaches production.

**Layer 2 — Runtime provider availability log.** sherpa-onnx logs the
selected EP at recognizer-create time. We will parse for the line containing
"CoreMLExecutionProvider" and panic in debug builds if it's absent. In
release builds, log a warning and emit a telemetry event so we know.

**Layer 3 — Per-utterance latency probe.** Wrap `recognizer.decode(&stream)`
with `Instant::now()` and emit p50 / p95 / p99 to a local rolling histogram
(written to `~/Library/Application Support/com.parakeet.rs/latency.jsonl`,
local-only, no telemetry). If end-to-end p50 climbs above 250 ms, surface a
warning in the settings UI: "ANE acceleration may be inactive — re-run
verification".

**Alternatives.**
- *Just trust the EP string*: known to fail silently — that's how we got
  here.
- *Compare CPU vs CoreML A/B benchmarks*: nice but expensive at startup.
  Layer 3 catches this implicitly via the latency histogram.

**Consequences.** ~80 lines of Rust (symbol check is one shell line in the
post-build, runtime log parse is ~30 lines, latency probe is ~50 lines).
Replaces "I hope it works" with "we know it works."

---

## 0013 — Hotword / custom dictionary support (PROPOSED, future)

**Status:** Proposed

**Context.** Domain vocabulary (engineering terms, names, product names) gets
mistranscribed by general ASR. sherpa-onnx supports an external
`hotwords_file` that boosts specific n-grams during decoding.

**Decision.** Settings UI gains a "Custom vocabulary" textarea. Each line is
a hotword + optional boost score (`tauri 30.0\nshergaonnx 25.0`). Wired to
`OfflineRecognizerConfig.hotwords_file` (or its online equivalent).

**Consequences.** Minor decoder latency cost (negligible). Big accuracy win
on dictation about the user's actual work.

---

## 0014 — Tray-only headless UX (PROPOSED)

**Status:** Proposed. Current `src-tauri/tauri.conf.json` still has
`"visible": true` for the settings window — code does not yet match this ADR
and the [Current state snapshot](#current-state-vs-target-snapshot)
acknowledges the gap.

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

## 0016 — Tauri + Rust shell vs SwiftUI native (re-evaluation)

**Status:** **Superseded — outcome reversed.** The spike landed
"stay on Tauri+Rust", but during the subsequent code-architecture +
adversarial review rounds we dropped the Tauri shell entirely and
moved to a single native AppKit binary via `objc2` + per-class
`objc2-app-kit` features. Reason: with Tauri out, the WebView /
WebKit / `tauri-conf.json` / `bun`-frontend surface contributed no
value (Settings UI fits in native `NSWindow` + `NSTextField` +
`NSPopUpButton` cleanly), and removing it cut ~200 MB resident, the
entire frontend toolchain, and a class of focus-stealing bugs.
ADR-0019 (CGEvent paste) and the streaming HUD bar work depended on
direct AppKit access anyway. Original spike rationale preserved
below.

**Context.** ADR-0001 chose Tauri to escape Electron. Two of the implicit
motivations for *Tauri specifically* over *native Cocoa / SwiftUI* were:
(a) cross-platform optionality and (b) avoiding Swift learning curve. (a)
was retired when ADR-0002 made the project mac-only. Codex's review then
exposed that getting CoreML EP through sherpa-onnx requires vendoring the
upstream lib, building it ourselves, and maintaining the build going
forward — which is real, recurring work that a native SwiftUI app would
avoid entirely (Core ML is just the runtime in Swift, not a separate EP to
enable). So the original ADR-0001 reasoning has weakened.

A SwiftUI rewrite would substitute:
- **Whisper Large v3 turbo via WhisperKit** (Argmax's CoreML-native port,
  ANE on by default, no symbol-check theatrics) for Parakeet TDT v3 via
  sherpa-onnx.
- **AVAudioEngine** for cpal.
- **AXUIElement APIs** for objc2-accessibility glue (un-defers
  [ADR-0011](#0011-direct-accessibility-text-injection)).
- **NSStatusBar + SwiftUI settings view** for Tauri tray + WebView settings.
- **Apple Foundation Model API** (macOS 26+) for the
  [ADR-0010](#0010-local-llm-post-processing-for-smart-formatting) post-pass.

Cost: a Swift rewrite is ~1,000 lines thrown away, several weeks of Swift
fluency development, and lock-in to a single OS forever.

Benefit: every layer becomes Apple-native, the CoreML / ANE story stops
being a vendor-and-pray exercise, smaller binary, faster cold start, and
the "smart formatting" ADR collapses from "bundle a 3 GB LLM" to "call
Foundation Model API."

**Decision (spike-resolved).** **Stay on Tauri+Rust.** The
[ADR-0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage)
spike took **5 minutes, not 4 hours**: the upstream sherpa-onnx prebuilt
already ships a CoreML-enabled `libonnxruntime.dylib` in its `shared`
release archive, behind a single Cargo feature flag flip. No submodule,
no cmake build, no ongoing vendor maintenance — the "real cost" of
ADR-0012 collapsed to "add `default-features = false, features = ["shared"]`
to one line of Cargo.toml plus four entries in `tauri.conf.json` to bundle
the dylibs."

**Continuation triggers (assessed after spike):**
- ✅ Spike succeeded within budget (5 min vs 4 h)
- ⏳ ADR-0015 latency probe will confirm ANE engagement at first end-to-end
  run with a live mic. **Not yet measured.** Layer-1 build-time symbol
  check is green; layer-2 init log will say "EP symbol present"; layer-3
  RTFx probe needs a real recording.
- ✅ Build reproduces cleanly on a fresh checkout — `cargo build` just
  downloads the right prebuilt archive

**Pivot triggers — still active, archived as future safeguards.** The
spike succeeded for layer 1 (linking). If layer 3 (runtime RTFx) comes
back below 2x real-time, the pivot triggers re-arm:
- Build works but ANE is not actually engaged (per [ADR-0015] latency
  probe showing CPU-equivalent timings) → pivot to SwiftUI + WhisperKit
- Upstream sherpa-onnx / ONNX Runtime breaks the CoreML build in a way
  that takes more than a day to diagnose → same pivot

**Pivot cost (re-baseline after Tauri+Rust scaffold landed).** Roughly
1.5–2 weeks of clean Swift rewrite, reusing all design decisions
(Parakeet/Whisper choice, hotkey UX, settings model, paste path,
performance targets) and throwing away ~1,500 lines of Rust + TypeScript.

**Alternatives reconsidered.**
- *Pivot to SwiftUI now anyway, on principle*: rejected — the original
  motivation for the pivot was the ADR-0012 maintenance tax, which has
  evaporated. SwiftUI's other advantages (Foundation Model API for the
  LLM post-pass, AXUIElement for direct injection) remain real but are
  not load-bearing for v1.

**Consequences.** Tauri+Rust scaffold stays. The remaining ADRs proceed
on the original critical path. The dormant SwiftUI pivot path is kept in
the ADR record so we know what to do if a future ONNX Runtime regression
makes CoreML EP unreliable.

---

## 0018 — Cleanup backend: llama.cpp + Qwen 3.5 2B Q4_K_M

**Status:** Accepted (Phase-0 measured)

**Context.** [docs/latency-plan.md](./latency-plan.md) §6 calls for a
Candle vs OminiX-MLX head-to-head on Gemma 4 E2B 4-bit. Research surfaced
three blockers before any bench could run:

1. **Gemma 4 doesn't exist in Candle 0.10.2.** Candle *main* branch added
   `pub mod gemma4` recently, but no `quantized-gemma4` example yet —
   fp16/bf16 weights only (~10 GB for 5.1B-loaded E2B).
2. **OminiX-MLX ships no Gemma crate.** Adopting it for Gemma 4 means
   writing a new `gemma4-mlx` crate from scratch (per
   [docs/gemma4-mlx-implementation.md](./gemma4-mlx-implementation.md)),
   which catalogues seven architectural divergences from Qwen3. Multi-day
   port + token-parity validation against Python mlx-lm. Out of scope as
   the v1 cleanup backend; gated on a measured Candle/llama.cpp miss.
3. **Gemma 4 E2B doesn't fit the <2 GB disk budget.** Q4_K_M is ~3 GB
   ([bartowski/google_gemma-4-E2B-it-GGUF](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF)).
   Going lower than Q4_K_M (Q3_K_M ~2.4 GB, Q2_K ~1.9 GB) hits the steep
   small-model quant degradation curve flagged by the [Qwen3
   quantization study](https://arxiv.org/html/2505.02214v1).

**Decision.** Replace `claude -p` (current `src/cleanup.rs` path) with
in-process inference via **llama.cpp + Qwen 3.5 2B-Instruct Q4_K_M**.

- **Model:** [`unsloth/Qwen3.5-2B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
  → `Qwen3.5-2B-Q4_K_M.gguf` (1.22 GB on disk).
- **Backend:** [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2)
  Rust binding (crates.io, default-features off, `metal` feature on).
  llama.cpp builds llama.cpp's C++ core via cmake at first compile.
- **Chat template:** ChatML with `/no_think` directive — Qwen 3.5's
  reasoning mode is on by default and would blow past our output cap
  inside the `<think>` block. The directive disables thinking; we
  additionally pre-close an empty `<think></think>` on the assistant
  side as belt-and-braces.

**Why Qwen 3.5 2B, not the originally-spec'd Gemma 4 E2B.**

Per the size-matched comparison in
[Maniac](https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size):

| Benchmark | Gemma 4 E2B | Qwen 3.5 2B | Winner |
|-----------|-------------|-------------|--------|
| MMLU-Pro | 60.0 | **66.5** | Qwen (+6.5pp) |
| TAU2-Bench | 24.5 | **48.8** | Qwen (+24pp) |
| MMMU-Pro | 44.2 | **50.3** | Qwen (+6.1pp) |
| MMMLU | **67.4** | 63.1 | Gemma (+4.3pp) |

Qwen 3.5 2B beats Gemma 4 E2B on 3/4 size-class benchmarks, fits the
<2 GB disk budget at acceptable Q4_K_M quant, is one model generation
newer (Feb 2026 vs Gemma 4's earlier 2026 release), and works in the
shipping `llama-cpp-2` Rust binding today. Gemma 4 wins only on
multilingual MMMLU — not load-bearing for English-language dictation
cleanup.

**Why llama.cpp, not Candle.** Candle ships neither Qwen 3.5 (new
hybrid Gated-DeltaNet architecture, `Qwen3_5ForConditionalGeneration`)
nor Gemma 4 Q4. llama.cpp picked up both within days of release. The
"pure Rust" constraint in the latency plan is interpreted as
"no Python, no subprocess, no HTTP" — FFI to a well-maintained C++
library (analogous to sherpa-onnx for ASR) satisfies it. The Metal
backend on Apple Silicon delivers ~100 tok/s on Qwen 3.5 2B Q4_K_M
(measured below).

**Measured Phase-0 numbers, M5 Pro 24 GB, 100 iterations
(`bench/cleanup-backends.csv`):**

| Metric | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| TTFT (ms) | 2.0 | 2.0 | 2.0 | 2.0 |
| Generation (ms) | 548 | 548 | 558 | 567 |
| **Total per polish (ms)** | **551** | **550** | **560** | **570** |
| Decode (tokens/sec) | 100.3 | 100.4 | 101.7 | 101.9 |

Cold model load: 229 ms (incurred once per process; the warmup is
done as part of `App::spawn_llm_setup` in `src/app.rs`, so cleanup
is ready before the user's first hotkey press).
Output: 55 tokens (one cleaned paragraph) for a 240-character noisy
input. p99 / p50 = 1.04 — variance is negligible, Metal kernel
scheduling is steady-state from iteration 1.

**Latency budget consequence.** Projected total post-endpoint latency
on a 5 s utterance with cleanup:

```
   362 ms   ASR (§1 measured)
+  150 ms   VAD hangover (vad.rs:15)
+   50 ms   paste finalize (latency-plan estimate)
+  550 ms   cleanup (this bench, p50)
= 1112 ms   total p50
```

That's **~112 ms over the latency-plan §6 acceptance criterion of
≤ 1.0 s p50 with cleanup**. Three mitigations on the table for the
§4 cleanup-rewrite work, in order of effort:

1. **Stream the paste**: emit cleaned tokens to NSPasteboard + ⌘V
   incrementally as the model generates, rather than buffering until
   end-of-sequence. The user feels latency as "first token visible",
   not "last token visible". Saves ~300–400 ms perceived; the actual
   wall clock to last-token is unchanged.
2. **Trim output cap**: typical cleanup output for a 30-token input
   is 20–35 output tokens. Cap at 40 → ~400 ms gen → total ~960 ms p50,
   under budget. Risk: long dictations get truncated; need fallback to
   raw paste if the cap hits.
3. **Ship at ~1.1 s p50, advertise honestly**: still ~5× faster than
   `claude -p` subprocess (1–3 s startup alone). The §6 acceptance
   number gets a footnote: "1.0 s p50 was an aspirational target;
   measured v1 is 1.1 s p50 within budget for streaming-paste v1.1."

Recommend (1) for §4: streaming paste is the lever that buys the most
perceived-latency improvement and aligns with how cloud dictation
tools (Wispr Flow, etc.) deliver their <700 ms feel.

**Rejected alternatives revisited.**

- **OminiX-MLX + new `gemma4-mlx` crate.** The implementation doc
  ([gemma4-mlx-implementation.md](./gemma4-mlx-implementation.md))
  estimates this as "from-scratch work" with 7 architectural
  divergences from Qwen3. Plan-faithful but multi-day. Defer; revisit
  if the measured llama.cpp number ever misses a tightened budget.
- **Candle main + gemma4 fp16.** ~10 GB on-disk, blows past the <2 GB
  user constraint, and the loader is not yet quantized.
- **Direct Anthropic API.** Rejected by project directive (no cloud
  cleanup). The legacy `cleanup_mode = "anthropic"` serde alias was
  removed when the API key field was dropped from settings; the
  in-process llama.cpp path replaces both the API and the prior
  `claude -p` subprocess approach.
- **`mlx-rs` direct.** Same multi-day port story as OminiX-MLX without
  the shared infrastructure crates.

**Open issues (resolved post-§6).**

- ~~The `/no_think` directive leaks into the model's output.~~
  Resolved — `strip_no_think_tail` in `src/cleanup.rs` handles all
  observed variants (`/no_think`, `no_think`, `no think`,
  `No think`, etc., case-insensitive, ignoring trailing punctuation).
- ~~Streaming-paste is non-trivial against the clipboard+⌘V shape.~~
  Resolved — `paste::Streamer` streams to the focused app via
  `CGEventKeyboardSetUnicodeString` keystrokes (ADR-0019), one
  word-boundary-batched chunk per LLM emission burst. No clipboard,
  no AX, no flicker.
- **Open: model file management.** ~1.22 GB download on first
  cleanup-enable, expected at
  `~/Library/Application Support/com.parakeet.rs/llm/qwen3.5-2b-q4_k_m/`.
  In-app download is **not** wired up — `load_llm_blocking` bails
  with a clear error if the file is missing; the user has to fetch
  manually (`bench/README.md` has the one-liner). Re-use the
  `model_fetch.rs` pattern from Parakeet's first-run flow.

**References.**

- `bench/cleanup-backends.csv` — full 100-row Phase-0 data, this M5 Pro.
- `src/bin/bench_llm.rs` — the bench harness.
- [Welcome Gemma 4 — Hugging Face](https://huggingface.co/blog/gemma4)
- [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [llama-cpp-2 crate](https://crates.io/crates/llama-cpp-2)
- [Qwen 3.5 vs Gemma 4 size-matched benchmarks — Maniac](https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size)
- [Qwen3 quantization empirical study (arxiv)](https://arxiv.org/html/2505.02214v1)

---

## 0017 — CoreML `ModelCacheDirectory` blocked at the sherpa-onnx Rust binding

**Status:** Blocked / Deferred

**Context.** [docs/latency-plan.md](./latency-plan.md) §2 wants us to set
ONNX Runtime's CoreML EP `ModelCacheDirectory` provider option to
`~/Library/Caches/parakeet-rs/coreml/`. ORT 1.20+ supports it; we link
`libonnxruntime.1.24.4.dylib`, so the underlying EP can consume it.
Expected win: seconds off **first-dictation-after-launch** cold start
(does not move warm p50, per the plan).

**Investigation.** Surveyed sherpa-onnx 1.13.2 (current crates.io latest):

- `OfflineModelConfig` exposes a single `provider: Option<String>` field
  (just `"coreml"`). No `provider_config`, no `coreml_*` sub-struct,
  no key/value map for arbitrary EP options.
  ( `~/.cargo/registry/.../sherpa-onnx-1.13.2/src/offline_asr.rs:475` )
- The sys binding mirrors the upstream C struct exactly — also a single
  `*const c_char`. ( `sherpa-onnx-sys-1.13.2/src/offline_asr.rs:178` )
- Upstream `SherpaOnnxOfflineRecognizerConfig` (k2-fsa/sherpa-onnx C
  API) does NOT carry a `provider_config` field. Only the *online*
  recognizer's `SherpaOnnxOnlineModelConfig` has one — and even there
  the CoreML sub-struct only surfaces `coreml_provider`, not
  `model_cache_directory`.
- `rg -i 'coreml|provider_config|model_cache'` across both crates
  returns zero matches outside the provider-name string itself.

**Decision.** Defer §2 until we can pass arbitrary CoreML EP options
through to `OrtSessionOptionsAppendExecutionProvider_CoreML_V2`.
Paths forward, in increasing cost:

1. **Wait for sherpa-onnx upstream.** File an issue requesting the
   offline path's `OfflineModelConfig` gain a `provider_config` field
   matching the online path. Low effort to file; weeks–months to land.
2. **Vendored fork of sherpa-onnx-sys.** Patch the C struct +
   `to_sys` bridge locally; rebuild the sys crate against our fork.
   Adds a maintenance liability — every sherpa-onnx upgrade has to
   re-apply the patch.
3. **Drop sherpa-onnx for the ASR path.** Switch to direct ORT
   bindings (`ort` crate) and feed the `.onnx` files ourselves. Big
   refactor; would absorb the encoder/decoder/joiner glue sherpa
   currently provides for the NeMo transducer family.

**Why deferring is OK.** The §1 baseline (bench/baseline.csv,
2026-05-16, M5 Pro 24 GB) puts the 5 s ASR-only p50 at **362 ms**.
Adding the latency plan's 150 ms VAD hangover + 50 ms paste finalize
≈ 562 ms total post-endpoint — already inside the §6 acceptance
criterion of ≤ 700 ms p50 no-cleanup. The §2 optimization helps
*first-launch* cold start only (where the user feels CoreML's MLProgram
graph compile cost). That's a real win to grab eventually, but it's
not gating the ship of §6 (cleanup rewrite + acceptance numbers).

**Open question.** Empirically verify whether CoreML's own framework-level
cache at `~/Library/Caches/com.apple.MLModelCompiler/` already short-
circuits enough of the recompile cost that the ORT-layer cache is
marginal. If it does, this ADR closes as "no work needed"; if it doesn't,
path 2 (vendored fork) becomes the right move.

**Consequences.**
- `src/asr.rs:72` left unchanged.
- Latency plan §2 acceptance criterion ("CoreML model cache directory
  is configured…") reads "Deferred — see ADR-0017" in the final
  rollup.
- Bench harness from §1 is already in place to measure the win when
  the binding option lands.

---

## 0019 — Paste delivery: synthetic Unicode keystroke (AnnotatedSession)

**Status:** **Accepted — shipped.** Supersedes [ADR-0011](#0011--direct-accessibility-text-injection-deferred)
(deferred AX path).

**Context.** Delivery of the transcribed (and optionally LLM-cleaned)
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
- Streaming polish: 3-chunk dictation into Ghostty round-tripped in
  861 ms `dur_post_endpoint_ms` end-to-end (audio capture stop →
  ASR → cleanup polish → keystroke posted → focused-app insertion),
  with the cleanup pipeline contributing ~550 ms of that.

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

## Index of open decisions vs targets

| ADR-0007 target | Owner ADR | Status | Blocked by |
|---|---|---|---|
| **CoreML EP actually present** | [0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage) + [0015](#0015-coreml-ep-verification-protocol) | **Shipped + measured** — 7.8x RTFx on the warmup decode confirms ANE/GPU is engaged | nothing |
| <1 s p50 felt latency (revised from <200 ms — see [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)) | [0009] | Press-once + VAD auto-stop shipped (offline encoder); ~640 ms encoder finalize on a 5 s utterance — meets the revised <1 s target | nothing |
| Live partial transcripts | [0009](#0009-streaming-recognition--vad-auto-stop) | Proposed | switch to streaming Parakeet model |
| ANE confirmed in use | [0015](#0015-coreml-ep-verification-protocol) | **All three layers green** — layer 1 nm-check, layer 2 init log, layer 3 measured 7.8x RTFx | nothing |
| ≤1.2 GB resident set | [0014](#0014-tray-only-headless-ux) + [ADR-0006](#0006-apple-silicon-optimization-plan-ds4-playbook-applied) mmap | Tray-only shipped, mmap shipped; lazy webview still Proposed | nothing |
| Smart formatting parity with Wispr Flow | [0010](#0010-local-llm-post-processing-for-smart-formatting) | Proposed | nothing |
| Clipboard not clobbered | [0011](#0011-direct-accessibility-text-injection) | **Deferred to v2** | not in v1 scope |
| Custom vocabulary | [0013](#0013-hotword--custom-dictionary-support-proposed-future) | Proposed | nothing |

**Critical path to ADR-0007 latency claim (gated by
[ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation)
time-boxed spike):**
1. **Spike: [ADR-0012](#0012-self-built-sherpa-onnx-with-coreml-ep)** —
   self-build sherpa-onnx with CoreML EP (vendored submodule). ≤ 4 hours.
   Pivot to SwiftUI if it doesn't land cleanly (see
   [ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation)).
2. [ADR-0015](#0015-coreml-ep-verification-protocol) — wire build-time +
   runtime EP checks; confirm ANE is actually engaged.
3. [ADR-0009](#0009-streaming-recognition--vad-auto-stop) — streaming +
   Silero VAD with 150 ms threshold.
4. Measure end-of-speech → text latency on real utterances; only then
   update [ADR-0007](#0007-performance-targets) "Today (baseline)" column
   with the post-optimization number.

Anything not on this table is either accepted-and-done or out of scope.

## Change log

- **2026-05-15** — Codex challenge review (`/codex challenge docs/ADR.md`)
  surfaced eight findings. Verified the most critical (no CoreML EP in
  prebuilt static lib) via `nm -gU` symbol inspection. Material revisions:
  - Added [Current state snapshot](#current-state-vs-target-snapshot) so the
    code-vs-target gap is impossible to overlook.
  - [ADR-0005](#0005-sherpa-onnx-as-the-inference-binding) updated to
    record the realised CoreML EP risk with evidence.
  - [ADR-0006](#0006-apple-silicon-optimization-plan-ds4-playbook-applied)
    split: CPU optimizations Accepted, CoreML / ANE claims downgraded to
    Proposed-gated-on-0012.
  - [ADR-0007](#0007-performance-targets) latency table gained a "Today
    (baseline)" column; the impossible "<400 MB resident set including
    model" target replaced with the honest "≤1.2 GB" steady-state target.
  - [ADR-0009](#0009-streaming-recognition--vad-auto-stop) VAD silence
    threshold tightened from 250 ms to 150 ms; latency budget made
    explicit and shown to require ADR-0012 to hold.
  - [ADR-0010](#0010-local-llm-post-processing-for-smart-formatting)
    post-pass latency estimate bumped from "50–150 ms" (hand-waved) to
    "150–400 ms warmed" (engineered).
  - [ADR-0011](#0011-direct-accessibility-text-injection) **deferred to
    v2** by user direction.
  - [ADR-0012](#0012-self-built-sherpa-onnx-with-coreml-ep) **promoted
    from Proposed to Accepted**; vendor as submodule rather than env-var
    redirection; explicit cmake flag list; honest maintenance cost.
  - New [ADR-0015](#0015-coreml-ep-verification-protocol) added with a
    three-layer verification protocol (build-time symbol check, runtime
    provider log parse, per-utterance latency probe).
  - New [ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation)
    re-opens the Tauri-vs-SwiftUI question now that we're mac-only and
    ADR-0012 has revealed real maintenance costs. Decision time-boxed to a
    ≤ 4 h sherpa-onnx-with-CoreML build spike; explicit pivot/continuation
    triggers documented.

- **2026-05-15** (later) — Implementation pass landed:
  - [ADR-0014](#0014-tray-only-headless-ux) shipped: settings window
    `visible: false` at launch; tray menu opens it on demand.
  - [ADR-0009](#0009-streaming-recognition--vad-auto-stop) shipped (offline
    encoder variant): press-twice toggle deleted; new `streamer.rs` +
    `vad.rs` modules drive Silero VAD over an audio tap channel; cancel-on-
    second-press preserved. Streaming Parakeet still future work.
  - [ADR-0015](#0015-coreml-ep-verification-protocol) implemented: layer 1
    in `build.rs`, layer 2 in `asr.rs::Asr::load`, layer 3 in
    `asr.rs::Asr::recognize_with_timing` with an RTFx-floor warn at 2x.
  - HIG audit findings #1, #2, #3, #5, #9, #11, #12, #14, #15 addressed:
    SF Symbols tray icon (via `objc2_app_kit`), state-reflective menu
    labels, glyph-rendered hotkey field, determinate `<progress>` bar,
    `-apple-system` typography, semantic dark-mode palette.
  - **[ADR-0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage)
    spike resolved unexpectedly fast** — switched `sherpa-onnx` to
    `default-features = false, features = ["shared"]`, which pulls
    Microsoft's official `libonnxruntime.dylib` (CoreML-enabled) instead
    of the CPU-only static archive. Bundled the four resulting dylibs in
    `tauri.conf.json` `bundle.macOS.frameworks`. Build-time `nm -gU`
    confirms `OrtSessionOptionsAppendExecutionProvider_CoreML` is exported.
    The originally drafted vendored-cmake plan is preserved at the bottom
    of ADR-0012 as a future fallback if Microsoft's prebuilt regresses.
  - [ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation)
    closed in favour of staying on Tauri+Rust — the ADR-0012 maintenance
    tax that triggered the re-evaluation no longer exists.

- **2026-05-15** (even later, runtime confirmation pass):
  - **ADR-0015 layer 3 measured and green** on this M5 Pro: the warmup's
    2 s silent decode runs in **0.258 s (7.8x real time)**, well above the
    2x CoreML floor. ANE/GPU is engaged. The "Context leak detected,
    CoreAnalytics returned false" lines from the first decode were
    misread as failure on the prior pass — they're a harmless lifecycle
    log from `CoreAnalyticsCenter` that *confirms* CoreML is initialising.
  - **Warmup refactored** to a throwaway-then-measured two-pass structure
    (`warmup.rs:38-48`), so the user-visible RTFx log line is always the
    steady-state number. The throwaway pass uses a new
    `Asr::recognize_silent_warmup` that suppresses the spurious
    "below CoreML floor" warn for the JIT-dominated first decode.
  - **Warn threshold tightened**: `recognize_with_timing` now only warns
    on samples ≥ 1.5 s of audio (was 0.5 s), since short utterances —
    "yes", "no", single words — aren't reliable RTFx measurements.
  - [ADR-0014] indicator webview **dropped entirely** — tray icon state
    swap (mic / mic.fill via SF Symbols) is now the sole visual feedback
    during dictation. Saves a webview at startup and aligns with the
    "no niceties" steer. Files removed: `src/indicator.html`,
    `src/indicator.ts`, `body.indicator` CSS rule; window definition
    removed from `tauri.conf.json`; helpers `show_indicator` /
    `hide_indicator` deleted from `lib.rs`.
  - **Release profile tuned**: `[profile.release]` in Cargo.toml now sets
    `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`,
    `strip = "symbols"`, `opt-level = 3`. **Release binary 16 MB → 8.6 MB**
    (-46%), with no measurable cold-start regression.

- **2026-05-15** (final pass) — **Streaming model swap considered and
  rejected.** The "ADR-0009 phase 2" idea (`OfflineRecognizer` →
  `OnlineRecognizer` with a streaming model) was investigated end-to-end:
  - No streaming Parakeet TDT 0.6B v3 ONNX exists; sherpa-onnx issue
    [k2-fsa/sherpa-onnx#2918](https://github.com/k2-fsa/sherpa-onnx/issues/2918)
    is open and unresolved.
  - The available substitutes — NeMo FastConformer streaming-multi @ 480 ms
    (114 M params, 5.7% test-other, no test-clean published, no
    punctuation) and Kroko Streaming Zipformer (~50 M, no published WER,
    no punctuation) — both regress accuracy meaningfully and lose
    Parakeet TDT v3's native punctuation/capitalization.
  - NVIDIA's high-quality streaming option (`nemotron-speech-streaming-en-0.6b`)
    is **NVIDIA-GPU only** by license and runtime; not deployable on
    Apple Silicon. Rejected.
  - Building our own streaming variant from the existing Parakeet
    checkpoint would require retraining with cache-aware attention masks
    — multi-week ML project, not justified by the ~640 ms latency saving.
  - [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)
    re-titled and rewritten to record the reasoning, the WER trade-off
    table, and the new accepted scope (Silero VAD auto-stop only, no
    recognizer swap).
  - [ADR-0007](#0007-performance-targets) latency table updated:
    **<200 ms p50 target retired** in favour of **<1 s p50 with WER ≤ 2%**,
    which the current shipped build already meets (~840 ms p50 on a 5 s
    utterance: 150 ms VAD hangover + 640 ms offline encoder + ~50 ms
    finalize).
