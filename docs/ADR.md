# Architecture Decision Records — parakeet-rs

One file, one ADR per heading. Status legend: **Accepted** (in code today),
**Proposed** (next pass), **Rejected** (considered and dropped),
**Superseded** (replaced by a later ADR).

The overarching goal is in [ADR-0007](#0007--performance-targets-beat-wispr-flow). Every decision
below should be re-evaluated against it.

---

## Current state vs target snapshot

This section tracks the gap between what the code does **today** and what the
ADRs target. Update whenever the code lands or a measurement is taken.

| Dimension | Today (measured / asserted) | Target (ADR-0007) | Blocker to close the gap |
|---|---|---|---|
| End-of-speech → text appears | Resident Parakeet Unified Core ML recognition starts speculatively behind an unchanged Silero stop authority. Tap Fast measured **182.0 ms p50 / 203.0 ms p95** on the representative 5 s production-path replay; pause-friendly Tap measured **637.0 ms p50 / 658.1 ms p95** on the 14.225 s endpoint fixture. | **<1 s p50** with the representative-speech quality gate intact | none — ADR-0022, ADR-0023, and ADR-0025 shipped and measured |
| Recognition acceleration | A resident native Swift worker owns the int8 Parakeet Unified Core ML graph on CPU+ANE. The sherpa CoreML backend remains the automatic load-failure and contextual-vocabulary fallback. An explicit per-chip tuner keeps CPU+ANE unless another bounded plan clears quality, memory, and ≥5% performance gates. | Native Apple Silicon execution with evidence-backed placement | none — ADR-0022 + ADR-0026 shipped; current M5 Pro evidence correctly retains CPU+ANE |
| Resident set | ~800 MB (640 MB mmap'd ASR model + ORT arenas + audio buffers); +~4 GB when polish is On (Qwen 3.5 4B Q6_K weights + KV cache); ~50 MB bundled dylibs | ≤5 GB steady state with polish On (revised with the 4B bump, ADR-0018 amendment) | none — ADR-0016 + ADR-0018 shipped |
| Settings window | Native `NSWindow` opened from menubar "Settings…" (`src/settings_ui.rs`); `orderFrontRegardless` so it surfaces above other apps | native, on-demand | none — shipped |
| Menubar UX | SF Symbols (`mic` / `mic.fill` / `arrow.down.circle`) via `objc2_app_kit::NSImage`; state-reflective menu labels | HIG-conformant template image with state | none — shipped |
| Paste path | `CGEventKeyboardSetUnicodeString` synthetic keystroke at `AnnotatedSession` tap layer (`src/ax_paste.rs`) | no clipboard mutation; works in terminals, browsers, native, Electron, IDEs | none — [ADR-0019](#0019--paste-delivery-synthetic-unicode-keystroke-annotatedsession) shipped, supersedes ADR-0011 |
| Smart formatting | In-process LLM polish pass: Qwen 3.5 4B Q6_K via llama-cpp-2 + Metal (`src/polish.rs`); opt-in via Settings → Polish → On | optional local polish, streaming output to cursor on word boundaries | none — [ADR-0018](#0018--polish-backend-llamacpp--qwen-35-2b-q4_k_m) shipped + amended (4B bump) |
| Custom vocabulary | A plain-text vocabulary selects sherpa beam search with validated hotword encoding; the default empty vocabulary keeps the faster native worker | explicit specialization without silently changing the generic model | none — ADR-0020 + ADR-0028 shipped and bounded by measured evidence |
| macOS permissions | Contextual Input Monitoring onboarding, just-in-time Microphone/Accessibility requests, a permanent dashboard, settings recovery links, and activation-time revocation detection | explain before requesting and remain usable when a grant is absent | implementation shipped in ADR-0029; destructive revocation confirmation remains issue #23 |

**Primary acceleration path complete.** ADR-0012 and ADR-0015 first proved the
sherpa fallback's CoreML execution. ADR-0022 then moved the default empty-
vocabulary path to a resident native Parakeet Unified worker, ADR-0023
overlapped recognition with endpoint confirmation, and ADR-0026 added safe
per-chip runtime selection without changing model weights.

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
its complexity is per-OS native helpers. The development hardware here is
M5 Pro and the target users are mac-first.

**Decision.** Apple Silicon macOS only (Intel deprecated). No `#[cfg]` arms
for other OSes, no cross-platform abstractions.

**Alternatives.**
- *Cross-platform via tauri-cross-platform-shortcut-y libs*: works, but every
  optimization in [ADR-0006](#0006--apple-silicon-optimization-plan-ds4-playbook-applied) becomes
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

**Status:** **Accepted as fallback; superseded as the default backend by
[ADR-0022](#0022--resident-native-core-ml-parakeet-unified-backend).**

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

**Status:** **Accepted as the fallback/contextual-vocabulary binding; superseded
as the default empty-vocabulary path by [ADR-0022](#0022--resident-native-core-ml-parakeet-unified-backend).**
The body below records the original binding decision and the static-library
risk that ADR-0012 later resolved for the fallback.

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
include the CoreML EP — see [ADR-0006](#0006--apple-silicon-optimization-plan-ds4-playbook-applied) for
the verification path. We do not own the inference code, so we cannot easily
add custom kernels.

**Historical risk — realised, then resolved.** The static upstream prebuilt did
not include the CoreML EP. This was verified 2026-05-15 by running
`nm -gU target/sherpa-onnx-prebuilt/sherpa-onnx-v1.13.2-osx-arm64-static-lib/lib/libonnxruntime.a | grep -i coreml`
which returned zero matches; available providers were CPU / CUDA / DML / Dnnl /
MIGraphX / Nnapi / OpenVINO / ROCM / TensorRT / VitisAI / CANN. Setting
`provider="coreml"` against that artifact was a silent no-op. ADR-0012 switched
the fallback to shared linkage with a CoreML-capable ONNX Runtime dylib, and
ADR-0015 added build/runtime verification.

---

## 0006 — Apple Silicon optimization plan (ds4 playbook applied)

**Status:** **Accepted for the CPU-side baseline; accelerator selection is
superseded by [ADR-0022](#0022--resident-native-core-ml-parakeet-unified-backend)
and [ADR-0026](#0026--evidence-gated-per-chip-core-ml-runtime-plans).** The
P-core scheduling, thread count, page-touch warmup, mmap'd weights, and
long-lived runtime object remain in the fallback path.

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
| Privacy | cloud | zero network calls after installed model artifacts pass local verification | **zero network calls after required/optional model downloads** | [ADR-0003](#0003--100-local-inference-no-cloud-apis-as-defaults) |
| Smart formatting | yes, cloud LLM | Qwen 3.5 4B Q6_K, local and streamed | **yes, local LLM post-pass** | met by ADR-0018/0019 |
| Resident set (steady state) | ~150 MB | ~800 MB asserted without polish; ~4 GB with Qwen loaded | **≤5 GB with polish On** | met after the ADR-0018 4B amendment |
| Battery cost / 30 min dictation | n/a | unmeasured | **<2% on M5 Pro** | resident CPU+ANE worker and idle QoS policy exist; a controlled energy protocol is still required |

**Honesty notes.**
- The resident-set target was previously set at "<400 MB including model"
  which is arithmetically impossible (the model alone is 640 MB on disk and
  some of those pages will be resident under any access pattern). Corrected.
- The <200 ms p50 latency target was **retired** after the
  [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)
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
[ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected). Clipboard +
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
VAD-auto-stop in [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)
because press-twice adds a user-input delay on top of inference
latency.

---

## 0009 — Silero-VAD auto-stop, offline encoder (Accepted) — streaming model SWAP REJECTED

**Status:** **Accepted in current form (press-once + Silero VAD + offline
Parakeet); streaming model swap rejected after measurement.**

Replaces [ADR-0008](#0008--hotkey-press-to-toggle--clipboard-paste) on the
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
   Losing it means we'd need [ADR-0010](#0010--local-llm-post-processing-for-smart-formatting)
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

**Revised target.** [ADR-0007](#0007--performance-targets-beat-wispr-flow) "<200 ms p50 felt
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
recognizer swap. ADR-0010 LLM post-pass became less urgent because Parakeet
already emitted punctuation. [ADR-0007]'s original sub-200 ms p50 target was
retired in favour of a more honest sub-1 s target. The 150 ms endpoint remains
available through explicit Tap Fast mode; normal Tap uses ADR-0025's
pause-friendly policy.

---

## 0010 — Local LLM post-processing for smart formatting

**Status:** **Superseded by [ADR-0018](#0018--polish-backend-llamacpp--qwen-35-2b-q4_k_m),
which shipped the local polish pass.** The body below is the original proposal.

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
- *Cloud LLM* (Anthropic/OpenAI): violates [ADR-0003](#0003--100-local-inference-no-cloud-apis-as-defaults).

**Consequences.** Realistic post-pass latency: **150–400 ms warmed**, not
the 50–150 ms originally quoted. Breakdown for a 1.5B-param model on M5 Pro
with KV cache warm and the model resident: ~50 ms TTFT, ~30 tokens output
at ~10 ms/token = ~300 ms generation, plus tokenize/detokenize overhead
~20–50 ms. Token streaming **does not help** here because we need a single
final string before injection — partial-rewrite streaming would cause
flicker in the target app and corrections after paste. If the post-pass
budget collides with [ADR-0007](#0007--performance-targets-beat-wispr-flow), default it off
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
surface in v1, especially since [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)
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

**Status:** **Accepted — shipped.** Spike per [ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation)
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

**Verification.** All three [ADR-0015](#0015--coreml-ep-verification-protocol)
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
  (#0007--performance-targets-beat-wispr-flow) latency story. Rejected.
- *Different inference binding* — see
  [ADR-0005](#0005--sherpa-onnx-as-the-inference-binding); already rejected.

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

**Context.** After [ADR-0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage)
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

## 0013 — Hotword / custom dictionary support

**Status:** **Superseded by [ADR-0020](#0020--vocabulary-sherpa-contextual-biasing-generated-from-a-plain-text-list),
which shipped a plain-text vocabulary and measured decoder policy.**

**Context.** Domain vocabulary (engineering terms, names, product names) gets
mistranscribed by general ASR. sherpa-onnx supports an external
`hotwords_file` that boosts specific n-grams during decoding.

**Decision.** Settings UI gains a "Custom vocabulary" textarea. Each line is
a hotword + optional boost score (`tauri 30.0\nshergaonnx 25.0`). Wired to
`OfflineRecognizerConfig.hotwords_file` (or its online equivalent).

**Consequences.** Minor decoder latency cost (negligible). Big accuracy win
on dictation about the user's actual work.

---

## 0014 — Tray-only headless UX

**Status:** **Superseded in implementation by [ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation).**
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
  [ADR-0011](#0011--direct-accessibility-text-injection-deferred)).
- **NSStatusBar + SwiftUI settings view** for Tauri tray + WebView settings.
- **Apple Foundation Model API** (macOS 26+) for the
  [ADR-0010](#0010--local-llm-post-processing-for-smart-formatting) post-pass.

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

## 0018 — Polish backend: llama.cpp + Qwen 3.5 2B Q4_K_M

**Status:** **Accepted — shipped and measured; amended to Qwen 3.5 4B Q6_K.**

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
   the v1 polish backend; gated on a measured Candle/llama.cpp miss.
3. **Gemma 4 E2B doesn't fit the <2 GB disk budget.** Q4_K_M is ~3 GB
   ([bartowski/google_gemma-4-E2B-it-GGUF](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF)).
   Going lower than Q4_K_M (Q3_K_M ~2.4 GB, Q2_K ~1.9 GB) hits the steep
   small-model quant degradation curve flagged by the [Qwen3
   quantization study](https://arxiv.org/html/2505.02214v1).

**Decision.** Replace `claude -p` (current `src/polish.rs` path) with
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
polish.

**Why llama.cpp, not Candle.** Candle ships neither Qwen 3.5 (new
hybrid Gated-DeltaNet architecture, `Qwen3_5ForConditionalGeneration`)
nor Gemma 4 Q4. llama.cpp picked up both within days of release. The
"pure Rust" constraint in the latency plan is interpreted as
"no Python, no subprocess, no HTTP" — FFI to a well-maintained C++
library (analogous to sherpa-onnx for ASR) satisfies it. The Metal
backend on Apple Silicon delivers ~100 tok/s on Qwen 3.5 2B Q4_K_M
(measured below).

**Measured Phase-0 numbers, M5 Pro 24 GB, 100 iterations
(`bench/polish-backends.csv`):**

| Metric | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| TTFT (ms) | 2.0 | 2.0 | 2.0 | 2.0 |
| Generation (ms) | 548 | 548 | 558 | 567 |
| **Total per polish (ms)** | **551** | **550** | **560** | **570** |
| Decode (tokens/sec) | 100.3 | 100.4 | 101.7 | 101.9 |

Cold model load: 229 ms (incurred once per process; the warmup is
done as part of `App::spawn_llm_setup` in `src/app.rs`, so polish
is ready before the user's first hotkey press).
Output: 55 tokens (one cleaned paragraph) for a 240-character noisy
input. p99 / p50 = 1.04 — variance is negligible, Metal kernel
scheduling is steady-state from iteration 1.

**Latency budget consequence.** Projected total post-endpoint latency
on a 5 s utterance with polish:

```
   362 ms   ASR (§1 measured)
+  150 ms   VAD hangover (vad.rs:15)
+   50 ms   paste finalize (latency-plan estimate)
+  550 ms   polish (this bench, p50)
= 1112 ms   total p50
```

That's **~112 ms over the latency-plan §6 acceptance criterion of
≤ 1.0 s p50 with polish**. Three mitigations on the table for the
§4 polish-rewrite work, in order of effort:

1. **Stream the paste**: emit cleaned tokens to NSPasteboard + ⌘V
   incrementally as the model generates, rather than buffering until
   end-of-sequence. The user feels latency as "first token visible",
   not "last token visible". Saves ~300–400 ms perceived; the actual
   wall clock to last-token is unchanged.
2. **Trim output cap**: typical polish output for a 30-token input
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
  polish). The in-process llama.cpp path replaces both the API and the
  prior `claude -p` subprocess approach.
- **`mlx-rs` direct.** Same multi-day port story as OminiX-MLX without
  the shared infrastructure crates.

**Open issues (resolved post-§6).**

- ~~The `/no_think` directive leaks into the model's output.~~
  Resolved — `strip_no_think_tail` in `src/polish.rs` handles all
  observed variants (`/no_think`, `no_think`, `no think`,
  `No think`, etc., case-insensitive, ignoring trailing punctuation).
- ~~Streaming-paste is non-trivial against the clipboard+⌘V shape.~~
  Resolved — `paste::Streamer` streams to the focused app via
  `CGEventKeyboardSetUnicodeString` keystrokes (ADR-0019), one
  word-boundary-batched chunk per LLM emission burst. No clipboard,
  no AX, no flicker.
- ~~Open: model file management.~~ Resolved 2026-06-11 alongside the
  4B bump — `model_fetch::ensure_polish_model` auto-downloads the
  GGUF on first polish-enable (same `.part`-validate-rename flow as
  the ASR first-run fetch), wired into `load_llm_blocking` so the
  boot and Settings-toggle paths share it. A settings-test pins the
  download URL's filename to `Settings::polish_model_path`.

**References.**

- `bench/polish-backends.csv` — full 100-row Phase-0 data, this M5 Pro.
- `src/bin/bench_llm.rs` — the bench harness.
- [Welcome Gemma 4 — Hugging Face](https://huggingface.co/blog/gemma4)
- [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [llama-cpp-2 crate](https://crates.io/crates/llama-cpp-2)
- [Qwen 3.5 vs Gemma 4 size-matched benchmarks — Maniac](https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size)
- [Qwen3 quantization empirical study (arxiv)](https://arxiv.org/html/2505.02214v1)

**Amendment (2026-06-11): model bumped to Qwen 3.5 4B Q6_K.**

The shipped 2B's instruction-following misses were the polish pass's
dominant quality complaint: paraphrasing, over-deleting legitimate
"like", fumbling `scratch that`. With the disk budget relaxed from
<2 GB to ≤5 GB, the swap went to
[`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
→ `Qwen3.5-4B-Q6_K.gguf` (3.53 GB). Same family ⇒ the ChatML +
`/no_think` template, tail-strip, and `PolishBackend` plumbing carry
over unchanged; only `Settings::polish_model_path` moved.

Why 4B Q6_K and not the alternatives:

- **9B**: every 4-bit quant exceeds 5 GB (Q4_K_S 5.39 GB); only Q3
  fits, which re-enters the small-model quant-degradation curve this
  ADR already rejected. Decode would also drop to ~30 tok/s.
- **4B at Q4_K_M (2.74 GB)**: fits easily, but the relaxed budget
  buys Q6_K's negligible-loss quant for free.
- **Other families (Llama 3.3 8B, Phi-4-mini, Mistral)**: nothing
  at ≤5 GB clearly beats Qwen3.5-4B on instruction following, and
  all cost chat-template migration + no-think revalidation.
- **Apple Foundation Models (WWDC26 AFM 3)**: zero-download wildcard;
  Swift-only API needs an `@objc` shim dylib. Tracked as a possible
  second `PolishBackend`, not a blocker for this bump.

Measured on the same harness (`bench/README.md` §6 follow-up):
total p50 550 ms → 1225 ms, decode 100 → 43 tok/s, TTFT 29 ms.
Streaming paste (ADR-0019) keeps perceived latency at
time-to-first-words, so the 2.2× wall-clock cost lands after the
user already sees text flowing. Resident memory with polish On rises
~1.6 GB → ~4 GB (weights + KV); the steady-state budget row at the
top of this file is updated accordingly.

---

## 0017 — CoreML `ModelCacheDirectory` blocked at the sherpa-onnx Rust binding

**Status:** **Superseded as the primary optimization path by
[ADR-0022](#0022--resident-native-core-ml-parakeet-unified-backend).** The
sherpa binding limitation remains true for the fallback backend.

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
criterion of ≤ 700 ms p50 no-polish. The §2 optimization helps
*first-launch* cold start only (where the user feels CoreML's MLProgram
graph compile cost). That's a real win to grab eventually, but it's
not gating the ship of §6 (polish rewrite + acceptance numbers).

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
  ASR → polish polish → keystroke posted → focused-app insertion),
  with the polish pipeline contributing ~550 ms of that.

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

## 0020 — Vocabulary: sherpa contextual biasing, generated from a plain-text list

**Status:** **Accepted — shipped.**

**Context.** Dictation fails predictably on the same class of words:
proper nouns, product names, trade jargon, colleagues' surnames. The
recognizer is fixed and small; nothing in the pipeline let a user tell
it "this word exists". sherpa-onnx supports contextual biasing
(hotwords) for offline transducers, so the capability was already in
the tree — but three properties of it had to be measured against
`parakeet-tdt-0.6b-v3-int8` before it could be exposed, because getting
any of them wrong produces a feature that silently does nothing.

**Measurements** (M5 Pro, `bench/audio/5s.wav`, int8 + CoreML):

1. **Greedy decoding rejects a hotwords file outright.**
   `OfflineRecognizer::create` returns `None` with
   `Please use --decoding-method=modified_beam_search if you provide
   --hotwords-file`. Biasing therefore forces beam search — it is not
   an opt-in refinement on top of the existing decode path.

2. **Beam search costs ~13%.** 396 ms → 448 ms per 5 s decode
   (greedy vs `modified_beam_search`, `max_active_paths=4`). Real but
   affordable, and only paid when the user has terms.

3. **A plain word does nothing — silently.** The model ships no
   `bpe.model` (the HF repo contains only the three ONNX files and
   `tokens.txt`), so sherpa falls back to `modeling_unit="cjkchar"` and
   looks up each whitespace-separated piece as a single token.
   `Kubernetes` is not a token, so it fails to encode and the decode is
   unchanged **even at `hotwords_score=50`**. This is the trap: the
   naive implementation ships, appears to work, and does nothing.

   The tokens are SentencePiece pieces, so the correct encoding is
   per-character with `▁` (U+2581) marking word-initial position:

   | vocabulary.txt | encoded | decode at score 50 |
   |---|---|---|
   | `Kubernetes` | `Kubernetes` | *(unchanged — no effect)* |
   | `Kubernetes` | `K u b e r n e t e s` | `KubernetesKubernetes…` (glued) |
   | `Kubernetes` | `▁K u b e r n e t e s` | `Kubernetes Kubernetes …` ✓ |
   | `New York` | `▁N e w ▁Y o r k` | `New York New York …` ✓ |

4. **Safe boost range.** With a realistic 3-term vocabulary, scores
   1.0–6.0 left all five bench fixtures byte-identical to the greedy
   baseline. 10.0 began injecting hotwords into audio that did not
   contain them; 50.0 replaced the transcript with them entirely.
   **Default: 2.0.**

**Decision.**

- The user edits `vocabulary.txt` — one term or phrase per line, `#`
  comments, written naturally. `crate::vocabulary` owns the translation
  into sherpa's format. The raw format is not exposed; measurement 3 is
  precisely why asking a user to write it themselves would be a bug
  farm.
- **Every encoded piece is validated against `tokens.txt` before the
  hotwords file is written.** Measurement 3's failure mode is not
  limited to whole words: any character with no token (emoji, `Ω`, a
  decomposed `é`, smart punctuation) makes sherpa drop that hotword
  with a stderr log the bundled app discards. Validating up front turns
  "your term silently does nothing" into a named warning identifying
  the term and the offending piece. If validation rejects everything,
  we fall back to greedy rather than paying beam-search cost for an
  empty context graph.
- A plain file opened via `NSWorkspace`, not an in-window text view.
  This is a list people paste into, sort, and keep in dotfiles;
  `NSTextView` would be a worse editor than the one they already have.
- Greedy stays the default. `AsrConfig.hotwords: Option<&Path>` selects
  the decoding method, so an empty or absent vocabulary costs nothing —
  no beam search, no behaviour change, byte-identical transcripts.
- Biasing is baked in at recognizer construction, so a vocabulary edit
  requires a rebuild. Settings-Save rebuilds in the background via
  `DictationFsm::try_claim_model_reload`, which atomically takes the
  app out of `Idle` so a hotkey press can't start a session against a
  recognizer that is about to be dropped. The previous recognizer is
  replaced only on success — a malformed vocabulary costs the user
  their custom terms, never their ability to dictate.

**Staleness tracking (`App::loaded_biasing`).** Three bugs here were
caught in adversarial review, and all three share one root cause:
comparing against *what settings said* rather than *what was loaded*.

  1. Comparing `prev` vs `new` settings meant a **failed** rebuild
     never retried — the second Save saw `prev == new` and did nothing,
     leaving a recognizer that didn't match the config indefinitely.
  2. Fingerprinting the vocabulary file *after* the build recorded a
     newer file than was actually read if the user edited mid-build.
     That erased the very mismatch that would have triggered the retry,
     so it was permanent.
  3. A rebuild requested while the app was busy printed "applies after
     the current dictation" and was then dropped — nothing drained it.

The fix is one type: `Biasing { vocab: Option<(len, mtime)>, score }`,
sampled **before** the build, returned by `load_asr_blocking`, and
stored only on success. Staleness is `loaded != Biasing::sample(...)`.
Failure leaves the old value, so the mismatch persists and the retry
happens; a mid-build edit compares unequal to the pre-build sample and
gets picked up by the re-check the worker runs on completion; and a
busy app sets `reload_pending`, drained by `on_session_finished`.

**Consequences.** Verified by `asr_diff` (ADR-0021): the default
vocabulary + score is transparent on all five fixtures, and the
harness catches the drift at score 20 with 100% divergence.

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

**Status:** **Accepted — implemented and measured.**

**Context.** The sherpa-onnx/CoreML path was already usable at roughly
13–14× real time, but its ONNX execution and decoder path left a large gap to
hardware-specialized Parakeet runtimes. The target for this change was not an
external benchmark number: it was at least **3× the frozen previous backend**
under the same local harness, while preserving the gold-reference WER/CER
gate. The implementation also needed to keep Python out of the shipping app
and retain the existing custom-vocabulary behavior.

**Decision.** Prefer FluidAudio's int8 Parakeet Unified EN 0.6B offline model
through a resident Swift worker. Rust remains the application and policy
layer; `AsrBackend` is the stable seam. Rust owns the pinned download and
integrity gate. The worker owns model load, native mel extraction, Core ML
CPU+ANE execution, and greedy RNNT decode, then accepts framed little-endian
Float32 audio over stdin and returns framed JSON over stdout. It is pinned to
FluidAudio commit `00a9aa771900ea09c485659663be31019e293e47`.

- Model load and ANE plan compilation happen once per worker, not once per
  utterance.
- The worker is built as a 15 MB arm64 helper, copied into
  `Contents/MacOS/`, signed before the main executable, and verified with the
  complete app bundle.
- A non-empty custom vocabulary deliberately selects sherpa, whose modified
  beam-search hotword graph preserves ADR-0020. Empty vocabulary selects the
  optimized backend.
- Specialized model download/load failure falls back to sherpa. The fallback
  remains downloaded on first launch so the app does not become unusable when
  a hardware-specific plan cannot load.
- The Core ML weights are a separate ~595 MB on-demand artifact under
  CC-BY-4.0. FluidAudio is Apache-2.0; attribution is shipped in
  `THIRD_PARTY_NOTICES.md`.

**Evidence.** M5 Pro, 24 GB, macOS 26.5.1, release builds, identical 48 kHz
fixtures, three warmups, 30 measured repetitions per bucket. Both rows include
Rust↔worker IPC and resampling because the outer timer wraps
`Asr::recognize()`.

| bucket | sherpa p50 | unified p50 | speedup | sherpa p95 | unified p95 | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1 s | 112.0 ms | 35.0 ms | **3.20×** | 116.5 ms | 36.5 ms | **3.19×** |
| 3 s | 226.0 ms | 50.0 ms | **4.52×** | 239.6 ms | 51.0 ms | **4.70×** |
| 5 s | 361.5 ms | 66.0 ms | **5.48×** | 384.3 ms | 67.0 ms | **5.74×** |
| 10 s | 580.0 ms | 90.0 ms | **6.44×** | 597.8 ms | 91.5 ms | **6.53×** |
| 20 s | 1195.0 ms | 188.0 ms | **6.36×** | 1245.3 ms | 192.6 ms | **6.47×** |

The versioned five-fixture smoke gate passed at **2.38% WER / 2.22% CER**
against 4% / 3% limits, with punctuation/capitalization retained. This corpus
is macOS `say` output and proves regression-harness compatibility, not
production WER; representative recorded speech remains a release gate.

**Consequences.** The 3× target holds at p50 and p95 in every measured bucket,
including one-second audio where fixed IPC cost dominates. Optimized model
cold load measured about 10 seconds on first Core ML plan creation and about
0.12 seconds from a warm compiled cache; startup warmup keeps both off the
dictation path. The helper requires macOS 14; older supported systems fail the
specialized spawn/load and use sherpa.

The later real-speech gold gate measured 5.43% WER / 3.57% CER with zero
ten-repeat spread. Against sherpa greedy it was 6.21× faster at corpus-decode
p50 and 6.39× at p95; model load was 0.130 s versus 3.647 s, first result
0.278 s versus 4.582 s, and observed full-process-tree RSS 0.10 GiB versus
4.25 GiB. Production now verifies the complete 15-file Core ML bundle from
model revision `4252711f6f060f9a2f91e5f081a806d7f45eebd8` through the Rust
integrity gate before the worker may load it. `PARAKEET_ASR_BACKEND=sherpa`
keeps the fallback one setting away. The exact graph, tokenizer, decoder,
license, compatibility, and conversion limitation are recorded in
[`docs/asr/COREML_MODEL.md`](asr/COREML_MODEL.md).

Paired outer/internal instrumentation also prices the worker boundary rather
than inferring it from bucket scaling. At 30 repetitions the boundary was
0.128 ms p50 / 0.143 ms p95 for one-second audio and 1.111 ms / 1.209 ms for
20-second audio. The 35 ms short-utterance floor is model-side work, so shared
memory or an in-process rewrite is not justified by current measurements.

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
| frozen sherpa + serial ASR | 613.6 ms | 613.0 ms | 652.5 ms | 657.3 ms |
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

**Status:** **Accepted — implemented.**

**Context.** The first-run downloader previously treated an existing path as a
valid model. New downloads were checked against the response's optional
`Content-Length`, but that length was supplied by the same server as the bytes.
A truncated response without the header, a wrong upstream object of the same
length, or a locally changed file could therefore reach an inference runtime.

**Decision.** Give every artifact fetched by `model_fetch.rs` a code-reviewed
identity: an immutable Hugging Face revision where available, a published byte
length, and a SHA-256 digest. Hash each response while streaming it to a
same-directory `.part` file, flush it, and rename it only after the pinned
digest and length match. `Content-Length` is not an integrity input. A mismatch
deletes the partial or existing artifact and the normal first-use path fetches
it again.

Existing artifacts are hashed before their first use by this version. A JSON
sidecar beside each model records the expected digest plus the file's size,
mtime, ctime, device, and inode. Later launches may skip the multi-gigabyte hash
only when all cached identity fields still match. Malformed/unreadable cache
data is a cache miss, never proof of integrity; cache writes use fsync plus an
atomic same-directory rename.

| Artifact | Immutable revision / release | Bytes | SHA-256 |
|---|---|---:|---|
| `tokens.txt` | sherpa `2bda32e` | 93,939 | `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |
| `decoder.int8.onnx` | sherpa `2bda32e` | 11,845,275 | `179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e` |
| `joiner.int8.onnx` | sherpa `2bda32e` | 6,355,277 | `3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3` |
| `encoder.int8.onnx` | sherpa `2bda32e` | 652,184,281 | `acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247` |
| `silero_vad.onnx` | sherpa-onnx `asr-models` release asset | 643,854 | `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6` |
| `Qwen3.5-4B-Q6_K.gguf` | Qwen `e87f176` | 3,525,956,768 | `fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66` |

The sherpa and Qwen revisions are the upstream repository commits
[`2bda32ec70b097a55adaa07d9a7173915b43cc78`](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/commit/2bda32ec70b097a55adaa07d9a7173915b43cc78)
and
[`e87f176479d0855a907a41277aca2f8ee7a09523`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/commit/e87f176479d0855a907a41277aca2f8ee7a09523).
Silero's mutable release URL is made byte-immutable by the pinned digest.

**Scope.** This ADR's original table covers six artifacts. ADR-0022 now adds
the separately reviewed 15-file Core ML truth pack and reuses this verifier,
cache, and atomic-publish path before the Swift worker may load it. Its exact
manifest is maintained in `model_fetch.rs` and
[`docs/asr/COREML_MODEL.md`](asr/COREML_MODEL.md).

**Consequences.** A first launch after upgrading reads and hashes each existing
Rust-managed model once. On later launches the integrity gate is a handful of
metadata reads and small sidecar reads. Changing a model now requires an
intentional code review of revision, size, and digest together.

Measured on the M5 Pro release bundle against the installed artifacts, the
one-time SHA-256 pass took 1.291 s for the 671 MB sherpa/VAD set and 6.313 s for
the 3.53 GB polish GGUF. The next launch's metadata-cache checks took 0.738 ms
and 2.632 ms respectively. This is a launch/first-use gate, not part of
recording, endpointing, or inference.

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
  gate exactly.
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

**Status:** **Rejected — measured no-go; no production code added.**

**Context.** Qwen3-ASR 0.6B offers multilingual offline/streaming recognition
and quantized MLX conversions. Its q8 artifact could improve offline lexical
quality while applying the same fixed-model/hardware-specialization thesis as
ADR-0026, but the shipping app cannot add Python and no reviewed native Rust or
Core ML implementation exists.

**Decision.** Retain native Core ML Parakeet Unified as the default and
sherpa-onnx as the contextual-vocabulary/load-failure fallback. Treat q8 as the
only default Qwen quantization candidate; reject q4 unless it wins quality as
well as memory/latency. Do not fund a native Qwen port, package its weights, or
expose it through the backend selector until a maintained native runtime passes
the same per-category, real-boundary streaming, process-cold startup, latency,
memory, immutable-artifact, signing, and fallback gates.

**Evidence.** On M5 Pro, the pinned q8 MLX oracle measured 4.35% WER / 3.15%
CER offline, 0.840 s corpus p50, 40.5× RTFx, 1.12 GiB peak RSS, and a 960 MiB
weight artifact. It improved aggregate offline quality but was 1.86× slower and
about 11.0× larger in resident memory than shipping Parakeet, while regressing
the `noisy` and `numbers` category rows. q4 added only 17.2% throughput over q8
and regressed WER to 6.52%; fp16 was 3.52× slower than q8.

At actual 2-second model boundaries, q8 measured 23.91% WER / 21.01% CER under
exact, unpaced 100 ms segmented, and jittered transport writes. The 14.225 s
fixture was exact offline and 41.86% WER streaming. A native `mlx-rs` 0.25.3
release spike bound
the needed primitive ops but failed before linking because Xcode's separately
downloaded Metal Toolchain was absent; the crate also raises the MSRV from 1.77
to 1.82 and still requires a several-thousand-line architecture/tokenizer/
cache/streaming port plus a separately packaged `mlx.metallib`.

**Consequences.** No Qwen, MLX, Candle, Python, model download, runtime switch,
or MSRV change enters the app. The PEP 723 oracle and losing measurements stay
checked in for future comparison. Multilingual evaluation is deferred because
the English product, streaming, latency, memory, and native-build gates already
fail. Full evidence and the reopen conditions are in
[`docs/asr/QWEN3_ASR_EVALUATION.md`](asr/QWEN3_ASR_EVALUATION.md).

---

## 0028 — Generic ASR base with evidence-gated domain and user adaptation

**Status:** **Accepted architecture; training rejected on current evidence.**

**Context.** A generic model can be specialized along two independent axes:
semantic behavior for a domain or speaker, and deployment/runtime behavior for
hardware. Calling both “fine-tuning for the Mac” risks training on too little
user data, binding personal identity to a chip artifact, or using QAT to solve a
vocabulary problem. The product already has the cheapest semantic mechanism:
sherpa contextual vocabulary with a configurable score.

The frozen shipping Core ML baseline has one repeated lexical class in this
seven-fixture corpus: three word edits across two custom-vocabulary fixtures.
The other lexical failure is a single spoken-number-form example. That is
enough to test the existing vocabulary mechanism, not enough to estimate a
trainable error distribution.

**Decision.** Retain the immutable generic int8 Parakeet Unified Core ML model
and its CPU+ANE default. Do not collect training data, train an adapter,
distill, run QAT, or raise the global hotword score from current evidence.
Before learned weights, attempt a constrained native vocabulary/lexical-
rescoring layer when newly separated data proves a repeated target class.

Keep specialization layers distinct:

- a domain adapter is global/organization-specific and portable across Macs;
- a personal adapter is user-specific, local, removable, and also portable;
- a QAT/distilled export may target an Apple hardware family but never encodes
  user ownership;
- the existing runtime profile remains keyed to exact chip, OS, model digest,
  and workload, with CPU+ANE fallback.

Any future collection must preregister consent/retention, speaker- or session-
disjoint train/development/blind-test splits, category thresholds, and artifact
identity. The current seven fixtures were used to locate vocabulary-score
transitions, so they are diagnostic rather than a fresh blind test for future
adaptation and remain prohibited from training/calibration/QAT/distillation.

**Evidence.** A 27-point score exploration from 0 through 50 was narrowed to
six ten-repeat boundary rows on M5 Pro. Score 0 and the default score 2 were
transcript-identical to greedy while costing 13.2% and 13.9% p50. The first
effect at 2.75 improved custom-vocabulary WER from 54.55% to 27.27%, but changed
noisy `Amy` to `80`, worsened CER from 5.46% to 6.09%, regressed noisy and
numbers categories, and cost 14.2% p50. Score 4.5 began producing `Olly` while
injecting `IBM` into unrelated audio. Score 6 reached 42.39% WER; scores 8–50
measured 96.74–119.57% WER. All repeated outputs were deterministic, and no
sherpa row cleared the frozen shipping Core ML overall/per-category gate.

Report schema v3 now embeds decoder method, requested/active vocabulary state,
score, term count, source vocabulary digest, and generated-hotword digest. The
raw reports, compact summary, and PEP 723/`uv` verifier live under
`bench/domain-adaptation/`.

**Consequences.** The architecture supports the requested generic-base-plus-
specialization design without manufacturing a training project. A future
adapter has explicit ownership, deletion, blind-test, accuracy, repeatability,
latency, memory, and fallback gates before collection begins. Any QAT or
distillation result must report compiled artifact bytes, process-tree RSS,
load/first-result/p50/p95 latency, and overall/per-category quality, and must
reduce a targeted deployment resource by at least 10% without quality loss.
Full thresholds and reopen conditions are in
[`docs/asr/DOMAIN_ADAPTATION.md`](asr/DOMAIN_ADAPTATION.md).

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

## Target status index

| ADR-0007 target | Owner ADR | Status | Blocked by |
|---|---|---|---|
| Accelerated Core ML path present | [0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage) + [0015](#0015--coreml-ep-verification-protocol) + [0022](#0022--resident-native-core-ml-parakeet-unified-backend) | **Shipped + measured** — native worker quality/latency gates pass; the sherpa fallback retains its CoreML symbol/runtime checks | exact per-op placement remains Apple-managed |
| <1 s p50 felt latency (revised from <200 ms — see [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)) | [0009] + [0022](#0022--resident-native-core-ml-parakeet-unified-backend) + [0023](#0023--speculative-decode-behind-an-unchanged-endpoint-authority) + [0025](#0025--pause-friendly-tap-with-an-explicit-low-latency-mode) | **Shipped + measured** — Tap Fast is 182.0 ms p50 on the representative 5 s bucket; pause-friendly Tap remains below 1 s on the endpoint corpus | nothing |
| Live partial transcripts | [0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected) + [0023](#0023--speculative-decode-behind-an-unchanged-endpoint-authority) | **Out of scope** — no quality-preserving streaming model won; speculative final decode provides the measured latency win | a new candidate must beat the existing quality/latency/resource gates |
| CPU+ANE requested and performance-gated | [0022](#0022--resident-native-core-ml-parakeet-unified-backend) + [0026](#0026--evidence-gated-per-chip-core-ml-runtime-plans) | **Shipped** — explicit tuner retained CPU+ANE on the M5 Pro after all challengers failed the ≥5%/quality/memory gates | Core ML owns final per-op placement |
| ≤5 GB resident set with polish On | [0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation) + [0018](#0018--polish-backend-llamacpp--qwen-35-2b-q4_k_m) + [0022](#0022--resident-native-core-ml-parakeet-unified-backend) | **Shipped** — native tray shell, resident Core ML worker, and Qwen mmap/lifecycle | nothing |
| Smart formatting parity with Wispr Flow | [0018](#0018--polish-backend-llamacpp--qwen-35-2b-q4_k_m) | **Shipped** — optional local Qwen polish streams on word boundaries | strict last-token latency remains above the original target |
| Clipboard not clobbered | [0019](#0019--paste-delivery-synthetic-unicode-keystroke-annotatedsession) | **Shipped** on the normal path; clipboard is rescue-only for observable delivery failure | `CGEventPost` has no delivery receipt |
| Custom vocabulary | [0020](#0020--vocabulary-sherpa-contextual-biasing-generated-from-a-plain-text-list) + [0022](#0022--resident-native-core-ml-parakeet-unified-backend) + [0028](#0028--generic-asr-base-with-evidence-gated-domain-and-user-adaptation) | **Shipped + bounded** — non-empty vocabulary selects sherpa; no measured global score clears the quality gate | constrained native Unified biasing requires new separated evidence |

**Completed path to the ADR-0007 latency claim:**
1. [ADR-0022](#0022--resident-native-core-ml-parakeet-unified-backend) — move
   the generic default path to a pinned resident Parakeet Unified worker while
   retaining sherpa for fallback and vocabulary.
2. [ADR-0026](#0026--evidence-gated-per-chip-core-ml-runtime-plans) — keep
   hardware placement evidence-gated, quality-safe, and reversible.
3. [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected)
   + [ADR-0025](#0025--pause-friendly-tap-with-an-explicit-low-latency-mode) —
   retain Silero endpoint authority with pause-friendly and explicit fast modes.
4. [ADR-0023](#0023--speculative-decode-behind-an-unchanged-endpoint-authority)
   — overlap recognition with confirmation and clear the matched 3× p50/p95 gate.

Anything not on this table is either accepted-and-done or out of scope.

## Change log

- **2026-08-11** — [ADR-0029](#0029--contextual-macos-permission-onboarding-and-recovery)
  added: ZoomItForMac-inspired permission-state separation, contextual
  request scopes, explicit denied/restricted/revoked recovery, targeted
  settings links with fallback, activation refresh, and a signed manual QA
  matrix without adding Screen Recording or Camera access.

- **2026-08-11** — [ADR-0028](#0028--generic-asr-base-with-evidence-gated-domain-and-user-adaptation)
  added: a 27-point vocabulary sweep with six repeated boundary reports, a
  measured no-training decision, explicit global/domain/personal/hardware
  ownership, pre-collection split/privacy gates, and report-schema-v3 decoding
  provenance.

- **2026-08-11** — [ADR-0027](#0027--qwen3-asr-q8-apple-silicon-challenger)
  added: pinned q8/q4/fp16 offline evidence, actual mid-chunk streaming
  measurements, `mlx-rs`/Candle/Core ML feasibility and packaging analysis,
  and a measured no-go that leaves the production backend unchanged.

- **2026-08-11** — [ADR-0026](#0026--evidence-gated-per-chip-core-ml-runtime-plans)
  added: bounded Core ML candidate tuning, short/long regime selection,
  hardware/OS/model/tuner cache invalidation, quality/category/memory gates,
  an inspectable atomic profile, and measured M5 Pro negative evidence that
  keeps CPU+ANE.

- **2026-08-11** — [ADR-0025](#0025--pause-friendly-tap-with-an-explicit-low-latency-mode)
  added: 750 ms pause-friendly Tap default, explicit 150 ms Tap Fast, and a
  versioned real-speech gate coupling zero false stops with sub-second p95
  final-pause latency.

- **2026-08-10** — [ADR-0024](#0024--immutable-identities-for-rust-managed-model-artifacts)
  added: immutable Hugging Face revisions, pinned length/SHA-256 identities,
  streaming `.part` verification, corrupt-file refetch, and metadata-cached
  first-use verification for all six Rust-managed model artifacts.

- **2026-08-10** — [ADR-0023](#0023--speculative-decode-behind-an-unchanged-endpoint-authority)
  added: dual-VAD speculative decode with the original confirmer retained as
  stop authority, true acoustic-end timing, deterministic BlackHole replay,
  and a passing 30-repetition 3.37× p50 / 3.21× p95 end-to-end gate.

- **2026-08-10** — [ADR-0022](#0022--resident-native-core-ml-parakeet-unified-backend)
  added: pinned resident FluidAudio worker, int8 Parakeet Unified CPU+ANE
  backend, automatic sherpa fallback, vocabulary-aware backend selection,
  matched 30-repetition evidence showing 3.20–6.44× p50 speedups, and a
  passing 2.38% WER / 2.22% CER smoke gate.

- **2026-05-15** — Codex challenge review (`/codex challenge docs/ADR.md`)
  surfaced eight findings. Verified the most critical (no CoreML EP in
  prebuilt static lib) via `nm -gU` symbol inspection. Material revisions:
  - Added [Current state snapshot](#current-state-vs-target-snapshot) so the
    code-vs-target gap is impossible to overlook.
  - [ADR-0005](#0005--sherpa-onnx-as-the-inference-binding) updated to
    record the realised CoreML EP risk with evidence.
  - [ADR-0006](#0006--apple-silicon-optimization-plan-ds4-playbook-applied)
    split: CPU optimizations Accepted, CoreML / ANE claims downgraded to
    Proposed-gated-on-0012.
  - [ADR-0007](#0007--performance-targets-beat-wispr-flow) latency table gained a "Today
    (baseline)" column; the impossible "<400 MB resident set including
    model" target replaced with the honest "≤1.2 GB" steady-state target.
  - [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected) VAD silence
    threshold tightened from 250 ms to 150 ms; latency budget made
    explicit and shown to require ADR-0012 to hold.
  - [ADR-0010](#0010--local-llm-post-processing-for-smart-formatting)
    post-pass latency estimate bumped from "50–150 ms" (hand-waved) to
    "150–400 ms warmed" (engineered).
  - [ADR-0011](#0011--direct-accessibility-text-injection-deferred) **deferred to
    v2** by user direction.
  - [ADR-0012](#0012--sherpa-onnx-prebuilt-with-coreml-ep-shared-linkage) **promoted
    from Proposed to Accepted**; vendor as submodule rather than env-var
    redirection; explicit cmake flag list; honest maintenance cost.
  - New [ADR-0015](#0015--coreml-ep-verification-protocol) added with a
    three-layer verification protocol (build-time symbol check, runtime
    provider log parse, per-utterance latency probe).
  - New [ADR-0016](#0016--tauri--rust-shell-vs-swiftui-native-re-evaluation)
    re-opens the Tauri-vs-SwiftUI question now that we're mac-only and
    ADR-0012 has revealed real maintenance costs. Decision time-boxed to a
    ≤ 4 h sherpa-onnx-with-CoreML build spike; explicit pivot/continuation
    triggers documented.

- **2026-05-15** (later) — Implementation pass landed:
  - [ADR-0014](#0014--tray-only-headless-ux) shipped: settings window
    `visible: false` at launch; tray menu opens it on demand.
  - [ADR-0009](#0009--silero-vad-auto-stop-offline-encoder-accepted--streaming-model-swap-rejected) shipped (offline
    encoder variant): press-twice toggle deleted; new `streamer.rs` +
    `vad.rs` modules drive Silero VAD over an audio tap channel; cancel-on-
    second-press preserved. Streaming Parakeet still future work.
  - [ADR-0015](#0015--coreml-ep-verification-protocol) implemented: layer 1
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
  - [ADR-0007](#0007--performance-targets-beat-wispr-flow) latency table updated:
    **<200 ms p50 target retired** in favour of **<1 s p50 with WER ≤ 2%**,
    which the current shipped build already meets (~840 ms p50 on a 5 s
    utterance: 150 ms VAD hangover + 640 ms offline encoder + ~50 ms
    finalize).
