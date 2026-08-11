# parakeet-rs — Big-Improvement Review (2026-07-12)

> Historical point-in-time review. Several recommendations were implemented or superseded after publication. See the project [README](../README.md) and [ADR](../docs/ADR.md) for current behavior and decisions.

**Build/test status:** `cargo build --release` clean (1m08s); `cargo test --release` 84/84 lib tests pass, 0 doc-test/bin failures. Toolchain fine at MSRV 1.77 / edition 2021 — no issues.

**Current stack (verified in code):** Parakeet TDT 0.6B v3 **int8** (csukuangfj sherpa-onnx export) via `sherpa-onnx` crate 1.13.2, `provider="coreml"` (shared `libonnxruntime.dylib`), offline greedy transducer decode, measured **7.8x RTFx** on this M5 Pro. Silero VAD auto-stop, 30 s hard cap per utterance (both modes). Polish: Qwen 3.5 4B Q6_K via `llama-cpp-2` + Metal, greedy, ctx 2048 / 768-token output cap, streamed to cursor via CGEvent keystrokes.

Only improvements I'd defend are listed. Weak ideas killed at the bottom.

---

## Ranked improvements

### 1. Swap the ASR model to Qwen3-ASR (0.6B now, 1.7B when exported) — biggest accuracy win

**What.** Alibaba's Qwen3-ASR (open-weighted Jan 2026, Apache 2.0): AuT conformer encoder (180 M) + Qwen3 LLM decoder. 52 languages, and — unlike Parakeet — **context-token hotword biasing** (arbitrary text in the system prompt customizes recognition; solves ADR-0013 for free) and a **streaming-unified architecture** (2 s chunks; 1.7B degrades only 2.69 % → 3.33 % WER in streaming mode), which would let ADR-0009's "no viable streaming model" verdict be re-opened later without a WER sacrifice.

**Impact (quantified).**
- Official: LibriSpeech test-clean **1.63 %** (1.7B) / 2.11 % (0.6B) vs Parakeet v3's 1.93 %; test-other **3.38 %** (1.7B) vs ~4.5 %.
- Independent M5 Pro benchmark ([Soniqo](https://soniqo.audio/benchmarks), first 200 test-clean utts): Qwen3-ASR 1.7B 5-bit **1.32 % WER at 36x real-time** vs Parakeet TDT v3 int8 **2.37 %** — i.e. ~45 % relative error reduction against the *same int8 Parakeet this app ships*. 36x RT ⇒ a 5 s utterance decodes in ~140 ms, still well inside the <1 s budget.
- Hotword biasing: user dictionary / app-specific jargon (names, product terms) recognized at ASR time instead of hoping the polish LLM fixes them.

**Integration path — already cheap.** `sherpa-onnx` crate **1.13.4 (published 2026-07-08) already exposes `OfflineQwen3ASRModelConfig`** (also Canary and CohereTranscribe configs), and k2-fsa published `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`. The 1.7B int8 export is requested upstream ([k2-fsa/sherpa-onnx#3535](https://github.com/k2-fsa/sherpa-onnx/issues/3535)); until it lands, 0.6B is the drop-in.

**Caveats (verify in the A/B):**
- Punctuation/capitalization of the *open* checkpoints' raw output: expected (LLM decoder, same family as the punctuating Qwen3-ASR-Flash API) but **UNVERIFIED** — must confirm on the sherpa-onnx export before committing, since native punctuation was a load-bearing reason for choosing Parakeet (ADR-0004).
- int8 quality of the k2-fsa 0.6B export vs the fp numbers above: unmeasured.
- LLM-decoder ASR is autoregressive — slower than TDT (RTF 0.064 at the 0.6B per the tech report on server GPUs; CPU int8 RTFx on M5 Pro needs measuring, expect 10–30x — fine, but bench it).

**Effort.** Small for 0.6B: bump crate 1.13.2 → 1.13.4, add the Qwen3 config arm in `asr.rs`, change `model_fetch.rs` URLs. Medium for 1.7B (wait for upstream export, or export yourself with their scripts).

**First step.** Bump `sherpa-onnx` to 1.13.4, pull the 0.6B int8 export, and run `bench_asr` A/B (WER on your own dictation recordings + RTFx) against Parakeet v3. Check punctuation in the raw output.

Sources: [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1) · [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Soniqo M5 Pro benchmark](https://soniqo.audio/benchmarks) · [sherpa-onnx Qwen3-ASR docs](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/index.html)

---

### 2. Fix ASR execution: the CoreML EP is leaving ~15x on the table — biggest latency win

**What.** The app measures **7.8x RTFx** and treats that as "CoreML engaged" (ADR-0015's floor is 2x). Two independent sources show the *same Parakeet v3* running **110–146x real-time on Apple Silicon** when executed as a native CoreML model on the ANE: [FluidAudio's parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) reports ~110x RTF on M4 Pro (145.8x overall on LibriSpeech), and Soniqo measured **117x on an M5 Pro**. 7.8x is a CPU-class number — the likely cause is that ONNX Runtime's CoreML EP can't map the **int8 QDQ** conformer ops to ANE (ANE wants fp16), so most of the graph silently falls back to CPU despite the EP being linked. ADR-0015's RTFx floor detects *total* fallback, not *partial* placement.

**Impact (quantified).** 5 s utterance: encoder+decode 640 ms → ~50 ms. End-of-speech → text drops from ~840 ms to **~250 ms p50** (VAD hangover + paste dominate) — Wispr-Flow-class felt latency, from ADR-0007's original retired target. Also large energy savings (ANE vs 10 P-cores).

**Two-stage plan.**
1. **Hours-level experiment first:** run the **fp16/fp32 (non-int8) Parakeet v3 ONNX export** through the existing stack with `provider="coreml"` and compare RTFx. If ANE placement is the blocker, this alone may yield a several-fold speedup for a 1-line model-URL change (model grows 640 MB → ~1.2 GB fp16). Also dump ORT's node-placement log to confirm the diagnosis.
2. **If ONNX/CoreML EP still underperforms:** drive FluidAudio's CoreML `.mlpackage`s (encoder/decoder/joint) directly from Rust via the `objc2-core-ml` framework crate, reimplementing the TDT greedy loop (~few hundred lines; the tri-model split matches the current encoder/decoder/joiner structure). Alternative: a small Swift shim linking FluidAudio behind a C ABI — same pattern as the AFM idea already tracked in ADR-0018.

**Effort.** Stage 1: hours. Stage 2: 1–2 weeks.

**Interplay with #1:** these compete for the same slot. If #1's A/B shows Qwen3-ASR-0.6B int8 accuracy ≥ Parakeet at acceptable RTFx, accuracy wins (latency is already in budget). If you want both, FluidAudio also publishes CoreML Parakeet variants, and Qwen3-ASR runs 36x on MLX — but pick one primary path; don't maintain two inference stacks.

---

### 3. Unbounded dictation: VAD-segmented incremental decode — biggest capability unlock

**What.** Today every dictation hard-stops at **30 s** (`vad.rs MAX_SPEECH_S`, `streamer.rs MANUAL_MAX_RECORDING`), and the polish path independently breaks around ~45 s (768-token output cap → error → raw-paste fallback; 2048 ctx ceiling). Long-form dictation — the thing people actually do in documents — is structurally capped.

Change the session to emit **one `Outcome::Speech` per completed VAD segment** while capture continues: decode segment N (at 7.8–117x RT, per #2) and polish/paste it while the user speaks segment N+1. This is *not* the "chunked pseudo-streaming" ADR-0009 rejected — segments are disjoint (no overlapping-context recompute; total compute is linear), and Silero already produces the boundaries; `drain_segments()` currently throws them away.

**Impact.** Removes the 30 s wall entirely; perceived latency for long dictations improves dramatically (text lands paragraph-by-paragraph instead of after a long wait); each polish call stays comfortably inside the 2048-token ctx; the 768-token truncation failure mode disappears for segmented input.

**Effort.** Medium — `streamer.rs` refactor (multi-outcome sessions), `app.rs` pipeline to serialize per-segment decode→polish→paste (the `polish_lock` already serializes polish), cross-segment capitalization handled by the polish pass or a joining heuristic. No new models, no new deps.

**First step.** Make `Outcome::Speech` repeatable per session, keep `finalize()` as "flush current segment and stop", and wire `app.rs` to loop on the outcome receiver.

---

### 4. Polish pass: three defensible upgrades (model choice itself is still sound)

Qwen 3.5 4B (Feb 2026) remains the strongest instruction-follower at ≤5 GB as of July 2026 — no swap-for-swap candidate surfaced that clearly beats it, so keep it. The real gains:

**4a. Context-aware polish (quality).** Feed the polish prompt (i) the user's custom dictionary and (ii) surrounding text from the focused field (the app already lives at the AX/CGEvent layer). Correct casing of names/jargon, tone continuity with the document, and correct joins at segment boundaries from #3. This is Wispr Flow's headline "context-aware" feature, locally. Effort: small–medium (AX read of focused element + prompt template extension + re-run the `bench_llm` quality set). Guard the prompt-injection surface: the transcript must stay data, not instructions — the existing system prompt already leans this way.

**4b. Apple Foundation Models (AFM-3) as a second `PolishBackend` (speed/RAM).** Post-WWDC26, the Foundation Models framework exposes Apple's third-gen on-device model to third-party apps on macOS 26. Zero download (vs 3.5 GB GGUF), ANE execution (vs 4 GB resident Metal weights + KV), likely faster than 43 tok/s. Already sketched in ADR-0018 as a wildcard; AFM-3 makes it real. Quality vs Qwen3.5-4B: **UNVERIFIED — bench first** with the existing harness via a throwaway Swift CLI before writing the `@objc` shim. Effort: medium (Swift shim dylib behind the existing `PolishBackend` trait). Kill it if the bench shows instruction-following regressions on `scratch that` / filler-word cases.

**4c. Speculative decoding for the 4B (latency).** llama.cpp supports draft-model speculative decoding; a Qwen3.5-0.6B draft typically yields 1.5–2x decode speedup on this workload class → polish p50 ~1225 ms → ~700–800 ms wall clock, which matters for the final-flush (the truncation-check path) and short dictations where TTFT dominates perception less. Risk: `llama-cpp-2` may not expose the speculative API (the eugenehp fork tracks it; upstream binding unverified) — timebox a spike; if the binding fight exceeds a day, drop it, since streaming paste already hides most generation latency.

---

## Rust-ecosystem notes (no action forced, one warning)

- **`ort` is still 2.0.0-rc (rc.12)** — going direct-to-ONNX-Runtime remains wrong for this app; sherpa-onnx (crate now 1.13.4, actively released) stays the right call. Candle still has no Parakeet/conformer-transducer implementation.
- **Name collision:** a crate literally named **`parakeet-rs` (0.3.2) exists on crates.io** — someone else's Parakeet-via-`ort` library. If this app is ever published to crates.io or promoted publicly, the name is taken; rename or namespace before any release.
- Pure-Rust Qwen3-ASR implementations exist (`qwen-asr` CPU-only, `qwen3-asr-rs` libtorch/MLX) — useful reference code for #1, not production paths.

## Killed ideas (evaluated, not worth it)

- **Cohere Transcribe 03-2026** (2B, Apache 2.0, 5.42 avg leaderboard WER, sherpa-onnx export exists): dominated by Qwen3-ASR-1.7B — worse English WER than 1.7B, 2.4 GB int8 encoder, AED decoder no faster, 14 langs vs 52, no hotword story.
- **Granite Speech 4.1 2B** (Apache 2.0, leaderboard #1 at 5.33 avg, 1.33 clean): no ONNX/sherpa-onnx path; the llama.cpp `mtmd` route exists upstream but `llama-cpp-2` doesn't expose mtmd — high integration cost for accuracy roughly at Qwen3-ASR-1.7B level.
- **Canary-Qwen 2.5B** (5.1 avg): still CC-BY-NC — unusable in a distributed app.
- **Streaming zipformer/FastConformer swaps**: ADR-0009's analysis still holds (WER regression, no punctuation); Qwen3-ASR's unified streaming mode (#1) is the eventual streaming path instead.
- **Beam search / TDT decoding tweaks on Parakeet**: sherpa-onnx offline TDT is greedy-only in practice and TDT gains from beam are marginal at this WER level; superseded by #1.
- **KV-cache reuse of the fixed polish system prompt**: TTFT is already 29 ms — noise.

## Sources

- [Soniqo — Qwen3-ASR vs Parakeet vs Whisper on M5 Pro](https://soniqo.audio/benchmarks)
- [Qwen3-ASR Technical Report (arXiv 2601.21337)](https://arxiv.org/html/2601.21337v1) · [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [sherpa-onnx Qwen3-ASR docs](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/index.html) · [sherpa-onnx crate 1.13.4 on docs.rs](https://docs.rs/sherpa-onnx/latest/sherpa_onnx/) · [k2-fsa/sherpa-onnx#3535 (1.7B export request)](https://github.com/k2-fsa/sherpa-onnx/issues/3535)
- [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) · [FluidAudio benchmarks](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md)
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (no newer Parakeet since v3, Aug 2025)
- [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) · [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
- [Apple — Third-generation Foundation Models](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models) · [9to5Mac WWDC26 explainer](https://9to5mac.com/2026/06/11/apples-new-foundation-models-explained-on-device-ai-cloud-ai-and-everything-in-between/)
- [llama.cpp speculative decoding docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) · [ort releases (2.0 still rc)](https://github.com/pykeio/ort/releases) · [parakeet-rs crate name collision](https://crates.io/crates/parakeet-rs/0.3.2)
