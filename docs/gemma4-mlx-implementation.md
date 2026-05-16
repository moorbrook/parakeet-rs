# Implementing a `gemma4-mlx` crate on OminiX-MLX

> **Status: DISQUALIFIED for parakeet-rs cleanup tier.** Gemma 4 E2B 4-bit on-disk size exceeds 2 GB, the cap parakeet-rs accepts for the first-run cleanup-model download. The candidate has been dropped from the Phase-0 benchmark in [latency-plan.md §6](./latency-plan.md).
>
> This document is preserved as a reference in case (a) a smaller Gemma 4 quant (sub-2 GB int3 / mixed-precision) becomes available, or (b) the 2 GB cap is revisited. Everything below is still accurate for someone wanting to write a `gemma4-mlx` crate on OminiX-MLX — only the recommendation that parakeet-rs should is rescinded.

Companion to [latency-plan.md](./latency-plan.md). This doc was originally the implementation map for the OminiX-MLX side of the Phase-0 cleanup-backend benchmark.

**Key fact up front:** [OminiX-MLX](https://github.com/OminiX-ai/OminiX-MLX) ships model crates for Qwen3, GLM4, Mixtral, Mistral, etc. **It does not ship a Gemma crate.** Choosing OminiX-MLX for Gemma 4 means writing a new `gemma4-mlx` workspace member from scratch, following the existing `qwen3-mlx` pattern.

## Workspace context

OminiX-MLX is a Cargo monorepo:

- `mlx-sys/` — FFI bindings to Apple's MLX C++ via bindgen
- `mlx-rs/` — safe Rust bindings on top of `mlx-sys`
- `mlx-rs-core/` — shared inference infrastructure (KV cache, RoPE, attention masks, SDPA, sampling)
- per-model crates: `qwen3-mlx`, `glm4-mlx`, `mixtral-mlx`, `mistral-mlx`, …
- `ominix-api/` — unified OpenAI-compatible HTTP server (we don't need this; we call the model crate directly from `src/cleanup.rs`)

License: MIT/Apache-2.0 dual. Contributing a new model crate back upstream is the friendly path.

## What `mlx-rs-core` gives you (don't rewrite these)

| Module | Provides |
|---|---|
| `cache` | `KeyValueCache` trait, `ConcatKeyValueCache` impl for autoregressive KV |
| `utils` | `AttentionMask`, `SdpaMask`, causal & sliding-window mask generators, RoPE init |
| `utils` (SDPA) | Scaled-dot-product attention with mask support |
| `Sampler` trait + `DefaultSampler` | Greedy / temperature / top-p sampling |
| Metal fused kernels | Fused SwiGLU (won't help Gemma — see [Activation](#mlp-activation-geglu-not-swiglu)) |

**Not provided** (you write it): tokenizer loading, safetensors weight loading, model forward pass, generation loop.

## Crate layout (mirror `qwen3-mlx`)

```
gemma4-mlx/
├── Cargo.toml
├── ominix.toml                   # registry entry — copy qwen3-mlx's, swap names
├── README.md
├── src/
│   ├── lib.rs                    # public re-exports
│   ├── config.rs                 # GemmaConfig — matches HF config.json schema
│   ├── model.rs                  # Gemma4Model: forward(), embed, decoder stack
│   ├── attention.rs              # interleaved local + global attention
│   ├── layer.rs                  # DecoderLayer: norm → attn → norm → MLP
│   ├── loader.rs                 # safetensors → mlx tensors
│   ├── tokenizer.rs              # thin wrapper around the `tokenizers` crate
│   └── generate.rs               # generation loop using core's Sampler + KVCache
└── examples/
    ├── generate_gemma4.rs        # one-shot prompt → completion
    └── chat_gemma4.rs            # interactive REPL
```

`Cargo.toml` deps:

```toml
[dependencies]
mlx-rs = { path = "../mlx-rs" }
mlx-rs-core = { path = "../mlx-rs-core" }
tokenizers = "0.20"
safetensors = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
thiserror = "2"

[dev-dependencies]
clap = { version = "4", features = ["derive"] }
```

## Gemma 4 vs Qwen3 — the seven things you must change

The high-level forward pass is the same (embed → N × (norm + attn + norm + MLP) → final norm → lm_head). Below are the places where blindly copying Qwen3 will silently break Gemma 4.

### 1. Interleaved local + global attention (biggest item)

Gemma 4 alternates **local sliding-window attention** (typically 512–1024 token window) with **global attention**, in a 5:1 ratio. Every 6th layer is global, the rest are local.

- Local layers: use `mlx-rs-core`'s sliding-window mask.
- Global layers: use the standard causal mask.
- Dispatch by layer index inside `attention.rs`.

This is the single most error-prone part of the port. Get the layer-index pattern wrong and quality silently degrades; the model still produces fluent text, just worse.

### 2. MLP activation = GeGLU, not SwiGLU

Qwen/Llama: `down(silu(gate(x)) * up(x))`
Gemma 4:    `down(gelu(gate(x)) * up(x))`

One function swap. `mlx-rs` has GELU; if not, it's a one-liner:

```
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

The fused-SwiGLU Metal kernel in `mlx-rs-core` doesn't help here — either write a fused-GeGLU kernel (later optimization) or use unfused GELU + element-wise multiply (fine for v0).

### 3. RMSNorm "+1" trick

Standard RMSNorm:
```
out = x / rms(x) * weight
```

Gemma's RMSNorm:
```
out = x / rms(x) * (weight + 1.0)
```

One line. Easy to miss. Getting it wrong yields a model that produces grammatical garbage.

### 4. Dual RoPE base

Gemma 4 uses different RoPE θ for local vs global layers — typically 10 000 for local, 1 000 000 for global. Call `mlx-rs-core`'s RoPE init twice with the two bases, store both, dispatch per-layer same as the attention mask.

### 5. Tied input/output embeddings

Gemma reuses the embedding matrix for `lm_head`. Don't allocate a separate `lm_head` weight tensor — reference the embedding. Saves memory and matches the published checkpoints.

### 6. Tokenizer = Gemma's SentencePiece variant

The HuggingFace `tokenizers` crate loads `tokenizer.json` directly. No special work needed as long as you grab the official HF Gemma 4 repo's tokenizer file. Cache it next to the weights.

### 7. Skip multimodal entirely

Gemma 4 E2B/E4B support audio + vision through additional encoders. For cleanup-only use you only need the text tower. **Do not port the vision encoder or audio frontend.** Avoids a large chunk of porting work and gigabytes of weights.

### Things the same as Qwen3 — lift wholesale

- Grouped-query attention head layout (just different head counts; check config)
- Decoder-only causal mask shape
- Generation loop (greedy / sampling / KV cache update)
- Quantization op surface

## Weights source

Gemma 4 E2B-it 4-bit MLX-format weights are on Hugging Face under [`mlx-community/gemma-4-e2b-it-4bit`](https://huggingface.co/mlx-community) (and `-e4b-`, `-26b-`, `-31b-` variants).

Loader options:

- **(a) Load MLX-format int4 weights directly.** Faster load, no quant drift. Match the loader pattern in `mistral-mlx` or `qwen3-mlx`.
- **(b) Load HF safetensors fp16/bf16 and run `mlx-rs`'s quantize op at load time.** Slower first launch, but flexible if MLX-format weights aren't available for a given size.

Start with (a).

First-launch download flow: pull from HF, cache to `~/Library/Application Support/parakeet-rs/llm/gemma-4-e2b-it-4bit/`. Same first-run pattern as Parakeet's existing model fetch in `src/model_fetch.rs`.

## Validation strategy

The hardest part of model porting is silent correctness bugs. Use Python `mlx-lm` as the oracle:

1. Pick 20 short prompts (mix of completion, code, multilingual).
2. Run through Python `mlx_lm.generate` with **greedy decoding** (`temp=0`) and fixed seed.
3. Run the same prompts through your Rust crate with greedy decoding.
4. **Token-by-token equality for at least 50 tokens of output is the bar.** Anything less and you have a bug.

Common divergence sources, in rough frequency order:

| Symptom | Likely cause |
|---|---|
| Diverges by token 3-5 | RMSNorm "+1" missing, or wrong RoPE base on global layers |
| Diverges by token 10-20 | Sliding-window mask off-by-one, wrong activation, attention scale wrong |
| Diverges by token 30+ | KV cache bug, sampling implementation differs from oracle |
| Garbage from token 0 | Wrong tokenizer, transposed weight, wrong tie-embeddings handling |

When tokens diverge, dump per-layer logits from both sides and binary-search to find the divergence layer. Tedious but deterministic.

## Implication for the latency plan

The Phase-0 benchmark in [latency-plan.md §6](./latency-plan.md) currently frames Candle vs OminiX-MLX as a symmetric head-to-head. **It isn't.**

| | Scope |
|---|---|
| **Candle** | Add `candle-core` + `candle-transformers` deps; use existing Gemma 4 module. Library-deep work. |
| **OminiX-MLX (with new `gemma4-mlx`)** | Write a new model crate from scratch (architecture port + weight loader + validation against the Python oracle). From-scratch work. |

**Recommendation:** ship Candle integration first as the working baseline for the cleanup tier. Only invest in writing `gemma4-mlx` if a measured Candle Gemma 4 E2B 4-bit polish on M5 Pro fails the **≤300 ms p50** acceptance threshold from latency-plan.md §6. If Candle hits the number, OminiX-MLX isn't worth the from-scratch port for an expected ~10–25% speed bump.

The latency-plan ADR for the cleanup backend should reflect this asymmetry: not "pick the fastest of two equivalent options" but "ship Candle; gate OminiX-MLX investment on a measured Candle miss."

## References

- [OminiX-MLX repo](https://github.com/OminiX-ai/OminiX-MLX)
- [`qwen3-mlx` (closest analog — small dense LLM)](https://github.com/OminiX-ai/OminiX-MLX/tree/main/qwen3-mlx)
- [`mistral-mlx` (alternative analog with similar loader pattern)](https://github.com/OminiX-ai/OminiX-MLX/tree/main/mistral-mlx)
- [`mlx-rs-core` (shared infra)](https://github.com/OminiX-ai/OminiX-MLX/tree/main/mlx-rs-core)
- [`mlx-community` on Hugging Face](https://huggingface.co/mlx-community) — Gemma 4 MLX weights
- [`mlx-lm` (Python oracle for token-parity validation)](https://github.com/ml-explore/mlx-lm)
- [Gemma 4 on Hugging Face blog](https://huggingface.co/blog/gemma4) — architecture overview, official numbers
- [latency-plan.md](./latency-plan.md) — parent plan; this doc is the implementation arm of §6 Phase-0 benchmark
