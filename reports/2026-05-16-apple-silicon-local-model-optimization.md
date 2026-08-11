# Apple Silicon Local Model Optimization: Tricks of the Trade

> Research snapshot last updated May 16, 2026. Techniques and upstream details may evolve; validate them against the target hardware and current dependencies before implementation. See the project [README](../README.md) and [ADR](../docs/ADR.md) for the shipped architecture.

This is a low-level optimizer playbook for running local models on Apple Silicon, inspired by antirez/ds4, llama.cpp/GGML/Metal practice, Apple’s Metal guidance, and first-principles reasoning about Apple’s unified-memory SoCs.

Short thesis: the biggest wins on Apple Silicon are not “use Metal” or “convert to MLX.” They come from treating the model, quantization, KV cache, command graph, activation hooks, and hardware memory system as one co-designed artifact.

The boring runner mindset is:

```text
model weights -> generic runtime -> generic kernels -> hope
```

The ds4-style optimizer mindset is:

```text
target model architecture
+ target quant layout
+ target memory topology
+ target inference pattern
+ target validation suite
= custom engine
```

## Sources checked

- antirez/ds4 README, GGUF tooling, imatrix docs, quality testing docs, speed-bench docs, and `ds4_metal.m`
- DeepWiki index of `antirez/ds4`, checked against actual repository pages and symbols
- llama.cpp/GGML Metal and optimization guidance
- Apple Metal overview and Metal Best Practices Guide
- Apple WWDC “Bring your machine learning and AI models to Apple silicon”
- MLX project docs/README for Apple Silicon array/runtime assumptions
- Sean Goedecke, “DeepSeek-V4-Flash means LLM steering is interesting again” (https://www.seangoedecke.com/steering-vectors/), for the local-model steering-vector framing

## 1. Apple Silicon first principles

Apple Silicon is not a small NVIDIA GPU glued to a normal PC.

Important properties:

1. Unified memory is the defining feature.
   CPU and GPU share one memory pool. This avoids explicit PCIe copies and lets very large models fit if system RAM is large enough.

   But “unified” does not mean “free.” GPU reads still consume memory bandwidth. CPU page faults, virtual-memory behavior, residency, and cache locality still matter.

2. Memory bandwidth is often the limiter.
   Autoregressive decode is usually memory-bandwidth dominated, especially for large quantized models. Each generated token touches a large fraction of model weights, but does not do enough arithmetic to fully occupy the GPU like a big batched GEMM would.

   Rule of thumb:
   - prefill is more compute/GEMM friendly
   - decode is more memory-streaming / latency sensitive

3. Apple GPUs like regular, fused, wide work.
   They do not like:
   - tiny kernels
   - excessive command buffer boundaries
   - CPU/GPU synchronization
   - scattered irregular memory reads
   - generic dispatch overhead
   - CPU intervention between layers

4. The CPU is still important.
   Tokenization, sampling, prompt rendering, routing metadata, I/O, mmap, state management, and server overhead can destroy latency if the GPU is otherwise fast.

5. The Neural Engine is usually not the answer for hobbyist LLM inference.
   It is powerful, but not a general low-level programmable target like Metal. For custom local model hacking, Metal/GPU + CPU is the practical path.

## 2. The most important ds4 lesson: specialize narrowly

ds4 / DwarfStar 4 is deliberately not a general GGUF runner. It targets DeepSeek V4 Flash specifically, with fixed model-shape assumptions such as `DS4_N_LAYER = 43`, `DS4_N_EMBD = 4096`, `DS4_N_VOCAB = 129280`, `DS4_N_EXPERT = 256`, and `DS4_N_EXPERT_USED = 6`. That narrowness is the point: arbitrary DeepSeek or GGUF files do not automatically have the tensor layout, metadata, quantization mix, and optional MTP state expected by the engine.

Generic runtimes carry abstraction cost:

- dynamic tensor graphs
- many model architectures
- generic tensor layouts
- many quant formats
- generic scheduling
- fallback paths
- shape polymorphism
- repeated dispatch decisions

A specialized runtime can bake in:

- exact layer count
- exact tensor names
- exact expert count
- exact hidden sizes
- exact top-k routing behavior
- exact quantization mix
- exact KV format
- exact prompt template
- exact tool-call format
- exact validation vectors

This lets you replace a graph interpreter with a hand-planned execution path.

Trick: if you want frontier local performance, pick one important model family and make it feel “finished,” not merely “supported.”

That means:

- custom GGUF creation
- custom quant recipe
- custom Metal kernels
- custom KV cache policy
- custom validation
- custom server/prompt integration

The engine and model artifact should be co-designed. In ds4 this includes the downloadable `q2-imatrix` and `q4-imatrix` artifacts, the optional MTP draft model, model-specific prompt/tool formatting, and disk-capable KV state.

## 3. Mixed quantization beats uniform quantization

A naive optimizer asks:

```text
Can I quantize the whole model to 4-bit or 2-bit?
```

A serious optimizer asks:

```text
Which tensors are quality-critical, which tensors dominate size, which tensors dominate bandwidth, and which tensors can survive brutal quantization?
```

ds4’s DeepSeek V4 Flash recipe is a great example:

- only routed MoE experts are heavily quantized
- routed experts dominate model size
- shared experts, projections, routing, and output-related components are kept at higher precision
- for ds4’s q2 path, routed MoE gate/up are `IQ2_XXS` and routed MoE down is `Q2_K`
- router/shared experts/output stay higher precision in ds4’s checked q2 path, e.g. `F16` or `Q8_0`

Why this works:

MoE models have many total parameters but only a subset active per token. The routed experts consume enormous storage. Quantizing those gives huge memory savings. But router/projection/output errors can globally corrupt every token, so keep those cleaner.

General trick: classify tensors into buckets.

A. Very quality-sensitive:
- token embeddings
- output head / lm_head
- attention projections in some models
- router/gating logits
- normalization weights
- small but globally used tensors

B. Huge and somewhat redundant:
- FFN up/down/gate matrices
- MoE expert matrices
- some MLP blocks

C. Bandwidth-critical during decode:
- whatever is streamed every token

D. Context-size critical:
- KV cache

Then quantize each bucket differently. Do not blindly use one global quant type.

## 4. Importance-aware quantization is real

The ds4 imatrix pipeline is one of the most important ideas.

Instead of quantizing weights only by static weight magnitude, ds4 uses activation-importance data from the actual inference graph. Its imatrix tooling lives under `gguf-tools/imatrix/README.md`, and the runtime path lines up with the core idea: routed gate/up quantization is driven by FFN-normalized inputs, while routed down quantization is driven by the post-routing SwiGLU rows that feed the down projection.

The principle: a weight column or row that is frequently multiplied by large activations matters more than one that is rarely active or usually near zero.

So the quantizer should minimize activation-weighted error, not just raw-weight error. If you are reimplementing ds4’s exact imatrix file format or packing, verify the details directly in `gguf-tools/imatrix/README.md` rather than inferring them from the runtime alone.

This especially matters for ultra-low-bit quantization like 2-bit.

General trick: build a calibration corpus that resembles your real use:

- coding prompts if optimizing for coding agents
- tool-call prompts if you need tool use
- long-context prompts if you care about long context
- multilingual prompts if relevant
- thinking/non-thinking variants if the model has those modes

Then collect activation stats through your actual runtime path, not an abstract reference path.

Quantization should be downstream of runtime behavior.

The same “activations are data, not magic” mindset also opens up local model steering. Steering vectors directly modify activations during inference: collect paired activations for prompts with and without a target behavior such as “respond tersely,” subtract the baseline activations to get a direction, then add that direction back at the same layer during later generations. More sophisticated versions can learn sparse-autoencoder features and boost those features instead of using raw prompt-pair differences.

This is only practical when you control the runtime. API models hide weights and activations; a local Apple Silicon engine can expose layer hooks, capture calibration activations, and inject steering deltas in the hot path. That makes steering a sibling to imatrix quantization: both depend on measuring the model’s internal activations under the actual inference path, then feeding those measurements back into the runtime.

Optimization caveat: activation steering is not free. Injecting vectors can add memory reads/writes, extra command boundaries, CPU-GPU synchronization, and validation burden. If steering is a product feature, design it like any other runtime feature: fixed hook points, preloaded steering vectors in GPU-friendly layout, no decode-time allocation, no CPU readback per token, and quality tests that prove the steering effect survives quantization and kernel fusion.

## 5. Validate against logits, not vibes

A common local inference failure mode: the model “seems to answer,” but a kernel, tokenizer, RoPE, KV cache, or quantization detail is subtly wrong.

ds4 uses official continuations, official logprob vectors, and local negative log-likelihood scoring. In the repo this shows up as `tests/ds4_test.c --logprob-vectors`, `tests/test-vectors/official.vec`, and `gguf-tools/quality-testing/score_official.c`. That is the right idea.

Good validation levels:

1. Tensor/kernel numerical tests: does each Metal kernel match a CPU/reference implementation?
2. Logprob vector tests: given known prompt, do local top-token logprobs match official/reference vectors?
3. Token probability tests: given official continuation, how much probability does the local model assign token by token?
4. First-token match: does the greedy first token match reference?
5. Greedy longest common prefix: how long does local greedy decode agree with official/reference decode?
6. Behavioral tests: tool calls, JSON, code patches, long-context retrieval.
7. Steering tests if activation hooks are enabled: does a known vector move behavior/logits in the expected direction without corrupting unrelated capabilities?
8. Speed regression tests: same model, same prompt, same context frontiers, same thermal state.

Do not judge quantization from a few sampled chats. Sampling noise hides regressions.

## 6. Treat KV cache as a first-class artifact

For long-context inference, KV cache can dominate memory.

ds4’s README makes a strong point: with compressed KV caches and fast Apple SSDs, KV should not automatically be considered RAM-only. It treats KV as a first-class disk citizen; the server can use SHA1-rendered-prefix keyed `.kv` files through options such as `--kv-disk-dir` and `--kv-disk-space-mb`.

This is important on Apple Silicon because:

- unified memory is large but finite
- SSDs are fast
- local agents often revisit similar long contexts
- long-context sessions may benefit from persistence
- compressed KV formats can make huge contexts practical

Optimization directions:

- quantize KV cache
- compress KV cache asymmetrically
- persist KV cache to disk
- mmap KV cache
- reuse prefix states
- store session states
- separate hot recent KV from cold older KV
- benchmark context frontiers, not just whole-run averages

KV policy should be model-specific:

- Some architectures tolerate KV quantization well.
- Others are sensitive.
- Some use MLA/compressed attention structures that change the tradeoff entirely.

Aspirational trick: think of long-context inference as a database/cache problem, not just a transformer problem.

## 7. mmap the model, but understand the cost

ds4 wraps mmap-backed model views as Metal buffers with `newBufferWithBytesNoCopy`. That avoids copying huge model weights.

Benefits:

- faster startup
- lower peak memory
- OS can page model data
- large models can be represented without explicit full copy
- plays well with unified memory

Pitfalls:

- page alignment matters
- first-touch can be expensive
- page faults during inference can destroy latency
- virtual-memory behavior can be surprising
- very large mmap + CPU access paths can hit OS bugs
- many tiny mapped ranges are bad

ds4 has logic around:

- page-aligned model views
- maximum Metal buffer sizes / device `maxBufferLength`
- stable small number of page-aligned views
- warming model-backed pages, including a `kernel_touch_u8_stride` path
- avoiding splitting tensors across command encoders

General tricks:

- Use mmap for huge static weights.
- Align views to page boundaries.
- Group tensor ranges to reduce buffer/view count.
- Warm pages before timing.
- Avoid random first-touch during measured decode.
- Keep mapped weight views stable for the process lifetime.
- Know Metal buffer size limits.
- Measure cold-start and warm-start separately.

## 8. Minimize command buffer and encoder overhead

Apple’s Metal Best Practices says to submit the fewest command buffers possible without starving the GPU. ds4 reflects this with command batching machinery such as `g_batch_cb`, `g_batch_enc`, `g_pending_cbs`, `ds4_gpu_begin_commands()`, `ds4_gpu_flush_commands()`, and `ds4_gpu_synchronize()`.

For model inference, this means:

- do not commit/wait after every tiny op
- batch layer operations where possible
- avoid CPU-GPU sync points
- keep command encoders open when possible
- prebuild/cache pipeline states
- reuse buffers
- avoid allocating during decode

Bad pattern:

```text
dispatch kernel
wait
dispatch kernel
wait
read tiny value on CPU
dispatch next kernel
```

Good pattern:

```text
encode a whole useful graph section
commit
let GPU run
only synchronize at necessary boundaries
```

The CPU should orchestrate; it should not be in the critical path every layer.

## 9. Fuse kernels aggressively, but selectively

Fusing is one of the biggest low-level wins.

Why:

- fewer memory round trips
- fewer intermediate buffers
- fewer command dispatches
- more arithmetic per byte loaded
- less CPU overhead
- better cache locality

ds4 has fused paths for things like:

- shared gate/up + SwiGLU paths
- routed gate/up pair processing
- down + sum over experts, e.g. the Q2_K expert summation path `kernel_mul_mv_id_q2_K_sum6_f32`
- HC split / weighted-sum / norm paths
- HC expand paths
- shared-down HC expansion
- specialized attention paths

General fusion targets:

- RMSNorm + projection setup
- Q/K/V projection preparation
- RoPE application
- dequant + matvec
- matmul + bias
- gate/up + activation
- SwiGLU multiply
- expert weighted sum
- residual add + norm
- softmax pieces where shape is fixed
- output projection + logits postprocessing if practical

But do not fuse blindly. Fused kernels can reduce occupancy or increase register pressure. Always benchmark.

## 10. Separate prefill and decode paths

Prefill and decode are different workloads.

Prefill:

- many tokens at once
- GEMM-like
- benefits from batching
- more compute-dense
- GPU-friendly

Decode:

- one/few tokens at a time
- GEMV/matvec-like
- memory bandwidth dominated
- latency-sensitive
- dispatch overhead matters more

Therefore, use separate kernels and scheduling.

For prefill:

- larger tiles
- batched matmul
- process multiple tokens
- exploit parallelism over sequence

For decode:

- specialize for batch=1 or small batch
- fuse matvec/dequant/activation
- minimize memory loads
- minimize command count
- optimize KV access
- avoid generic GEMM kernels if they are bad for skinny shapes

There are multiple inference regimes; optimize each separately.

## 11. Exploit model architecture

The best low-level optimizations come from model-specific structure.

For MoE:

- only load/use selected experts
- specialize for top-k count
- group selected expert work
- use expert-major layouts for prefill
- use token-major or selected-expert layouts for decode depending shape
- fuse expert summation
- keep router high precision
- quantize routed experts harder than shared path

For GQA/MQA:

- KV cache is smaller than full MHA
- optimize head grouping
- avoid duplicated K/V loads

For MLA or compressed attention:

- exploit compressed latent KV
- do not emulate classic KV if architecture has cheaper native form

For sliding-window/local attention:

- avoid attending to unavailable tokens
- keep ring buffers

For long context:

- use chunked prefill
- reuse prefix states
- benchmark frontiers: 2k, 4k, 8k, 32k, 128k, etc.

For speculative decoding:

- only worth it if draft acceptance is high and verification is cheap
- ds4 notes its MTP/speculative path is experimental, correctness-gated, and currently only a slight speedup rather than a meaningful generation-speed win
- do not assume speculation helps; measure it

## 12. Design the GGUF/layout for the engine, not vice versa

Generic GGUFs are convenient. Serious optimization may require a custom GGUF.

ds4 explicitly says arbitrary DeepSeek/GGUF files will not work because tensor layout, quantization mix, metadata, and optional MTP state are expected by the engine.

That is a feature, not a bug, for maximum performance.

Layout questions:

- Are tensors stored in the order decode needs them?
- Are expert tensors contiguous?
- Are offsets page-aligned?
- Are quant blocks aligned to GPU access patterns?
- Are small critical tensors grouped?
- Can one Metal buffer view cover many consecutive tensors?
- Are tensor names mapped cleanly to custom kernels?
- Does layout avoid runtime reshaping/transposition?

Weight format is part of the runtime.

## 13. Avoid dequantizing to fp16/fp32 as a separate step

A common mistake:

```text
quantized weights -> dequantize into fp16 buffer -> matmul
```

Better:

```text
quantized weights -> dequantize inside matvec/matmul kernel -> accumulate
```

Why:

- separate dequant writes huge intermediate data
- doubles memory bandwidth pressure
- increases latency
- creates extra kernels
- destroys the point of quantization

The kernel should understand the quant format directly.

For Apple Silicon, the key is not only lower storage size; it is lower bytes read per token.

## 14. Pick quant block sizes for GPU access, not just compression

Quantization format is a hardware interface.

Consider:

- block size
- scale layout
- zero/min layout
- lookup table usage
- alignment
- vectorized loads
- threadgroup memory usage
- whether one SIMD group can consume a block cleanly
- whether rows are multiples of 64/128/256
- whether dequant math is branchless
- whether scales are reused efficiently

Some ds4 Metal paths require expert dimensions to be multiples of 256. That is the kind of shape-specific constraint that unlocks simpler kernels.

General principle: if the quant format makes the kernel ugly, it may lose despite smaller bits/weight.

## 15. Use higher precision where errors amplify

Some tensors or operations amplify small errors:

- router logits
- final output head
- attention score calculation
- normalization
- low-rank/compression/indexer components
- embeddings
- early layers in some models
- layers near output in some models

For those:

- keep F16/BF16/Q8
- avoid ultra-low-bit
- test with NLL/logprob-vector metrics

For bulk MLP/expert weights:

- Q4, Q3, IQ2, etc. may be acceptable
- especially with imatrix

This is why mixed quantization usually beats uniform quantization at the same file size.

## 16. Optimize for bandwidth per generated token

A useful mental model:

```text
decode speed ~= memory_bandwidth / bytes_streamed_per_token
```

This is simplified, but good enough to guide work.

To improve decode:

- reduce bytes per weight via quantization
- avoid reading unused experts
- avoid dequant intermediates
- keep small tensors resident/cache-friendly
- fuse ops to avoid writing intermediates
- compress KV
- reduce KV reads with architectural tricks
- avoid CPU sync
- avoid page faults
- batch concurrent users only if latency target allows it

When comparing models, active parameters matter more than total parameters for MoE decode.

This is why a huge MoE can feel surprisingly fast if active params and KV are small.

## 17. Prefill wants batching; decode wants latency

For local chat, users feel decode speed. For RAG/agents/long prompts, prefill also matters.

Optimize both separately.

Prefill tricks:

- chunk long prompts
- use large enough batch/ubatch
- use GEMM-oriented kernels
- keep GPU saturated
- reduce graph overhead
- use prefix caching
- avoid repeated prompt rendering/tokenization
- use incremental prefill benchmarks

Decode tricks:

- batch=1 specialized kernels
- matvec kernels
- persistent/reused buffers
- small number of command buffers
- no CPU readbacks
- fused dequant/activation
- KV cache locality
- minimal sampling overhead

## 18. Sampling can matter

When model generation gets fast, sampler overhead becomes visible.

Low-level sampler optimizations:

- keep logits on GPU if doing heavy filtering
- or copy only needed logits/top-k to CPU
- avoid full vocab sort
- use partial top-k
- avoid repeated allocations
- precompute repetition penalty structures
- optimize tokenizer detokenization
- stream efficiently

For coding agents:

- greedy or low-temperature generation is common
- constrained tool-call formats can reduce sampling complexity
- grammar constraints can add CPU overhead if implemented naively

## 19. Memory allocation discipline

Do not allocate in the hot path.

Preallocate:

- KV cache
- attention scratch
- router buffers
- expert scratch
- logits
- temporary reductions
- command resources
- pipeline states
- token buffers

Runtime allocation during decode is poison:

- allocator overhead
- memory pressure
- unpredictable latency
- page faults
- synchronization

## 20. Pipeline caching matters

Metal pipeline creation is expensive. Build/cache pipeline states before hot inference.

ds4 has pipeline caches and many statically referenced hot pipelines.

Tricks:

- compile Metal libraries at build time where possible
- cache `MTLComputePipelineState` objects
- use function constants for shape-specialized variants
- avoid creating pipeline states in decode
- provide environment flags to disable/compare hot specialized paths
- warm up kernels before benchmarking

## 21. Use Apple’s resource model intentionally

Apple’s Metal Best Practices:

- choose storage modes carefully
- use few command buffers
- reduce CPU overhead
- manage resources effectively
- keep CPU/GPU parallel

For Apple Silicon-era Macs, practical guidance:

- static weights: mmap/no-copy shared buffers can work well for huge models
- GPU-only scratch: consider private where appropriate, but shared is often used for CPU-filled or mmap-backed buffers
- CPU-updated small data: shared
- avoid needless CPU access to GPU-written buffers
- avoid synchronizing unless needed
- avoid copying huge static weights if mmap is viable

The right choice may differ between:

- Mac integrated/unified memory
- older discrete-GPU Macs
- iOS/tvOS unified memory
- Metal API storage-mode semantics

Measure on the target hardware.

## 22. Benchmark correctly

Bad benchmark:

```text
I got 45 tok/s once.
```

Good benchmark includes:

- model file hash
- quant type
- exact commit
- machine
- macOS version
- power mode
- thermal state
- prompt length
- context frontier
- prefill t/s
- decode t/s
- batch size
- tokens generated
- first run vs warm run
- memory pressure
- whether model pages were warm
- whether app was foreground/background

ds4’s `ds4-bench` method is good:

- load once
- walk a fixed token sequence to context frontiers
- measure incremental prefill at each frontier
- save/restore KV state after each frontier
- generate a fixed greedy non-EOS probe
- report prefill and generation separately

For long context, speed is a curve, not a scalar.

## 23. Use quality-speed Pareto curves

Every optimization should be plotted on:

- size
- RAM use
- prefill t/s
- decode t/s
- NLL/logprob-vector deviation
- behavioral pass/fail
- long-context pass/fail

A 2-bit model that is fast but loses tool-call reliability may be useless. A Q4 model that is 10% slower but preserves coding quality may be better. A model with better active-parameter structure may beat a smaller dense model.

Do not optimize speed alone.

## 24. Architecture-specific cheats are fair game

Examples:

- DeepSeek V4 Flash has compressed KV and MoE sparsity
- MLA-style models can have radically different KV economics
- top-k experts can be hardcoded
- fixed number of layers can be unrolled in scheduling
- fixed hidden sizes can use specialized kernels
- exact prompt templates can be compiled into server logic
- exact tool-call syntax can be validated/tested
- context windows can be managed with model-specific cache logic

Generic elegance loses to architecture-aware ugliness if your goal is local performance.

## 25. Disk is part of the memory hierarchy

Especially on modern Macs:

- SSD is fast
- mmap is powerful
- VM subsystem can be leveraged
- persistent KV can save repeated prefill
- local agents reuse contexts

But disk is not RAM:

- random page faults hurt
- first-token latency can spike
- OS pressure matters
- kernel/VM bugs can exist
- throughput differs from latency

Use disk for:

- model mmap
- cold KV
- persisted sessions
- prefix cache
- quant artifacts

Do not let disk page faults surprise the decode loop.

## 26. Optimize end-to-end agent usage, not just kernels

ds4’s README emphasizes server API, CLI, prompt rendering, tool calling, KV state handling, traces, and coding-agent integration.

That is important. A model runner can be fast and still bad for agents.

Agent-relevant optimizations:

- prompt rendering must exactly match model expectations
- tool-call syntax should be robust
- server streaming must be low overhead
- session/KV reuse should be easy
- traces should capture failures
- long-context retrieval should be tested
- state save/restore should be reliable
- startup should be fast enough for workflows
- optional activation steering should be a first-class runtime path, not a debug hack: capture hooks, injection hooks, vector storage, and per-vector validation should be explicit

The actual product is not tokens/sec. It is useful local work/sec.

## 27. Think in rooflines

For each kernel, ask:

1. How many bytes does it read/write?
2. How many FLOPs/integer ops does it do?
3. Is it bandwidth-bound or compute-bound?
4. Is occupancy limited by registers/threadgroup memory?
5. Is launch overhead dominant?
6. Is it reading contiguous memory?
7. Can it fuse with neighbor ops?
8. Can it avoid writing an intermediate?
9. Can it exploit fixed shape?
10. Can it reduce precision safely?

If arithmetic intensity is low, quantization and fusion matter most. If arithmetic intensity is high, tiling and occupancy matter more. If launch overhead dominates, fuse/batch. If page faults dominate, warm/pin/restructure memory.

## 28. Practical priority order

If building an Apple Silicon local optimizer from scratch:

1. Pick one model architecture. Do not start generic.
2. Build correctness harness first: reference logits, tokenization, NLL scoring, kernel tests.
3. Make a simple Metal path. Correct before fast.
4. Separate prefill and decode. Benchmark both.
5. mmap weights. Avoid huge copies; handle page alignment.
6. Implement direct quantized matvec/matmul. No separate dequant buffer.
7. Add mixed quantization. Keep sensitive tensors high precision.
8. Add imatrix/activation-aware quantization, especially for 2-3 bit.
9. Fuse decode kernels: norm/projection/activation/expert sum/residual where possible.
10. Optimize KV: quantize/compress/persist/reuse.
11. Reduce command buffer count and CPU sync. Batch graph sections.
12. Preallocate all scratch. No hot-path allocation.
13. Add long-context state management: prefix cache, session save/restore, disk KV.
14. If you want controllable local behavior, add activation-steering hooks: capture paired activations, store steering vectors in a GPU-friendly format, inject without CPU readback, and validate both behavioral effect and logit drift.
15. Build speed regression suite: context frontier benchmarks.
16. Integrate with real agent workloads: tool calls, coding edits, long prompts.

This order avoids the trap of optimizing a wrong implementation.

## 29. What not to waste time on early

Avoid early obsession with:

- generic framework integration
- supporting many architectures
- perfect server API
- fancy UI
- dozens of quant formats
- speculative decoding before baseline is excellent
- micro-optimizing tokenizer before GPU path is correct
- giant benchmark spreadsheets without quality metrics
- one-off chat vibes as evaluation

First get:

```text
correct logits
stable decode
known quality
known speed
one architecture
```

## 30. antirez/ds4’s most transferable tricks

From ds4 specifically, the big transferable ideas are:

1. Narrow engine for one model, not generic runtime.
2. Custom GGUFs matched to the engine.
3. Aggressive mixed quantization.
4. Ultra-low-bit only where the model can tolerate it.
5. imatrix activation-aware quantization for routed experts.
6. Official-continuation NLL scoring.
7. Logprob-vector regression tests.
8. Long-context tests.
9. Metal-specific graph path.
10. mmap-backed model views.
11. Page-aligned model wrapping.
12. Warm model-backed pages.
13. Cached Metal pipelines.
14. Preallocated scratch buffers.
15. Fused kernels.
16. Separate prefill/decode benchmarking.
17. KV cache as RAM/disk state, not an afterthought.
18. End-to-end agent integration as a performance target.
19. Activation steering as a local-only control surface when you can afford the hook and validation costs.

## 31. Apple Silicon-specific caveats

1. The best settings differ by chip.
   M1/M2/M3/M4, Max/Ultra, RAM size, memory bandwidth, GPU cores, and thermal envelope all matter.

2. MacBook vs Mac Studio matters.
   Sustained thermals can change results a lot.

3. Unified memory can encourage overcommit.
   Running too close to memory capacity may trigger paging and destroy latency.

4. macOS VM behavior matters.
   mmap is powerful but can have surprising first-touch and paging effects.

5. Metal profiling is mandatory.
   Use Instruments / Metal System Trace / Xcode GPU tools. Guessing is not enough.

6. “Fits in RAM” is not the same as “fast.”
   A 180GB model on a 192GB machine may run, but memory pressure can make it miserable.

7. Generic ML frameworks may hide the problem.
   MLX/Core ML are excellent for many uses, but if the goal is ds4-level optimization, inspect kernels, memory movement, synchronization, and layout.

## 32. Mental model for choosing models for local Apple Silicon

Prefer models with:

- low active parameters
- efficient KV cache
- GQA/MQA/MLA or other KV-saving attention
- good low-bit quant tolerance
- known prompt format
- stable tokenizer
- architecture simple enough to specialize
- strong quality at small active compute
- long-context behavior that can be tested

Be skeptical of:

- huge dense models
- giant KV footprint
- models that collapse under quantization
- architectures requiring unsupported exotic ops
- models with poor tool-use format
- models where quality depends on very long hidden reasoning tokens

## 33. The core design equation

For local Apple Silicon inference:

```text
Useful speed =
  model quality
  × quantization tolerance
  × active parameter efficiency
  × KV efficiency
  × memory bandwidth efficiency
  × kernel fusion
  × low synchronization
  × controllable activation hooks when steering is needed
  × correct prompt/tool integration
```

If any term is near zero, the system feels bad.

## Bottom line

The trick of the trade is co-design.

The model file, quantizer, calibration set, Metal kernels, memory map, KV cache, benchmark harness, prompt template, and server should all know about each other.

If you also want steering, add activation traces and steering-vector injection to that co-design loop. Directly modifying activations is powerful precisely because it is below prompting and above retraining; on a local model runner, it becomes another optimized runtime path that has to be measured, fused around, and regression-tested.

That is why ds4 is interesting: it is not just a faster runner. It is a complete, narrow inference product for one model family.

If you want to become good at this, do not start by writing a generic ONNX/MLX wrapper. Start by taking one model architecture and making it:

- bit-exact enough to trust
- quantized with architecture-aware mixed precision
- activation-calibrated
- mmap/page-aware
- Metal-kernel-aware
- KV-cache-aware
- benchmarked at context frontiers
- validated against reference continuations
- usable by real agents
