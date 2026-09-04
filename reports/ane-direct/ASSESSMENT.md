# Direct ANE access below Core ML, assessed for Parakeet

Research spike for kata 1wet. Machine: M5 Pro, macOS 26.5.1 (25F80).
Page citations use the printed folio of each paper, which for arXiv 2606.22283 sits 6 below the PDF page.

## Recommendation

No-go for now, with one narrow exception worth keeping open.

The path works. Every private class and C entry point the two papers describe is present on this
machine, and a hand-authored MIL program compiled, loaded and evaluated on the engine from an
ad-hoc signed binary with no entitlements and no Core ML in the process. That answers the
feasibility question the issue asked.

It does not answer the latency question in our favour. The decoder we would want to move is a
two-layer LSTM, and `lstm` has no engine path on any Apple silicon family including M5
(2606.22283, Table A.12, p.255). Core ML is already running that step off the engine, so a direct
port would not remove a Core ML ANE dispatch, it would replace a CPU LSTM with a hand-unrolled
gate graph paying a 0.07 to 0.23 ms engine dispatch floor per step. Orion measured exactly this
shape of loss on GPT-2 124M, where CPU decode at 283 tok/s beat ANE decode at 170 tok/s
(2603.06728, p.13).

Revisit only if g38m reports both of: decoder-loop time dominating the 66 ms at 5 s, and a
measured per-call Core ML overhead well above 0.25 ms. If instead the encoder or a fixed
per-utterance cost owns the time, the encoder bucket path is the thing to look at, not the decoder.

## 1. The call sequence

Two independent routes reach the engine below Core ML.

**Objective-C route (Orion).** `dlopen` on
`/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine`, then
`objc_getClass` for `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, `_ANERequest`,
`_ANEIOSurfaceObject` (2603.06728, Table 2, p.3). The sequence, confirmed by runtime
introspection on this machine and by Orion's `core/ane_runtime.m`:

1. `+[_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]`. MIL text must be
   `NSData`, not `NSString` (constraint #9), and the weights dictionary must be `@{}` rather than
   `nil` (#11).
2. `+[_ANEInMemoryModel inMemoryModelWithDescriptor:]`.
3. Stage `model.mil` and the weight blobs under `NSTemporaryDirectory()/<hexStringIdentifier>`,
   because the out-of-process compiler service reads them from disk.
4. `-compileWithQoS:options:error:` with QoS 21.
5. `-loadWithQoS:options:error:`.
6. `+[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:]`
   over `_ANEIOSurfaceObject` wrappers, then `-evaluateWithQoS:options:request:error:`.

Weights are BLOBFILE: a 128-byte container whose chunk header sits at byte 64 (sentinel
`0xDEADBEEF`, dtype 1 for fp16, payload size at 72, data offset 128 at byte 80). MIL references it
as `BLOBFILE(path=..., offset=uint64(64))`, pointing at the chunk header rather than the payload
(#8, p.5). I verified this byte-for-byte against the shipped Parakeet decoder's `weights/weight.bin`.
Tensors are fp16 in a `[1, C, 1, S]` layout over IOSurface-backed memory (p.3). Multi-input and
multi-output programs need uniform allocation sizes and alphabetically ordered names (#2, #3, #18,
#19), and evaluation needs roughly 49 KB of surface regardless of tensor size (#4).

**C route (ANEForge, and the arch paper's own listings).** The `e5rt_*` family exported from
`Espresso.framework`: `e5rt_e5_compiler_create_with_config`, `e5rt_e5_compiler_compile`,
`e5rt_program_library_retain_program_function`, `e5rt_program_function_load_for_execution`,
`e5rt_precompiled_compute_op_create_options_create_with_program_function`,
`e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options`,
`e5rt_buffer_object_alloc`, `e5rt_io_port_bind_buffer_object`, `e5rt_execution_stream_create`,
then a hot loop of `prepare_op_for_encode` / `encode_operation` / `execute_sync` / `reset`
(2606.22283, Listing 6.1, p.38). Device mask 0x4 selects the engine.

## 2. Reachable from Rust, and what signing is needed

Yes, and the C route is the easier one.

No entitlement is required for compute. The compiler and dispatch path are reachable from ordinary
user space, and the entitlement family gates loader-tier features instead (2606.22283, §6.3 p.40
and §8.1 p.47). The hard limit is elsewhere: a self-built program binary is rejected at load with
`0xe00002e2`, so the daemon must compile and sign on your behalf, and the reachable surface is
whatever the daemon's compiler accepts (§8.5, p.49).

My prototype is an ad-hoc, linker-signed `clang` binary with no entitlements and no Team ID.
Compile, load and evaluate all returned success. That is the empirical answer for a personal build.

For Rust: `e5rt_api.h` in ANEForge (MIT) is a recovered header of plain C signatures returning
`int64_t`, which binds directly with `libloading` or a `#[link]` block and needs no `objc2` at all.
The Objective-C route is also reachable through the `objc2` 0.6 already in our `Cargo.toml`, though
`_ANERequest`'s seven-argument class method is more comfortable through raw `objc_msgSend` casts.

Unknown: whether a Developer ID or notarized bundle behaves the same. I only tested ad-hoc.

## 3. The prototype

`reports/ane-direct/proto/ane_direct_matmul.m`. A single fp16 64x64 matmul expressed as a 1x1
convolution over a `[1,64,1,16]` activation, weights in a BLOBFILE, IOSurface I/O, checked against
an fp32 CPU reference.

```
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
  -o ane_direct_matmul reports/ane-direct/proto/ane_direct_matmul.m
./ane_direct_matmul
```

Ran twice, per the hardware constraint. Output:

```
staged program at BF598BB8...
compile: ok
load: ok
evaluate: ok
best: stride=16 channel-major, max relative error vs fp32 CPU reference 0.51953
PARITY FAIL
```

The API path is proven. Numeric parity is not. The second run scored the output buffer under four
row strides in both channel-major and transposed order, and none reproduced the reference; packed
stride 16 was merely the least wrong. The remaining suspect is the engine's internal activation
tiling, which the papers describe as packed `[1,C,1,S]` from byte 0 (#20) but which the netplist
grammar exposes as an `InputInterleave` / `OutputInterleave` factor (2606.22283, §6.2, p.38). Input
and output layout are wrong together if they are wrong at all, so sweeping output strides alone
cannot recover it.

The next experiment is an identity program, `out = identity(x)` over the same surfaces, which
reveals the permutation in one dispatch. Deferred while g38m owns the bench.

## 4. Resident state against the Parakeet decoder

The mechanism is buffer aliasing: compile a program declaring the state as both input and output,
then bind one buffer object to both ports, so the engine updates it in place across dispatches
(2606.22283, §14.7 p.85, Listing 14.2). Verified on M1 by the author with an accumulator returning
1, 2, 3, 4 over four dispatches. The native `state` type is gated and does not lower.

Mapped onto our decoder: `parakeet_unified_decoder` takes `h_in`, `c_in` as fp32 `[2,1,640]` and
returns `h_out`, `c_out`. Aliasing those would keep 2 x 2 x 640 fp16 values resident, saving about
10 KB of copies per step. That is the whole saving.

Dispatch count does not improve. The emit-versus-advance decision after the joint's argmax is host
control flow, so the loop stays at roughly 62 frames x (1 decoder + 1 to n joint) per 5 s utterance,
the same count g38m is measuring today. What changes is the cost of each dispatch, from Core ML's
per-prediction overhead down toward the 0.07 to 0.23 ms engine floor (2606.22283, p.85; ANEForge
reports about 90 us for a small fused program). Whether that is a saving at all depends on what
g38m measures Core ML actually charging.

Against that sits the LSTM. Table A.12 (p.255) gives `lstm` as no-path on M1 through M5, note
"unroll on host". A direct-route decoder means hand-unrolling both layers into conv, matmul,
sigmoid and tanh, roughly 8 to 16 ops per layer, then fusing them into one program so the floor is
paid once. Doable. Not obviously faster than the CPU path Core ML already uses for a 640-wide cell.

## 5. Risks

| Risk | Severity | Note |
|---|---|---|
| Private API breaks on a macOS update | High | Undocumented and version-fragile by the author's own statement (2606.22283, §6.3 p.40). Every selector and symbol would need re-probing per release. |
| MIL dialect drift | Medium | Orion pins `program(1.3)` and `func main<ios18>`; our shipped models emit `program(1.0)` / `ios17`. The accepted grammar is whatever the daemon's compiler happens to take. |
| Signing | Low | Ad-hoc signing worked here with no entitlements. Notarized-bundle behaviour untested. |
| Undiscovered layout contract | High, current blocker | Parity failed and the papers do not fully specify the activation tiling. |
| Compile and program caps | Medium | About 119 compiles per process (#5) and near 128 loaded programs per process (2606.22283, p.85). A bucketed encoder would need a cache budget. |
| Maintenance | High | Two of us would be maintaining a private-API runtime for a single-user dictation app. |

## Deferred timing, to run when g38m releases the bench

```
# 1. Fix layout first: identity program probe, one dispatch.
./ane_identity_probe

# 2. Bare dispatch floor, 1000 evaluations of the matmul program, p50/p95.
./ane_direct_matmul --repeat 1000 --report-percentiles

# 3. Compare against the same shape through Core ML MLComputeUnits.cpuAndNeuralEngine.
cargo run --release --bin bench_asr -- --stage-timers
```

## Sources

- Kumaresan, R. *Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference.* arXiv:2603.06728. Code: https://github.com/mechramc/Orion (MIT).
- Bryngelson, S. H. *Apple Neural Engine: Architecture, Programming, and Performance.* arXiv:2606.22283.
- *ANEForge: Python for direct computation on the Apple Neural Engine.* arXiv:2606.17090. Code: https://github.com/sbryngelson/ANEForge (MIT), notably `aneforge/_lib/e5rt_api.h`.
