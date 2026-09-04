# Direct ANE access below Core ML, assessed for Parakeet

Research spike for kata 1wet. Machine: M5 Pro, macOS 26.5.1 (25F80).
Page citations use the printed folio of each paper, which for arXiv 2606.22283 sits 6 below the PDF page.

## Recommendation

No-go for now, with one narrow exception worth keeping open.

The path works. Every private class and C entry point the two papers describe is present on this
machine, and a hand-authored MIL program compiled, loaded and evaluated on the engine from an
ad-hoc signed binary with no entitlements and no Core ML in the process. That answers the
feasibility question the issue asked.

The latency case is weaker. The decoder we would want to move is a two-layer LSTM, and `lstm`
has no engine path on any Apple silicon family including M5 (2606.22283, Table A.12, p.255).
Almost certainly Core ML runs that step off the engine already, which g38m is measuring. A direct
port replaces a CPU LSTM with a hand-unrolled gate graph paying a 0.07 to 0.23 ms engine dispatch
floor per step. Orion measured this shape of loss on GPT-2 124M, where CPU decode at 283 tok/s beat
ANE decode at 170 tok/s (2603.06728, p.14 and p.16).

The one design that would change the arithmetic is described in section 4: a single fused program
per frame, with the emit-versus-advance branch turned into arithmetic. Every operation it needs is
engine-native. That is a real lever, and it is also several weeks of work on a private API.

Revisit if g38m reports the decoder loop owning the 66 ms at 5 s. If a fixed per-utterance cost or
the encoder owns it instead, the encoder bucket path is the thing to look at.

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
`0xDEADBEEF`, dtype 1 for fp16, payload size at 72, data offset 128 at byte 80), which MIL
references as `BLOBFILE(path=..., offset=uint64(64))`, pointing at the chunk header rather than the
payload (#8, p.5). I verified that layout byte-for-byte against the shipped Parakeet decoder's
`weights/weight.bin`. Tensors are fp16 `[1, C, 1, S]` over IOSurface memory (p.3). Multi-input and
multi-output programs need uniform allocation sizes and alphabetically ordered names (#2, #3, #18,
#19), and evaluation needs roughly 49 KB of surface regardless of tensor size (#4).

**C route (ANEForge, and the arch paper's own listings).** The `e5rt_*` family exported from
`Espresso.framework`, in four phases: compile (`e5rt_e5_compiler_create_with_config`,
`e5rt_e5_compiler_compile`), load (`e5rt_program_library_retain_program_function`,
`e5rt_program_function_load_for_execution`, then the precompiled-compute-op options and operation
constructors), bind (`e5rt_buffer_object_alloc`, `e5rt_..._retain_input_port`,
`e5rt_io_port_bind_buffer_object`), and a hot loop of `prepare_op_for_encode` / `encode_operation`
/ `execute_sync` / `reset` (2606.22283, Listing 6.1, p.38). Device mask 0x4 selects the engine.
All 17 of these symbols resolve on this machine.

## 2. Reachable from Rust, and what signing is needed

Yes, and the C route is the easier one.

No entitlement is required for compute. The compiler and dispatch path are reachable from ordinary
user space, and the entitlement family gates loader-tier features instead (2606.22283, §6.3 p.40
and §8.1 p.47). The hard limit sits elsewhere: a self-built program binary is rejected at load with
`0xe00002e2`, so the daemon compiles and signs on your behalf and the reachable surface is whatever
its compiler accepts (§8.5, p.49).

My prototype is an ad-hoc, linker-signed `clang` binary with no entitlements and no Team ID.
Compile, load and evaluate all returned success, which is the empirical answer for a personal
build. Whether a Developer ID or notarized bundle behaves the same is unknown; I only tested ad-hoc.

For Rust, `e5rt_api.h` in ANEForge (MIT) is a recovered header of plain C signatures returning
`int64_t`, which binds with `libloading` or a `#[link]` block and needs no `objc2`. The
Objective-C route is reachable through the `objc2` 0.6 already in our `Cargo.toml`, though
`_ANERequest`'s seven-argument class method is more comfortable through raw `objc_msgSend` casts.

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

The API path is proven and numeric parity remains open. The second run scored the output buffer
under four row strides in both channel-major and transposed order, and none reproduced the
reference; packed stride 16 was merely the least wrong. The suspect is the engine's internal activation
tiling, described in the papers as packed `[1,C,1,S]` from byte 0 (#20) yet exposed by the netplist
grammar as an `InputInterleave` / `OutputInterleave` factor (2606.22283, §6.2, p.38). Input and
output layout are wrong together if they are wrong at all, so sweeping output strides alone cannot
recover it. Orion reached full token agreement with the same 32-byte rows (C=768, S=16), so row
padding alone is an unlikely culprit. The next experiment is an identity program, `out =
identity(x)`, run over several values of C as well as S so channel tiling and row stride separate.
Deferred while g38m owns the bench.

## 4. Resident state against the Parakeet decoder

The mechanism is buffer aliasing: compile a program declaring the state as both input and output,
then bind one buffer object to both ports, so the engine updates it in place across dispatches
(2606.22283, §14.7 p.85, Listing 14.2). Verified on M1 by the author with an accumulator returning
1, 2, 3, 4 over four dispatches. The native `state` type is gated and does not lower.

Mapped onto our decoder: `parakeet_unified_decoder` takes `h_in`, `c_in` as fp32 `[2,1,640]` and
returns `h_out`, `c_out`. Aliasing those keeps 2 x 2 x 640 values resident, saving about 10 KB of
copies per step, which is the whole of what aliasing alone buys.

Ported one-for-one, dispatch count is unchanged. The emit-versus-advance decision after the joint's
argmax is host control flow, so the loop stays at roughly 62 frames x (1 decoder + 1 to n joint)
per 5 s utterance, the count g38m is measuring today. What changes is the cost of each dispatch,
from Core ML's per-prediction overhead down toward the 0.07 to 0.23 ms engine floor (2606.22283,
p.85; ANEForge reports about 90 us for a small fused program).

The issue's "one dispatch per frame" is reachable by a different construction. Unroll `max_symbols`
decoder-plus-joint steps into a single program and turn the branch into arithmetic: `reduce_argmax`
over the joint logits, `equal` against a constant index vector to build the embedding selector,
`matmul` against the embedding table in place of a `gather`, and `select` on `token != blank` to
choose between the updated and the unchanged LSTM state. Compute is then fixed and no step depends
on a host decision, so h and c stay resident and the frame costs one dispatch. Appendix A gives
every one of those operations as native on all families through M5: `equal`, `not_equal` and
`select` (p.250), `reduce_argmax` and `reduce_max` (p.252), `cast` (p.254), `matmul` (p.249).
`one_hot` has no path (p.253) and is exactly what the equal-against-constant construction replaces.
At 62 frames that is 62 dispatches against 62 x (1 + n), and the wasted compute from running all
`max_symbols` steps every frame is small next to the floor for a 640-wide cell.

The LSTM still has to be unrolled by hand. Table A.12 (p.255) gives `lstm` as no-path on M1 through
M5, note "unroll on host", so both layers become conv, matmul, sigmoid and tanh, roughly 8 to 16
ops per layer. Combined with the unrolled symbol steps that is a program of a few hundred
operations, well inside the 16 to 64 op depth range where the engine reaches 94% utilization
(2603.06728, p.4).

## 5. Risks

| Risk | Severity | Note |
|---|---|---|
| Private API breaks on a macOS update | High | Undocumented and version-fragile by the author's own statement (2606.22283, §6.3 p.40). Every selector and symbol would need re-probing per release. |
| MIL dialect drift | Medium | Orion pins `program(1.3)` and `func main<ios18>`; our shipped models emit `program(1.0)` / `ios17`. The accepted grammar is whatever the daemon's compiler happens to take. |
| Signing | Low | Ad-hoc signing worked here with no entitlements. Notarized-bundle behaviour untested. |
| Undiscovered layout contract | High, current blocker | Parity failed and the papers do not fully specify the activation tiling. |
| Compile and program caps | Medium | About 119 compiles per process (#5) and near 128 loaded programs per process (2606.22283, p.85). A bucketed encoder would need a cache budget. |
| Maintenance | High | A private-API runtime for a single-user app is standing maintenance, re-verified every macOS release. |

## Deferred timing, to run when g38m releases the bench

Both probes below still need writing; neither exists yet.

```
# 1. Fix layout first. Identity program over several C and S values, one dispatch each,
#    so channel tiling and row stride separate.
./ane_identity_probe --channels 64,128,768 --seq 16,32

# 2. Bare dispatch floor. 1000 evaluations of the already-compiled matmul program, p50/p95.
./ane_direct_matmul --repeat 1000 --report-percentiles

# 3. The same shape through Core ML with MLComputeUnits.cpuAndNeuralEngine, for the delta.
cargo run --release --bin bench_asr -- --stage-timers
```

## Sources

- Kumaresan, R. *Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference.* arXiv:2603.06728. Code: https://github.com/mechramc/Orion (MIT).
- Bryngelson, S. H. *Apple Neural Engine: Architecture, Programming, and Performance.* arXiv:2606.22283.
- *ANEForge: Python for direct computation on the Apple Neural Engine.* arXiv:2606.17090. Code: https://github.com/sbryngelson/ANEForge (MIT), notably `aneforge/_lib/e5rt_api.h`.
