# Latency bench

`scripts/bench-latency.sh` drives `bench_asr` over generated TTS WAVs at
{1, 3, 5, 10, 20}s, 30 reps each, and emits `phase_timer` log lines that
`scripts/bench-aggregate.py` reduces into `baseline.csv` (or `$OUT_CSV`). It
also writes the matching `*-boundary.csv` through the PEP 723
`scripts/bench-boundary.py` aggregator.

See `docs/latency-plan.md` §1 for design and acceptance criteria.

## Gold-reference quality gate

The checked-in real-speech corpus gates model, quantization, backend, and
hotword changes against human-authored transcripts:

```bash
REPETITIONS=10 scripts/bench-gold.sh
```

`bench/gold/manifest.json` contains independent absolute and baseline-regression
WER/CER limits. `asr_diff` applies them to the worst repeated result, prints
per-fixture and per-category summaries, exits non-zero when either threshold
fails, and writes schema-v3 machine reports with:

- normalized lexical WER/CER, exact formatting matches, and separate insertion,
  deletion, and substitution counts;
- categories such as names, commands, numbers, punctuation, and vocabulary;
- repeat-run transcript/WER/CER spread, p50/p95 decode time, and real-time factor;
- backend/model/quantization/provider labels;
- decoder method plus requested/active vocabulary state, score, term count,
  and source/generated-hotword SHA-256 identities;
- app version, macOS version, chip, memory, CPU count, model-load time, and
  warmup, first-result, and full-process-tree peak-resident memory.

Lexical normalization preserves Unicode letters/numbers and accents,
lowercases them, removes apostrophes without splitting words, and treats other
punctuation as a separator. Raw reference and hypothesis strings remain in the
report so capitalization and punctuation changes are still auditable.

The representative corpus contains seven real human recordings from pinned,
licensed LibriSpeech and SLURP revisions, including proper nouns, commands,
numbers, distant/noisy audio, and custom-vocabulary cases. Sources, hashes, and
reproduction instructions are under `bench/gold/`. `gold.example.json` remains
only as the older macOS `say` format/smoke example.

Measured M5 Pro baselines, fairness rules, threshold derivation, known errors,
and rejected variants live in `docs/asr/{PERF,DISCREPANCIES,NEGATIVE_EVIDENCE}.md`.

The developer-only Qwen3-ASR challenger uses pinned PEP 723/`uv` oracle scripts
and does not affect the shipping dependency graph. Its immutable artifact
summary and no-go evidence are under `bench/qwen3-asr/`; the full interpretation
is in `docs/asr/QWEN3_ASR_EVALUATION.md`.

The measured vocabulary-score sweep and no-training decision are under
`bench/domain-adaptation/`; interpretation and future adapter/data gates are in
`docs/asr/DOMAIN_ADAPTATION.md`.

## Quick start

```bash
# First time only: launch Parakeet.app once so the model bundle downloads
# into ~/Library/Application Support/com.parakeet.rs/models/.
open target/release/bundle/osx/Parakeet.app   # or however you launch it

# Then:
scripts/bench-latency.sh                           # → bench/baseline.csv
# Shipping native int8 Parakeet Unified backend (builds the pinned worker):
BACKEND=coreml-unified OUT_CSV=bench/coreml-unified.csv \
    scripts/bench-latency.sh

# Production capture + VAD + ASR, frozen serial baseline vs optimized path.
# Requires a duplex Core Audio loopback named "BlackHole 2ch".
scripts/bench-end-to-end.sh

# Hold mode: release-to-transcript through the same production path.
# Also requires the "BlackHole 2ch" loopback.
scripts/bench-hold.sh
```

## What is and isn't measured

The bench loads pre-recorded WAVs, converts them to 16 kHz once at load, and
runs `Asr::recognize()` directly. The conversion sits outside the measured loop
because that is where production does it: since ADR-0030 `AudioCapture`
resamples inside its cpal callbacks, so `recognize()` is always handed 16 kHz
mono. Fixtures stay at 48 kHz on disk so these runs stay comparable with the
published baselines.
Each repetition also emits `asr_boundary` with worker-internal
resample-plus-inference time, outer Rust wall time, and their difference. For
the Core ML worker that difference prices Float32 pipe transfer, scheduling,
framing, response JSON, and Rust-side handling instead of attributing the full
short-utterance floor to IPC.
It **does not** exercise:

- `cpal` mic-capture callback latency
- the Silero VAD endpoint policy (750 ms for Tap; 90 ms for Tap Fast)
- the `CGEventKeyboardSetUnicodeString` keystroke insertion step
  (sub-ms per chord — see ADR-0019)

So `scripts/bench-latency.sh` is **ASR-only**. Use
`scripts/bench-end-to-end.sh` for the production capture, resampling, dual-VAD,
endpoint, session-shutdown, and ASR path, and `scripts/bench-hold.sh` for the
Hold path, where the hotkey release replaces the VAD endpoint. Both harnesses
stop at transcript-ready rather than typing into the user's focused app; the
only excluded production step is the synchronous synthetic-Unicode event post
(sub-ms per chord; see ADR-0019).

## End-to-end 3x result: M5 Pro 24 GB (2026-08-10)

The end-to-end harness selects `BlackHole 2ch` directly for both input and
output without changing macOS defaults. It trims only sub-threshold trailing
fixture silence, uses Core Audio's predicted playback timestamp for the last
non-silent sample, and feeds the production `cpal` capture and `streamer` path.
Both variants receive the same measurement-only acoustic endpoint marker.

The representative fixture is `5s_48000.wav` (4.854 s measured). Release
builds used two warmups and 30 measured repetitions per variant. Every
measured transcript had an exact lexical match to the reviewed reference.

| pipeline | n | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| sherpa + serial endpoint | 30 | 613.6 ms | 613.0 ms | 652.5 ms | 657.3 ms |
| Core ML + speculative decode | 30 | 189.7 ms | 182.0 ms | 203.0 ms | 203.0 ms |
| speedup | | **3.23×** | **3.37×** | **3.21×** | **3.24×** |

The optimized path starts ASR from the early 32 ms detector, discards the
provisional result whenever speech resumes, and lets an independent Silero
state remain the sole stop authority. Re-run on 2026-09-04 after ADR-0030
retired the resample stage, the gate reads 630.0 ms against 192.0 ms (3.28x
p50) and 952.3 ms against 203.0 ms (4.69x p95), every transcript matching.

Tap did not collect the resample saving, and the `phase_timer` lines said why:
`t_asr_start=4966`, `t_asr_done=5013`, `t_vad_endpoint=5094`. The speculative
decode finishes about 80 ms before the endpoint policy confirms, and
`dur_post_endpoint_ms` is 0 to 1 ms, so this path is endpoint-bound rather than
decode-bound. Taking 22 ms out of the decode widens that margin instead of
shortening the result. The saving lands where the decode is not hidden: Hold,
the serial fallback, and every utterance long enough that the decode would
otherwise outrun the confirmation window. ADR-0031 acted on the endpoint side
instead and moved Tap Fast to 90 ms; the window sweep is below.

This frozen comparison pins `--confirmation-ms 150`, Tap Fast's original policy,
so the historical 3× result stays like-for-like whatever the shipping window
becomes. Re-run on 2026-09-04 it reads, baseline before optimized, 594.5 → 182.0 ms p50
(3.27×) and 635.1 → 203.6 ms p95 (3.12×). The gate fails unless both p50 and p95 are at
least 3.0× and every transcript matches:

```bash
REPS=30 WARMUP_REPS=2 scripts/bench-end-to-end.sh
```

## Long-pause endpoint gate

Normal Tap uses a 750 ms confirmation policy; Tap Fast uses 90 ms for short
commands (ADR-0031). The separate endpoint gate replays versioned human LibriSpeech
audio through production capture, VAD, speculative Core ML inference, and
session shutdown. Its 14.225 s fixture includes a reviewed 544 ms natural
pause that the former policy cut. A pass requires zero early stops and p95
final-pause latency below one second for both the single- and multi-sentence
fixtures:

```bash
REPS=30 WARMUP_REPS=2 scripts/bench-endpoint-policy.sh
```

M5 Pro 24 GB release results (re-run 2026-09-04):

| fixture | repetitions | false stops | p50 | p95 |
|---|---:|---:|---:|---:|
| 3.505 s single sentence | 30 | **0** | 667.0 ms | 672.9 ms |
| 14.225 s multi sentence | 30 | **0** | 635.0 ms | 647.5 ms |

The unchanged Tap Fast comparison was also re-run for 30 repetitions after
this policy split. It retained **3.24× p50 / 3.18× p95** speedups (589.5 →
182.0 ms p50; 644.8 → 203.0 ms p95), so the representative no-polish gate
remains above its accepted 3× target.

The fixture manifest, source revision, hashes, references, and license are in
[`bench/endpointing/`](endpointing/). This gate isolates endpoint behavior;
transcript WER/CER remains the responsibility of `asr_diff`.

## Confirmation-window sweep

Tap's end-to-end number is the confirmation window plus Silero's detection lag;
`scripts/bench-endpoint-sweep.sh` sweeps that window over all three fixtures.
Each row reports two oracles, because neither alone is sound. `false_cuts`
counts commits landing before Core Audio's predicted instant for the fixture's
last sample above -80 dBFS — the LibriSpeech fixtures carry room tone above that
floor, so a short window can miss the marker with every word intact.
`mismatches` compares the transcript against the fixture reference and is the
oracle for lost speech.

The curve stops improving at 90 ms and reverses at 60, where every repetition
reports `t_asr_done == t_vad_endpoint`: the synchronous speculative decode
blocks the VAD watcher, so below about 90 ms the decode is the floor. ADR-0031
takes Tap Fast to 90 ms and rejects a punctuation-aware early commit that was
built and measured alongside it. Full tables are in
[`docs/asr/PERF.md`](../docs/asr/PERF.md).

```bash
REPS=15 WARMUP_REPS=2 scripts/bench-endpoint-sweep.sh
```

## Native Core ML result: M5 Pro 24 GB (2026-08-10)

Matched release runs used the same 48 kHz WAVs, resident recognizer, three
warmups, 30 measured repetitions, outer `Asr::recognize()` timer, and CSV
aggregator. `sherpa` is the frozen previous backend; `coreml-unified` is the
resident FluidAudio worker with the int8 offline encoder on CPU+ANE.

| bucket | sherpa p50 | unified p50 | p50 speedup | sherpa p95 | unified p95 | p95 speedup |
|--------|-----------:|------------:|------------:|-----------:|------------:|------------:|
| 1 s    | 112.0 ms   | 35.0 ms     | **3.20×**   | 116.5 ms   | 36.5 ms     | **3.19×**   |
| 3 s    | 226.0 ms   | 50.0 ms     | **4.52×**   | 239.6 ms   | 51.0 ms     | **4.70×**   |
| 5 s    | 361.5 ms   | 66.0 ms     | **5.48×**   | 384.3 ms   | 67.0 ms     | **5.74×**   |
| 10 s   | 580.0 ms   | 90.0 ms     | **6.44×**   | 597.8 ms   | 91.5 ms     | **6.53×**   |
| 20 s   | 1195.0 ms  | 188.0 ms    | **6.36×**   | 1245.3 ms  | 192.6 ms    | **6.47×**   |

The companion gold run passed at **2.38% WER / 2.22% CER** against limits of
4% / 3%, with 40% exact formatting and 74.4× aggregate model-reported RTFx.
This is a five-item macOS `say` smoke corpus, not a claim about real-user WER;
the representative-speech gate described above still applies before a release.

## Per-stage breakdown: M5 Pro 24 GB (2026-09-04)

`bench_asr --stage-timings` starts the worker with `--emit-stage-timings`, and
the worker then reports where each decode went. `scripts/bench-latency.sh`
passes the flag automatically for `BACKEND=coreml-unified` and reduces the
`asr_stages` log lines into `*-stages.csv` through `scripts/bench-stages.py`.

The decode pipeline lives in FluidAudio's `UnifiedAsrManager` and
`UnifiedRnntDecoder`, a pinned dependency this project depends on and patches
rather than vendors (`native/ParakeetCoreMLWorker/patches/fluidaudio.patch`, two
changes: the offline encoder window, and the hook the native decode loop is
injected through). The worker measures from outside all the same: at startup it
replaces the prediction implementations of `MLModel` and its registered
subclasses with timing wrappers that call straight through, and attributes each
dispatch to a stage by the input feature names FluidAudio's providers declare
(`mel` for the encoder, `targets` for the decoder, `encoder_step` for the
joint). Stage boundaries come from the resulting dispatch timeline: mel is the
gap before an encoder dispatch, the RNNT loop is everything from an encoder
dispatch to the last dispatch of that window. The shipping dictation path never
installs the wrappers.

With `--rnnt-engine native` the decode loop issues no Core ML dispatches at all,
so it reports its own steps instead: `native_decoder_steps` and
`native_joint_steps` stand in for `decoder_calls` and `joint_calls`, and
`decode_loop_native_ms` for `decode_loop_dispatch_ms`. The counts mean the same
thing, so the frame identity below reads either pair; a row with both populated
is rejected, since the loop runs on one engine per utterance.

Medians over 30 measured repetitions per bucket, three warmups, 48 kHz
fixtures, release build (`bench/coreml-unified-stages.csv`). The resample column
was retired by ADR-0030 and its measurement is reproduced under
"Retiring the resample stage" below.

| fixture | resample | mel | encoder | RNNT loop | post | worker total | IPC | ASR p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.816 s | 0.001 ms | 3.20 ms | **25.99 ms** | 3.25 ms | 0.04 ms | 32.7 ms | 0.11 ms | 32.0 ms |
| 2.828 s | 0.001 ms | 3.23 ms | **25.94 ms** | 9.04 ms | 0.05 ms | 38.2 ms | 0.17 ms | 38.0 ms |
| 4.854 s | 0.001 ms | 3.19 ms | **25.97 ms** | 15.43 ms | 0.06 ms | 44.8 ms | 0.21 ms | 44.5 ms |
| 8.062 s | 0.001 ms | 3.20 ms | **25.89 ms** | 23.57 ms | 0.07 ms | 52.7 ms | 0.27 ms | 52.5 ms |
| 16.513 s | 0.001 ms | 6.22 ms | **51.95 ms** | 53.04 ms | 0.12 ms | 111.5 ms | 0.47 ms | 111.5 ms |

The 2026-09-04 g38m table this replaces was measured on a differently
synthesized fixture set (0.740, 2.507, 4.967, 8.150, 15.691 s), so it is not a
like-for-like row-by-row comparison; the before/after below re-measured the old
code on these exact fixtures instead.

Worker total is resample plus the profiled transcribe interval, and with
resample retired the two are the same number. It now sits within about 0.01 ms
of the `asr_boundary` internal time (44.789 against 44.795 ms at 5 s), where the
g38m run had a 0.7 to 1.0 ms gap: that gap was the Swift work around the
conversion, not around the profiled window. IPC halved with the payload, from
0.42 to 0.21 ms at 5 s and 1.14 to 0.47 ms at 20 s, and stays under 0.5% of the
call.

Enabling the profiler costs nothing measurable. Matched 30-repetition runs with
and without `--stage-timings` measured 35.51 against 35.51 ms at 1 s and 67.09
against 66.92 ms at 5 s. The 10 s bucket, which has the most dispatches of any
single-window fixture, was measured with eight interleaved on/off blocks of 15
repetitions so that machine drift hits both arms equally: the per-block delta
has a median of -0.17 ms and a mean of -0.38 ms, and its sign flips across
blocks, so the effect is below the noise. The interception is two clock reads
and a lock per dispatch; the class sweep that finds the entry points stops once
all three stages have dispatched, and the model placement is read once per stage
rather than per call.

These absolute numbers need a quiet machine, which matters more than the
profiler does. In the g38m run, which still paid the resample, the 1, 3, 5, and
20 s buckets landed within about 2 ms of the 2026-08-11 published baseline and
10 s was roughly 5 ms higher; the interleaved A/B above ruled the profiler out
as the cause. A separate repeat under a
competing job reproduced the dispatch counts exactly and kept the encoder flat
at 26.4 to 28.5 ms, with resample and the decode loop 10 to 20% higher. The
shape of the breakdown is stable; the millisecond values are not, so compare
against the ASR-only p50 column from the same run.

The encoder cost does not depend on utterance length. `UnifiedAsrManager`
zero-pads every window to a fixed 15 s buffer (240,000 samples, 1,501 mel
frames) and runs the full offline encoder graph on it, so a 0.816 s utterance
pays the same 26.0 ms as an 8.062 s one. The 16.513 s fixture exceeds the 15 s
window and needs a second one, which doubles both mel and encoder.
`chunkStarts` adds that second window only past 240,000 samples, so a 14 s
utterance still runs a single encoder pass.

### Core ML dispatch counts per utterance

| fixture | windows | encoder calls | decoder calls | joint calls | decoded frames | per joint | per decoder |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.816 s | 1 | 1 | 7 | 17 | 11 | 113 µs | 170 µs |
| 2.828 s | 1 | 1 | 20 | 55 | 36 | 105 µs | 147 µs |
| 4.854 s | 1 | 1 | 36 | 96 | 61 | 101 µs | 145 µs |
| 8.062 s | 1 | 1 | 55 | 155 | 101 | 98 µs | 139 µs |
| 16.513 s | 2 | 2 | 119 | 349 | 232 | 98 µs | 142 µs |

Decoded frames are `joint_calls - (decoder_calls - windows)`, since the greedy
loop issues one joint per frame plus one per emitted token, and one decoder call
per window plus one per emitted token. Every single-window bucket matches its
80 ms frame arithmetic: 4.854 s of audio decodes 61 frames. That agreement is
what validates the counts.

These counts are also identical to the pre-ADR-0030 arm at every bucket, which
is the sharpest available check that moving the resample into Rust did not
change what the encoder sees: a different filter would perturb the mel frames
and, sooner or later, the token the greedy loop emits.

The agreement is also checked at runtime, because a stage whose Core ML entry
point stops being intercepted reports zero cost while every other number stays
plausible. `bench_asr` fails the run when a report has no encoder dispatches,
when `windows` and `encoder_calls` disagree, or when any prediction goes
unattributed, and `scripts/bench-stages.py` repeats those checks before writing
a CSV. The encoder is the stage worth naming here: losing it also drives
`windows` to zero, which leaves the frame identity intact and the row
believable.

Per-frame loop cost is flat at roughly 0.25 ms across all lengths. The dispatch
floor of about 100 µs per joint call is the dominant term inside the loop:
`decode_loop_dispatch_ms` accounts for 95% of `decode_loop_ms` at every bucket,
so the Swift-side loop work (encoder-step extraction, `MLMultiArray`
allocation) is not where the loop time goes.

### Which compute unit runs each stage

The encoder runs on `cpuAndNeuralEngine`; the decoder and the joint-decision
model both run `cpuOnly`. The `compute_units` column reports
`MLModelConfiguration.computeUnits` read off each live model object at dispatch
time, so it records the placement Core ML actually used rather than the one the
worker asked for. It matches the pinned FluidAudio
source, where `UnifiedAsrManager.loadModels` builds a separate `cpuOnly`
configuration for the decoder and joint and comments that only the encoder uses
ANE or GPU.

The ANE is genuinely carrying the encoder. Eight repetitions on the 4.967 s
fixture at each placement:

| encoder placement | encoder median |
|---|---:|
| `cpu-and-neural-engine` | 26.05 ms |
| `cpu-only` | 86.01 ms |

`cpu-and-gpu` produces no result at all. The worker dies during warmup with
`MPSGraphExecutable.mm:5070: failed assertion 'Error: MLIR pass manager failed'`,
the int8-on-MPSGraph failure FluidAudio's own loader comments on and coerces
away from for `.all`.

```bash
BACKEND=coreml-unified OUT_CSV=bench/coreml-unified.csv scripts/bench-latency.sh
./target/release/bench_asr --backend coreml-unified --wav bench/audio/5s_48000.wav \
    --reps 8 --warmup-reps 3 --stage-timings --compute-units cpu-only
```

### Where the 45 ms at 5 s goes

Two stages own it. On the 4.854 s fixture the 44.8 ms of worker-internal time is
encoder 26.0 ms (58%), RNNT decode loop 15.4 ms (34%), and mel 3.2 ms (7%), with
IPC at 0.21 ms and post-processing under 0.1 ms. The 32 ms floor at 1 s is owned
by the encoder: the offline path pads every utterance to the fixed 15 s window,
so 26.0 ms of encoder plus 3.2 ms of mel is length-independent work that a
one-word utterance pays in full, and that 29.2 ms is 91% of the 1 s result. What
grows with length is now only the decode loop, at 0.25 ms per 80 ms frame,
entirely on the CPU, with roughly 100 µs of each frame being a single joint
dispatch.

### Retiring the resample stage

Before ADR-0030 a third stage sat between them. The worker received device-rate
audio and let FluidAudio's `AudioConverter` convert it, which builds a fresh
`AVAudioConverter` per call and cost a linear 4.6 ms per second of 48 kHz input:
22.4 ms of a 67 ms result at 5 s, more than the whole decode loop at every
measured length, and all of it after the endpoint. Rust already converted the
same audio for Silero VAD, so the work happened twice and the copy on the
critical path was the redundant one.

The fix converts once, in the cpal capture callback, so 16 kHz audio is ready
when the user stops speaking. Both arms below ran back to back in one session on
the same five fixtures; "before" is commit `9857547`.

| fixture | resample before | resample after | ASR p50 before | ASR p50 after | change |
|---|---:|---:|---:|---:|---:|
| 0.816 s | 3.87 ms | 0.001 ms | 36.0 ms | 32.0 ms | −4.0 ms |
| 2.828 s | 13.04 ms | 0.001 ms | 51.0 ms | 38.0 ms | −13.0 ms |
| 4.854 s | 22.37 ms | 0.001 ms | 67.0 ms | 44.5 ms | −22.5 ms |
| 8.062 s | 36.98 ms | 0.001 ms | 91.0 ms | 52.5 ms | −38.5 ms |
| 16.513 s | 76.50 ms | 0.001 ms | 190.5 ms | 111.5 ms | −79.0 ms |

The change is the retired resample plus a smaller second term: the pipe carries
16 kHz floats instead of 48 kHz, so measured IPC falls from 0.42 to 0.21 ms at
5 s and from 1.14 to 0.47 ms at 20 s. At 5 s, 22.37 + 0.21 accounts for 22.6 of
the 22.5 ms measured. Every other stage held: mel 3.19 against 3.23 ms, encoder
25.97 against 26.04, profiled transcribe interval 44.79 against 44.87.

## Bucketed short-window encoders: M5 Pro 24 GB (2026-09-04)

With the resample retired, the encoder is the whole of the short-utterance
floor: 26.0 ms of a 32.7 ms one-second result, and flat, because the offline
encoder is compiled at one fixed 15 s mel window and `UnifiedAsrManager`
zero-pads every utterance to it. `scripts/build-bucket-encoder.py` exports the
same NVIDIA checkpoint at shorter windows through FluidInference's `mobius`
pipeline, and the worker sends each utterance to the narrowest compiled window
that holds it. Buckets are discovered by filename in the model directory, so the
two arms below differ only in which `--model-dir` the worker was given.

Encoder cost against compiled window, `parakeet-encoder-probe` on zero inputs,
ten predictions after three warmups:

| window | mel frames | encoder frames | predict p50 |
|---:|---:|---:|---:|
| 2 s | 201 | 26 | 7.70 ms |
| 5 s | 501 | 63 | 9.64 ms |
| 8 s | 801 | 101 | 12.21 ms |
| 12 s | 1201 | 151 | 24.48 ms |
| 15 s (shipped) | 1501 | 188 | 26.11 ms |

The curve bends: 7.5 µs per frame from 201 to 801, 30.7 µs from 801 to 1201,
5.4 µs from 1201 to 1501. That bend is unexplained. It is why an 8 s window is
worth having and a 12 s one is not, and why extrapolating from the short end
would have been wrong: about 6 ms of the encoder is fixed cost no shorter window
removes. Derivation and method are in
[`docs/asr/COMPUTE_PLAN.md`](../docs/asr/COMPUTE_PLAN.md).

Matched 30-repetition runs, three warmups, release build
(`bench/coreml-unified-rerun-stages.csv` against
`bench/coreml-unified-buckets-stages.csv`). Machine 80 to 92% idle throughout.

| fixture | captured | bucket taken | encoder before | encoder after | mel before | mel after |
|---|---:|---|---:|---:|---:|---:|
| `1s_48000` | 0.816 s | 2 s | 25.98 ms | **7.71 ms** | 3.18 ms | 0.71 ms |
| `3s_48000` | 2.828 s | 5 s | 25.31 ms | **9.74 ms** | 3.06 ms | 1.28 ms |
| `5s_48000` | 4.854 s | 5 s | 25.45 ms | **9.98 ms** | 3.11 ms | 1.34 ms |
| 7 s cut | 7.000 s | 8 s | 25.60 ms | **12.45 ms** | 3.10 ms | 1.89 ms |
| `10s_48000` | 8.062 s | none | 25.55 ms | 25.60 ms | 3.14 ms | 3.16 ms |
| `20s_48000` | 15.755 s | none | 51.88 ms | 51.16 ms | 6.22 ms | 6.05 ms |

Mel falls with the encoder because `UnifiedMelExtractor` is built at the
layout's window, so a 2 s bucket computes 201 mel frames instead of 1501.

These runs predate the merge of ADR-0030, so both arms still paid the
worker-side resample. That does not touch these two columns, and the check is in
the table: the "before" arm's encoder and mel reproduce the post-ADR-0030
per-stage table above within 0.7 ms at every fixture, which is what licenses
reading the two tables together.

Worker total, which excludes resample in both arms and is therefore directly
comparable to the post-ADR-0030 numbers above:

| fixture | captured | worker total before | after | saved |
|---|---:|---:|---:|---:|
| `1s_48000` | 0.816 s | 32.40 ms | **11.26 ms** | −21.1 ms |
| `3s_48000` | 2.828 s | 37.08 ms | **19.78 ms** | −17.3 ms |
| `5s_48000` | 4.854 s | 43.98 ms | **27.01 ms** | −17.0 ms |
| 7 s cut | 7.000 s | 49.70 ms | **35.61 ms** | −14.1 ms |
| `10s_48000` | 8.062 s | 52.91 ms | 52.89 ms | 0.0 ms |
| `20s_48000` | 15.755 s | 112.61 ms | 110.77 ms | −1.8 ms |

The "before" column agrees with the independently measured post-ADR-0030 worker
totals in the per-stage table above to within 1.1 ms at every fixture, and the
"after" column was checked directly on the merged code: a single 4.854 s
decode with ADR-0030's 16 kHz capture path reports resample 0.001 ms, mel
1.43 ms, encoder 10.01 ms and worker total 27.17 ms, against the 27.01 ms in
the table. ASR p50 is worker total plus IPC, which is 0.11 to 0.27 ms across
this range, so a one-second utterance goes from about 32 ms to about 11 ms of
ASR. Repeated end-to-end p50 on the merged code has not been re-measured; when
it is, it belongs in the per-stage table above rather than here.

The 8.062 s fixture is unchanged twice over: it is 128,992 samples against the
8 s bucket's 128,000, missing by 62 ms of audio, and it is also past the 8 s
long-regime threshold. No stock fixture lands between 5 and 8 seconds, so the
8 s bucket is measured on a 7.0 s cut of the gold corpus's 14.225 s
`librispeech-multi` recording — real speech, trimmed with `soundfile` rather
than newly synthesized. The 15.755 s fixture needs two 15 s windows either way.

Every row has identical Core ML dispatch counts in both arms, which is what says
the decode path did not change: 7/17, 20/55, 36/96, 48/135, 55/155 and 119/349
decoder/joint calls respectively.

At one second the remaining 11.3 ms is encoder 7.71, decode loop 2.83, mel 0.71
and post 0.03. The length-independent encoder-and-mel share of the ASR call
falls from 91% to 75%; what is left is a smaller fixed cost of the same kind,
and the decode loop, which grows at 0.25 ms per 80 ms frame and now dominates
past about 3 seconds of audio.

Quality is unchanged. Matched ten-repetition gold runs, same corpus and worker,
differing only in model directory:

| arm | WER | CER | corpus decode p50 | p95 | RTFx p50 | peak RSS | load |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 s only | 5.434783% | 3.571429% | 0.4534 s | 0.4682 s | 75.1× | 0.10 GiB | 0.107 s |
| buckets 2/5/8 | 5.434783% | 3.571429% | **0.3397 s** | **0.3879 s** | **100.2×** | 0.19 GiB | 0.495 s |

Both are 5 word edits of 92 and 17 character edits of 476, with zero WER and CER
spread and zero changed outputs across ten repetitions, so the bucket arm sits
exactly on the frozen baseline and passes its 0.00-point regression cap. Every
hypothesis is byte-identical between the two arms, checked field by field rather
than inferred from equal scores. The gold corpus routes one fixture to the 2 s
bucket, five to the 5 s, and the 14.225 s fixture to the unbucketed path; the
7 s cut covers the 8 s bucket, and it is byte-identical too. Those corpus
timings also predate ADR-0030 in both arms.

Each bucket is a separate 590 MB compiled program, so 2/5/8 costs 1.77 GB of
disk on top of the shipped 569 MB encoder. Resident memory is far cheaper
because Core ML maps the weights: 0.10 to 0.19 GiB. The first load of a bucket
compiles a Core ML plan and takes about 6 s; the plan cache is persistent, so
warm load is 0.495 s against 0.107 s. That cold cost is behind a blocking read
in the Rust worker handshake and is tracked as kata hrs0. The ANE's ~128
loaded-program cap is not close at three buckets.

```bash
scripts/build-bucket-encoder.py --seconds 5    # ~90 s per bucket, 1 to 14
BACKEND=coreml-unified OUT_CSV=bench/coreml-unified-buckets.csv \
    PARAKEET_COREML_MODEL_DIR=<dir with bucket encoders> scripts/bench-latency.sh
REPETITIONS=10 COREML_WORKER=target/release/parakeet-coreml-worker \
    COREML_MODEL_DIR=<dir with bucket encoders> scripts/bench-gold.sh
```

Bucket artifacts are not on Hugging Face and the Rust download and verification
path knows nothing about them, so a stock model directory has no buckets and
behaves exactly as the tables above describe.

## Native RNNT decode loop: M5 Pro 24 GB (2026-09-04)

With bucketing and the resample retired, the greedy transducer loop was the
largest stage left on a short utterance. It issued one `decoderModel.prediction`
per emitted token and one `jointDecisionModel.prediction` per frame and per
token, each a separate Core ML call against a dispatch floor near 100 µs.
Neither model has a Neural Engine path — the prediction network is a two-layer
LSTM, which Core ML places `cpuOnly` by necessity ([`COMPUTE_PLAN.md`](../docs/asr/COMPUTE_PLAN.md))
— so the dispatch bought nothing but the driver round trip.

`--rnnt-engine native` runs both programs in the worker process on the weights
read out of the same two `.mlmodelc` bundles. `--rnnt-engine coreml` keeps
FluidAudio's loop, so both arms are one build apart.

Medians over 30 measured repetitions per bucket, three warmups, 48 kHz fixtures,
release build, 2/5/8 s bucket encoders, machine 84 to 93% idle throughout
(`bench/coreml-unified-rnnt-coreml-stages.csv` against
`bench/coreml-unified-rnnt-native-stages.csv`).

| fixture | windows | decoder steps | joint steps | Core ML calls in the loop | loop, `coreml` | loop, `native` |
|---|---:|---:|---:|---:|---:|---:|
| 0.740 s | 1 | 7 | 16 | 23 → **0** | 2.91 ms | **1.40 ms** |
| 2.507 s | 1 | 20 | 51 | 71 → **0** | 8.15 ms | **3.73 ms** |
| 4.967 s | 2 | 49 | 134 | 183 → **0** | 20.77 ms | **8.63 ms** |
| 8.150 s | 1 | 57 | 158 | 215 → **0** | 24.21 ms | **9.92 ms** |
| 15.691 s | 2 | 122 | 342 | 464 → **0** | 52.33 ms | **21.42 ms** |

The step counts are identical in both arms at every length, which is the check
that says the loop decoded the same way rather than merely faster: the native
engine reports them as `native_decoder_steps` and `native_joint_steps`, and the
decoded-frame identity reads either pair. The gold corpus produces
byte-identical hypotheses on all seven fixtures, at 5.43% WER / 3.57% CER over
10 repetitions with no nondeterministic output.

Whole-decode effect, same runs:

| fixture | worker total, `coreml` | worker total, `native` | ASR p50, `coreml` | ASR p50, `native` |
|---|---:|---:|---:|---:|
| 0.740 s | 11.71 ms | **10.22 ms** | 11.0 ms | **10.0 ms** |
| 2.507 s | 19.09 ms | **15.89 ms** | 19.0 ms | **15.5 ms** |
| 4.967 s | 42.41 ms | **30.16 ms** | 42.0 ms | **30.0 ms** |
| 8.150 s | 52.82 ms | **38.36 ms** | 53.0 ms | **38.0 ms** |
| 15.691 s | 109.37 ms | **78.11 ms** | 109.0 ms | **78.0 ms** |

### The 5 ms target was missed

Kata 2564 asked for the loop under 5 ms at 5 s. It is 8.63 ms: 2.4x faster, not
4x. The remaining cost is weight traffic, not dispatch. One prediction-network step
reads 13.1 MB of fp16 weights and the 4.967 s fixture takes 49 of them; with 49
decoder-side projections at 0.8 MB and 134 joint decisions at 1.3 MB that is
642 + 40 + 176, about 860 MB for one utterance, which no amount of dispatch
removal touches.

The next lever is the embedding-input product. The first LSTM layer computes
`W_ih · embed[token]`, which depends only on the token, so all 1025 of them can
be precomputed into a 10.5 MB fp32 table of partial sums. That drops the first
layer's input matrix — 3.3 MB of the 13.1 MB — from every step, about 25% of the
loop's traffic, and changes nothing but the order the fp32 sum is accumulated
in. It is not implemented here.

### How many threads the row product splits across

The gate product is split across row slices with `concurrentPerform`; slices are
independent and accumulate separately, so the split cannot change a result.
`PARAKEET_RNNT_THREADS` sets the count, `PARAKEET_RNNT_JOINT_THREADS` the count
for the joint's two smaller products. Medians of 20 repetitions on the 4.967 s
fixture:

| slices | joint slices 1 | joint slices 2 |
|---:|---:|---:|
| 1 | 16.25 ms | 16.13 ms |
| 2 | 11.30 ms | 11.05 ms |
| 3 | 9.49 ms | 9.41 ms |
| 4 | **8.46 ms** | 8.38 ms |
| 5 | 8.14 ms | — |
| 6 | 10.85 ms | 10.93 ms |

Scaling holds to four and breaks at six, which is where the work starts landing
on efficiency cores. Five is 4% faster than four and one slice from that cliff;
the compiled default is four, on the shoulder rather than the edge. Splitting
the joint's products buys about 1% and widens the spread, so it defaults to one.

```bash
MD=<dir with bucket encoders>
REPS=30 BACKEND=coreml-unified MODEL_DIR="$MD" RNNT_ENGINE=coreml \
    OUT_CSV=bench/coreml-unified-rnnt-coreml.csv scripts/bench-latency.sh
REPS=30 BACKEND=coreml-unified MODEL_DIR="$MD" RNNT_ENGINE=native \
    OUT_CSV=bench/coreml-unified-rnnt-native.csv scripts/bench-latency.sh
PARAKEET_RNNT_THREADS=4 ./target/release/bench_asr --backend coreml-unified \
    --model-dir "$MD" --wav bench/audio/5s_48000.wav --reps 20 --warmup-reps 3 \
    --stage-timings --rnnt-engine native
```

### A bucket artifact these runs exposed

The 4.967 s fixture runs **two** encoder windows, and pays 19.3 ms of encoder
rather than 9.6. `EncoderBuckets.select` routes on `sampleCount <= seconds *
16_000`, but a window only decodes `windowSamples / frameSamples * frameSamples`
— 79,360 samples for the 5 s bucket, not 80,000. An utterance between those two
numbers is routed to a bucket that cannot cover it in one window, and the last
112 samples cost a second full encoder pass. It affects both arms equally, so
the comparison above stands; it belongs to the bucketing work rather than here.

## Parakeet TDT 0.6B v3 challenger: M5 Pro 24 GB (2026-09-04)

TDT predicts a duration per emitted token and skips encoder frames, so its
decoder work scales with tokens rather than frames. FluidAudio's published
benchmarks put TDT v3 at 155.6× RTFx against 123.3× for Unified batch on the
same int8-on-ANE setup, which is the reason to try it. It is loaded through the
same worker behind `--model-variant tdt-v3` (kata f0zg); the default is
unchanged.

The graph set is `FluidInference/parakeet-tdt-0.6b-v3-coreml` at revision
`7dd20fe6b1797d35f5e3307e8b1732d9a178edfe`, fetched by
`scripts/fetch-tdt-v3-model.py`; per-file byte lengths and SHA-256 digests are
in [`tdt-v3-model-manifest.json`](tdt-v3-model-manifest.json). Four compiled
graphs plus the vocabulary, 469 MB on disk.

**The encoder window is the same 15 s.** `Encoder.mlmodelc` takes a fixed
`[1, 128, 1501]` mel — 1,501 frames at a 10 ms hop — and emits 188 encoder
frames at 8× subsampling, which is byte-for-byte the shape the Unified offline
encoder takes. So the bucketed short-window work above does not carry over:
TDT would need its own re-conversion at each window before it could pay the
same short-utterance saving.

### Latency against the shipping backend

Matched 30-repetition runs, three warmups, release build, stage timings on,
same fixtures. Machine 85 to 93% idle throughout. Both arms exclude nothing:
the outer timer wraps `Asr::recognize()`, and the Rust-to-worker boundary was
0.10 to 0.44 ms in both, so the gap below is entirely model-side.

| fixture | Unified p50 | TDT v3 p50 | Unified p95 | TDT v3 p95 |
|---|---:|---:|---:|---:|
| 1 s | **31.0 ms** | 45.0 ms | **33.5 ms** | 49.0 ms |
| 3 s | **36.0 ms** | 51.0 ms | **38.0 ms** | 56.5 ms |
| 5 s | **44.5 ms** | 58.5 ms | **57.2 ms** | 64.3 ms |
| 10 s | **53.0 ms** | 68.0 ms | **57.7 ms** | 76.0 ms |
| 20 s | **110.5 ms** | 182.0 ms | **116.9 ms** | 193.9 ms |

TDT loses at every length: 14 to 15 ms flat from 1 to 10 seconds, and 71.5 ms
at 20 s where two windows double every per-window cost. The 155.6× versus
123.3× RTFx that motivated the trial does not appear here at any length.

Where the flat 14 ms goes, from the same runs:

| stage | Unified 1 s | TDT v3 1 s | Unified 5 s | TDT v3 5 s |
|---|---:|---:|---:|---:|
| mel (host) | 3.04 | 0.24 | 3.12 | 0.24 |
| mel graph | 0.00 | 1.04 | 0.00 | 1.02 |
| encoder | 25.49 | **23.93** | 26.00 | **23.82** |
| decode loop | **2.88** | 5.11 | **15.43** | 18.78 |
| post | **0.03** | 14.72 | **0.06** | 15.06 |
| total | **31.53** | 45.01 | **44.78** | 58.90 |

TDT's encoder is 1.6 to 2.2 ms *faster* on these fixtures and its mel front end
is 1.8 ms cheaper. Both are erased by a post-dispatch tail — the interval
between the last Core ML dispatch and the end of `transcribe` — which Unified
pays 0.03 to 0.12 ms for and TDT pays about 15 ms for.

That tail is **unexplained**. It is close to constant where a per-token cost
could not be: across the one-window fixtures it moves from 14.72 to 15.16 ms
while decoder calls go from 5 to 52 and joint calls from 13 to 59, so tokenizer
decode and token-timing assembly are ruled out as the bulk of it. It rises to
19.58 ms on the two-window 20 s fixture, so some of it is per-window. Candidates
not separated here: overlap-merge and hypothesis assembly in FluidAudio's
`ChunkProcessor`, the progress-emitter session it opens and closes per
utterance, or teardown of the per-utterance decoder state. None is measured;
the profiler's timeline ends at the last dispatch and nothing instruments what
follows. Whatever it is, it is worth more than the joint graph to anyone
revisiting TDT, and it is not the duration head.

The encoder result should be read narrowly too: 23.8 to 23.9 ms against 25.4 to
26.0 ms is a real and repeatable difference on this hardware at this precision,
but both are the same 15 s graph shape and the gap is under 10%.

### The duration head skips frames, and the joint graph eats the saving

Paired 14.225 s fixture, one 15 s window, 10 repetitions after 3 warmups:

| arm | decoder calls | joint calls | joint dispatch | post | total |
|---|---:|---:|---:|---:|---:|
| Unified | 84 | **261** | 25.55 ms | 0.10 ms | **67.05 ms** |
| TDT v3 | 82 | **92** | 27.58 ms | 14.48 ms | 82.42 ms |
| TDT v3, decode pinned CPU-only | 82 | **92** | 28.90 ms | 13.99 ms | 84.19 ms |

The frame skipping is real and large: 92 joint predictions against 261, 2.84×
fewer on identical audio. It returns nothing, because each TDT joint call costs
3.06× what a Unified one does — 0.300 ms against 0.098 ms.

The third row settles why. FluidAudio's TDT loader places the decoder and joint
on CPU+ANE where the Unified loader pins them CPU-only, which was the other
candidate explanation. Pinning them (`--tdt-decode-compute-units cpu-only`,
encoder left on the Neural Engine) moves joint dispatch from 27.58 ms to
28.90 ms — slightly worse, not better. Placement is not the cause. What is left
is the graph: `JointDecisionv3` computes `top_k_ids` and `top_k_logits` at
K=64 on every call for script-aware language filtering, and the v3 vocabulary
is 8,192 entries against Unified's 1,024. A TDT conversion without the top-K
outputs is the only version of this model worth re-measuring.

### Quality: a hard fail on the English gate

Matched ten-repetition gold runs, same corpus and worker, differing only in
model directory. Zero WER and CER spread and zero changed outputs in both arms,
so these are exact, not sampled.

| arm | WER | CER | corpus p50 | p95 | RTFx p50 | peak RSS | load | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Unified | **5.434783%** | **3.571429%** | **0.3048 s** | **0.4239 s** | **111.7×** | **0.10 GiB** | 0.135 s | pass |
| TDT v3 | 7.608696% | 3.991597% | 0.4161 s | 0.4990 s | 81.8× | 0.12 GiB | **0.100 s** | **fail** |

Seven word edits of 92 against five, and 19 character edits of 476 against 17.
The manifest's regression cap is 0.00 points, so the bar is WER ≤ 5.434783%
exactly and TDT is 2.17 points over. One word edit on this corpus is 1.09
points, so the margin is two whole errors, not rounding. TDT's absolute 7.61%
is still under the 8.00% ceiling; the gate that fails is the regression one.

Read per fixture, the difference is narrower than the aggregate suggests. Four
of seven are clean in both arms. Of the three that fail, two fail *identically*:

| fixture | reference | Unified | TDT v3 |
|---|---|---|---|
| `slurp-alarm-seven-thirty-close` | "Please wake me up at seven thirty AM." | "…at seven hundred and thirty AM" — 2 edits | "…at seven hundred and thirty AM." — 2 edits |
| `slurp-music-olly-tactics-close` | "Hey Olly, play playlist Tactics from music." | "Hey Ollie, play playlist tactics for music" — 2 edits | "Hey Ollie, play playlist tactics for music." — 2 edits |
| `slurp-stock-ibm-close` | "Is IBM up today?" | "Is IPM up today?" — **1 edit** | "It's I PM up today." — **3 edits** |

**The entire 5 → 7 regression is the IBM fixture.** Both models mishear the
acronym; Unified corrupts one word and TDT splits it into three. Every
per-category difference below traces to that one fixture, because it is the
only one the two arms disagree on:

| category | Unified WER | TDT v3 WER |
|---|---:|---:|
| commands | **12.20%** | 17.07% |
| custom-vocabulary | **27.27%** | 45.45% |
| numbers | 6.67% | 6.67% |
| proper-nouns | **3.57%** | 5.95% |
| punctuation | **5.43%** | 7.61% |
| general, long, noisy | 0.00% | 0.00% |

That is worth stating plainly: on this corpus the gate is decided by a single
four-word utterance. The no-go stands — the cap is 0.00 points and a regression
is a regression — but "TDT is worse at English" is more than these seven
fixtures can support. What they do support is that TDT is not *better*, and
that it fails a gate Unified passes. A wider corpus would be needed before
claiming a general English quality gap, and it is not worth building one while
the latency case below also fails.

TDT's one win is warm model load, 0.100 s against 0.135 s, which is off the
dictation path either way.

### Long-form chunk concurrency

FluidAudio's TDT path decodes long-form chunks four at a time; Unified's
offline windowing is serial. On the two-window 20 s fixture (15.691 s captured)
that is a real wall-clock win — 121.29 ms parallel against 183.85 ms serial —
and it is also why the worker defaults TDT to `--tdt-chunk-concurrency 1`. Two
Core ML predictions in flight make the stage columns timeline sums over
overlapping intervals, so they double-count and stop partitioning the decode
interval: the parallel run reported 100.10 ms of decode-loop dispatch inside a
52.93 ms wall interval. `StageProfiler` now measures that overlap directly and
both `bench_asr` and `scripts/bench-stages.py` refuse a report containing any,
rather than publishing a split that does not add up. The guard is checked in
both directions: at concurrency 1 the identity holds
(15.111 + 2.093 + 48.789 + 105.172 + 19.542 == 190.708 ms) and at
`--tdt-chunk-concurrency 4` the run is rejected with 85.550 ms of overlap.

Serial is also the honest default for this workload. A dictation utterance is
decoded on its own, chunking only engages past 15 s, and the long-regime
threshold is 8 s. Raise the flag to measure the parallel arm deliberately.

### Reproducing

Raw artifacts for every arm above are under `bench/f0zg/`.

```bash
scripts/fetch-tdt-v3-model.py    # 469 MB, pinned revision, SHA-256 verified
TDT="$HOME/Library/Application Support/com.parakeet.rs/models/coreml/parakeet-tdt-0.6b-v3"

BACKEND=coreml-unified OUT_CSV=bench/f0zg/unified.csv scripts/bench-latency.sh
BACKEND=coreml-tdt-v3 OUT_CSV=bench/f0zg/tdt-v3.csv \
    PARAKEET_COREML_MODEL_DIR="$TDT" scripts/bench-latency.sh
BACKEND=coreml-tdt-v3 OUT_CSV=bench/f0zg/tdt-v3-cpudecode.csv \
    PARAKEET_COREML_MODEL_DIR="$TDT" \
    EXTRA_ARGS="--tdt-decode-compute-units cpu-only" scripts/bench-latency.sh

REPETITIONS=10 COREML_WORKER=target/release/parakeet-coreml-worker \
    scripts/bench-gold.sh            # runs the TDT arm when the model is present
```

`--tdt-chunk-concurrency N` raises TDT's long-form chunk parallelism for a
wall-clock arm; the per-stage rows are correctly refused at anything above 1.
`PARAKEET_COREML_TDT_V3_MODEL_DIR` overrides the TDT directory alone, so a
shell that already exports `PARAKEET_COREML_MODEL_DIR` for the shipping pack
does not have to be unset to run the challenger.

## Hold-mode baseline: M5 Pro 24 GB (2026-09-04)

Hold (press-and-hold) had no measured release-to-text number; the tables above
and the end-to-end gate both cover Tap, where Silero VAD owns the endpoint. In
Hold the hotkey release is the endpoint, so `scripts/bench-hold.sh` runs
`bench_e2e --mode hold`: it plays each fixture through the loopback device,
waits for Core Audio's predicted instant of the last audible sample, releases
there, and stops the clock at transcript-ready. Releasing at the acoustic end
is the earliest a user could, so these are floor numbers for the mode.

30 measured repetitions per bucket, two warmups, `BlackHole 2ch` loopback,
resident Core ML worker (`bench/hold.csv`). Re-measured 2026-09-04 after
ADR-0030; the 2026-09-04 pre-ADR-0030 column is kept beside it because Hold,
unlike Tap, has nothing overlapping the decode and so collects the saving in
full.

| bucket | captured audio | n | mean | p50 | p95 | p99 | p50 before |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 s | 0.885 s | 30 | 49.0 ms | **48.0 ms** | 55.0 ms | 117.5 ms | 54.0 ms |
| 3 s | 2.901 s | 30 | 56.6 ms | **55.0 ms** | 76.0 ms | 90.7 ms | 67.0 ms |
| 5 s | 4.922 s | 30 | 84.5 ms | **72.5 ms** | 137.6 ms | 141.1 ms | 106.5 ms |
| 10 s | 8.128 s | 30 | 117.7 ms | **125.0 ms** | 161.2 ms | 166.6 ms | 137.5 ms |
| 20 s | 16.587 s | 30 | 204.4 ms | **192.5 ms** | 240.0 ms | 352.9 ms | 231.5 ms |

The before column came from a run whose captured durations differ by up to
0.8 s per bucket, so treat it as a trend rather than a controlled comparison;
the controlled before/after is the ASR-only table under "Retiring the resample
stage".

Bucket labels are nominal. The captured-audio column is the median measured
duration, and it is what these latencies belong to: the "20 s" row is a 15.755 s
utterance, which is only just over the 15 s encoder window, and the "10 s" row is
8.2 s. Budget against the captured column, not the label.

Medians of the parts, from the same `phase_timer` lines:

| bucket | captured audio | release to observed | capture stop and join | ASR | total p50 |
|---|---:|---:|---:|---:|---:|
| 1 s | 0.885 s | 14.5 ms | 0.0 ms | 33.0 ms | 48.0 ms |
| 3 s | 2.901 s | 12.0 ms | 0.0 ms | 40.0 ms | 55.0 ms |
| 5 s | 4.922 s | 12.0 ms | 0.0 ms | 61.0 ms | 72.5 ms |
| 10 s | 8.128 s | 12.5 ms | 0.0 ms | 115.5 ms | 125.0 ms |
| 20 s | 16.587 s | 14.0 ms | 0.0 ms | 181.0 ms | 192.5 ms |

The capture callback itself is measured, because one that overruns its buffer
period drops audio. `AudioCapture` keeps a lock-free duration histogram and logs
`capture_callback` at stop; on the 48 kHz loopback, 461 callbacks over 4.917 s
measured mean 9.8 µs, p99 30 µs, max 103 µs against the 10.67 ms period of a
512-frame chunk. That is 0.92 ms per audio-second for the whole callback — mono
fold, level meter, filter, buffer append, channel send — all of it during
capture and none at the endpoint.

`run_manual` polled its signal channel every 15 ms in this build, which is the
12 to 14 ms median seen in the first column; ADR-0032 replaced that sleep with a
3 ms blocking read on the audio tap. Capture shutdown now rounds to 0 ms at every
bucket, where it used to cost about 1 ms: `finish_with_recording` no longer
folds the whole recording to mono, because capture did that per callback, and
all that remains after the stream is dropped is the resampler's tail flush.
Everything else is ASR, which runs slower here than in the isolated bench, not
monotonically in length, because the capture stream is still live in the same
process. In this build Hold never set `early_transcript`, so unlike Tap it could
not overlap any decode with the utterance; that is what ADR-0032 changed and
what the next section measures.

```bash
REPS=30 WARMUP_REPS=2 BACKEND=coreml-unified scripts/bench-hold.sh
```

## Hold-mode incremental windows (ADR-0032)

The baseline above is the serial path: nothing decodes until the key comes up,
so the ASR column grows with the recording. ADR-0032 cuts the held recording
into windows at Silero-confirmed pauses, and at a length cap when the speaker
does not pause, decodes each in the background while capture continues, and
joins them on the words neighbouring windows agree on. What the user waits for
on release is then the tail window plus the merge, not the whole recording.

`bench_e2e --hold-windows off|MIN,MAX` selects the path, so both columns come
from one binary on one sitting. `scripts/bench-hold.sh` passes `HOLD_WINDOWS`
straight through and defaults to the shipping `3,6`.

`bench/audio/multipause_48000.wav` is generated by `scripts/bench-hold.sh`: four
clauses separated by explicit 800 ms gaps, long enough for Silero to confirm a
pause. The `say`-generated fixtures above have at most one sentence boundary
each, so without it the pause-cut path would only ever be exercised by the
length cap. It is aggregated into its own CSV because `bench-aggregate.py`
buckets by measured duration to the nearest of {1,3,5,10,20}s and this fixture
is none of those.

```bash
REPS=30 WARMUP_REPS=2 BACKEND=coreml-unified scripts/bench-hold.sh
HOLD_WINDOWS=off RAW_LOG=bench/hold-serial.log OUT_CSV=bench/hold-serial.csv \
    REPS=30 WARMUP_REPS=2 BACKEND=coreml-unified scripts/bench-hold.sh
```

Transcript quality is a separate question from latency, and the seam merge is
where a windowed decode can lose. `asr_diff --hold-windows MIN,MAX` decodes each
fixture the windowed way from a buffer — same VAD, same planner, same merge, no
loopback device — so the gold corpus can be scored both ways:

```bash
COREML_WORKER=target/release/parakeet-coreml-worker scripts/bench-gold.sh
```

The worker must be the one built from this checkout: `token_spans` is what the
merge aligns on, and a worker built before ADR-0032 does not report them.

### Release-to-text: M5 Pro 24 GB (2026-09-04)

30 measured repetitions per bucket, two warmups, `BlackHole 2ch` loopback,
`--arm warm`. Both arms ran back to back on the same quiet machine (load average
1.40 before the windowed arm, 1.30 before the serial one), so this is a
controlled before/after rather than a comparison against the older baseline
table above. `bench/hold.csv` and `bench/hold-serial.csv`.

Three arms: windowing off, the shipping forced-cut config, and the pause-cut
config that the WER section below rejects. `bench/hold-serial.csv`,
`bench/hold-forced.csv`, `bench/hold.csv`.

| bucket | captured audio | serial p50 | shipping `6,6` p50 | pause `3,6` p50 | serial p95 | shipping p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1 s | 0.875 s | 48.5 ms | **36.0 ms** | 36.0 ms | 57.5 ms | 38.0 ms |
| 3 s | 2.891 s | 59.5 ms | **47.0 ms** | 47.0 ms | 69.8 ms | 65.5 ms |
| 5 s | 4.917 s | 64.5 ms | **58.5 ms** | 58.5 ms | 96.3 ms | 88.2 ms |
| 10 s | 8.128 s | 104.5 ms | **66.0 ms** | 63.0 ms | 146.6 ms | 76.0 ms |
| 20 s | 16.58 s | 180.5 ms | **65.0 ms** | 64.5 ms | 231.8 ms | 84.0 ms |
| multipause | 16.14 s | 177.5 ms | **62.0 ms** | 78.5 ms | 197.9 ms | 90.0 ms |

The 1, 3, and 5 s fixtures are all shorter than the 6 s cap, so no cut is
possible and the two windowed configurations are the same code path on them;
those three rows were measured once and are repeated in both columns.

The shape is the point. Serial p50 grows with the recording because the whole
recording is encoded after release. Windowed p50 stops growing after 5 s,
because what is left at release is the tail window and nothing else. The
shipping config costs 1 to 3 ms against the pause config on the two `say`
fixtures and is 16 ms faster on the multipause one, where pause cutting left a
3.69 s tail against a 2.83 s one.

The 1, 3, and 5 s rows never cut a window at all — the per-session log confirms
`windows=1` on every repetition — so their 6 to 13 ms improvement is entirely
the `run_manual` poll change, a 15 ms sleep replaced by a 3 ms blocking read on
the audio tap. That is the same 12 to 14 ms the Hold baseline section attributes
to the sleep, recovered.

### Windows and seams per session

Read out of `bench/hold.log`, which carries one `hold windows=... seams=[...]`
line per repetition beside its `phase_timer`:

At the shipping config (`bench/hold-forced.log`):

| bucket | windows | seams over 30 repetitions | tail p50 | queue wait p95 |
|---|---:|---|---:|---:|
| 1 s | 1 | none | 0.87 s | 0.0 ms |
| 3 s | 1 | none | 2.89 s | 0.0 ms |
| 5 s | 1 | none | 4.92 s | 0.0 ms |
| 10 s | 2 | 30 agreed, 30 empty | 3.61 s | 0.0 ms |
| 20 s + multipause | 4 | 180 agreed, 60 empty | 2.83 s | 0.0 ms |

240 seams, 180 of them resolved by word agreement, none duplicating or dropping
a word. The `EmptyOverlap` entries are the first seam of each session, where
there is no previous window to reconcile against. The tail never waited behind
an in-flight window at any percentile: a window closes at least a second before
release and the worker is idle again by the time the tail arrives.

At the pause config every seam is `EmptyOverlap` instead, because a pause cut's
overlap is the confirmation silence and neither window puts a word in it. That
join is a concatenation with nothing to reconcile, which is why the agreement
path only appears once cuts land mid-speech.

### WER: pause cuts lose, forced cuts are free

`asr_diff --gold bench/gold/manifest.json --repetitions 3`, three arms on the
same corpus and worker. The manifest gates on WER and CER with a zero-regression
cap against a 5.43% / 3.57% baseline.

| arm | WER | CER | exact | gate |
|---|---:|---:|---:|---|
| plain single-pass | 5.43% | 3.57% | 28.57% | PASS |
| windowed, pause cuts (`3,6`) | **6.52%** | **4.62%** | 28.57% | **FAIL** |
| windowed, forced cuts only (`6,6`) | 5.43% | 3.57% | 28.57% | PASS |

Per category, the pause arm's damage is not at the seams:

| category | fixtures | plain WER | `3,6` WER | `6,6` WER |
|---|---:|---:|---:|---:|
| commands | 5 | 12.20% | **14.63%** | 12.20% |
| numbers | 3 | 6.67% | **10.00%** | 6.67% |
| proper-nouns | 6 | 3.57% | 3.57% | 3.57% |
| long | 1 | 0.00% | 0.00% | 0.00% |
| custom-vocabulary | 2 | 27.27% | 27.27% | 27.27% |

`commands` and `numbers` are two to four second fixtures. At
`hold_window_min_seconds: 3.0` a pause inside one of them closes a window, and
the two halves decode worse than the whole did. Cutting only at the cap cannot
touch a fixture that short, and reproduces the plain transcript on every
category.

Read the `6,6` PASS honestly: six of the seven gold fixtures are under 4.3 s and
so decode as a single window in that arm, identical to plain by construction.
The one fixture long enough to be cut, `librispeech-multi` at 14.2 s, was cut and
still scored 0.00% WER. That is one fixture of real multi-window evidence, which
is why `bench/audio/multipause_48000.wav` and the 20 s bucket carry the rest.

This reproduces FluidAudio's own finding, recorded in
`UnifiedAsrManager.decodedTokens`: silence-aligned window starts measured about
1 WER point worse than a fixed stride on the 15 s offline encoder, with no
artifact benefit.

The `6,6` arm was re-run after the seam-merge fixes (seam-nearest tie-break, the
no-drop disagreement path, the two-word agreement floor) to confirm the result
still belongs to the shipped code. Every transcript and every per-fixture and
per-category score came back identical; only timing fields moved. The plain and
`3,6` rows above are from the original sitting and were not re-run — the
comparison they support is unaffected, since none of those fixtures reaches a
seam the fixes touch.
## ANE idle re-wake A/B: cold, prime, keep-alive (kata snx0)

The Neural Engine hard power-gates when idle. Published measurements put the
cold re-wake after an idle gap at tens to hundreds of milliseconds, with
per-dispatch cost climbing once the gap reaches about 100 ms (arXiv 2606.22283,
p.58-59 and p.84). Between dictations this app is idle for seconds to minutes,
so the endpoint decode may be paying that re-wake on every utterance.

`scripts/bench-idle.sh` measures four arms. They differ only in how long the
engine has gone without a dispatch when the measured decode starts:

| arm | silence before the decode |
|---|---|
| `warm` | none - repetitions run back to back |
| `cold` | the idle gap plus the recording interval |
| `prime` | the recording interval; one dispatch fires at the hotkey-down edge |
| `cadence` | at most `KEEPALIVE_MS` |

The recording interval stands in for the user speaking, and it is the whole
point of the ladder: a hotkey-down prime helps only if the engine's warmth
survives the 1 s or 5 s of talking that follows it. `bench_asr` simulates that
interval with `--record-gap-ms` (default: the fixture's own length); in
`bench_e2e --mode hold` it is real, because the fixture plays through the
loopback in the time it takes.

Arms are tagged with an `idle_arm` log marker rather than a new `phase_timer`
field, so nothing in the production timing path changes.

The shipping prime drops a second request while one is in flight rather than
queueing it. It cannot cancel the first, so a press-release short enough to end
while a prime is still running puts that endpoint decode behind one dispatch on
the worker's single pipe, about 30 to 50 ms. It is bounded at one dispatch and
only reachable on a press that found the engine cold - the case that was going
to pay a re-wake regardless.
`scripts/bench-idle.py` attributes each timed line to the marker above it,
warns about any line it cannot label, and carries `encoder_ms` from the stage
profiler alongside the latency percentiles - the encoder is the only stage on
the engine, so its delta separates a Neural Engine re-wake from a cold CPU.

```bash
# Find the cool-down knee first. If it sits well below 60 s, the matrix runs
# at that gap instead and takes minutes rather than hours.
scripts/bench-idle.sh sweep

IDLE_GAP_MS=60000 REPS=20 scripts/bench-idle.sh tap
IDLE_GAP_MS=60000 REPS=20 scripts/bench-idle.sh hold
scripts/bench-idle.sh energy
```

**The numbers in the four tables below were measured before fajz moved
resampling into the capture callbacks and before fgzt's bucketed short-window
encoders landed.** They were taken with one 15 s encoder and with the 48 kHz to
16 kHz conversion still inside the measured decode, so the absolute
milliseconds no longer describe the current path and should not be compared
against any table elsewhere in this file. Every arm within a table paid the
same conditions, so the comparisons between arms - which are what the decision
rests on - still hold. Re-run `scripts/bench-idle.sh` to refresh the absolute
figures.

One interaction to know about if bucket artifacts are installed. The prime
sends 0.5 s of silence, and `EncoderBuckets.select` routes a request to the
narrowest window that holds it, so with buckets present the prime warms the
narrowest bucket's encoder rather than the one a 5 s utterance will use. The
engine's power gate is a hardware unit and any dispatch lifts it, so the
re-wake this experiment measured should still be paid by the prime; a
per-program load cost, which a single 15 s encoder could not expose, would not
be. No bucket artifacts were installed on the machine that produced these
tables, so the worker used the stock encoder throughout and the numbers are
unaffected. Worth re-measuring once buckets ship.

### Cool-down knee: M5 Pro 24 GB (2026-09-04)

`scripts/bench-idle.sh sweep`, 1 s fixture, 8 repetitions per gap, three
warmups, `--record-gap-ms 0` so the decode follows the idle interval directly
and the measured cost is the re-wake alone. Machine at 87.6% idle, load average
6.35 (`bench/idle-sweep.csv`):

| idle gap | n | p50 | p95 | encoder p50 | encoder p95 |
|---|---:|---:|---:|---:|---:|
| 0 (back to back) | 8 | **37.0 ms** | 38.3 ms | **25.59 ms** | 26.33 ms |
| 100 ms | 8 | 37.0 ms | 40.3 ms | 25.63 ms | 29.00 ms |
| 500 ms | 8 | 38.0 ms | 39.6 ms | 26.30 ms | 29.05 ms |
| 2 s | 8 | 40.0 ms | 43.3 ms | 29.38 ms | 32.81 ms |
| 5 s | 8 | 52.5 ms | 59.6 ms | 41.15 ms | 49.67 ms |
| 10 s | 8 | 55.5 ms | 109.5 ms | 44.75 ms | 75.38 ms |
| 60 s | 8 | **63.5 ms** | 90.7 ms | **51.00 ms** | 62.13 ms |

The re-wake is real and it is on the engine. A fully cold decode costs 26.5 ms
more than a back-to-back one at p50, and the encoder - the only stage this
pipeline places on the Neural Engine - accounts for 25.4 ms of that. The other
stages are flat across the whole sweep.

The decay is gradual rather than a cliff: nothing measurable at 100 ms, about
3 ms by 2 s, half the total by 5 s, and the plateau by 10 s. That shape is what
decides whether a hotkey-down prime can work, because the prime has to survive
the user talking. The 10 s and 60 s rows are within noise of each other at
n=8, so the matrices below use a 10 s gap as fully cold.

### Tap Fast: M5 Pro 24 GB (2026-09-04)

`IDLE_GAP_MS=10000 REPS=15 scripts/bench-idle.sh tap`, three warmups, ASR
decode only. Machine at 79.4% idle, load average 4.07 (`bench/idle-tap.csv`):

| fixture | arm | n | p50 | p95 | encoder p50 |
|---|---|---:|---:|---:|---:|
| 1 s | warm | 15 | **36.0 ms** | **37.0 ms** | 25.84 ms |
| 1 s | cold | 15 | 63.0 ms | 102.5 ms | 51.91 ms |
| 1 s | prime | 15 | **39.0 ms** | **41.6 ms** | 28.52 ms |
| 1 s | cadence | 15 | 37.0 ms | 65.2 ms | 26.62 ms |
| 5 s | warm | 15 | **65.0 ms** | **71.4 ms** | 25.36 ms |
| 5 s | cold | 15 | 102.0 ms | 144.7 ms | 49.64 ms |
| 5 s | prime | 15 | **80.0 ms** | **117.3 ms** | 30.77 ms |
| 5 s | cadence | 15 | 90.0 ms | 138.2 ms | 29.17 ms |

Against cold, the hotkey-down prime removes 60.9 ms at p95 and 24.0 ms at p50
on the 1 s fixture, and 27.4 ms at p95 and 22.0 ms at p50 on the 5 s fixture.
The encoder is where it comes from: 51.9 ms cold against 28.5 ms primed at 1 s.

At 1 s the prime lands within 4.6 ms of the back-to-back floor at p95, because
only about a second passes between it and the decode. At 5 s it recovers most
but not all of the gap, which the sweep predicts: five seconds of talking is
already half the cool-down.

The cadence arm has the warmest encoder of the three treated arms at both
lengths and still loses to the prime on total latency, badly in the tail: 65.2
against 41.6 ms at p95 on the 1 s fixture. Its dispatches contend for the
worker's single pipe and for the CPU with the decode that follows them, and
that costs more than the engine warmth it buys.

### Hold: M5 Pro 24 GB (2026-09-04)

`IDLE_GAP_MS=10000 REPS=12 scripts/bench-idle.sh hold`, two warmups,
`BlackHole 2ch` loopback, release-to-transcript. Machine at 89.8% idle, load
average 2.66 (`bench/idle-hold.csv`). `bench_e2e` does not run the stage
profiler, so there is no encoder column here:

| fixture | arm | n | p50 | p95 |
|---|---|---:|---:|---:|
| 1 s | warm | 12 | 53.0 ms | 70.0 ms |
| 1 s | cold | 12 | 126.0 ms | 140.6 ms |
| 1 s | prime | 12 | **53.0 ms** | **70.0 ms** |
| 1 s | cadence | 12 | 50.0 ms | 69.6 ms |
| 5 s | warm | 12 | 122.0 ms | 160.0 ms |
| 5 s | cold | 12 | 117.0 ms | 188.2 ms |
| 5 s | prime | 12 | **94.5 ms** | **146.6 ms** |
| 5 s | cadence | 12 | 121.5 ms | 136.7 ms |

The 1 s row is the clean one and it is the largest effect measured anywhere in
this experiment: the prime removes 73.0 ms at p50 and 70.6 ms at p95, matching
the warm arm exactly.

That 73 ms is larger than the mechanism accounts for. The sweep prices the
encoder re-wake at 25.4 ms, and Tap Fast's cold penalty at 1 s is 27.0 ms at
p50, which matches it. Hold's is nearly three times that, and `bench_e2e` does
not run the stage profiler, so there is no encoder column here to attribute the
remainder to. Something else in the Hold path is also paying for the idle gap -
the capture stream, the loopback device, or CPU frequency, none of which the
isolated ASR bench exercises. The direction and the ordering are not in doubt,
and the prime removes whatever it is along with the encoder cost, but the 25.4
ms re-wake explains only about a third of the Hold number. Running
`bench_e2e` with stage timings would settle it.

Read the `warm` column here differently than in the Tap Fast table. `warm`
skips the idle gap but still plays the fixture, so each of its decodes follows
the previous one by the playback plus session setup - the same gap structure
the `prime` arm has. That is why warm and prime are identical at 1 s, and it
means the Hold `warm` column is not the back-to-back floor the Tap Fast one is.

It also explains the 5 s rows, which should not be read as a ranking. Warm
there sits about 5 s from its previous dispatch, which the sweep prices at
roughly 15 ms of encoder, so the expected cold-to-warm separation is only about
10 ms - inside the spread the p95 column shows at n=12. Cold landing below warm
at p50 is noise around a small expected difference, not a contradiction. Only
the 1 s Hold row and the Tap Fast table above carry the decision, and the
acceptance metric is prime against cold, which separates cleanly in every row
that counts.

### Cadence energy: M5 Pro 24 GB (2026-09-04)

`ENERGY_WINDOW_S=60 scripts/bench-idle.sh energy`. Two matched 60 s windows
differing only in whether the 250 ms keep-alive is dispatching, measured as the
resident worker's cumulative CPU time:

| window | worker CPU over 60 s |
|---|---:|
| idle control | **0.00 s** |
| 250 ms keep-alive | **2.41 s** |

The cadence costs 2.41 seconds of worker CPU per minute it runs - about 4% of
one core, continuously, against a genuine zero when the app is idle. At roughly
240 dispatches per window that is about 10 ms of host CPU per keep-alive.

**The Neural Engine's own power draw was not measured.** `powermetrics
--samplers ane_power` requires root and no interactive sudo was available on
this machine, so the figure above is the host-side cost only and the true cost
of the cadence is higher by whatever the engine draws to stay awake.

Results and the ship/no-ship decision are recorded in
[`../docs/asr/PERF.md`](../docs/asr/PERF.md).

## Baseline: M5 Pro 24 GB (2026-05-16, pre-§2 CoreML cache)

| length | n  | mean ms | p50 ms | p95 ms | p99 ms |
|--------|----|---------|--------|--------|--------|
| 1 s    | 30 | 121     | 121    | 136    | 146    |
| 3 s    | 30 | 229     | 227    | 237    | 263    |
| 5 s    | 30 | 364     | **362**| 376    | 405    |
| 10 s   | 30 | 573     | 572    | 589    | 591    |
| 20 s   | 30 | 1120    | 1114   | 1162   | 1185   |

**Steady-state RTFx** ≈ 13–14× real time on the 5 s bucket. This historical
sherpa baseline is retained for comparison; the shipping native Core ML and
end-to-end measurements above supersede it as the current performance record.

**Implied total post-endpoint latency on 5 s (pre-cache):**
362 ms ASR + 150 ms VAD ≈ **512 ms** — under the former 700 ms acceptance
target before the native worker and speculative-decode work landed.

## §6 Phase-0 polish-backend bench: Qwen 3.5 2B Q4_K_M (2026-05-16, M5 Pro 24 GB)

Driven by `src/bin/bench_llm.rs`. 100 polish iterations of a fixed
240-char noisy transcript through `llama-cpp-2` (Metal feature)
loading `unsloth/Qwen3.5-2B-Q4_K_M.gguf`. Output: 55 tokens cleaned.

| Metric | Mean | p50 | p95 | p99 |
|--------|------|-----|-----|-----|
| TTFT (ms) | 2.0 | 2.0 | 2.0 | 2.0 |
| Generation (ms) | 548 | 548 | 558 | 567 |
| Total per polish (ms) | 551 | **550** | 560 | 570 |
| Decode (tokens/sec) | 100.3 | 100.4 | 101.7 | 101.9 |

Cold model load: 229 ms. p99/p50 = 1.04 (variance negligible).

Replay:

```bash
./target/release/bench_llm \
    --model ~/Library/Application\ Support/com.parakeet.rs/llm/qwen3.5-2b-q4_k_m/Qwen3.5-2B-Q4_K_M.gguf \
    --reps 100 --warmup-reps 3 2> bench/llm-raw.log
# then aggregate inline — see ADR-0018 for the one-shot Python snippet
```

Background and library-selection rationale: [ADR-0018](../docs/ADR.md#0018--polish-backend-llamacpp--qwen-35-2b-q4_k_m).

## §6 follow-up: Qwen 3.5 4B Q6_K (2026-06-11, M5 Pro 24 GB) — shipped

The 2B's instruction-following misses (paraphrasing, over-deleted
"like", fumbled `scratch that`) motivated a bump to **Qwen3.5-4B at
Q6_K** (3.53 GB) — same family, so the ChatML + `/no_think` template
carries over unchanged. See the ADR-0018 amendment.

Enable Polish once in the app to download and SHA-256 verify the pinned GGUF;
the replay command below uses that standard application path.

30 reps, same 240-char sample transcript, same `bench_llm` harness:

| Metric | Mean | p50 | p95 |
|--------|------|-----|-----|
| TTFT (ms) | 40 | 29 | 33 |
| Generation (ms) | 1200 | 1197 | 1231 |
| Total per polish (ms) | 1240 | **1225** | 1262 |
| Decode (tokens/sec) | 43.3 | 43.4 | 43.8 |

vs the 2B: total p50 550 ms → 1225 ms (2.2×), decode 100 → 43 tok/s.
Streaming paste (ADR-0019) absorbs the difference — perceived latency
is time-to-first-words (TTFT 29 ms + first chunks), not last-token.
No truncations at the 768-token output cap across the run.

Replay:

```bash
./target/release/bench_llm \
    --model ~/Library/Application\ Support/com.parakeet.rs/llm/qwen3.5-4b-q6_k/Qwen3.5-4B-Q6_K.gguf \
    --reps 30 --warmup-reps 3 2> bench/llm-4b-raw.log
```

## §6 follow-up: polish latency and quality on an eval set (2026-09-04, M5 Pro 24 GB)

Kata 0tpp. Everything above measures **one** transcript — the hardcoded
`SAMPLE_INPUT`, 55 output tokens at the model's decode rate. That is the
structural bound, not a p50 over anything a user dictates, and it is
what the "1000 ms NOT MET" verdict was recorded against.

`bench/polish/eval.json` is a 26-item eval set with the polished text
each item should produce. `bench_llm --eval` runs it, scores word-level
error rate against `expected` (casing and punctuation included — those
are two of the three things polish exists to fix), and reports latency
and quality over the same population.

Machine load before each run: `Load Avg` 1.0–2.3, CPU ≥ 87 % idle.
3 repetitions × 26 items = 78 measured decodes per row. Repetitions are
low deliberately: within-item variance is negligible (§6's p99/p50 =
1.04), so the percentiles are driven by *which item* it is, not by
run-to-run noise. `p99` over 78 samples is a single item and is not
quoted below.

**Skipped items enter the percentiles as 0 ms.** The `skip < 4 words`
row bypasses the model for three items and records them at zero, which
is the latency the user experiences. It is not a decode time, and that
row's distribution is therefore not the same shape as the others.

> ### Quality-column correction (2026-09-04)
>
> The WER column below was first measured against `eval.json` schema 1
> with a whitespace-splitting scorer. Both had defects, found in review:
>
> - **`technical` was measured wrong.** `technical-01` and `technical-03`
>   expected spoken-to-written conversion (`one point zero point two one
>   nine` → `1.0.219`, `colon colon` → `::`) that system-prompt rule 7
>   explicitly forbids — "Preserve technical terms, names, and code-like
>   fragments exactly as transcribed". They penalised the model for
>   obeying the prompt. Schema 2 derives their expected text from the
>   prompt rules alone. The category's WER falls **0.847 → 0.091**.
> - **`command` was scored blind.** The scorer split on whitespace, so a
>   model that emitted a space where `new paragraph` required a line
>   break scored zero errors. It now tokenises breaks. The category's
>   WER rises **0.000 → 0.059**, exposing a real defect: on
>   `command-01` and `command-04` the 4B produced
>   `Ship the parts Monday. Invoice follows separately.` — correct
>   punctuation, no line break.
>
> Inputs did not change, so **every latency number below is unaffected**.
> Only the 4B row's quality has been re-measured under schema 2
> (blended **WER 0.061, 18/26 exact**, recomputed from the unchanged
> categories plus a deterministic re-run of the two corrected ones).
> The 2B, 0.8B, and edits-only rows still carry schema-1 quality and are
> **not comparable to it**; re-measuring them needs another bench turn.

| Variant | p50 | p95 | `legacy-bench-sample` | mean WER | exact | mean out tokens |
|---|---|---|---|---|---|---|
| **4B Q6_K full-text (shipping)** | **444 ms** | 1219 ms | 1209 ms | **0.139** | 17/26 | 18.7 |
| 4B full-text + skip < 4 words | 446 ms | 1222 ms | 1221 ms | 0.139 | 17/26 | 18.3 |
| 4B full-text + context reuse | 424 ms | 1220 ms | 1198 ms | 0.139 | 17/26 | 18.7 |
| 4B edits-only | 600 ms | 2438 ms | 2421 ms | 0.321 | 9/26 | 26.3 |
| 2B Q6_K full-text | 230 ms | 572 ms | 571 ms | 0.185 | 13/26 | 20.5 |
| 0.8B Q6_K full-text | 141 ms | 338 ms | 330 ms | 0.249 | 11/26 | 21.6 |

**The shipping configuration meets the <1000 ms p50 target on the eval
set.** The `legacy-bench-sample` column shows why the two verdicts
differ: that one item costs 1209 ms in the same run whose p50 is 444 ms.

### Composition caveat — read this before quoting the p50

The eval set's composition was a judgement call, and composition
determines the blended p50. Nine of 26 items are zero-change by
construction. Per-category numbers are published so the blended figure
can be re-weighted against a different view of what real dictation looks
like:

| Category | n | 4B p50 | 4B max | 4B WER | 2B p50 | 2B WER |
|---|---|---|---|---|---|---|
| clean | 6 | 429 ms | 514 ms | 0.000 | 197 ms | 0.000 |
| short | 3 | 299 ms | 326 ms | 0.000 | 147 ms | 0.000 |
| filler-light | 6 | 446 ms | 492 ms | 0.042 | 212 ms | 0.173 |
| command | 4 | 439 ms | 503 ms | 0.000 | 251 ms | 0.211 |
| technical | 3 | 609 ms | 652 ms | 0.847 | 298 ms | 0.681 |
| filler-heavy | 3 | 857 ms | 1219 ms | 0.257 | 391 ms | 0.258 |
| long | 1 | 2812 ms | 2812 ms | 0.060 | 1400 ms | 0.103 |

(WER columns here are schema 1; see the correction box above. Latency is
unaffected.)

Latency tracks output length almost exactly; quality does not.

**Retracted:** an earlier revision of this section claimed the
`technical` category showed "spoken version numbers and identifiers are
where polish does real damage". That was wrong, and backwards. Re-run
with `--show-output`, the 4B **preserves** identifiers exactly as rule 7
instructs:

```
input    : Um, bump serde to one point zero point two one nine in Cargo dot toml.
produced : Bump serde to one point zero point two one nine in Cargo dot toml.
```

Filler removed, casing fixed, identifier untouched — correct on every
count. The 0.847 was my eval set demanding a conversion the prompt
forbids. Whether polish damages identifiers is **unmeasured**: no item
in this set tests it, because the prompt tells the model not to touch
them.

The genuine defects the corrected metrics surface are smaller and
different: the 4B leaves `and, you know,` in `technical-03`, and ignores
the `new line` / `new paragraph` commands in two of four `command`
items.

### Findings per avenue

- **Edits-only is worse on both axes.** The model answers with the
  corrected transcript instead of an edit list in 17 of 26 items, on the
  4B, with an explicit prohibition and a worked example in the prompt.
  The arithmetic does not favour it even when it works: full-text
  averages 18.7 output tokens, an `OLD ==> NEW` line costs 5–8 tokens
  per fix because `OLD` must carry enough context to be unique, and the
  variant measured 26.3. A GBNF grammar (`LlamaSampler::grammar`, and
  the `sampler` feature is already on) would force the `==>` structure
  but cannot stop `<whole input> ==> <corrected>`, which parses cleanly
  and doubles the tokens.
- **Prompt caching is unavailable on this model family.** Qwen 3.5 is a
  hybrid Gated-DeltaNet architecture (ADR-0018); its recurrent state
  carries no per-token position, so `llama_memory_seq_rm` refuses
  partial sequence removal. `mean_reused_prompt_tokens=0.0` on every
  run. The same refused operation is what draft-model speculative
  decoding would need for rollback, so that avenue is blocked too. What
  the `--prompt-cache` row actually measures is **context reuse** — not
  re-allocating a `LlamaContext` per call — worth 20 ms (444 → 424),
  consistent with the 29 ms TTFT bounding it.
- **Skipping short utterances helps the mean, not the p50.** It fires on
  3/26 items with identical quality, and moves the mean 591 → 556 ms.
  The p50 is unchanged (444 → 446) because the items it removes were
  already the fastest ones; taking three items off the bottom does not
  move the middle.
- **Neither smaller model is a free swap.** The 2B cuts p95 from 1219 to
  572 ms and the structural-bound item from 1209 to 571 ms, at WER 0.185
  vs 0.139 and 13/26 vs 17/26 exact. It is a tail fix bought with
  quality. The 0.8B is faster still and clearly worse.

Fixed-sample replays for cross-checking against §6 above (15 reps):
4B Q6_K **p50 1202 ms** (§6 recorded 1225), 2B Q6_K p50 565 ms.

Replay:

```bash
LLM="$HOME/Library/Application Support/com.parakeet.rs/llm"
./target/release/bench_llm \
    --model "$LLM/qwen3.5-4b-q6_k/Qwen3.5-4B-Q6_K.gguf" \
    --eval bench/polish/eval.json --variant full-text \
    --reps 3 --tag qwen3.5-4b-q6_k --csv bench/polish/variants.csv \
    2> bench/llm-4b-fulltext-raw.log
```

Swap `--variant edits-only`, `--skip-min-words 4`, `--prompt-cache`, or
`--model` for the other rows.

## Files

| Path                         | Purpose                                          |
|------------------------------|--------------------------------------------------|
| `audio/{1,3,5,10,20}s_*.wav` | Generated fixtures (gitignored). Filename includes sample rate (e.g. `5s_16000.wav`). |
| `raw.log`                    | All `phase_timer` lines from the last ASR run.   |
| `gold/manifest.json`       | Shipping real-speech quality manifest and thresholds. |
| `gold/sources.json`        | Immutable source provenance, licenses, and hashes. |
| `gold.example.json`        | Historical `say` smoke-example schema. |
| `asr-quality.json`          | Generated machine-readable quality/latency report (gitignored). |
| `llm-raw.log`                | All `llm_timer` lines from the last LLM run.     |
| `baseline.csv`               | Aggregated ASR baseline (pre-CoreML-cache).      |
| `coreml-unified.csv`         | Generated shipping-backend ASR percentiles.     |
| `*-boundary.csv`             | Generated Rust/worker boundary measurements.    |
| `*-stages.csv`               | Generated per-stage breakdown and Core ML dispatch counts. |
| `coreml-unified-buckets*.csv` | Generated bucketed short-window encoder runs. |
| `hold.{log,csv}`             | Generated Hold-mode release-to-transcript runs. |
| `hold-multipause.{log,csv}`  | Generated Hold-mode runs on the four-clause pause fixture. |
| `hold-serial*.{log,csv}`     | Generated Hold-mode runs with windowing off (the before arm). |
| `hold-forced.{log,csv}`      | Generated Hold-mode runs at the shipping forced-cut config. |
| `asr-quality-windowed-*.json` | Generated gold reports for the windowed decode arms. |
| `idle-*.{log,csv}`           | Generated ANE idle re-wake A/B runs (sweep, tap, hold, energy). |
| `e2e-*.{log,csv}`            | Generated serial/speculative production-path runs. |
| `endpoint-*.{log,csv}`       | Generated pause-friendly endpoint gate runs.   |
| `endpoint-sweep*.{csv,/}`    | Generated confirmation-window sweep rows and logs. |
| `polish-backends.csv`        | Historical §6 Phase-0 2B polish measurements.  |
| `polish/eval.json`           | Polish quality eval set: 26 transcripts with expected output. |
| `polish/variants.csv`        | Generated per-variant polish latency/quality summary. |
| `llm-*-raw.log`              | Generated `llm_timer` / `llm_eval_*` lines per polish variant run. |
