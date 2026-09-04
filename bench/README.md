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
- the Silero VAD endpoint policy (750 ms for Tap; 150 ms for Tap Fast)
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

Tap does not collect the resample saving, and the `phase_timer` lines say why:
`t_asr_start=4966`, `t_asr_done=5013`, `t_vad_endpoint=5094`. The speculative
decode finishes about 80 ms before the endpoint policy confirms, and
`dur_post_endpoint_ms` is 0 to 1 ms, so this path is endpoint-bound rather than
decode-bound. Taking 22 ms out of the decode widens that margin instead of
shortening the result. The saving lands where the decode is not hidden: Hold,
the serial fallback, and every utterance long enough that the decode would
otherwise outrun the confirmation window. This frozen comparison explicitly uses
Tap Fast's original 150 ms policy so the historical 3× result stays
like-for-like. The gate fails unless both p50 and p95 are at least 3.0× and
every transcript matches:

```bash
REPS=30 WARMUP_REPS=2 scripts/bench-end-to-end.sh
```

## Long-pause endpoint gate

Normal Tap now uses a 750 ms confirmation policy; Tap Fast retains 150 ms for
short commands. The separate endpoint gate replays versioned human LibriSpeech
audio through production capture, VAD, speculative Core ML inference, and
session shutdown. Its 14.225 s fixture includes a reviewed 544 ms natural
pause that the former policy cut. A pass requires zero early stops and p95
final-pause latency below one second for both the single- and multi-sentence
fixtures:

```bash
REPS=30 WARMUP_REPS=2 scripts/bench-endpoint-policy.sh
```

M5 Pro 24 GB release results (2026-08-11):

| fixture | repetitions | false stops | p50 | p95 |
|---|---:|---:|---:|---:|
| 3.505 s single sentence | 30 | **0** | 668.0 ms | 668.0 ms |
| 14.225 s multi sentence | 30 | **0** | 637.0 ms | 658.1 ms |

The unchanged Tap Fast comparison was also re-run for 30 repetitions after
this policy split. It retained **3.24× p50 / 3.18× p95** speedups (589.5 →
182.0 ms p50; 644.8 → 203.0 ms p95), so the representative no-polish gate
remains above its accepted 3× target.

The fixture manifest, source revision, hashes, references, and license are in
[`bench/endpointing/`](endpointing/). This gate isolates endpoint behavior;
transcript WER/CER remains the responsibility of `asr_diff`.

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
`UnifiedRnntDecoder`, which are a pinned dependency this project depends on
rather than vendors. The worker therefore measures from outside: at startup it
replaces the prediction implementations of `MLModel` and its registered
subclasses with timing wrappers that call straight through, and attributes each
dispatch to a stage by the input feature names FluidAudio's providers declare
(`mel` for the encoder, `targets` for the decoder, `encoder_step` for the
joint). Stage boundaries come from the resulting dispatch timeline: mel is the
gap before an encoder dispatch, the RNNT loop is everything from an encoder
dispatch to the last dispatch of that window. Nothing in the pinned package is
patched, and the shipping dictation path never installs the wrappers.

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

`run_manual` polls its signal channel every 15 ms, which is the 12 to 14 ms
median seen in the first column. Capture shutdown now rounds to 0 ms at every
bucket, where it used to cost about 1 ms: `finish_with_recording` no longer
folds the whole recording to mono, because capture did that per callback, and
all that remains after the stream is dropped is the resampler's tail flush.
Everything else is ASR, which runs slower here than in the isolated bench, not
monotonically in length, because the capture stream is still live in the same
process. Hold also never sets `early_transcript`, so unlike Tap it
cannot overlap any decode with the tail of the utterance.

```bash
REPS=30 WARMUP_REPS=2 BACKEND=coreml-unified scripts/bench-hold.sh
```

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
| `idle-*.{log,csv}`           | Generated ANE idle re-wake A/B runs (sweep, tap, hold, energy). |
| `e2e-*.{log,csv}`            | Generated serial/speculative production-path runs. |
| `endpoint-*.{log,csv}`       | Generated pause-friendly endpoint gate runs.   |
| `polish-backends.csv`        | Historical §6 Phase-0 2B polish measurements.  |
| `polish/eval.json`           | Polish quality eval set: 26 transcripts with expected output. |
| `polish/variants.csv`        | Generated per-variant polish latency/quality summary. |
| `llm-*-raw.log`              | Generated `llm_timer` / `llm_eval_*` lines per polish variant run. |
