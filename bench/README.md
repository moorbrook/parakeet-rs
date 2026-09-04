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
12 to 14 ms median seen in the first column; ADR-0031 replaced that sleep with a
3 ms blocking read on the audio tap. Capture shutdown now rounds to 0 ms at every
bucket, where it used to cost about 1 ms: `finish_with_recording` no longer
folds the whole recording to mono, because capture did that per callback, and
all that remains after the stream is dropped is the resampler's tail flush.
Everything else is ASR, which runs slower here than in the isolated bench, not
monotonically in length, because the capture stream is still live in the same
process. In this build Hold never set `early_transcript`, so unlike Tap it could
not overlap any decode with the utterance; that is what ADR-0031 changed and
what the next section measures.

```bash
REPS=30 WARMUP_REPS=2 BACKEND=coreml-unified scripts/bench-hold.sh
```

## Hold-mode incremental windows (ADR-0031)

The baseline above is the serial path: nothing decodes until the key comes up,
so the ASR column grows with the recording. ADR-0031 cuts the held recording
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
merge aligns on, and a worker built before ADR-0031 does not report them.

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
| `hold.{log,csv}`             | Generated Hold-mode release-to-transcript runs. |
| `hold-multipause.{log,csv}`  | Generated Hold-mode runs on the four-clause pause fixture. |
| `e2e-*.{log,csv}`            | Generated serial/speculative production-path runs. |
| `endpoint-*.{log,csv}`       | Generated pause-friendly endpoint gate runs.   |
| `polish-backends.csv`        | Historical §6 Phase-0 2B polish measurements.  |
