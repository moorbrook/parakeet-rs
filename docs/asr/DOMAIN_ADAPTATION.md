# Domain adaptation and hardware specialization

## Decision

Do not train, distill, or quantization-aware-train a new model from the current
evidence. Keep the generic int8 Parakeet Unified Core ML model as the shipping
default. Contextual vocabulary runs inside the native worker with score 2.0.
Before changing that score, run the gold corpus with and without vocabulary.
The small diagnostic corpus does not justify training or a global score change.

## What the generic model gets wrong

The frozen shipping Core ML baseline is 5/92 word edits (5.43% WER) and 17/476
character edits (3.57% CER), with zero output spread over ten runs. Its three
lexical failures are:

| class | repeated evidence | shipping output | interpretation |
|---|---|---|---|
| custom names/terms | 2 fixtures, 3 word edits | `IBM` → `IPM`; `Olly` → `Ollie`; `from` → `for` | the only repeated lexical class, but just two examples |
| spoken-number form | 1 fixture, 2 word insertions | `seven thirty AM` → `seven hundred and thirty AM` | semantic form preference, not a hardware failure |
| formatting | 5 fixtures are not exact; 2 differ only in formatting | terminal punctuation and comma differences | WER/CER intentionally normalize punctuation; exact-match remains separate |

The noisy `Amy` fixture and both general-speech fixtures are lexically exact on
the shipping backend. That matters: an intervention cannot trade those wins
away to repair a name.

## Locked gate before any new collection

No training data was collected in this issue. Before a future collection
starts, its protocol and scorecard must be committed. The minimum protocol is:

1. Define one repeated target class from real opt-in error reports. A global
   domain adapter needs at least 25 reviewed errors across at least five
   speakers. A personal adapter needs at least 25 reviewed errors across at
   least five recording sessions. Until then, use removable vocabulary rules.
2. Split before optimization. Global/domain work separates speakers across
   train, development, and blind test. Personal work separates recording
   sessions by time and retains a multi-speaker generic safety suite.
3. Seal the blind test transcripts and hashes before training. Training,
   score selection, early stopping, calibration, QAT, and prompt/rule selection
   may use only train/development data.
4. Keep consent, retention, deletion, and provenance beside every recording.
   Raw user audio stays local by default; derived examples remain deletable.

The seven current fixtures were used to choose vocabulary transition points,
so they are now diagnostic/development evidence for that choice—not a fresh
blind test for a future adapter. They must never enter training, calibration,
distillation, or QAT.

A candidate graduates only if a newly sealed test set satisfies all of:

- overall WER ≤ 5.4347826% and CER ≤ 3.5714286%;
- no WER or CER regression in any frozen category relative to
  `bench/qwen3-asr/shipping-coreml-baseline.json`;
- at least 20% relative error reduction in the declared target class, with a
  paired 95% bootstrap interval whose lower bound is above zero;
- zero nondeterministic outputs in ten repeated passes;
- corpus p50 and p95 no more than 5% slower and peak RSS no more than 1.25×
  the shipping Core ML baseline.

For QAT or distillation, the report must additionally pin every artifact and
record compiled artifact bytes, peak process-tree RSS, load/first-result/p50/
p95 latency, overall and per-category WER/CER, and repeatability. It must reduce
at least one targeted deployment resource by 10% without a quality regression.
There is no QAT/distillation result in this issue because the evidence did not
authorize either experiment.

## Which layer owns adaptation

“Fine-tune for this Mac” combines distinct concerns that should not share an
identity or lifecycle:

| layer | scope | hardware-specific? | removal/fallback |
|---|---|---|---|
| generic acoustic/language model | all users | no | immutable shipping baseline |
| domain adapter | product, organization, or vocabulary domain | no; train on speech/domain evidence | remove adapter and use generic model |
| personal adapter | one consenting user | no; user identity is not a chip property | delete local adapter/data and use generic model |
| QAT/distilled export | deployment hardware family and precision budget | yes, at export/package level | retain full-quality generic artifact |
| Core ML runtime plan | exact chip, OS, model digest, and workload | yes | existing tuner falls back to CPU+ANE |

The semantic adapter should remain portable across Macs. A separate export may
quantize or compile that model for an Apple hardware family, and the existing
runtime tuner may select execution for the exact machine. This preserves the
useful “generic model plus local specialization” idea without pretending that
a speaker's vocabulary is an M5-specific weight.

## Reproduction

`COREML_WORKER=target/release/parakeet-coreml-worker scripts/bench-gold.sh`
records the native baseline and vocabulary rows. Reports include decoder
method, requested and accepted vocabulary counts, score, source digest, model
identity, hardware, quality, latency, and memory.
