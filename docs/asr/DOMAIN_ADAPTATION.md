# Domain adaptation and hardware specialization

## Decision

Do not train, distill, or quantization-aware-train a new model from the current
evidence. Keep the generic int8 Parakeet Unified Core ML model as the shipping
default. Do not change the global sherpa `hotword_score` from 2.0.

The smallest measured intervention was already in the product: sherpa
contextual vocabulary. A 27-point exploratory score sweep followed by six
ten-repeat measurements found no score that improves a repeated vocabulary
class without worsening another category or CER. Because the existing corpus
contains only seven fixtures and the remaining errors are heterogeneous,
neural training would be fitting anecdotes rather than a demonstrated error
distribution.

The next intervention, if new data shows the same class repeatedly, should be
a constrained vocabulary/lexical-rescoring layer on the native Core ML path.
It is smaller, removable, and easier to gate than an adapter. Learned weights
come only after that route fails on a newly separated development set.

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

## Contextual-vocabulary sweep

All rows use the int8 sherpa fallback because the native Core ML backend does
not expose contextual decoding. Each checked row is a release build, one
warmup, and ten corpus repetitions, run serially on an M5 Pro. All produced one
unique transcript per fixture. The complete per-category rows and provenance
are in `bench/domain-adaptation/raw/`.

| decoder | WER | CER | custom-vocabulary WER / CER | p50 / p95 | p50 cost vs greedy | result |
|---|---:|---:|---:|---:|---:|---|
| greedy | 10.87% | 5.46% | 54.55% / 12.50% | 2.843 / 2.923 s | — | fallback baseline; fails shipping gate |
| modified beam, score 0 | 10.87% | 5.46% | 54.55% / 12.50% | 3.219 / 3.249 s | +13.2% | search cost with no lexical effect |
| modified beam, score 2 | 10.87% | 5.46% | 54.55% / 12.50% | 3.237 / 3.287 s | +13.9% | current default; no lexical effect |
| modified beam, score 2.75 | 8.70% | 6.09% | 27.27% / 12.50% | 3.245 / 3.314 s | +14.2% | first effect; WER improves, CER/categories regress |
| modified beam, score 4.5 | 11.96% | 8.40% | 36.36% / 23.21% | 3.261 / 3.309 s | +14.7% | first `Olly`; also injects `IBM` elsewhere |
| modified beam, score 6 | 42.39% | 33.82% | 45.45% / 28.57% | 3.324 / 3.387 s | +16.9% | broad vocabulary injection |

The first-effect score changes `It's I B M up today` to `IBM up today`, but
also changes the noisy phrase ending `with Amy` to `with 80`. Its per-category
delta against greedy makes the trade explicit; negative is better:

| category | Δ WER points | Δ CER points | gate |
|---|---:|---:|---|
| commands | -4.88 | +1.50 | fail |
| custom vocabulary | -27.27 | 0.00 | WER win only |
| general | 0.00 | 0.00 | tie |
| long | 0.00 | 0.00 | tie |
| noisy | +9.09 | +5.56 | fail |
| numbers | +3.33 | +2.08 | fail |
| proper nouns | -2.38 | +0.68 | fail |
| punctuation-tagged corpus | -2.17 | +0.63 | fail |

The exploratory sweep covered scores 0, 0.5, 1, 2, every 0.25 from 2.25
through 5.75, then 6, 8, 10, 12, 15, 20, 30, and 50. Scores through 2.5 were
identical to greedy. Scores 2.75–4.25 shared the same `IBM` win/`Amy` loss.
Scores 8–50 measured 96.74–119.57% WER. There is no missing safe interval
among the sampled points; the two transition regions are bracketed to 0.25,
which is sufficient to reject a global change on this already-tuned diagnostic
set, not to claim every real-valued score has been exhaustively evaluated.

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

The compact summary, raw reports, source/input digests, and verifier live in
`bench/domain-adaptation/`:

```bash
uv run scripts/verify-domain-adaptation-evidence.py
```

`asr_diff` report schema v3 makes each result self-describing: decoder method,
whether contextual vocabulary was requested and active, score, term count,
source vocabulary SHA-256, and generated hotword SHA-256 are embedded with the
existing model, hardware, quality, latency, and memory metadata.
