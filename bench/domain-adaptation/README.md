# Domain-adaptation decision evidence

`m5-pro-2026-08-11.json` is the machine-verifiable decision record for issue
#6. `raw/` retains the greedy baseline and five vocabulary-score candidates,
each measured ten times on the same M5 Pro. Report schema v3 embeds the decoder
method, requested and active vocabulary state, score, term count, source
vocabulary digest, and generated sherpa-hotword digest.

The 27-point exploratory sweep located transition boundaries; its compact
score list and observations are retained in the summary. The six repeated raw
reports—not the exploratory single runs—are authoritative. The sweep used the
seven existing gold fixtures, so they are now a diagnostic set for vocabulary
selection and must not be presented as a fresh blind test of a future adapter.
They also remain prohibited from training, calibration, QAT, and distillation.

Verify every digest, copied metric, category score, transcript reference,
decoder setting, and the no-training decision with:

```bash
uv run scripts/verify-domain-adaptation-evidence.py
```

Replay one point after building `asr_diff`:

```bash
cargo build --release --locked --bin asr_diff
target/release/asr_diff \
  --gold bench/gold/manifest.json --audio-dir bench/gold/audio \
  --backend sherpa --repetitions 10 \
  --vocabulary bench/gold/vocabulary.txt --hotword-score 2.75 \
  --json-out bench/domain-adaptation/raw/sherpa-score-2.75.json
```

The expected exit status is 1: all sherpa rows fail the frozen shipping
quality gate, and the report is written before that failure is returned.
Interpretation, future data separation, and adapter ownership are documented
in [`docs/asr/DOMAIN_ADAPTATION.md`](../../docs/asr/DOMAIN_ADAPTATION.md).
