# Contextual-biasing evidence (ADR-0033)

Raw reports behind the tables in `bench/README.md` and ADR-0033. M5 Pro, 24 GB,
macOS 26.5.1, release worker and binaries built from the branch that introduced
the biasing hook.

| file | what produced it |
|---|---|
| `coreml-gold-quality.json` | `REPETITIONS=10 scripts/bench-gold.sh`, Core ML, no vocabulary |
| `coreml-vocabulary-gold-quality.json` | same run, Core ML biased toward `bench/gold/vocabulary.txt` at score 2.0 |
| `sherpa-gold-quality.json` | same run, sherpa greedy |
| `sherpa-vocabulary-gold-quality.json` | same run, sherpa `modified_beam_search` with the generated hotwords file at score 2.0 |
| `coreml-unified-p94q-base*.csv` | `BACKEND=coreml-unified scripts/bench-latency.sh`, 30 repetitions |
| `coreml-unified-p94q-vocab50*.csv` | same, `VOCABULARY=bench/gold/vocabulary-50.txt HOTWORD_SCORE=2.0` |

`COREML_WORKER=target/release/parakeet-coreml-worker` was set for the gold run.
Without it `bench-gold.sh` uses the installed app's worker, which predates the
`PRKV` vocabulary frame: the unbiased rows would measure stale code and the
biased row would fail on the magic check.
