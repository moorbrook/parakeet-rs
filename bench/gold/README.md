# Representative ASR gold corpus

This directory is the checked-in, offline quality corpus used by `asr_diff`.
It contains seven real human recordings: two longer LibriSpeech utterances and
five SLURP commands. The command set includes a matched close/distant recording
and covers names, spoken numbers, an acronym, and custom-vocabulary terms.
Although SLURP names its upstream partition `train`, these exact selected rows
are evaluation-only in this project and must not enter fine-tuning, calibration,
quantization-aware training, prompt selection, or threshold-tuning data.
Issue #6's vocabulary-score sweep also used them to locate decoder transition
points, so they are diagnostic/development evidence for that setting and are
not a fresh blind test for any future adaptation candidate.

The manifest references are human-reviewed display text. Lexical WER/CER
ignores case and punctuation according to `asr_eval::normalize_lexical`; raw
strings remain in reports so formatting differences are still visible.

The absolute product limits are 8% WER and 5% CER. The recorded shipping
baseline is 5/92 word edits and 17/476 character edits. Ten repeated passes
produced zero transcript, WER, or CER spread, so the independent regression
tolerance is zero: a new run may improve the baseline but may not add an error.
See `docs/asr/PERF.md` for the complete derivation and A/B.

`sources.json` pins corpus revisions and both source/final hashes. Run
`uv run scripts/fetch-gold-corpus.py` to re-fetch the five SLURP files, verify
the selected rows and hashes, and reproduce their 48 kHz PCM WAV conversion.
The two LibriSpeech WAVs are shared byte-for-byte with `bench/endpointing/`.

The SLURP audio is CC BY-NC 4.0, is present only as a test asset, and is not
bundled into Parakeet.app. See `THIRD_PARTY_NOTICES.md` for attribution.
