# Third-party notices

## FluidAudio

The native Core ML speech-recognition worker links FluidAudio at commit
`00a9aa771900ea09c485659663be31019e293e47`.

- Project: <https://github.com/FluidInference/FluidAudio>
- Copyright: FluidInference contributors
- License: Apache License 2.0; see `LICENSE-APACHE`

## Parakeet Unified EN 0.6B Core ML model

The optimized backend downloads the `FluidInference/parakeet-unified-en-0.6b-coreml`
model on demand. The weights are not stored in this repository or bundled in the
application.

- Model (exact revision): <https://huggingface.co/FluidInference/parakeet-unified-en-0.6b-coreml/tree/4252711f6f060f9a2f91e5f081a806d7f45eebd8>
- Core ML conversion and integration: FluidInference contributors
- Based on NVIDIA Parakeet
- Model license: Creative Commons Attribution 4.0 International
  (<https://creativecommons.org/licenses/by/4.0/>)

## LibriSpeech endpoint regression fixtures

The two human-speech WAV files under `bench/endpointing/` come from the
LibriSpeech `test.clean` split (OpenSLR SLR12), prepared by Vassil Panayotov,
Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur from LibriVox recordings.
They are test assets only and are not bundled in Parakeet.app.

- Corpus: <https://www.openslr.org/12>
- Mirror and exact source revision: <https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1>
- Source utterance IDs: `6930-75918-0000`, `6930-75918-0001`
- License: Creative Commons Attribution 4.0 International
  (<https://creativecommons.org/licenses/by/4.0/>)

## SLURP gold-corpus fixtures

The five human-speech WAV files whose names begin with `slurp-` under
`bench/gold/audio/` come from the real-audio train split of SLURP by Emanuele
Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser. They are test
assets only and are not bundled in Parakeet.app.

- Corpus: <https://github.com/pswietojanski/slurp>
- Official annotation revision: <https://github.com/pswietojanski/slurp/tree/8eb16545762be97ace75334109d73824217311f1>
- Audio mirror and exact revision: <https://huggingface.co/datasets/qmeeus/slurp/tree/91b0abfee2e735282967ee00d631d6d5f0fb7ff9>
- Source IDs, row numbers, capture types, and hashes: `bench/gold/sources.json`
- Text license: Creative Commons Attribution 4.0 International
- Audio license: Creative Commons Attribution-NonCommercial 4.0 International
  (<https://creativecommons.org/licenses/by-nc/4.0/>)
