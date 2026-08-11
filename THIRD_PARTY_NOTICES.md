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

## Parakeet TDT 0.6B v3 int8 fallback model

The fallback and contextual-vocabulary backend downloads the sherpa-onnx ONNX
conversion of NVIDIA Parakeet TDT 0.6B v3. The weights are not stored in this
repository or bundled in the application.

- Model (exact revision): <https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tree/2bda32ec70b097a55adaa07d9a7173915b43cc78>
- Original model: NVIDIA Parakeet TDT 0.6B v3
- Conversion: sherpa-onnx contributors
- Model license: Creative Commons Attribution 4.0 International
  (<https://creativecommons.org/licenses/by/4.0/>)

## Silero VAD

Parakeet downloads the Silero voice-activity detector distributed with the
sherpa-onnx ASR model releases. The exact downloaded bytes are pinned in
`src/model_fetch.rs` by length and SHA-256.

- Upstream project: <https://github.com/snakers4/silero-vad>
- Distributed artifact: <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>
- License: MIT

## Qwen 3.5 4B polish model

The optional Polish feature downloads an Unsloth GGUF conversion of Qwen 3.5
4B Instruct. The weights are not stored in this repository or bundled in the
application.

- Model (exact revision): <https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523>
- Original model: Qwen 3.5 4B Instruct
- Conversion: Unsloth contributors
- Model license: Apache License 2.0; see `LICENSE-APACHE`

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
