# Real-speech endpoint fixtures

These are human-read, 16-bit mono excerpts from the LibriSpeech `test.clean`
split, versioned specifically for the auto-stop false-cut gate. They are not
macOS `say` output. OpenSLR publishes LibriSpeech under CC BY 4.0; attribution
and exact source identities are recorded in `manifest.json` and
`THIRD_PARTY_NOTICES.md`.

The source WAVs were fetched from the Hugging Face dataset viewer at immutable
dataset revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. They were resampled
from 16 kHz to the benchmark host's 48 kHz BlackHole format without trimming,
normalization, denoising, or inserted silence:

```bash
ffmpeg -i SOURCE.wav -ar 48000 -ac 1 -c:a pcm_s16le OUTPUT.wav
```

The 14.225 s fixture contains a reviewed natural 544 ms VAD pause beginning
near 8.064 s. The former 150 ms production policy stops there; Long-form mode
waits through it and captures the complete recording. The 3.505 s fixture
guards the ordinary single-sentence case.

Run the production capture/VAD/Core ML gate with:

```bash
scripts/bench-endpoint-policy.sh
```

The gate selects `BlackHole 2ch` explicitly and never changes the system input
or output defaults. A run fails on the first false stop because `bench_e2e`
cannot observe the fixture's reviewed acoustic endpoint until playback reaches
it. Passing all repetitions therefore proves a zero false-stop count for this
corpus; the same run reports p50/p95 final-pause latency.
