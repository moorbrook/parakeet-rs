# Qwen3-ASR challenger evidence

`m5-pro-2026-08-11.json` is the sanitized, reviewable summary for issue #5. It
pins the oracle commit, complete model artifacts, gold manifest, and shipping
quality baseline. `raw/` preserves the four losing Qwen reports plus a fresh
shipping Core ML gold report, without home-directory paths. It records the
exact local ffmpeg binary used for resampling. `uv run
scripts/verify-qwen3-asr-evidence.py` checks their digests and every copied
metric against the summary.

The evaluators require Python 3.14.2 and pin the oracle's direct dependencies
in PEP 723 metadata. Their adjacent `uv` lockfiles freeze the complete resolved
graph and artifact hashes.

Regenerate the full per-repetition q8 reports with:

```bash
uv run scripts/eval-qwen3-asr-offline.py \
  --model mlx-community/Qwen3-ASR-0.6B-8bit \
  --manifest bench/gold/manifest.json --audio-dir bench/gold/audio \
  --sources bench/gold/sources.json \
  --baseline bench/qwen3-asr/shipping-coreml-baseline.json \
  --expected-snapshot 89e96d92ba34aca20b3e29fb10cc284097d1219f \
  --expected-weight-sha256 b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8 \
  --expected-artifact-sha256 901975e84de875144dabd3a64655ed8f7335562626f816e68409bd61c50f278a \
  --expected-ffmpeg-sha256 1332dc2de372bade9a8a63da0d6cdfab9de97fcefbae707bcc0b0506e1203327 \
  --repetitions 10 --json-out bench/qwen3-asr/raw/q8-offline.json

uv run scripts/eval-qwen3-asr-streaming.py \
  --model mlx-community/Qwen3-ASR-0.6B-8bit \
  --manifest bench/gold/manifest.json --audio-dir bench/gold/audio \
  --sources bench/gold/sources.json \
  --baseline bench/qwen3-asr/shipping-coreml-baseline.json \
  --expected-snapshot 89e96d92ba34aca20b3e29fb10cc284097d1219f \
  --expected-weight-sha256 b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8 \
  --expected-artifact-sha256 901975e84de875144dabd3a64655ed8f7335562626f816e68409bd61c50f278a \
  --expected-ffmpeg-sha256 1332dc2de372bade9a8a63da0d6cdfab9de97fcefbae707bcc0b0506e1203327 \
  --repetitions 10 --json-out bench/qwen3-asr/raw/q8-streaming.json
```

The expected exit status is 2 because the reports are written before the
quality gate rejects each candidate. Repeat the offline command for q4 and
fp16 with the model, snapshot, weight digest, and composite artifact digest in
the summary, then update the source-report digests and run the verifier. These
PEP 723 tools are reference-only; Python, MLX, and Qwen are not shipped.
Interpretation and distribution details are in
[`docs/asr/QWEN3_ASR_EVALUATION.md`](../../docs/asr/QWEN3_ASR_EVALUATION.md).
