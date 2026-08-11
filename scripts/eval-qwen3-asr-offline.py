#!/usr/bin/env python3
# /// script
# requires-python = "==3.14.2"
# dependencies = [
#   "mlx-qwen3-asr @ git+https://github.com/moona3k/mlx-qwen3-asr.git@d1a035514e1d6ac31da7658b273482656eacba61",
#   "huggingface-hub==1.27.0",
#   "mlx==0.32.0",
#   "numpy==2.5.2",
#   "regex==2026.7.19",
# ]
# ///
"""Evaluate a pinned MLX Qwen3-ASR artifact with Parakeet's gold policy.

This is a developer-only reference oracle. It deliberately does not add Python
or MLX to Parakeet.app. Run it with ``uv run`` so the PEP 723 dependency stays
isolated from the shipping Rust dependency graph.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_qwen3_asr import load_model
from mlx_qwen3_asr.audio import load_audio_np
from mlx_qwen3_asr.session import Session
from qwen3_asr_eval_common import (
    RUNTIME_COMMIT,
    evaluate_quality_gate,
    evaluator_identity,
    load_gold_corpus,
    load_quality_baseline,
    peak_rss_bytes,
    percentile,
    resolve_model_source,
    score_pairs,
    verify_audio_runtime_identity,
    verify_loaded_model_source,
    verify_model_artifact,
    verify_runtime_identity,
)

SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--expected-snapshot", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--expected-ffmpeg-sha256", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    verified_runtime = verify_runtime_identity()
    audio_runtime = verify_audio_runtime_identity(args.expected_ffmpeg_sha256)
    fixtures, thresholds, gold_input_identity = load_gold_corpus(
        args.manifest, args.audio_dir, args.sources
    )
    baseline = load_quality_baseline(args.baseline, args.manifest, fixtures)
    model_source = resolve_model_source(args.model, args.expected_snapshot)
    audios = {
        fixture["file"]: np.asarray(
            load_audio_np(args.audio_dir / fixture["file"], sr=SAMPLE_RATE),
            dtype=np.float32,
        )
        for fixture in fixtures
    }

    (
        resolved_model_path,
        weight_path,
        weight_sha256,
        model_artifact_manifest,
        model_artifact_sha256,
    ) = verify_model_artifact(
        model_source,
        expected_snapshot=args.expected_snapshot,
        expected_weight_sha256=args.expected_weight_sha256,
        expected_artifact_sha256=args.expected_artifact_sha256,
    )

    load_started = time.perf_counter()
    model, config = load_model(str(resolved_model_path), dtype=mx.float16)
    # The pinned oracle currently evaluates parameters before returning. Keep
    # this explicit boundary so a future oracle change cannot move lazy weight
    # materialization from model load into the first transcription.
    mx.eval(model.parameters())
    verify_loaded_model_source(model, resolved_model_path)
    session = Session(
        model=model,
        dtype=mx.float16,
        tokenizer_model=str(resolved_model_path),
    )
    load_seconds = time.perf_counter() - load_started

    warmup_audio = audios[fixtures[0]["file"]]
    warmup_started = time.perf_counter()
    session.transcribe(warmup_audio, language="English")
    warmup_seconds = time.perf_counter() - warmup_started

    runs: dict[str, list[dict[str, object]]] = defaultdict(list)
    corpus_wall_seconds: list[float] = []
    for _ in range(args.repetitions):
        corpus_started = time.perf_counter()
        for fixture in fixtures:
            audio = audios[fixture["file"]]
            started = time.perf_counter()
            result = session.transcribe(audio, language="English")
            elapsed = time.perf_counter() - started
            runs[fixture["file"]].append(
                {
                    "text": result.text,
                    "language": result.language,
                    "wall_seconds": elapsed,
                    "audio_seconds": len(audio) / SAMPLE_RATE,
                }
            )
        corpus_wall_seconds.append(time.perf_counter() - corpus_started)

    repetition_scores = []
    category_scores: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    categories = sorted(
        {category for fixture in fixtures for category in fixture["categories"]}
    )
    for repetition in range(args.repetitions):
        pairs = [
            (fixture["reference"], runs[fixture["file"]][repetition]["text"])
            for fixture in fixtures
        ]
        repetition_scores.append(score_pairs(pairs))
        for category in categories:
            category_pairs = [
                (fixture["reference"], runs[fixture["file"]][repetition]["text"])
                for fixture in fixtures
                if category in fixture["categories"]
            ]
            category_scores[category].append(score_pairs(category_pairs))

    nondeterministic_outputs = sum(
        sum(run["text"] != fixture_runs[0]["text"] for run in fixture_runs[1:])
        for fixture_runs in runs.values()
    )
    total_audio_seconds = sum(len(audio) / SAMPLE_RATE for audio in audios.values())
    wer_percent_max = max(item["wer_percent"] for item in repetition_scores)
    cer_percent_max = max(item["cer_percent"] for item in repetition_scores)
    category_summary = {
        category: {
            "wer_percent": max(item["wer_percent"] for item in scores),
            "cer_percent": max(item["cer_percent"] for item in scores),
        }
        for category, scores in category_scores.items()
    }
    quality_gate = evaluate_quality_gate(
        thresholds=thresholds,
        wer_percent=float(wer_percent_max),
        cer_percent=float(cer_percent_max),
        nondeterministic_outputs=nondeterministic_outputs,
        categories=category_summary,
        baseline=baseline,
    )
    report = {
        "schema_version": 1,
        "runtime_commit": RUNTIME_COMMIT,
        "runtime_versions": verified_runtime,
        "audio_runtime": audio_runtime,
        "gold_input_identity": gold_input_identity,
        "evaluator_identity": evaluator_identity(Path(__file__)),
        "model": args.model,
        "model_snapshot": resolved_model_path.name,
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": weight_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "model_artifact_manifest": model_artifact_manifest,
        "model_config": {
            "model_type": getattr(config, "model_type", None),
            "classify_num": getattr(config, "classify_num", None),
        },
        "repetitions": args.repetitions,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "corpus_wall_seconds": corpus_wall_seconds,
        "corpus_wall_p50_seconds": percentile(corpus_wall_seconds, 50),
        "corpus_wall_p95_seconds": percentile(corpus_wall_seconds, 95),
        "rtfx_p50": total_audio_seconds / percentile(corpus_wall_seconds, 50),
        "peak_rss_bytes": peak_rss_bytes(),
        "wer_percent_max": wer_percent_max,
        "cer_percent_max": cer_percent_max,
        "nondeterministic_outputs": nondeterministic_outputs,
        "categories": category_summary,
        "quality_gate": quality_gate,
        "fixtures": {
            fixture["file"]: {
                "reference": fixture["reference"],
                "categories": fixture["categories"],
                "runs": runs[fixture["file"]],
            }
            for fixture in fixtures
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not quality_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
