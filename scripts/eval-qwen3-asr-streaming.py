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
"""Measure pinned Qwen3-ASR q8 at real 2 s streaming boundaries.

The schedules exercise exact, 100 ms segmented, and jittered transport writes.
The model still receives its own 2 s chunks, and every checked-in fixture ends
mid-chunk. This developer oracle is never bundled into Parakeet.app.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_qwen3_asr import load_model
from mlx_qwen3_asr.audio import load_audio_np
from mlx_qwen3_asr.session import Session
from mlx_qwen3_asr.streaming import streaming_metrics
from qwen3_asr_eval_common import (
    RUNTIME_COMMIT,
    evaluate_quality_gate,
    evaluator_identity,
    load_gold_corpus,
    load_quality_baseline,
    normalize_lexical,
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
MODEL_CHUNK_SECONDS = 2.0
SCHEDULES = {
    "exact-2s": [2.0],
    "segmented-100ms": [0.1],
    "jittered": [0.37, 0.83, 1.41, 0.29, 2.17, 0.53],
}


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


def run_schedule(
    *,
    audio: np.ndarray,
    session: Session,
    schedule_name: str,
    schedule_seconds: list[float],
) -> dict[str, object]:
    state = session.init_streaming(
        chunk_size_sec=MODEL_CHUNK_SECONDS,
        language="English",
        enable_tail_refine=True,
    )
    schedule_samples = [
        max(1, round(seconds * SAMPLE_RATE)) for seconds in schedule_seconds
    ]
    events: list[dict[str, object]] = []
    cursor = 0
    schedule_index = 0
    feed_wall = 0.0
    transport_calls = 0
    while cursor < len(audio):
        count = min(
            schedule_samples[schedule_index % len(schedule_samples)],
            len(audio) - cursor,
        )
        before_chunks = state.chunk_id
        started = time.perf_counter()
        session.feed_audio(audio[cursor : cursor + count], state)
        elapsed = time.perf_counter() - started
        feed_wall += elapsed
        cursor += count
        transport_calls += 1
        if state.chunk_id != before_chunks:
            events.append(
                {
                    "fed_audio_seconds": cursor / SAMPLE_RATE,
                    "decode_wall_seconds": elapsed,
                    "chunks_processed": state.chunk_id,
                    "buffered_seconds": len(state.buffer) / SAMPLE_RATE,
                    "text": state.text,
                    "stable_text": state.stable_text,
                }
            )
        schedule_index += 1

    pre_finalize_text = state.text
    pending_tail_seconds = len(state.buffer) / SAMPLE_RATE
    started = time.perf_counter()
    finish_streaming_with_verified_session(session, state)
    finalize_wall = time.perf_counter() - started
    return {
        "schedule": schedule_name,
        "transport_chunk_seconds": schedule_seconds,
        "transport_calls": transport_calls,
        "audio_seconds": len(audio) / SAMPLE_RATE,
        "ended_mid_model_chunk": pending_tail_seconds > 0.0,
        "pending_tail_seconds": pending_tail_seconds,
        "boundary_events": events,
        "pre_finalize_text": pre_finalize_text,
        "final_text": state.text,
        "feed_wall_seconds": feed_wall,
        "finalize_wall_seconds": finalize_wall,
        "total_wall_seconds": feed_wall + finalize_wall,
        "metrics": streaming_metrics(state),
    }


def finish_streaming_with_verified_session(session: Session, state: object) -> None:
    """Keep the oracle's tail-refinement decode on the verified tokenizer.

    The pinned runtime's tail path calls its module-level ``transcribe`` with a
    preloaded model object, which otherwise selects the runtime's default
    tokenizer. Bind only that synchronous call to this verified Session.
    """
    transcribe_module = importlib.import_module("mlx_qwen3_asr.transcribe")
    original_transcribe = transcribe_module.transcribe

    def session_transcribe(
        *,
        audio: np.ndarray,
        context: str = "",
        max_new_tokens: int | None = None,
        verbose: bool = False,
        **_: object,
    ) -> object:
        return session.transcribe(
            audio,
            context=context,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )

    transcribe_module.transcribe = session_transcribe
    try:
        session.finish_streaming(state)
    finally:
        transcribe_module.transcribe = original_transcribe


def summarize_schedule(
    fixture_reports: list[dict[str, object]],
    schedule_name: str,
    repetitions: int,
) -> dict[str, object]:
    results = [
        schedule
        for fixture in fixture_reports
        for schedule in fixture["schedules"]
        if schedule["schedule"] == schedule_name
    ]
    repetition_reports: list[dict[str, float | int]] = []
    category_repetition_reports: dict[str, list[dict[str, float | int]]] = {}
    category_names = sorted(
        {category for fixture in fixture_reports for category in fixture["categories"]}
    )
    for repetition in range(repetitions):
        repetition_results = [
            item for item in results if item["repetition"] == repetition
        ]
        word_edits = sum(
            item["reference_score"]["word_edits"] for item in repetition_results
        )
        reference_words = sum(
            item["reference_score"]["reference_words"] for item in repetition_results
        )
        char_edits = sum(
            item["reference_score"]["char_edits"] for item in repetition_results
        )
        reference_chars = sum(
            item["reference_score"]["reference_chars"] for item in repetition_results
        )
        repetition_reports.append(
            {
                "word_edits": word_edits,
                "reference_words": reference_words,
                "wer_percent": 100.0 * word_edits / reference_words,
                "char_edits": char_edits,
                "reference_chars": reference_chars,
                "cer_percent": 100.0 * char_edits / reference_chars,
                "offline_mismatch_fixtures": sum(
                    not item["matches_offline_lexically"] for item in repetition_results
                ),
                "corpus_wall_seconds": sum(
                    item["total_wall_seconds"] for item in repetition_results
                ),
                "finalize_wall_seconds": sum(
                    item["finalize_wall_seconds"] for item in repetition_results
                ),
                "boundary_events": sum(
                    len(item["boundary_events"]) for item in repetition_results
                ),
            }
        )
        for category in category_names:
            category_results = [
                item for item in repetition_results if category in item["categories"]
            ]
            word_edits = sum(
                item["reference_score"]["word_edits"] for item in category_results
            )
            reference_words = sum(
                item["reference_score"]["reference_words"] for item in category_results
            )
            char_edits = sum(
                item["reference_score"]["char_edits"] for item in category_results
            )
            reference_chars = sum(
                item["reference_score"]["reference_chars"] for item in category_results
            )
            category_repetition_reports.setdefault(category, []).append(
                {
                    "wer_percent": 100.0 * word_edits / reference_words,
                    "cer_percent": 100.0 * char_edits / reference_chars,
                }
            )
    corpus_wall_seconds = [
        float(item["corpus_wall_seconds"]) for item in repetition_reports
    ]
    finalize_wall_seconds = [
        float(item["finalize_wall_seconds"]) for item in repetition_reports
    ]
    nondeterministic_outputs = 0
    for fixture in fixture_reports:
        fixture_results = [
            item for item in fixture["schedules"] if item["schedule"] == schedule_name
        ]
        nondeterministic_outputs += sum(
            item["final_text"] != fixture_results[0]["final_text"]
            for item in fixture_results[1:]
        )
    return {
        "schedule": schedule_name,
        "fixtures": len(fixture_reports),
        "repetitions": repetitions,
        "wer_percent_max": max(item["wer_percent"] for item in repetition_reports),
        "cer_percent_max": max(item["cer_percent"] for item in repetition_reports),
        "offline_mismatch_fixtures_max": max(
            item["offline_mismatch_fixtures"] for item in repetition_reports
        ),
        "nondeterministic_outputs": nondeterministic_outputs,
        "corpus_wall_p50_seconds": percentile(corpus_wall_seconds, 50),
        "corpus_wall_p95_seconds": percentile(corpus_wall_seconds, 95),
        "finalize_wall_p50_seconds": percentile(finalize_wall_seconds, 50),
        "finalize_wall_p95_seconds": percentile(finalize_wall_seconds, 95),
        "boundary_events_min": min(
            item["boundary_events"] for item in repetition_reports
        ),
        "boundary_events_max": max(
            item["boundary_events"] for item in repetition_reports
        ),
        "repetition_reports": repetition_reports,
        "categories": {
            category: {
                "wer_percent": max(item["wer_percent"] for item in scores),
                "cer_percent": max(item["cer_percent"] for item in scores),
            }
            for category, scores in category_repetition_reports.items()
        },
    }


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
    model, _ = load_model(str(resolved_model_path), dtype=mx.float16)
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
    run_schedule(
        audio=warmup_audio,
        session=session,
        schedule_name="warmup",
        schedule_seconds=[MODEL_CHUNK_SECONDS],
    )
    warmup_seconds = time.perf_counter() - warmup_started

    fixture_reports: list[dict[str, object]] = []
    for fixture in fixtures:
        audio = audios[fixture["file"]]
        schedules: list[dict[str, object]] = []
        for repetition in range(args.repetitions):
            for schedule_name, schedule_seconds in SCHEDULES.items():
                result = run_schedule(
                    audio=audio,
                    session=session,
                    schedule_name=schedule_name,
                    schedule_seconds=schedule_seconds,
                )
                result["repetition"] = repetition
                result["reference_score"] = score_pairs(
                    [(fixture["reference"], str(result["final_text"]))]
                )
                result["categories"] = fixture["categories"]
                schedules.append(result)
                mx.clear_cache()
        fixture_reports.append(
            {
                "file": fixture["file"],
                "reference": fixture["reference"],
                "categories": fixture["categories"],
                "audio_seconds": len(audio) / SAMPLE_RATE,
                "schedules": schedules,
            }
        )

    streaming_peak_rss_bytes = peak_rss_bytes()
    for fixture, fixture_report in zip(fixtures, fixture_reports, strict=True):
        offline_started = time.perf_counter()
        offline = session.transcribe(audios[fixture["file"]], language="English")
        fixture_report["offline_text"] = offline.text
        fixture_report["offline_wall_seconds"] = time.perf_counter() - offline_started
        for result in fixture_report["schedules"]:
            result["offline_score"] = score_pairs(
                [(offline.text, str(result["final_text"]))]
            )
            result["matches_offline_lexically"] = normalize_lexical(
                offline.text
            ) == normalize_lexical(str(result["final_text"]))

    schedule_summary = [
        summarize_schedule(fixture_reports, schedule_name, args.repetitions)
        for schedule_name in SCHEDULES
    ]
    for summary in schedule_summary:
        summary["quality_gate"] = evaluate_quality_gate(
            thresholds=thresholds,
            wer_percent=float(summary["wer_percent_max"]),
            cer_percent=float(summary["cer_percent_max"]),
            nondeterministic_outputs=int(summary["nondeterministic_outputs"]),
            categories=summary["categories"],
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
        "model_chunk_seconds": MODEL_CHUNK_SECONDS,
        "tail_refine_enabled": True,
        "tail_refine_tokenizer_binding": "verified-session",
        "repetitions": args.repetitions,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "streaming_peak_rss_bytes": streaming_peak_rss_bytes,
        "post_reference_peak_rss_bytes": peak_rss_bytes(),
        "schedule_summary": schedule_summary,
        "fixtures": fixture_reports,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not all(summary["quality_gate"]["passed"] for summary in schedule_summary):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
