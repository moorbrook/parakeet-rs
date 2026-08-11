#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Verify that the checked-in Qwen summary exactly matches raw evaluator output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from qwen3_asr_eval_common import (
    RUNTIME_COMMIT,
    evaluate_quality_gate,
    evaluator_identity,
    load_gold_corpus,
    load_quality_baseline,
    normalize_lexical,
    percentile,
    resolve_contained_file,
    score_pairs,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def repo_file(value: object) -> Path:
    if isinstance(value, Path):
        if value.is_absolute():
            try:
                value = value.resolve().relative_to(ROOT.resolve()).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"evidence path escapes the repository: {value}"
                ) from error
        else:
            value = value.as_posix()
    return resolve_contained_file(ROOT, value, max_bytes=MAX_EVIDENCE_BYTES)


def repo_directory(value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("evidence directory must be a relative path string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"evidence directory escapes the repository: {value!r}")
    root = ROOT.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise ValueError(f"evidence directory is outside the repository: {value!r}")
    return resolved


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def require_close(label: str, actual: object, expected: object) -> None:
    if (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(expected, (int, float))
        or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def manifest_digest(files: dict[str, dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for relative, record in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_offline(
    name: str,
    raw: dict[str, object],
    summary: dict[str, object],
    fixtures: list[dict[str, object]],
    thresholds: dict[str, float],
    baseline: dict[str, object],
) -> None:
    field_map = {
        "repetitions": "repetitions",
        "load_seconds": "load_seconds",
        "warmup_seconds": "warmup_seconds",
        "corpus_wall_p50_seconds": "corpus_wall_p50_seconds",
        "corpus_wall_p95_seconds": "corpus_wall_p95_seconds",
        "rtfx_p50": "rtfx_p50",
        "peak_rss_bytes": "peak_rss_bytes",
        "wer_percent": "wer_percent_max",
        "cer_percent": "cer_percent_max",
        "nondeterministic_outputs": "nondeterministic_outputs",
        "categories": "categories",
    }
    for summary_field, raw_field in field_map.items():
        require_equal(
            f"offline.{name}.{summary_field}",
            summary[summary_field],
            raw[raw_field],
        )
    repetitions = raw["repetitions"]
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError(f"offline {name} repetitions must be an integer")
    raw_fixtures = raw["fixtures"]
    if not isinstance(raw_fixtures, dict):
        raise TypeError(f"offline {name} fixtures must be an object")
    require_equal(
        f"offline {name} fixture set",
        list(raw_fixtures),
        [fixture["file"] for fixture in fixtures],
    )
    repetition_scores: list[dict[str, float | int]] = []
    category_scores: dict[str, list[dict[str, float | int]]] = {
        category: []
        for category in sorted(
            {category for fixture in fixtures for category in fixture["categories"]}
        )
    }
    for fixture in fixtures:
        record = raw_fixtures[fixture["file"]]
        require_equal(
            f"offline {name} {fixture['file']} reference",
            record["reference"],
            fixture["reference"],
        )
        require_equal(
            f"offline {name} {fixture['file']} categories",
            record["categories"],
            fixture["categories"],
        )
        require_equal(
            f"offline {name} {fixture['file']} run count",
            len(record["runs"]),
            repetitions,
        )
    for repetition in range(repetitions):
        pairs = [
            (
                fixture["reference"],
                raw_fixtures[fixture["file"]]["runs"][repetition]["text"],
            )
            for fixture in fixtures
        ]
        repetition_scores.append(score_pairs(pairs))
        for category in category_scores:
            category_scores[category].append(
                score_pairs(
                    [
                        (
                            fixture["reference"],
                            raw_fixtures[fixture["file"]]["runs"][repetition]["text"],
                        )
                        for fixture in fixtures
                        if category in fixture["categories"]
                    ]
                )
            )
    expected_categories = {
        category: {
            "wer_percent": max(score["wer_percent"] for score in scores),
            "cer_percent": max(score["cer_percent"] for score in scores),
        }
        for category, scores in category_scores.items()
    }
    expected_nondeterministic = sum(
        sum(run["text"] != record["runs"][0]["text"] for run in record["runs"][1:])
        for record in raw_fixtures.values()
    )
    expected_wer = max(score["wer_percent"] for score in repetition_scores)
    expected_cer = max(score["cer_percent"] for score in repetition_scores)
    require_equal(
        f"offline {name} recomputed WER", raw["wer_percent_max"], expected_wer
    )
    require_equal(
        f"offline {name} recomputed CER", raw["cer_percent_max"], expected_cer
    )
    require_equal(
        f"offline {name} recomputed categories", raw["categories"], expected_categories
    )
    require_equal(
        f"offline {name} recomputed nondeterminism",
        raw["nondeterministic_outputs"],
        expected_nondeterministic,
    )
    corpus_wall = raw["corpus_wall_seconds"]
    require_equal(f"offline {name} corpus timing count", len(corpus_wall), repetitions)
    require_equal(
        f"offline {name} recomputed p50",
        raw["corpus_wall_p50_seconds"],
        percentile(corpus_wall, 50),
    )
    require_equal(
        f"offline {name} recomputed p95",
        raw["corpus_wall_p95_seconds"],
        percentile(corpus_wall, 95),
    )
    total_audio_seconds = sum(
        raw_fixtures[fixture["file"]]["runs"][0]["audio_seconds"]
        for fixture in fixtures
    )
    require_equal(
        f"offline {name} recomputed RTFx",
        raw["rtfx_p50"],
        total_audio_seconds / percentile(corpus_wall, 50),
    )
    expected_gate = evaluate_quality_gate(
        thresholds=thresholds,
        wer_percent=float(expected_wer),
        cer_percent=float(expected_cer),
        nondeterministic_outputs=expected_nondeterministic,
        categories=expected_categories,
        baseline=baseline,
    )
    require_equal(f"offline {name} recomputed gate", raw["quality_gate"], expected_gate)
    if expected_gate["passed"] is not False:
        raise ValueError(f"offline {name} no longer records the expected gate failure")


def verify_shipping_source(
    source: dict[str, object], fixtures: list[dict[str, object]]
) -> None:
    source_fixtures = source["fixtures"]
    pairs = [
        (fixture["reference"], source_fixture["hypothesis"])
        for fixture, source_fixture in zip(fixtures, source_fixtures, strict=True)
    ]
    overall = score_pairs(pairs)
    for metric in ("wer_percent", "cer_percent"):
        require_equal(
            f"shipping source recomputed {metric}",
            source["overall"][metric],
            overall[metric],
        )
    categories = {
        category for fixture in fixtures for category in fixture["categories"]
    }
    for category in categories:
        score = score_pairs(
            [
                (fixture["reference"], source_fixture["hypothesis"])
                for fixture, source_fixture in zip(
                    fixtures, source_fixtures, strict=True
                )
                if category in fixture["categories"]
            ]
        )
        for metric in ("wer_percent", "cer_percent"):
            require_equal(
                f"shipping source {category} recomputed {metric}",
                source["categories"][category][metric],
                score[metric],
            )


def verify_streaming_raw(
    raw: dict[str, object],
    fixtures: list[dict[str, object]],
    thresholds: dict[str, float],
    baseline: dict[str, object],
) -> dict[str, dict[str, object]]:
    require_equal("streaming tail refinement", raw["tail_refine_enabled"], True)
    require_equal(
        "streaming tail tokenizer binding",
        raw["tail_refine_tokenizer_binding"],
        "verified-session",
    )
    repetitions = raw["repetitions"]
    raw_fixtures = raw["fixtures"]
    require_equal(
        "streaming fixture order",
        [fixture["file"] for fixture in raw_fixtures],
        [fixture["file"] for fixture in fixtures],
    )
    schedules = [summary["schedule"] for summary in raw["schedule_summary"]]
    if len(set(schedules)) != len(schedules):
        raise ValueError("streaming schedule summary contains duplicates")

    first_results: dict[str, dict[str, object]] = {}
    for fixture, raw_fixture in zip(fixtures, raw_fixtures, strict=True):
        for field in ("file", "reference", "categories"):
            require_equal(
                f"streaming {fixture['file']} {field}",
                raw_fixture[field],
                fixture[field],
            )
        fixture_schedules = raw_fixture["schedules"]
        require_equal(
            f"streaming {fixture['file']} schedule result count",
            len(fixture_schedules),
            repetitions * len(schedules),
        )
        for schedule_name in schedules:
            results = [
                result
                for result in fixture_schedules
                if result["schedule"] == schedule_name
            ]
            require_equal(
                f"streaming {fixture['file']} {schedule_name} repetition indexes",
                [result["repetition"] for result in results],
                list(range(repetitions)),
            )
            for result in results:
                expected_reference_score = score_pairs(
                    [(fixture["reference"], result["final_text"])]
                )
                require_equal(
                    f"streaming {fixture['file']} reference score",
                    result["reference_score"],
                    expected_reference_score,
                )
                expected_offline_score = score_pairs(
                    [(raw_fixture["offline_text"], result["final_text"])]
                )
                require_equal(
                    f"streaming {fixture['file']} offline score",
                    result["offline_score"],
                    expected_offline_score,
                )
                require_equal(
                    f"streaming {fixture['file']} offline match",
                    result["matches_offline_lexically"],
                    normalize_lexical(raw_fixture["offline_text"])
                    == normalize_lexical(result["final_text"]),
                )
            first_results.setdefault(fixture["file"], results[0])

    raw_summaries = {
        summary["schedule"]: summary for summary in raw["schedule_summary"]
    }
    category_names = sorted(
        {category for fixture in fixtures for category in fixture["categories"]}
    )
    for schedule_name in schedules:
        repetition_reports: list[dict[str, float | int]] = []
        category_reports: dict[str, list[dict[str, float | int]]] = {
            category: [] for category in category_names
        }
        per_fixture_results: dict[str, list[dict[str, object]]] = {}
        for fixture, raw_fixture in zip(fixtures, raw_fixtures, strict=True):
            per_fixture_results[fixture["file"]] = [
                result
                for result in raw_fixture["schedules"]
                if result["schedule"] == schedule_name
            ]
        for repetition in range(repetitions):
            selected = [
                per_fixture_results[fixture["file"]][repetition] for fixture in fixtures
            ]
            score = score_pairs(
                [
                    (fixture["reference"], result["final_text"])
                    for fixture, result in zip(fixtures, selected, strict=True)
                ]
            )
            repetition_reports.append(
                {
                    **score,
                    "offline_mismatch_fixtures": sum(
                        not result["matches_offline_lexically"] for result in selected
                    ),
                    "corpus_wall_seconds": sum(
                        result["total_wall_seconds"] for result in selected
                    ),
                    "finalize_wall_seconds": sum(
                        result["finalize_wall_seconds"] for result in selected
                    ),
                    "boundary_events": sum(
                        len(result["boundary_events"]) for result in selected
                    ),
                }
            )
            for category in category_names:
                category_reports[category].append(
                    score_pairs(
                        [
                            (fixture["reference"], result["final_text"])
                            for fixture, result in zip(fixtures, selected, strict=True)
                            if category in fixture["categories"]
                        ]
                    )
                )
        expected_nondeterministic = sum(
            sum(
                result["final_text"] != results[0]["final_text"]
                for result in results[1:]
            )
            for results in per_fixture_results.values()
        )
        expected_categories = {
            category: {
                "wer_percent": max(score["wer_percent"] for score in category_scores),
                "cer_percent": max(score["cer_percent"] for score in category_scores),
            }
            for category, category_scores in category_reports.items()
        }
        expected = raw_summaries[schedule_name]
        require_equal(
            f"streaming {schedule_name} repetition reports",
            expected["repetition_reports"],
            repetition_reports,
        )
        require_equal(
            f"streaming {schedule_name} categories",
            expected["categories"],
            expected_categories,
        )
        require_equal(
            f"streaming {schedule_name} nondeterminism",
            expected["nondeterministic_outputs"],
            expected_nondeterministic,
        )
        corpus_wall = [report["corpus_wall_seconds"] for report in repetition_reports]
        finalize_wall = [
            report["finalize_wall_seconds"] for report in repetition_reports
        ]
        for label, values in (
            ("corpus_wall", corpus_wall),
            ("finalize_wall", finalize_wall),
        ):
            require_equal(
                f"streaming {schedule_name} {label} p50",
                expected[f"{label}_p50_seconds"],
                percentile(values, 50),
            )
            require_equal(
                f"streaming {schedule_name} {label} p95",
                expected[f"{label}_p95_seconds"],
                percentile(values, 95),
            )
        expected_gate = evaluate_quality_gate(
            thresholds=thresholds,
            wer_percent=float(
                max(report["wer_percent"] for report in repetition_reports)
            ),
            cer_percent=float(
                max(report["cer_percent"] for report in repetition_reports)
            ),
            nondeterministic_outputs=expected_nondeterministic,
            categories=expected_categories,
            baseline=baseline,
        )
        require_equal(
            f"streaming {schedule_name} quality gate",
            expected["quality_gate"],
            expected_gate,
        )
        if expected_gate["passed"] is not False:
            raise ValueError(
                f"streaming {schedule_name} no longer records the expected gate failure"
            )
    return first_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "bench/qwen3-asr/m5-pro-2026-08-11.json",
    )
    args = parser.parse_args()
    summary_path = repo_file(args.summary)
    summary = read_json(summary_path)
    require_equal(
        "reference runtime commit",
        summary["reference_runtime"]["commit"],
        RUNTIME_COMMIT,
    )

    corpus = summary["corpus"]
    manifest_path = repo_file(corpus["manifest"])
    audio_dir = repo_directory(corpus["audio_dir"])
    sources_path = repo_file(corpus["sources"])
    require_equal(
        "gold manifest SHA-256",
        sha256_file(manifest_path),
        corpus["manifest_sha256"],
    )
    fixtures, thresholds, gold_input_identity = load_gold_corpus(
        manifest_path, audio_dir, sources_path
    )
    reference_score = score_pairs([(fixture["reference"], "") for fixture in fixtures])
    require_equal("corpus fixture count", corpus["fixtures"], len(fixtures))
    require_equal(
        "corpus reference words",
        corpus["reference_words"],
        reference_score["reference_words"],
    )
    require_equal(
        "corpus reference characters",
        corpus["reference_chars"],
        reference_score["reference_chars"],
    )
    require_equal(
        "gold sources SHA-256",
        gold_input_identity["sources_sha256"],
        corpus["sources_sha256"],
    )
    require_equal(
        "gold audio artifact SHA-256",
        gold_input_identity["audio_artifact_sha256"],
        corpus["audio_artifact_sha256"],
    )
    baseline_record = summary["quality_baseline"]
    baseline_path = repo_file(baseline_record["path"])
    require_equal(
        "quality baseline SHA-256",
        sha256_file(baseline_path),
        baseline_record["sha256"],
    )
    baseline = read_json(baseline_path)
    verified_baseline = load_quality_baseline(baseline_path, manifest_path, fixtures)

    raw_reports: dict[str, dict[str, object]] = {}
    require_equal(
        "source report set",
        set(summary["source_reports"]),
        {
            "shipping_coreml_gold",
            "q8_offline",
            "q4_offline",
            "fp16_offline",
            "q8_streaming",
        },
    )
    for name, record in summary["source_reports"].items():
        path = repo_file(record["path"])
        require_equal(
            f"source report {name} SHA-256", sha256_file(path), record["sha256"]
        )
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "file://" in text:
            raise ValueError(f"source report {name} contains a machine-local path")
        raw_reports[name] = read_json(path)

    source_report = raw_reports["shipping_coreml_gold"]
    verify_shipping_source(source_report, fixtures)
    shipping_summary = summary["shipping_coreml"]
    shipping_fields = {
        "load_seconds": ("metadata", "model_load_seconds"),
        "first_result_seconds": ("metadata", "first_result_seconds"),
        "corpus_wall_p50_seconds": (
            "repeatability",
            "corpus_decode_seconds_p50",
        ),
        "corpus_wall_p95_seconds": (
            "repeatability",
            "corpus_decode_seconds_p95",
        ),
        "rtfx_p50": ("repeatability", "corpus_rtfx_p50"),
        "peak_rss_bytes": ("metadata", "peak_resident_bytes"),
        "wer_percent": ("overall", "wer_percent"),
        "cer_percent": ("overall", "cer_percent"),
        "nondeterministic_outputs": (
            "repeatability",
            "nondeterministic_outputs",
        ),
    }
    for summary_field, (section, raw_field) in shipping_fields.items():
        require_equal(
            f"shipping_coreml.{summary_field}",
            shipping_summary[summary_field],
            source_report[section][raw_field],
        )
    for field in ("wer_percent", "cer_percent"):
        require_equal(
            f"quality baseline {field}",
            baseline[field],
            source_report["overall"][field],
        )
    for category, expected in baseline["categories"].items():
        for field in ("wer_percent", "cer_percent"):
            require_equal(
                f"quality baseline {category}.{field}",
                expected[field],
                source_report["categories"][category][field],
            )

    artifacts = {item["name"]: item for item in summary["artifacts"]}
    require_equal("artifact set", set(artifacts), {"q8", "q4", "fp16"})
    require_equal(
        "offline summary set",
        [item["name"] for item in summary["offline"]],
        ["q8", "q4", "fp16"],
    )
    offline_identity = evaluator_identity(ROOT / "scripts/eval-qwen3-asr-offline.py")
    for name in ("q8", "q4", "fp16"):
        raw = raw_reports[f"{name}_offline"]
        require_equal(
            f"offline {name} runtime commit", raw["runtime_commit"], RUNTIME_COMMIT
        )
        require_equal(
            f"offline {name} evaluator identity",
            raw["evaluator_identity"],
            offline_identity,
        )
        require_equal(
            f"offline {name} Python runtime",
            raw["runtime_versions"]["python"],
            summary["oracle_environment"]["python"],
        )
        require_equal(
            f"offline {name} distribution runtime",
            raw["runtime_versions"]["distributions"],
            summary["oracle_environment"]["distributions"],
        )
        require_equal(
            f"offline {name} gold input identity",
            raw["gold_input_identity"],
            gold_input_identity,
        )
        require_equal(
            f"offline {name} audio runtime",
            raw["audio_runtime"],
            summary["audio_runtime"],
        )
        artifact = artifacts[name]
        for summary_field, raw_field in {
            "model": "model",
            "snapshot": "model_snapshot",
            "weight_bytes": "weight_bytes",
            "weight_sha256": "weight_sha256",
            "artifact_sha256": "model_artifact_sha256",
        }.items():
            require_equal(
                f"artifacts.{name}.{summary_field}",
                artifact[summary_field],
                raw[raw_field],
            )
        require_equal(
            f"artifacts.{name}.composite digest",
            raw["model_artifact_sha256"],
            manifest_digest(raw["model_artifact_manifest"]),
        )
        offline = next(item for item in summary["offline"] if item["name"] == name)
        verify_offline(
            name,
            raw,
            offline,
            fixtures,
            thresholds,
            verified_baseline,
        )

    q8_fixture_map = raw_reports["q8_offline"]["fixtures"]
    q8_total_audio_seconds = sum(
        q8_fixture_map[fixture["file"]]["runs"][0]["audio_seconds"]
        for fixture in fixtures
    )
    require_equal(
        "corpus audio seconds", corpus["audio_seconds"], q8_total_audio_seconds
    )

    streaming_raw = raw_reports["q8_streaming"]
    q8_raw = raw_reports["q8_offline"]
    require_equal(
        "streaming evaluator identity",
        streaming_raw["evaluator_identity"],
        evaluator_identity(ROOT / "scripts/eval-qwen3-asr-streaming.py"),
    )
    require_equal(
        "streaming gold input identity",
        streaming_raw["gold_input_identity"],
        gold_input_identity,
    )
    require_equal(
        "streaming audio runtime",
        streaming_raw["audio_runtime"],
        summary["audio_runtime"],
    )
    for field in (
        "runtime_commit",
        "runtime_versions",
        "model",
        "model_snapshot",
        "weight_bytes",
        "weight_sha256",
        "model_artifact_sha256",
        "model_artifact_manifest",
    ):
        require_equal(f"streaming q8 {field}", streaming_raw[field], q8_raw[field])
    streaming_summary = summary["streaming_q8"]
    for field in (
        "model_chunk_seconds",
        "repetitions",
        "tail_refine_enabled",
        "tail_refine_tokenizer_binding",
        "streaming_peak_rss_bytes",
        "post_reference_peak_rss_bytes",
    ):
        require_equal(
            f"streaming_q8.{field}", streaming_summary[field], streaming_raw[field]
        )

    first_streaming_results = verify_streaming_raw(
        streaming_raw,
        fixtures,
        thresholds,
        verified_baseline,
    )
    all_mid_chunk = all(
        result["ended_mid_model_chunk"]
        for fixture in streaming_raw["fixtures"]
        for result in fixture["schedules"]
    )
    require_equal(
        "streaming_q8.all_fixtures_end_mid_chunk",
        streaming_summary["all_fixtures_end_mid_chunk"],
        all_mid_chunk,
    )

    raw_schedules = {
        schedule["schedule"]: schedule for schedule in streaming_raw["schedule_summary"]
    }
    require_equal(
        "streaming schedule set",
        list(raw_schedules),
        [schedule["name"] for schedule in streaming_summary["schedules"]],
    )
    for schedule in streaming_summary["schedules"]:
        raw = raw_schedules[schedule["name"]]
        first_raw_schedule = next(
            item
            for item in streaming_raw["fixtures"][0]["schedules"]
            if item["schedule"] == schedule["name"]
        )
        require_equal(
            f"streaming_q8.{schedule['name']}.transport_chunk_seconds",
            schedule["transport_chunk_seconds"],
            first_raw_schedule["transport_chunk_seconds"],
        )
        field_map = {
            "boundary_events_min": "boundary_events_min",
            "boundary_events_max": "boundary_events_max",
            "wer_percent": "wer_percent_max",
            "cer_percent": "cer_percent_max",
            "offline_mismatch_fixtures": "offline_mismatch_fixtures_max",
            "nondeterministic_outputs": "nondeterministic_outputs",
            "corpus_wall_p50_seconds": "corpus_wall_p50_seconds",
            "corpus_wall_p95_seconds": "corpus_wall_p95_seconds",
            "finalize_wall_p50_seconds": "finalize_wall_p50_seconds",
            "finalize_wall_p95_seconds": "finalize_wall_p95_seconds",
        }
        for summary_field, raw_field in field_map.items():
            require_equal(
                f"streaming_q8.{schedule['name']}.{summary_field}",
                schedule[summary_field],
                raw[raw_field],
            )
    raw_fixture_map = {
        fixture["file"]: fixture for fixture in streaming_raw["fixtures"]
    }
    q8_fixture_map = q8_raw["fixtures"]
    mismatch_files = [
        fixture["file"]
        for fixture in streaming_raw["fixtures"]
        if not first_streaming_results[fixture["file"]]["matches_offline_lexically"]
    ]
    failures = streaming_summary["failures"]
    require_equal(
        "streaming_q8 failure fixture set",
        [failure["file"] for failure in failures],
        mismatch_files,
    )
    for failure in failures:
        file = failure["file"]
        fixture = raw_fixture_map[file]
        result = first_streaming_results[file]
        require_close(
            f"streaming_q8 {file} audio_seconds",
            failure["audio_seconds"],
            fixture["audio_seconds"],
        )
        require_equal(
            f"streaming_q8 {file} pending_tail_seconds",
            failure["pending_tail_seconds"],
            result["pending_tail_seconds"],
        )
        if "offline_text" in failure:
            require_equal(
                f"streaming_q8 {file} offline_text",
                failure["offline_text"],
                fixture["offline_text"],
            )
        if "streaming_text" in failure:
            require_equal(
                f"streaming_q8 {file} streaming_text",
                failure["streaming_text"],
                result["final_text"],
            )
        if "offline_wer_percent" in failure:
            offline_score = score_pairs(
                [
                    (
                        fixture["reference"],
                        q8_fixture_map[file]["runs"][0]["text"],
                    )
                ]
            )
            require_equal(
                f"streaming_q8 {file} offline WER",
                failure["offline_wer_percent"],
                offline_score["wer_percent"],
            )
        if "streaming_wer_percent" in failure:
            require_equal(
                f"streaming_q8 {file} streaming WER",
                failure["streaming_wer_percent"],
                result["reference_score"]["wer_percent"],
            )
        if "observed" in failure and not str(failure["observed"]).strip():
            raise ValueError(f"streaming_q8 {file} observation is empty")

    require_equal("decision.result", summary["decision"]["result"], "no-go")
    print("Qwen3-ASR evidence verified: 5 raw reports exactly match the summary")


if __name__ == "__main__":
    main()
