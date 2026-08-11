#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Verify issue #6's vocabulary sweep and no-training decision."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from qwen3_asr_eval_common import resolve_contained_file, score_pairs, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "bench/domain-adaptation/m5-pro-2026-08-11.json"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def read_json(file: Path) -> dict[str, Any]:
    value = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{file} must contain a JSON object")
    return value


def repo_file(value: object) -> Path:
    return resolve_contained_file(ROOT, value, max_bytes=MAX_EVIDENCE_BYTES)


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


def verify_file_record(label: str, record: dict[str, Any]) -> Path:
    file = repo_file(record["path"])
    require_equal(f"{label} bytes", file.stat().st_size, record["bytes"])
    require_equal(f"{label} SHA-256", sha256_file(file), record["sha256"])
    return file


def category_scores(
    fixtures: list[dict[str, Any]], raw_fixtures: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    names = sorted({name for fixture in fixtures for name in fixture["categories"]})
    output: dict[str, dict[str, float]] = {}
    for name in names:
        pairs = [
            (fixture["reference"], measured["hypothesis"])
            for fixture, measured in zip(fixtures, raw_fixtures, strict=True)
            if name in fixture["categories"]
        ]
        score = score_pairs(pairs)
        output[name] = {
            "wer_percent": float(score["wer_percent"]),
            "cer_percent": float(score["cer_percent"]),
        }
    return output


def verify_shipping_baseline(
    baseline_file: Path,
    baseline: dict[str, Any],
    manifest: dict[str, Any],
    manifest_digest: str,
) -> None:
    require_equal("shipping baseline schema", baseline["schema_version"], 1)
    require_equal(
        "shipping baseline manifest SHA-256",
        baseline["gold_manifest_sha256"],
        manifest_digest,
    )
    source_relative = baseline_file.parent.relative_to(ROOT) / baseline["source_report"]
    source_file = repo_file(source_relative.as_posix())
    require_equal(
        "shipping source report SHA-256",
        sha256_file(source_file),
        baseline["source_report_sha256"],
    )
    source = read_json(source_file)
    require_equal("shipping source manifest schema", source["manifest_version"], 2)
    fixtures = manifest["fixtures"]
    raw_fixtures = source["fixtures"]
    pairs = [
        (expected["reference"], measured["hypothesis"])
        for expected, measured in zip(fixtures, raw_fixtures, strict=True)
    ]
    overall = score_pairs(pairs)
    categories = category_scores(fixtures, raw_fixtures)
    require_close("shipping baseline recomputed WER", baseline["wer_percent"], overall["wer_percent"])
    require_close("shipping baseline recomputed CER", baseline["cer_percent"], overall["cer_percent"])
    for name, score in categories.items():
        require_close(f"shipping baseline {name} WER", baseline["categories"][name]["wer_percent"], score["wer_percent"])
        require_close(f"shipping baseline {name} CER", baseline["categories"][name]["cer_percent"], score["cer_percent"])


def verify_candidate(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    hardware: dict[str, Any],
    vocabulary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    report_file = repo_file(candidate["report"])
    require_equal(
        f"{candidate['name']} report SHA-256",
        sha256_file(report_file),
        candidate["report_sha256"],
    )
    raw = read_json(report_file)
    require_equal(f"{candidate['name']} schema", raw["schema_version"], 3)
    require_equal(f"{candidate['name']} manifest schema", raw["manifest_version"], 2)
    require_equal(f"{candidate['name']} thresholds", raw["thresholds"], manifest["thresholds"])
    require_equal(f"{candidate['name']} gate", raw["passed"], False)

    raw_hardware = {
        "chip": raw["metadata"]["chip"],
        "memory_bytes": raw["metadata"]["memory_bytes"],
        "logical_cpus": raw["metadata"]["logical_cpus"],
        "operating_system": raw["metadata"]["operating_system"],
        "architecture": raw["metadata"]["architecture"],
    }
    require_equal(f"{candidate['name']} hardware", raw_hardware, hardware)
    require_equal(
        f"{candidate['name']} backend",
        raw["metadata"]["backend"],
        {
            "backend": "sherpa-onnx",
            "model": "NVIDIA Parakeet TDT 0.6B v3",
            "quantization": "int8",
            "execution_provider": "coreml-requested",
        },
    )

    decoding = raw["metadata"]["decoding"]
    require_equal(f"{candidate['name']} method", decoding["method"], candidate["method"])
    require_equal(
        f"{candidate['name']} score", decoding["hotword_score"], candidate["hotword_score"]
    )
    biased = candidate["name"] != "greedy"
    require_equal(f"{candidate['name']} vocabulary requested", decoding["contextual_vocabulary_requested"], biased)
    require_equal(f"{candidate['name']} vocabulary active", decoding["contextual_vocabulary_active"], biased)
    require_equal(f"{candidate['name']} term count", decoding["vocabulary_terms_requested"], 3 if biased else 0)
    require_equal(f"{candidate['name']} vocabulary digest", decoding["vocabulary_sha256"], vocabulary["sha256"] if biased else None)
    require_equal(f"{candidate['name']} hotword digest", decoding["generated_hotwords_sha256"], vocabulary["generated_hotwords_sha256"] if biased else None)

    repeatability = raw["repeatability"]
    require_equal(f"{candidate['name']} repetitions", repeatability["repetitions"], 10)
    require_equal(f"{candidate['name']} nondeterministic fixtures", repeatability["nondeterministic_fixtures"], 0)
    require_equal(f"{candidate['name']} nondeterministic outputs", repeatability["nondeterministic_outputs"], candidate["nondeterministic_outputs"])
    require_close(f"{candidate['name']} WER spread", repeatability["wer_spread_percent"], 0.0)
    require_close(f"{candidate['name']} CER spread", repeatability["cer_spread_percent"], 0.0)

    fixtures = manifest["fixtures"]
    raw_fixtures = raw["fixtures"]
    require_equal(f"{candidate['name']} fixture count", len(raw_fixtures), len(fixtures))
    for expected, measured in zip(fixtures, raw_fixtures, strict=True):
        require_equal(f"{candidate['name']} fixture name", measured["file"], expected["file"])
        require_equal(f"{candidate['name']} fixture reference", measured["reference"], expected["reference"])
        require_equal(f"{candidate['name']} fixture categories", measured["categories"], expected["categories"])
        require_equal(f"{candidate['name']} fixture repetitions", measured["repetitions"], 10)
        require_equal(f"{candidate['name']} fixture unique output", measured["unique_hypotheses"], 1)

    pairs = [
        (expected["reference"], measured["hypothesis"])
        for expected, measured in zip(fixtures, raw_fixtures, strict=True)
    ]
    overall = score_pairs(pairs)
    categories = category_scores(fixtures, raw_fixtures)
    require_close(f"{candidate['name']} recomputed WER", raw["overall"]["wer_percent"], overall["wer_percent"])
    require_close(f"{candidate['name']} recomputed CER", raw["overall"]["cer_percent"], overall["cer_percent"])
    for name, score in categories.items():
        require_close(f"{candidate['name']} {name} WER", raw["categories"][name]["wer_percent"], score["wer_percent"])
        require_close(f"{candidate['name']} {name} CER", raw["categories"][name]["cer_percent"], score["cer_percent"])

    copied = {
        "wer_percent": repeatability["wer_percent_max"],
        "cer_percent": repeatability["cer_percent_max"],
        "custom_vocabulary_wer_percent": raw["categories"]["custom-vocabulary"]["wer_percent"],
        "custom_vocabulary_cer_percent": raw["categories"]["custom-vocabulary"]["cer_percent"],
        "corpus_decode_seconds_p50": repeatability["corpus_decode_seconds_p50"],
        "corpus_decode_seconds_p95": repeatability["corpus_decode_seconds_p95"],
        "corpus_rtfx_p50": repeatability["corpus_rtfx_p50"],
        "peak_rss_bytes": raw["metadata"]["peak_resident_bytes"],
        "passed_shipping_gate": raw["passed"],
    }
    for field, value in copied.items():
        if isinstance(value, float):
            require_close(f"{candidate['name']} copied {field}", candidate[field], value)
        else:
            require_equal(f"{candidate['name']} copied {field}", candidate[field], value)
    return raw, categories


def main() -> None:
    summary = read_json(SUMMARY)
    require_equal("summary schema", summary["schema_version"], 1)
    require_equal("issue", summary["issue"], 6)
    require_equal("report schema contract", summary["measurement_contract"]["report_schema_version"], 3)
    require_equal("repetition contract", summary["measurement_contract"]["repetitions"], 10)

    manifest_file = verify_file_record("gold manifest", summary["inputs"]["gold_manifest"])
    verify_file_record("gold sources", summary["inputs"]["gold_sources"])
    vocabulary_file = verify_file_record("vocabulary", summary["inputs"]["vocabulary"])
    baseline_file = verify_file_record("shipping baseline", summary["inputs"]["shipping_quality_baseline"])
    require_equal("vocabulary terms", vocabulary_file.read_text(encoding="utf-8").splitlines(), summary["inputs"]["vocabulary"]["terms"])

    for relative, expected_digest in summary["evaluator_sources"].items():
        require_equal(f"source {relative} SHA-256", sha256_file(repo_file(relative)), expected_digest)

    manifest = read_json(manifest_file)
    baseline = read_json(baseline_file)
    verify_shipping_baseline(
        baseline_file,
        baseline,
        manifest,
        summary["inputs"]["gold_manifest"]["sha256"],
    )
    candidates = summary["candidates"]
    require_equal(
        "candidate order",
        [candidate["name"] for candidate in candidates],
        ["greedy", "score-0", "score-2", "score-2.75", "score-4.5", "score-6"],
    )
    reports: dict[str, dict[str, Any]] = {}
    category_results: dict[str, dict[str, dict[str, float]]] = {}
    for candidate in candidates:
        reports[candidate["name"]], category_results[candidate["name"]] = verify_candidate(
            candidate, manifest, summary["hardware"], summary["inputs"]["vocabulary"]
        )

    greedy_p50 = reports["greedy"]["repeatability"]["corpus_decode_seconds_p50"]
    for candidate in candidates:
        p50 = reports[candidate["name"]]["repeatability"]["corpus_decode_seconds_p50"]
        regression = 100.0 * (p50 / greedy_p50 - 1.0)
        require_close(
            f"{candidate['name']} latency regression",
            candidate["latency_regression_vs_greedy_percent"],
            regression,
        )

    delta_record = summary["first_effect_category_delta_points_vs_greedy"]
    require_equal("first-effect candidate", delta_record["candidate"], "score-2.75")
    for name, greedy_score in category_results["greedy"].items():
        candidate_score = category_results["score-2.75"][name]
        require_close(f"{name} WER delta", delta_record[name]["wer"], candidate_score["wer_percent"] - greedy_score["wer_percent"])
        require_close(f"{name} CER delta", delta_record[name]["cer"], candidate_score["cer_percent"] - greedy_score["cer_percent"])

    for name, raw in reports.items():
        categories = category_results[name]
        clears_locked_gate = (
            raw["overall"]["wer_percent"] <= baseline["wer_percent"]
            and raw["overall"]["cer_percent"] <= baseline["cer_percent"]
            and all(
                categories[category][metric] <= limits[metric]
                for category, limits in baseline["categories"].items()
                for metric in ("wer_percent", "cer_percent")
            )
        )
        require_equal(f"{name} locked shipping gate", clears_locked_gate, False)

    decision = summary["decision"]
    require_equal("training authorization", decision["training_authorized"], False)
    require_equal("QAT/distillation performed", decision["qat_or_distillation_performed"], False)
    require_equal("global score change", decision["global_hotword_score_change"], False)

    serialized = SUMMARY.read_text(encoding="utf-8")
    if "/Users/" in serialized or "/private/" in serialized:
        raise ValueError("summary contains a machine-local absolute path")
    print("domain-adaptation evidence verified: 6 reports, 60 corpus runs, no candidate clears the locked gate")


if __name__ == "__main__":
    main()
