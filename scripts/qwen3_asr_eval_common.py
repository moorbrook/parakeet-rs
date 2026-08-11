"""Shared, dependency-free helpers for the Qwen3-ASR evaluation oracles."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import shutil
import subprocess
import unicodedata
from collections.abc import Iterable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

RUNTIME_DISTRIBUTIONS = (
    "mlx-qwen3-asr",
    "mlx",
    "numpy",
    "regex",
    "huggingface-hub",
)
RUNTIME_COMMIT = "d1a035514e1d6ac31da7658b273482656eacba61"
EXPECTED_PYTHON_VERSION = "3.14.2"
EXPECTED_DISTRIBUTIONS = {
    "mlx-qwen3-asr": "0.3.5",
    "mlx": "0.32.0",
    "numpy": "2.5.2",
    "regex": "2026.7.19",
    "huggingface-hub": "1.27.0",
}
GOLD_MANIFEST_VERSION = 2
THRESHOLD_FIELDS = {
    "max_wer_percent",
    "max_cer_percent",
    "baseline_wer_percent",
    "baseline_cer_percent",
    "max_wer_regression_percent",
    "max_cer_regression_percent",
}


class GoldFixture(TypedDict):
    file: str
    reference: str
    categories: list[str]


class QualityBaseline(TypedDict):
    backend: str
    wer_percent: float
    cer_percent: float
    categories: dict[str, dict[str, float]]


class GoldInputIdentity(TypedDict):
    sources_schema_version: int
    sources_sha256: str
    conversion: str
    audio_artifact_sha256: str
    audio_artifact_manifest: dict[str, dict[str, object]]


def normalize_lexical(value: str) -> str:
    """Match the lexical normalization in ``src/asr_eval.rs``."""
    normalized: list[str] = []
    separator_pending = False
    for source_character in unicodedata.normalize("NFC", value):
        for character in source_character.lower():
            if character.isalnum():
                if separator_pending and normalized:
                    normalized.append(" ")
                normalized.append(character)
                separator_pending = False
            elif (
                unicodedata.category(character).startswith("M")
                and normalized
                and not separator_pending
            ):
                normalized.append(character)
            elif character in {"'", "’"}:
                continue
            elif normalized:
                separator_pending = True
    return unicodedata.normalize("NFC", "".join(normalized))


def load_gold_policy(
    manifest_path: Path, audio_dir: Path
) -> tuple[list[GoldFixture], dict[str, float]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("gold manifest must be a JSON object")
    if set(manifest) != {"version", "thresholds", "fixtures"}:
        raise ValueError("gold manifest fields do not match schema version 2")
    if manifest["version"] != GOLD_MANIFEST_VERSION:
        raise ValueError(
            f"unsupported gold manifest version {manifest['version']!r}; "
            f"expected {GOLD_MANIFEST_VERSION}"
        )

    thresholds = manifest["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != THRESHOLD_FIELDS:
        raise ValueError("gold manifest threshold fields do not match schema version 2")
    for name, value in thresholds.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"thresholds.{name} must be a finite non-negative number")
    if thresholds["baseline_wer_percent"] > thresholds["max_wer_percent"]:
        raise ValueError("baseline WER exceeds the absolute WER limit")
    if thresholds["baseline_cer_percent"] > thresholds["max_cer_percent"]:
        raise ValueError("baseline CER exceeds the absolute CER limit")

    raw_fixtures = manifest["fixtures"]
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ValueError("gold manifest must contain at least one fixture")
    fixtures: list[GoldFixture] = []
    seen_files: set[str] = set()
    for raw_fixture in raw_fixtures:
        if not isinstance(raw_fixture, dict):
            raise TypeError("each gold fixture must be a JSON object")
        if not {"file", "reference"} <= set(raw_fixture) or not set(raw_fixture) <= {
            "file",
            "reference",
            "categories",
        }:
            raise ValueError("gold fixture fields do not match schema version 2")
        file = raw_fixture["file"]
        reference = raw_fixture["reference"]
        categories = raw_fixture.get("categories", [])
        if not isinstance(file, str) or not isinstance(reference, str):
            raise TypeError("gold fixture file and reference must be strings")
        fixture_path = Path(file)
        if (
            not file
            or fixture_path.is_absolute()
            or len(fixture_path.parts) != 1
            or fixture_path.suffix.lower() != ".wav"
        ):
            raise ValueError(
                f"fixture file {file!r} must be one WAV filename relative to --audio-dir"
            )
        if file in seen_files:
            raise ValueError(f"duplicate fixture file {file!r}")
        seen_files.add(file)
        if not (audio_dir / file).is_file():
            raise ValueError(f"fixture audio is missing: {audio_dir / file}")
        if not normalize_lexical(reference):
            raise ValueError(
                f"fixture {file!r} must have non-empty lexical reference text"
            )
        if not isinstance(categories, list) or not all(
            isinstance(category, str) for category in categories
        ):
            raise ValueError(f"fixture {file!r} categories must be strings")
        if any(not category.strip() for category in categories):
            raise ValueError(f"fixture {file!r} has an empty category")
        if len(set(categories)) != len(categories):
            raise ValueError(f"fixture {file!r} repeats a category")
        fixtures.append(
            {"file": file, "reference": reference, "categories": categories}
        )
    return fixtures, {name: float(value) for name, value in thresholds.items()}


def load_gold_fixtures(manifest_path: Path, audio_dir: Path) -> list[GoldFixture]:
    fixtures, _ = load_gold_policy(manifest_path, audio_dir)
    return fixtures


def _manifest_digest(files: dict[str, dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for relative, record in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_gold_input_identity(
    sources_path: Path,
    audio_dir: Path,
    fixtures: Sequence[GoldFixture],
) -> GoldInputIdentity:
    """Verify every gold WAV against its source record before decoding it."""
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if not isinstance(sources, dict) or set(sources) != {
        "version",
        "conversion",
        "sources",
    }:
        raise ValueError("gold sources fields do not match schema version 1")
    if sources["version"] != 1:
        raise ValueError("unsupported gold sources schema version")
    if not isinstance(sources["conversion"], str) or not sources["conversion"]:
        raise ValueError("gold sources conversion must be a non-empty string")
    records = sources["sources"]
    if not isinstance(records, list):
        raise TypeError("gold sources must be a list")

    expected_files = [fixture["file"] for fixture in fixtures]
    observed_files: list[str] = []
    audio_manifest: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each gold source must be a JSON object")
        file = record.get("file")
        expected_sha256 = record.get("sha256")
        if not isinstance(file, str) or not isinstance(expected_sha256, str):
            raise TypeError("gold source file and sha256 must be strings")
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(f"gold source {file!r} has an invalid SHA-256")
        observed_files.append(file)
        audio_path = audio_dir / file
        actual_sha256 = sha256_file(audio_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"gold audio SHA-256 mismatch for {file!r}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        audio_manifest[file] = {
            "bytes": audio_path.stat().st_size,
            "sha256": actual_sha256,
        }
    if observed_files != expected_files:
        raise ValueError("gold sources do not exactly match manifest fixture order")

    return {
        "sources_schema_version": 1,
        "sources_sha256": sha256_file(sources_path),
        "conversion": sources["conversion"],
        "audio_artifact_sha256": _manifest_digest(audio_manifest),
        "audio_artifact_manifest": audio_manifest,
    }


def load_gold_corpus(
    manifest_path: Path,
    audio_dir: Path,
    sources_path: Path,
) -> tuple[list[GoldFixture], dict[str, float], GoldInputIdentity]:
    fixtures, thresholds = load_gold_policy(manifest_path, audio_dir)
    identity = load_gold_input_identity(sources_path, audio_dir, fixtures)
    return fixtures, thresholds, identity


def load_quality_baseline(
    baseline_path: Path,
    manifest_path: Path,
    fixtures: Sequence[GoldFixture],
) -> QualityBaseline:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or set(baseline) != {
        "schema_version",
        "backend",
        "gold_manifest_sha256",
        "source_report",
        "source_report_sha256",
        "wer_percent",
        "cer_percent",
        "categories",
    }:
        raise ValueError("quality baseline fields do not match schema version 1")
    if baseline["schema_version"] != 1:
        raise ValueError("unsupported quality baseline schema version")
    expected_manifest_sha256 = sha256_file(manifest_path)
    if baseline["gold_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("quality baseline was measured against another gold manifest")
    if not isinstance(baseline["backend"], str) or not baseline["backend"]:
        raise ValueError("quality baseline backend must be a non-empty string")
    categories = baseline["categories"]
    if not isinstance(categories, dict) or not categories:
        raise ValueError("quality baseline must contain category scores")
    for name, score in {
        "overall": baseline,
        **categories,
    }.items():
        if not isinstance(score, dict):
            raise TypeError(f"quality baseline score {name!r} must be an object")
        for metric in ("wer_percent", "cer_percent"):
            value = score.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"quality baseline {name!r}.{metric} must be non-negative"
                )
    result: QualityBaseline = {
        "backend": baseline["backend"],
        "wer_percent": float(baseline["wer_percent"]),
        "cer_percent": float(baseline["cer_percent"]),
        "categories": {
            category: {
                "wer_percent": float(score["wer_percent"]),
                "cer_percent": float(score["cer_percent"]),
            }
            for category, score in categories.items()
        },
    }
    source_path = resolve_contained_file(
        baseline_path.parent, baseline["source_report"], max_bytes=16 * 1024 * 1024
    )
    if sha256_file(source_path) != baseline["source_report_sha256"]:
        raise ValueError("quality baseline source report SHA-256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        not isinstance(source, dict)
        or source.get("passed") is not True
        or not isinstance(source.get("overall"), dict)
        or not isinstance(source.get("categories"), dict)
        or source.get("repeatability", {}).get("nondeterministic_outputs") != 0
    ):
        raise ValueError("quality baseline source report is not a passing gold run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("manifest_version") != manifest.get("version"):
        raise ValueError("quality baseline source report manifest version mismatch")
    if source.get("thresholds") != manifest.get("thresholds"):
        raise ValueError("quality baseline source report thresholds mismatch")
    source_repeatability = source["repeatability"]
    repetitions = source_repeatability.get("repetitions")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 10
    ):
        raise ValueError(
            "quality baseline source report must contain at least 10 repetitions"
        )
    source_fixtures = source.get("fixtures")
    if not isinstance(source_fixtures, list):
        raise TypeError("quality baseline source fixtures must be a list")
    observed_fixtures: list[GoldFixture] = []
    for source_fixture in source_fixtures:
        if not isinstance(source_fixture, dict):
            raise TypeError("quality baseline source fixture must be an object")
        if source_fixture.get("repetitions") != repetitions:
            raise ValueError("quality baseline fixture repetition count mismatch")
        observed_fixtures.append(
            {
                "file": source_fixture.get("file"),
                "reference": source_fixture.get("reference"),
                "categories": source_fixture.get("categories"),
            }
        )
    if observed_fixtures != list(fixtures):
        raise ValueError("quality baseline source fixtures do not match the manifest")
    expected_categories = {
        category for fixture in fixtures for category in fixture["categories"]
    }
    if set(result["categories"]) != expected_categories:
        raise ValueError("quality baseline categories do not match the manifest")
    if set(source["categories"]) != expected_categories:
        raise ValueError("quality baseline source categories do not match the manifest")
    if source["overall"].get("fixtures") != len(fixtures):
        raise ValueError("quality baseline source fixture total mismatch")
    source_quality = source["overall"]
    require_fields = {
        "wer_percent": result["wer_percent"],
        "cer_percent": result["cer_percent"],
    }
    for field, expected in require_fields.items():
        if source_quality.get(field) != expected:
            raise ValueError(
                f"quality baseline {field} does not match its source report"
            )
    for category, expected in result["categories"].items():
        source_score = source["categories"].get(category)
        if not isinstance(source_score, dict) or any(
            source_score.get(metric) != expected[metric]
            for metric in ("wer_percent", "cer_percent")
        ):
            raise ValueError(
                f"quality baseline category {category!r} does not match its source report"
            )
    return result


def evaluate_quality_gate(
    *,
    thresholds: dict[str, float],
    wer_percent: float,
    cer_percent: float,
    nondeterministic_outputs: int,
    categories: dict[str, dict[str, float | int]],
    baseline: QualityBaseline,
) -> dict[str, object]:
    failures: list[str] = []
    wer_limit = min(
        thresholds["max_wer_percent"],
        thresholds["baseline_wer_percent"] + thresholds["max_wer_regression_percent"],
    )
    cer_limit = min(
        thresholds["max_cer_percent"],
        thresholds["baseline_cer_percent"] + thresholds["max_cer_regression_percent"],
    )
    if wer_percent > wer_limit:
        failures.append(f"overall WER {wer_percent:.6f}% exceeds {wer_limit:.6f}%")
    if cer_percent > cer_limit:
        failures.append(f"overall CER {cer_percent:.6f}% exceeds {cer_limit:.6f}%")
    if nondeterministic_outputs != 0:
        failures.append(f"observed {nondeterministic_outputs} nondeterministic outputs")
    missing_categories = sorted(set(baseline["categories"]) - set(categories))
    extra_categories = sorted(set(categories) - set(baseline["categories"]))
    if missing_categories:
        failures.append(f"missing categories: {', '.join(missing_categories)}")
    if extra_categories:
        failures.append(
            f"baseline is missing categories: {', '.join(extra_categories)}"
        )
    for category, baseline_score in baseline["categories"].items():
        score = categories.get(category)
        if score is None:
            failures.append(f"missing category {category!r}")
            continue
        for metric in ("wer_percent", "cer_percent"):
            value = float(score[metric])
            limit = baseline_score[metric]
            if value > limit + 1e-9:
                failures.append(
                    f"{category} {metric} {value:.6f}% exceeds "
                    f"{baseline['backend']} {limit:.6f}%"
                )
    return {
        "passed": not failures,
        "thresholds": thresholds,
        "baseline_backend": baseline["backend"],
        "failures": failures,
    }


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def score_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, float | int]:
    word_edits = 0
    reference_words = 0
    char_edits = 0
    reference_chars = 0
    for reference, hypothesis in pairs:
        normalized_reference = normalize_lexical(reference)
        normalized_hypothesis = normalize_lexical(hypothesis)
        reference_word_tokens = normalized_reference.split()
        hypothesis_word_tokens = normalized_hypothesis.split()
        word_edits += edit_distance(reference_word_tokens, hypothesis_word_tokens)
        reference_words += len(reference_word_tokens)
        char_edits += edit_distance(
            list(normalized_reference), list(normalized_hypothesis)
        )
        reference_chars += len(normalized_reference)
    if reference_words == 0 or reference_chars == 0:
        raise ValueError("references must contain lexical words and characters")
    return {
        "reference_words": reference_words,
        "word_edits": word_edits,
        "wer_percent": 100.0 * word_edits / reference_words,
        "reference_chars": reference_chars,
        "char_edits": char_edits,
        "cer_percent": 100.0 * char_edits / reference_chars,
    }


def percentile(values: Sequence[float], percent: int) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent / 100) - 1)
    return ordered[index]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_contained_file(root: Path, value: object, *, max_bytes: int) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("evidence path must be a non-empty relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"evidence path escapes its root: {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(
            f"evidence path is not a regular file inside its root: {value!r}"
        )
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ValueError(f"evidence file is too large ({size} bytes): {value!r}")
    return resolved


def artifact_manifest(model_path: Path) -> tuple[dict[str, dict[str, object]], str]:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(
        (path for path in model_path.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(model_path).as_posix(),
    ):
        relative = path.relative_to(model_path).as_posix()
        file_sha256 = sha256_file(path)
        size = path.stat().st_size
        files[relative] = {"bytes": size, "sha256": file_sha256}
    if not files:
        raise RuntimeError(f"model snapshot contains no files: {model_path}")
    return files, _manifest_digest(files)


def resolve_model_source(model_reference: str, expected_snapshot: str | None) -> Path:
    """Resolve downloads before timed model loading and pin remote revisions."""
    local_path = Path(model_reference)
    if local_path.is_dir() and (local_path / "config.json").is_file():
        return local_path
    if expected_snapshot is None:
        raise ValueError("remote --model requires --expected-snapshot")

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_reference,
            revision=expected_snapshot,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    )


def verify_model_artifact(
    model_path: Path,
    *,
    expected_snapshot: str | None,
    expected_weight_sha256: str | None,
    expected_artifact_sha256: str | None,
) -> tuple[Path, Path, str, dict[str, dict[str, object]], str]:
    """Verify the complete resolved snapshot before any model data is loaded."""
    weight_path = model_path / "model.safetensors"
    if not weight_path.is_file():
        raise RuntimeError(f"model weight is missing: {weight_path}")
    if expected_snapshot is not None and model_path.name != expected_snapshot:
        raise RuntimeError(
            f"model snapshot mismatch: expected {expected_snapshot}, got {model_path.name}"
        )
    weight_sha256 = sha256_file(weight_path)
    if (
        expected_weight_sha256 is not None
        and weight_sha256 != expected_weight_sha256.lower()
    ):
        raise RuntimeError(
            "model weight SHA-256 mismatch: "
            f"expected {expected_weight_sha256.lower()}, got {weight_sha256}"
        )
    files, artifact_sha256 = artifact_manifest(model_path)
    if (
        expected_artifact_sha256 is not None
        and artifact_sha256 != expected_artifact_sha256.lower()
    ):
        raise RuntimeError(
            "model artifact SHA-256 mismatch: "
            f"expected {expected_artifact_sha256.lower()}, got {artifact_sha256}"
        )
    return model_path, weight_path, weight_sha256, files, artifact_sha256


def verify_loaded_model_source(model: object, expected_path: Path) -> None:
    value = getattr(model, "_resolved_model_path", None)
    if not isinstance(value, (str, Path)):
        raise TypeError("pinned MLX oracle did not expose its resolved model path")
    if Path(value).resolve() != expected_path.resolve():
        raise RuntimeError(
            f"oracle loaded {Path(value)!s}; verified snapshot was {expected_path!s}"
        )


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def runtime_versions() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "distributions": {
            distribution: version(distribution)
            for distribution in RUNTIME_DISTRIBUTIONS
        },
    }


def verify_runtime_identity() -> dict[str, object]:
    observed = runtime_versions()
    if observed["python"] != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"expected Python {EXPECTED_PYTHON_VERSION}, got {observed['python']}"
        )
    if observed["distributions"] != EXPECTED_DISTRIBUTIONS:
        raise RuntimeError(
            "runtime distribution versions do not match the benchmark contract: "
            f"{observed['distributions']!r}"
        )
    from importlib.metadata import distribution

    direct_url_text = distribution("mlx-qwen3-asr").read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("mlx-qwen3-asr lacks direct_url.json commit provenance")
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != RUNTIME_COMMIT:
        raise RuntimeError(
            f"expected mlx-qwen3-asr commit {RUNTIME_COMMIT}, got {commit!r}"
        )
    return observed


def verify_audio_runtime_identity(expected_sha256: str) -> dict[str, str]:
    """Pin the ffmpeg binary used to resample the 48 kHz gold WAVs."""
    executable_value = shutil.which("ffmpeg")
    if executable_value is None:
        raise RuntimeError("ffmpeg is required to resample the gold corpus")
    executable = Path(executable_value).resolve()
    observed_sha256 = sha256_file(executable)
    if observed_sha256 != expected_sha256.lower():
        raise RuntimeError(
            "ffmpeg SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {observed_sha256}"
        )
    result = subprocess.run(
        [str(executable), "-version"],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    version_line = result.stdout.splitlines()[0] if result.stdout else ""
    if not version_line.startswith("ffmpeg version "):
        raise RuntimeError("ffmpeg did not report a recognizable version")
    return {
        "executable": str(executable),
        "version": version_line,
        "sha256": observed_sha256,
    }


def evaluator_identity(evaluator_path: Path) -> dict[str, str]:
    helper_path = Path(__file__).resolve()
    evaluator_path = evaluator_path.resolve()
    lock_path = evaluator_path.with_name(f"{evaluator_path.name}.lock")
    return {
        "evaluator_sha256": sha256_file(evaluator_path),
        "shared_helper_sha256": sha256_file(helper_path),
        "uv_lock_sha256": sha256_file(lock_path),
    }
