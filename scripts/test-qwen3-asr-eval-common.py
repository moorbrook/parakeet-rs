#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Unit tests for the dependency-free Qwen3-ASR evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qwen3_asr_eval_common import (
    artifact_manifest,
    evaluate_quality_gate,
    load_gold_fixtures,
    load_gold_input_identity,
    load_quality_baseline,
    normalize_lexical,
    percentile,
    resolve_model_source,
    score_pairs,
    verify_loaded_model_source,
    verify_model_artifact,
)


class DummyModel:
    def __init__(self, model_path: Path) -> None:
        self._resolved_model_path = str(model_path)


class Qwen3AsrEvalCommonTests(unittest.TestCase):
    def test_normalization_matches_unicode_and_apostrophe_policy(self) -> None:
        self.assertEqual(normalize_lexical("  CAFÉ—don't!  "), "café dont")
        self.assertEqual(normalize_lexical("Cafe\u0301"), "café")
        self.assertEqual(normalize_lexical("ΟΣ"), "οσ")

    def test_score_pairs_aggregates_word_and_character_edits(self) -> None:
        result = score_pairs([("hello world", "hello there"), ("café", "cafe")])
        self.assertEqual(result["reference_words"], 3)
        self.assertEqual(result["word_edits"], 2)
        self.assertEqual(result["reference_chars"], 15)
        self.assertEqual(result["char_edits"], 6)

    def test_score_pairs_rejects_empty_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "references must contain"):
            score_pairs([("...", "anything")])

    def test_percentile_matches_rust_nearest_rank_policy(self) -> None:
        values = [float(value) for value in range(1, 11)]
        self.assertEqual(percentile(values, 50), 5.0)
        self.assertEqual(percentile(values, 95), 10.0)

    def test_artifact_identity_accepts_exact_snapshot_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "snapshot-id"
            model_path.mkdir()
            weight_path = model_path / "model.safetensors"
            weight_path.write_bytes(b"weights")
            expected = hashlib.sha256(b"weights").hexdigest()
            _, artifact_sha256 = artifact_manifest(model_path)

            resolved, weight, digest, files, whole_digest = verify_model_artifact(
                model_path,
                expected_snapshot="snapshot-id",
                expected_weight_sha256=expected,
                expected_artifact_sha256=artifact_sha256,
            )

            self.assertEqual(resolved, model_path)
            self.assertEqual(weight, weight_path)
            self.assertEqual(digest, expected)
            self.assertEqual(set(files), {"model.safetensors"})
            self.assertEqual(whole_digest, artifact_sha256)

    def test_artifact_identity_rejects_moved_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "different-snapshot"
            model_path.mkdir()
            (model_path / "model.safetensors").write_bytes(b"weights")

            with self.assertRaisesRegex(RuntimeError, "snapshot mismatch"):
                verify_model_artifact(
                    model_path,
                    expected_snapshot="expected-snapshot",
                    expected_weight_sha256=None,
                    expected_artifact_sha256=None,
                )

    def test_artifact_identity_rejects_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "snapshot-id"
            model_path.mkdir()
            (model_path / "model.safetensors").write_bytes(b"weights")

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_model_artifact(
                    model_path,
                    expected_snapshot="snapshot-id",
                    expected_weight_sha256="0" * 64,
                    expected_artifact_sha256=None,
                )

    def test_artifact_identity_covers_configuration_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "snapshot-id"
            model_path.mkdir()
            (model_path / "model.safetensors").write_bytes(b"weights")
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "vocab.json").write_text("{}", encoding="utf-8")
            _, expected = artifact_manifest(model_path)
            (model_path / "vocab.json").write_text('{"changed":true}', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 mismatch"):
                verify_model_artifact(
                    model_path,
                    expected_snapshot="snapshot-id",
                    expected_weight_sha256=None,
                    expected_artifact_sha256=expected,
                )

    def test_loaded_model_must_match_the_preverified_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "expected"
            different = root / "different"
            expected.mkdir()
            different.mkdir()
            verify_loaded_model_source(DummyModel(expected), expected)
            with self.assertRaisesRegex(RuntimeError, "verified snapshot"):
                verify_loaded_model_source(DummyModel(different), expected)

    def test_local_model_source_does_not_require_hub_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "snapshot-id"
            model_path.mkdir()
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_model_source(str(model_path), None), model_path)

    def test_manifest_validation_accepts_checked_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "fixture.wav").write_bytes(b"RIFF")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(self._valid_manifest()), encoding="utf-8"
            )

            self.assertEqual(
                load_gold_fixtures(manifest_path, audio_dir),
                [
                    {
                        "file": "fixture.wav",
                        "reference": "Hello world.",
                        "categories": ["general"],
                    }
                ],
            )

    def test_manifest_validation_rejects_unsafe_or_ambiguous_fixtures(self) -> None:
        cases = {
            "traversal": [{"file": "../fixture.wav"}],
            "duplicate": [{}, {}],
            "empty reference": [{"reference": "..."}],
            "empty category": [{"categories": [""]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "fixture.wav").write_bytes(b"RIFF")
            manifest_path = root / "manifest.json"
            for name, overrides in cases.items():
                with self.subTest(name=name):
                    manifest = self._valid_manifest()
                    manifest["fixtures"] = [
                        {**manifest["fixtures"][0], **override}
                        for override in overrides
                    ]
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_gold_fixtures(manifest_path, audio_dir)

    def test_manifest_validation_rejects_missing_fields_files_and_fixtures(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "fixture.wav").write_bytes(b"RIFF")
            manifest_path = root / "manifest.json"

            invalid_manifests = []
            empty = self._valid_manifest()
            empty["fixtures"] = []
            invalid_manifests.append(empty)

            missing_field = self._valid_manifest()
            del missing_field["fixtures"][0]["reference"]
            invalid_manifests.append(missing_field)

            absolute_path = self._valid_manifest()
            absolute_path["fixtures"][0]["file"] = str(
                (audio_dir / "fixture.wav").resolve()
            )
            invalid_manifests.append(absolute_path)

            missing_audio = self._valid_manifest()
            missing_audio["fixtures"][0]["file"] = "missing.wav"
            invalid_manifests.append(missing_audio)

            for manifest in invalid_manifests:
                with self.subTest(manifest=manifest):
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_gold_fixtures(manifest_path, audio_dir)

    def test_gold_input_identity_binds_every_audio_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            audio_path = audio_dir / "fixture.wav"
            audio_path.write_bytes(b"RIFF-pinned-audio")
            fixtures = self._valid_manifest()["fixtures"]
            sources_path = root / "sources.json"
            sources_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "conversion": "test conversion",
                        "sources": [
                            {
                                "file": "fixture.wav",
                                "sha256": hashlib.sha256(
                                    b"RIFF-pinned-audio"
                                ).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            identity = load_gold_input_identity(sources_path, audio_dir, fixtures)
            self.assertEqual(
                identity["audio_artifact_manifest"]["fixture.wav"]["bytes"],
                len(b"RIFF-pinned-audio"),
            )
            audio_path.write_bytes(b"RIFF-tampered")
            with self.assertRaisesRegex(ValueError, "audio SHA-256 mismatch"):
                load_gold_input_identity(sources_path, audio_dir, fixtures)

    def test_quality_baseline_binds_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest = self._valid_manifest()
            manifest_text = json.dumps(manifest) + "\n"
            manifest_path.write_text(manifest_text, encoding="utf-8")
            baseline_path = root / "baseline.json"
            report_path = root / "report.json"
            report = {
                "manifest_version": 2,
                "thresholds": manifest["thresholds"],
                "passed": True,
                "overall": {
                    "fixtures": 1,
                    "wer_percent": 1.0,
                    "cer_percent": 2.0,
                },
                "repeatability": {
                    "repetitions": 10,
                    "nondeterministic_outputs": 0,
                },
                "categories": {"general": {"wer_percent": 1.0, "cer_percent": 2.0}},
                "fixtures": [
                    {
                        **manifest["fixtures"][0],
                        "repetitions": 10,
                    }
                ],
            }
            report_bytes = json.dumps(report).encode()
            report_path.write_bytes(report_bytes)
            baseline = {
                "schema_version": 1,
                "backend": "baseline",
                "gold_manifest_sha256": hashlib.sha256(
                    manifest_text.encode()
                ).hexdigest(),
                "source_report": "report.json",
                "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "wer_percent": 1.0,
                "cer_percent": 2.0,
                "categories": {"general": {"wer_percent": 1.0, "cer_percent": 2.0}},
            }
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            self.assertEqual(
                load_quality_baseline(
                    baseline_path, manifest_path, manifest["fixtures"]
                )["backend"],
                "baseline",
            )
            manifest_path.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another gold manifest"):
                load_quality_baseline(
                    baseline_path, manifest_path, manifest["fixtures"]
                )

    def test_quality_baseline_rejects_escaping_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            baseline = {
                "schema_version": 1,
                "backend": "baseline",
                "gold_manifest_sha256": hashlib.sha256(b"{}\n").hexdigest(),
                "source_report": "../outside.json",
                "source_report_sha256": "0" * 64,
                "wer_percent": 1.0,
                "cer_percent": 2.0,
                "categories": {"general": {"wer_percent": 1.0, "cer_percent": 2.0}},
            }
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_quality_baseline(baseline_path, manifest_path, [])

    def test_quality_gate_rejects_category_regression(self) -> None:
        result = evaluate_quality_gate(
            thresholds=self._valid_manifest()["thresholds"],
            wer_percent=4.0,
            cer_percent=4.0,
            nondeterministic_outputs=0,
            categories={"general": {"wer_percent": 1.1, "cer_percent": 0.5}},
            baseline={
                "backend": "baseline",
                "wer_percent": 5.0,
                "cer_percent": 5.0,
                "categories": {"general": {"wer_percent": 1.0, "cer_percent": 1.0}},
            },
        )
        self.assertFalse(result["passed"])
        self.assertIn("general wer_percent", result["failures"][0])

    def test_quality_gate_requires_identical_category_sets(self) -> None:
        result = evaluate_quality_gate(
            thresholds=self._valid_manifest()["thresholds"],
            wer_percent=1.0,
            cer_percent=1.0,
            nondeterministic_outputs=0,
            categories={
                "general": {"wer_percent": 0.0, "cer_percent": 0.0},
                "new": {"wer_percent": 0.0, "cer_percent": 0.0},
            },
            baseline={
                "backend": "baseline",
                "wer_percent": 5.0,
                "cer_percent": 5.0,
                "categories": {"general": {"wer_percent": 1.0, "cer_percent": 1.0}},
            },
        )
        self.assertFalse(result["passed"])
        self.assertIn("baseline is missing categories: new", result["failures"])

    @staticmethod
    def _valid_manifest() -> dict[str, object]:
        return {
            "version": 2,
            "thresholds": {
                "max_wer_percent": 10.0,
                "max_cer_percent": 10.0,
                "baseline_wer_percent": 5.0,
                "baseline_cer_percent": 5.0,
                "max_wer_regression_percent": 1.0,
                "max_cer_regression_percent": 1.0,
            },
            "fixtures": [
                {
                    "file": "fixture.wav",
                    "reference": "Hello world.",
                    "categories": ["general"],
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
