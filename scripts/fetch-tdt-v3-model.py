#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests>=2.32"]
# ///
"""Fetch the pinned Parakeet TDT 0.6B v3 Core ML graph set for the kata f0zg bench.

This is an evaluation artifact, not a shipping one. The app's Rust integrity
gate owns the Unified pack; TDT has no Rust-managed manifest, so this script
carries the same discipline in one place: a pinned Hugging Face revision, an
explicit file list, streamed SHA-256 verification, and atomic publication.

Usage:
    scripts/fetch-tdt-v3-model.py            # download and verify against the manifest
    scripts/fetch-tdt-v3-model.py --record   # (re)generate the manifest from a fresh download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

REPO = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
REVISION = "7dd20fe6b1797d35f5e3307e8b1732d9a178edfe"
# FluidAudio derives the on-disk folder from `Repo.parakeetV3.folderName`,
# which strips the `-coreml` suffix. The name is load-bearing: `AsrModels.load`
# re-appends it to the parent of the directory it is handed.
FOLDER = "parakeet-tdt-0.6b-v3"

# The int8 split-frontend v3 set: `ModelNames.ASR.requiredModelsV3(.int8)` plus
# the shared vocabulary. `EncoderInt4`, `Encoder_v2`, `MelEncoder`,
# `JointDecision{,v2}`, `RNNTJoint`, `Parakeet*` and every `.mlpackage` in the
# repository are other conversions FluidAudio's v3 loader never opens.
BUNDLES = [
    "Preprocessor.mlmodelc",
    "Encoder.mlmodelc",
    "Decoder.mlmodelc",
    "JointDecisionv3.mlmodelc",
]
BUNDLE_MEMBERS = [
    "analytics/coremldata.bin",
    "coremldata.bin",
    "metadata.json",
    "model.mil",
    "weights/weight.bin",
]
LOOSE_FILES = ["parakeet_vocab.json", "config.json"]

MANIFEST = Path(__file__).resolve().parent.parent / "bench" / "tdt-v3-model-manifest.json"


def default_model_root() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "com.parakeet.rs"
        / "models"
        / "coreml"
    )


def wanted_paths() -> list[str]:
    paths = [f"{bundle}/{member}" for bundle in BUNDLES for member in BUNDLE_MEMBERS]
    return paths + LOOSE_FILES


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(session: requests.Session, relative: str, destination: Path) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{relative}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=default_model_root())
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    target = args.model_root / FOLDER
    expected: dict[str, dict[str, object]] = {}
    if not args.record:
        if not MANIFEST.exists():
            print(f"missing manifest {MANIFEST}; rerun with --record", file=sys.stderr)
            return 2
        recorded = json.loads(MANIFEST.read_text())
        if recorded["revision"] != REVISION:
            print("manifest revision does not match this script", file=sys.stderr)
            return 2
        expected = recorded["files"]

    session = requests.Session()
    observed: dict[str, dict[str, object]] = {}
    for relative in wanted_paths():
        destination = target / relative
        if not destination.exists():
            print(f"fetching {relative}")
            download(session, relative, destination)
        entry = {"bytes": destination.stat().st_size, "sha256": digest(destination)}
        if expected:
            want = expected.get(relative)
            if want is None:
                print(f"{relative} is not in the manifest", file=sys.stderr)
                return 1
            if want != entry:
                print(
                    f"{relative} does not match the manifest: {entry} != {want}",
                    file=sys.stderr,
                )
                return 1
        observed[relative] = entry

    if args.record:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(
            json.dumps(
                {"repo": REPO, "revision": REVISION, "folder": FOLDER, "files": observed},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"recorded {len(observed)} files to {MANIFEST}")
    else:
        print(f"verified {len(observed)} files under {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
