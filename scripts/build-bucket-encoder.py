#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build a short-window offline Parakeet Unified encoder for bucketed dispatch.

The shipped offline encoder is compiled at one fixed 15 s mel window and
`UnifiedAsrManager` zero-pads every utterance to it, so a 1 s dictation
utterance pays the whole 25.5 ms encoder. This exports the same checkpoint at a
shorter window, quantizes it with the upstream int8 recipe, and compiles it to
the `.mlmodelc` the worker loads.

The conversion itself is FluidInference's `mobius` pipeline, run unmodified
apart from one added flag. It is a uv project with its own lockfile and a NeMo
git overlay that PEP 723 cannot express, so this script drives that project
rather than declaring the dependencies itself: it clones mobius at a pinned
commit, reproduces the README's environment steps, verifies the checkpoint by
SHA-256, and records the provenance of everything it produced.

    scripts/build-bucket-encoder.py --seconds 5

Output lands in `--work-dir` (default: a cache under the user's home) as
`parakeet_unified_encoder_<N>s_int8.mlmodelc`, plus `provenance.json`. Copy it
into the model directory the worker is pointed at; the worker discovers bucket
encoders by filename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MOBIUS_REPO = "https://github.com/FluidInference/mobius.git"
MOBIUS_COMMIT = "4040a39f760290bb1d43a72dc82e5894f27b0f5c"
MOBIUS_SUBDIR = "models/stt/parakeet-unified-en-0.6b/coreml"
NEMO_GIT = (
    "nemo_toolkit @ git+https://github.com/NVIDIA-NeMo/NeMo.git"
    "@95f92737cfb8ee0123bb328b07a2d24c6d859aff"
)
CHECKPOINT_URL = (
    "https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/"
    "parakeet-unified-en-0.6b.nemo"
)
CHECKPOINT_SHA256 = "ec23ed9150c8fde49072c3e2d61678ab903dbcef389d658db833420cbc1da35b"

# convert-coreml.py hardcodes the offline window. Upstream takes every other
# export parameter on the command line, so this adds the one that is missing
# rather than forking the file.
WINDOW_PATCH_FROM = "        max_audio_seconds=15.0,\n"
WINDOW_PATCH_TO = "        max_audio_seconds=args.max_audio_seconds,\n"
ARG_PATCH_FROM = '    args = parser.parse_args()\n\n    settings = ExportSettings('
ARG_PATCH_TO = (
    '    parser.add_argument("--max-audio-seconds", type=float, default=15.0)\n'
    "    args = parser.parse_args()\n\n    settings = ExportSettings("
)

QUANTIZE_SOURCE = '''
import argparse, sys
from pathlib import Path
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
)

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()

# The upstream recipe from quantize_int8.py, applied to one encoder.
config = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", granularity="per_channel", dtype="int8"
    )
)
model = ct.models.MLModel(str(args.source), compute_units=ct.ComputeUnit.CPU_ONLY)
linear_quantize_weights(model, config).save(str(args.destination))
print(f"saved {args.destination}")
'''


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    # This script runs under `uv run --script`, which exports VIRTUAL_ENV for
    # its own ephemeral environment. `uv pip install` honours that variable, so
    # leaving it set silently installs the NeMo overlay into this script's
    # environment instead of the mobius project's .venv, and the export then
    # fails on `att_chunk_context_size` as if the overlay had never run.
    environment = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(root: Path) -> str:
    """A stable digest over a bundle: every file's relative path and content."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def prepare_mobius(work_dir: Path) -> Path:
    checkout = work_dir / "mobius"
    if not checkout.exists():
        run(["git", "clone", "--quiet", MOBIUS_REPO, str(checkout)])
    run(["git", "-C", str(checkout), "fetch", "--quiet", "origin", MOBIUS_COMMIT])
    run(["git", "-C", str(checkout), "checkout", "--quiet", MOBIUS_COMMIT])
    run(["git", "-C", str(checkout), "checkout", "--quiet", "--", "."])

    project = checkout / MOBIUS_SUBDIR
    run(["uv", "sync"], cwd=project)
    # NeMo main is required for this checkpoint's attention config, and its own
    # pyproject pins torch to an index uv will not resolve against, hence
    # --no-deps. `uv run --no-sync` afterwards keeps the overlay in place.
    run(["uv", "pip", "install", "--no-deps", "--force-reinstall", NEMO_GIT], cwd=project)

    convert = project / "convert-coreml.py"
    text = convert.read_text()
    if "--max-audio-seconds" not in text:
        if WINDOW_PATCH_FROM not in text or ARG_PATCH_FROM not in text:
            sys.exit(
                "convert-coreml.py no longer matches the expected shape; "
                f"re-check mobius at {MOBIUS_COMMIT}"
            )
        text = text.replace(ARG_PATCH_FROM, ARG_PATCH_TO)
        text = text.replace(WINDOW_PATCH_FROM, WINDOW_PATCH_TO)
        convert.write_text(text)
    return project


def fetch_checkpoint(project: Path) -> Path:
    checkpoint = project / "parakeet-unified-en-0.6b.nemo"
    if not checkpoint.exists():
        run(["curl", "-L", "--retry", "3", "-o", str(checkpoint), CHECKPOINT_URL])
    digest = sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        sys.exit(f"checkpoint SHA-256 is {digest}, expected {CHECKPOINT_SHA256}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds", type=int, required=True, help="encoder window in whole seconds"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.home() / ".cache" / "parakeet-bucket-encoders",
        help="where the mobius checkout, checkpoint and build products live",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.seconds <= 15:
        sys.exit("--seconds must be between 1 and 15")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    work_dir = arguments.work_dir.expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    project = prepare_mobius(work_dir)
    checkpoint = fetch_checkpoint(project)

    export_dir = work_dir / f"export-{arguments.seconds}s"
    package = export_dir / "parakeet_unified_encoder.mlpackage"
    if not package.exists():
        run(
            [
                "uv", "run", "--no-sync", "python", "convert-coreml.py",
                "--nemo-path", str(checkpoint),
                "--output-dir", str(export_dir),
                "--skip-streaming",
                "--max-audio-seconds", str(float(arguments.seconds)),
            ],
            cwd=project,
        )

    quantized = export_dir / f"parakeet_unified_encoder_{arguments.seconds}s_int8.mlpackage"
    if not quantized.exists():
        script = work_dir / "quantize_one_encoder.py"
        script.write_text(QUANTIZE_SOURCE)
        run(
            [
                "uv", "run", "--no-sync", "python", str(script),
                "--source", str(package),
                "--destination", str(quantized),
            ],
            cwd=project,
        )

    compiled_name = f"parakeet_unified_encoder_{arguments.seconds}s_int8.mlmodelc"
    compiled = work_dir / compiled_name
    if compiled.exists():
        shutil.rmtree(compiled)
    # coremltools writes .mlpackage; the worker loads compiled .mlmodelc, the
    # same form the Hugging Face artifacts ship in.
    run(["xcrun", "coremlcompiler", "compile", str(quantized), str(work_dir)])
    produced = work_dir / f"{quantized.stem}.mlmodelc"
    if produced != compiled:
        produced.rename(compiled)

    provenance = {
        "window_seconds": arguments.seconds,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {
            "url": CHECKPOINT_URL,
            "sha256": CHECKPOINT_SHA256,
            "bytes": checkpoint.stat().st_size,
        },
        "converter": {
            "repository": MOBIUS_REPO,
            "commit": MOBIUS_COMMIT,
            "subdirectory": MOBIUS_SUBDIR,
            "nemo": NEMO_GIT,
            "local_change": "convert-coreml.py gains a --max-audio-seconds flag",
        },
        "quantization": "linear_symmetric, per_channel, int8 (upstream quantize_int8.py recipe)",
        "artifact": {
            "name": compiled_name,
            "sha256": directory_sha256(compiled),
            "bytes": sum(p.stat().st_size for p in compiled.rglob("*") if p.is_file()),
        },
    }
    record = work_dir / f"provenance-{arguments.seconds}s.json"
    record.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    print(f"\nartifact: {compiled}")
    print(f"provenance: {record}")


if __name__ == "__main__":
    main()
