#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reproduce and verify the checked-in human-speech gold corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "bench" / "gold"
AUDIO = GOLD / "audio"
SOURCES = json.loads((GOLD / "sources.json").read_text())
SLURP_REVISION = "91b0abfee2e735282967ee00d631d6d5f0fb7ff9"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=qmeeus/slurp&config=default&split=train&offset={row}&length=1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def verify(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="parakeet-gold-") as temporary:
        temp = Path(temporary)
        for source in SOURCES["sources"]:
            destination = AUDIO / source["file"]
            if source["corpus"].startswith("OpenSLR"):
                shared = ROOT / "bench" / "endpointing" / source["file"]
                verify(shared, source["sha256"], source["source_id"])
                shutil.copyfile(shared, destination)
                verify(destination, source["sha256"], source["file"])
                print(f"verified {source['file']}")
                continue

            row_number = source["source_row"]
            payload = fetch_json(ROWS_URL.format(row=row_number))
            row = payload["rows"][0]["row"]
            if row["slurp_id"] != source["slurp_id"]:
                raise RuntimeError(f"SLURP row {row_number} ID changed")
            audio_url = row["audio"][0]["src"]
            if SLURP_REVISION not in audio_url:
                raise RuntimeError(
                    f"SLURP row {row_number} is not from pinned revision {SLURP_REVISION}"
                )

            source_flac = temp / f"{row_number}.flac"
            download(audio_url, source_flac)
            verify(source_flac, source["source_sha256"], source["source_id"])
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_flac),
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-c:a",
                    "pcm_s16le",
                    str(destination),
                ],
                check=True,
            )
            verify(destination, source["sha256"], source["file"])
            print(f"verified {source['file']}")


if __name__ == "__main__":
    main()
