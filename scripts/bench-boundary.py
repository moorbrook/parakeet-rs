#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Summarize worker-internal time versus the outer ASR call boundary."""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

BOUNDARY_RE = re.compile(r"asr_boundary\s+(.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")
BUCKET_RE = re.compile(r"^bench-(\d+)s_")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    buckets: dict[int, list[tuple[float, float, float]]] = {}
    for line in path.read_text().splitlines():
        match = BOUNDARY_RE.search(line)
        if not match:
            continue
        fields = dict(KV_RE.findall(match.group(1)))
        session_id = fields.get("session_id", "")
        bucket_match = BUCKET_RE.match(session_id)
        if not bucket_match or session_id.startswith("warmup-"):
            continue
        try:
            row = (
                float(fields["internal_ms"]),
                float(fields["wall_ms"]),
                float(fields["boundary_ms"]),
            )
        except (KeyError, ValueError):
            continue
        buckets.setdefault(int(bucket_match.group(1)), []).append(row)
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1
    buckets = parse(args.log)
    if not buckets:
        print(f"no measured asr_boundary lines in {args.log}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "target_length_s",
                "n",
                "internal_mean_ms",
                "internal_p50_ms",
                "internal_p95_ms",
                "wall_mean_ms",
                "wall_p50_ms",
                "wall_p95_ms",
                "boundary_mean_ms",
                "boundary_p50_ms",
                "boundary_p95_ms",
            ]
        )
        for bucket, rows in sorted(buckets.items()):
            internal, wall, boundary = (list(values) for values in zip(*rows, strict=True))
            writer.writerow(
                [
                    bucket,
                    len(rows),
                    f"{statistics.mean(internal):.3f}",
                    f"{percentile(internal, 0.50):.3f}",
                    f"{percentile(internal, 0.95):.3f}",
                    f"{statistics.mean(wall):.3f}",
                    f"{percentile(wall, 0.50):.3f}",
                    f"{percentile(wall, 0.95):.3f}",
                    f"{statistics.mean(boundary):.3f}",
                    f"{percentile(boundary, 0.50):.3f}",
                    f"{percentile(boundary, 0.95):.3f}",
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
