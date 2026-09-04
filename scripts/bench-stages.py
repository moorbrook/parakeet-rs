#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Aggregate `asr_stages` log lines into a per-length stage-breakdown CSV.

`bench_asr --stage-timings` emits one line per measured repetition:

    [...] INFO  bench_asr asr_stages session_id=bench-5s_48000-r000-... \\
                audio_s=4.967 resample_ms=22.976 windows=1 encoder_calls=1 \\
                decoder_calls=35 joint_calls=96 other_calls=0 mel_ms=3.080 \\
                encoder_ms=25.959 decode_loop_ms=15.840 \\
                decode_loop_dispatch_ms=15.135 decoder_dispatch_ms=5.359 \\
                joint_dispatch_ms=9.776 post_ms=0.066 total_ms=44.944 \\
                boundary_ms=0.404 compute_units=encoder=...,decoder=...

Warmup repetitions (`session_id` starting `warmup-`) are dropped, rows are
bucketed to the nearest of {1, 3, 5, 10, 20} s by measured `audio_s`, and each
numeric field is reduced to its median. Dispatch counts are integers and are
reported as medians too, so a mismatched count shows up rather than being
averaged away.

Uses stdlib only. Invoke via `uv run --quiet scripts/bench-stages.py ...`.
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

STAGE_TAG_RE = re.compile(r"asr_stages\s+(.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")
TARGETS_S = [1, 3, 5, 10, 20]

COUNT_FIELDS = [
    "windows",
    "encoder_calls",
    "decoder_calls",
    "joint_calls",
    "other_calls",
]
TIME_FIELDS = [
    "resample_ms",
    "mel_ms",
    "encoder_ms",
    "decode_loop_ms",
    "decode_loop_dispatch_ms",
    "decoder_dispatch_ms",
    "joint_dispatch_ms",
    "post_ms",
    "total_ms",
    "boundary_ms",
]


def bucket_for(audio_s: float) -> int:
    """Snap measured audio_s to the nearest target bucket."""
    return min(TARGETS_S, key=lambda t: abs(t - audio_s))


def parse_log(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        match = STAGE_TAG_RE.search(line)
        if not match:
            continue
        kv = dict(KV_RE.findall(match.group(1)))
        if kv.get("session_id", "").startswith("warmup-"):
            continue
        try:
            record = {"audio_s": float(kv["audio_s"])}
            for field in COUNT_FIELDS:
                record[field] = int(kv[field])
            for field in TIME_FIELDS:
                record[field] = float(kv[field])
        except (KeyError, ValueError):
            continue
        record["compute_units"] = kv.get("compute_units", "")
        rows.append(record)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    rows = parse_log(args.log)
    if not rows:
        print(f"no asr_stages lines in {args.log}", file=sys.stderr)
        return 1

    by_bucket: dict[int, list[dict]] = {}
    for row in rows:
        by_bucket.setdefault(bucket_for(row["audio_s"]), []).append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = ["target_length_s", "n"] + COUNT_FIELDS + TIME_FIELDS + ["compute_units"]
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for bucket in sorted(by_bucket):
            group = by_bucket[bucket]
            record = [bucket, len(group)]
            for field in COUNT_FIELDS:
                record.append(int(statistics.median(row[field] for row in group)))
            for field in TIME_FIELDS:
                record.append(f"{statistics.median(row[field] for row in group):.3f}")
            record.append(group[0]["compute_units"])
            writer.writerow(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
