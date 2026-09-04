#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Aggregate `asr_stages` log lines into a per-length stage-breakdown CSV.

`bench_asr --stage-timings` emits one line per measured repetition:

    [...] INFO  bench_asr asr_stages session_id=bench-5s_48000-r000-... \\
                audio_s=4.967 resample_ms=22.976 windows=1 encoder_calls=1 \\
                preprocessor_calls=0 decoder_calls=35 joint_calls=96 \\
                native_decoder_steps=0 native_joint_steps=0 other_calls=0 \\
                mel_ms=3.080 preprocessor_ms=0.000 encoder_ms=25.959 \\
                decode_loop_ms=15.840 \\
                decode_loop_dispatch_ms=15.135 decoder_dispatch_ms=5.359 \\
                decode_loop_native_ms=0.000 \\
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
    "preprocessor_calls",
    "encoder_calls",
    "decoder_calls",
    "joint_calls",
    "native_decoder_steps",
    "native_joint_steps",
    "other_calls",
]
TIME_FIELDS = [
    "resample_ms",
    "mel_ms",
    "preprocessor_ms",
    "encoder_ms",
    "decode_loop_ms",
    "decode_loop_dispatch_ms",
    "decoder_dispatch_ms",
    "joint_dispatch_ms",
    "decode_loop_native_ms",
    "post_ms",
    "overlapped_dispatch_ms",
    "total_ms",
    "boundary_ms",
]
# Emitted only since the TDT v3 comparison added a graph mel front end. A log
# captured before that is still aggregatable; Unified reports zero for both.
OPTIONAL_FIELDS = {"preprocessor_calls", "preprocessor_ms", "overlapped_dispatch_ms"}


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
                if field in OPTIONAL_FIELDS and field not in kv:
                    record[field] = 0
                    continue
                record[field] = int(kv[field])
            for field in TIME_FIELDS:
                if field in OPTIONAL_FIELDS and field not in kv:
                    record[field] = 0.0
                    continue
                record[field] = float(kv[field])
        except (KeyError, ValueError):
            continue
        record["compute_units"] = kv.get("compute_units", "")
        rows.append(record)
    return rows


def validate(by_bucket: dict[int, list[dict]]) -> list[str]:
    """Reject rows that cannot describe a Parakeet Core ML pipeline.

    A stage whose Core ML entry point stops being intercepted reports zero cost
    while every other number stays plausible, so the invariants are checked here
    as well as in `bench_asr`: aggregating a broken run into a CSV is how a
    wrong baseline gets published.
    """
    problems = []
    for bucket, group in sorted(by_bucket.items()):
        for row in group:
            if row["encoder_calls"] < 1:
                problems.append(
                    f"{bucket}s bucket: no encoder dispatches were intercepted, so encoder "
                    f"and mel time are missing"
                )
                break
            if row["windows"] != row["encoder_calls"]:
                problems.append(
                    f"{bucket}s bucket: {row['windows']} windows against "
                    f"{row['encoder_calls']} encoder calls"
                )
                break
            if row["preprocessor_calls"] not in (0, row["windows"]):
                problems.append(
                    f"{bucket}s bucket: {row['preprocessor_calls']} mel front-end dispatches "
                    f"against {row['windows']} windows"
                )
                break
            if row["overlapped_dispatch_ms"] > 0:
                problems.append(
                    f"{bucket}s bucket: {row['overlapped_dispatch_ms']:.3f} ms of overlapping "
                    f"Core ML dispatch, so the per-stage columns double-count"
                )
                break
            if row["other_calls"] != 0:
                problems.append(
                    f"{bucket}s bucket: {row['other_calls']} unattributed Core ML predictions"
                )
                break
            if row["decoder_calls"] and row["native_decoder_steps"]:
                problems.append(
                    f"{bucket}s bucket: {row['decoder_calls']} Core ML decoder calls and "
                    f"{row['native_decoder_steps']} native steps in one utterance"
                )
                break
            # The loop runs on one engine or the other, so the frame identity
            # holds over whichever pair of counters is populated.
            decoder_steps = row["decoder_calls"] + row["native_decoder_steps"]
            joint_steps = row["joint_calls"] + row["native_joint_steps"]
            frames = joint_steps - (decoder_steps - row["windows"])
            if frames < 1:
                problems.append(f"{bucket}s bucket: implied decoded-frame count {frames}")
                break
    return problems


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

    problems = validate(by_bucket)
    if problems:
        for problem in problems:
            print(f"stage report is not trustworthy: {problem}", file=sys.stderr)
        return 1

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
