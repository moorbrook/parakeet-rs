#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Summarize one `bench_e2e` sweep log into a single CSV row.

Reads the `phase_timer` lines for the latency distribution and the
`bench_e2e_summary` line for the false-cut and transcript-mismatch counts a
percentile cannot express: a repetition that committed early emits no timer
row at all, so latency alone would silently reward cutting.

Appends to `--out`, writing the header only when the file is new.
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

PHASE_RE = re.compile(r"phase_timer\s+(.*)$")
SUMMARY_RE = re.compile(r"bench_e2e_summary\s+(.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")

COLUMNS = [
    "label",
    "fixture",
    "policy",
    "confirmation_ms",
    "reps",
    "false_cuts",
    "mismatches",
    "n",
    "mean_ms",
    "p50_ms",
    "p95_ms",
]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, identical to bench-aggregate.py so
    these rows stay comparable with the published end-to-end numbers."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def parse(path: Path) -> tuple[list[float], dict[str, str]]:
    durations: list[float] = []
    summary: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if match := SUMMARY_RE.search(line):
            summary = dict(KV_RE.findall(match.group(1)))
            continue
        if not (match := PHASE_RE.search(line)):
            continue
        kv = dict(KV_RE.findall(match.group(1)))
        if kv.get("session_id", "").startswith("warmup-"):
            continue
        try:
            durations.append(float(kv["dur_end_to_end_ms"]))
        except (KeyError, ValueError):
            continue
    return durations, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--confirmation-ms", required=True)
    args = ap.parse_args()

    durations, summary = parse(args.log)
    if not summary:
        print(f"no bench_e2e_summary line in {args.log}", file=sys.stderr)
        return 1

    row = {
        "label": args.label,
        "fixture": args.fixture,
        "policy": args.policy,
        "confirmation_ms": args.confirmation_ms,
        "reps": summary.get("reps", ""),
        "false_cuts": summary.get("false_cuts", ""),
        "mismatches": summary.get("mismatches", ""),
        "n": len(durations),
        "mean_ms": f"{statistics.fmean(durations):.1f}" if durations else "",
        "p50_ms": f"{percentile(durations, 0.50):.1f}" if durations else "",
        "p95_ms": f"{percentile(durations, 0.95):.1f}" if durations else "",
    }

    is_new = not args.out.exists()
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"{row['label']:<28} false_cuts={row['false_cuts']}/{row['reps']} "
        f"mismatches={row['mismatches']} p50={row['p50_ms']}ms p95={row['p95_ms']}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
