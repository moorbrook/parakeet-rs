#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Aggregate phase_timer log lines into per-length percentile CSV.

Reads stderr-style lines like:

    [...] INFO  phase_timer mode=bench session_id=bench-5s-r03-... \\
                audio_s=5.024 t_capture_end=0 t_vad_endpoint=0 \\
                t_asr_start=0 t_asr_done=638 t_paste_done=638 \\
                dur_post_endpoint_ms=638

Skips:
    * lines without `phase_timer ` tag
    * lines whose session_id begins with `warmup-` (CoreML graph compile)
    * lines missing audio_s or dur_post_endpoint_ms

Writes CSV columns: mode,target_length_s,n,mean_ms,p50_ms,p95_ms,p99_ms.
Target length = nearest of {1, 3, 5, 10, 20} to the line's audio_s.

Uses stdlib only (statistics, csv, re, pathlib, argparse). Invoke via
`uv run --quiet scripts/bench-aggregate.py ...` — uv resolves the
PEP-723 frontmatter and spins up an interpreter with zero install.
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

PHASE_TAG_RE = re.compile(r"phase_timer\s+(.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")
TARGETS_S = [1, 3, 5, 10, 20]


def bucket_for(audio_s: float) -> int:
    """Snap measured audio_s to the nearest target bucket."""
    return min(TARGETS_S, key=lambda t: abs(t - audio_s))


def parse_log(path: Path):
    rows: list[tuple[str, float, int]] = []
    for line in path.read_text().splitlines():
        m = PHASE_TAG_RE.search(line)
        if not m:
            continue
        kv = dict(KV_RE.findall(m.group(1)))
        sid = kv.get("session_id", "")
        if sid.startswith("warmup-"):
            continue
        try:
            audio_s = float(kv.get("audio_s", "-"))
            dur = int(kv.get("dur_post_endpoint_ms", "-"))
        except ValueError:
            continue
        rows.append((kv.get("mode", "?"), audio_s, dur))
    return rows


def percentile(xs, p: float) -> float:
    """Linear-interpolation percentile (same shape as numpy's default)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    rows = parse_log(args.log)
    if not rows:
        print(f"no phase_timer lines in {args.log}", file=sys.stderr)
        return 1

    by_bucket: dict[tuple[str, int], list[int]] = {}
    for mode, audio_s, dur in rows:
        b = bucket_for(audio_s)
        by_bucket.setdefault((mode, b), []).append(dur)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "target_length_s", "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms"])
        for (mode, b), xs in sorted(by_bucket.items()):
            w.writerow([
                mode, b, len(xs),
                f"{statistics.mean(xs):.1f}",
                f"{percentile(xs, 0.50):.1f}",
                f"{percentile(xs, 0.95):.1f}",
                f"{percentile(xs, 0.99):.1f}",
            ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
