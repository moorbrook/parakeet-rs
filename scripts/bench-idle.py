#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Aggregate the idle re-wake A/B (kata snx0) into a per-arm percentile CSV.

`scripts/bench-aggregate.py` buckets by (mode, length) and would merge every
arm of this experiment into one row. This one attributes each `phase_timer`
line to the arm announced by the most recent `idle_arm` marker, which is how
`bench_asr` and `bench_e2e` tag their arms without adding a field to the
PhaseTimer format shared with the production path.

Reads two line shapes:

    [...] INFO  idle_arm arm=prime idle_gap_ms=60000 record_gap_ms=audio \\
                keepalive_ms=250
    [...] INFO  phase_timer mode=bench session_id=bench-5s-prime-g60000-r003-... \\
                audio_s=5.024 [...] dur_post_endpoint_ms=64

and, when the stage profiler is on, splits out the encoder so a cold decode's
Neural Engine re-wake can be told apart from a cold CPU:

    [...] INFO  asr_stages session_id=... encoder_ms=25.959 [...]

Skips lines whose session_id begins with `warmup-`, and any `phase_timer`
line that appears before the first `idle_arm` marker: without a marker the
arm is unknown, and guessing one would silently mislabel a row.

Writes: arm,idle_gap_ms,target_length_s,n,mean_ms,p50_ms,p95_ms,p99_ms,
        encoder_p50_ms,encoder_p95_ms.
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

TAG_RE = re.compile(r"\b(idle_arm|phase_timer|asr_stages)\s+(.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")
TARGETS_S = [1, 3, 5, 10, 20]
METRIC_FIELDS = {
    "post-endpoint": "dur_post_endpoint_ms",
    "end-to-end": "dur_end_to_end_ms",
}


def bucket_for(audio_s: float) -> int:
    """Snap measured audio_s to the nearest target bucket."""
    return min(TARGETS_S, key=lambda t: abs(t - audio_s))


def percentile(xs, p: float) -> float:
    """Linear-interpolation percentile, matching scripts/bench-aggregate.py."""
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


def parse_log(path: Path, duration_field: str):
    """Return (latency_rows, encoder_rows, unlabelled_count).

    Both row lists are keyed the same way, so a run without the stage
    profiler simply produces empty encoder columns rather than dropping rows.
    """
    arm = None
    gap = None
    latency: list[tuple[str, int, float, int]] = []
    encoder: list[tuple[str, int, float, float]] = []
    unlabelled = 0

    for line in path.read_text().splitlines():
        m = TAG_RE.search(line)
        if not m:
            continue
        tag, rest = m.group(1), m.group(2)
        kv = dict(KV_RE.findall(rest))

        if tag == "idle_arm":
            arm = kv.get("arm", "?")
            try:
                gap = int(kv.get("idle_gap_ms", "-"))
            except ValueError:
                gap = None
            continue

        sid = kv.get("session_id", "")
        if sid.startswith("warmup-"):
            continue
        if arm is None or gap is None:
            unlabelled += 1
            continue
        try:
            audio_s = float(kv.get("audio_s", "-"))
        except ValueError:
            continue

        if tag == "phase_timer":
            try:
                latency.append((arm, gap, audio_s, int(kv[duration_field])))
            except (KeyError, ValueError):
                continue
        else:
            try:
                encoder.append((arm, gap, audio_s, float(kv["encoder_ms"])))
            except (KeyError, ValueError):
                continue

    return latency, encoder, unlabelled


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--metric",
        choices=METRIC_FIELDS,
        default="post-endpoint",
        help="duration to aggregate (default: post-endpoint)",
    )
    args = ap.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    latency, encoder, unlabelled = parse_log(args.log, METRIC_FIELDS[args.metric])
    if unlabelled:
        # Loud, because the usual cause is a binary that emits phase_timer
        # lines without the marker — every one of those rows is missing from
        # the table below.
        print(
            f"warning: {unlabelled} timed lines had no preceding idle_arm marker "
            "and were dropped",
            file=sys.stderr,
        )
    if not latency:
        print(f"no arm-labelled phase_timer lines in {args.log}", file=sys.stderr)
        return 1

    by_key: dict[tuple[str, int, int], list[int]] = {}
    for arm, gap, audio_s, dur in latency:
        by_key.setdefault((arm, gap, bucket_for(audio_s)), []).append(dur)
    enc_by_key: dict[tuple[str, int, int], list[float]] = {}
    for arm, gap, audio_s, ms in encoder:
        enc_by_key.setdefault((arm, gap, bucket_for(audio_s)), []).append(ms)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arm", "idle_gap_ms", "target_length_s", "n",
            "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "encoder_p50_ms", "encoder_p95_ms",
        ])
        for key, xs in sorted(by_key.items()):
            arm, gap, bucket = key
            enc = enc_by_key.get(key, [])
            w.writerow([
                arm, gap, bucket, len(xs),
                f"{statistics.mean(xs):.1f}",
                f"{percentile(xs, 0.50):.1f}",
                f"{percentile(xs, 0.95):.1f}",
                f"{percentile(xs, 0.99):.1f}",
                f"{percentile(enc, 0.50):.2f}" if enc else "",
                f"{percentile(enc, 0.95):.2f}" if enc else "",
            ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
