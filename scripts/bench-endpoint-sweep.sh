#!/usr/bin/env bash
# Confirmation-window sweep for Tap: latency against false early cuts.
#
# Each row replays a fixture through production capture, dual VAD, speculative
# Core ML inference, and shutdown, with one explicit pair of silence thresholds.
# `bench_e2e --tolerate-false-cuts` counts early commits instead of aborting,
# because a rate needs every repetition and restarting the process per
# repetition would reload and re-warm the Core ML worker.
#
# Latency alone cannot decide this: a shorter window always looks faster,
# and a repetition that cut early emits no timer row at all. Read the
# false_cuts column first.
#
# Requires the duplex "BlackHole 2ch" loopback. Never changes system defaults.

set -euo pipefail

cd "$(dirname "$0")/.."

REPS="${REPS:-15}"
WARMUP_REPS="${WARMUP_REPS:-2}"
DEVICE="${DEVICE:-BlackHole 2ch}"
OUT_DIR="${OUT_DIR:-bench/endpoint-sweep}"
CSV="${CSV:-bench/endpoint-sweep.csv}"

FIVE_S="bench/audio/5s_48000.wav"
FIVE_S_EXPECTED="The latency benchmark measures end to end speech recognition pipeline performance."
SINGLE="bench/endpointing/librispeech-single-6930-75918-0000-48000.wav"
SINGLE_EXPECTED="CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"
MULTI="bench/endpointing/librispeech-multi-6930-75918-0001-48000.wav"
MULTI_EXPECTED="THE ENGLISH FORWARDED TO THE FRENCH BASKETS OF FLOWERS OF WHICH THEY HAD MADE A PLENTIFUL PROVISION TO GREET THE ARRIVAL OF THE YOUNG PRINCESS THE FRENCH IN RETURN INVITED THE ENGLISH TO A SUPPER WHICH WAS TO BE GIVEN THE NEXT DAY"

command -v uv >/dev/null || { echo "missing uv" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$(dirname "$FIVE_S")"
rm -f "$CSV"

# bench/audio is generated, not versioned. Synthesize the five-second fixture
# exactly the way scripts/bench-latency.sh does so a fresh worktree produces
# the same audio the published end-to-end numbers used.
if [[ ! -f "$FIVE_S" ]]; then
    command -v say >/dev/null || { echo "missing macOS \`say\`" >&2; exit 1; }
    command -v afconvert >/dev/null || { echo "missing macOS \`afconvert\`" >&2; exit 1; }
    echo "Generating $FIVE_S (48000 Hz mono PCM16)..."
    say -o bench/audio/5s.aiff "$FIVE_S_EXPECTED"
    afconvert -f WAVE -d "LEI16@48000" -c 1 bench/audio/5s.aiff "$FIVE_S"
    rm bench/audio/5s.aiff
fi

scripts/build-coreml-worker.sh
cargo build --release --locked --bin bench_e2e

# label fixture policy confirmation_ms punctuated_ms
run_row() {
    local label="$1" fixture="$2" policy="$3" confirmation="$4" punctuated="$5"
    # Every row carries a reference transcript. The acoustic-end marker fires
    # at the fixture's last sample above -80 dBFS, and the LibriSpeech room
    # tone sits above that floor, so a short window can miss the marker with
    # every word intact. `mismatches` is the oracle for lost speech;
    # `false_cuts` is the marker, and the two are reported separately.
    # The LibriSpeech references are the corpus text, not this model's output,
    # so a row's mismatch count is read against its own control row rather
    # than against zero.
    #
    # bash 3.2 treats an empty array under `set -u` as unbound, so this array
    # is never allowed to be empty.
    local wav reference
    case "$fixture" in
        5s) wav="$FIVE_S"; reference="$FIVE_S_EXPECTED" ;;
        single) wav="$SINGLE"; reference="$SINGLE_EXPECTED" ;;
        multi) wav="$MULTI"; reference="$MULTI_EXPECTED" ;;
        *) echo "unknown fixture: $fixture" >&2; exit 1 ;;
    esac
    [[ -f "$wav" ]] || { echo "missing fixture: $wav" >&2; exit 1; }

    local log="$OUT_DIR/${label}.log"
    RUST_LOG=info ./target/release/bench_e2e \
        --backend coreml-unified \
        --strategy speculative \
        --endpoint-policy "$policy" \
        --confirmation-ms "$confirmation" \
        --punctuated-ms "$punctuated" \
        --tolerate-false-cuts \
        --device "$DEVICE" \
        --wav "$wav" \
        --expected "$reference" \
        --warmup-reps "$WARMUP_REPS" \
        --reps "$REPS" \
        2>"$log"

    uv run --quiet scripts/bench-endpoint-sweep.py \
        --log "$log" --out "$CSV" --label "$label" --fixture "$fixture" \
        --policy "$policy" --confirmation-ms "$confirmation" \
        --punctuated-ms "$punctuated"
}

echo "== Tap Fast confirmation curve (5 s fixture) =="
run_row fast-150-off   5s fast 150 off
run_row fast-120-off   5s fast 120 off
run_row fast-90-off    5s fast 90  off
run_row fast-60-off    5s fast 60  off

echo
echo "== Punctuation-aware commit behind the unchanged 150 ms window (5 s) =="
run_row fast-150-p120  5s fast 150 120
run_row fast-150-p90   5s fast 150 90
run_row fast-150-p60   5s fast 150 60

echo
echo "== Tap Fast false cuts on the single-sentence human fixture =="
run_row single-fast-150-off single fast 150 off
run_row single-fast-90-off  single fast 90  off
run_row single-fast-150-p90 single fast 150 90

echo
echo "== Long-form window and the adversarial 544 ms intra-utterance pause =="
run_row multi-long-750-off multi long-form 750 off
run_row multi-long-750-p90 multi long-form 750 90
run_row multi-long-500-off multi long-form 500 off
run_row multi-long-300-off multi long-form 300 off

echo
echo "Sweep written to $CSV"
column -s, -t "$CSV"
