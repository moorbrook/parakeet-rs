# ASR discrepancy ledger

This ledger records known disagreements with the reviewed gold references. An
entry is evidence, not a correction to the reference; reference text changes
require listening to the source recording and updating its provenance review.

## FluidAudio Core ML baseline — 2026-08-11

| fixture | reference fragment | hypothesis fragment | word alignment | disposition |
|---|---|---|---|---|
| SLURP alarm | `seven thirty AM` | `seven hundred and thirty AM` | +`hundred`, +`and` | Open model error; counted. |
| SLURP stock | `IBM` | `IPM` | 1 substitution | Open proper-noun error; counted. |
| SLURP music | `Olly … from music` | `Ollie … for music` | 2 substitutions | Open name/preposition errors; counted. |
| LibriSpeech single | final period | final comma | 0 lexical edits | Formatting-only; raw strings retain it. |
| LibriSpeech multi | two optional comma locations | commas omitted | 0 lexical edits | Formatting-only; raw strings retain it. |

The aggregate is five word edits: two insertions, zero deletions, and three
substitutions. These errors are inside the absolute product budget, but the
zero-point regression cap prevents accepting an additional error silently.
