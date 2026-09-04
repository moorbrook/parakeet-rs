# Agent contract

Rules for any coding agent working in this repository. The README owns
install, layout, and the verification commands; this file owns the rules
that are not derivable from the code.

## Tracker

- Kata is the tracker. Project `moorbrook/parakeet-rs`; pass
  `--project moorbrook/parakeet-rs` on every command, the cwd does not bind.
- Claim before editing. Close only with a commit, the test command and its
  result, and the observed behavior. Never `kata delete` or `kata purge`.
- One assignment, one tuple: Kata ref, agent name, branch, worktree. Record
  it as the first comment on the issue.

## Commits and review

- Small green slices on `main`, pushed as soon as they pass. Do not hold a
  batch of merges locally; reviews arrive per push and are cheapest while
  the change is still in your head.
- Roborev reviews every pushed commit. About three minutes after a push run
  `roborev show <sha>`; fix every High and Medium before the next commit
  builds on the same code, in at most three follow-up commits naming the
  same Kata ref. A wrong finding is closed with the evidence, not ignored.
- Commit messages name the Kata ref, for example `(kata 0bb3)`.
- No new top-level files and no rewrites outside the issue's named files
  without a Kata comment first.

## Tests

- Never modify an existing test to make code pass. If the specification is
  wrong, say so in the commit message, change the test first, show it fails
  on the old code, then fix the code.
- Definition of done for a bug fix: the close message includes the command
  that reintroduces the bug (revert of the fix hunk or `git stash` of it)
  and the failing test output. A test that cannot fail is not evidence.
- When a test file is replaced, list every deleted test in the commit
  message as `obsolete` or `reinstated`.
- Verification commands are in the README. Anything that can change what the
  recogniser says must pass `asr_diff` and, for encoder or feature changes,
  the gold run.

## Benchmarks and hardware

- The Neural Engine, the microphone, and the BlackHole loopback are one
  shared resource. One benchmark at a time on this machine; check `top` for
  other load first and record it with the numbers.
- Rebuild `bench_asr` and the worker before measuring. A stale binary reports
  retired stages as alive.
- The worker build and the worker tests cannot overlap in one checkout;
  `scripts/reconstitute-fluidaudio.sh` rewrites `.fluidaudio-local`.
- Measured numbers in docs carry the fixture, the commit, and the load
  conditions. Do not round a single run into a claim.

## Repository hygiene

- This repository is public. No real names, email addresses, or home
  directory paths in tracked files, commit messages, or scripts. Keychain
  identity names are fine; the "Apple Development" identity is not.
- FluidAudio is a pinned revision plus checked-in patches reconstituted at
  build time. Do not vendor it and do not bump it without an ADR.
- Single-user app: no backwards-compatibility shims, serde aliases, or
  migration paths. Rename freely.
- Python runs through `uv run` with PEP 723 inline metadata.
- Decisions go in `docs/ADR.md`, numbered by merge order. Docs state
  conclusions; they do not narrate what the code used to do.

## macOS specifics

- Local builds sign with the `Parakeet Local Dev` keychain identity so the
  privacy database recognises rebuilds as the same app. Ad hoc signing
  invalidates every granted permission.
- Privacy tests must launch the app through `open -a Parakeet`. A shell
  launch attributes permission checks to the terminal.
- Recovery for a stale grant: `tccutil reset ListenEvent com.parakeet.rs`.
