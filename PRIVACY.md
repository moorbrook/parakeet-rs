# Privacy

Parakeet asks for three of the most powerful permissions macOS grants:
it listens to your microphone, watches every keystroke you type, and
synthesizes keystrokes into whatever app is focused. That combination
deserves a straight answer about what it does with them, not a bullet
in a README.

Short version: **your audio and transcripts are never transmitted by
Parakeet, and no audio or transcript is written to disk.** Settings and
downloaded models are on disk, and there is one failure path that puts
a transcript on your clipboard — both covered below. The rest of this
document is the detail behind that claim, including the parts that are
less flattering.

## Network access

Parakeet makes network requests in exactly two situations, both of them
model downloads over HTTPS:

| When | What | Where from |
|---|---|---|
| First launch | Parakeet TDT 0.6B v3 int8 + tokens (~640 MB) | `huggingface.co` |
| First launch | Silero VAD (~2 MB) | `github.com` |
| First time you enable Polish | Qwen 3.5 4B Q6_K GGUF (~3.5 GB) | `huggingface.co` |

Once those files are on disk there are no further network calls. There
is no telemetry, no crash reporting, no analytics, no update check, and
no license or activation server. There is no account to create and no
API key to enter, because there is no service to authenticate to.

You can verify this: `src/model_fetch.rs` contains every URL the app
knows about, and it is the only module that links the HTTP client.

## Your audio

Recorded into memory while the hotkey session is active, passed to the
local recognizer, and dropped when the session ends. It is never
written to a file, never buffered between sessions, and never
transmitted.

Parakeet records only while a session is live — from the hotkey press
until the VAD detects end-of-speech (Tap mode) or you release the key
(Hold mode). The microphone is not held open between sessions, and the
macOS microphone indicator in the menu bar reflects this accurately.

## Your transcripts

Held in memory, delivered to the focused app, and dropped. Parakeet
keeps **no transcript history** — there is no log file, no database, no
"recent dictations" list, and therefore nothing to browse, export, or
leak. If you dictate a password into a text field, Parakeet retains no
record of it.

Two consequences worth being explicit about:

- If delivery to the focused app fails, Parakeet's fallback is to put
  the transcript on your clipboard so your words aren't lost (see
  below). That is the one case where a transcript outlives the session,
  and it lives exactly as long as anything else you copy.
- There is no way to recover a transcript after the fact. That is the
  intended tradeoff, not an oversight.

Transcript text is written to the log at INFO level as a **32-character
preview** (`src/ax_paste.rs`) for debugging delivery problems. That goes
to stderr; the app does not write a log file, so this is only visible
if you launch the binary from a terminal.

## Permissions, and what each one is actually used for

Parakeet does **not** ask macOS for all three merely because it launched.
Onboarding first explains Input Monitoring, which enables the global hotkey;
the app and its menu remain usable if you defer it. Microphone and
Accessibility are explained and requested only when you try to start
dictation. A **Check Permissions…** menu command always shows current state
and recovery actions. Parakeet refreshes that state after you return from
System Settings and if a previously granted permission is later revoked.

Microphone and Accessibility are required for a complete dictation. Input
Monitoring is required only for the global hotkey; menu-bar dictation works
without it. macOS requires one explained quit/reopen when Input Monitoring is
granted after Parakeet has already created its event tap.

**Microphone** — capture audio while a session is active. Nothing else.

**Input Monitoring** — Parakeet installs a `CGEventTap` to detect its
global hotkey (`src/hotkey.rs`). An event tap at this level *does* see
every keystroke in every app, which is what makes this permission
frightening, and the concern is legitimate.

Two things constrain what Parakeet can do with it:

- The tap is registered as **`CGEventTapOptions::ListenOnly`**. That is
  an OS-enforced mode, not a promise about our code: a listen-only tap
  is structurally incapable of modifying or suppressing events. It
  cannot alter what you type.
- The callback reads two fields per event — the keycode and the
  modifier flags — compares them against your configured shortcut, and
  returns. Nothing is accumulated across events. There is no buffer, no
  counter, and no data structure holding keystrokes to inspect.

One disclosure: the callback contains a `log::trace!` that prints each
event's keycode and modifier flags, for diagnosing "I pressed my
shortcut and nothing happened". It is off unless you explicitly set
`RUST_LOG=parakeet_rs::hotkey=trace`, and even then it goes to stderr,
which the bundled app discards. If you turn it on from a terminal, that
terminal will show keycodes for everything you type until you close it.

**Accessibility** — required to post the synthesized keystroke that
inserts your transcript at the cursor. Parakeet does not read the
contents of other applications through the Accessibility API. An
earlier version did use `AXUIElementSetAttributeValue` to write text
directly; that path was removed (see
[ADR-0019](docs/ADR.md#0019--paste-delivery-synthetic-unicode-keystroke))
because it silently failed in terminals, not for privacy reasons — but
the practical effect is that the current code touches less of the AX
API, not more.

The exact macOS state model and recovery/QA matrix are documented in
[`docs/macos-permissions.md`](docs/macos-permissions.md).

## Clipboard

On the normal path, Parakeet **does not touch your clipboard.** Text is
inserted as a synthetic Unicode keystroke, so whatever you had copied
stays copied.

The exception is delivery failure. If the keystroke can't be
constructed or posted, or the Polish pass dies part-way through
streaming, Parakeet writes the transcript to the clipboard and says so
in the menu bar, because losing what you just said is worse than
overwriting a clipboard entry. Your previous clipboard contents are
**not** restored afterwards: the transcript has to stay there until you
paste it.

**This is the one path where a transcript can leave your Mac.** The
general pasteboard participates in Handoff / Universal Clipboard, so if
you have that enabled and are signed into iCloud, macOS may sync the
rescued transcript to your other Apple devices. Parakeet does not
initiate or control that — it is the OS behaving normally for anything
you copy — but it is a real consequence of the rescue and you should
know about it. Turn off Handoff in System Settings → General →
AirDrop & Handoff if that matters to you.

Note also what the rescue **cannot** catch. `CGEventPost` is
fire-and-forget: macOS gives no delivery receipt, so if the focused app
receives the keystroke and discards it (password fields, apps that
filter synthetic input), Parakeet has no way to know. It reports
success and no clipboard copy is made. The rescue covers failures we
can observe, not every way text can fail to appear.

## Files on disk

Everything lives under
`~/Library/Application Support/com.parakeet.rs/`:

| Path | Contents |
|---|---|
| `settings.json` | Your hotkey, trigger mode, polish mode, hotword score. No secrets — there is no API key to store. |
| `vocabulary.txt` | Custom terms you added, if any. Parakeet creates it with a commented-out template the first time you click "Edit Vocabulary…", and only reads it after that — it never modifies content you write. |
| `hotwords.generated.txt` | Machine translation of the above. Regenerated on every model load; safe to delete. |
| `models/` | Downloaded ASR + VAD weights (~640 MB). |
| `llm/` | Downloaded Polish GGUF (~3.5 GB), only if you enabled Polish. |

No audio, no transcripts. Deleting the whole directory resets Parakeet
to a fresh install; it will re-download the models on next launch.

Note that `vocabulary.txt` is unencrypted plain text, and its contents
are inferable from what Parakeet transcribes well. If the names you
dictate are themselves sensitive, that file is worth knowing about.

## The Polish pass

Optional, off by default. When enabled, it runs Qwen 3.5 4B in-process
via llama.cpp on the Metal GPU — the same process, no subprocess, no
socket, no server. Your transcript is put in a prompt and the cleaned
text comes back. Nothing is sent anywhere, and the model has no
mechanism to persist anything between calls.

## What this document does not promise

- Parakeet is not audited. These claims are checkable by reading the
  source, which is the level of assurance a single-maintainer project
  can honestly offer.
- Builds are ad-hoc signed, not notarized. You are trusting the build
  you made from source, which is the right way around, but it does mean
  there is no Apple-verified chain from this repository to the binary
  you ran.
- macOS itself, your keyboard, and the apps you dictate into all have
  their own privacy behavior that Parakeet has no visibility into or
  control over.

Found something that contradicts the above? That is a bug, and a
serious one. Please open an issue.
