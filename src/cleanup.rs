//! Optional LLM post-processing pass.
//!
//! Sits between `Asr::recognize()` and `paste::deliver()` in the dictation
//! pipeline. Takes the raw ASR transcript and (when enabled) cleans it
//! through Claude to:
//!
//! - strip filler words (`um`, `uh`, `you know`, `like`),
//! - fix punctuation and capitalisation,
//! - honour inline editing commands (`new paragraph`, `scratch that`).
//!
//! **Transport: `claude -p`**, not the Anthropic Messages API. Per project
//! directive (see [[no-anthropic-api]] in agent memory): no direct API
//! calls, no `x-api-key`, no separate API-key provisioning. The user's
//! existing Claude Code OAuth login is what bills the request, and a key
//! that we don't hold is a key that can't leak.

use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};

use crate::settings::{CleanupMode, Settings};

const SYSTEM_PROMPT: &str = "You clean up raw speech-to-text transcriptions for direct insertion into the user's document. Output only the cleaned text. No preamble, no commentary, no quotes around the output, no Markdown formatting.\n\
\n\
Rules:\n\
1. Remove filler words: um, uh, er, ah, like, you know, sort of, kind of, I mean (when used as filler).\n\
2. Fix punctuation and capitalisation. Add commas, periods, question marks.\n\
3. Honour inline editing commands: 'new paragraph' or 'new line' becomes a literal newline; 'scratch that', 'delete that', or 'strike that' removes the immediately preceding sentence; 'period' / 'question mark' / 'comma' become the literal punctuation.\n\
4. Preserve the speaker's meaning, tone, and vocabulary. Do NOT paraphrase, summarise, expand, or 'improve' the content.\n\
5. Do NOT add information the speaker did not say.\n\
6. If the input is empty, single-word, or unintelligible, return it unchanged.\n\
7. Preserve technical terms, names, and code-like fragments exactly as transcribed.\n\
8. Do not call any tools. Output text only.";

/// `claude -p` startup + model load is ~1-3 s on a warm cache. 30 s is
/// generous enough to cover a cold cache + a long transcript without
/// letting a hung subprocess wedge the dictation pipeline forever.
const SUBPROCESS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub fn polish(text: &str, settings: &Settings) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(text.to_string());
    }
    match settings.cleanup_mode {
        CleanupMode::Off => Ok(text.to_string()),
        CleanupMode::Claude => polish_via_claude_cli(text, &settings.cleanup_model),
    }
}

fn polish_via_claude_cli(text: &str, model: &str) -> Result<String> {
    let mut child = Command::new("claude")
        .args([
            "-p",
            "--model",
            model,
            "--no-session-persistence",
            "--append-system-prompt",
            SYSTEM_PROMPT,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                anyhow!(
                    "`claude` CLI not found on PATH. Install Claude Code \
                     (https://docs.claude.com/en/docs/claude-code) or turn \
                     Cleanup off in Settings."
                )
            } else {
                anyhow!("spawning `claude -p`: {e}")
            }
        })?;

    // Pipe the raw transcript via stdin so we don't have to worry about
    // shell quoting, argv length limits, or accidentally re-interpreting
    // the user's spoken `--flag` as a CLI argument.
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("claude subprocess has no stdin pipe"))?;
        stdin
            .write_all(text.as_bytes())
            .context("write transcript to claude stdin")?;
        // stdin is dropped here, closing the pipe — claude knows we're done.
    }

    let output = wait_with_timeout(child, SUBPROCESS_TIMEOUT)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "claude -p exited with {}: {}",
            output.status,
            truncate(stderr.trim(), 240)
        ));
    }

    let cleaned = String::from_utf8(output.stdout)
        .context("claude -p stdout was not valid UTF-8")?
        .trim()
        .to_string();
    if cleaned.is_empty() {
        // Treat empty output as failure — pasting nothing would be worse
        // than pasting the raw transcript (which the caller falls back to).
        return Err(anyhow!("claude -p returned an empty response"));
    }
    Ok(cleaned)
}

/// Wait for the subprocess with a wall-clock timeout. `std::process::Child`
/// has no `wait_timeout` of its own, so poll `try_wait` on a short
/// interval and kill the child if the budget is exceeded.
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: std::time::Duration,
) -> Result<std::process::Output> {
    let start = std::time::Instant::now();
    loop {
        match child.try_wait().context("polling claude subprocess")? {
            Some(_status) => {
                // Use `wait_with_output` for the captured stdout/stderr —
                // it internally re-waits, which is fine since the process
                // has already exited.
                return child
                    .wait_with_output()
                    .context("collecting claude subprocess output");
            }
            None if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!(
                    "claude -p timed out after {:.0}s",
                    timeout.as_secs_f32()
                ));
            }
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{CleanupMode, Settings};

    #[test]
    fn polish_off_is_identity() {
        let s = Settings {
            cleanup_mode: CleanupMode::Off,
            ..Settings::default()
        };
        assert_eq!(polish("Hello, world.", &s).unwrap(), "Hello, world.");
    }

    #[test]
    fn polish_empty_is_identity_even_when_mode_is_claude() {
        // Empty input short-circuits before we'd spawn the subprocess,
        // which keeps cleanup-mode-on safe against the recogniser
        // returning "" for a failed decode.
        let s = Settings {
            cleanup_mode: CleanupMode::Claude,
            ..Settings::default()
        };
        assert_eq!(polish("", &s).unwrap(), "");
        assert_eq!(polish("   \n  ", &s).unwrap(), "   \n  ");
    }

    #[test]
    fn truncate_is_charwise_safe() {
        assert_eq!(truncate("abc", 10), "abc");
        assert_eq!(truncate("abcdefghij", 5), "abcde…");
    }

    // Note: we don't unit-test the actual subprocess invocation here.
    // The `claude` CLI talks to the network and bills the user's
    // account on every call — wrong shape for a unit test. The
    // integration test is "Set Cleanup = Claude in Settings, dictate
    // something filler-heavy, watch the paste come out clean."
}
