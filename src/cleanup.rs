//! Optional LLM post-processing pass.
//!
//! Sits between `Asr::recognize()` and `paste::deliver()` in the dictation
//! pipeline. Takes the raw ASR transcript and (when enabled) sends it to a
//! cloud LLM to:
//!
//! - strip filler words (`um`, `uh`, `you know`, `like`),
//! - fix punctuation and capitalisation,
//! - honour inline editing commands (`new paragraph`, `scratch that`).
//!
//! Cloud path uses Anthropic's Messages API directly (no SDK). The HTTP
//! call is async — the caller is expected to drive it via the shared tokio
//! runtime handle stashed on `App`.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::settings::{CleanupMode, Settings};

const SYSTEM_PROMPT: &str = "You clean up raw speech-to-text transcriptions for direct insertion into the user's document. Output only the cleaned text. No preamble, no commentary, no quotes around the output.\n\
\n\
Rules:\n\
1. Remove filler words: um, uh, er, ah, like, you know, sort of, kind of, I mean (when used as filler).\n\
2. Fix punctuation and capitalisation. Add commas, periods, question marks.\n\
3. Honour inline editing commands: 'new paragraph' or 'new line' becomes a literal newline; 'scratch that', 'delete that', or 'strike that' removes the immediately preceding sentence; 'period' / 'question mark' / 'comma' become the literal punctuation.\n\
4. Preserve the speaker's meaning, tone, and vocabulary. Do NOT paraphrase, summarise, expand, or 'improve' the content.\n\
5. Do NOT add information the speaker did not say.\n\
6. If the input is empty, single-word, or unintelligible, return it unchanged.\n\
7. Preserve technical terms, names, and code-like fragments exactly as transcribed.";

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
/// Hard upper bound to keep the bill bounded. A 30-second dictation at
/// normal speaking speed is ~80 words ≈ ~120 tokens; 2048 leaves plenty of
/// slack while still capping a runaway response.
const MAX_TOKENS: u32 = 2048;

pub async fn polish(text: &str, settings: &Settings) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(text.to_string());
    }
    match settings.cleanup_mode {
        CleanupMode::Off => Ok(text.to_string()),
        CleanupMode::Anthropic => {
            polish_anthropic(text, &settings.anthropic_api_key, &settings.cleanup_model).await
        }
    }
}

async fn polish_anthropic(text: &str, api_key: &str, model: &str) -> Result<String> {
    if api_key.is_empty() {
        return Err(anyhow!(
            "cleanup mode is Anthropic but no API key is set in Settings"
        ));
    }
    let client = reqwest::Client::builder()
        .user_agent("parakeet-rs/0.1")
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("build reqwest client")?;

    let body = serde_json::json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{ "role": "user", "content": text }],
    });

    let resp = client
        .post(API_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("Anthropic API request failed")?;

    let status = resp.status();
    if !status.is_success() {
        // Pull the error body so the menu-bar status surfaces something
        // useful (e.g. invalid key, model not found, rate limit).
        let detail = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Anthropic API {status}: {}",
            truncate(&detail, 240)
        ));
    }

    let parsed: ApiResponse = resp.json().await.context("parse Anthropic response")?;
    let cleaned: String = parsed
        .content
        .into_iter()
        .filter_map(|c| if c.kind == "text" { Some(c.text) } else { None })
        .collect::<Vec<_>>()
        .join("");
    Ok(cleaned.trim().to_string())
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
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

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn polish_off_is_identity() {
        let s = Settings {
            cleanup_mode: CleanupMode::Off,
            ..Settings::default()
        };
        let out = rt().block_on(polish("Hello, world.", &s)).unwrap();
        assert_eq!(out, "Hello, world.");
    }

    #[test]
    fn polish_empty_is_identity_even_when_mode_is_anthropic() {
        // Empty input short-circuits before we hit the network or check the
        // API key — important because the recogniser can hand us "" for a
        // failed decode and we don't want to surface a "missing key" error
        // for what is effectively a no-op.
        let s = Settings {
            cleanup_mode: CleanupMode::Anthropic,
            anthropic_api_key: String::new(),
            ..Settings::default()
        };
        assert_eq!(rt().block_on(polish("", &s)).unwrap(), "");
        assert_eq!(rt().block_on(polish("   \n  ", &s)).unwrap(), "   \n  ");
    }

    #[test]
    fn polish_anthropic_without_key_errors() {
        let s = Settings {
            cleanup_mode: CleanupMode::Anthropic,
            anthropic_api_key: String::new(),
            ..Settings::default()
        };
        let err = rt()
            .block_on(polish("real transcript", &s))
            .expect_err("missing key should error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("API key") || msg.contains("api key"),
            "error should mention the key: {msg}"
        );
    }

    #[test]
    fn truncate_is_charwise_safe() {
        assert_eq!(truncate("abc", 10), "abc");
        assert_eq!(truncate("abcdefghij", 5), "abcde…");
    }
}
