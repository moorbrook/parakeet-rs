//! macOS Keychain Services wrapper for the Anthropic API key.
//!
//! Never persist secrets to `settings.json`. The settings file is plain
//! text in `~/Library/Application Support/com.parakeet.rs/` and gets
//! casually backed up, synced, screen-shared, and grep'd — entirely the
//! wrong threat model for a long-lived API credential.
//!
//! This module stores the key as a `kSecClassGenericPassword` item under
//! a single service / account pair so the same key persists across
//! launches of the app and across Settings dialog opens. Empty string
//! ("set to nothing") deletes the keychain item — the absence of an
//! item is what tells `cleanup` mode that no key is configured.

use anyhow::{Context, Result};
use security_framework::passwords::{
    delete_generic_password, get_generic_password, set_generic_password,
};

/// Keychain service name. Stable for the life of the app — changing it
/// would orphan everyone's existing entries.
const SERVICE: &str = "com.parakeet.rs";
/// Logical account name within the service. Lets us add a second slot
/// later (OpenAI key, Gemini key, etc.) without colliding.
const ACCOUNT_ANTHROPIC: &str = "anthropic_api_key";

/// Read the Anthropic API key from the user's login keychain. Returns an
/// empty string (not an error) if no key has been configured yet — that's
/// the equivalent of "no key set" and the cleanup path treats it as such.
pub fn read_anthropic_key() -> Result<String> {
    match get_generic_password(SERVICE, ACCOUNT_ANTHROPIC) {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).to_string()),
        Err(e) => {
            // `security-framework` doesn't break out errSecItemNotFound
            // as its own variant; the error code is -25300. Anything
            // else (denied, locked keychain) is a real error worth
            // surfacing.
            if e.code() == -25300 {
                Ok(String::new())
            } else {
                Err(anyhow::Error::new(e)).context("read Anthropic key from Keychain")
            }
        }
    }
}

/// Persist the Anthropic API key. Empty string deletes the keychain
/// item entirely — "save with the field cleared" is how the user
/// removes a previously-stored key without an extra "Clear" button.
pub fn write_anthropic_key(key: &str) -> Result<()> {
    if key.is_empty() {
        match delete_generic_password(SERVICE, ACCOUNT_ANTHROPIC) {
            Ok(()) => Ok(()),
            // -25300 = errSecItemNotFound. Deleting a non-existent item
            // is the same desired outcome ("no key stored"); not an error.
            Err(e) if e.code() == -25300 => Ok(()),
            Err(e) => Err(anyhow::Error::new(e)).context("delete Anthropic key from Keychain"),
        }
    } else {
        set_generic_password(SERVICE, ACCOUNT_ANTHROPIC, key.as_bytes())
            .map_err(anyhow::Error::new)
            .context("store Anthropic key in Keychain")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_and_account_constants_are_stable() {
        // Pin the strings: changing either would orphan every existing
        // user's keychain entry. If a future PR genuinely needs to
        // change them, it should add a migration step rather than
        // silently flipping the constant.
        assert_eq!(SERVICE, "com.parakeet.rs");
        assert_eq!(ACCOUNT_ANTHROPIC, "anthropic_api_key");
    }

    // Note: we don't unit-test the actual Keychain round-trip here.
    // The system keychain prompts for user consent on writes from
    // unsigned binaries, which would block CI. The integration test is
    // "launch the app, set a key in Settings, restart, observe the
    // key still works".
}
