//! Persisted user preferences + model paths.
//!
//! Drops the Tauri `AppHandle` previously used to find the per-bundle data
//! directory. On macOS, that path is now hand-derived from `dirs::data_dir()`
//! ( `~/Library/Application Support` ) plus our `com.parakeet.rs` bundle
//! identifier — so a `cargo run`-built binary points at the same directory
//! that the previous Tauri-bundled build used. No re-download needed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Bundle-id-style namespace for our on-disk state. Matches what the previous
/// Tauri build wrote (`tauri.conf.json` `identifier`), so the model files
/// downloaded under that name still resolve.
const BUNDLE_NAMESPACE: &str = "com.parakeet.rs";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TriggerMode {
    /// Tap once to start; Silero VAD detects end-of-speech and auto-pastes.
    /// A second tap during dictation cancels.
    #[default]
    Tap,
    /// Press and hold to dictate; release to immediately paste. No VAD.
    Hold,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    pub inject_mode: String,
    /// Language hint for the recognizer, e.g. "eng_Latn". Empty = autodetect.
    pub language: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "CmdOrCtrl+Shift+Space".to_string(),
            trigger_mode: TriggerMode::default(),
            inject_mode: "paste".to_string(),
            language: String::new(),
        }
    }
}

#[derive(Clone)]
pub struct SettingsStore {
    #[allow(dead_code)] // Reserved for the future settings-save UI.
    settings_path: PathBuf,
    data_dir: PathBuf,
    cache: Arc<Mutex<Settings>>,
}

impl SettingsStore {
    pub fn new() -> Result<Self> {
        let data_root = dirs::data_dir().context("locating Application Support dir")?;
        let data_dir = data_root.join(BUNDLE_NAMESPACE);
        // On macOS the conventional split is config under `~/Library/Preferences`
        // and data under `~/Library/Application Support`. We collapse both into
        // a single per-bundle directory so a fresh checkout finds its settings
        // next to its models.
        std::fs::create_dir_all(&data_dir).context("create data dir")?;
        let settings_path = data_dir.join("settings.json");
        let cache = if settings_path.exists() {
            let raw = std::fs::read_to_string(&settings_path)?;
            serde_json::from_str::<Settings>(&raw).unwrap_or_default()
        } else {
            Settings::default()
        };
        Ok(Self {
            settings_path,
            data_dir,
            cache: Arc::new(Mutex::new(cache)),
        })
    }

    pub fn load(&self) -> Settings {
        self.cache.lock().clone()
    }

    #[allow(dead_code)] // Reserved for the future settings-save UI.
    pub fn save(&self, s: &Settings) -> Result<()> {
        *self.cache.lock() = s.clone();
        let raw = serde_json::to_string_pretty(s)?;
        std::fs::write(&self.settings_path, raw)?;
        Ok(())
    }

    #[allow(dead_code)] // Useful in tests / future debugging code.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn model_dir(&self) -> PathBuf {
        self.data_dir
            .join("models")
            .join("parakeet-tdt-0.6b-v3-int8")
    }

    pub fn encoder_path(&self) -> PathBuf {
        self.model_dir().join("encoder.int8.onnx")
    }

    pub fn decoder_path(&self) -> PathBuf {
        self.model_dir().join("decoder.int8.onnx")
    }

    pub fn joiner_path(&self) -> PathBuf {
        self.model_dir().join("joiner.int8.onnx")
    }

    pub fn tokens_path(&self) -> PathBuf {
        self.model_dir().join("tokens.txt")
    }

    /// Silero VAD ONNX model — drives the press-once / auto-stop flow.
    pub fn vad_path(&self) -> PathBuf {
        self.data_dir.join("models").join("silero_vad.onnx")
    }

    pub fn model_present(&self) -> bool {
        self.encoder_path().exists()
            && self.decoder_path().exists()
            && self.joiner_path().exists()
            && self.tokens_path().exists()
            && self.vad_path().exists()
    }
}

#[cfg(test)]
mod tests {
    //! Settings live in `~/Library/Application Support/com.parakeet.rs/settings.json`
    //! and persist across upgrades, so the serialisation shape is part of
    //! our compatibility contract. These tests guard against accidental
    //! breakage: renamed fields, removed defaults, or model-path layout
    //! drift that would silently invalidate an existing user's install.
    use super::*;
    use std::path::PathBuf;

    fn synthetic_store(data_dir: PathBuf) -> SettingsStore {
        // Build a store without touching `dirs::data_dir()` so the test
        // doesn't depend on a real `$HOME`. The path layout helpers
        // (`encoder_path`, `vad_path`, …) read `data_dir` only.
        SettingsStore {
            settings_path: data_dir.join("settings.json"),
            data_dir,
            cache: Arc::new(Mutex::new(Settings::default())),
        }
    }

    #[test]
    fn default_settings_roundtrip_through_json() {
        let s = Settings::default();
        let raw = serde_json::to_string(&s).expect("serialise");
        let back: Settings = serde_json::from_str(&raw).expect("parse default round-trip");
        assert_eq!(back.hotkey, s.hotkey);
        assert_eq!(back.inject_mode, s.inject_mode);
        assert_eq!(back.language, s.language);
        assert_eq!(back.trigger_mode, s.trigger_mode);
    }

    #[test]
    fn trigger_mode_serialises_lowercase() {
        // `#[serde(rename_all = "lowercase")]` — drift would silently break
        // anyone whose settings.json was written by a previous build.
        let s = Settings {
            trigger_mode: TriggerMode::Hold,
            ..Settings::default()
        };
        let raw = serde_json::to_string(&s).unwrap();
        assert!(
            raw.contains("\"trigger_mode\":\"hold\""),
            "expected lowercase enum, got: {raw}"
        );
    }

    #[test]
    fn missing_trigger_mode_falls_back_to_default() {
        // settings.json written by a build that predates the Hold UI must
        // still parse — the field has `#[serde(default)]`. If someone
        // removes that attribute this test fails loudly.
        let raw = r#"{
            "hotkey": "CmdOrCtrl+Shift+Space",
            "inject_mode": "paste",
            "language": ""
        }"#;
        let s: Settings = serde_json::from_str(raw).expect("legacy file should parse");
        assert_eq!(s.trigger_mode, TriggerMode::default());
        assert_eq!(s.trigger_mode, TriggerMode::Tap);
    }

    #[test]
    fn unknown_keys_are_ignored_for_forward_compat() {
        // If a newer build adds a field and the user downgrades, the older
        // build still needs to parse the file (it just drops the unknown
        // key). serde defaults to permissive parsing; this test pins that
        // behaviour so nobody accidentally adds `#[serde(deny_unknown_fields)]`.
        let raw = r#"{
            "hotkey": "F5",
            "trigger_mode": "hold",
            "inject_mode": "paste",
            "language": "",
            "future_feature_flag": true,
            "another_one": "value"
        }"#;
        let s: Settings = serde_json::from_str(raw).expect("should ignore unknown keys");
        assert_eq!(s.hotkey, "F5");
        assert_eq!(s.trigger_mode, TriggerMode::Hold);
    }

    #[test]
    fn model_paths_share_a_single_root() {
        // The downloader (model_fetch.rs) writes the four ASR files into
        // `model_dir()` and the VAD into `data_dir/models/`. If anyone
        // renames the model subdirectory, every existing user has to
        // re-download 640 MB. Pin the layout.
        let store = synthetic_store(PathBuf::from("/tmp/parakeet-test"));
        let model_dir = store.model_dir();
        assert!(model_dir.starts_with(store.data_dir()));
        assert_eq!(
            model_dir,
            store
                .data_dir()
                .join("models")
                .join("parakeet-tdt-0.6b-v3-int8"),
        );
        assert_eq!(store.encoder_path(), model_dir.join("encoder.int8.onnx"));
        assert_eq!(store.decoder_path(), model_dir.join("decoder.int8.onnx"));
        assert_eq!(store.joiner_path(), model_dir.join("joiner.int8.onnx"));
        assert_eq!(store.tokens_path(), model_dir.join("tokens.txt"));
        assert_eq!(
            store.vad_path(),
            store.data_dir().join("models").join("silero_vad.onnx"),
        );
    }

    #[test]
    fn bundle_namespace_is_stable() {
        // The on-disk path is derived from this constant. Changing it
        // strands every existing user's downloaded model + settings.
        assert_eq!(BUNDLE_NAMESPACE, "com.parakeet.rs");
    }

    #[test]
    fn download_set_matches_presence_check() {
        // The downloader fetches `model_fetch::ASR_FILES` into
        // `model_dir()`. Startup then refuses to launch unless all four
        // `*_path()` accessors point at files that exist. If those two
        // lists ever drift, first-run reports "Model ready." and the
        // recogniser fails to load on the next launch.
        let store = synthetic_store(PathBuf::from("/tmp/parakeet-test"));
        let model_dir = store.model_dir();

        let mut gated: Vec<PathBuf> = vec![
            store.encoder_path(),
            store.decoder_path(),
            store.joiner_path(),
            store.tokens_path(),
        ];
        gated.sort();

        let mut fetched: Vec<PathBuf> = crate::model_fetch::ASR_FILES
            .iter()
            .map(|name| model_dir.join(name))
            .collect();
        fetched.sort();

        assert_eq!(
            gated, fetched,
            "model_fetch::ASR_FILES and SettingsStore presence accessors have drifted"
        );
    }
}
