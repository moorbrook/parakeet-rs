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

/// Optional LLM post-processing pass between ASR output and paste. Used to
/// strip filler words, fix punctuation, and honour inline editing commands
/// (e.g. "new paragraph", "scratch that").
///
/// v1 ships a single backend: Qwen 3.5 2B Q4_K_M via in-process
/// `llama-cpp-2` (Metal on Apple Silicon). See [ADR-0018](../docs/ADR.md).
/// No `cleanup_model` setting — the model is fixed at the backend's
/// expected path and changing it requires a code change, not a config
/// change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CleanupMode {
    /// No post-processing; paste the raw ASR transcript.
    #[default]
    Off,
    /// In-process Qwen 3.5 2B Q4_K_M via `llama-cpp-2` Metal backend.
    On,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    /// Language hint for the recognizer, e.g. "eng_Latn". Empty = autodetect.
    pub language: String,
    #[serde(default)]
    pub cleanup_mode: CleanupMode,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "CmdOrCtrl+Shift+Space".to_string(),
            trigger_mode: TriggerMode::default(),
            language: String::new(),
            cleanup_mode: CleanupMode::default(),
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

    pub fn save(&self, s: &Settings) -> Result<()> {
        // Atomic write: serialise + fsync to a unique temp file in the
        // same directory, then rename over the target, then fsync the
        // directory so the rename itself is durable across power loss.
        // The in-memory cache is updated AFTER the rename succeeds —
        // otherwise an I/O failure would leave the running process and
        // the on-disk file disagreeing, and the next boot's
        // `unwrap_or_default()` parse would silently revert to defaults.
        //
        // Tmp path is unique per save (pid + nonce) so two concurrent
        // saves can't truncate each other's tmp file mid-write or race
        // on `rename`.
        //
        // Pretty-printed so it's grep-friendly when debugging. There is
        // no secret to special-case — the in-process llama.cpp path has
        // no API key.
        use std::io::Write as _;
        use std::sync::atomic::{AtomicU64, Ordering};
        static SAVE_NONCE: AtomicU64 = AtomicU64::new(0);

        let raw = serde_json::to_string_pretty(s)?;
        let nonce = SAVE_NONCE.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let parent = self
            .settings_path
            .parent()
            .context("settings_path must have a parent dir")?;
        let stem = self
            .settings_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("settings.json");
        let tmp_path = parent.join(format!(".{stem}.tmp.{pid}.{nonce}"));
        {
            let mut tmp = std::fs::File::create(&tmp_path)
                .with_context(|| format!("creating temp file {}", tmp_path.display()))?;
            tmp.write_all(raw.as_bytes())
                .context("writing settings tmp file")?;
            // fsync the file so the rename below isn't observed before
            // the bytes hit the platter.
            tmp.sync_all().context("fsync settings tmp file")?;
        }
        std::fs::rename(&tmp_path, &self.settings_path)
            .with_context(|| format!("renaming {} into place", tmp_path.display()))?;
        // fsync the directory so the rename is durable. macOS HFS+/APFS
        // honour this; on a crash between rename and the next dir-sync,
        // the file would otherwise revert to its pre-rename inode.
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        *self.cache.lock() = s.clone();
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

    /// Cleanup-pass GGUF weights, downloaded on first cleanup-enabled
    /// launch into the same per-bundle data directory as the ASR model.
    /// v1 ships exactly one supported model (Qwen 3.5 2B Q4_K_M) — see
    /// [ADR-0018](../../docs/ADR.md). The directory + filename are
    /// fixed; changing them requires a model-fetch update, not a
    /// settings change.
    pub fn cleanup_model_path(&self) -> PathBuf {
        self.data_dir
            .join("llm")
            .join("qwen3.5-2b-q4_k_m")
            .join("Qwen3.5-2B-Q4_K_M.gguf")
    }

    /// True iff the cleanup-pass GGUF is on disk. The cleanup loader
    /// gates on this before attempting to call llama.cpp.
    pub fn cleanup_model_present(&self) -> bool {
        self.cleanup_model_path().exists()
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
        assert_eq!(back.language, s.language);
        assert_eq!(back.trigger_mode, s.trigger_mode);
    }

    #[test]
    fn legacy_inject_mode_field_is_ignored_for_forward_compat() {
        // Older settings.json files have an `inject_mode` key (the
        // pre-AX clipboard/⌘V delivery had a "paste" / "type" /
        // "clipboard" debug knob that was never wired to the UI).
        // ADR-0019 removes it, but existing files MUST still parse
        // cleanly. serde's default behaviour ignores unknown keys —
        // pin that so nobody accidentally adds `#[serde(deny_unknown_fields)]`.
        let raw = r#"{
            "hotkey": "CmdOrCtrl+Shift+Space",
            "trigger_mode": "tap",
            "inject_mode": "paste",
            "language": ""
        }"#;
        let s: Settings = serde_json::from_str(raw).expect("legacy field should parse-and-ignore");
        assert_eq!(s.hotkey, "CmdOrCtrl+Shift+Space");
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
    fn cleanup_defaults_to_off() {
        // The cleanup pass is opt-in: a fresh install doesn't load the
        // ~1.2 GB Qwen GGUF until the user enables cleanup explicitly.
        // Pin the default so an accidental flip doesn't silently start
        // a download + warmup the user didn't ask for.
        let s = Settings::default();
        assert_eq!(s.cleanup_mode, CleanupMode::Off);
    }

    #[test]
    fn cleanup_mode_on_round_trips_lowercase() {
        let s = Settings {
            cleanup_mode: CleanupMode::On,
            ..Settings::default()
        };
        let raw = serde_json::to_string(&s).unwrap();
        // Lowercase enum form, same as TriggerMode.
        assert!(
            raw.contains("\"cleanup_mode\":\"on\""),
            "expected lowercase cleanup_mode, got: {raw}"
        );
        let back: Settings = serde_json::from_str(&raw).unwrap();
        assert_eq!(back.cleanup_mode, CleanupMode::On);
    }

    #[test]
    fn settings_json_never_contains_api_key_or_model_name() {
        // The Anthropic-API era and the `claude -p` era both wrote
        // identifier fields (`anthropic_api_key`, `cleanup_model`) into
        // settings.json. The in-process llama.cpp path has neither —
        // no key, exactly one supported model. Pin that nothing in the
        // wire shape lets a future change reintroduce them by accident.
        let s = Settings {
            cleanup_mode: CleanupMode::On,
            ..Settings::default()
        };
        let raw = serde_json::to_string(&s).unwrap();
        assert!(
            !raw.contains("api_key"),
            "settings.json must not contain any *api_key* field: {raw}"
        );
        assert!(
            !raw.contains("cleanup_model"),
            "settings.json must not contain cleanup_model: {raw}"
        );
    }

    #[test]
    fn missing_cleanup_fields_default_to_off() {
        // A settings.json from before the cleanup feature existed must
        // still parse — and importantly, it must come out with cleanup
        // OFF so an upgrade doesn't silently start downloading 1.2 GB
        // of model weights without consent.
        let raw = r#"{
            "hotkey": "CmdOrCtrl+Shift+Space",
            "trigger_mode": "tap",
            "inject_mode": "paste",
            "language": ""
        }"#;
        let s: Settings = serde_json::from_str(raw).expect("pre-cleanup file should parse");
        assert_eq!(s.cleanup_mode, CleanupMode::Off);
    }

    #[test]
    fn cleanup_model_path_lives_under_data_dir() {
        // The cleanup GGUF cache layout is a shipped invariant — a
        // launch that changes this strands existing users' downloads.
        let store = synthetic_store(tempfile::tempdir().unwrap().keep());
        let cm = store.cleanup_model_path();
        assert!(cm.starts_with(store.data_dir()));
        assert_eq!(
            cm,
            store
                .data_dir()
                .join("llm")
                .join("qwen3.5-2b-q4_k_m")
                .join("Qwen3.5-2B-Q4_K_M.gguf")
        );
    }

    #[test]
    fn model_paths_share_a_single_root() {
        // The downloader (model_fetch.rs) writes the four ASR files into
        // `model_dir()` and the VAD into `data_dir/models/`. If anyone
        // renames the model subdirectory, every existing user has to
        // re-download 640 MB. Pin the layout.
        let store = synthetic_store(tempfile::tempdir().unwrap().keep());
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
        let store = synthetic_store(tempfile::tempdir().unwrap().keep());
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
