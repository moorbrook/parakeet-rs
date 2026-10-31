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

use crate::windows::HoldWindowConfig;

/// Bundle-id-style namespace for our on-disk state. Matches what the previous
/// Tauri build wrote (`tauri.conf.json` `identifier`), so the model files
/// downloaded under that name still resolve.
const BUNDLE_NAMESPACE: &str = "com.parakeet.rs";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TriggerMode {
    /// Tap once to start; pause-friendly Silero VAD waits through natural
    /// clause/sentence pauses before auto-pasting. A second tap cancels.
    #[default]
    Tap,
    /// 90 ms auto-stop for short commands where minimum final-pause
    /// latency matters more than tolerating a long pause.
    #[serde(rename = "tap_fast")]
    TapFast,
    /// Press and hold to dictate; release to immediately paste. No VAD.
    Hold,
}

impl TriggerMode {
    pub fn endpoint_policy(self) -> crate::endpointing::EndpointPolicy {
        match self {
            Self::Tap => crate::endpointing::EndpointPolicy::LongForm,
            Self::TapFast | Self::Hold => crate::endpointing::EndpointPolicy::Fast,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    /// Language hint for the recognizer, e.g. "eng_Latn". Empty = autodetect.
    pub language: String,
    /// Rewrite each transcript through the OpenCode service that is
    /// already running on this Mac before pasting it. Off by default; when
    /// on and no service answers within the deadline, the raw transcript is
    /// pasted unchanged. See [`crate::rewrite`].
    #[serde(default)]
    pub rewrite_enabled: bool,
    /// OpenCode model for the rewrite as `provider/model`, for example
    /// `zai-coding-plan/glm-5.3-flash`. Empty uses the service's default
    /// model.
    #[serde(default)]
    pub rewrite_model: String,
    /// The model's variant, for example `low` reasoning effort. Empty uses
    /// the model's default variant. Ignored when `rewrite_model` is empty.
    #[serde(default)]
    pub rewrite_variant: String,
    /// Contextual-biasing boost applied to every term in
    /// `vocabulary.txt`. Ignored when the vocabulary is empty.
    ///
    /// Measured against `parakeet-tdt-0.6b-v3-int8` on the bench
    /// fixtures: 1.0–6.0 leaves clean speech untouched, 10.0 starts
    /// injecting hotwords into audio that doesn't contain them, and
    /// 50.0 replaces the transcript with them outright. 2.0 is the
    /// conservative default; raise it only if a term is still being
    /// missed, and re-check with `asr_diff` that nothing else moved.
    #[serde(default = "default_hotword_score")]
    pub hotword_score: f32,
    /// Turn off to restore the original Hold behaviour: nothing is decoded
    /// until the key is released.
    #[serde(default = "default_hold_windows_enabled")]
    pub hold_windows_enabled: bool,
    /// Shortest window a pause may close. Defaults to the same value as
    /// `hold_window_max_seconds`, which turns pause cutting off: cutting at
    /// pauses measured 1.09 points of gold WER worse than cutting at the cap.
    /// Lower it only with fresh evidence. See ADR-0032.
    #[serde(default = "default_hold_window_min_seconds")]
    pub hold_window_min_seconds: f32,
    /// Longest window Hold mode accumulates before cutting. Hold decodes each
    /// closed window in the background while the key is still down, so this
    /// bounds the tail the user actually waits for on release and is the number
    /// that sets the release-to-text ceiling for a long utterance.
    #[serde(default = "default_hold_window_max_seconds")]
    pub hold_window_max_seconds: f32,
    /// Fire one throwaway dispatch at hotkey-down so the Neural Engine is
    /// awake by the time the endpoint arrives.
    ///
    /// The engine power-gates when idle and a cold encoder costs about 25 ms
    /// more than a warm one. Measured on an M5 Pro, the prime removes 60.9 ms
    /// at p95 on a 1 s Tap Fast utterance and 27.4 ms at 5 s; see
    /// `bench/README.md`. Off makes every dictation after a pause pay the
    /// re-wake.
    #[serde(default = "default_prime_engine_on_keydown")]
    pub prime_engine_on_keydown: bool,
}

/// Default for [`Settings::prime_engine_on_keydown`]. On: the measured win is
/// far outside noise and the cost is one silent dispatch per press.
fn default_prime_engine_on_keydown() -> bool {
    true
}

/// Default contextual-biasing boost. See [`Settings::hotword_score`].
fn default_hotword_score() -> f32 {
    2.0
}

fn default_hold_windows_enabled() -> bool {
    HoldWindowConfig::default().enabled
}

fn default_hold_window_min_seconds() -> f32 {
    HoldWindowConfig::default().min_seconds
}

fn default_hold_window_max_seconds() -> f32 {
    HoldWindowConfig::default().max_seconds
}

impl Settings {
    /// Hold-mode windowing policy. An out-of-range pair is rejected by
    /// [`HoldWindowConfig::validate`] at session start, which turns windowing
    /// off rather than failing the session.
    pub fn hold_windows(&self) -> HoldWindowConfig {
        HoldWindowConfig {
            enabled: self.hold_windows_enabled,
            min_seconds: self.hold_window_min_seconds,
            max_seconds: self.hold_window_max_seconds,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "CmdOrCtrl+Shift+Space".to_string(),
            trigger_mode: TriggerMode::default(),
            language: String::new(),
            rewrite_enabled: false,
            rewrite_model: String::new(),
            rewrite_variant: String::new(),
            hotword_score: default_hotword_score(),
            hold_windows_enabled: default_hold_windows_enabled(),
            hold_window_min_seconds: default_hold_window_min_seconds(),
            hold_window_max_seconds: default_hold_window_max_seconds(),
            prime_engine_on_keydown: default_prime_engine_on_keydown(),
        }
    }
}

#[derive(Clone)]
pub struct SettingsStore {
    settings_path: PathBuf,
    data_dir: PathBuf,
    cache: Arc<Mutex<Settings>>,
}

impl SettingsStore {
    pub fn new() -> Result<Self> {
        let data_root = dirs::data_dir().context("locating Application Support dir")?;
        Self::with_data_root(&data_root)
    }

    /// The store under `data_root` (Application Support in production).
    fn with_data_root(data_root: &Path) -> Result<Self> {
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
        // Pretty-printed so it's grep-friendly when debugging. Settings
        // hold no secret.
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

    /// Root passed to the native worker's model hub. The worker appends the
    /// pinned Parakeet Unified repository folder beneath it.
    pub fn coreml_model_root(&self) -> PathBuf {
        self.data_dir.join("models").join("coreml")
    }

    pub fn coreml_model_dir(&self) -> PathBuf {
        self.coreml_model_root()
            .join(crate::coreml_worker::COREML_MODEL_FOLDER)
    }

    /// Inspectable/removable, machine-derived ASR tuning evidence.
    pub fn asr_tuning_profile_path(&self) -> PathBuf {
        self.data_dir.join("asr-tuning-profile.json")
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

    /// User-editable vocabulary list — one term or phrase per line,
    /// written naturally. Opened by the Settings UI's "Edit
    /// Vocabulary…" button and translated into sherpa's hotwords
    /// format by `crate::vocabulary`.
    ///
    /// Lives directly in the data dir (not under `models/`) because
    /// it's user content, not a downloaded artefact — a user clearing
    /// out `models/` to force a re-download must not lose it.
    pub fn vocabulary_path(&self) -> PathBuf {
        self.data_dir.join("vocabulary.txt")
    }

    /// Machine-generated sherpa hotwords file, derived from
    /// [`Self::vocabulary_path`] at recogniser-load time. Regenerated
    /// from scratch on every load, so it's safe to delete.
    pub fn hotwords_path(&self) -> PathBuf {
        self.data_dir.join("hotwords.generated.txt")
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
    fn an_existing_settings_file_keeps_the_engine_prime_on() {
        // The field was added after users already had a settings.json. Without
        // the serde default, every existing install would silently read it as
        // false and go on paying the Neural Engine re-wake on every dictation.
        let existing = r#"{
            "hotkey": "CmdOrCtrl+Shift+Space",
            "language": "",
            "hotword_score": 2.0
        }"#;
        let settings: Settings = serde_json::from_str(existing).expect("old file must load");
        assert!(settings.prime_engine_on_keydown);
        assert!(Settings::default().prime_engine_on_keydown);
    }

    #[test]
    fn the_engine_prime_flag_round_trips_when_turned_off() {
        let settings = Settings {
            prime_engine_on_keydown: false,
            ..Settings::default()
        };
        let json = serde_json::to_string(&settings).expect("serialise");
        let back: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(!back.prime_engine_on_keydown);
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
    fn tap_modes_select_their_documented_endpoint_policies() {
        use crate::endpointing::EndpointPolicy;

        let legacy_tap: TriggerMode = serde_json::from_str("\"tap\"").unwrap();
        assert_eq!(legacy_tap, TriggerMode::Tap);
        assert_eq!(legacy_tap.endpoint_policy(), EndpointPolicy::LongForm);
        assert_eq!(TriggerMode::TapFast.endpoint_policy(), EndpointPolicy::Fast);
        let raw = serde_json::to_string(&Settings {
            trigger_mode: TriggerMode::TapFast,
            ..Settings::default()
        })
        .unwrap();
        assert!(raw.contains("\"trigger_mode\":\"tap_fast\""));
    }

    #[test]
    fn a_settings_file_from_the_polish_era_still_loads() {
        // settings.json written before the local polish model was removed
        // carries `polish_mode`. The loader ignores unknown keys, so the
        // user's other settings survive; a parse failure would silently
        // reset them all to defaults (`SettingsStore::new` falls back to
        // `Settings::default()`).
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join(BUNDLE_NAMESPACE);
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(
            data_dir.join("settings.json"),
            r#"{"hotkey":"Alt+Space","trigger_mode":"hold","language":"eng_Latn","polish_mode":"on","hotword_score":2.5}"#,
        )
        .unwrap();
        let store = SettingsStore::with_data_root(dir.path()).unwrap();
        let s = store.load();
        assert_eq!(s.hotkey, "Alt+Space");
        assert_eq!(s.trigger_mode, TriggerMode::Hold);
        assert_eq!(s.hotword_score, 2.5);
        assert!(!s.rewrite_enabled);
        assert_eq!(s.rewrite_model, "");
        assert_eq!(s.rewrite_variant, "");
    }

    #[test]
    fn settings_json_never_contains_api_key() {
        // The Anthropic-API era wrote an `anthropic_api_key` field into
        // settings.json. The OpenCode rewrite reads the service's own
        // loopback registration and needs no key. Pin that the wire shape
        // never leaks any *api_key* field.
        let s = Settings {
            rewrite_enabled: true,
            rewrite_model: "zai-coding-plan/glm-5.3-flash".to_string(),
            rewrite_variant: "low".to_string(),
            ..Settings::default()
        };
        let raw = serde_json::to_string(&s).unwrap();
        assert!(
            !raw.contains("api_key"),
            "settings.json must not contain any *api_key* field: {raw}"
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
        assert_eq!(
            store.coreml_model_dir(),
            store
                .data_dir()
                .join("models")
                .join("coreml")
                .join("parakeet-unified-en-0.6b")
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
        // `model_dir()`. Benchmark/evaluation binaries use the existence-only
        // `model_present()` preflight, while app startup performs the stronger
        // downloader integrity gate. If these lists drift, those entry points
        // disagree about whether the fallback model is installed.
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
