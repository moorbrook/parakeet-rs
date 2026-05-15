use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

/// Persisted user preferences. No API key, no provider — all transcription is
/// local via Omnilingual ASR.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
    pub inject_mode: String,
    /// Language hint for the recognizer, e.g. "eng_Latn". Empty = autodetect.
    pub language: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "CmdOrCtrl+Shift+Space".to_string(),
            inject_mode: "paste".to_string(),
            language: String::new(),
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
    pub fn new(app: AppHandle) -> Result<Self> {
        let config_dir = app.path().app_config_dir().context("app config dir")?;
        let data_dir = app.path().app_data_dir().context("app data dir")?;
        std::fs::create_dir_all(&config_dir).context("create config dir")?;
        std::fs::create_dir_all(&data_dir).context("create data dir")?;
        let settings_path = config_dir.join("settings.json");
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
        *self.cache.lock() = s.clone();
        let raw = serde_json::to_string_pretty(s)?;
        std::fs::write(&self.settings_path, raw)?;
        Ok(())
    }

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
