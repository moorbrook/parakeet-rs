//! Silero VAD wrapper.
//!
//! One `Vad` instance is built per dictation session — Silero state is short
//! enough that recreating it is cheap, and a fresh detector means no carry-over
//! from a previous utterance.

use std::path::Path;

use anyhow::{Result, anyhow};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

/// Silero operates at 16 kHz natively.
pub const VAD_SAMPLE_RATE: i32 = 16_000;

/// 150 ms of trailing silence ends a segment (matches ADR-0009 latency budget).
pub const MIN_SILENCE_S: f32 = 0.150;
/// Discard segments shorter than 200 ms — almost always a stray click / breath.
pub const MIN_SPEECH_S: f32 = 0.200;
/// Hard cap so a microphone left hot can't grow an unbounded segment.
pub const MAX_SPEECH_S: f32 = 30.0;
/// Silero's expected window size, in samples at 16 kHz.
pub const WINDOW_SIZE: i32 = 512;

pub struct Vad {
    inner: VoiceActivityDetector,
}

impl Vad {
    pub fn load(model: &Path, num_threads: i32) -> Result<Self> {
        if !model.exists() {
            return Err(anyhow!("silero VAD model missing: {}", model.display()));
        }
        let cfg = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model.to_string_lossy().into_owned()),
                threshold: 0.5,
                min_silence_duration: MIN_SILENCE_S,
                min_speech_duration: MIN_SPEECH_S,
                window_size: WINDOW_SIZE,
                max_speech_duration: MAX_SPEECH_S,
            },
            sample_rate: VAD_SAMPLE_RATE,
            num_threads,
            // Silero is a tiny RNN; CPU is faster than CoreML round-trips here.
            provider: Some("cpu".to_string()),
            debug: false,
            ..Default::default()
        };
        let detector = VoiceActivityDetector::create(&cfg, 30.0)
            .ok_or_else(|| anyhow!("VoiceActivityDetector::create returned None"))?;
        Ok(Self { inner: detector })
    }

    /// Feed 16 kHz samples; one Silero window-sized slice at a time is ideal.
    pub fn accept_waveform(&self, samples: &[f32]) {
        self.inner.accept_waveform(samples);
    }

    /// True while Silero believes speech is currently happening.
    pub fn detected(&self) -> bool {
        self.inner.detected()
    }

    /// Pull all completed speech segments out and drop them — we don't use
    /// Silero's own segmentation, only the start-of-speech edge to know we
    /// can stop waiting on silence.
    pub fn drain_segments(&self) {
        while !self.inner.is_empty() {
            self.inner.pop();
        }
    }

}
