//! Session-local Silero VAD through a dedicated native Core ML worker.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use crate::coreml_worker::{CoreMlWorkerConfig, WorkerProcess};
use crate::endpointing::SPECULATIVE_MIN_SILENCE_S;

pub const VAD_SAMPLE_RATE: i32 = 16_000;
pub const MIN_SPEECH_S: f32 = 0.200;
pub const CANDIDATE_MIN_SPEECH_S: f32 = SPECULATIVE_MIN_SILENCE_S;
pub const MAX_SPEECH_S: f32 = 30.0;
/// Capture, Core ML inference, and endpoint policy advance in 32 ms frames.
pub const WINDOW_SIZE: i32 = 512;

pub struct Vad {
    process: WorkerProcess,
    state: SpeechState,
}

/// Duration confirmation stays independent of the inference implementation.
struct SpeechState {
    sample: u64,
    start: Option<u64>,
    end: Option<u64>,
    detected: bool,
    min_speech: u64,
    min_silence: u64,
}

impl SpeechState {
    fn observe(&mut self, probability: f32) {
        self.sample += WINDOW_SIZE as u64;
        if probability > 0.5 {
            self.end = None;
            match self.start {
                None => self.start = Some(self.sample),
                Some(start) if self.sample - start >= self.min_speech => self.detected = true,
                Some(_) => {}
            }
        } else if probability < 0.5 && !self.detected {
            self.start = None;
            self.end = None;
        } else if probability <= 0.35 {
            let end = *self.end.get_or_insert(self.sample);
            if self.sample - end >= self.min_silence {
                self.detected = false;
                self.start = None;
                self.end = None;
            }
        }
    }
}

impl Vad {
    pub fn load_candidate(model: &Path) -> Result<Self> {
        Self::load_with_durations(model, SPECULATIVE_MIN_SILENCE_S, CANDIDATE_MIN_SPEECH_S)
    }

    pub fn load_confirming(model: &Path, confirmation_ms: u32) -> Result<Self> {
        Self::load_with_durations(model, confirmation_ms as f32 / 1_000.0, MIN_SPEECH_S)
    }

    fn load_with_durations(model: &Path, min_silence: f32, min_speech: f32) -> Result<Self> {
        if !model.is_dir() {
            return Err(anyhow!(
                "Core ML Silero VAD model missing: {}",
                model.display()
            ));
        }
        let worker = CoreMlWorkerConfig::discover()?.worker_path;
        let mut child = Command::new(worker)
            .arg("--vad-model")
            .arg(model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting Core ML VAD worker")?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("VAD worker stdin missing"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("VAD worker stdout missing"))?;
        let mut process = WorkerProcess::new(child, stdin, stdout)?;
        process.transaction(Duration::from_secs(60), |p, deadline| {
            p.read_ready(deadline)
        })?;
        Ok(Self {
            process,
            state: SpeechState {
                sample: 0,
                start: None,
                end: None,
                detected: false,
                min_speech: (min_speech * VAD_SAMPLE_RATE as f32) as u64,
                min_silence: (min_silence * VAD_SAMPLE_RATE as f32) as u64,
            },
        })
    }

    pub fn accept_waveform(&mut self, samples: &[f32]) -> Result<()> {
        if samples.len() != WINDOW_SIZE as usize {
            return Err(anyhow!("VAD expects a 512-sample capture frame"));
        }
        let probability = self
            .process
            .transaction(Duration::from_secs(2), |p, deadline| {
                p.vad_probability(samples, deadline)
            })?;
        self.state.observe(probability);
        Ok(())
    }

    pub fn detected(&self) -> bool {
        self.state.detected
    }

    /// Stop inference now; Drop reaps the process after the tail ASR completes.
    pub(crate) fn request_shutdown(&mut self) {
        self.process.request_shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirmation_rejects_a_click_and_waits_through_a_short_pause() {
        let mut state = SpeechState {
            sample: 0,
            start: None,
            end: None,
            detected: false,
            min_speech: 3200,
            min_silence: 8800,
        };
        state.observe(0.9);
        assert!(!state.detected);
        state.observe(0.0);
        assert!(!state.detected);
        for _ in 0..8 {
            state.observe(0.9);
        }
        assert!(state.detected);
        state.observe(0.0);
        state.observe(0.0);
        assert!(state.detected);
        state.observe(0.9);
        assert!(state.detected);
        for _ in 0..18 {
            state.observe(0.0);
        }
        assert!(state.detected);
        state.observe(0.0);
        assert!(!state.detected);
    }
}
