//! Local Parakeet TDT 0.6B v3 ASR via sherpa-onnx.
//!
//! Holds a single `OfflineRecognizer` for the life of the app; reused across
//! every hotkey press so CoreML doesn't recompile its graph each time.
//!
//! ADR-0015 layer 3: every `recognize` call records decode-time vs audio-time
//! (RTFx). On this M5 Pro, CoreML-resident execution should sit comfortably
//! above 5x real-time. A sustained drop below 2x is the signal that
//! provider="coreml" silently fell back to CPU — surfaced as a `log::warn`.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
};

/// Below this real-time factor we assume CoreML is not engaged.
const RTFX_COREML_FLOOR: f32 = 2.0;

pub struct Asr {
    inner: Arc<Mutex<OfflineRecognizer>>,
}

pub struct Decoded {
    pub text: String,
    pub audio_seconds: f32,
    pub decode_seconds: f32,
}

impl Decoded {
    pub fn rtfx(&self) -> f32 {
        if self.decode_seconds > 0.0 {
            self.audio_seconds / self.decode_seconds
        } else {
            f32::INFINITY
        }
    }
}

impl Asr {
    pub fn load(
        encoder: &Path,
        decoder: &Path,
        joiner: &Path,
        tokens: &Path,
        num_threads: i32,
    ) -> Result<Self> {
        for p in [encoder, decoder, joiner, tokens] {
            if !p.exists() {
                return Err(anyhow!("missing model file: {}", p.display()));
            }
        }

        // ADR-0015 layer 2: log what we *asked* for and what build.rs
        // (layer 1) detected in the static lib. sherpa-onnx's Rust surface
        // doesn't expose the effective provider after creation, so the
        // empirical signal lives in the per-utterance RTFx probe below.
        if cfg!(parakeet_coreml_ep_present) {
            log::info!("ASR provider requested: coreml (EP symbol present in libonnxruntime.a)");
        } else {
            log::warn!(
                "ASR provider requested: coreml — but libonnxruntime.a has \
                 no CoreML EP symbol (see build.rs warning). Expect CPU \
                 fallback."
            );
        }

        let config = OfflineRecognizerConfig {
            model_config: OfflineModelConfig {
                transducer: OfflineTransducerModelConfig {
                    encoder: Some(encoder.to_string_lossy().into_owned()),
                    decoder: Some(decoder.to_string_lossy().into_owned()),
                    joiner: Some(joiner.to_string_lossy().into_owned()),
                },
                tokens: Some(tokens.to_string_lossy().into_owned()),
                num_threads,
                provider: Some("coreml".to_string()),
                model_type: Some("nemo_transducer".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("OfflineRecognizer::create returned None"))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(recognizer)),
        })
    }

    pub fn recognize(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        let decoded =
            self.recognize_with_timing(samples, sample_rate, /* warmup = */ false)?;
        Ok(decoded.text)
    }

    /// Like `recognize` but doesn't log RTFx — used by the throwaway-first
    /// pass of `warmup::dummy_decode`, where timing is dominated by CoreML
    /// graph compilation rather than steady-state inference.
    pub fn recognize_silent_warmup(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        let decoded = self.recognize_with_timing(samples, sample_rate, /* warmup = */ true)?;
        Ok(decoded.text)
    }

    fn recognize_with_timing(
        &self,
        samples: &[f32],
        sample_rate: u32,
        warmup: bool,
    ) -> Result<Decoded> {
        if samples.is_empty() {
            return Ok(Decoded {
                text: String::new(),
                audio_seconds: 0.0,
                decode_seconds: 0.0,
            });
        }
        let recognizer = self.inner.lock();
        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate as i32, samples);

        let start = Instant::now();
        recognizer.decode(&stream);
        let decode_seconds = start.elapsed().as_secs_f32();

        let result = stream
            .get_result()
            .ok_or_else(|| anyhow!("get_result returned None"))?;
        let audio_seconds = samples.len() as f32 / sample_rate as f32;

        let decoded = Decoded {
            text: result.text.trim().to_string(),
            audio_seconds,
            decode_seconds,
        };

        if !warmup {
            let rtfx = decoded.rtfx();
            // Only warn on segments long enough for steady-state inference to
            // dominate setup cost. ≥1.5 s catches typical dictation utterances
            // and skips single-word "yes"/"no" replies + the warmup pass.
            if decoded.audio_seconds >= 1.5 && rtfx < RTFX_COREML_FLOOR {
                log::warn!(
                    "ASR RTFx {rtfx:.2}x on {:.2}s of audio is below the CoreML \
                     floor of {RTFX_COREML_FLOOR:.1}x — provider=\"coreml\" is \
                     almost certainly falling back to CPU. See ADR-0015.",
                    decoded.audio_seconds
                );
            } else {
                log::info!(
                    "ASR decoded {:.2}s in {:.3}s ({:.1}x real time)",
                    decoded.audio_seconds,
                    decoded.decode_seconds,
                    rtfx
                );
            }
        }
        Ok(decoded)
    }
}
