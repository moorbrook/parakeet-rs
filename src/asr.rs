//! Stable local-ASR facade and the sherpa fallback backend.
//!
//! The app prefers the resident native Core ML Parakeet Unified worker and
//! selects sherpa-onnx when contextual vocabulary is active or specialized
//! model setup fails. Each backend remains resident across hotkey presses so
//! model loading and Core ML graph compilation stay off the dictation path.
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
use serde::Serialize;
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
};

/// Below this real-time factor we assume CoreML is not engaged.
const RTFX_COREML_FLOOR: f32 = 2.0;

/// Recognition engine used by [`Asr`].
///
/// Backends return their own measured inference time so the common facade can
/// apply one quality/performance policy without hiding backend-specific work.
/// Implementations should not log timing themselves.
pub trait AsrBackend: Send + Sync {
    fn metadata(&self) -> &AsrBackendMetadata;
    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded>;
}

/// Identity of the exact model/runtime artifact behind an [`AsrBackend`].
///
/// Benchmark and quality reports read this from the backend rather than from
/// CLI constants, so adding a challenger cannot accidentally label its results
/// as the shipping sherpa model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AsrBackendMetadata {
    pub backend: String,
    pub model: String,
    pub quantization: String,
    pub execution_provider: String,
}

/// Stable recognition facade used by the app.
///
/// The current default backend is sherpa-onnx. Keeping callers behind this
/// facade lets native Core ML or MLX experiments use the same capture, warmup,
/// timing, and transcript-handling path.
pub struct Asr {
    backend: Arc<dyn AsrBackend>,
}

#[derive(Debug, Clone, PartialEq)]
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

/// Everything needed to construct the recogniser. A struct rather than
/// a positional argument list because the biasing fields only make
/// sense together, and `load(a, b, c, d, 8, None, 0.0)` at the call
/// site says nothing about which `0.0` that is.
pub struct AsrConfig<'a> {
    pub encoder: &'a Path,
    pub decoder: &'a Path,
    pub joiner: &'a Path,
    pub tokens: &'a Path,
    pub num_threads: i32,
    /// Generated sherpa hotwords file (see `crate::vocabulary`).
    ///
    /// `None` selects greedy decoding. `Some` selects modified beam
    /// search, which is the **only** decoding method sherpa accepts
    /// alongside a hotwords file — passing one with greedy makes
    /// `OfflineRecognizer::create` fail outright — and which measured
    /// ~13% slower on the 5 s bench fixture. See ADR-0020.
    pub hotwords: Option<&'a Path>,
    /// Boost applied to biased terms. Only read when `hotwords` is
    /// `Some`. See [`crate::settings::Settings::hotword_score`] for the
    /// measured safe range.
    pub hotwords_score: f32,
}

struct SherpaBackend {
    inner: Mutex<OfflineRecognizer>,
    metadata: AsrBackendMetadata,
}

impl AsrBackend for SherpaBackend {
    fn metadata(&self) -> &AsrBackendMetadata {
        &self.metadata
    }

    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
        let recognizer = self.inner.lock();
        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate as i32, samples);

        let start = Instant::now();
        recognizer.decode(&stream);
        let decode_seconds = start.elapsed().as_secs_f32();

        let result = stream
            .get_result()
            .ok_or_else(|| anyhow!("get_result returned None"))?;

        Ok(Decoded {
            text: result.text,
            audio_seconds: samples.len() as f32 / sample_rate as f32,
            decode_seconds,
        })
    }
}

impl Asr {
    pub fn load(cfg: &AsrConfig<'_>) -> Result<Self> {
        let AsrConfig {
            encoder,
            decoder,
            joiner,
            tokens,
            num_threads,
            hotwords,
            hotwords_score,
        } = *cfg;
        for p in [encoder, decoder, joiner, tokens] {
            if !p.exists() {
                return Err(anyhow!("missing model file: {}", p.display()));
            }
        }

        // Greedy unless the user actually has vocabulary terms — beam
        // search is a real cost to pay for a feature nobody enabled.
        let (decoding_method, hotwords_file, hotwords_score) = match hotwords {
            Some(path) => {
                log::info!(
                    "ASR contextual biasing ON (score {hotwords_score}): {}",
                    path.display()
                );
                (
                    "modified_beam_search",
                    Some(path.to_string_lossy().into_owned()),
                    hotwords_score,
                )
            }
            None => ("greedy_search", None, 0.0),
        };

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
            decoding_method: Some(decoding_method.to_string()),
            hotwords_file,
            hotwords_score,
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("OfflineRecognizer::create returned None"))?;
        Ok(Self::from_backend(Arc::new(SherpaBackend {
            inner: Mutex::new(recognizer),
            metadata: AsrBackendMetadata {
                backend: "sherpa-onnx".to_string(),
                model: "NVIDIA Parakeet TDT 0.6B v3".to_string(),
                quantization: "int8".to_string(),
                execution_provider: "coreml-requested".to_string(),
            },
        })))
    }

    /// Construct the app-facing recognizer around an explicit backend.
    ///
    /// Production startup uses [`Self::load`]. This constructor is the seam
    /// for benchmark challengers and deterministic test doubles.
    pub fn from_backend(backend: Arc<dyn AsrBackend>) -> Self {
        Self { backend }
    }

    pub fn backend_metadata(&self) -> &AsrBackendMetadata {
        self.backend.metadata()
    }

    pub fn recognize(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        let decoded = self.recognize_with_metrics(samples, sample_rate)?;
        Ok(decoded.text)
    }

    /// Recognize an utterance and return the metrics needed by quality and
    /// backend-comparison harnesses.
    pub fn recognize_with_metrics(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
        self.recognize_with_timing(samples, sample_rate, /* warmup = */ false)
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
        let mut decoded = self.backend.transcribe(samples, sample_rate)?;
        decoded.text = decoded.text.trim().to_string();

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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct FakeBackend {
        calls: AtomicUsize,
        metadata: AsrBackendMetadata,
    }

    impl AsrBackend for FakeBackend {
        fn metadata(&self) -> &AsrBackendMetadata {
            &self.metadata
        }

        fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Decoded {
                text: "  hello from a challenger  ".to_string(),
                audio_seconds: samples.len() as f32 / sample_rate as f32,
                decode_seconds: 0.25,
            })
        }
    }

    #[test]
    fn explicit_backend_is_used_and_text_is_normalized() {
        let backend = Arc::new(FakeBackend {
            calls: AtomicUsize::new(0),
            metadata: AsrBackendMetadata {
                backend: "fake".to_string(),
                model: "test model".to_string(),
                quantization: "none".to_string(),
                execution_provider: "cpu".to_string(),
            },
        });
        let asr = Asr::from_backend(backend.clone());

        let decoded = asr
            .recognize_with_metrics(&vec![0.0; 16_000], 16_000)
            .expect("fake backend should transcribe");

        assert_eq!(decoded.text, "hello from a challenger");
        assert_eq!(decoded.audio_seconds, 1.0);
        assert_eq!(decoded.decode_seconds, 0.25);
        assert_eq!(asr.backend_metadata(), &backend.metadata);
        assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn empty_audio_does_not_call_backend() {
        let backend = Arc::new(FakeBackend {
            calls: AtomicUsize::new(0),
            metadata: AsrBackendMetadata {
                backend: "fake".to_string(),
                model: "test model".to_string(),
                quantization: "none".to_string(),
                execution_provider: "cpu".to_string(),
            },
        });
        let asr = Asr::from_backend(backend.clone());

        let decoded = asr
            .recognize_with_metrics(&[], 16_000)
            .expect("empty audio should be accepted");

        assert_eq!(
            decoded,
            Decoded {
                text: String::new(),
                audio_seconds: 0.0,
                decode_seconds: 0.0,
            }
        );
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    }
}
