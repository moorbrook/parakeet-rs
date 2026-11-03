//! Resident Core ML recognition facade, timing, and token-span handling.

use std::sync::Arc;

use crate::windows::{words_from_tokens, TokenSpan, Word};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Below this real-time factor, report unexpectedly slow recognition.
const RTFX_COREML_FLOOR: f32 = 2.0;

/// Recognition engine used by [`Asr`].
///
/// Backends return their own measured inference time so the common facade can
/// apply one quality/performance policy without hiding backend-specific work.
/// Implementations should not log timing themselves.
pub trait AsrBackend: Send + Sync {
    fn metadata(&self) -> &AsrBackendMetadata;
    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded>;

    /// Whether [`AsrBackend::transcribe_with_token_spans`] reports real spans.
    ///
    /// Hold-mode incremental windowing needs word boundaries to join two
    /// windows on their overlap, so it asks this before it starts cutting and
    /// otherwise leaves the session on the plain single-decode path.
    fn reports_token_spans(&self) -> bool {
        false
    }

    /// Decode, and additionally report the span of every RNNT emission on the
    /// clip's own timeline. Backends that cannot report them return an empty
    /// vector; callers must check [`AsrBackend::reports_token_spans`] rather
    /// than reading an empty result as "no speech".
    fn transcribe_with_token_spans(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<(Decoded, Vec<TokenSpan>)> {
        Ok((self.transcribe(samples, sample_rate)?, Vec::new()))
    }

    /// Resident bytes owned by helper processes outside this Rust process.
    fn auxiliary_resident_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    /// Per-stage breakdown of the most recent `transcribe`, when the backend
    /// was asked to collect one. Only the native Core ML worker reports this,
    /// and only when the bench enables its stage profiler.
    fn last_stage_report(&self) -> Option<StageReport> {
        None
    }

    /// What the backend made of a custom vocabulary it was given, or `None`
    /// when it was given none.
    fn contextual_vocabulary(&self) -> Option<VocabularyStatus> {
        None
    }
}

/// A backend's verdict on the custom vocabulary it was asked to bias toward.
///
/// `encoded` and `rejected` carry the user's own words, so they are for logs
/// and for anything that shows the user what happened to their vocabulary;
/// quality reports record only the counts.
///
/// Acceptance is weaker than it looks. The worker splits each term with the
/// model bundle's piece inventory by longest match, because the bundle ships
/// the inventory but not the merge ranks a faithful BPE encoder would need. A
/// term whose longest-match split differs from the segmentation the model
/// actually emits is *accepted* and biases a path the joint never walks, and
/// nothing downstream detects that. `encoded` is the only handle on it: it
/// reports the split the worker chose, per term, so a term that is doing
/// nothing can at least be recognized by eye.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VocabularyStatus {
    /// Terms the model's token inventory could represent.
    pub accepted: u32,
    /// Each accepted term and the pieces it was split into.
    pub encoded: Vec<EncodedTerm>,
    /// Terms it could not represent at all, which are therefore boosting
    /// nothing.
    pub rejected: Vec<String>,
}

/// One accepted vocabulary term, and the pieces the worker split it into.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
pub struct EncodedTerm {
    pub term: String,
    pub pieces: Vec<String>,
}

/// Where one utterance's decode time went inside the recognizer.
///
/// The native worker derives these from its Core ML dispatch timeline, so the
/// call counts are exact and the durations partition the worker-internal decode
/// interval:
/// `mel_ms + preprocessor_ms + encoder_ms + decode_loop_ms + post_ms == total_ms`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StageReport {
    /// Sample-rate conversion inside the worker, before any model runs. Near
    /// zero since ADR-0030: Rust hands the worker 16 kHz and FluidAudio's
    /// converter returns matching-rate input untouched.
    pub resample_ms: f64,
    /// Fixed 15 s encoder windows the utterance was split into.
    pub windows: u32,
    /// Mel-front-end Core ML dispatches. Zero for Parakeet Unified, which
    /// computes mel in Swift; one per window for TDT v3, whose front end is a
    /// `Preprocessor` graph. Defaulted so a payload captured before the stage
    /// existed still deserializes.
    #[serde(default)]
    pub preprocessor_calls: u32,
    pub encoder_calls: u32,
    pub decoder_calls: u32,
    pub joint_calls: u32,
    /// Prediction-network steps run natively, without a Core ML dispatch. The
    /// native and Core ML counts are kept apart so a row cannot claim both
    /// engines ran; exactly one of each pair is nonzero.
    pub native_decoder_steps: u32,
    /// Native joint evaluations: one decoder-side projection per step, plus one
    /// decision per frame and per emitted token.
    pub native_joint_steps: u32,
    /// Core ML predictions that matched none of the three known input shapes.
    /// A nonzero value means the pipeline changed and the split is suspect.
    pub other_calls: u32,
    /// Host-side work before each mel or encoder call. For Unified that is the
    /// whole Swift log-mel extraction; for TDT it is only the marshalling
    /// around `preprocessor_ms`.
    pub mel_ms: f64,
    /// Time inside the TDT mel front-end dispatches. Zero for Unified.
    #[serde(default)]
    pub preprocessor_ms: f64,
    pub encoder_ms: f64,
    /// Wall time inside the greedy RNNT loop, dispatch plus loop overhead.
    pub decode_loop_ms: f64,
    /// The Core ML dispatches alone, without the Swift-side loop overhead.
    pub decode_loop_dispatch_ms: f64,
    pub decoder_dispatch_ms: f64,
    pub joint_dispatch_ms: f64,
    /// Wall time inside the native steps, the counterpart of
    /// `decode_loop_dispatch_ms` when the loop runs without Core ML.
    pub decode_loop_native_ms: f64,
    /// Tokenizer decode and overlap merge after the last dispatch.
    pub post_ms: f64,
    /// Dispatch time counted more than once because two Core ML predictions
    /// were in flight at once. The stage durations are timeline sums, so they
    /// partition the decode interval only while dispatch is serial; FluidAudio's
    /// TDT long-form path can decode chunks concurrently. Nonzero means the
    /// wall totals are still right and the per-stage split is not.
    #[serde(default)]
    pub overlapped_dispatch_ms: f64,
    pub total_ms: f64,
    /// Compute units read off each live model, e.g.
    /// `encoder=cpu-and-neural-engine decoder=cpu-only joint=cpu-only`. A
    /// natively run stage reports `native`, which is not a Core ML placement.
    pub compute_units: String,
}

/// Identity of the exact model/runtime artifact behind an [`AsrBackend`].
///
/// Benchmark and quality reports read this from the backend rather than from
/// CLI constants, so adding a challenger cannot accidentally label its results
/// as the shipping model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AsrBackendMetadata {
    pub backend: String,
    pub model: String,
    pub quantization: String,
    pub execution_provider: String,
}

/// Stable recognition facade used by the app.
///
/// Production and evaluation share capture, warmup, timing, and transcript
/// handling through this facade.
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

impl Asr {
    /// Construct the app-facing recognizer around an explicit backend.
    ///
    /// This constructor is the seam
    /// for benchmark challengers and deterministic test doubles.
    pub fn from_backend(backend: Arc<dyn AsrBackend>) -> Self {
        Self { backend }
    }

    pub fn backend_metadata(&self) -> &AsrBackendMetadata {
        self.backend.metadata()
    }

    pub fn auxiliary_resident_bytes(&self) -> Result<u64> {
        self.backend.auxiliary_resident_bytes()
    }

    /// Stage breakdown of the most recent decode, when the backend collects one.
    pub fn last_stage_report(&self) -> Option<StageReport> {
        self.backend.last_stage_report()
    }

    /// The active contextual vocabulary, when this backend has one.
    pub fn contextual_vocabulary(&self) -> Option<VocabularyStatus> {
        self.backend.contextual_vocabulary()
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

    /// Whether this recognizer can report word boundaries, which is what
    /// Hold-mode incremental windowing needs to join two windows.
    pub fn reports_token_spans(&self) -> bool {
        self.backend.reports_token_spans()
    }

    /// Decode one window of a longer recording and return its words placed on
    /// the recording's timeline, `offset_s` being where the window starts.
    ///
    /// Deliberately not routed through `recognize_with_timing`: a window is a
    /// fragment, so its RTFx is not the utterance's and the CoreML-fallback
    /// warning would fire on short tails. The session logs window timings
    /// itself.
    pub fn recognize_window(
        &self,
        samples: &[f32],
        sample_rate: u32,
        offset_s: f32,
    ) -> Result<Vec<Word>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let (_, spans) = self
            .backend
            .transcribe_with_token_spans(samples, sample_rate)?;
        Ok(words_from_tokens(&spans, offset_s))
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
                     floor of {RTFX_COREML_FLOOR:.1}x; inference is slower than expected.",
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
