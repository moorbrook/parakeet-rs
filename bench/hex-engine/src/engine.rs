//! Hex's transcription engine, reduced to what decides its speed and output.
//!
//! Each choice below mirrors Hex at commit 46c3c762 (`src/parakeet.rs`,
//! `src/gguf_session.rs`, `src/transcription_engine.rs`, `src/dictation.rs`)
//! rather than transcribe.cpp's defaults, so this arm measures the engine
//! Hex ships:
//!
//! - Metal backend, and the load fails if transcribe.cpp picks anything else.
//! - One ggml thread on a GPU device (`session_options`).
//! - Timestamps requested at the finest kind the model offers, capped at word.
//! - No vocabulary extension, matching our greedy shipping row.
//! - Logging off, as the app runs it outside diagnostics.
//! - Offline clips padded with silence to 1.5 s (`pad_for_parakeet`).
//! - The session is dropped and recreated around every run and stream.
//! - A decoder stopped for repetition returns its partial text.
//! - Clips longer than the model's `max_audio_ms` are cut into chunks.
//! - Streams use Parakeet's default buffered context (no extension) and are
//!   fed in blocks of at least 2,560 samples; the finish feeds whatever the
//!   stream has not seen, then finalizes and returns the trimmed full text.
//!
//! Not mirrored: Hex's pause cap (it only acts on pauses over 2.5 s, which no
//! fixture contains), its 450 ms pre-roll and 200 ms post-release tail (app
//! capture policy, kept identical across both arms by the harness), and its
//! rubato resampler (both arms get the same audio from our resampler).

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use parakeet_dictation::asr::{AsrBackend, AsrBackendMetadata, Decoded};
use transcribe_cpp::{
    Backend, DeviceType, Model, ModelOptions, RunOptions, Session, SessionOptions, StreamOptions,
    TimestampKind,
};

pub const SAMPLE_RATE: u32 = 16_000;
/// `MINIMUM_PARAKEET_DURATION` in Hex's `dictation.rs`.
const MINIMUM_SAMPLES: usize = 24_000;
/// `STREAM_FEED_SAMPLES` in Hex's `dictation.rs`.
pub const STREAM_FEED_SAMPLES: usize = 2_560;
/// `STREAM_PREWARM_SAMPLES` in Hex's `gguf_session.rs`.
const STREAM_PREWARM_SAMPLES: usize = 8 * 16_000;

pub struct HexEngine {
    model: Model,
    session: Option<Session>,
    session_options: SessionOptions,
    run_options: RunOptions,
    stream_options: StreamOptions,
    max_audio_samples: Option<usize>,
    pub metadata: AsrBackendMetadata,
}

pub fn pad_for_parakeet(samples: &mut Vec<f32>) {
    samples.resize(samples.len().max(MINIMUM_SAMPLES), 0.0);
}

impl HexEngine {
    pub fn load(path: &Path) -> Result<Self> {
        transcribe_cpp::disable_logging();
        let model = Model::load_with(
            path,
            &ModelOptions {
                backend: Backend::Metal,
                ..Default::default()
            },
        )
        .with_context(|| format!("loading {}", path.display()))?;
        let device = model.device()?;
        if device.kind != "metal" {
            bail!(
                "transcribe.cpp selected {} ({}) instead of Metal",
                device.kind,
                device.name
            );
        }
        let capabilities = model.capabilities()?;
        if !capabilities.supports_streaming {
            bail!("{} does not advertise streaming", path.display());
        }
        let session_options = match device.device_type {
            DeviceType::Gpu | DeviceType::Igpu => SessionOptions {
                n_threads: 1,
                ..Default::default()
            },
            _ => SessionOptions::default(),
        };
        let run_options = RunOptions {
            timestamps: match capabilities.max_timestamp_kind {
                TimestampKind::Word | TimestampKind::Token => TimestampKind::Word,
                TimestampKind::Auto | TimestampKind::Segment => TimestampKind::Segment,
                TimestampKind::None => TimestampKind::None,
            },
            ..Default::default()
        };
        let max_audio_samples = usize::try_from(capabilities.max_audio_ms)
            .ok()
            .filter(|ms| *ms > 0)
            .and_then(|ms| ms.checked_mul(16));
        let metadata = AsrBackendMetadata {
            backend: "transcribe.cpp (Hex engine)".to_string(),
            model: format!("{}/{}", model.arch(), model.variant()),
            quantization: "Q8_0 GGUF".to_string(),
            execution_provider: format!("{} ({})", device.kind, device.description),
        };
        let session = model.session_with(&session_options)?;
        Ok(Self {
            model,
            session: Some(session),
            session_options,
            run_options,
            stream_options: StreamOptions::default(),
            max_audio_samples,
            metadata,
        })
    }

    fn take_session(&mut self) -> Result<Session> {
        match self.session.take() {
            Some(session) => Ok(session),
            None => Ok(self.model.session_with(&self.session_options)?),
        }
    }

    fn run(&mut self, samples: &[f32]) -> Result<String> {
        let mut session = self.take_session()?;
        let result = match session.run(samples, &self.run_options) {
            Ok(transcript) => Ok(transcript.text),
            Err(transcribe_cpp::Error::OutputRepetition {
                partial: Some(partial),
                ..
            }) => Ok(partial.text),
            Err(error) => Err(anyhow!(error)),
        };
        drop(session);
        self.session = Some(self.model.session_with(&self.session_options)?);
        result
    }

    /// Hex's offline path: `pad_for_parakeet`, then `transcribe_chunked`.
    pub fn transcribe_offline(&mut self, samples: &[f32]) -> Result<String> {
        let mut clip = samples.to_vec();
        pad_for_parakeet(&mut clip);
        match self.max_audio_samples.filter(|max| clip.len() > *max) {
            None => self.run(&clip),
            Some(max) => {
                let mut output = Vec::new();
                for chunk in clip.chunks(max) {
                    let mut chunk = chunk.to_vec();
                    pad_for_parakeet(&mut chunk);
                    let text = self.run(&chunk)?;
                    let text = text.trim();
                    if !text.is_empty() {
                        output.push(text.to_string());
                    }
                }
                Ok(output.join(" "))
            }
        }
    }

    /// One live stream. `drive` feeds it; the session rotates afterwards.
    pub fn with_stream<R>(
        &mut self,
        drive: impl FnOnce(&mut LiveFeed<'_, '_>) -> Result<R>,
    ) -> Result<R> {
        let mut session = self.take_session()?;
        let result = {
            let mut stream = session.stream(&self.run_options, &self.stream_options)?;
            let mut feed = LiveFeed {
                stream: &mut stream,
                pending: Vec::with_capacity(STREAM_FEED_SAMPLES * 4),
                fed: 0,
            };
            drive(&mut feed)
        };
        drop(session);
        self.session = Some(self.model.session_with(&self.session_options)?);
        result
    }

    /// Hex's stream probe, which also compiles the stream graph.
    pub fn prewarm_stream(&mut self) -> Result<()> {
        self.with_stream(|feed| {
            feed.push(&vec![0.0; STREAM_PREWARM_SAMPLES])?;
            feed.finish().map(drop)
        })
    }

    /// Feeds a whole clip in capture-sized blocks, then finishes.
    pub fn transcribe_streamed(&mut self, samples: &[f32]) -> Result<String> {
        self.with_stream(|feed| {
            for block in samples.chunks(STREAM_FEED_SAMPLES) {
                feed.push(block)?;
            }
            feed.finish()
        })
    }
}

/// Hex's capture-side feeding: audio accumulates until a block of at least
/// [`STREAM_FEED_SAMPLES`] is ready, and the whole backlog goes in one feed.
pub struct LiveFeed<'s, 'm> {
    stream: &'s mut transcribe_cpp::Stream<'m>,
    pending: Vec<f32>,
    pub fed: usize,
}

impl LiveFeed<'_, '_> {
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.pending.extend_from_slice(samples);
        if self.pending.len() >= STREAM_FEED_SAMPLES {
            self.stream.feed(&self.pending)?;
            self.fed += self.pending.len();
            self.pending.clear();
        }
        Ok(())
    }

    /// `finish_stream`: feed the unseen remainder, finalize, return the text.
    pub fn finish(&mut self) -> Result<String> {
        if !self.pending.is_empty() {
            self.stream.feed(&self.pending)?;
            self.fed += self.pending.len();
            self.pending.clear();
        }
        self.stream.finalize()?;
        Ok(self.stream.text().full.trim().to_string())
    }
}

/// The engine behind our `Asr` facade, so the Hold harness can drive it
/// through the app's own capture and release path. It reports no token
/// spans, which keeps the streamer on its single-decode Hold path.
pub struct HexOfflineBackend {
    engine: Mutex<HexEngine>,
    metadata: AsrBackendMetadata,
}

impl HexOfflineBackend {
    pub fn new(engine: HexEngine) -> Self {
        let metadata = engine.metadata.clone();
        Self {
            engine: Mutex::new(engine),
            metadata,
        }
    }

    /// Exclusive access for the stream paths, which the facade cannot express.
    pub fn with_engine<R>(&self, work: impl FnOnce(&mut HexEngine) -> Result<R>) -> Result<R> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| anyhow!("Hex engine mutex poisoned"))?;
        work(&mut engine)
    }
}

impl AsrBackend for HexOfflineBackend {
    fn metadata(&self) -> &AsrBackendMetadata {
        &self.metadata
    }

    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
        if sample_rate != SAMPLE_RATE {
            bail!("the Hex engine takes {SAMPLE_RATE} Hz audio, got {sample_rate} Hz");
        }
        let started = Instant::now();
        let text = self
            .engine
            .lock()
            .map_err(|_| anyhow!("Hex engine mutex poisoned"))?
            .transcribe_offline(samples)?;
        Ok(Decoded {
            text,
            audio_seconds: samples.len() as f32 / SAMPLE_RATE as f32,
            decode_seconds: started.elapsed().as_secs_f32(),
        })
    }
}
