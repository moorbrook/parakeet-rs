//! Deterministic, inspectable Core ML runtime-plan selection.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::coreml_worker::{
    CoreMlComputeUnits, DEFAULT_LONG_REGIME_SECONDS, MAX_LONG_REGIME_SECONDS,
};
use crate::endpointing::{SAMPLE_RATE, WINDOW_SAMPLES};
use crate::model_fetch::EncoderBucket;
use crate::performance;
use crate::windows::{HoldWindowConfig, WindowPlanner};

pub const PROFILE_SCHEMA_VERSION: u32 = 1;
pub const TUNER_VERSION: u32 = 1;
pub const BACKEND_ID: &str = "fluid-audio-parakeet-unified-en-0.6b-int8";
pub const BASELINE_COMPUTE_UNITS: CoreMlComputeUnits = CoreMlComputeUnits::CpuAndNeuralEngine;

const SESSION_UTTERANCES_PER_REGIME: f64 = 20.0;
const MINIMUM_SCORE_IMPROVEMENT_PERCENT: f64 = 5.0;
const MAXIMUM_MEMORY_RATIO: f64 = 1.25;
const QUALITY_EPSILON: f64 = 1.0e-9;
static SAVE_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceLevel {
    pub index: u32,
    pub name: String,
    pub logical_cpus: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareFingerprint {
    pub chip: String,
    pub architecture: String,
    pub memory_bytes: u64,
    pub logical_cpus: usize,
    pub performance_levels: Vec<PerformanceLevel>,
    pub macos_version: String,
}

impl HardwareFingerprint {
    pub fn current() -> Result<Self> {
        let level_count = performance::sysctl_i32("hw.nperflevels")
            .unwrap_or(0)
            .max(0) as u32;
        let performance_levels = (0..level_count)
            .filter_map(|index| {
                let name = performance::sysctl_string(&format!("hw.perflevel{index}.name")).ok()?;
                let logical_cpus =
                    performance::sysctl_i32(&format!("hw.perflevel{index}.logicalcpu")).ok()?;
                (logical_cpus > 0).then_some(PerformanceLevel {
                    index,
                    name,
                    logical_cpus: logical_cpus as u32,
                })
            })
            .collect();
        Ok(Self {
            chip: performance::sysctl_string("machdep.cpu.brand_string")
                .context("reading Apple chip identity")?,
            architecture: std::env::consts::ARCH.to_string(),
            memory_bytes: performance::sysctl_u64("hw.memsize")
                .context("reading physical memory")?,
            logical_cpus: std::thread::available_parallelism()
                .context("reading logical CPU count")?
                .get(),
            performance_levels,
            macos_version: performance::sysctl_string("kern.osproductversion")
                .context("reading macOS product version")?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CategoryQuality {
    pub wer_percent: Option<f64>,
    pub cer_percent: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityMeasurement {
    pub passed: bool,
    pub wer_percent: Option<f64>,
    pub cer_percent: Option<f64>,
    pub nondeterministic_outputs: usize,
    pub categories: BTreeMap<String, CategoryQuality>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeMeasurement {
    pub fixture_count: usize,
    pub audio_seconds: f64,
    pub wall_p50_ms: f64,
    pub wall_p95_ms: f64,
    pub rtfx_p50: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum CandidateStatus {
    Completed,
    Failed { error: String },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateMeasurement {
    pub compute_units: CoreMlComputeUnits,
    pub status: CandidateStatus,
    pub first_observed_load_ms: Option<f64>,
    pub load_p50_ms: Option<f64>,
    pub warmup_ms: Option<f64>,
    pub peak_resident_bytes: Option<u64>,
    pub quality: Option<QualityMeasurement>,
    pub short: Option<RegimeMeasurement>,
    pub long: Option<RegimeMeasurement>,
}

impl CandidateMeasurement {
    pub fn failed(compute_units: CoreMlComputeUnits, error: impl Into<String>) -> Self {
        Self {
            compute_units,
            status: CandidateStatus::Failed {
                error: error.into(),
            },
            first_observed_load_ms: None,
            load_p50_ms: None,
            warmup_ms: None,
            peak_resident_bytes: None,
            quality: None,
            short: None,
            long: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    Short,
    Long,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeSelection {
    pub compute_units: CoreMlComputeUnits,
    pub session_score_ms: f64,
    pub baseline_score_ms: f64,
    pub score_improvement_percent: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub short: RegimeSelection,
    pub long: RegimeSelection,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TuningProfile {
    pub schema_version: u32,
    pub tuner_version: u32,
    pub cache_key: String,
    pub hardware: HardwareFingerprint,
    pub backend_id: String,
    pub artifact_digest: String,
    pub created_at_unix_seconds: u64,
    pub repetitions: usize,
    pub long_regime_seconds: u32,
    pub baseline_compute_units: CoreMlComputeUnits,
    pub selection: Selection,
    pub candidates: Vec<CandidateMeasurement>,
    /// Encoder buckets and Hold window, chosen after the compute units.
    /// Absent from a profile written before kata 4s2q, which then keeps the
    /// untuned behaviour: every hosted bucket and the 6 s window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_tuning: Option<WindowTuning>,
}

impl TuningProfile {
    pub fn new(
        hardware: HardwareFingerprint,
        artifact_digest: String,
        repetitions: usize,
        candidates: Vec<CandidateMeasurement>,
    ) -> Result<Self> {
        if repetitions == 0 {
            bail!("tuning profile needs at least one measured repetition");
        }
        validate_candidate_inventory(&candidates)?;
        let cache_key = profile_cache_key(&hardware, &artifact_digest, None)?;
        let selection = select(&candidates)?;
        Ok(Self {
            schema_version: PROFILE_SCHEMA_VERSION,
            tuner_version: TUNER_VERSION,
            cache_key,
            hardware,
            backend_id: BACKEND_ID.to_string(),
            artifact_digest,
            created_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            repetitions,
            long_regime_seconds: DEFAULT_LONG_REGIME_SECONDS,
            baseline_compute_units: BASELINE_COMPUTE_UNITS,
            selection,
            candidates,
            window_tuning: None,
        })
    }

    /// Record the bucket and Hold-window choice, which joins the cache key.
    pub fn set_window_tuning(&mut self, tuning: WindowTuning) -> Result<()> {
        if tuning.selection != select_window(&tuning.evidence)? {
            bail!("window selection does not match its evidence");
        }
        self.cache_key = profile_cache_key(&self.hardware, &self.artifact_digest, Some(&tuning))?;
        self.window_tuning = Some(tuning);
        Ok(())
    }

    /// The bucket set and Hold window this profile chose, when it chose them
    /// and every chosen bucket is still the one `manifest` hosts. Call only on
    /// a profile [`Self::validate_for`] accepted. `Ok(None)` means the profile
    /// predates window tuning; an error means its window evidence is stale or
    /// inconsistent, and the caller keeps the untuned behaviour for windows
    /// while still using the compute units.
    pub fn window_selection_for(
        &self,
        manifest: &[EncoderBucket],
    ) -> Result<Option<&WindowSelection>> {
        let Some(tuning) = &self.window_tuning else {
            return Ok(None);
        };
        if tuning.selection != select_window(&tuning.evidence)? {
            bail!("window selection does not match its evidence");
        }
        for window in &tuning.selection.encoder_buckets {
            let measured = tuning
                .evidence
                .encoders
                .iter()
                .find(|cost| cost.window_seconds == *window)
                .with_context(|| format!("selected {window} s bucket has no measurement"))?;
            let hosted = manifest
                .iter()
                .find(|bucket| bucket.seconds == *window)
                .with_context(|| format!("selected {window} s bucket is no longer hosted"))?;
            if measured.archive_sha256.as_deref() != Some(hosted.archive_sha256())
                || measured.installed_bytes != hosted.installed_bytes()
            {
                bail!("the {window} s bucket was re-released after tuning");
            }
        }
        Ok(Some(&tuning.selection))
    }

    pub fn validate_for(
        &self,
        hardware: &HardwareFingerprint,
        artifact_digest: &str,
    ) -> Result<()> {
        if self.schema_version != PROFILE_SCHEMA_VERSION {
            bail!(
                "unsupported tuning profile schema {}; expected {PROFILE_SCHEMA_VERSION}",
                self.schema_version
            );
        }
        if self.tuner_version != TUNER_VERSION {
            bail!("tuning profile was produced by a different tuner version");
        }
        if self.backend_id != BACKEND_ID || self.artifact_digest != artifact_digest {
            bail!("tuning profile backend or artifact identity is stale");
        }
        if &self.hardware != hardware {
            bail!("tuning profile hardware or macOS fingerprint is stale");
        }
        if self.cache_key
            != profile_cache_key(hardware, artifact_digest, self.window_tuning.as_ref())?
        {
            bail!("tuning profile cache key does not match its contents");
        }
        if !(1..=MAX_LONG_REGIME_SECONDS).contains(&self.long_regime_seconds) {
            bail!("tuning profile has an invalid long-regime threshold");
        }
        if self.repetitions == 0 {
            bail!("tuning profile has no measured repetitions");
        }
        if self.baseline_compute_units != BASELINE_COMPUTE_UNITS {
            bail!("tuning profile changed the safe baseline");
        }
        validate_candidate_inventory(&self.candidates)?;
        if self.selection != select(&self.candidates)? {
            bail!("tuning profile selection does not match its evidence");
        }
        Ok(())
    }
}

pub fn cache_key(hardware: &HardwareFingerprint, artifact_digest: &str) -> Result<String> {
    profile_cache_key(hardware, artifact_digest, None)
}

/// The key adds the window choice only when the profile makes one, and then
/// only the chosen buckets' archive digests and the chosen Hold window. A
/// profile without window tuning keeps the key it was written with, and one
/// that chose 5 s alone does not go stale when the 8 s bucket is re-released.
fn profile_cache_key(
    hardware: &HardwareFingerprint,
    artifact_digest: &str,
    window_tuning: Option<&WindowTuning>,
) -> Result<String> {
    #[derive(Serialize)]
    struct WindowKey<'a> {
        encoder_buckets: Vec<(u32, Option<&'a str>)>,
        hold_window_seconds: f32,
    }
    #[derive(Serialize)]
    struct Key<'a> {
        hardware: &'a HardwareFingerprint,
        backend_id: &'static str,
        artifact_digest: &'a str,
        tuner_version: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        window: Option<WindowKey<'a>>,
    }
    let window = window_tuning.map(|tuning| WindowKey {
        encoder_buckets: tuning
            .selection
            .encoder_buckets
            .iter()
            .map(|window| {
                let digest = tuning
                    .evidence
                    .encoders
                    .iter()
                    .find(|cost| cost.window_seconds == *window)
                    .and_then(|cost| cost.archive_sha256.as_deref());
                (*window, digest)
            })
            .collect(),
        hold_window_seconds: tuning.selection.hold_window_seconds,
    });
    let bytes = serde_json::to_vec(&Key {
        hardware,
        backend_id: BACKEND_ID,
        artifact_digest,
        tuner_version: TUNER_VERSION,
        window,
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn load(path: &Path) -> Result<Option<TuningProfile>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing tuning profile {}", path.display()))
        .map(Some)
}

pub fn save(path: &Path, profile: &TuningProfile) -> Result<()> {
    let parent = path
        .parent()
        .context("tuning profile path must have a parent directory")?;
    fs::create_dir_all(parent)?;
    let nonce = SAVE_NONCE.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(
        ".asr-tuning-profile.{}.{}.tmp",
        std::process::id(),
        nonce
    ));
    let result = (|| -> Result<()> {
        let mut output = File::create(&temp)?;
        output.write_all(&serde_json::to_vec_pretty(profile)?)?;
        output.write_all(b"\n")?;
        output.sync_all()?;
        fs::rename(&temp, path)?;
        if let Ok(directory) = File::open(parent) {
            directory.sync_all()?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.with_context(|| format!("saving tuning profile {}", path.display()))
}

pub fn remove(path: &Path) -> Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("removing {}", path.display())),
    }
}

pub fn select(candidates: &[CandidateMeasurement]) -> Result<Selection> {
    for candidate in candidates {
        validate_candidate_evidence(candidate)?;
    }
    let baseline = candidates
        .iter()
        .find(|candidate| candidate.compute_units == BASELINE_COMPUTE_UNITS)
        .context("safe baseline candidate is missing")?;
    require_complete_and_quality_safe(baseline)?;
    Ok(Selection {
        short: select_regime(candidates, baseline, Regime::Short)?,
        long: select_regime(candidates, baseline, Regime::Long)?,
    })
}

fn validate_candidate_inventory(candidates: &[CandidateMeasurement]) -> Result<()> {
    if candidates.len() != CoreMlComputeUnits::CANDIDATES.len() {
        bail!("tuning profile does not contain the complete bounded candidate set");
    }
    for expected in CoreMlComputeUnits::CANDIDATES {
        if candidates
            .iter()
            .filter(|candidate| candidate.compute_units == expected)
            .count()
            != 1
        {
            bail!(
                "tuning profile must contain exactly one {} candidate",
                expected.as_str()
            );
        }
    }
    for candidate in candidates {
        validate_candidate_evidence(candidate)?;
    }
    Ok(())
}

fn validate_candidate_evidence(candidate: &CandidateMeasurement) -> Result<()> {
    match &candidate.status {
        CandidateStatus::Failed { error } => {
            if error.trim().is_empty() {
                bail!("failed candidate omitted its error");
            }
            if candidate.first_observed_load_ms.is_some()
                || candidate.load_p50_ms.is_some()
                || candidate.warmup_ms.is_some()
                || candidate.peak_resident_bytes.is_some()
                || candidate.quality.is_some()
                || candidate.short.is_some()
                || candidate.long.is_some()
            {
                bail!("failed candidate contains partial measurement evidence");
            }
            return Ok(());
        }
        CandidateStatus::Completed => {}
    }

    for (label, value) in [
        ("first load", candidate.first_observed_load_ms),
        ("median load", candidate.load_p50_ms),
        ("warmup", candidate.warmup_ms),
    ] {
        let value = value.with_context(|| format!("completed candidate omitted {label}"))?;
        if !value.is_finite() || value < 0.0 {
            bail!("completed candidate has invalid {label}");
        }
    }
    if candidate.peak_resident_bytes.unwrap_or(0) == 0 {
        bail!("completed candidate has invalid peak resident memory");
    }
    let quality = candidate
        .quality
        .as_ref()
        .context("completed candidate omitted quality")?;
    validate_optional_percentage("WER", quality.wer_percent)?;
    validate_optional_percentage("CER", quality.cer_percent)?;
    if quality.categories.is_empty() {
        bail!("completed candidate omitted per-category quality");
    }
    for (category, score) in &quality.categories {
        if category.trim().is_empty() {
            bail!("completed candidate contains an empty quality category");
        }
        validate_optional_percentage("category WER", score.wer_percent)?;
        validate_optional_percentage("category CER", score.cer_percent)?;
    }
    validate_regime("short", candidate.short.as_ref())?;
    validate_regime("long", candidate.long.as_ref())?;
    Ok(())
}

fn validate_optional_percentage(label: &str, value: Option<f64>) -> Result<()> {
    let value = value.with_context(|| format!("completed candidate omitted {label}"))?;
    if !value.is_finite() || value < 0.0 {
        bail!("completed candidate has invalid {label}");
    }
    Ok(())
}

fn validate_regime(label: &str, measurement: Option<&RegimeMeasurement>) -> Result<()> {
    let measurement =
        measurement.with_context(|| format!("completed candidate omitted {label}"))?;
    if measurement.fixture_count == 0
        || !measurement.audio_seconds.is_finite()
        || measurement.audio_seconds <= 0.0
        || !measurement.wall_p50_ms.is_finite()
        || measurement.wall_p50_ms <= 0.0
        || !measurement.wall_p95_ms.is_finite()
        || measurement.wall_p95_ms < measurement.wall_p50_ms
        || !measurement.rtfx_p50.is_finite()
        || measurement.rtfx_p50 <= 0.0
    {
        bail!("completed candidate has invalid {label} evidence");
    }
    Ok(())
}

fn select_regime(
    candidates: &[CandidateMeasurement],
    baseline: &CandidateMeasurement,
    regime: Regime,
) -> Result<RegimeSelection> {
    let baseline_score = session_score(baseline, regime)?;
    let baseline_memory = baseline
        .peak_resident_bytes
        .context("safe baseline omitted peak resident memory")? as f64;
    let baseline_quality = baseline
        .quality
        .as_ref()
        .context("safe baseline omitted quality")?;

    let mut best = baseline;
    let mut best_score = baseline_score;
    for candidate_kind in CoreMlComputeUnits::CANDIDATES {
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.compute_units == candidate_kind)
        else {
            continue;
        };
        if require_complete_and_quality_safe(candidate).is_err()
            || !category_quality_no_worse(
                candidate.quality.as_ref().expect("checked above"),
                baseline_quality,
            )
            || candidate.peak_resident_bytes.expect("checked above") as f64
                > baseline_memory * MAXIMUM_MEMORY_RATIO
        {
            continue;
        }
        let score = session_score(candidate, regime)?;
        if score < best_score {
            best = candidate;
            best_score = score;
        }
    }

    let raw_improvement = percent_improvement(baseline_score, best_score);
    if best.compute_units != BASELINE_COMPUTE_UNITS
        && raw_improvement + QUALITY_EPSILON < MINIMUM_SCORE_IMPROVEMENT_PERCENT
    {
        best = baseline;
        best_score = baseline_score;
    }
    Ok(RegimeSelection {
        compute_units: best.compute_units,
        session_score_ms: best_score,
        baseline_score_ms: baseline_score,
        score_improvement_percent: percent_improvement(baseline_score, best_score),
    })
}

fn require_complete_and_quality_safe(candidate: &CandidateMeasurement) -> Result<()> {
    if !matches!(candidate.status, CandidateStatus::Completed) {
        bail!("candidate did not complete");
    }
    let quality = candidate
        .quality
        .as_ref()
        .context("candidate omitted quality")?;
    if !quality.passed || quality.nondeterministic_outputs != 0 {
        bail!("candidate failed quality or repeatability");
    }
    if candidate.load_p50_ms.is_none()
        || candidate.peak_resident_bytes.is_none()
        || candidate.short.is_none()
        || candidate.long.is_none()
    {
        bail!("candidate omitted required performance evidence");
    }
    Ok(())
}

fn category_quality_no_worse(
    candidate: &QualityMeasurement,
    baseline: &QualityMeasurement,
) -> bool {
    baseline
        .categories
        .iter()
        .all(|(category, baseline_score)| {
            candidate.categories.get(category).is_some_and(|score| {
                option_no_worse(score.wer_percent, baseline_score.wer_percent)
                    && option_no_worse(score.cer_percent, baseline_score.cer_percent)
            })
        })
}

fn option_no_worse(candidate: Option<f64>, baseline: Option<f64>) -> bool {
    match (candidate, baseline) {
        (Some(candidate), Some(baseline)) => candidate <= baseline + QUALITY_EPSILON,
        (None, None) => true,
        _ => false,
    }
}

fn session_score(candidate: &CandidateMeasurement, regime: Regime) -> Result<f64> {
    let load_ms = candidate
        .load_p50_ms
        .context("candidate omitted median model load")?;
    let warmup_ms = candidate
        .warmup_ms
        .context("candidate omitted first-decode warmup")?;
    let measurement = match regime {
        Regime::Short => candidate.short.as_ref(),
        Regime::Long => candidate.long.as_ref(),
    }
    .context("candidate omitted regime measurement")?;
    let per_utterance_ms = measurement.wall_p50_ms / measurement.fixture_count as f64;
    Ok(load_ms + warmup_ms + SESSION_UTTERANCES_PER_REGIME * per_utterance_ms)
}

fn percent_improvement(baseline: f64, candidate: f64) -> f64 {
    if baseline > 0.0 {
        100.0 * (baseline - candidate) / baseline
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Encoder buckets and Hold window (kata 4s2q)
//
// The tuner measures each encoder's cost on this chip, predicts the release
// cost of every bucket subset and Hold window from it, and then runs the gold
// gate on those configurations fastest first. The first whose output matches
// the 15 s encoder at the 6 s window byte for byte is chosen; the baseline
// itself is the fallback and needs no gate.

/// The Hold window the app uses when no profile chooses one (ADR-0032).
pub const BASELINE_HOLD_WINDOW_SECONDS: f32 = 6.0;

/// Hold windows the tuner may choose. Each runs with pause cutting off
/// (minimum equal to maximum), the measured setting of ADR-0032.
pub const HOLD_WINDOW_CANDIDATES: [f32; 5] = [3.0, 4.0, 5.0, 6.0, 8.0];

/// Utterance lengths the predicted release cost averages over, equally
/// weighted: what bench-hold captures of its 1, 3, 5, 10 and 20 s fixtures
/// (kata ktfa), so a prediction can be checked against that bench. Fixed in
/// the tuner, never learned from what the user dictates.
pub const RELEASE_WORKLOAD_SECONDS: [f32; 5] = [0.875, 2.891, 4.917, 8.128, 16.58];

/// Samples per encoder frame (80 ms). An encoder window decodes only whole
/// frames, so the 5 s bucket holds 79,360 samples although it is routed
/// anything up to 80,000, and runs twice for the gap between (kata qgwm).
pub const ENCODER_FRAME_SAMPLES: u64 = 1_280;

/// Window of the stock encoder, which takes every audio no bucket holds.
pub const STOCK_ENCODER_WINDOW_SECONDS: u32 = 15;

/// One encoder's measured cost. The encoder pads every input to its window,
/// so this cost does not depend on how much audio it was given.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderCost {
    pub window_seconds: u32,
    /// Median of the worker's encoder stage over the probe repetitions.
    pub encoder_ms_p50: f64,
    /// The release archive this bucket was measured from. `None` for the
    /// stock encoder, which the artifact digest already covers.
    pub archive_sha256: Option<String>,
    /// Disk the bucket occupies once installed; zero for the stock encoder.
    pub installed_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum GateOutcome {
    /// Every repetition produced the same outputs; this is their digest.
    Output {
        sha256: String,
    },
    /// Repetitions disagreed, so no single output can match the baseline.
    Nondeterministic,
    Failed {
        error: String,
    },
}

/// The gold gate's result for one bucket subset at one Hold window.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GateVerdict {
    pub encoder_buckets: Vec<u32>,
    pub hold_window_seconds: f32,
    pub outcome: GateOutcome,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindowEvidence {
    pub disk_budget_bytes: u64,
    pub hold_window_candidates: Vec<f32>,
    pub workload_seconds: Vec<f32>,
    /// Every encoder measured: the stock one and each installed bucket.
    pub encoders: Vec<EncoderCost>,
    /// Release cost that grows with the tail's audio (mel and decode), per
    /// second, from the stock encoder's probes.
    pub decode_ms_per_audio_second: f64,
    /// Digest of the gold outputs with no bucket at the baseline Hold window,
    /// both whole-utterance and Hold-windowed. A configuration is eligible
    /// only when its digest is this one.
    pub baseline_output_sha256: String,
    /// Verdicts in the order they ran: fastest predicted first, stopping at
    /// the first match.
    pub gates: Vec<GateVerdict>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindowSelection {
    pub encoder_buckets: Vec<u32>,
    pub hold_window_seconds: f32,
    pub installed_bytes: u64,
    pub predicted_release_ms: f64,
    pub baseline_predicted_release_ms: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindowTuning {
    pub evidence: WindowEvidence,
    pub selection: WindowSelection,
}

impl WindowTuning {
    pub fn new(evidence: WindowEvidence) -> Result<Self> {
        let selection = select_window(&evidence)?;
        Ok(Self {
            evidence,
            selection,
        })
    }
}

/// The encoder buckets the app keeps and the Hold window it cuts at.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowPlan {
    pub encoder_buckets: Vec<u32>,
    pub hold_window_seconds: f32,
}

impl WindowPlan {
    /// No profile, or one that predates window tuning: every hosted bucket and
    /// the 6 s window, the app's behaviour before kata 4s2q.
    pub fn untuned(manifest: &[EncoderBucket]) -> Self {
        Self {
            encoder_buckets: manifest.iter().map(|bucket| bucket.seconds).collect(),
            hold_window_seconds: BASELINE_HOLD_WINDOW_SECONDS,
        }
    }

    /// The plan of a profile [`TuningProfile::validate_for`] accepted. Stale
    /// window evidence falls back to [`Self::untuned`] and says why.
    pub fn for_profile(profile: &TuningProfile, manifest: &[EncoderBucket]) -> Self {
        match profile.window_selection_for(manifest) {
            Ok(Some(selection)) => Self {
                encoder_buckets: selection.encoder_buckets.clone(),
                hold_window_seconds: selection.hold_window_seconds,
            },
            Ok(None) => Self::untuned(manifest),
            Err(error) => {
                log::warn!(
                    "ASR tuning profile's bucket and Hold-window choice is stale; using every \
                     bucket and the {BASELINE_HOLD_WINDOW_SECONDS} s window: {error:#}"
                );
                Self::untuned(manifest)
            }
        }
    }

    pub fn hold_windows(&self, enabled: bool) -> HoldWindowConfig {
        HoldWindowConfig {
            enabled,
            min_seconds: self.hold_window_seconds,
            max_seconds: self.hold_window_seconds,
        }
    }
}

/// A bucket subset at a Hold window, with its predicted release cost.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowCandidate {
    pub encoder_buckets: Vec<u32>,
    pub hold_window_seconds: f32,
    pub installed_bytes: u64,
    pub predicted_release_ms: f64,
}

impl WindowCandidate {
    pub fn is_baseline(&self) -> bool {
        self.encoder_buckets.is_empty() && self.hold_window_seconds == BASELINE_HOLD_WINDOW_SECONDS
    }
}

/// Samples left to decode on release once Hold has cut every window it can,
/// for continuous speech of `utterance_samples` at `hold_window_seconds`. Runs
/// the session's own planner on all-speech frames, so a forced cut lands
/// where `streamer::decode_hold_windowed` would put it.
pub fn tail_samples(utterance_samples: u64, hold_window_seconds: f32) -> u64 {
    let config = HoldWindowConfig {
        enabled: true,
        min_seconds: hold_window_seconds,
        max_seconds: hold_window_seconds,
    };
    let mut planner = WindowPlanner::new(config, u64::from(WINDOW_SAMPLES));
    for _ in 0..utterance_samples / u64::from(WINDOW_SAMPLES) {
        let _ = planner.observe_frame(true);
    }
    utterance_samples - planner.open_window_start().min(utterance_samples)
}

/// The encoder window that takes `samples`: the narrowest bucket that holds
/// them, else the stock encoder. The worker's `EncoderBuckets.select` rule.
pub fn routed_window(samples: u64, buckets: &[u32]) -> u32 {
    let mut sorted = buckets.to_vec();
    sorted.sort_unstable();
    sorted
        .into_iter()
        .find(|window| samples <= u64::from(*window) * u64::from(SAMPLE_RATE))
        .unwrap_or(STOCK_ENCODER_WINDOW_SECONDS)
}

/// Encoder runs `samples` costs on the `window` s encoder: one per whole-frame
/// capacity it fills.
pub fn encoder_passes(samples: u64, window: u32) -> u64 {
    let capacity =
        u64::from(window) * u64::from(SAMPLE_RATE) / ENCODER_FRAME_SAMPLES * ENCODER_FRAME_SAMPLES;
    samples.div_ceil(capacity).max(1)
}

/// Mean release-time ASR cost over [`RELEASE_WORKLOAD_SECONDS`]: the routed
/// encoder's measured cost plus the per-second cost of the tail's audio. The
/// fixed per-request overhead is the same for every configuration and left
/// out.
pub fn predicted_release_ms(
    evidence: &WindowEvidence,
    buckets: &[u32],
    hold_window_seconds: f32,
) -> Result<f64> {
    let mut total = 0.0;
    for seconds in &evidence.workload_seconds {
        let samples = (f64::from(*seconds) * f64::from(SAMPLE_RATE)).round() as u64;
        let tail = tail_samples(samples, hold_window_seconds);
        let window = routed_window(tail, buckets);
        let encoder = evidence
            .encoders
            .iter()
            .find(|cost| cost.window_seconds == window)
            .with_context(|| format!("no measured cost for the {window} s encoder"))?;
        total += encoder_passes(tail, window) as f64 * encoder.encoder_ms_p50
            + evidence.decode_ms_per_audio_second * tail as f64 / f64::from(SAMPLE_RATE);
    }
    Ok(total / evidence.workload_seconds.len() as f64)
}

/// Every configuration the gate may try, in the order it tries them, ending
/// with the baseline: measured buckets only, within the disk budget, and at
/// least [`MINIMUM_SCORE_IMPROVEMENT_PERCENT`] faster than the baseline, so
/// noise cannot move the app off it. Ties go to fewer bytes, then the window
/// nearer the baseline.
pub fn rank_window_candidates(evidence: &WindowEvidence) -> Result<Vec<WindowCandidate>> {
    validate_window_evidence(evidence)?;
    let mut buckets: Vec<&EncoderCost> = evidence
        .encoders
        .iter()
        .filter(|cost| cost.window_seconds != STOCK_ENCODER_WINDOW_SECONDS)
        .collect();
    buckets.sort_by_key(|cost| cost.window_seconds);
    let baseline_ms = predicted_release_ms(evidence, &[], BASELINE_HOLD_WINDOW_SECONDS)?;

    let mut ranked = Vec::new();
    for mask in 0..1_u32 << buckets.len() {
        let chosen: Vec<&EncoderCost> = buckets
            .iter()
            .enumerate()
            .filter(|(index, _)| mask & (1 << index) != 0)
            .map(|(_, cost)| *cost)
            .collect();
        let installed_bytes: u64 = chosen.iter().map(|cost| cost.installed_bytes).sum();
        if installed_bytes > evidence.disk_budget_bytes {
            continue;
        }
        let windows: Vec<u32> = chosen.iter().map(|cost| cost.window_seconds).collect();
        for hold_window_seconds in &evidence.hold_window_candidates {
            let candidate = WindowCandidate {
                predicted_release_ms: predicted_release_ms(
                    evidence,
                    &windows,
                    *hold_window_seconds,
                )?,
                encoder_buckets: windows.clone(),
                hold_window_seconds: *hold_window_seconds,
                installed_bytes,
            };
            let improvement = percent_improvement(baseline_ms, candidate.predicted_release_ms);
            if !candidate.is_baseline()
                && improvement + QUALITY_EPSILON >= MINIMUM_SCORE_IMPROVEMENT_PERCENT
            {
                ranked.push(candidate);
            }
        }
    }
    ranked.sort_by(|a, b| {
        a.predicted_release_ms
            .total_cmp(&b.predicted_release_ms)
            .then(a.installed_bytes.cmp(&b.installed_bytes))
            .then(
                (a.hold_window_seconds - BASELINE_HOLD_WINDOW_SECONDS)
                    .abs()
                    .total_cmp(&(b.hold_window_seconds - BASELINE_HOLD_WINDOW_SECONDS).abs()),
            )
            .then(a.hold_window_seconds.total_cmp(&b.hold_window_seconds))
            .then(a.encoder_buckets.cmp(&b.encoder_buckets))
    });
    ranked.push(WindowCandidate {
        encoder_buckets: Vec::new(),
        hold_window_seconds: BASELINE_HOLD_WINDOW_SECONDS,
        installed_bytes: 0,
        predicted_release_ms: baseline_ms,
    });
    Ok(ranked)
}

/// The verdict recorded for `candidate`, if the gate ran it.
pub fn gate_verdict<'a>(
    evidence: &'a WindowEvidence,
    candidate: &WindowCandidate,
) -> Option<&'a GateVerdict> {
    evidence.gates.iter().find(|verdict| {
        verdict.encoder_buckets == candidate.encoder_buckets
            && verdict.hold_window_seconds == candidate.hold_window_seconds
    })
}

/// The fastest ranked configuration whose gold output is the baseline's,
/// byte for byte. Every configuration ranked ahead of the choice must carry a
/// verdict, so a profile cannot skip a faster configuration it never gated.
pub fn select_window(evidence: &WindowEvidence) -> Result<WindowSelection> {
    let ranked = rank_window_candidates(evidence)?;
    let baseline_ms = ranked
        .last()
        .context("ranking always ends with the baseline")?
        .predicted_release_ms;
    for candidate in ranked {
        let passes = candidate.is_baseline()
            || match gate_verdict(evidence, &candidate) {
                Some(verdict) => matches!(
                    &verdict.outcome,
                    GateOutcome::Output { sha256 } if *sha256 == evidence.baseline_output_sha256
                ),
                None => bail!(
                    "no gold verdict for buckets {:?} at a {} s Hold window, which ranks \
                     ahead of any passing configuration",
                    candidate.encoder_buckets,
                    candidate.hold_window_seconds
                ),
            };
        if passes {
            return Ok(WindowSelection {
                encoder_buckets: candidate.encoder_buckets,
                hold_window_seconds: candidate.hold_window_seconds,
                installed_bytes: candidate.installed_bytes,
                predicted_release_ms: candidate.predicted_release_ms,
                baseline_predicted_release_ms: baseline_ms,
            });
        }
    }
    unreachable!("the ranking ends with the baseline, which always passes")
}

fn validate_window_evidence(evidence: &WindowEvidence) -> Result<()> {
    if evidence.hold_window_candidates != HOLD_WINDOW_CANDIDATES
        || evidence.workload_seconds != RELEASE_WORKLOAD_SECONDS
    {
        bail!("window evidence was gathered over a different candidate set or workload");
    }
    for seconds in &evidence.hold_window_candidates {
        HoldWindowConfig {
            enabled: true,
            min_seconds: *seconds,
            max_seconds: *seconds,
        }
        .validate()
        .map_err(|reason| anyhow::anyhow!(reason))?;
    }
    if !evidence
        .encoders
        .iter()
        .any(|cost| cost.window_seconds == STOCK_ENCODER_WINDOW_SECONDS)
    {
        bail!("window evidence has no stock-encoder measurement");
    }
    for (index, cost) in evidence.encoders.iter().enumerate() {
        if evidence.encoders[..index]
            .iter()
            .any(|other| other.window_seconds == cost.window_seconds)
        {
            bail!("the {} s encoder is measured twice", cost.window_seconds);
        }
        if !cost.encoder_ms_p50.is_finite() || cost.encoder_ms_p50 < 0.0 {
            bail!("the {} s encoder has an invalid cost", cost.window_seconds);
        }
        let stock = cost.window_seconds == STOCK_ENCODER_WINDOW_SECONDS;
        let identified = cost
            .archive_sha256
            .as_deref()
            .is_some_and(|digest| digest.len() == 64);
        if stock != (cost.archive_sha256.is_none() && cost.installed_bytes == 0)
            || (!stock && (!identified || cost.installed_bytes == 0))
            || cost.window_seconds == 0
            || cost.window_seconds > STOCK_ENCODER_WINDOW_SECONDS
        {
            bail!(
                "the {} s encoder has an invalid identity or size",
                cost.window_seconds
            );
        }
    }
    if !evidence.decode_ms_per_audio_second.is_finite() || evidence.decode_ms_per_audio_second < 0.0
    {
        bail!("window evidence has an invalid per-second decode cost");
    }
    if evidence.baseline_output_sha256.len() != 64 {
        bail!("window evidence has no baseline output digest");
    }
    for (index, verdict) in evidence.gates.iter().enumerate() {
        if evidence.gates[..index].iter().any(|other| {
            other.encoder_buckets == verdict.encoder_buckets
                && other.hold_window_seconds == verdict.hold_window_seconds
        }) {
            bail!("a configuration was gated twice");
        }
        if let GateOutcome::Failed { error } = &verdict.outcome {
            if error.trim().is_empty() {
                bail!("a failed gate omitted its error");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn hardware() -> HardwareFingerprint {
        HardwareFingerprint {
            chip: "Apple Test".into(),
            architecture: "aarch64".into(),
            memory_bytes: 24 * 1024 * 1024 * 1024,
            logical_cpus: 12,
            performance_levels: vec![PerformanceLevel {
                index: 0,
                name: "Performance".into(),
                logical_cpus: 8,
            }],
            macos_version: "26.5".into(),
        }
    }

    fn candidate(
        compute_units: CoreMlComputeUnits,
        load_ms: f64,
        short_ms: f64,
        long_ms: f64,
    ) -> CandidateMeasurement {
        let categories = BTreeMap::from([(
            "general".into(),
            CategoryQuality {
                wer_percent: Some(1.0),
                cer_percent: Some(0.5),
            },
        )]);
        CandidateMeasurement {
            compute_units,
            status: CandidateStatus::Completed,
            first_observed_load_ms: Some(load_ms),
            load_p50_ms: Some(load_ms),
            warmup_ms: Some(20.0),
            peak_resident_bytes: Some(100),
            quality: Some(QualityMeasurement {
                passed: true,
                wer_percent: Some(1.0),
                cer_percent: Some(0.5),
                nondeterministic_outputs: 0,
                categories,
            }),
            short: Some(RegimeMeasurement {
                fixture_count: 1,
                audio_seconds: 3.0,
                wall_p50_ms: short_ms,
                wall_p95_ms: short_ms,
                rtfx_p50: 3_000.0 / short_ms,
            }),
            long: Some(RegimeMeasurement {
                fixture_count: 1,
                audio_seconds: 12.0,
                wall_p50_ms: long_ms,
                wall_p95_ms: long_ms,
                rtfx_p50: 12_000.0 / long_ms,
            }),
        }
    }

    #[test]
    fn deterministic_selection_is_bucketed_and_keeps_near_ties_safe() {
        let baseline = candidate(BASELINE_COMPUTE_UNITS, 100.0, 50.0, 100.0);
        let near_tie = candidate(CoreMlComputeUnits::All, 100.0, 49.0, 99.0);
        let bucketed = candidate(CoreMlComputeUnits::CpuOnly, 100.0, 40.0, 120.0);
        let selection = select(&[baseline, near_tie, bucketed]).unwrap();
        assert_eq!(selection.short.compute_units, CoreMlComputeUnits::CpuOnly);
        assert_eq!(
            selection.long.compute_units,
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
    }

    #[test]
    fn quality_or_memory_regression_cannot_win_on_latency() {
        let baseline = candidate(BASELINE_COMPUTE_UNITS, 100.0, 50.0, 100.0);
        let mut bad_quality = candidate(CoreMlComputeUnits::All, 50.0, 1.0, 1.0);
        bad_quality
            .quality
            .as_mut()
            .unwrap()
            .categories
            .get_mut("general")
            .unwrap()
            .wer_percent = Some(2.0);
        let mut bad_memory = candidate(CoreMlComputeUnits::CpuOnly, 50.0, 1.0, 1.0);
        bad_memory.peak_resident_bytes = Some(126);
        let selection = select(&[baseline, bad_quality, bad_memory]).unwrap();
        assert_eq!(
            selection.short.compute_units,
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
        assert_eq!(
            selection.long.compute_units,
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
    }

    #[test]
    fn session_score_normalizes_corpus_latency_to_one_utterance() {
        let mut measured = candidate(BASELINE_COMPUTE_UNITS, 100.0, 300.0, 100.0);
        measured.short.as_mut().unwrap().fixture_count = 6;
        assert_eq!(session_score(&measured, Regime::Short).unwrap(), 1_120.0);
    }

    #[test]
    fn profile_key_invalidates_on_hardware_os_model_or_tuner_inputs() {
        let a = cache_key(&hardware(), "artifact-a").unwrap();
        assert_eq!(a, cache_key(&hardware(), "artifact-a").unwrap());
        assert_ne!(a, cache_key(&hardware(), "artifact-b").unwrap());
        let mut other = hardware();
        other.macos_version = "27.0".into();
        assert_ne!(a, cache_key(&other, "artifact-a").unwrap());
    }

    #[test]
    fn profile_is_atomic_inspectable_removable_and_strict() {
        let profile = TuningProfile::new(
            hardware(),
            "artifact".into(),
            3,
            vec![
                candidate(BASELINE_COMPUTE_UNITS, 100.0, 50.0, 100.0),
                CandidateMeasurement::failed(CoreMlComputeUnits::All, "unsupported"),
                CandidateMeasurement::failed(CoreMlComputeUnits::CpuAndGpu, "unsupported"),
                CandidateMeasurement::failed(CoreMlComputeUnits::CpuOnly, "unsupported"),
            ],
        )
        .unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("profile.json");
        save(&path, &profile).unwrap();
        let loaded = load(&path).unwrap().unwrap();
        assert_eq!(loaded, profile);
        loaded.validate_for(&hardware(), "artifact").unwrap();
        assert!(loaded.validate_for(&hardware(), "other").is_err());
        let mut corrupted = loaded.clone();
        corrupted.candidates[0].load_p50_ms = Some(-1.0);
        assert!(corrupted.validate_for(&hardware(), "artifact").is_err());
        let mut altered_selection = loaded.clone();
        altered_selection.selection.short.compute_units = CoreMlComputeUnits::All;
        assert!(altered_selection
            .validate_for(&hardware(), "artifact")
            .is_err());
        assert!(remove(&path).unwrap());
        assert!(!remove(&path).unwrap());
    }

    const BASELINE_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    const CHANGED_SHA: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn hosted(window: u32) -> &'static EncoderBucket {
        crate::model_fetch::ENCODER_BUCKETS
            .iter()
            .find(|bucket| bucket.seconds == window)
            .unwrap()
    }

    /// Encoder costs in ms: the 15 s encoder 26, buckets 2 s 6, 5 s 10, 8 s 13.
    fn window_evidence(disk_budget_bytes: u64) -> WindowEvidence {
        let bucket = |window: u32, encoder_ms_p50: f64| EncoderCost {
            window_seconds: window,
            encoder_ms_p50,
            archive_sha256: Some(hosted(window).archive_sha256().to_string()),
            installed_bytes: hosted(window).installed_bytes(),
        };
        WindowEvidence {
            disk_budget_bytes,
            hold_window_candidates: HOLD_WINDOW_CANDIDATES.to_vec(),
            workload_seconds: RELEASE_WORKLOAD_SECONDS.to_vec(),
            encoders: vec![
                EncoderCost {
                    window_seconds: 15,
                    encoder_ms_p50: 26.0,
                    archive_sha256: None,
                    installed_bytes: 0,
                },
                bucket(2, 6.0),
                bucket(5, 10.0),
                bucket(8, 13.0),
            ],
            decode_ms_per_audio_second: 2.0,
            baseline_output_sha256: BASELINE_SHA.into(),
            gates: Vec::new(),
        }
    }

    /// Record a verdict for every configuration the ranking puts ahead of the
    /// baseline, as tune_asr would when nothing passes before the end.
    fn gate_everything(
        evidence: &mut WindowEvidence,
        output: impl Fn(&[u32], f32) -> &'static str,
    ) {
        for candidate in rank_window_candidates(evidence).unwrap() {
            if candidate.is_baseline() {
                break;
            }
            evidence.gates.push(GateVerdict {
                outcome: GateOutcome::Output {
                    sha256: output(&candidate.encoder_buckets, candidate.hold_window_seconds)
                        .into(),
                },
                encoder_buckets: candidate.encoder_buckets,
                hold_window_seconds: candidate.hold_window_seconds,
            });
        }
    }

    #[test]
    fn hold_tail_and_routing_follow_the_planner_and_frame_rounding() {
        // 8.128 s at 6 s: one forced cut at frame 188 (96,256 samples), the
        // tail starts 1.5 s before it, at 72,256.
        assert_eq!(tail_samples(130_048, 6.0), 57_792);
        assert_eq!(tail_samples(78_672, 6.0), 78_672);
        assert_eq!(routed_window(80_000, &[8, 2, 5]), 5);
        assert_eq!(routed_window(80_001, &[2, 5, 8]), 8);
        assert_eq!(routed_window(14_000, &[]), 15);
        assert_eq!(encoder_passes(79_360, 5), 1);
        assert_eq!(encoder_passes(79_361, 5), 2);
        assert_eq!(encoder_passes(128_000, 8), 1);
        assert_eq!(
            BASELINE_HOLD_WINDOW_SECONDS,
            HoldWindowConfig::default().max_seconds
        );
    }

    #[test]
    fn a_budget_that_fits_one_bucket_takes_the_largest_measured_saving() {
        // About 588 MB each, so a 600 MB budget holds exactly one. At 6 s every
        // workload tail fits 5 s: 16 ms saved on all five, against 13 for 8 s
        // and 20 on the one sub-2 s tail for 2 s.
        let mut evidence = window_evidence(600_000_000);
        gate_everything(&mut evidence, |_, hold| {
            if hold == 6.0 {
                BASELINE_SHA
            } else {
                CHANGED_SHA
            }
        });
        let selection = select_window(&evidence).unwrap();
        assert_eq!(selection.encoder_buckets, [5]);
        assert_eq!(selection.hold_window_seconds, 6.0);
        assert_eq!(selection.installed_bytes, hosted(5).installed_bytes());
    }

    #[test]
    fn a_configuration_that_changes_gold_output_is_never_chosen() {
        let mut evidence = window_evidence(u64::MAX);
        gate_everything(&mut evidence, |buckets, hold| {
            if buckets == [8] && hold == 6.0 {
                BASELINE_SHA
            } else {
                CHANGED_SHA
            }
        });
        let selection = select_window(&evidence).unwrap();
        assert_eq!(selection.encoder_buckets, [8]);
        assert_eq!(selection.hold_window_seconds, 6.0);
        let fastest = &rank_window_candidates(&evidence).unwrap()[0];
        assert!(fastest.predicted_release_ms < selection.predicted_release_ms);

        // Nothing passes: the baseline, which needs no gate.
        let mut evidence = window_evidence(u64::MAX);
        gate_everything(&mut evidence, |_, _| CHANGED_SHA);
        let selection = select_window(&evidence).unwrap();
        assert!(selection.encoder_buckets.is_empty());
        assert_eq!(selection.hold_window_seconds, BASELINE_HOLD_WINDOW_SECONDS);
    }

    #[test]
    fn a_faster_configuration_without_a_verdict_makes_the_evidence_invalid() {
        let evidence = window_evidence(u64::MAX);
        let error = select_window(&evidence).unwrap_err().to_string();
        assert!(error.contains("no gold verdict"), "{error}");
    }

    #[test]
    fn a_profile_written_before_window_tuning_loads_and_keeps_every_bucket_at_6_s() {
        // The profile tune_asr wrote on the M5 Pro on 2026-08-11, verbatim.
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/tuning/m5-pro-profile-pre-4s2q.json");
        let profile = load(&path).unwrap().unwrap();
        assert_eq!(profile.window_tuning, None);
        assert_eq!(
            profile.cache_key,
            "ef53785151eaf33a82c9c14e1eb1c4ca25ece1bec4c5bddafe35fe067c39fe1c"
        );
        profile
            .validate_for(
                &profile.hardware,
                "df2587d4ec1a768b78f11235e51298cc3d44c947b6f0a151729b9e94e7da0002",
            )
            .unwrap();
        let plan = WindowPlan::for_profile(&profile, crate::model_fetch::ENCODER_BUCKETS);
        assert_eq!(plan.encoder_buckets, [2, 5, 8]);
        assert_eq!(plan.hold_window_seconds, 6.0);
        assert!(!serde_json::to_string(&profile)
            .unwrap()
            .contains("window_tuning"));
    }

    #[test]
    fn window_choice_joins_the_key_and_a_re_released_bucket_falls_back() {
        let mut profile = TuningProfile::new(
            hardware(),
            "artifact".into(),
            3,
            vec![
                candidate(BASELINE_COMPUTE_UNITS, 100.0, 50.0, 100.0),
                CandidateMeasurement::failed(CoreMlComputeUnits::All, "unsupported"),
                CandidateMeasurement::failed(CoreMlComputeUnits::CpuAndGpu, "unsupported"),
                CandidateMeasurement::failed(CoreMlComputeUnits::CpuOnly, "unsupported"),
            ],
        )
        .unwrap();
        let untuned_key = profile.cache_key.clone();
        let mut evidence = window_evidence(600_000_000);
        gate_everything(&mut evidence, |buckets, hold| {
            if buckets == [5] && hold == 6.0 {
                BASELINE_SHA
            } else {
                CHANGED_SHA
            }
        });
        profile
            .set_window_tuning(WindowTuning::new(evidence).unwrap())
            .unwrap();
        assert_ne!(profile.cache_key, untuned_key);

        let directory = tempdir().unwrap();
        let path = directory.path().join("profile.json");
        save(&path, &profile).unwrap();
        let loaded = load(&path).unwrap().unwrap();
        assert_eq!(loaded, profile);
        loaded.validate_for(&hardware(), "artifact").unwrap();
        let manifest = crate::model_fetch::ENCODER_BUCKETS;
        let plan = WindowPlan::for_profile(&loaded, manifest);
        assert_eq!(plan.encoder_buckets, [5]);
        assert_eq!(plan.hold_window_seconds, 6.0);

        // The 5 s bucket re-released: compute units stay valid, windows untuned.
        let mut stale = loaded.clone();
        let tuning = stale.window_tuning.as_mut().unwrap();
        let five = tuning
            .evidence
            .encoders
            .iter_mut()
            .find(|cost| cost.window_seconds == 5)
            .unwrap();
        five.archive_sha256 = Some(CHANGED_SHA.into());
        stale.cache_key =
            profile_cache_key(&stale.hardware, "artifact", stale.window_tuning.as_ref()).unwrap();
        stale.validate_for(&hardware(), "artifact").unwrap();
        assert!(stale.window_selection_for(manifest).is_err());
        assert_eq!(
            WindowPlan::for_profile(&stale, manifest),
            WindowPlan::untuned(manifest)
        );

        // An edited choice no longer matches the key it was saved under.
        let mut edited = loaded.clone();
        edited
            .window_tuning
            .as_mut()
            .unwrap()
            .selection
            .hold_window_seconds = 5.0;
        assert!(edited.validate_for(&hardware(), "artifact").is_err());
    }

    #[test]
    fn current_hardware_fingerprint_names_every_available_level() {
        let fingerprint = HardwareFingerprint::current().unwrap();
        assert!(!fingerprint.chip.is_empty());
        assert!(!fingerprint.macos_version.is_empty());
        assert!(fingerprint.memory_bytes > 0);
        assert!(!fingerprint.performance_levels.is_empty());
        assert!(fingerprint
            .performance_levels
            .iter()
            .all(|level| !level.name.is_empty() && level.logical_cpus > 0));
    }
}
