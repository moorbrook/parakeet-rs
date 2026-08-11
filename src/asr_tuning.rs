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
use crate::performance;

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
        let cache_key = cache_key(&hardware, &artifact_digest)?;
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
        })
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
        if self.cache_key != cache_key(hardware, artifact_digest)? {
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
    #[derive(Serialize)]
    struct Key<'a> {
        hardware: &'a HardwareFingerprint,
        backend_id: &'static str,
        artifact_digest: &'a str,
        tuner_version: u32,
    }
    let bytes = serde_json::to_vec(&Key {
        hardware,
        backend_id: BACKEND_ID,
        artifact_digest,
        tuner_version: TUNER_VERSION,
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
