//! Apple-Silicon P-core count + per-dictation `PhaseTimer` (one structured
//! log line per dictation, parsed by `scripts/bench-aggregate.py`).
//! See `docs/latency-plan.md` §1 for the log-line contract.

use std::ffi::CString;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// `sysctlbyname("hw.perflevel0.logicalcpu")` returns the P-core count on
/// Apple Silicon (M1+). Falls back to half of total logicals if the sysctl
/// is missing — non-Apple-Silicon Macs, hypothetically.
pub fn performance_core_count() -> i32 {
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let name = CString::new("hw.perflevel0.logicalcpu")
        .expect("static sysctl name contains no interior NUL");
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        value
    } else {
        (num_cpus_total() / 2).max(2) as i32
    }
}

// The closure can't become `NonZero::get` (what the lint wants): that
// path form is only stable since 1.79 and MSRV is 1.77 — the lint and
// `clippy::incompatible_msrv` are in direct conflict here.
#[allow(clippy::redundant_closure_for_method_calls)]
fn num_cpus_total() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Tag used at the start of every PhaseTimer log line — the aggregator
/// in `scripts/bench-aggregate.py` keys on this exact string.
pub const PHASE_TIMER_TAG: &str = "phase_timer";

/// Whether this dictation came from the real menu-bar app or from
/// `bench-asr`. Aggregator buckets separately so synthetic numbers
/// can't contaminate real-user telemetry.
#[derive(Clone, Copy, Debug)]
pub enum PhaseTimerMode {
    /// Real dictation triggered by the user's hotkey.
    Real,
    /// Synthetic bench iteration. The ASR-only harness uses a pre-loaded WAV;
    /// the end-to-end harness uses an explicit loopback capture device.
    Bench,
}

impl PhaseTimerMode {
    fn as_str(self) -> &'static str {
        match self {
            PhaseTimerMode::Real => "real",
            PhaseTimerMode::Bench => "bench",
        }
    }
}

/// Per-dictation timing snapshot. Anchored at `t0 = Instant::now()` at
/// construction (session start: audio capture begin in Real mode, iter
/// begin in Bench mode). Markers populate as the pipeline advances; the
/// final `emit()` writes a single log line in this shape:
///
/// ```text
/// phase_timer mode=real session_id=68291f4b-0007 audio_s=4.83 \
///   t_capture_end=4831 t_vad_endpoint=4831 t_asr_start=4704 \
///   t_asr_done=4770 t_paste_done=4835 dur_post_endpoint_ms=4 \
///   t_speech_end=4671 dur_end_to_end_ms=164
/// ```
///
/// All `t_*` are ms offsets from `t0`. Absent markers print as `-` —
/// `scripts/bench-aggregate.py` skips lines with `-` in fields it needs.
///
/// `dur_post_endpoint_ms` is retained for compatibility and starts when VAD
/// finishes its silence confirmation. `dur_end_to_end_ms` starts at the
/// estimated last voiced sample, so it includes the VAD hangover and is the
/// user-facing end-of-speech-to-text measurement.
pub struct PhaseTimer {
    session_id: String,
    mode: PhaseTimerMode,
    t0: Instant,
    audio_s: Option<f32>,
    t_capture_end_ms: Option<u32>,
    t_speech_end_ms: Option<u32>,
    t_vad_endpoint_ms: Option<u32>,
    t_asr_start_ms: Option<u32>,
    t_asr_done_ms: Option<u32>,
    t_paste_done_ms: Option<u32>,
}

impl PhaseTimer {
    /// Start the clock. `session_id` should be unique per dictation so
    /// log lines can be cross-referenced. Use `next_session_id()` for a
    /// monotonically-increasing default.
    pub fn start(mode: PhaseTimerMode, session_id: String) -> Self {
        Self {
            session_id,
            mode,
            t0: Instant::now(),
            audio_s: None,
            t_capture_end_ms: None,
            t_speech_end_ms: None,
            t_vad_endpoint_ms: None,
            t_asr_start_ms: None,
            t_asr_done_ms: None,
            t_paste_done_ms: None,
        }
    }

    fn elapsed_ms(&self) -> u32 {
        // 49 days saturates a u32 ms; we'll never get close on a dictation.
        self.t0.elapsed().as_millis().min(u128::from(u32::MAX)) as u32
    }

    /// Capture-end: AudioCapture buffer was just collected. `audio_s` is
    /// the captured duration in seconds.
    pub fn mark_capture_end(&mut self, audio_s: f32) {
        self.t_capture_end_ms = Some(self.elapsed_ms());
        self.audio_s = Some(audio_s);
    }

    /// Record the estimated last voiced sample on the captured-audio
    /// timeline. The VAD watcher confirms this timestamp only after its full
    /// silence-safety window survives without resumed speech.
    pub fn mark_speech_end_at_audio_offset(&mut self, audio_seconds: f32) {
        let offset_ms = (audio_seconds.max(0.0) * 1_000.0)
            .round()
            .clamp(0.0, u32::MAX as f32) as u32;
        self.t_speech_end_ms = Some(offset_ms.min(self.elapsed_ms()));
    }

    /// Override the acoustic endpoint with an externally observed monotonic
    /// instant. The loopback benchmark uses Core Audio's predicted playback
    /// timestamp for the fixture's final non-silent sample, avoiding any
    /// dependency on the VAD implementation being measured.
    pub fn mark_speech_end_at_instant(&mut self, speech_end: Instant) {
        let offset = speech_end
            .saturating_duration_since(self.t0)
            .as_millis()
            .min(u128::from(u32::MAX)) as u32;
        self.t_speech_end_ms = Some(offset.min(self.elapsed_ms()));
    }

    /// VAD confirmed the endpoint, or the user explicitly finalized a Hold
    /// session. When no earlier acoustic endpoint exists, this moment is the
    /// best available end-of-speech timestamp.
    pub fn mark_vad_endpoint(&mut self) {
        let now = self.elapsed_ms();
        self.t_speech_end_ms.get_or_insert(now);
        self.t_vad_endpoint_ms = Some(now);
    }

    pub fn mark_asr_start(&mut self) {
        self.t_asr_start_ms = Some(self.elapsed_ms());
    }

    pub fn mark_asr_done(&mut self) {
        self.t_asr_done_ms = Some(self.elapsed_ms());
    }

    /// Transcript delivery returned (currently a synthetic Unicode event).
    /// Latency clock stops here.
    pub fn mark_paste_done(&mut self) {
        self.t_paste_done_ms = Some(self.elapsed_ms());
    }

    /// Post-endpoint latency: VAD endpoint (or capture end in Manual mode)
    /// → paste done. The "<1 s" acceptance criterion in §6 of the latency
    /// plan is stated in terms of this number.
    fn post_endpoint_dur_ms(&self) -> Option<u32> {
        let endpoint = self.t_vad_endpoint_ms.or(self.t_capture_end_ms)?;
        let done = self.t_paste_done_ms?;
        Some(done.saturating_sub(endpoint))
    }

    fn end_to_end_dur_ms(&self) -> Option<u32> {
        let speech_end = self
            .t_speech_end_ms
            .or(self.t_vad_endpoint_ms)
            .or(self.t_capture_end_ms)?;
        let done = self.t_paste_done_ms?;
        Some(done.saturating_sub(speech_end))
    }

    /// Write the structured log line. Idempotent — consumes `self`.
    pub fn emit(self) {
        let post_endpoint = self.post_endpoint_dur_ms();
        let end_to_end = self.end_to_end_dur_ms();
        // Single-line shape is the contract with the aggregator. Field
        // order doesn't matter (parser is k=v based), but new fields
        // should be appended so older log files remain parseable.
        log::info!(
            "{tag} mode={mode} session_id={sid} audio_s={audio} t_capture_end={tce} t_vad_endpoint={tve} t_asr_start={tas} t_asr_done={tad} t_paste_done={tpd} dur_post_endpoint_ms={dpe} t_speech_end={tse} dur_end_to_end_ms={de2e}",
            tag = PHASE_TIMER_TAG,
            mode = self.mode.as_str(),
            sid = self.session_id,
            audio = OptF32(self.audio_s),
            tce = OptU32(self.t_capture_end_ms),
            tve = OptU32(self.t_vad_endpoint_ms),
            tas = OptU32(self.t_asr_start_ms),
            tad = OptU32(self.t_asr_done_ms),
            tpd = OptU32(self.t_paste_done_ms),
            dpe = OptU32(post_endpoint),
            tse = OptU32(self.t_speech_end_ms),
            de2e = OptU32(end_to_end),
        );
    }
}

/// `Display` newtypes that render `Some(n)` and `None` (as `-`) without
/// allocating a `String` per field — the eager `format!` path showed up
/// on the latency hot path even when info-level was filtered out.
struct OptU32(Option<u32>);
struct OptF32(Option<f32>);

impl fmt::Display for OptU32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(n) => write!(f, "{n}"),
            None => f.write_str("-"),
        }
    }
}

impl fmt::Display for OptF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(n) => write!(f, "{n:.3}"),
            None => f.write_str("-"),
        }
    }
}

/// Monotonically-increasing session ID. The shape is
/// `<epoch_secs:x>-<launch_nonce:x>-<counter:04x>` so:
///
/// - The epoch prefix keeps logs roughly time-sorted across launches.
/// - The launch nonce (PID xor a one-shot per-process random value)
///   disambiguates two app launches that land in the same second —
///   without it both would start at counter `0000` and collide in the
///   aggregator. The nonce is computed once per process.
/// - The counter disambiguates two dictations within a single launch.
pub fn next_session_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    static LAUNCH_NONCE: OnceLock<u32> = OnceLock::new();
    let nonce = *LAUNCH_NONCE.get_or_init(|| {
        let pid = std::process::id();
        // Mix in a per-process random so two launches with sequential
        // PIDs (common in CI) still diverge.
        let rand = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        pid ^ rand
    });
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("{ts:x}-{nonce:x}-{n:04x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn post_endpoint_uses_vad_when_present() {
        let mut t = PhaseTimer::start(PhaseTimerMode::Real, "test-1".into());
        // Simulate 1 s of audio capture before VAD endpoint.
        sleep(Duration::from_millis(5));
        t.mark_capture_end(1.0);
        t.mark_vad_endpoint();
        sleep(Duration::from_millis(5));
        t.mark_asr_start();
        t.mark_asr_done();
        t.mark_paste_done();
        // VAD endpoint should be the anchor for post_endpoint.
        let post = t.post_endpoint_dur_ms().expect("set");
        // Capture+VAD happened together at ~5ms; paste done at ~10ms;
        // so post-endpoint should be small (under 50 ms is generous).
        assert!(post < 50, "post_endpoint_ms = {post}, expected small");
    }

    #[test]
    fn post_endpoint_falls_back_to_capture_end_in_manual_mode() {
        let mut t = PhaseTimer::start(PhaseTimerMode::Real, "test-2".into());
        t.mark_capture_end(2.0);
        // Manual mode: no VAD endpoint marker.
        t.mark_asr_start();
        t.mark_asr_done();
        t.mark_paste_done();
        assert!(t.post_endpoint_dur_ms().is_some());
    }

    #[test]
    fn post_endpoint_is_none_until_paste_done() {
        let mut t = PhaseTimer::start(PhaseTimerMode::Real, "test-3".into());
        t.mark_capture_end(1.0);
        t.mark_vad_endpoint();
        // No paste-done mark yet.
        assert!(t.post_endpoint_dur_ms().is_none());
        assert!(t.end_to_end_dur_ms().is_none());
    }

    #[test]
    fn end_to_end_includes_time_before_vad_confirmation() {
        let mut t = PhaseTimer::start(PhaseTimerMode::Real, "test-e2e".into());
        t.mark_speech_end_at_audio_offset(0.0);
        sleep(Duration::from_millis(5));
        t.mark_vad_endpoint();
        sleep(Duration::from_millis(5));
        t.mark_paste_done();
        let post = t.post_endpoint_dur_ms().expect("set");
        let end_to_end = t.end_to_end_dur_ms().expect("set");
        assert!(end_to_end > post);
    }

    #[test]
    fn external_acoustic_endpoint_overrides_vad_estimate() {
        let mut t = PhaseTimer::start(PhaseTimerMode::Bench, "test-external-end".into());
        let speech_end = Instant::now();
        sleep(Duration::from_millis(5));
        t.mark_speech_end_at_audio_offset(99.0);
        t.mark_speech_end_at_instant(speech_end);
        sleep(Duration::from_millis(5));
        t.mark_paste_done();
        assert!(t.end_to_end_dur_ms().is_some_and(|duration| duration >= 5));
    }

    #[test]
    fn session_ids_are_unique() {
        let a = next_session_id();
        let b = next_session_id();
        assert_ne!(a, b);
    }

    #[test]
    fn opt_formatters_render_dash_when_absent() {
        assert_eq!(OptU32(None).to_string(), "-");
        assert_eq!(OptU32(Some(42)).to_string(), "42");
        assert_eq!(OptF32(None).to_string(), "-");
        assert_eq!(OptF32(Some(4.83)).to_string(), "4.830");
    }
}
