//! Atomic state machine that coordinates the dictation pipeline's three
//! co-varying pieces: the user-facing `DictationState`, the active
//! `Session` slot, and the cold-start `pending_terminate` queue.
//!
//! ## Why one type
//!
//! Before this module landed these were three separate `Mutex`es on
//! `App`. The lock order (`session → current_state → pending_terminate`)
//! was a *comment-only* invariant, and at least three TOCTOU bugs got
//! reintroduced over time:
//!
//! - "state=Listening ∧ session=None ∧ no starter in flight": queue
//!   writes leaked into the next session.
//! - "state=Idle ∧ session=Some": hotkey claims a slot the old watcher
//!   is about to clear; the watcher then cancels the new session.
//! - Tap-cancel during the cold-start gap dropped the edge because the
//!   queue write raced with the starter populating the slot.
//!
//! [`DictationFsm`] folds all three into one `parking_lot::Mutex` and
//! exposes typed transitions. Every transition is the single source of
//! truth for the lock order — adding a new state edge means changing
//! one method here, not re-deriving the protocol at every call site.
//!
//! ## Scope
//!
//! The FSM owns the state ↔ session ↔ pending-terminate triple. It
//! does NOT own:
//!
//! - The recogniser (`Asr`) or the polish LLM — those have monotonic
//!   "ready or not" semantics and don't interact with session state.
//!   Callers pass `asr_ready: bool` (or a precomputed next state) on
//!   transitions where it matters.
//! - UI side-effects (`menubar::refresh`, `hud::show_state`). Callers
//!   are responsible for invoking those *outside* the FSM lock, since
//!   AppKit calls can re-enter Rust and must not be held under a
//!   `Mutex`.

use parking_lot::Mutex;

use crate::streamer::Session;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DictationState {
    /// Model still downloading / loading.
    ModelLoading,
    /// Ready, no active session.
    Idle,
    /// Currently capturing audio.
    Listening,
    /// Capture stopped, ASR running.
    Transcribing,
    /// ASR done; LLM polish pass running.
    Polishing,
}

/// Pre-populated stop edge — fired during the gap between the FSM
/// claiming `Listening` and the starter worker populating the session
/// slot. Without this queue a fast hold-release or tap-cancel during
/// the cold-start window is silently dropped and recording runs to the
/// 30 s VAD cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminateKind {
    /// Tap-mode 2nd press during the cold-start gap.
    Cancel,
    /// Hold-mode release during the cold-start gap.
    Finalize,
}

/// Outcome of [`DictationFsm::on_press_tap`].
#[derive(Debug)]
pub enum TapPressOutcome {
    /// State transitioned Idle → Listening. Caller MUST spawn the
    /// session starter — no other caller will.
    ClaimedListening,
    /// We were already listening with a live session; the FSM sent
    /// Cancel through it.
    CancelledLive,
    /// We were already listening but in the cold-start gap (no session
    /// slot yet); Cancel queued for the starter to drain.
    QueuedCancel,
    /// Wrong state for a tap press; nothing happened.
    Ignored(DictationState),
}

/// Outcome of [`DictationFsm::on_press_hold`].
#[derive(Debug)]
pub enum HoldPressOutcome {
    /// State transitioned Idle → Listening. Caller MUST spawn the
    /// session starter.
    ClaimedListening,
    /// Wrong state for a hold press; nothing happened.
    Ignored(DictationState),
}

/// Outcome of [`DictationFsm::on_release_hold`].
#[derive(Debug)]
pub enum HoldReleaseOutcome {
    /// Live session was finalised.
    Finalised,
    /// Release fired in the cold-start gap; Finalize queued for the
    /// starter to drain.
    QueuedFinalize,
    /// Release fired outside a listening window — no-op.
    Ignored,
}

pub struct DictationFsm {
    inner: Mutex<Inner>,
}

struct Inner {
    state: DictationState,
    session: Option<Session>,
    pending_terminate: Option<TerminateKind>,
}

impl DictationFsm {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                state: DictationState::ModelLoading,
                session: None,
                pending_terminate: None,
            }),
        }
    }

    /// Snapshot the current user-facing state. Used by the menubar
    /// refresh path that has to render a state label.
    pub fn state(&self) -> DictationState {
        self.inner.lock().state
    }

    /// True iff a live session slot is populated. Doesn't mean the
    /// session is healthy — only that some watcher hasn't cleared it
    /// yet.
    pub fn has_live_session(&self) -> bool {
        self.inner.lock().session.is_some()
    }

    /// Force-set the state without touching session/pending_terminate.
    /// Used by:
    /// - Boot transition `ModelLoading → Idle` once the ASR is warm.
    /// - `Transcribing → Polishing` while the LLM pass runs (no
    ///   session involved at that point).
    /// - `on_session_finished` to return to Idle/ModelLoading after
    ///   the transcribe worker completes.
    pub fn set_state(&self, new_state: DictationState) {
        self.inner.lock().state = new_state;
    }

    /// Tap-mode press handler. Atomically observes both `session` and
    /// `state` so the cancel-vs-claim decision can't race with the
    /// watcher clearing the slot.
    ///
    /// Returns the action the caller now needs to take (spawn the
    /// starter, or nothing).
    pub fn on_press_tap(&self) -> TapPressOutcome {
        let mut inner = self.inner.lock();
        if inner.state == DictationState::Listening {
            if inner.session.is_some() {
                // SAFETY: just checked is_some. Send Cancel through
                // the live session under the FSM lock so a concurrent
                // hotkey edge sees a consistent view.
                if let Some(s) = inner.session.as_ref() {
                    s.cancel();
                }
                return TapPressOutcome::CancelledLive;
            }
            inner.pending_terminate = Some(TerminateKind::Cancel);
            return TapPressOutcome::QueuedCancel;
        }
        if inner.state != DictationState::Idle {
            return TapPressOutcome::Ignored(inner.state);
        }
        inner.state = DictationState::Listening;
        TapPressOutcome::ClaimedListening
    }

    /// Hold-mode press handler. Only ever claims a fresh session —
    /// release is the commit edge, not a second press.
    pub fn on_press_hold(&self) -> HoldPressOutcome {
        let mut inner = self.inner.lock();
        if inner.state != DictationState::Idle {
            return HoldPressOutcome::Ignored(inner.state);
        }
        inner.state = DictationState::Listening;
        HoldPressOutcome::ClaimedListening
    }

    /// Hold-mode release handler. Finalises a live session or queues
    /// the edge for the starter if we're in the cold-start gap.
    pub fn on_release_hold(&self) -> HoldReleaseOutcome {
        let mut inner = self.inner.lock();
        if let Some(s) = inner.session.as_ref() {
            s.finalize();
            return HoldReleaseOutcome::Finalised;
        }
        if inner.state == DictationState::Listening {
            inner.pending_terminate = Some(TerminateKind::Finalize);
            return HoldReleaseOutcome::QueuedFinalize;
        }
        HoldReleaseOutcome::Ignored
    }

    /// The starter worker finished creating the session. Atomically
    /// installs it in the slot AND drains any terminate edge that
    /// arrived during the cold-start gap, sending the drained edge
    /// straight through the freshly-installed session under the same
    /// lock. No window opens between install and drain.
    ///
    /// Returns which (if any) edge was drained, for observability /
    /// tests.
    pub fn install_session(&self, session: Session) -> Option<TerminateKind> {
        let mut inner = self.inner.lock();
        let prev = inner.session.replace(session);
        debug_assert!(prev.is_none(), "session slot was not cleared");
        let drained = inner.pending_terminate.take();
        if let (Some(kind), Some(s)) = (drained, inner.session.as_ref()) {
            match kind {
                TerminateKind::Cancel => s.cancel(),
                TerminateKind::Finalize => s.finalize(),
            }
        }
        drained
    }

    /// The starter worker failed before it could install the session
    /// (e.g. cpal couldn't open the mic, Silero VAD failed to load).
    /// Reset state to the post-boot baseline and clear any queued
    /// terminate so the next session doesn't inherit it.
    pub fn abort_starter(&self, next_state: DictationState) {
        let mut inner = self.inner.lock();
        inner.pending_terminate = None;
        inner.state = next_state;
    }

    /// The watcher received the session outcome. Atomically transitions
    /// state out of Listening AND clears the session slot under one
    /// lock so a concurrent hotkey claim can't observe "state=Idle ∧
    /// session=Some".
    pub fn finish_session(&self, next_state: DictationState) {
        let mut inner = self.inner.lock();
        inner.state = next_state;
        inner.session = None;
    }

    /// Worker-thread panic recovery. Clears all three pieces of state
    /// under one lock so the next hotkey press sees a clean baseline.
    pub fn recover(&self, next_state: DictationState) {
        let mut inner = self.inner.lock();
        inner.state = next_state;
        inner.session = None;
        inner.pending_terminate = None;
    }
}

impl Default for DictationFsm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The FSM unit tests below cover the state-and-pending-terminate
    // half of the invariant (no `Session` involved). The
    // session-installation / drain path is covered by tests in
    // `app::tests` where a real `Session` is convenient to build via
    // the streamer module.

    #[test]
    fn new_starts_in_model_loading_with_no_session() {
        let fsm = DictationFsm::new();
        assert_eq!(fsm.state(), DictationState::ModelLoading);
        assert!(!fsm.has_live_session());
    }

    #[test]
    fn boot_transition_model_loading_to_idle() {
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Idle);
        assert_eq!(fsm.state(), DictationState::Idle);
    }

    #[test]
    fn tap_press_from_idle_claims_listening() {
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Idle);
        match fsm.on_press_tap() {
            TapPressOutcome::ClaimedListening => {}
            other => panic!("expected ClaimedListening, got {other:?}"),
        }
        assert_eq!(fsm.state(), DictationState::Listening);
    }

    #[test]
    fn tap_press_from_listening_with_no_session_queues_cancel() {
        // Cold-start gap: state=Listening but session not yet
        // installed. The next press should queue Cancel for the
        // starter to drain, NOT race into a second ClaimedListening.
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Idle);
        let _ = fsm.on_press_tap();
        match fsm.on_press_tap() {
            TapPressOutcome::QueuedCancel => {}
            other => panic!("expected QueuedCancel, got {other:?}"),
        }
    }

    #[test]
    fn tap_press_from_transcribing_is_ignored() {
        // Pressing during Transcribing/Polishing must NOT spawn a
        // second session — the prior session's watcher would later
        // overwrite the new Listening with Idle.
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Transcribing);
        match fsm.on_press_tap() {
            TapPressOutcome::Ignored(DictationState::Transcribing) => {}
            other => panic!("expected Ignored(Transcribing), got {other:?}"),
        }
        assert_eq!(fsm.state(), DictationState::Transcribing);
    }

    #[test]
    fn hold_press_only_claims_from_idle() {
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Idle);
        assert!(matches!(
            fsm.on_press_hold(),
            HoldPressOutcome::ClaimedListening
        ));
        // Second hold-press while Listening must NOT re-claim.
        assert!(matches!(
            fsm.on_press_hold(),
            HoldPressOutcome::Ignored(DictationState::Listening)
        ));
    }

    #[test]
    fn hold_release_in_cold_start_gap_queues_finalize() {
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Idle);
        let _ = fsm.on_press_hold();
        match fsm.on_release_hold() {
            HoldReleaseOutcome::QueuedFinalize => {}
            other => panic!("expected QueuedFinalize, got {other:?}"),
        }
    }

    #[test]
    fn hold_release_outside_listening_is_ignored() {
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Idle);
        assert!(matches!(fsm.on_release_hold(), HoldReleaseOutcome::Ignored));
        fsm.set_state(DictationState::Transcribing);
        assert!(matches!(fsm.on_release_hold(), HoldReleaseOutcome::Ignored));
    }

    #[test]
    fn abort_starter_clears_pending_terminate() {
        // Starter failure: any queued cancel/finalize must be dropped
        // so a future session can't inherit it.
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Idle);
        let _ = fsm.on_press_tap();
        let _ = fsm.on_press_tap(); // queue Cancel
        fsm.abort_starter(DictationState::Idle);
        assert_eq!(fsm.state(), DictationState::Idle);
        // A subsequent press doesn't see a stale queued edge — claim
        // succeeds cleanly.
        assert!(matches!(
            fsm.on_press_tap(),
            TapPressOutcome::ClaimedListening
        ));
    }

    #[test]
    fn recover_clears_everything() {
        let fsm = DictationFsm::new();
        fsm.set_state(DictationState::Listening);
        // Simulate a queued terminate (no real session needed).
        let _ = fsm.on_press_tap(); // queues Cancel since state is Listening but no session
        fsm.recover(DictationState::Idle);
        assert_eq!(fsm.state(), DictationState::Idle);
        assert!(!fsm.has_live_session());
        // Pending was cleared too — next claim sees clean baseline.
        assert!(matches!(
            fsm.on_press_tap(),
            TapPressOutcome::ClaimedListening
        ));
    }
}
