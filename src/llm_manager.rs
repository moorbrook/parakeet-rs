//! Single-mutex state machine for the cleanup-LLM lifecycle.
//!
//! Replaces the previous `App::llm: Mutex<Option<Arc<dyn CleanupBackend>>>`
//! and `App::llm_loading: Mutex<bool>` pair. Those two mutexes encoded
//! one logical state — `{ Disabled, Loading, Ready(Arc) }` — but the
//! ordering between them had to be re-derived at every call site:
//! `try_claim_llm_load`, `finalize_llm_load`, the Settings-toggle
//! handler, and the panic recovery. Folding both fields into one
//! `LlmState` enum behind one `parking_lot::Mutex` makes the invalid
//! combinations unrepresentable.
//!
//! ## Scope
//!
//! This module owns the *bookkeeping* — claiming the load slot,
//! finalising it, racing it against a user-driven `disable()`. It does
//! NOT own the actual loader (the GGUF mmap + Metal init); the caller
//! passes a `Result<Arc<dyn CleanupBackend>>` into `finalize_load`.
//! That keeps the manager testable without a real 1.2 GB model on
//! disk.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::cleanup::CleanupBackend;

/// Single source of truth for the cleanup-LLM lifecycle.
pub struct LlmManager {
    inner: Mutex<LlmState>,
}

enum LlmState {
    /// Cleanup mode is `Off`, or we've never tried to load.
    Disabled,
    /// A load is in flight; another claim must wait.
    Loading,
    /// The model is loaded and `try_get` returns it.
    Ready(Arc<dyn CleanupBackend>),
}

/// Result of [`LlmManager::try_claim_load`]. Names beat booleans —
/// callers can't accidentally treat "already loaded" the same as
/// "another load in flight".
#[derive(Debug, PartialEq, Eq)]
pub enum LoadClaim {
    /// THIS caller owns the load slot. Caller MUST spawn the loader
    /// thread and eventually call `finalize_load`.
    Claimed,
    /// A load is already in flight; do nothing (the existing load will
    /// finalise on its own).
    AlreadyLoading,
    /// The model is already loaded; do nothing.
    AlreadyLoaded,
}

/// Result of [`LlmManager::finalize_load`]. The caller uses this to
/// drive the menubar status text.
#[derive(Debug)]
pub enum FinalizeOutcome {
    /// Load succeeded AND the user still wants cleanup ON; the
    /// backend is now stored.
    Stored,
    /// Load succeeded but the user toggled cleanup OFF while we were
    /// loading — the backend was discarded.
    DiscardedDisabled,
    /// Loader returned an error. State is now `Disabled` so a retry
    /// (Off→On toggle, or app restart) can try again cleanly.
    Failed(anyhow::Error),
}

impl LlmManager {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(LlmState::Disabled),
        }
    }

    /// Returns the loaded backend if one is ready. Used by
    /// `deliver_cleaned` to decide between cleanup-streaming and raw
    /// paste fallback.
    pub fn try_get(&self) -> Option<Arc<dyn CleanupBackend>> {
        match &*self.inner.lock() {
            LlmState::Ready(b) => Some(Arc::clone(b)),
            _ => None,
        }
    }

    /// Atomically advance from `Disabled` to `Loading` if no load is
    /// in flight and no model is already loaded. The returned variant
    /// tells the caller whether to actually spawn the loader work.
    pub fn try_claim_load(&self) -> LoadClaim {
        let mut state = self.inner.lock();
        match &*state {
            LlmState::Loading => LoadClaim::AlreadyLoading,
            LlmState::Ready(_) => LoadClaim::AlreadyLoaded,
            LlmState::Disabled => {
                *state = LlmState::Loading;
                LoadClaim::Claimed
            }
        }
    }

    /// The loader thread is done. `result` is what the loader produced;
    /// `keep_if_loaded` is the user's *current* cleanup mode at the
    /// moment we finalise — `true` means "user still wants cleanup on,
    /// store the backend"; `false` means "user toggled Off while we
    /// were loading, discard the backend." Reading this *inside* the
    /// critical section closes the race where the Settings toggle
    /// would clear the slot a tick after we wrote it.
    pub fn finalize_load(
        &self,
        result: anyhow::Result<Arc<dyn CleanupBackend>>,
        keep_if_loaded: bool,
    ) -> FinalizeOutcome {
        let mut state = self.inner.lock();
        match result {
            Ok(backend) => {
                if keep_if_loaded {
                    *state = LlmState::Ready(backend);
                    FinalizeOutcome::Stored
                } else {
                    *state = LlmState::Disabled;
                    drop(backend);
                    FinalizeOutcome::DiscardedDisabled
                }
            }
            Err(e) => {
                *state = LlmState::Disabled;
                FinalizeOutcome::Failed(e)
            }
        }
    }

    /// User toggled cleanup OFF. Drops the loaded backend if any. If
    /// a load is in flight, the in-flight loader's `finalize_load`
    /// will see `keep_if_loaded == false` (the Settings layer wrote
    /// `Off` before calling `disable`) and discard its result.
    pub fn disable(&self) {
        let mut state = self.inner.lock();
        // Replace whatever's there with Disabled. If it was Ready,
        // the Arc drops here — but a polish-in-flight clone still
        // keeps the model alive until that call completes.
        *state = LlmState::Disabled;
    }

    /// Reset `Loading` back to `Disabled` after a panic in the loader
    /// thread. Without this the toggle would silently no-op forever
    /// (`try_claim_load` would observe `Loading` and refuse to spawn).
    /// No-op if state isn't `Loading` (e.g. a concurrent `disable`
    /// already moved us to `Disabled`).
    pub fn clear_loading_after_panic(&self) {
        let mut state = self.inner.lock();
        if matches!(*state, LlmState::Loading) {
            *state = LlmState::Disabled;
        }
    }
}

impl Default for LlmManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    struct DummyBackend;
    impl CleanupBackend for DummyBackend {
        fn polish_into(
            &self,
            _text: &str,
            _on_chunk: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<()> {
            Ok(())
        }
        fn warmup(&self) -> Result<()> {
            Ok(())
        }
    }

    fn backend() -> Arc<dyn CleanupBackend> {
        Arc::new(DummyBackend)
    }

    #[test]
    fn fresh_manager_starts_disabled() {
        let m = LlmManager::new();
        assert!(m.try_get().is_none());
    }

    #[test]
    fn first_claim_wins_second_sees_already_loading() {
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        assert_eq!(m.try_claim_load(), LoadClaim::AlreadyLoading);
        // Until finalize_load runs, the model isn't visible.
        assert!(m.try_get().is_none());
    }

    #[test]
    fn finalize_stores_when_user_still_wants_it() {
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        let outcome = m.finalize_load(Ok(backend()), /* keep */ true);
        assert!(matches!(outcome, FinalizeOutcome::Stored));
        assert!(m.try_get().is_some());
        // Subsequent claim sees AlreadyLoaded — no duplicate load.
        assert_eq!(m.try_claim_load(), LoadClaim::AlreadyLoaded);
    }

    #[test]
    fn finalize_discards_when_user_flipped_off_during_load() {
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        let outcome = m.finalize_load(Ok(backend()), /* keep */ false);
        assert!(matches!(outcome, FinalizeOutcome::DiscardedDisabled));
        assert!(m.try_get().is_none());
        // State is back to Disabled — a fresh On toggle can claim
        // again.
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
    }

    #[test]
    fn finalize_error_resets_to_disabled_so_retry_works() {
        // Loader failure (missing GGUF, OOM, …) must NOT leave the
        // manager stuck in `Loading` — a future On toggle has to be
        // able to retry. Mutation-survivable: a forget-to-reset bug
        // makes the second claim observe AlreadyLoading and silently
        // no-op forever.
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        let outcome = m.finalize_load(Err(anyhow::anyhow!("missing gguf")), true);
        match outcome {
            FinalizeOutcome::Failed(e) => assert!(e.to_string().contains("missing gguf")),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert!(m.try_get().is_none());
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
    }

    #[test]
    fn disable_drops_loaded_backend() {
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        let _ = m.finalize_load(Ok(backend()), true);
        assert!(m.try_get().is_some());
        m.disable();
        assert!(m.try_get().is_none());
    }

    #[test]
    fn clear_loading_after_panic_resets_only_loading_state() {
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        m.clear_loading_after_panic();
        // After the panic-cleanup, a new claim succeeds.
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
    }

    #[test]
    fn clear_loading_after_panic_no_op_when_already_ready() {
        // If by some race the loader managed to store before the
        // panic handler ran, the panic-cleanup must NOT clear the
        // freshly-loaded model.
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        let _ = m.finalize_load(Ok(backend()), true);
        m.clear_loading_after_panic();
        assert!(m.try_get().is_some(), "loaded backend must survive");
    }

    #[test]
    fn rapid_toggle_off_then_on_during_load_discards_and_retries() {
        // Realistic scenario: boot starts a load, user flips Off
        // before it finishes (we call disable()), then On again.
        let m = LlmManager::new();
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
        // User flips Off while we're loading.
        m.disable();
        // The in-flight loader finishes successfully but settings
        // now says Off → we tell finalize_load not to keep.
        let _ = m.finalize_load(Ok(backend()), /* keep */ false);
        assert!(m.try_get().is_none());
        // User flips back to On — claim succeeds again.
        assert_eq!(m.try_claim_load(), LoadClaim::Claimed);
    }
}
