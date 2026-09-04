//! Startup warmup, ds4-style.
//!
//! Two passes:
//! 1. mmap the .onnx file and walk one byte per 16 KB page. The kernel populates
//!    the page cache before the first recognition needs it, so the first decode
//!    doesn't pay a cold-read tax (~hundreds of ms on a 1 GB model).
//! 2. Run one tiny silent decode through the recognizer. CoreML compiles its
//!    graph the first time it sees a shape; we eat that cost during startup so
//!    the first user press feels instant.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use memmap2::Mmap;

use crate::asr::Asr;

/// Walk the mmap'd model file, touching one byte every `STRIDE` bytes.
pub fn page_touch(model_path: &Path) -> Result<u64> {
    let file = std::fs::File::open(model_path)
        .with_context(|| format!("opening {} for warmup", model_path.display()))?;
    // SAFETY: model artifacts are immutable after verified installation, the
    // file stays alive while the read-only map is created, and this function
    // never writes through or aliases the mapping mutably.
    let map = unsafe { Mmap::map(&file).context("mmap model file")? };

    // 16 KiB matches macOS' default page size on Apple Silicon.
    const STRIDE: usize = 16 * 1024;
    let bytes = &map[..];
    let mut acc: u64 = 0;
    let mut i = 0;
    while i < bytes.len() {
        // `std::hint::black_box` keeps LLVM from optimising the read away.
        acc = acc.wrapping_add(u64::from(std::hint::black_box(bytes[i])));
        i += STRIDE;
    }
    Ok(acc)
}

/// Two-pass startup warmup against the recognizer:
/// 1. **Throwaway pass.** 0.5 s of silence pays the CoreML graph-compile cost
///    and primes CPU/ANE caches. Its timing is meaningless and we ignore it.
/// 2. **Measured pass.** 2 s of silence runs through the now-warm graph.
///    `recognize_with_timing` logs the resulting RTFx; that's the steady-state
///    number we want users (and ADR-0015 layer 3) to see in the log.
pub fn dummy_decode(asr: &Asr) -> Result<()> {
    // Pass 1: small sample, timing discarded.
    let throwaway = vec![0.0_f32; 16_000 / 2];
    let _ = asr.recognize_silent_warmup(&throwaway, 16_000)?;
    // Pass 2: longer sample, timing logged via `recognize`.
    let measured = vec![0.0_f32; 16_000 * 2];
    let _ = asr.recognize(&measured, 16_000)?;
    Ok(())
}

/// Sample rate the priming buffers are built at. The worker resamples anything
/// else, so generating at 16 kHz keeps the prime off the resampler.
const PRIME_SAMPLE_RATE: u32 = 16_000;

/// Length of one priming buffer.
///
/// 0.5 s matches the throwaway pass of [`dummy_decode`], the shape already
/// known to compile and run cleanly on this model.
///
/// The size used to be irrelevant: one 15 s encoder ran the same window
/// whatever the input length. Bucketed encoders changed that —
/// `EncoderBuckets.select` routes a request to the narrowest compiled window
/// that holds it, so this buffer picks the smallest bucket rather than the one
/// the utterance to come will use. The engine's power gate is a hardware unit
/// that any dispatch lifts, so the re-wake this exists to avoid is still
/// covered; a per-program load cost across buckets would not be. Kata xt0y
/// follows up once bucket artifacts ship.
pub const PRIME_SECONDS: f32 = 0.5;

/// Run one minimal dispatch through the recognizer.
///
/// The Apple Neural Engine hard power-gates when idle, so the first dispatch
/// after a gap of seconds to minutes pays a re-wake the user feels as endpoint
/// latency. This is the cheapest program that still touches the encoder, which
/// is the only stage this pipeline places on the engine.
pub fn prime_engine(asr: &Asr) -> Result<()> {
    let samples = vec![0.0_f32; (PRIME_SAMPLE_RATE as f32 * PRIME_SECONDS) as usize];
    let _ = asr.recognize_silent_warmup(&samples, PRIME_SAMPLE_RATE)?;
    Ok(())
}

/// Fires [`prime_engine`] on a worker thread, at most one at a time.
///
/// The hotkey-down edge runs on the AppKit main thread under a hard ~250 ms
/// budget before macOS disables the event tap, so the prime can never run
/// inline. The in-flight flag matters because the Core ML worker serializes on
/// one pipe: a second prime queued behind the first would make a fast
/// press-release wait for both.
#[derive(Clone, Debug, Default)]
pub struct EnginePrimer {
    in_flight: Arc<AtomicBool>,
}

impl EnginePrimer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn a prime unless one is still running.
    ///
    /// Returns `true` when a thread was spawned. A `false` return is the
    /// normal outcome of a rapid second press, not an error.
    ///
    /// The guard drops the second request rather than queueing it, but it
    /// cannot cancel the first: a press-release short enough to end while a
    /// prime is still running puts that endpoint decode behind one dispatch on
    /// the worker's single pipe, about 30 to 50 ms. It is bounded at one, and
    /// it can only happen inside a session the press itself started, which is
    /// the case that was going to pay a re-wake anyway. A press arriving while
    /// a decode is in flight never reaches here: `App::on_hotkey_press` primes
    /// only on an FSM outcome that changed state, so a prime cannot queue
    /// ahead of a final or tail-window decode.
    pub fn prime_in_background(&self, asr: Arc<Asr>) -> bool {
        if self
            .in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            log::debug!("engine prime skipped: one is already in flight");
            return false;
        }
        let clear_on_exit = InFlightGuard {
            flag: Arc::clone(&self.in_flight),
        };
        std::thread::spawn(move || {
            // Held across the dispatch so the flag is cleared by unwinding
            // too. A panic crossing the worker's IPC boundary would otherwise
            // latch it true and silently disable priming for the rest of the
            // process lifetime.
            let _clear_on_exit = clear_on_exit;
            if let Err(error) = prime_engine(&asr) {
                log::warn!("engine prime failed; the next decode pays the re-wake: {error:#}");
            }
        });
        true
    }

    /// True while a spawned prime has not yet returned.
    pub fn is_in_flight(&self) -> bool {
        self.in_flight.load(Ordering::Acquire)
    }
}

/// Clears [`EnginePrimer`]'s in-flight flag on both the normal and the
/// unwinding path out of a prime.
struct InFlightGuard {
    flag: Arc<AtomicBool>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// A background thread that re-primes the engine on a fixed cadence.
///
/// [`Self::stop`] joins the thread. Callers must join before the decode they
/// are about to measure or serve: the worker's pipe is a single mutex, so an
/// unjoined keep-alive can put a whole encoder pass in front of the real
/// request.
#[derive(Debug)]
pub struct KeepAlive {
    stop: Arc<AtomicBool>,
    dispatches: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl KeepAlive {
    pub fn start(asr: Arc<Asr>, period: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let dispatches = Arc::new(AtomicU64::new(0));
        let thread_stop = Arc::clone(&stop);
        let thread_dispatches = Arc::clone(&dispatches);
        let handle = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                match prime_engine(&asr) {
                    Ok(()) => {
                        thread_dispatches.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(error) => {
                        log::warn!("keep-alive dispatch failed, stopping cadence: {error:#}");
                        break;
                    }
                }
                // Sleep in short slices so `stop` is observed promptly rather
                // than one full period late.
                let mut remaining = period;
                let slice = Duration::from_millis(10);
                while remaining > Duration::ZERO && !thread_stop.load(Ordering::Acquire) {
                    let step = remaining.min(slice);
                    std::thread::sleep(step);
                    remaining -= step;
                }
            }
        });
        Self {
            stop,
            dispatches,
            handle: Some(handle),
        }
    }

    /// Signal the cadence to stop, wait for its in-flight dispatch to finish,
    /// and report how many dispatches it fired.
    pub fn stop(mut self) -> u64 {
        self.signal_and_join();
        self.dispatches.load(Ordering::Relaxed)
    }

    fn signal_and_join(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            // A panicking keep-alive thread has already been reported by the
            // default hook; losing its count is not worth propagating a panic
            // into a caller that is about to serve a real utterance.
            let _ = handle.join();
        }
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.signal_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::{AsrBackend, AsrBackendMetadata, Decoded};

    /// Counts dispatches and holds each one for a fixed time, so a test can
    /// tell the difference between "signalled" and "actually finished".
    struct CountingBackend {
        metadata: AsrBackendMetadata,
        calls: Arc<AtomicU64>,
        hold: Duration,
    }

    impl CountingBackend {
        fn new(hold: Duration) -> (Arc<Self>, Arc<AtomicU64>) {
            let calls = Arc::new(AtomicU64::new(0));
            let backend = Arc::new(Self {
                metadata: AsrBackendMetadata {
                    backend: "counting".into(),
                    model: "none".into(),
                    quantization: "none".into(),
                    execution_provider: "test".into(),
                },
                calls: Arc::clone(&calls),
                hold,
            });
            (backend, calls)
        }
    }

    impl AsrBackend for CountingBackend {
        fn metadata(&self) -> &AsrBackendMetadata {
            &self.metadata
        }

        fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Decoded> {
            std::thread::sleep(self.hold);
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Decoded {
                text: String::new(),
                audio_seconds: samples.len() as f32 / sample_rate as f32,
                decode_seconds: 0.0,
            })
        }
    }

    #[test]
    fn the_prime_buffer_is_silent_and_nonempty() {
        // An empty buffer short-circuits inside `Asr::recognize_with_timing`
        // and never reaches the backend, which would make the prime a no-op
        // that still reads as success.
        let (backend, calls) = CountingBackend::new(Duration::ZERO);
        let asr = Asr::from_backend(backend);
        prime_engine(&asr).expect("prime must succeed");
        // Reaching the backend at all is the assertion: a zero-length buffer
        // would return Ok without ever dispatching.
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_second_prime_is_dropped_while_the_first_is_in_flight() {
        let (backend, calls) = CountingBackend::new(Duration::from_millis(200));
        let asr = Arc::new(Asr::from_backend(backend));
        let primer = EnginePrimer::new();

        assert!(primer.prime_in_background(Arc::clone(&asr)));
        // The first prime holds the fake backend for 200 ms, so this second
        // request lands squarely inside its window.
        assert!(!primer.prime_in_background(Arc::clone(&asr)));

        while primer.is_in_flight() {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // Once the first one has drained, priming is available again.
        assert!(primer.prime_in_background(asr));
    }

    /// Panics on the first dispatch, succeeds afterwards.
    struct PanicOnceBackend {
        metadata: AsrBackendMetadata,
        calls: Arc<AtomicU64>,
    }

    impl AsrBackend for PanicOnceBackend {
        fn metadata(&self) -> &AsrBackendMetadata {
            &self.metadata
        }

        fn transcribe(&self, _samples: &[f32], _sample_rate: u32) -> Result<Decoded> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                panic!("worker IPC exploded");
            }
            Ok(Decoded {
                text: String::new(),
                audio_seconds: 0.5,
                decode_seconds: 0.0,
            })
        }
    }

    #[test]
    fn a_panicking_dispatch_does_not_latch_the_in_flight_flag() {
        // Without the drop guard the flag stays true for the process lifetime
        // and every later prime is dropped in silence — the app would keep
        // paying the re-wake it was built to avoid, with nothing in the log to
        // say why.
        let calls = Arc::new(AtomicU64::new(0));
        let asr = Arc::new(Asr::from_backend(Arc::new(PanicOnceBackend {
            metadata: AsrBackendMetadata {
                backend: "panic-once".into(),
                model: "none".into(),
                quantization: "none".into(),
                execution_provider: "test".into(),
            },
            calls: Arc::clone(&calls),
        })));
        let primer = EnginePrimer::new();

        // The spawned thread panics; the default hook prints to stderr, which
        // is noise in the test output and not a failure.
        assert!(primer.prime_in_background(Arc::clone(&asr)));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while primer.is_in_flight() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !primer.is_in_flight(),
            "the guard must clear the flag while unwinding"
        );

        assert!(primer.prime_in_background(asr));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while primer.is_in_flight() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "priming must still work after a panicking dispatch"
        );
    }

    #[test]
    fn stopping_the_keep_alive_waits_for_its_in_flight_dispatch() {
        let (backend, calls) = CountingBackend::new(Duration::from_millis(120));
        let asr = Arc::new(Asr::from_backend(backend));
        let keep_alive = KeepAlive::start(asr, Duration::from_millis(10));
        std::thread::sleep(Duration::from_millis(30));

        let reported = keep_alive.stop();
        // `stop` joins, so no dispatch can still be running: the count the
        // caller was handed is final. Without the join the backend would keep
        // incrementing after this read and the assert below would flake.
        let after_stop = calls.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(calls.load(Ordering::Relaxed), after_stop);
        assert_eq!(reported, after_stop);
        assert!(reported >= 1, "the cadence must fire at least once");
    }
}
