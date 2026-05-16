//! Headless ASR bench harness.
//!
//! Loads a WAV file, runs it through `parakeet_rs::asr::Asr` N times, and
//! emits one `phase_timer` line per iteration to stderr. `scripts/bench-latency.sh`
//! drives it across {1, 3, 5, 10, 20} s fixtures and `scripts/bench-aggregate.py`
//! reduces the log into p50/p95/p99 per length.
//!
//! Uses the same `SettingsStore` paths as the menu-bar app, so the model
//! must already be downloaded (launch Parakeet.app once and let the
//! first-run fetch finish). This binary does NOT request mic permissions,
//! touch the clipboard, or synthesize keystrokes — it isolates the ASR
//! decode cost so the bench number is comparable across runs.
//!
//! Usage:
//!   bench_asr --wav bench/audio/5s.wav --reps 30 [--warmup-reps 3]

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use parakeet_rs::asr::Asr;
use parakeet_rs::performance::{
    self, next_session_id, PhaseTimer, PhaseTimerMode,
};
use parakeet_rs::settings::SettingsStore;
use parakeet_rs::warmup;

const DEFAULT_REPS: usize = 30;
const DEFAULT_WARMUP_REPS: usize = 3;

struct Args {
    wav: PathBuf,
    reps: usize,
    warmup_reps: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut wav: Option<PathBuf> = None;
    let mut reps: usize = DEFAULT_REPS;
    let mut warmup_reps: usize = DEFAULT_WARMUP_REPS;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--wav" => {
                wav = Some(PathBuf::from(
                    it.next().ok_or("--wav needs a path")?,
                ));
            }
            "--reps" => {
                reps = it
                    .next()
                    .ok_or("--reps needs a number")?
                    .parse()
                    .map_err(|e| format!("--reps: {e}"))?;
            }
            "--warmup-reps" => {
                warmup_reps = it
                    .next()
                    .ok_or("--warmup-reps needs a number")?
                    .parse()
                    .map_err(|e| format!("--warmup-reps: {e}"))?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    let wav = wav.ok_or("--wav is required")?;
    Ok(Args {
        wav,
        reps,
        warmup_reps,
    })
}

fn print_usage() {
    eprintln!(
        "usage: bench_asr --wav PATH [--reps N] [--warmup-reps N]\n\
         \n\
         Runs the loaded Parakeet recognizer over WAV PATH `--reps` times,\n\
         emitting one `phase_timer` log line per iteration on stderr.\n\
         `--warmup-reps` decodes are run first and not recorded — they pay\n\
         the CoreML graph-compile cost so steady-state numbers are clean.\n\
         \n\
         The model must already be downloaded (launch Parakeet.app once\n\
         to trigger the first-run fetch)."
    );
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    if let Err(e) = run(&args) {
        eprintln!("bench_asr failed: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: &Args) -> anyhow::Result<()> {
    let store = SettingsStore::new()?;
    if !store.model_present() {
        anyhow::bail!(
            "ASR model not present at {}. Launch Parakeet.app once so it can \
             download the first-run model bundle.",
            store.encoder_path().display()
        );
    }

    let threads = performance::performance_core_count();
    log::info!("loading Asr (threads={threads}, provider=coreml)");
    let asr = Asr::load(
        &store.encoder_path(),
        &store.decoder_path(),
        &store.joiner_path(),
        &store.tokens_path(),
        threads,
    )?;

    // CoreML graph compile happens on first inference. The aggregator
    // ignores the warmup reps so steady-state numbers aren't contaminated.
    log::info!("warming recognizer (page-touch + silent decode)");
    warmup::page_touch(&store.encoder_path())?;
    warmup::dummy_decode(&asr)?;

    let (samples, sample_rate) = read_wav_mono(&args.wav)?;
    let audio_s = samples.len() as f32 / sample_rate as f32;
    log::info!(
        "loaded {} ({audio_s:.3}s mono @ {sample_rate} Hz, {} samples)",
        args.wav.display(),
        samples.len()
    );

    let stem = args
        .wav
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".into());

    // Warmup reps: emit phase_timer lines but tagged so the aggregator
    // can drop them. Using session_id with a `warmup-` prefix keeps the
    // log file self-describing.
    for i in 0..args.warmup_reps {
        run_one(&asr, &samples, sample_rate, audio_s, &format!("warmup-{stem}-r{i:03}"))?;
    }
    // Measured reps. session_id has no `warmup-` prefix → aggregator counts it.
    for i in 0..args.reps {
        run_one(&asr, &samples, sample_rate, audio_s, &format!("bench-{stem}-r{i:03}"))?;
    }
    Ok(())
}

fn run_one(
    asr: &Asr,
    samples: &[f32],
    sample_rate: u32,
    audio_s: f32,
    session_label: &str,
) -> anyhow::Result<()> {
    // Combine the label with a unique counter so the timer log can be
    // grouped without collisions across runs.
    let sid = format!("{session_label}-{}", next_session_id());
    let mut t = PhaseTimer::start(PhaseTimerMode::Bench, sid);
    // The WAV is already in hand; capture and VAD collapsed into t0.
    t.mark_capture_end(audio_s);
    t.mark_vad_endpoint();
    t.mark_asr_start();
    let _decoded = asr.recognize(samples, sample_rate)?;
    t.mark_asr_done();
    // No paste in bench mode — mark it equal to asr_done so the
    // `dur_post_endpoint_ms` field cleanly reads as "ASR-only latency".
    t.mark_paste_done();
    t.emit();
    Ok(())
}

/// Read a 16-bit PCM WAV (mono or stereo) into f32 samples in [-1, 1].
/// Folds multi-channel input to mono by averaging. Refuses anything other
/// than 16-bit PCM, since the bench fixtures are generated by `afconvert
/// -d LEI16@... -c 1` (single-channel little-endian 16-bit).
fn read_wav_mono(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    use std::io::Read;
    let file = File::open(path)
        .map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;
    let mut r = BufReader::new(file);

    let mut header = [0u8; 12];
    r.read_exact(&mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        anyhow::bail!("not a RIFF/WAVE file: {}", path.display());
    }

    // Cap per-chunk allocation: a bench fixture > 1 GiB indicates a
    // corrupt header (raw u32 sizes from disk are otherwise unbounded).
    const MAX_CHUNK_BYTES: usize = 1024 * 1024 * 1024;

    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut data: Vec<u8> = Vec::new();

    loop {
        let mut chunk_hdr = [0u8; 8];
        match r.read_exact(&mut chunk_hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let id = &chunk_hdr[0..4];
        let size = u32::from_le_bytes([chunk_hdr[4], chunk_hdr[5], chunk_hdr[6], chunk_hdr[7]]) as usize;
        if size > MAX_CHUNK_BYTES {
            anyhow::bail!(
                "WAV chunk size {size} exceeds {MAX_CHUNK_BYTES} byte cap (corrupt header?)"
            );
        }

        if id == b"fmt " {
            let mut fmt = vec![0u8; size];
            r.read_exact(&mut fmt)?;
            let fmt_tag = u16::from_le_bytes([fmt[0], fmt[1]]);
            if fmt_tag != 1 {
                anyhow::bail!("only PCM (fmt tag 1) supported, got {fmt_tag}");
            }
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if id == b"data" {
            data.resize(size, 0);
            r.read_exact(&mut data)?;
        } else {
            // Skip unknown chunks (e.g. LIST, JUNK).
            let mut skip = vec![0u8; size];
            r.read_exact(&mut skip)?;
        }
    }

    if bits_per_sample != 16 {
        anyhow::bail!(
            "only 16-bit PCM supported, got {bits_per_sample}-bit \
             (regenerate with `afconvert -d LEI16@RATE`)"
        );
    }
    if channels == 0 || sample_rate == 0 || data.is_empty() {
        anyhow::bail!("WAV missing fmt or data chunk");
    }

    let frame_bytes = 2 * channels as usize;
    let frames = data.len() / frame_bytes;
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let frame = &data[i * frame_bytes..(i + 1) * frame_bytes];
        let mut sum = 0i32;
        for c in 0..channels as usize {
            let s = i16::from_le_bytes([frame[2 * c], frame[2 * c + 1]]) as i32;
            sum += s;
        }
        let avg = sum as f32 / channels as f32;
        out.push(avg / i16::MAX as f32);
    }
    Ok((out, sample_rate))
}
