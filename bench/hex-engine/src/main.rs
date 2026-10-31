//! `bench_hex`: Hex's engine (transcribe.cpp, Parakeet Unified EN 0.6B Q8_0
//! GGUF on Metal) on the gold corpus and the Hold loopback. Developer-only;
//! see `scripts/bench-hex.sh` and the "Hex engine" section of
//! `docs/asr/PERF.md`.

mod engine;
mod gold;
mod hold;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, bail, Result};

use gold::{DecodePath, GoldArgs};
use hold::HoldArgs;

const USAGE: &str = "usage:
  bench_hex gold --model GGUF --gold MANIFEST --audio-dir DIR --json-out FILE
                 [--repetitions N] [--path offline|stream]
  bench_hex hold --model GGUF --wav FILE [--device NAME] [--reps N]
                 [--warmup-reps N] [--path offline|stream]";

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("bench_hex: {error:#}");
            ExitCode::from(2)
        }
    }
}

struct Flags(Vec<String>);

impl Flags {
    fn take(&mut self, name: &str) -> Option<String> {
        let index = self.0.iter().position(|arg| arg == name)?;
        self.0.remove(index);
        (index < self.0.len()).then(|| self.0.remove(index))
    }

    fn required(&mut self, name: &str) -> Result<String> {
        self.take(name)
            .ok_or_else(|| anyhow!("{name} is required\n{USAGE}"))
    }

    fn number(&mut self, name: &str, default: usize) -> Result<usize> {
        self.take(name).map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|_| anyhow!("{name} expects a number"))
        })
    }

    fn finish(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            bail!("unexpected arguments {:?}\n{USAGE}", self.0)
        }
    }
}

fn run() -> Result<bool> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        bail!("{USAGE}");
    }
    let command = argv.remove(0);
    let mut flags = Flags(argv);
    let path = DecodePath::parse(&flags.take("--path").unwrap_or_else(|| "offline".into()))?;
    let model = PathBuf::from(flags.required("--model")?);
    match command.as_str() {
        "gold" => {
            let args = GoldArgs {
                model,
                gold: flags.required("--gold")?.into(),
                audio_dir: flags.required("--audio-dir")?.into(),
                json_out: flags.required("--json-out")?.into(),
                repetitions: flags.number("--repetitions", 10)?,
                path,
            };
            flags.finish()?;
            gold::run(&args)
        }
        "hold" => {
            let args = HoldArgs {
                model,
                wav: flags.required("--wav")?.into(),
                device: flags
                    .take("--device")
                    .unwrap_or_else(|| "BlackHole 2ch".into()),
                reps: flags.number("--reps", 30)?,
                warmup_reps: flags.number("--warmup-reps", 2)?,
                path,
            };
            flags.finish()?;
            hold::run(&args).map(|()| true)
        }
        other => bail!("unknown command {other:?}\n{USAGE}"),
    }
}
