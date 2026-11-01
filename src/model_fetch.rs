//! First-run download and integrity verification for the Rust-managed model
//! artifacts: Parakeet TDT 0.6B v3 int8, Parakeet Unified Core ML and Silero
//! VAD.
//!
//! Every artifact has an immutable upstream identity (revision where the host
//! supports one, expected byte length, and SHA-256). Downloads are hashed while
//! streaming into a `.part` file and are renamed only after the digest matches.
//! Existing files are hashed once, then a size/mtime/ctime/inode cache beside
//! the artifact makes later launches a metadata-only check.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

#[cfg(test)]
const SHERPA_REVISION: &str = "2bda32ec70b097a55adaa07d9a7173915b43cc78";
#[cfg(test)]
const HF_REPO: &str = concat!(
    "https://huggingface.co/csukuangfj/",
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/",
    "2bda32ec70b097a55adaa07d9a7173915b43cc78"
);
const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

#[derive(Clone, Copy, Debug)]
struct Artifact {
    label: &'static str,
    url: &'static str,
    sha256: &'static str,
    size: u64,
}

const ASR_ARTIFACTS: &[Artifact] = &[
    Artifact {
        label: "tokens.txt",
        url: concat!(
            "https://huggingface.co/csukuangfj/",
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/",
            "2bda32ec70b097a55adaa07d9a7173915b43cc78/tokens.txt"
        ),
        sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
        size: 93_939,
    },
    Artifact {
        label: "decoder.int8.onnx",
        url: concat!(
            "https://huggingface.co/csukuangfj/",
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/",
            "2bda32ec70b097a55adaa07d9a7173915b43cc78/decoder.int8.onnx"
        ),
        sha256: "179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e",
        size: 11_845_275,
    },
    Artifact {
        label: "joiner.int8.onnx",
        url: concat!(
            "https://huggingface.co/csukuangfj/",
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/",
            "2bda32ec70b097a55adaa07d9a7173915b43cc78/joiner.int8.onnx"
        ),
        sha256: "3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3",
        size: 6_355_277,
    },
    Artifact {
        label: "encoder.int8.onnx",
        url: concat!(
            "https://huggingface.co/csukuangfj/",
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/",
            "2bda32ec70b097a55adaa07d9a7173915b43cc78/encoder.int8.onnx"
        ),
        sha256: "acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247",
        size: 652_184_281,
    },
];

const SILERO_VAD: Artifact = Artifact {
    label: "silero_vad.onnx",
    url: SILERO_VAD_URL,
    sha256: "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    size: 643_854,
};

pub const COREML_REVISION: &str = "4252711f6f060f9a2f91e5f081a806d7f45eebd8";

macro_rules! coreml_artifact {
    ($path:literal, $sha256:literal, $size:literal) => {
        Artifact {
            label: $path,
            url: concat!(
                "https://huggingface.co/FluidInference/",
                "parakeet-unified-en-0.6b-coreml/resolve/",
                "4252711f6f060f9a2f91e5f081a806d7f45eebd8/",
                $path
            ),
            sha256: $sha256,
            size: $size,
        }
    };
}

/// Exact offline int8 truth pack consumed by `UnifiedAsrManager`.
const COREML_ARTIFACTS: &[Artifact] = &[
    coreml_artifact!(
        "config.json",
        "6cbe6c76445410c5c6debf3d44c8c3b75e9966bf09bba5cd138c2378c62120f6",
        1_355
    ),
    coreml_artifact!(
        "metadata.json",
        "2b26a96b76fe1f7a04d3e867f50c75d6ce5dd1650d0dbcd4c35b591b22305f0e",
        1_046
    ),
    coreml_artifact!(
        "vocab.json",
        "e1a7bff4f5df133c0f4ad47b8e43c96f6bf1865d99126a4c4725ef51d0108bec",
        15_088
    ),
    coreml_artifact!(
        "parakeet_unified_decoder.mlmodelc/analytics/coremldata.bin",
        "9ae70f6559989f88b856b326e59315798f9f0d08207a19fcc2dd3287a30088a5",
        243
    ),
    coreml_artifact!(
        "parakeet_unified_decoder.mlmodelc/coremldata.bin",
        "ce99c4488840fc463d59f8d4d6d2a9e8ceae8138ead51e3c265dde4d2ba4a0e9",
        560
    ),
    coreml_artifact!(
        "parakeet_unified_decoder.mlmodelc/model.mil",
        "6e60965b89c93943aa2be2d991c2461108145851fde05e1d048223a32d4cb20d",
        13_102
    ),
    coreml_artifact!(
        "parakeet_unified_decoder.mlmodelc/weights/weight.bin",
        "96f990461a5986d5e7309ad1a0f36084fbf0f4b28aec35948f8b8d0dcbf8599e",
        14_429_952
    ),
    coreml_artifact!(
        "parakeet_unified_encoder_int8.mlmodelc/analytics/coremldata.bin",
        "57e116a9d5765e39c0cdf754137ab744ddae34d9c6d68a5fdcad6600ae3a7b6b",
        243
    ),
    coreml_artifact!(
        "parakeet_unified_encoder_int8.mlmodelc/coremldata.bin",
        "54f533d30343d5e62b324a0691e4c262a6768b07b6e88e7aa14c617a2baba8a3",
        492
    ),
    coreml_artifact!(
        "parakeet_unified_encoder_int8.mlmodelc/model.mil",
        "c1c5d71c6cbf4d35bba08458746bde3640da7b1b444e1229a269393a58222c10",
        1_110_902
    ),
    coreml_artifact!(
        "parakeet_unified_encoder_int8.mlmodelc/weights/weight.bin",
        "f984b81590a4deae041ae20fbab8981c2d2a5b528b2ac81fae81c432633535c6",
        595_051_904
    ),
    coreml_artifact!(
        "parakeet_unified_joint_decision_single_step.mlmodelc/analytics/coremldata.bin",
        "163877ad14af97ec4107cd854fd1c6d336ee5d40ad25a657cc764fb763f452f5",
        243
    ),
    coreml_artifact!(
        "parakeet_unified_joint_decision_single_step.mlmodelc/coremldata.bin",
        "68a081570a48b52ec9379e153bd56748a5408a50be16767601563f231eaeff03",
        556
    ),
    coreml_artifact!(
        "parakeet_unified_joint_decision_single_step.mlmodelc/model.mil",
        "03c21096090bcd0b71c896c5ae0eb815db31a91c6676f572a7868eee4299abe3",
        9_611
    ),
    coreml_artifact!(
        "parakeet_unified_joint_decision_single_step.mlmodelc/weights/weight.bin",
        "06831afa6d1beb0c0b10350ebf7886bc37638e951d14e738d7e06fbd2a05012f",
        3_446_978
    ),
];

/// Stable digest of the reviewed Core ML manifest, not a rehash of the 614 MB
/// model on every launch. `ensure_coreml_model` verifies the corresponding
/// bytes before a profile keyed by this identity can be used.
pub fn coreml_artifact_digest() -> String {
    let mut hasher = Sha256::new();
    hasher.update(COREML_REVISION.as_bytes());
    hasher.update(b"\n");
    for artifact in COREML_ARTIFACTS {
        hasher.update(artifact.label.as_bytes());
        hasher.update(b"\0");
        hasher.update(artifact.size.to_le_bytes());
        hasher.update(artifact.sha256.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

/// Offline encoders compiled at windows shorter than the stock 15 s, which the
/// worker routes short utterances to (kata fgzt, ADR-0037). Each is converted
/// once from the same NVIDIA checkpoint by `scripts/build-bucket-encoder.py`
/// and published as one archive per window, because a Core ML bundle is a
/// directory and release assets are flat files.
///
/// Deliberately not part of [`coreml_artifact_digest`]: that digest keys the
/// tuning profile, and the profile does not yet select buckets (kata 4s2q).
#[derive(Clone, Copy, Debug)]
pub struct EncoderBucket {
    pub seconds: u32,
    /// Directory name the worker discovers in the model directory.
    bundle: &'static str,
    archive: Artifact,
    /// Every regular file the archive holds, labelled relative to `bundle`.
    files: &'static [Artifact],
}

macro_rules! encoder_bucket {
    (
        $seconds:literal, $bundle:literal, $archive_sha256:literal, $archive_size:literal,
        [$(($path:literal, $sha256:literal, $size:literal)),+ $(,)?]
    ) => {
        EncoderBucket {
            seconds: $seconds,
            bundle: $bundle,
            archive: Artifact {
                label: concat!($bundle, ".tar"),
                url: concat!(
                    "https://github.com/moorbrook/parakeet-rs/releases/download/",
                    "encoder-buckets-v1/",
                    $bundle,
                    ".tar"
                ),
                sha256: $archive_sha256,
                size: $archive_size,
            },
            files: &[$(Artifact {
                label: $path,
                url: concat!(
                    "https://github.com/moorbrook/parakeet-rs/releases/download/",
                    "encoder-buckets-v1/",
                    $bundle,
                    ".tar"
                ),
                sha256: $sha256,
                size: $size,
            }),+],
        }
    };
}

pub const ENCODER_BUCKETS: &[EncoderBucket] = &[
    encoder_bucket!(
        2,
        "parakeet_unified_encoder_2s_int8.mlmodelc",
        "4113921d6e527e98fee315f2fdc12c23aaae917b987bd6309e8e3674790fea78",
        587_972_608,
        [
            (
                "analytics/coremldata.bin",
                "33c3b4145f8bdf8c2754cb9a875a3775454fb1a95fa0ec6ea247d44ef93ca631",
                243
            ),
            (
                "coremldata.bin",
                "3d0cdc36d3ef05e625fa7da6417bdfdac9bafed38fb4d1a7442bdd44aa00d647",
                491
            ),
            (
                "metadata.json",
                "aa09fdec103068ef57292b22c6306081a58f7d2378fc632a63fed93fee3e5342",
                2_965
            ),
            (
                "model.mil",
                "ecdb3bc6e24a438acd950249a66c43fddf17e260a040eef3152704361c5ed63f",
                888_271
            ),
            (
                "weights/weight.bin",
                "76911797d2d0eb3663f82a04ba532d913785e61e3fa1caf6cc3d727f4ee28a53",
                587_076_544
            ),
        ]
    ),
    encoder_bucket!(
        5,
        "parakeet_unified_encoder_5s_int8.mlmodelc",
        "7ffac58a220dae55ea321bbefa161339f84f3f42a64a259d8bf92b99378216de",
        589_814_784,
        [
            (
                "analytics/coremldata.bin",
                "ffccf8f64035ede4f95b2d9b89022ccef10adbf63229abce1babc0414f49de02",
                243
            ),
            (
                "coremldata.bin",
                "144b14f469f210ab52ab55b9d44d96f34f3710b3f4c6860011c0d0a9d14ca5cd",
                491
            ),
            (
                "metadata.json",
                "6ad1d20b9701bd7c9c2e3136e225d4b243577e6680c44bdde87a652e4f1a62e1",
                2_965
            ),
            (
                "model.mil",
                "9b69788b1141f7ea747aeefcabe20ce9bf5ac15bfa01310cd73ca302144aad10",
                911_233
            ),
            (
                "weights/weight.bin",
                "8208bfa068b9124310625bca7de635f56ebeee3d0a41ef05fa16053a51a8022e",
                588_895_360
            ),
        ]
    ),
    encoder_bucket!(
        8,
        "parakeet_unified_encoder_8s_int8.mlmodelc",
        "a093946f242040214c8c8fe0df6e86abd6ad62fde6dbb33567b665b1c4a248aa",
        591_728_640,
        [
            (
                "analytics/coremldata.bin",
                "c420c5b321cec4d332eac0a9cad920ea8bf5ea9307456f7c16db1ee817e3a66b",
                243
            ),
            (
                "coremldata.bin",
                "bec1d60006b8e5a9b334b7e54a660550fcd260fd3eb075eb2e39bbdb5903e228",
                491
            ),
            (
                "metadata.json",
                "bd25da9b3691ed564cd07e87dd5cf6c39203f51cf8339c16b920dc231efa989a",
                2_967
            ),
            (
                "model.mil",
                "07c8f63c4382e59f380b3c650ace56745de80d20c070cee58c2438ca550f1f78",
                952_797
            ),
            (
                "weights/weight.bin",
                "cf0862b953e4fe0c01588edfc7795c03ff1f34eb5ccae7d1a1c6e291b7e8893b",
                590_767_872
            ),
        ]
    ),
];

/// Whether [`ensure_encoder_buckets`] may go to the network.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BucketFetch {
    /// Only report bundles already present and intact. The startup path uses
    /// this so a missing bucket never delays the first dictation.
    VerifyOnly,
    /// Download and unpack any bucket that is missing or corrupt.
    Download,
}

/// Parakeet TDT triplet + tokens, all relative to the per-model dir.
/// `pub(crate)` so `settings.rs`'s tests can cross-check that this list
/// stays in sync with the four files `SettingsStore::model_present()`
/// gates startup on.
#[cfg(test)]
pub(crate) const ASR_FILES: &[&str] = &[
    "tokens.txt",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "encoder.int8.onnx",
];

const CACHE_SCHEMA_VERSION: u8 = 1;
static CACHE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub enum Progress {
    /// Status text — drives the menu-bar header label.
    Status(String),
    /// Streaming chunk update — fires at most ~5 Hz while bytes flow.
    Chunk {
        file: String,
        bytes: u64,
        total: u64,
        fraction: f32,
    },
}

pub type ProgressFn = Arc<dyn Fn(Progress) + Send + Sync + 'static>;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct VerificationCache {
    schema_version: u8,
    sha256: String,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
    device: u64,
    inode: u64,
}

impl VerificationCache {
    fn from_metadata(artifact: Artifact, metadata: &fs::Metadata) -> Self {
        Self {
            schema_version: CACHE_SCHEMA_VERSION,
            sha256: artifact.sha256.to_string(),
            size: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }

    fn matches(&self, artifact: Artifact, metadata: &fs::Metadata) -> bool {
        self == &Self::from_metadata(artifact, metadata)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ExistingVerification {
    Missing,
    CacheHit,
    Hashed,
    DigestMismatch {
        actual_sha256: Option<String>,
        actual_size: u64,
    },
}

pub async fn ensure_model(
    model_dir: &Path,
    vad_path: &Path,
    on_progress: ProgressFn,
) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;
    if let Some(parent) = vad_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    on_progress(Progress::Status("Checking model integrity…".to_string()));
    let mut downloaded = false;
    for artifact in ASR_ARTIFACTS {
        downloaded |= ensure_artifact(*artifact, &model_dir.join(artifact.label), &on_progress)
            .await
            .with_context(|| format!("ensuring {}", artifact.label))?;
    }
    downloaded |= ensure_artifact(SILERO_VAD, vad_path, &on_progress)
        .await
        .context("ensuring silero_vad.onnx")?;

    let status = if downloaded {
        "Model downloaded and verified."
    } else {
        "Model verified."
    };
    on_progress(Progress::Status(status.to_string()));
    Ok(())
}

/// Verify or download the pinned Core ML truth pack before the Swift worker is
/// allowed to load it. This deliberately bypasses FluidAudio's mutable-main,
/// layout-only cache check while continuing to use FluidAudio for inference.
pub async fn ensure_coreml_model(model_dir: &Path, on_progress: ProgressFn) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;
    on_progress(Progress::Status(
        "Checking optimized model integrity…".to_string(),
    ));
    let mut downloaded = false;
    for artifact in COREML_ARTIFACTS {
        downloaded |= ensure_artifact(*artifact, &model_dir.join(artifact.label), &on_progress)
            .await
            .with_context(|| format!("ensuring Core ML {}", artifact.label))?;
    }
    let status = if downloaded {
        "Optimized model downloaded and verified."
    } else {
        "Optimized model verified."
    };
    on_progress(Progress::Status(status.to_string()));
    Ok(())
}

/// Hidden directory beside the bundles where archives land and unpack. A
/// bundle reaches the model directory, where the worker looks for it, only by
/// one rename after every file in it has matched the manifest.
const BUCKET_STAGING_DIR: &str = ".encoder-buckets-staging";

/// Verify every encoder bucket and, with [`BucketFetch::Download`], fetch the
/// missing or corrupt ones. Returns the windows whose bundle is intact,
/// ascending.
///
/// Never fails: a bucket that cannot be verified or fetched is left out, and
/// the worker sends those utterances to the stock 15 s encoder. After the
/// first failed download no further bucket is fetched in the same call, so an
/// unreachable host costs one request rather than one per bucket.
pub async fn ensure_encoder_buckets(
    model_dir: &Path,
    fetch: BucketFetch,
    on_progress: ProgressFn,
) -> Vec<u32> {
    ensure_buckets(ENCODER_BUCKETS, model_dir, fetch, &on_progress).await
}

async fn ensure_buckets(
    buckets: &[EncoderBucket],
    model_dir: &Path,
    fetch: BucketFetch,
    on_progress: &ProgressFn,
) -> Vec<u32> {
    let mut ready = Vec::new();
    let mut may_download = fetch == BucketFetch::Download;
    for bucket in buckets {
        let final_dir = model_dir.join(bucket.bundle);
        if verify_bundle(*bucket, &final_dir).await {
            ready.push(bucket.seconds);
            continue;
        }
        if let Err(error) = remove_bundle(&final_dir).await {
            log::warn!("cannot remove damaged {}: {error:#}", final_dir.display());
            continue;
        }
        if !may_download {
            continue;
        }
        match fetch_bucket(*bucket, model_dir, on_progress).await {
            Ok(()) => ready.push(bucket.seconds),
            Err(error) => {
                log::warn!(
                    "{} s encoder bucket unavailable, using the 15 s encoder: {error:#}",
                    bucket.seconds
                );
                may_download = false;
            }
        }
    }
    ready.sort_unstable();
    ready
}

async fn fetch_bucket(
    bucket: EncoderBucket,
    model_dir: &Path,
    on_progress: &ProgressFn,
) -> Result<()> {
    let staging = model_dir.join(BUCKET_STAGING_DIR);
    let prepare = staging.clone();
    tokio::task::spawn_blocking(move || prepare_staging(&prepare))
        .await
        .context("staging preparer panicked")??;
    let archive_path = staging.join(bucket.archive.label);
    ensure_artifact(bucket.archive, &archive_path, on_progress)
        .await
        .with_context(|| format!("ensuring {}", bucket.archive.label))?;

    on_progress(Progress::Status(format!(
        "Unpacking {} s encoder…",
        bucket.seconds
    )));
    let unpack_archive = archive_path.clone();
    let unpack_staging = staging.clone();
    let unpacked = tokio::task::spawn_blocking(move || {
        unpack_bundle(bucket, &unpack_archive, &unpack_staging)
    })
    .await
    .context("bucket unpacker panicked")?;
    // The archive matched its digest, so an archive that does not unpack to
    // the manifest is wrong at the source; keeping it would only repeat this.
    if let Err(error) = unpacked {
        discard_artifact(&archive_path).await?;
        return Err(error);
    }
    let unpacked_dir = staging.join(bucket.bundle);
    if !verify_bundle(bucket, &unpacked_dir).await {
        discard_artifact(&archive_path).await?;
        remove_bundle(&unpacked_dir).await?;
        bail!(
            "{} unpacked to files that do not match the manifest",
            bucket.archive.label
        );
    }

    let final_dir = model_dir.join(bucket.bundle);
    tokio::fs::rename(&unpacked_dir, &final_dir)
        .await
        .with_context(|| format!("moving verified bundle to {}", final_dir.display()))?;
    sync_parent_directory(&final_dir).await?;
    discard_artifact(&archive_path).await?;
    log::info!("{} s encoder bucket installed", bucket.seconds);
    Ok(())
}

/// Make `staging` a real directory that only this user can enter. Unpacking
/// deletes and writes under it, so a symlink or file in its place is removed
/// (the link itself, never its target) rather than followed.
fn prepare_staging(staging: &Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    match fs::symlink_metadata(staging) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => fs::remove_file(staging)
            .with_context(|| format!("removing non-directory {}", staging.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("inspecting {}", staging.display()))
        }
    }
    match fs::DirBuilder::new().mode(0o700).create(staging) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).with_context(|| format!("creating {}", staging.display())),
    }
    fs::set_permissions(staging, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("restricting {}", staging.display()))
}

/// Unpack `archive` under `staging`, refusing any entry that is not a regular
/// file the manifest names inside the bucket's bundle.
fn unpack_bundle(bucket: EncoderBucket, archive: &Path, staging: &Path) -> Result<()> {
    remove_bundle_sync(&staging.join(bucket.bundle))?;
    let file = File::open(archive).with_context(|| format!("opening {}", archive.display()))?;
    let mut tar = tar::Archive::new(BufReader::with_capacity(1024 * 1024, file));
    tar.set_preserve_permissions(false);
    for entry in tar.entries().context("reading archive")? {
        let mut entry = entry.context("reading archive entry")?;
        let path = entry.path().context("archive entry path")?.into_owned();
        let label = path
            .strip_prefix(bucket.bundle)
            .ok()
            .and_then(Path::to_str)
            .ok_or_else(|| {
                anyhow!(
                    "archive entry {} is outside {}",
                    path.display(),
                    bucket.bundle
                )
            })?;
        match entry.header().entry_type() {
            tar::EntryType::Directory => {}
            tar::EntryType::Regular if bucket.files.iter().any(|f| f.label == label) => {}
            other => bail!("unexpected archive entry {} ({other:?})", path.display()),
        }
        if !entry
            .unpack_in(staging)
            .with_context(|| format!("unpacking {}", path.display()))?
        {
            bail!(
                "archive entry {} escapes the staging directory",
                path.display()
            );
        }
    }
    Ok(())
}

/// Whether every manifest file of `bucket` is present under `dir` and matches
/// its digest. Any problem, including an I/O error, reads as "not intact": the
/// caller's only choices are to use the bundle or not.
async fn verify_bundle(bucket: EncoderBucket, dir: &Path) -> bool {
    let dir = dir.to_path_buf();
    let verified = tokio::task::spawn_blocking(move || {
        if !dir.is_dir() {
            return false;
        }
        bucket.files.iter().all(|file| {
            let path = dir.join(file.label);
            match verify_existing_artifact_sync(*file, &path) {
                Ok(ExistingVerification::CacheHit | ExistingVerification::Hashed) => true,
                Ok(ExistingVerification::Missing) => {
                    log::warn!("{} is missing", path.display());
                    false
                }
                Ok(ExistingVerification::DigestMismatch {
                    actual_sha256,
                    actual_size,
                }) => {
                    log::warn!(
                        "{} does not match its manifest (size {actual_size}, sha256 {})",
                        path.display(),
                        actual_sha256.as_deref().unwrap_or("not computed")
                    );
                    false
                }
                Err(error) => {
                    log::warn!("cannot verify {}: {error:#}", path.display());
                    false
                }
            }
        })
    })
    .await;
    verified.unwrap_or(false)
}

async fn remove_bundle(dir: &Path) -> Result<()> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || remove_bundle_sync(&dir))
        .await
        .context("bundle remover panicked")?
}

/// Remove a bundle directory, or only the link when it is a symlink to one.
fn remove_bundle_sync(dir: &Path) -> Result<()> {
    let result = match fs::symlink_metadata(dir) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(dir),
        Ok(_) => fs::remove_file(dir),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => Err(error),
    };
    result.with_context(|| format!("removing {}", dir.display()))
}

async fn ensure_artifact(
    artifact: Artifact,
    dest: &Path,
    on_progress: &ProgressFn,
) -> Result<bool> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let verification_started = std::time::Instant::now();
    let verify_path = dest.to_path_buf();
    let verification =
        tokio::task::spawn_blocking(move || verify_existing_artifact_sync(artifact, &verify_path))
            .await
            .context("artifact verifier panicked")??;

    match verification {
        ExistingVerification::CacheHit => {
            log::debug!(
                "verified {} from metadata cache in {:.3} ms",
                artifact.label,
                verification_started.elapsed().as_secs_f64() * 1_000.0
            );
            return Ok(false);
        }
        ExistingVerification::Hashed => {
            log::info!(
                "verified {} with SHA-256 in {:.3} s",
                artifact.label,
                verification_started.elapsed().as_secs_f64()
            );
            return Ok(false);
        }
        ExistingVerification::Missing => {}
        ExistingVerification::DigestMismatch {
            actual_sha256,
            actual_size,
        } => {
            log::warn!(
                "discarding corrupt {} (size {actual_size}, sha256 {}); expected size {}, sha256 {}",
                dest.display(),
                actual_sha256.as_deref().unwrap_or("not computed"),
                artifact.size,
                artifact.sha256
            );
            discard_artifact(dest).await?;
        }
    }

    on_progress(Progress::Status(format!(
        "Downloading and verifying {}…",
        artifact.label
    )));
    download_to(artifact, dest, on_progress).await?;
    log::info!(
        "downloaded and SHA-256 verified {} in {:.3} s",
        artifact.label,
        verification_started.elapsed().as_secs_f64()
    );
    Ok(true)
}

fn verify_existing_artifact_sync(artifact: Artifact, dest: &Path) -> Result<ExistingVerification> {
    let metadata = match fs::metadata(dest) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ExistingVerification::Missing);
        }
        Err(error) => {
            return Err(error).with_context(|| format!("reading metadata for {}", dest.display()));
        }
    };
    if !metadata.is_file() {
        bail!(
            "model artifact path is not a regular file: {}",
            dest.display()
        );
    }
    if metadata.len() != artifact.size {
        return Ok(ExistingVerification::DigestMismatch {
            actual_sha256: None,
            actual_size: metadata.len(),
        });
    }

    if read_verification_cache(dest).is_some_and(|cache| cache.matches(artifact, &metadata)) {
        return Ok(ExistingVerification::CacheHit);
    }

    let before = VerificationCache::from_metadata(artifact, &metadata);
    let file = File::open(dest).with_context(|| format!("opening {}", dest.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let (actual_sha256, actual_size) =
        sha256_reader(&mut reader).with_context(|| format!("hashing {}", dest.display()))?;

    // Check both the opened inode and the current path. This refuses to cache a
    // digest if another process edited or replaced the file while it was read.
    let opened_metadata = reader
        .get_ref()
        .metadata()
        .with_context(|| format!("rechecking {}", dest.display()))?;
    let path_metadata =
        fs::metadata(dest).with_context(|| format!("rechecking path {}", dest.display()))?;
    let opened_identity = VerificationCache::from_metadata(artifact, &opened_metadata);
    let path_identity = VerificationCache::from_metadata(artifact, &path_metadata);
    if before != opened_identity || before != path_identity {
        bail!(
            "{} changed while its SHA-256 was being verified",
            dest.display()
        );
    }

    if actual_size != artifact.size || actual_sha256 != artifact.sha256 {
        return Ok(ExistingVerification::DigestMismatch {
            actual_sha256: Some(actual_sha256),
            actual_size,
        });
    }

    if let Err(error) = write_verification_cache(dest, &path_identity) {
        // The digest is authoritative; a cache failure only means the next
        // launch must hash again, so it must not make an otherwise safe model
        // unusable.
        log::warn!(
            "could not cache verification for {}: {error:#}",
            dest.display()
        );
    }
    Ok(ExistingVerification::Hashed)
}

fn sha256_reader(mut reader: impl Read) -> Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut size = 0_u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size += read as u64;
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}

async fn download_to(artifact: Artifact, dest: &Path, on_progress: &ProgressFn) -> Result<()> {
    let tmp = part_path(dest)?;
    remove_file_if_exists(&tmp).await?;

    let client = reqwest::Client::builder()
        .user_agent(concat!("parakeet-dictation/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let response = client.get(artifact.url).send().await?.error_for_status()?;

    let transfer = async {
        let mut file = tokio::fs::File::create(&tmp).await?;
        let mut stream = response.bytes_stream();
        let mut hasher = Sha256::new();
        let mut downloaded = 0_u64;
        let mut last_emit = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;
            if last_emit.elapsed() >= std::time::Duration::from_millis(200) {
                last_emit = std::time::Instant::now();
                on_progress(Progress::Chunk {
                    file: artifact.label.to_string(),
                    bytes: downloaded,
                    total: artifact.size,
                    fraction: (downloaded as f32 / artifact.size as f32).min(1.0),
                });
            }
        }
        file.flush().await?;
        file.sync_all().await?;
        drop(file);

        Ok::<_, anyhow::Error>((downloaded, format!("{:x}", hasher.finalize())))
    }
    .await;

    let (downloaded, actual_sha256) = match transfer {
        Ok(result) => result,
        Err(error) => {
            let _ = remove_file_if_exists(&tmp).await;
            return Err(error).with_context(|| format!("streaming {}", artifact.label));
        }
    };

    // Content-Length is intentionally not trusted or required. The pinned
    // length gives clearer diagnostics, while SHA-256 is the authority.
    if downloaded != artifact.size || actual_sha256 != artifact.sha256 {
        remove_file_if_exists(&tmp).await?;
        bail!(
            "{} integrity mismatch: got {downloaded} bytes, sha256 {actual_sha256}; expected {} bytes, sha256 {} (discarded)",
            artifact.label,
            artifact.size,
            artifact.sha256
        );
    }

    if let Err(error) = tokio::fs::rename(&tmp, dest).await {
        let _ = remove_file_if_exists(&tmp).await;
        return Err(error).context("renaming verified .part to final artifact");
    }
    sync_parent_directory(dest).await?;

    let cache_dest = dest.to_path_buf();
    let cache_result = tokio::task::spawn_blocking(move || {
        let metadata = fs::metadata(&cache_dest)?;
        let identity = VerificationCache::from_metadata(artifact, &metadata);
        write_verification_cache(&cache_dest, &identity)
    })
    .await
    .context("verification-cache writer panicked")?;
    if let Err(error) = cache_result {
        log::warn!(
            "could not cache verification for {}: {error:#}",
            dest.display()
        );
    }

    on_progress(Progress::Chunk {
        file: artifact.label.to_string(),
        bytes: downloaded,
        total: artifact.size,
        fraction: 1.0,
    });
    Ok(())
}

async fn discard_artifact(dest: &Path) -> Result<()> {
    remove_file_if_exists(dest).await?;
    remove_file_if_exists(&verification_cache_path(dest)?).await?;
    Ok(())
}

async fn remove_file_if_exists(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("removing {}", path.display())),
    }
}

fn part_path(dest: &Path) -> Result<PathBuf> {
    let filename = dest
        .file_name()
        .ok_or_else(|| anyhow!("artifact has no filename: {}", dest.display()))?;
    let mut part_name = filename.to_os_string();
    part_name.push(".part");
    Ok(dest.with_file_name(part_name))
}

fn verification_cache_path(dest: &Path) -> Result<PathBuf> {
    let filename = dest
        .file_name()
        .ok_or_else(|| anyhow!("artifact has no filename: {}", dest.display()))?;
    let mut cache_name = std::ffi::OsString::from(".");
    cache_name.push(filename);
    cache_name.push(".parakeet-verified.json");
    Ok(dest.with_file_name(cache_name))
}

fn read_verification_cache(dest: &Path) -> Option<VerificationCache> {
    let cache_path = match verification_cache_path(dest) {
        Ok(path) => path,
        Err(error) => {
            log::warn!("cannot resolve verification-cache path: {error:#}");
            return None;
        }
    };
    let raw = match fs::read(&cache_path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            log::warn!("cannot read {}: {error}", cache_path.display());
            return None;
        }
    };
    match serde_json::from_slice(&raw) {
        Ok(cache) => Some(cache),
        Err(error) => {
            log::warn!("ignoring malformed {}: {error}", cache_path.display());
            None
        }
    }
}

fn write_verification_cache(dest: &Path, cache: &VerificationCache) -> Result<()> {
    let cache_path = verification_cache_path(dest)?;
    let sequence = CACHE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut tmp_name = cache_path
        .file_name()
        .ok_or_else(|| anyhow!("verification cache has no filename"))?
        .to_os_string();
    tmp_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
    let tmp_path = cache_path.with_file_name(tmp_name);
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .with_context(|| format!("creating {}", tmp_path.display()))?;
        serde_json::to_writer_pretty(&mut file, cache)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp_path, &cache_path).with_context(|| {
            format!(
                "renaming verification cache {} to {}",
                tmp_path.display(),
                cache_path.display()
            )
        })?;
        sync_parent_directory_sync(&cache_path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

async fn sync_parent_directory(path: &Path) -> Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || sync_parent_directory_sync(&path))
        .await
        .context("directory sync task panicked")?
}

fn sync_parent_directory_sync(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("syncing directory {}", parent.display()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    const ABC: Artifact = Artifact {
        label: "abc.bin",
        url: "https://example.invalid/abc.bin",
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        size: 3,
    };

    #[test]
    fn sha256_reader_matches_standard_vector() {
        let (digest, size) = sha256_reader(Cursor::new(b"abc")).unwrap();
        assert_eq!(digest, ABC.sha256);
        assert_eq!(size, ABC.size);
    }

    #[test]
    fn first_verification_hashes_then_unchanged_file_hits_cache() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join(ABC.label);
        fs::write(&artifact, b"abc").unwrap();

        assert_eq!(
            verify_existing_artifact_sync(ABC, &artifact).unwrap(),
            ExistingVerification::Hashed
        );
        assert!(verification_cache_path(&artifact).unwrap().is_file());
        assert_eq!(
            verify_existing_artifact_sync(ABC, &artifact).unwrap(),
            ExistingVerification::CacheHit
        );
    }

    #[test]
    fn changed_same_size_file_is_rehashed_and_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join(ABC.label);
        fs::write(&artifact, b"abc").unwrap();
        assert_eq!(
            verify_existing_artifact_sync(ABC, &artifact).unwrap(),
            ExistingVerification::Hashed
        );

        fs::write(&artifact, b"abd").unwrap();
        let result = verify_existing_artifact_sync(ABC, &artifact).unwrap();
        assert!(matches!(
            result,
            ExistingVerification::DigestMismatch {
                actual_sha256: Some(_),
                actual_size: 3
            }
        ));
    }

    #[test]
    fn malformed_cache_is_never_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join(ABC.label);
        fs::write(&artifact, b"abc").unwrap();
        assert_eq!(
            verify_existing_artifact_sync(ABC, &artifact).unwrap(),
            ExistingVerification::Hashed
        );
        fs::write(verification_cache_path(&artifact).unwrap(), b"not json").unwrap();
        assert_eq!(
            verify_existing_artifact_sync(ABC, &artifact).unwrap(),
            ExistingVerification::Hashed
        );
    }

    #[tokio::test]
    async fn discard_removes_corrupt_artifact_and_its_cache() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join(ABC.label);
        fs::write(&artifact, b"abc").unwrap();
        assert_eq!(
            verify_existing_artifact_sync(ABC, &artifact).unwrap(),
            ExistingVerification::Hashed
        );
        fs::write(&artifact, b"abd").unwrap();

        discard_artifact(&artifact).await.unwrap();
        assert!(!artifact.exists());
        assert!(!verification_cache_path(&artifact).unwrap().exists());
    }

    #[tokio::test]
    async fn missing_content_length_still_downloads_by_pinned_digest() {
        let (url, server) = serve_once_without_content_length(b"abc");
        let artifact = Artifact {
            url: Box::leak(url.into_boxed_str()),
            ..ABC
        };
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(artifact.label);
        let progress: ProgressFn = Arc::new(|_| {});

        download_to(artifact, &dest, &progress).await.unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"abc");
        assert_eq!(
            verify_existing_artifact_sync(artifact, &dest).unwrap(),
            ExistingVerification::CacheHit
        );
    }

    #[tokio::test]
    async fn corrupt_existing_file_is_deleted_and_refetched() {
        let (url, server) = serve_once_without_content_length(b"abc");
        let artifact = Artifact {
            url: Box::leak(url.into_boxed_str()),
            ..ABC
        };
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(artifact.label);
        fs::write(&dest, b"abd").unwrap();
        let progress: ProgressFn = Arc::new(|_| {});

        assert!(ensure_artifact(artifact, &dest, &progress).await.unwrap());
        server.join().unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"abc");
        assert!(!part_path(&dest).unwrap().exists());
    }

    #[test]
    fn artifact_manifest_matches_public_file_list() {
        let manifest: Vec<&str> = ASR_ARTIFACTS
            .iter()
            .map(|artifact| artifact.label)
            .collect();
        assert_eq!(manifest, ASR_FILES);
        assert!(ASR_ARTIFACTS.iter().all(|artifact| {
            artifact.url.starts_with(HF_REPO)
                && artifact.url.contains(SHERPA_REVISION)
                && artifact.sha256.len() == 64
                && artifact.size > 0
        }));
    }

    #[test]
    fn silero_manifest_pins_published_release_digest() {
        assert_eq!(SILERO_VAD.url, SILERO_VAD_URL);
        assert_eq!(SILERO_VAD.sha256.len(), 64);
        assert_eq!(SILERO_VAD.size, 643_854);
    }

    #[test]
    fn coreml_truth_pack_is_complete_revision_pinned_and_unique() {
        assert_eq!(COREML_ARTIFACTS.len(), 15);
        assert!(COREML_ARTIFACTS.iter().all(|artifact| {
            artifact.url.contains(COREML_REVISION)
                && artifact.url.ends_with(artifact.label)
                && artifact.sha256.len() == 64
                && artifact.size > 0
        }));
        let mut paths: Vec<&str> = COREML_ARTIFACTS
            .iter()
            .map(|artifact| artifact.label)
            .collect();
        paths.sort_unstable();
        paths.dedup();
        assert_eq!(paths.len(), COREML_ARTIFACTS.len());
        assert!(paths.contains(&"vocab.json"));
        assert!(paths.contains(&"parakeet_unified_encoder_int8.mlmodelc/weights/weight.bin"));
        assert!(paths
            .contains(&"parakeet_unified_joint_decision_single_step.mlmodelc/weights/weight.bin"));
        assert_eq!(coreml_artifact_digest().len(), 64);
        assert_eq!(coreml_artifact_digest(), coreml_artifact_digest());
    }

    #[tokio::test]
    async fn nested_artifact_creates_destination_directories() {
        let (url, server) = serve_once_without_content_length(b"abc");
        let artifact = Artifact {
            label: "model.mlmodelc/weights/weight.bin",
            url: Box::leak(url.into_boxed_str()),
            ..ABC
        };
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(artifact.label);
        let progress: ProgressFn = Arc::new(|_| {});

        assert!(ensure_artifact(artifact, &dest, &progress).await.unwrap());
        server.join().unwrap();
        assert_eq!(fs::read(dest).unwrap(), b"abc");
    }

    #[test]
    fn asr_files_list_has_no_duplicates() {
        let mut copy: Vec<&&str> = ASR_FILES.iter().collect();
        copy.sort();
        let len_before = copy.len();
        copy.dedup();
        assert_eq!(len_before, copy.len(), "duplicate entry in ASR_FILES");
    }

    #[test]
    fn encoder_buckets_are_the_2_5_8_release_set() {
        let windows: Vec<u32> = ENCODER_BUCKETS.iter().map(|b| b.seconds).collect();
        assert_eq!(windows, [2, 5, 8]);
        for bucket in ENCODER_BUCKETS {
            assert_eq!(
                bucket.bundle,
                format!("parakeet_unified_encoder_{}s_int8.mlmodelc", bucket.seconds)
            );
            assert_eq!(
                bucket.archive.url,
                format!(
                    "https://github.com/moorbrook/parakeet-rs/releases/download/\
                     encoder-buckets-v1/{}.tar",
                    bucket.bundle
                )
            );
            let mut labels: Vec<&str> = bucket.files.iter().map(|f| f.label).collect();
            labels.sort_unstable();
            assert_eq!(
                labels,
                [
                    "analytics/coremldata.bin",
                    "coremldata.bin",
                    "metadata.json",
                    "model.mil",
                    "weights/weight.bin"
                ]
            );
            let unpacked: u64 = bucket.files.iter().map(|f| f.size).sum();
            assert!(bucket.archive.size > unpacked, "{}", bucket.bundle);
            for artifact in bucket.files.iter().chain([&bucket.archive]) {
                assert_eq!(artifact.sha256.len(), 64);
                assert!(artifact.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            }
        }
    }

    #[test]
    fn bucket_digests_do_not_change_the_tuning_profile_key() {
        // The tuning profile is keyed by this digest; buckets must not
        // invalidate a profile until kata 4s2q makes the profile select them.
        // The value is the one main's manifest produced, and it matches the
        // `artifact_digest` of the profile already written on the M5 Pro.
        assert_eq!(
            coreml_artifact_digest(),
            "df2587d4ec1a768b78f11235e51298cc3d44c947b6f0a151729b9e94e7da0002"
        );
    }

    const BUNDLE: &str = "enc_2s.mlmodelc";
    const BUNDLE_FILES: &[(&str, &[u8])] = &[
        ("coremldata.bin", b"plan"),
        ("weights/weight.bin", b"weights"),
    ];

    fn tar_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (path, body) in entries {
            let mut header = tar::Header::new_ustar();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, format!("{BUNDLE}/{path}"), *body)
                .unwrap();
        }
        builder.into_inner().unwrap()
    }

    fn leaked_artifact(label: &str, url: &str, body: &[u8]) -> Artifact {
        let (sha256, size) = sha256_reader(Cursor::new(body)).unwrap();
        Artifact {
            label: Box::leak(label.to_string().into_boxed_str()),
            url: Box::leak(url.to_string().into_boxed_str()),
            sha256: Box::leak(sha256.into_boxed_str()),
            size,
        }
    }

    /// A bucket whose manifest pins `pinned_files` and whose archive pin is
    /// the digest of `archive`, served at `{base}/{seconds}.tar`.
    fn test_bucket(
        seconds: u32,
        base: &str,
        archive: &[u8],
        pinned_files: &[(&str, &[u8])],
    ) -> EncoderBucket {
        let url = format!("{base}/{seconds}.tar");
        let files: Vec<Artifact> = pinned_files
            .iter()
            .map(|(path, body)| leaked_artifact(path, &url, body))
            .collect();
        EncoderBucket {
            seconds,
            bundle: BUNDLE,
            archive: leaked_artifact(&format!("{BUNDLE}.tar"), &url, archive),
            files: Box::leak(files.into_boxed_slice()),
        }
    }

    /// Serves `routes` (path to body; absent paths get 404) until the test
    /// process exits, recording every requested path.
    fn serve(routes: Vec<(String, Vec<u8>)>) -> (String, Arc<parking_lot::Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requested = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let log = Arc::clone(&requested);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut request = [0_u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let line = String::from_utf8_lossy(&request[..read]);
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                log.lock().push(path.clone());
                match routes.iter().find(|(route, _)| *route == path) {
                    Some((_, body)) => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        stream.write_all(head.as_bytes()).unwrap();
                        stream.write_all(body).unwrap();
                    }
                    None => stream
                        .write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .unwrap(),
                }
            }
        });
        (base, requested)
    }

    fn no_progress() -> ProgressFn {
        Arc::new(|_| {})
    }

    #[tokio::test]
    async fn a_fetched_bucket_is_installed_only_after_every_file_verifies() {
        let archive = tar_of(BUNDLE_FILES);
        let (base, requested) = serve(vec![("/2.tar".to_string(), archive.clone())]);
        let bucket = test_bucket(2, &base, &archive, BUNDLE_FILES);
        let dir = tempfile::tempdir().unwrap();

        let ready =
            ensure_buckets(&[bucket], dir.path(), BucketFetch::Download, &no_progress()).await;

        assert_eq!(ready, [2]);
        assert_eq!(
            fs::read(dir.path().join(BUNDLE).join("weights/weight.bin")).unwrap(),
            b"weights"
        );
        assert!(!dir
            .path()
            .join(BUCKET_STAGING_DIR)
            .join(bucket.archive.label)
            .exists());
        // A second launch verifies from the metadata cache and fetches nothing.
        let ready =
            ensure_buckets(&[bucket], dir.path(), BucketFetch::Download, &no_progress()).await;
        assert_eq!(ready, [2]);
        assert_eq!(requested.lock().as_slice(), ["/2.tar"]);
    }

    #[tokio::test]
    async fn a_symlinked_staging_directory_is_replaced_and_its_target_left_alone() {
        use std::os::unix::fs::PermissionsExt;
        let archive = tar_of(BUNDLE_FILES);
        let (base, _) = serve(vec![("/2.tar".to_string(), archive.clone())]);
        let bucket = test_bucket(2, &base, &archive, BUNDLE_FILES);
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let foreign = elsewhere.path().join(BUNDLE).join("keep.txt");
        fs::create_dir_all(foreign.parent().unwrap()).unwrap();
        fs::write(&foreign, b"not ours").unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), dir.path().join(BUCKET_STAGING_DIR)).unwrap();

        let ready =
            ensure_buckets(&[bucket], dir.path(), BucketFetch::Download, &no_progress()).await;

        assert_eq!(ready, [2]);
        assert_eq!(fs::read(&foreign).unwrap(), b"not ours");
        assert_eq!(fs::read_dir(elsewhere.path()).unwrap().count(), 1);
        let staging = fs::symlink_metadata(dir.path().join(BUCKET_STAGING_DIR)).unwrap();
        assert!(staging.is_dir(), "staging is still a symlink");
        assert_eq!(staging.permissions().mode() & 0o777, 0o700);
    }

    #[tokio::test]
    async fn an_archive_with_the_wrong_digest_is_refused() {
        let archive = tar_of(BUNDLE_FILES);
        let mut tampered = archive.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        let (base, _) = serve(vec![("/2.tar".to_string(), tampered)]);
        let bucket = test_bucket(2, &base, &archive, BUNDLE_FILES);
        let dir = tempfile::tempdir().unwrap();

        let ready =
            ensure_buckets(&[bucket], dir.path(), BucketFetch::Download, &no_progress()).await;

        assert!(ready.is_empty());
        assert!(!dir.path().join(BUNDLE).exists());
        let staging = dir.path().join(BUCKET_STAGING_DIR);
        assert!(!staging.join(bucket.archive.label).exists());
        assert!(!part_path(&staging.join(bucket.archive.label))
            .unwrap()
            .exists());
    }

    #[tokio::test]
    async fn an_archive_whose_files_differ_from_the_manifest_is_refused() {
        // The archive matches its own pin, but one file inside it does not
        // match the per-file manifest.
        let archive = tar_of(&[
            ("coremldata.bin", b"plan"),
            ("weights/weight.bin", b"WEIGHTS"),
        ]);
        let (base, _) = serve(vec![("/2.tar".to_string(), archive.clone())]);
        let bucket = test_bucket(2, &base, &archive, BUNDLE_FILES);
        let dir = tempfile::tempdir().unwrap();

        let ready =
            ensure_buckets(&[bucket], dir.path(), BucketFetch::Download, &no_progress()).await;

        assert!(ready.is_empty());
        assert!(!dir.path().join(BUNDLE).exists());
        let staging = dir.path().join(BUCKET_STAGING_DIR);
        assert!(!staging.join(BUNDLE).exists());
        assert!(!staging.join(bucket.archive.label).exists());
    }

    #[tokio::test]
    async fn an_archive_entry_outside_the_manifest_is_refused() {
        let archive = tar_of(&[
            ("coremldata.bin", b"plan"),
            ("weights/weight.bin", b"weights"),
            ("extra.bin", b"x"),
        ]);
        let (base, _) = serve(vec![("/2.tar".to_string(), archive.clone())]);
        let bucket = test_bucket(2, &base, &archive, BUNDLE_FILES);
        let dir = tempfile::tempdir().unwrap();

        let ready =
            ensure_buckets(&[bucket], dir.path(), BucketFetch::Download, &no_progress()).await;

        assert!(ready.is_empty());
        assert!(!dir.path().join(BUNDLE).exists());
    }

    #[tokio::test]
    async fn a_missing_bucket_is_left_out_and_stops_further_downloads() {
        let archive = tar_of(BUNDLE_FILES);
        let (base, requested) = serve(vec![]);
        let missing = test_bucket(2, &base, &archive, BUNDLE_FILES);
        let after = EncoderBucket {
            seconds: 5,
            bundle: "enc_5s.mlmodelc",
            ..test_bucket(5, &base, &archive, BUNDLE_FILES)
        };
        let dir = tempfile::tempdir().unwrap();

        let ready = ensure_buckets(
            &[missing, after],
            dir.path(),
            BucketFetch::Download,
            &no_progress(),
        )
        .await;

        assert!(ready.is_empty());
        assert!(!dir.path().join(BUNDLE).exists());
        assert_eq!(requested.lock().as_slice(), ["/2.tar"]);
    }

    #[tokio::test]
    async fn verify_only_reports_intact_bundles_and_never_downloads() {
        let archive = tar_of(BUNDLE_FILES);
        let (base, requested) = serve(vec![("/2.tar".to_string(), archive.clone())]);
        let bucket = test_bucket(2, &base, &archive, BUNDLE_FILES);
        let dir = tempfile::tempdir().unwrap();

        let absent = ensure_buckets(
            &[bucket],
            dir.path(),
            BucketFetch::VerifyOnly,
            &no_progress(),
        )
        .await;
        assert!(absent.is_empty());
        assert!(requested.lock().is_empty());

        for (path, body) in BUNDLE_FILES {
            let dest = dir.path().join(BUNDLE).join(path);
            fs::create_dir_all(dest.parent().unwrap()).unwrap();
            fs::write(dest, body).unwrap();
        }
        let present = ensure_buckets(
            &[bucket],
            dir.path(),
            BucketFetch::VerifyOnly,
            &no_progress(),
        )
        .await;
        assert_eq!(present, [2]);
        assert!(requested.lock().is_empty());
    }

    #[tokio::test]
    async fn a_damaged_installed_bundle_is_removed_so_the_worker_cannot_load_it() {
        let archive = tar_of(BUNDLE_FILES);
        let bucket = test_bucket(2, "http://127.0.0.1:9", &archive, BUNDLE_FILES);
        let dir = tempfile::tempdir().unwrap();
        for (path, body) in BUNDLE_FILES {
            let dest = dir.path().join(BUNDLE).join(path);
            fs::create_dir_all(dest.parent().unwrap()).unwrap();
            fs::write(dest, body).unwrap();
        }
        fs::write(
            dir.path().join(BUNDLE).join("weights/weight.bin"),
            b"weightz",
        )
        .unwrap();

        let ready = ensure_buckets(
            &[bucket],
            dir.path(),
            BucketFetch::VerifyOnly,
            &no_progress(),
        )
        .await;

        assert!(ready.is_empty());
        assert!(!dir.path().join(BUNDLE).exists());
    }

    fn serve_once_without_content_length(body: &'static [u8]) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.write_all(body).unwrap();
            stream.flush().unwrap();
        });
        (format!("http://{address}/artifact"), server)
    }
}
