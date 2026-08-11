//! First-run download and integrity verification for the Rust-managed model
//! artifacts: Parakeet TDT 0.6B v3 int8, Silero VAD, and the optional polish
//! GGUF.
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
#[cfg(test)]
const POLISH_REVISION: &str = "e87f176479d0855a907a41277aca2f8ee7a09523";
/// Polish-pass GGUF (ADR-0018, amended to the 4B). MUST stay in sync
/// with `Settings::polish_model_path` — the filename at the end of
/// this URL is the filename the loader expects on disk.
pub(crate) const POLISH_GGUF_URL: &str = concat!(
    "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/",
    "e87f176479d0855a907a41277aca2f8ee7a09523/",
    "Qwen3.5-4B-Q6_K.gguf"
);

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

const POLISH_GGUF: Artifact = Artifact {
    label: "Qwen3.5-4B-Q6_K.gguf",
    url: POLISH_GGUF_URL,
    sha256: "fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66",
    size: 3_525_956_768,
};

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

/// Verify the optional polish GGUF on every load and download it on first use.
/// A verified metadata cache keeps later loads cheap; changed files are hashed
/// again before llama.cpp is allowed to open them.
pub async fn ensure_polish_model(dest: &Path, on_progress: ProgressFn) -> Result<()> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    on_progress(Progress::Status(
        "Checking polish model integrity…".to_string(),
    ));
    let downloaded = ensure_artifact(POLISH_GGUF, dest, &on_progress)
        .await
        .context("ensuring polish GGUF")?;
    let status = if downloaded {
        "Polish model downloaded and verified."
    } else {
        "Polish model verified."
    };
    on_progress(Progress::Status(status.to_string()));
    Ok(())
}

async fn ensure_artifact(
    artifact: Artifact,
    dest: &Path,
    on_progress: &ProgressFn,
) -> Result<bool> {
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
        .user_agent(concat!("parakeet-rs/", env!("CARGO_PKG_VERSION")))
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
    fn polish_gguf_url_is_revision_pinned() {
        assert!(POLISH_GGUF_URL.contains(POLISH_REVISION));
        assert_eq!(POLISH_GGUF.sha256.len(), 64);
        assert_eq!(POLISH_GGUF.size, 3_525_956_768);
        assert!(POLISH_GGUF_URL.ends_with(POLISH_GGUF.label));
    }

    #[test]
    fn asr_files_list_has_no_duplicates() {
        let mut copy: Vec<&&str> = ASR_FILES.iter().collect();
        copy.sort();
        let len_before = copy.len();
        copy.dedup();
        assert_eq!(len_before, copy.len(), "duplicate entry in ASR_FILES");
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
