//! Optional rewrite of a transcript through the OpenCode service that is
//! already running on this Mac. Parakeet never starts OpenCode and never
//! handles a provider key: it reads the service's own owner-only loopback
//! registration, confirms the registered process answers, and posts one
//! stateless generation. OpenCode answers with whichever model it is
//! configured for. Absence, a slow answer, an error or an empty answer all
//! paste the local transcript unchanged. See ADR-0036.
//!
//! The discovery and wire protocol follow Hex (github.com/anomalyco/hex,
//! MIT): `src/opencode_service.rs` (registration, `/api/info` check,
//! loopback-only HTTP/1.1 over `TcpStream`) and `src/dictation_processor.rs`
//! (`/api/experimental/generate` with the `/api/generate` fallback, and the
//! `{"data":{"text"}}` / `{"message"}` answer shapes).

use std::fs;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Whole-rewrite budget: discovery, `/api/info` and the generation share it.
/// The hotkey is ignored while the rewrite runs, so this is also the longest
/// a rewrite can delay the next dictation. Measured on this Mac
/// (2026-10-31, OpenCode 2.0.32, a 25-word transcript with this prompt,
/// ten runs each; ADR-0036): `zai-coding-plan/glm-5.3-flash` with the `low`
/// variant answered in 0.98–3.38 s, median 1.45 s; without a variant it
/// took 6.7–26.6 s, so a reasoning model needs its low variant to fit. 5 s
/// covers the slowest low-variant run and is a sixth of Hex's 30 s
/// post-processing default (`src/app_settings.rs`), which is sized for
/// long prompts, not dictation.
pub const DEADLINE: Duration = Duration::from_secs(5);

const MAX_REGISTRATION_BYTES: u64 = 64 * 1024;
const MAX_REGISTRATIONS: usize = 16;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
/// How often a blocked read or write re-checks the deadline.
const READ_SLICE: Duration = Duration::from_millis(50);
/// Newest one-shot route first; older OpenCode builds serve only the second.
const GENERATE_ROUTES: [&str; 2] = ["/api/experimental/generate", "/api/generate"];

const PROMPT: &str = "You clean up a speech-to-text transcript for direct insertion at the cursor. Return only the text that should be pasted. Do not add an explanation, label, alternative, quotes, or Markdown.\n\
\n\
Rules:\n\
1. Remove filler words (um, uh, er, you know, I mean when used as filler).\n\
2. Fix punctuation and capitalisation.\n\
3. Honour spoken edits: \"new paragraph\" or \"new line\" becomes a line break; \"scratch that\" removes the preceding sentence.\n\
4. Preserve the speaker's meaning, tone and wording. Do not paraphrase, summarise, expand or answer the text.\n\
5. Keep technical terms, names and code exactly as spoken.";

/// What a rewrite attempt produced. `Raw` carries the reason for the log.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Rewritten(String),
    Raw(String),
}

impl Outcome {
    /// The text to paste: the rewrite, or `raw` unchanged.
    pub fn text(self, raw: &str) -> String {
        match self {
            Self::Rewritten(text) => text,
            Self::Raw(_) => raw.to_string(),
        }
    }
}

/// Which OpenCode model answers: `key` is `provider/model`, or empty for the
/// service's default; `variant` (for example `low`) is optional.
#[derive(Clone, Copy)]
pub struct ModelChoice<'a> {
    pub key: &'a str,
    pub variant: &'a str,
}

/// Rewrite `raw` through the running OpenCode service within [`DEADLINE`].
/// `vocabulary` is the user's custom terms, passed so the rewrite keeps
/// their spelling.
pub fn rewrite(raw: &str, model: ModelChoice<'_>, vocabulary: &[String]) -> Outcome {
    if raw.trim().is_empty() {
        return Outcome::Raw("empty transcript".to_string());
    }
    let started = Instant::now();
    let Some(service) = state_dir().and_then(|dir| find_in(&dir, started, DEADLINE)) else {
        return Outcome::Raw("OpenCode isn't running".to_string());
    };
    rewrite_with(&service, raw, model, vocabulary, started, DEADLINE)
}

/// The decision, against a known service: only a 200 answer carrying
/// non-empty text is a rewrite; everything else keeps the raw transcript.
fn rewrite_with(
    service: &Service,
    raw: &str,
    model: ModelChoice<'_>,
    vocabulary: &[String],
    started: Instant,
    deadline: Duration,
) -> Outcome {
    match generate(service, &prompt(raw, vocabulary), model, started, deadline) {
        Ok(text) if !text.trim().is_empty() => Outcome::Rewritten(text.trim().to_string()),
        Ok(_) => Outcome::Raw("OpenCode returned empty text".to_string()),
        Err(error) => Outcome::Raw(format!("{error:#}")),
    }
}

fn prompt(raw: &str, vocabulary: &[String]) -> String {
    let vocabulary = if vocabulary.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nVocabulary (spell these exactly when the speaker means them): {}",
            vocabulary.join("; ")
        )
    };
    format!("{PROMPT}{vocabulary}\n\nTranscript:\n<transcript>\n{raw}\n</transcript>")
}

#[derive(Serialize)]
struct Request<'a> {
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<ModelRef<'a>>,
}

#[derive(Serialize)]
struct ModelRef<'a> {
    #[serde(rename = "providerID")]
    provider: &'a str,
    id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    variant: Option<&'a str>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Answer {
    Success { data: AnswerText },
    Error { message: String },
}

#[derive(Deserialize)]
struct AnswerText {
    text: String,
}

fn generate(
    service: &Service,
    prompt: &str,
    model: ModelChoice<'_>,
    started: Instant,
    deadline: Duration,
) -> Result<String> {
    let model = match model.key.trim() {
        "" => None,
        key => {
            let (provider, id) = key
                .split_once('/')
                .filter(|(provider, id)| !provider.is_empty() && !id.is_empty())
                .ok_or_else(|| anyhow!("rewrite model {key:?} is not provider/model"))?;
            let variant = Some(model.variant.trim()).filter(|variant| !variant.is_empty());
            Some(ModelRef {
                provider,
                id,
                variant,
            })
        }
    };
    let body = serde_json::to_vec(&Request { prompt, model })?;
    let mut response = None;
    for route in GENERATE_ROUTES {
        let attempt = service.request("POST", route, Some(&body), started, deadline)?;
        let missing = attempt.status == 404;
        response = Some(attempt);
        if !missing {
            break;
        }
    }
    let response = response.context("no generation route tried")?;
    let answer: Option<Answer> = serde_json::from_slice(&response.body).ok();
    match (response.status, answer) {
        (200, Some(Answer::Success { data })) => Ok(data.text),
        (status, Some(Answer::Error { message })) => {
            bail!("OpenCode generation failed (HTTP {status}): {message}")
        }
        (status, _) => bail!("OpenCode generation answered HTTP {status} without text"),
    }
}

/// One running OpenCode service: where it listens and how to authenticate.
/// The password is the service's own loopback credential from its
/// registration, sent only to that loopback address.
#[derive(Clone)]
struct Service {
    address: SocketAddr,
    password: Option<String>,
}

struct Response {
    status: u16,
    body: Vec<u8>,
}

/// OpenCode's state directory: `$XDG_STATE_HOME/opencode`, else
/// `~/.local/state/opencode` (Hex `src/opencode_service.rs` `state_dir`).
fn state_dir() -> Option<PathBuf> {
    let root = std::env::var_os("XDG_STATE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".local").join("state")))?;
    Some(root.join("opencode"))
}

/// `service.json` first, then other channels' `service-<channel>.json`; the
/// first registration whose process answers `/api/info` wins.
fn find_in(state: &Path, started: Instant, deadline: Duration) -> Option<Service> {
    let mut others: Vec<PathBuf> = fs::read_dir(state)
        .ok()?
        .flatten()
        .take(512)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("service-") && name.ends_with(".json"))
        })
        .collect();
    others.sort();
    std::iter::once(state.join("service.json"))
        .chain(others)
        .take(MAX_REGISTRATIONS)
        .filter_map(|path| read_registration(&path))
        .find_map(|registration| confirm(&registration, started, deadline))
}

#[derive(Deserialize)]
struct Registration {
    url: String,
    pid: u32,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    password: Option<String>,
}

fn read_registration(path: &Path) -> Option<Registration> {
    let mut file = open_private(path)?;
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_REGISTRATION_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_REGISTRATION_BYTES {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

/// A regular file this user owns that no one else can read.
fn open_private(path: &Path) -> Option<fs::File> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .ok()?;
    let metadata = file.metadata().ok()?;
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    let uid = unsafe { libc::geteuid() };
    (metadata.is_file() && metadata.uid() == uid && metadata.permissions().mode() & 0o077 == 0)
        .then_some(file)
}

/// The registered service, when it is on loopback and the process answering
/// `/api/info` is the registered one. A stale registration means OpenCode
/// isn't running.
fn confirm(registration: &Registration, started: Instant, deadline: Duration) -> Option<Service> {
    #[derive(Deserialize)]
    struct Info {
        pid: u32,
        version: String,
    }
    let service = Service {
        address: loopback_address(&registration.url)?,
        password: registration
            .password
            .clone()
            .filter(|password| !password.is_empty()),
    };
    let response = service
        .request("GET", "/api/info", None, started, deadline)
        .ok()?;
    if response.status != 200 {
        return None;
    }
    let info: Info = serde_json::from_slice(&response.body).ok()?;
    let version_matches = match &registration.version {
        Some(version) => *version == info.version,
        None => true,
    };
    (info.pid == registration.pid && version_matches).then_some(service)
}

/// A registered `http://` URL's loopback socket; an unspecified address
/// (`0.0.0.0`, `::`) is reached through loopback. Anything else is refused.
fn loopback_address(url: &str) -> Option<SocketAddr> {
    let url = reqwest::Url::parse(url).ok()?;
    if url.scheme() != "http"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    let ip = match url.host()? {
        url::Host::Domain("localhost") => IpAddr::V4(Ipv4Addr::LOCALHOST),
        url::Host::Ipv4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        url::Host::Ipv6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        url::Host::Ipv4(ip) if ip.is_loopback() => IpAddr::V4(ip),
        url::Host::Ipv6(ip) if ip.is_loopback() => IpAddr::V6(ip),
        _ => return None,
    };
    let port = url.port_or_known_default().filter(|port| *port != 0)?;
    Some(SocketAddr::new(ip, port))
}

impl Service {
    /// One HTTP/1.1 request bounded by what is left of `deadline` since
    /// `started`. HTTP error statuses return normally; refusal, transport
    /// failure and the deadline are errors.
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
        started: Instant,
        deadline: Duration,
    ) -> Result<Response> {
        let remaining = || deadline.saturating_sub(started.elapsed());
        let check = || -> Result<()> {
            if remaining().is_zero() {
                bail!(
                    "OpenCode {path} exceeded the {} ms rewrite deadline",
                    deadline.as_millis()
                );
            }
            Ok(())
        };
        check()?;
        let mut head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nAccept: application/json\r\nConnection: close\r\n",
            self.address
        );
        if let Some(password) = &self.password {
            if password.contains(['\r', '\n']) {
                bail!("OpenCode registration has an invalid password");
            }
            head.push_str(&format!(
                "Authorization: Basic {}\r\n",
                base64(format!("opencode:{password}").as_bytes())
            ));
        }
        if let Some(body) = body {
            head.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                body.len()
            ));
        } else if method != "GET" {
            head.push_str("Content-Length: 0\r\n");
        }
        head.push_str("\r\n");
        let mut stream = TcpStream::connect_timeout(&self.address, remaining())
            .map_err(|error| anyhow!("OpenCode isn't running ({error})"))?;
        // One buffer, written in slices that each recheck the shared
        // deadline: a service that drains the request slowly keeps a single
        // `write_all` making partial progress past any per-call timeout.
        let mut request = head.into_bytes();
        request.extend_from_slice(body.unwrap_or_default());
        let mut sent = 0;
        while sent < request.len() {
            check()?;
            stream.set_write_timeout(Some(
                remaining().clamp(Duration::from_millis(1), READ_SLICE),
            ))?;
            match stream.write(&request[sent..]) {
                Ok(0) => bail!("OpenCode closed the connection while receiving {path}"),
                Ok(written) => sent += written,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock
                            | std::io::ErrorKind::TimedOut
                            | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(error) => {
                    return Err(error).with_context(|| format!("could not send OpenCode {path}"));
                }
            }
        }
        let mut bytes = Vec::new();
        let mut buffer = [0; 16 * 1024];
        loop {
            check()?;
            stream.set_read_timeout(Some(
                remaining().clamp(Duration::from_millis(1), READ_SLICE),
            ))?;
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    bytes.extend_from_slice(&buffer[..read]);
                    if bytes.len() > MAX_RESPONSE_BYTES {
                        bail!("OpenCode {path} answered more than 1 MiB");
                    }
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock
                            | std::io::ErrorKind::TimedOut
                            | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(error) => {
                    return Err(error).with_context(|| format!("could not read OpenCode {path}"));
                }
            }
        }
        parse_response(&bytes).with_context(|| format!("OpenCode {path} returned invalid HTTP"))
    }
}

fn parse_response(bytes: &[u8]) -> Option<Response> {
    let split = bytes.windows(4).position(|window| window == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&bytes[..split]).ok()?;
    let mut lines = head.split("\r\n");
    let status = lines.next()?.split_whitespace().nth(1)?.parse().ok()?;
    let mut length = None;
    let mut chunked = false;
    for line in lines {
        let (key, value) = line.split_once(':')?;
        let value = value.trim();
        if key.eq_ignore_ascii_case("content-length") {
            length = Some(value.parse::<usize>().ok()?);
        } else if key.eq_ignore_ascii_case("transfer-encoding") {
            chunked = value.eq_ignore_ascii_case("chunked");
        }
    }
    let body = &bytes[split + 4..];
    let body = if chunked {
        dechunk(body)?
    } else if let Some(length) = length {
        body.get(..length)?.to_vec()
    } else {
        body.to_vec()
    };
    Some(Response { status, body })
}

fn dechunk(mut body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line = body.windows(2).position(|window| window == b"\r\n")?;
        let size = std::str::from_utf8(&body[..line]).ok()?;
        let size = usize::from_str_radix(size.split(';').next()?.trim(), 16).ok()?;
        body = &body[line + 2..];
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(body.get(..size)?);
        body = body.get(size..)?.strip_prefix(b"\r\n")?;
    }
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = u32::from_be_bytes([
            0,
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ]);
        let ch = |shift: u32| char::from(TABLE[((n >> shift) & 63) as usize]);
        out.extend([
            ch(18),
            ch(12),
            if chunk.len() > 1 { ch(6) } else { '=' },
            if chunk.len() > 2 { ch(0) } else { '=' },
        ]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::thread;

    const RAW: &str = "um so i think we should uh ship it tomorrow";

    /// A fake OpenCode answering each request in turn with `(status, body)`;
    /// `None` accepts the connection and never answers. Returns the service
    /// and the request heads and bodies it received.
    fn serve(
        answers: Vec<Option<(u16, &'static str)>>,
    ) -> (Service, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut received = Vec::new();
            for answer in answers {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut head = String::new();
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse().unwrap();
                    }
                    if line == "\r\n" {
                        break;
                    }
                    head.push_str(&line);
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                received.push(format!("{head}\r\n{}", String::from_utf8(body).unwrap()));
                match answer {
                    Some((status, body)) => write!(
                        stream,
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .unwrap(),
                    // Hold the connection open past the client's deadline.
                    None => thread::sleep(Duration::from_secs(2)),
                }
            }
            received
        });
        (
            Service {
                address,
                password: Some("loopback-secret".to_string()),
            },
            server,
        )
    }

    const MODEL: ModelChoice<'static> = ModelChoice {
        key: "zai-coding-plan/glm-5.3-flash",
        variant: "low",
    };

    fn run(service: &Service, deadline: Duration) -> Outcome {
        rewrite_with(service, RAW, MODEL, &[], Instant::now(), deadline)
    }

    #[test]
    fn a_success_answer_is_pasted_as_the_rewrite() {
        let (service, server) = serve(vec![Some((
            200,
            r#"{"data":{"text":" So I think we should ship it tomorrow.\n"}}"#,
        ))]);
        let outcome = run(&service, Duration::from_secs(5));
        assert_eq!(
            outcome,
            Outcome::Rewritten("So I think we should ship it tomorrow.".to_string())
        );
        assert_eq!(outcome.text(RAW), "So I think we should ship it tomorrow.");
        let request = &server.join().unwrap()[0];
        assert!(request.starts_with("POST /api/experimental/generate HTTP/1.1\r\n"));
        // base64("opencode:loopback-secret")
        assert!(request.contains("Authorization: Basic b3BlbmNvZGU6bG9vcGJhY2stc2VjcmV0\r\n"));
        assert!(request.contains(
            r#""model":{"providerID":"zai-coding-plan","id":"glm-5.3-flash","variant":"low"}"#
        ));
        assert!(request.contains(RAW));
    }

    #[test]
    fn an_absent_harness_pastes_the_raw_transcript() {
        // Bind then drop: nothing listens on the port, so connect is refused.
        let address = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let service = Service {
            address,
            password: None,
        };
        let started = Instant::now();
        let outcome = run(&service, Duration::from_secs(5));
        assert!(
            matches!(&outcome, Outcome::Raw(reason) if reason.contains("isn't running")),
            "{outcome:?}"
        );
        assert_eq!(outcome.text(RAW), RAW);
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn a_harness_that_never_answers_pastes_the_raw_transcript_at_the_deadline() {
        let (service, _server) = serve(vec![None]);
        let started = Instant::now();
        let outcome = run(&service, Duration::from_millis(300));
        let elapsed = started.elapsed();
        assert!(
            matches!(&outcome, Outcome::Raw(reason) if reason.contains("deadline")),
            "{outcome:?}"
        );
        assert_eq!(outcome.text(RAW), RAW);
        assert!(elapsed >= Duration::from_millis(300), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(1_000), "{elapsed:?}");
    }

    #[test]
    fn a_harness_that_reads_slowly_pastes_the_raw_transcript_at_the_deadline() {
        // The service accepts, then drains the request a few KB at a time and
        // hangs up after 3 s. The prompt (about 2 MB of vocabulary) is far
        // larger than the socket buffers, so sending it keeps making partial
        // progress; the deadline must still bound the whole request.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let _server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let opened = Instant::now();
            let mut buffer = [0; 4096];
            while opened.elapsed() < Duration::from_secs(3) {
                if stream.read(&mut buffer).unwrap_or(0) == 0 {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
        });
        let service = Service {
            address,
            password: None,
        };
        let vocabulary: Vec<String> = (0..200_000).map(|i| format!("term{i}")).collect();
        let started = Instant::now();
        let outcome = rewrite_with(
            &service,
            RAW,
            MODEL,
            &vocabulary,
            started,
            Duration::from_millis(300),
        );
        let elapsed = started.elapsed();
        assert!(matches!(outcome, Outcome::Raw(_)), "{outcome:?}");
        assert_eq!(outcome.text(RAW), RAW);
        // The 300 ms deadline plus 200 ms for scheduling.
        assert!(elapsed < Duration::from_millis(500), "{elapsed:?}");
    }

    #[test]
    fn an_error_answer_pastes_the_raw_transcript() {
        let (service, _server) = serve(vec![Some((
            503,
            r#"{"_tag":"ServiceUnavailableError","message":"Free usage limit reached.","service":"opencode"}"#,
        ))]);
        let outcome = run(&service, Duration::from_secs(5));
        assert!(
            matches!(&outcome, Outcome::Raw(reason) if reason.contains("Free usage limit reached.")),
            "{outcome:?}"
        );
        assert_eq!(outcome.text(RAW), RAW);
    }

    #[test]
    fn an_error_status_carrying_text_is_not_a_rewrite() {
        let (service, _server) = serve(vec![Some((500, r#"{"data":{"text":"half an answer"}}"#))]);
        assert_eq!(run(&service, Duration::from_secs(5)).text(RAW), RAW);
    }

    #[test]
    fn an_empty_answer_pastes_the_raw_transcript() {
        let (service, _server) = serve(vec![Some((200, r#"{"data":{"text":"  \n"}}"#))]);
        let outcome = run(&service, Duration::from_secs(5));
        assert_eq!(
            outcome,
            Outcome::Raw("OpenCode returned empty text".to_string())
        );
        assert_eq!(outcome.text(RAW), RAW);
    }

    #[test]
    fn an_older_service_is_reached_on_the_fallback_route() {
        let (service, server) = serve(vec![
            Some((404, r#"{"message":"not found"}"#)),
            Some((200, r#"{"data":{"text":"Ship it."}}"#)),
        ]);
        assert_eq!(
            run(&service, Duration::from_secs(5)),
            Outcome::Rewritten("Ship it.".to_string())
        );
        let received = server.join().unwrap();
        assert!(received[1].starts_with("POST /api/generate HTTP/1.1\r\n"));
    }

    #[test]
    fn a_malformed_model_setting_pastes_the_raw_transcript() {
        let (service, _server) = serve(vec![]);
        let model = ModelChoice {
            key: "glm-5.3-flash",
            variant: "",
        };
        let outcome = rewrite_with(&service, RAW, model, &[], Instant::now(), DEADLINE);
        assert!(
            matches!(&outcome, Outcome::Raw(reason) if reason.contains("provider/model")),
            "{outcome:?}"
        );
    }

    #[test]
    fn the_prompt_carries_the_transcript_and_vocabulary() {
        let text = prompt(
            "ship parakeet",
            &["Parakeet".to_string(), "roborev".to_string()],
        );
        assert!(text.ends_with("Transcript:\n<transcript>\nship parakeet\n</transcript>"));
        assert!(text.contains(
            "Vocabulary (spell these exactly when the speaker means them): Parakeet; roborev"
        ));
        assert!(!prompt("x", &[]).contains("Vocabulary"));
    }

    fn register(state: &Path, url: &str, pid: u32, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        let path = state.join("service.json");
        fs::write(
            &path,
            format!(r#"{{"url":"{url}","pid":{pid},"password":"loopback-secret"}}"#),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
    }

    #[test]
    fn a_registered_service_is_confirmed_by_its_pid() {
        let state = tempfile::tempdir().unwrap();
        let (service, _server) = serve(vec![Some((200, r#"{"pid":42,"version":"2.0.32"}"#))]);
        register(
            state.path(),
            &format!("http://{}", service.address),
            42,
            0o600,
        );
        let found = find_in(state.path(), Instant::now(), DEADLINE).expect("confirmed service");
        assert_eq!(found.address, service.address);
        assert_eq!(found.password.as_deref(), Some("loopback-secret"));
    }

    #[test]
    fn a_stale_or_shared_registration_means_opencode_isnt_running() {
        let state = tempfile::tempdir().unwrap();
        assert!(find_in(state.path(), Instant::now(), DEADLINE).is_none());

        let (service, _server) = serve(vec![Some((200, r#"{"pid":7,"version":"2.0.32"}"#))]);
        register(
            state.path(),
            &format!("http://{}", service.address),
            42,
            0o600,
        );
        assert!(find_in(state.path(), Instant::now(), DEADLINE).is_none());

        register(
            state.path(),
            &format!("http://{}", service.address),
            7,
            0o644,
        );
        assert!(find_in(state.path(), Instant::now(), DEADLINE).is_none());
    }

    #[test]
    fn only_loopback_http_endpoints_are_used() {
        assert_eq!(
            loopback_address("http://127.0.0.1:49374"),
            Some("127.0.0.1:49374".parse().unwrap())
        );
        assert_eq!(
            loopback_address("http://0.0.0.0:4096"),
            Some("127.0.0.1:4096".parse().unwrap())
        );
        assert_eq!(
            loopback_address("http://localhost:4096/"),
            Some("127.0.0.1:4096".parse().unwrap())
        );
        for refused in [
            "https://127.0.0.1:4096",
            "http://192.168.1.2:4096",
            "http://example.com:4096",
            "http://user:pass@127.0.0.1:4096",
            "http://127.0.0.1:4096/prefix",
            "http://127.0.0.1:0",
        ] {
            assert_eq!(loopback_address(refused), None, "{refused}");
        }
    }

    #[test]
    fn chunked_answers_are_decoded() {
        let response = parse_response(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n",
        )
        .unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"hello world");
    }
}
