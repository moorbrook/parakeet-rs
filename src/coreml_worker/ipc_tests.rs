use super::*;
use std::os::unix::fs::PermissionsExt;

const BUDGET: Duration = Duration::from_millis(150);

fn fake(body: &str) -> WorkerProcess {
    let mut child = Command::new("/usr/bin/perl")
        .args(["-e", &format!("$|=1; alarm 3; {body}")])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    WorkerProcess::new(child, stdin, stdout).unwrap()
}

fn assert_timeout<T: std::fmt::Debug>(
    process: &mut WorkerProcess,
    result: Result<T>,
    start: Instant,
) {
    let error = result.unwrap_err();
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::TimedOut,
        "{error:#}"
    );
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(process.failed);
    assert_reaped(process);
}

fn assert_reaped(process: &mut WorkerProcess) {
    let mut status = 0;
    let pid = i32::try_from(process.child.id()).unwrap();
    // SAFETY: status is writable and the PID identifies this test's child.
    let result = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
    assert_eq!(
        result, -1,
        "child must already be reaped, not merely killed"
    );
    assert_eq!(
        io::Error::last_os_error().raw_os_error(),
        Some(libc::ECHILD)
    );
    assert!(
        process.child.try_wait().unwrap().is_some(),
        "worker must be reaped before returning"
    );
}

#[test]
fn ipc_stalled_frames_are_cancelled_and_reaped() {
    for body in [
        "sleep 10;",
        "print pack('C', 1); sleep 10;",
        "print pack('V', 100), '{'; sleep 10;",
    ] {
        let mut process = fake(body);
        let start = Instant::now();
        let result = process.transaction(BUDGET, WorkerProcess::read_response);
        assert_timeout(&mut process, result, start);
    }
}

#[test]
fn ipc_blocked_audio_and_vocabulary_writes_are_cancelled() {
    let mut process = fake("sleep 10;");
    let samples = vec![0.25; 1_000_000];
    let start = Instant::now();
    let result = process.transaction(BUDGET, |p, deadline| {
        p.write_request(&samples, 16_000, 1_000_000, deadline)
    });
    assert_timeout(&mut process, result, start);

    let mut process = fake("sleep 10;");
    let start = Instant::now();
    let result = process.transaction(BUDGET, |p, deadline| {
        p.set_vocabulary(&["x".repeat(500_000)], 1.0, deadline)
    });
    assert_timeout(&mut process, result, start);
}

#[test]
fn ipc_exit_is_an_error_and_is_reaped() {
    let mut process = fake("exit 0;");
    let error = process
        .transaction(BUDGET, WorkerProcess::read_response)
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert!(process.child.try_wait().unwrap().is_some());
}

#[test]
fn ipc_partial_progress_does_not_reset_the_deadline() {
    let mut process = fake("for (1..4) { print pack('C', 1); select undef,undef,undef,0.08; }");
    let start = Instant::now();
    let result = process.transaction(BUDGET, WorkerProcess::read_response);
    assert_timeout(&mut process, result, start);
}

#[test]
fn ipc_header_write_to_a_full_pipe_is_bounded() {
    let mut process = fake("sleep 10;");
    let bytes = [0; 4096];
    loop {
        match process.stdin.write(&bytes) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
            Err(error) => panic!("{error}"),
        }
    }
    let start = Instant::now();
    let result = process.transaction(BUDGET, |p, deadline| {
        p.write_request(&[], 16_000, 0, deadline)
    });
    assert_timeout(&mut process, result, start);
}

// The fake checks the actual wire headers and audio bytes before replying.
// An alarm bounds mutant tests even if a blocking pipe call is reintroduced.
const SERVER: &str = r#"
use strict;
use warnings;
$|=1;
alarm 5;
sub exact {
    my ($count) = @_;
    my $out = '';
    while (length($out) < $count) {
        my $n = read(STDIN, my $part, $count-length($out));
        die 'short request' unless $n;
        $out .= $part;
    }
    return $out;
}
sub frame { my ($json) = @_; print pack('V', length($json)), $json; }
frame('{"kind":"ready","ok":true,"load_seconds":0.1}');
my ($magic,$version,$length,$reserved) = unpack('a4VVV', exact(16));
die 'vocabulary header' unless $magic eq 'PRKV' && $version == 1 && $reserved == 0;
my $vocab = exact($length);
die 'missing vocabulary' unless $vocab eq '{"terms":["Parakeet"],"score":2.0}';
frame('{"kind":"vocabulary","ok":true,"vocabulary_accepted":1,"vocabulary_rejected":[]}');
my ($audio,$v,$rate,$count) = unpack('a4VVV', exact(16));
die 'audio header' unless $audio eq 'PRKT' && $v == 1 && $rate == 16000 && $count == 2;
die 'audio bytes' unless exact(8) eq pack('f<f<', 0.25, -0.5);
frame('{"kind":"result","ok":true,"text":"Hi.","decode_seconds":0.04,"token_spans":[{"text":" Hi","start_s":0.08,"end_s":0.16},{"text":".","start_s":0.16,"end_s":0.24}]}');
"#;

fn config_for(directory: &tempfile::TempDir, body: &str) -> CoreMlWorkerConfig {
    let worker = directory.path().join("worker");
    std::fs::write(&worker, format!("#!/usr/bin/perl\n{body}")).unwrap();
    std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut config = CoreMlWorkerConfig::new(worker, directory.path());
    config.vocabulary = vec!["Parakeet".into()];
    config.vocabulary_score = 2.0;
    config.startup_timeout = Duration::from_secs(2);
    config.request_timeout = BUDGET;
    config
}

#[test]
fn ipc_backend_preserves_vocabulary_audio_and_spans() {
    let directory = tempfile::tempdir().unwrap();
    let config = config_for(&directory, SERVER);
    let backend = CoreMlWorkerBackend::spawn(&config).unwrap();
    assert_eq!(backend.contextual_vocabulary().unwrap().accepted, 1);
    let (decoded, spans) = backend.decode(&[0.25, -0.5], 16_000, true).unwrap();
    assert_eq!(decoded.text, "Hi.");
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].text, " Hi");
    assert!((spans[1].end_s - 0.24).abs() < 1e-6);
}

#[test]
fn ipc_startup_has_its_own_budget() {
    let directory = tempfile::tempdir().unwrap();
    let config = config_for(
        &directory,
        &format!("select undef,undef,undef,0.3;\n{SERVER}"),
    );
    let backend = CoreMlWorkerBackend::spawn(&config).unwrap();
    assert_eq!(
        backend.decode(&[0.25, -0.5], 16_000, false).unwrap().0.text,
        "Hi."
    );
}

#[test]
fn ipc_backend_releases_lock_and_restarts_with_vocabulary() {
    let directory = tempfile::tempdir().unwrap();
    let stalled = SERVER.split("my ($audio").next().unwrap().to_owned() + "sleep 10;";
    let config = config_for(&directory, &stalled);
    let backend = CoreMlWorkerBackend::spawn(&config).unwrap();
    let error = backend.decode(&[0.25, -0.5], 16_000, false).unwrap_err();
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::TimedOut
    );
    let mut process = backend.process.try_lock().expect("decoder lock released");
    assert!(process.child.try_wait().unwrap().is_some());
    drop(process);
    config_for(&directory, SERVER);
    assert_eq!(
        backend.decode(&[0.25, -0.5], 16_000, true).unwrap().0.text,
        "Hi."
    );
}

#[test]
fn ipc_request_write_and_read_share_one_budget() {
    let mut process = fake("select undef,undef,undef,0.1; while (read(STDIN, my $part, 4096)) { last if length($part) < 4096; } sleep 10;");
    let start = Instant::now();
    let result = process.transaction(Duration::from_millis(250), |p, deadline| {
        p.write_request(&vec![0.25; 100_000], 16_000, 100_000, deadline)?;
        p.read_response(deadline)
    });
    assert_timeout(&mut process, result, start);
    assert!(start.elapsed() < Duration::from_millis(340));
}

#[test]
fn ipc_invalid_frames_invalidate_the_process() {
    for body in ["print pack('V', 4194305);", "print pack('V', 1), 'x';"] {
        let mut process = fake(body);
        assert!(process
            .transaction(BUDGET, WorkerProcess::read_response)
            .is_err());
        assert!(process.failed);
        assert!(process.child.try_wait().unwrap().is_some());
    }
}

#[test]
fn ipc_missing_text_is_not_a_successful_empty_transcript() {
    let directory = tempfile::tempdir().unwrap();
    let config = config_for(&directory, &SERVER.replace("\"text\":\"Hi.\",", ""));
    let backend = CoreMlWorkerBackend::spawn(&config).unwrap();
    assert!(backend
        .decode(&[0.25, -0.5], 16_000, false)
        .unwrap_err()
        .to_string()
        .contains("omitted text"));
}
