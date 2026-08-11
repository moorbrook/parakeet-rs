//! User vocabulary → sherpa-onnx contextual biasing ("hotwords").
//!
//! Dictation fails predictably on words the language model has never
//! weighted highly: proper nouns, product names, trade jargon,
//! colleagues' surnames. sherpa-onnx supports contextual biasing for
//! offline transducers, which is exactly the fix — but the on-disk
//! format it expects is not something a user can reasonably be asked to
//! write. This module owns that translation.
//!
//! ## The format, and why we generate it
//!
//! Two constraints were established empirically against
//! `parakeet-tdt-0.6b-v3-int8` (see [ADR-0020](../docs/ADR.md)):
//!
//! 1. **A plain word does nothing.** The model ships no `bpe.model`, so
//!    sherpa falls back to `modeling_unit="cjkchar"` and looks up each
//!    whitespace-separated piece as one token. `Kubernetes` is not a
//!    token, so it fails to encode — *silently*, with no change to the
//!    decode even at an absurd boost.
//! 2. **Word-initial position is marked with `▁` (U+2581).** The tokens
//!    are SentencePiece pieces, where `▁` means "start of word".
//!    `K u b e r n e t e s` boosts a mid-word spelling and renders
//!    glued to the preceding word; `▁K u b e r n e t e s` boosts the
//!    word-initial form, which is what dictation actually produces.
//!
//! So the user writes `Kubernetes` and we emit `▁K u b e r n e t e s`;
//! they write `New York` and we emit `▁N e w ▁Y o r k`.
//!
//! ## Cost
//!
//! Biasing requires `decoding_method="modified_beam_search"` — greedy
//! decoding rejects a hotwords file outright (the recognizer fails to
//! construct). Beam search measured **+13%** decode time on the 5 s
//! bench fixture (396 → 448 ms). We therefore only switch away from
//! greedy when the user actually has vocabulary entries; an empty or
//! absent file costs nothing.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// SentencePiece word-start marker. Prefixed to the first character of
/// every word so biasing targets the word-initial token.
const WORD_START: char = '\u{2581}';

/// Above this many terms the context graph starts costing real memory
/// and decode time. We don't refuse — it's the user's machine — but we
/// say so once at load.
const LARGE_VOCABULARY_WARN: usize = 500;

/// Strip comments and blank lines from a raw `vocabulary.txt`, yielding
/// the terms in file order.
///
/// `#` starts a comment only at the beginning of a line (after
/// trimming). A `#` mid-line is a literal character — some real terms
/// contain one, and there is no escaping syntax worth inventing here.
pub fn parse_terms(raw: &str) -> Vec<&str> {
    // Strip a UTF-8 BOM. Several editors write one, and it would
    // otherwise glue itself to the first line — turning the template's
    // opening `#` comment into an active term, which silently switches
    // decoding to beam search for a vocabulary the user never wrote.
    raw.trim_start_matches('\u{feff}')
        .lines()
        .map(|l| l.trim().trim_start_matches('\u{feff}'))
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// Whether the user's vocabulary contains at least one active term.
///
/// Backend selection uses this inexpensive check before loading a model:
/// Parakeet Unified is the fast default, while a non-empty vocabulary keeps
/// the sherpa backend whose contextual-biasing graph preserves that feature.
pub fn has_terms(path: &Path) -> Result<bool> {
    match std::fs::read_to_string(path) {
        Ok(raw) => Ok(!parse_terms(&raw).is_empty()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

/// Encode one natural-language term into sherpa's space-separated
/// token form, marking each word's first character as word-initial.
///
/// Returns `None` for a term with no non-whitespace content.
///
/// ```text
/// "Kubernetes"  -> "▁K u b e r n e t e s"
/// "New York"    -> "▁N e w ▁Y o r k"
/// ```
pub fn encode_term(term: &str) -> Option<String> {
    let mut pieces: Vec<String> = Vec::new();
    for word in term.split_whitespace() {
        for (i, ch) in word.chars().enumerate() {
            if i == 0 {
                pieces.push(format!("{WORD_START}{ch}"));
            } else {
                pieces.push(ch.to_string());
            }
        }
    }
    if pieces.is_empty() {
        None
    } else {
        Some(pieces.join(" "))
    }
}

/// The model's token inventory, read from `tokens.txt`.
///
/// Exists to catch the silent-failure mode: sherpa drops any hotword
/// containing a piece it can't encode, logging to stderr — which the
/// bundled app discards. Without this check a user adds `Ω` or a
/// decomposed `é` or an emoji to their vocabulary, sees no error, and
/// gets no biasing, with nothing anywhere to explain why.
pub struct TokenSet(std::collections::HashSet<String>);

impl TokenSet {
    /// Parse `tokens.txt`, whose lines are `<token> <id>`.
    pub fn load(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let set = raw
            .lines()
            // Split from the RIGHT: the id is the last field, and a
            // token may itself contain a space.
            .filter_map(|l| l.rsplit_once(' ').map(|(tok, _id)| tok.to_string()))
            .collect();
        Ok(Self(set))
    }

    fn contains(&self, piece: &str) -> bool {
        self.0.contains(piece)
    }
}

/// A term that couldn't be encoded, and the piece that killed it.
#[derive(Debug)]
pub struct RejectedTerm {
    pub term: String,
    pub piece: String,
}

/// Render a vocabulary file body into a sherpa hotwords body, dropping
/// terms the model can't represent.
///
/// Returns the encoded body plus the rejects, so the caller can tell
/// the user which of their words are doing nothing. Pure apart from the
/// borrowed `TokenSet`, so the encoding stays testable without disk.
///
/// `tokens = None` skips validation (used where the token inventory
/// isn't available); everything encodes and sherpa silently drops what
/// it can't handle, which is the old behaviour.
pub fn encode_file(raw: &str, tokens: Option<&TokenSet>) -> (String, Vec<RejectedTerm>) {
    let mut out = String::new();
    let mut rejected = Vec::new();
    for term in parse_terms(raw) {
        let Some(encoded) = encode_term(term) else {
            continue;
        };
        if let Some(tokens) = tokens {
            if let Some(bad) = encoded.split(' ').find(|p| !tokens.contains(p)) {
                rejected.push(RejectedTerm {
                    term: term.to_string(),
                    piece: bad.to_string(),
                });
                continue;
            }
        }
        out.push_str(&encoded);
        out.push('\n');
    }
    (out, rejected)
}

/// Read `vocab_path`, translate it, and write the result to
/// `generated_path`.
///
/// Returns the generated file's path when there is at least one term to
/// bias toward, and `None` when the vocabulary is absent or empty — the
/// caller uses that to decide between greedy decoding (cheap) and
/// modified beam search (biased, ~13% slower).
///
/// A stale generated file is removed in the `None` case so a
/// vocabulary the user emptied doesn't keep biasing forever.
pub fn prepare(
    vocab_path: &Path,
    generated_path: &Path,
    tokens_path: Option<&Path>,
) -> Result<Option<PathBuf>> {
    let raw = match std::fs::read_to_string(vocab_path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            return Err(e).with_context(|| format!("reading {}", vocab_path.display()));
        }
    };

    let terms = parse_terms(&raw);
    if terms.is_empty() {
        // Best-effort cleanup; a leftover file that we then don't point
        // the recognizer at is harmless, so failure here isn't fatal.
        let _ = std::fs::remove_file(generated_path);
        return Ok(None);
    }
    if terms.len() > LARGE_VOCABULARY_WARN {
        log::warn!(
            "vocabulary has {} terms (> {LARGE_VOCABULARY_WARN}); the biasing \
             context graph grows with this list and may slow decoding",
            terms.len()
        );
    }

    // Validation is best-effort: an unreadable tokens.txt shouldn't
    // block biasing, it just costs us the diagnostic.
    let tokens = match tokens_path {
        Some(p) => match TokenSet::load(p) {
            Ok(t) => Some(t),
            Err(e) => {
                log::warn!("vocabulary: can't validate terms against {p:?}: {e:#}");
                None
            }
        },
        None => None,
    };

    let (body, rejected) = encode_file(&raw, tokens.as_ref());
    for r in &rejected {
        log::warn!(
            "vocabulary: {:?} can't be represented by this model (no token for {:?}) — skipped",
            r.term,
            r.piece
        );
    }
    if body.is_empty() {
        log::warn!("vocabulary: no usable terms after validation; biasing off");
        let _ = std::fs::remove_file(generated_path);
        return Ok(None);
    }

    if let Some(parent) = generated_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(generated_path, &body)
        .with_context(|| format!("writing {}", generated_path.display()))?;
    log::info!(
        "vocabulary: {} terms encoded to {}",
        terms.len(),
        generated_path.display()
    );
    Ok(Some(generated_path.to_path_buf()))
}

/// The starter file dropped on first launch, so "Edit Vocabulary…"
/// opens something self-explanatory rather than an empty buffer.
pub const TEMPLATE: &str = "\
# Parakeet vocabulary — one term or phrase per line.
#
# Words listed here get boosted during recognition. Use it for names,
# jargon, and product names that Parakeet mishears. Write them exactly
# as you want them transcribed, including capitalisation:
#
#   Kubernetes
#   Ghostty
#   New York
#
# Lines starting with # are ignored. An empty list costs nothing;
# a non-empty one makes decoding roughly 13% slower.
";

/// Create `path` with [`TEMPLATE`] if it doesn't exist yet. Never
/// overwrites an existing file.
pub fn ensure_template(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    // `create_new`, not `write`: the `exists()` check above is a TOCTOU
    // window, and `std::fs::write` TRUNCATES. A vocabulary that landed
    // in that window (editor, sync client, dotfile manager) would be
    // destroyed by the very call whose contract is "never overwrites".
    // `create_new` makes the OS enforce it.
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut f) => {
            use std::io::Write as _;
            f.write_all(TEMPLATE.as_bytes())
                .with_context(|| format!("writing {}", path.display()))?;
            Ok(())
        }
        // Lost the race; the other writer's content wins, as intended.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e).with_context(|| format!("creating {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_word_gets_a_word_start_marker_then_bare_chars() {
        // The load-bearing encoding. `Kubernetes` verbatim does nothing
        // (not a token); bare chars boost the mid-word form and render
        // glued to the previous word. Only this form is correct.
        assert_eq!(encode_term("Kubernetes").unwrap(), "▁K u b e r n e t e s");
    }

    #[test]
    fn every_word_in_a_phrase_is_marked_word_initial() {
        // Without the marker on `York`, the decode renders "NewYork".
        assert_eq!(encode_term("New York").unwrap(), "▁N e w ▁Y o r k");
    }

    #[test]
    fn interior_whitespace_runs_collapse() {
        assert_eq!(encode_term("  New   York  ").unwrap(), "▁N e w ▁Y o r k");
    }

    #[test]
    fn whitespace_only_term_encodes_to_nothing() {
        assert!(encode_term("   ").is_none());
        assert!(encode_term("").is_none());
    }

    #[test]
    fn capitalisation_is_preserved_verbatim() {
        // The boost injects the exact characters listed, so casing in
        // the vocabulary file is the casing that gets transcribed.
        assert_eq!(encode_term("gRPC").unwrap(), "▁g R P C");
    }

    #[test]
    fn non_ascii_terms_encode_per_character() {
        assert_eq!(encode_term("Café").unwrap(), "▁C a f é");
    }

    #[test]
    fn comments_and_blank_lines_are_dropped() {
        let raw = "# a comment\n\nKubernetes\n   \n  # indented comment\nGhostty\n";
        assert_eq!(parse_terms(raw), vec!["Kubernetes", "Ghostty"]);
    }

    #[test]
    fn hash_inside_a_term_is_literal() {
        // `C#` is a real term someone would put in this file. Only a
        // leading `#` starts a comment.
        assert_eq!(parse_terms("C#\n"), vec!["C#"]);
        assert_eq!(encode_term("C#").unwrap(), "▁C #");
    }

    #[test]
    fn has_terms_treats_missing_and_comment_only_files_as_empty() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vocabulary.txt");
        assert!(!has_terms(&path).unwrap());
        std::fs::write(&path, "# template only\n\n").unwrap();
        assert!(!has_terms(&path).unwrap());
        std::fs::write(&path, "# template\nKubernetes\n").unwrap();
        assert!(has_terms(&path).unwrap());
    }

    #[test]
    fn encode_file_emits_one_newline_terminated_line_per_term() {
        let raw = "# header\nKubernetes\nNew York\n";
        assert_eq!(
            encode_file(raw, None).0,
            "▁K u b e r n e t e s\n▁N e w ▁Y o r k\n"
        );
    }

    #[test]
    fn the_shipped_template_parses_to_an_empty_vocabulary() {
        // Every line of the starter file must be a comment — otherwise
        // a fresh install silently pays the beam-search cost and boosts
        // words from the instructions.
        assert!(
            parse_terms(TEMPLATE).is_empty(),
            "template must contain no active terms: {:?}",
            parse_terms(TEMPLATE)
        );
    }

    #[test]
    fn a_utf8_bom_does_not_turn_the_first_comment_into_a_term() {
        // Several editors prepend a BOM. Glued to the leading `#` it
        // stops looking like a comment, which would silently switch
        // decoding to beam search and boost the instructions.
        let with_bom = format!("\u{feff}{TEMPLATE}");
        assert!(
            parse_terms(&with_bom).is_empty(),
            "BOM leaked into terms: {:?}",
            parse_terms(&with_bom)
        );
    }

    #[test]
    fn a_bom_before_a_real_term_does_not_corrupt_its_encoding() {
        assert_eq!(parse_terms("\u{feff}Kubernetes\n"), vec!["Kubernetes"]);
    }

    fn token_set(tokens: &[&str]) -> TokenSet {
        TokenSet(tokens.iter().map(|t| (*t).to_string()).collect())
    }

    #[test]
    fn terms_with_no_matching_token_are_rejected_not_silently_emitted() {
        // The whole trap this module exists to avoid: sherpa drops an
        // unencodable hotword with a stderr log the bundled app never
        // shows. We must surface it instead of writing a line that does
        // nothing.
        let tokens = token_set(&["▁K", "u", "b", "e", "r", "n", "t", "s"]);
        let (body, rejected) = encode_file("Kubernetes\nQ\n", Some(&tokens));
        assert_eq!(body, "▁K u b e r n e t e s\n");
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].term, "Q");
        assert_eq!(rejected[0].piece, "▁Q");
    }

    #[test]
    fn validation_is_skipped_when_no_token_set_is_supplied() {
        let (body, rejected) = encode_file("Q\n", None);
        assert_eq!(body, "▁Q\n");
        assert!(rejected.is_empty());
    }

    #[test]
    fn token_set_parses_tokens_containing_spaces() {
        // tokens.txt lines are `<token> <id>`; splitting from the LEFT
        // would mangle any token that itself contains a space.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tokens.txt");
        std::fs::write(&p, "▁K 12\n  7\nu 3\n").unwrap();
        let set = TokenSet::load(&p).unwrap();
        assert!(set.contains("▁K"));
        assert!(set.contains("u"));
        assert!(set.contains(" "), "a space token must survive parsing");
    }

    #[test]
    fn prepare_returns_none_when_validation_rejects_every_term() {
        // All terms unencodable == an empty hotwords file. Pointing the
        // recogniser at that would pay the beam-search cost for zero
        // biasing, so it must fall back to greedy.
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("vocabulary.txt");
        let generated = dir.path().join("hotwords.generated.txt");
        let tokens = dir.path().join("tokens.txt");
        std::fs::write(&vocab, "Zzz\n").unwrap();
        std::fs::write(&tokens, "▁K 1\nu 2\n").unwrap();

        assert!(prepare(&vocab, &generated, Some(&tokens))
            .unwrap()
            .is_none());
        assert!(!generated.exists());
    }

    #[test]
    fn prepare_returns_none_and_clears_stale_output_for_an_empty_vocabulary() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("vocabulary.txt");
        let generated = dir.path().join("hotwords.generated.txt");
        // Seed a stale generated file from a previous non-empty run.
        std::fs::write(&generated, "▁O l d\n").unwrap();
        std::fs::write(&vocab, "# everything commented out\n").unwrap();

        assert!(prepare(&vocab, &generated, None).unwrap().is_none());
        assert!(
            !generated.exists(),
            "a stale hotwords file would keep biasing after the user cleared their vocabulary"
        );
    }

    #[test]
    fn prepare_treats_a_missing_vocabulary_as_empty_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("does-not-exist.txt");
        let generated = dir.path().join("hotwords.generated.txt");
        assert!(prepare(&vocab, &generated, None).unwrap().is_none());
    }

    #[test]
    fn prepare_writes_the_encoded_file_and_returns_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("vocabulary.txt");
        let generated = dir.path().join("nested").join("hotwords.generated.txt");
        std::fs::write(&vocab, "Kubernetes\nNew York\n").unwrap();

        let out = prepare(&vocab, &generated, None)
            .unwrap()
            .expect("some terms");
        assert_eq!(out, generated);
        assert_eq!(
            std::fs::read_to_string(&generated).unwrap(),
            "▁K u b e r n e t e s\n▁N e w ▁Y o r k\n"
        );
    }

    #[test]
    fn ensure_template_does_not_clobber_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("vocabulary.txt");
        std::fs::write(&vocab, "Kubernetes\n").unwrap();
        ensure_template(&vocab).unwrap();
        assert_eq!(std::fs::read_to_string(&vocab).unwrap(), "Kubernetes\n");
    }

    #[test]
    fn ensure_template_creates_the_file_and_its_parent() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("nested").join("vocabulary.txt");
        ensure_template(&vocab).unwrap();
        assert_eq!(std::fs::read_to_string(&vocab).unwrap(), TEMPLATE);
    }
}
