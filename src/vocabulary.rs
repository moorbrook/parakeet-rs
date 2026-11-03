//! User-editable vocabulary terms sent verbatim to the native Core ML worker.

use std::path::Path;

use anyhow::{Context, Result};

/// Strip comments and blank lines from a raw `vocabulary.txt`, yielding
/// the terms in file order.
///
/// `#` starts a comment only at the beginning of a line (after
/// trimming). A `#` mid-line is a literal character — some real terms
/// contain one, and there is no escaping syntax worth inventing here.
pub fn parse_terms(raw: &str) -> Vec<&str> {
    // Strip a UTF-8 BOM. Several editors write one, and it would
    // otherwise glue itself to the first line — turning the template's
    // opening `#` comment into an active term the user never wrote.
    raw.trim_start_matches('\u{feff}')
        .lines()
        .map(|l| l.trim().trim_start_matches('\u{feff}'))
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// The user's active terms, in file order.
///
/// A missing file is an empty vocabulary, not an error: the template is
/// written on first launch and a user may delete it.
pub fn terms(path: &Path) -> Result<Vec<String>> {
    match std::fs::read_to_string(path) {
        Ok(raw) => Ok(parse_terms(&raw).into_iter().map(str::to_string).collect()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

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
# Lines starting with # are ignored.
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
    fn comments_and_blank_lines_are_dropped() {
        let raw = "# a comment\n\nKubernetes\n   \n  # indented comment\nGhostty\n";
        assert_eq!(parse_terms(raw), vec!["Kubernetes", "Ghostty"]);
    }

    #[test]
    fn hash_inside_a_term_is_literal() {
        // `C#` is a real term someone would put in this file. Only a
        // leading `#` starts a comment.
        assert_eq!(parse_terms("C#\n"), vec!["C#"]);
    }

    #[test]
    fn terms_treats_missing_and_comment_only_files_as_empty() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vocabulary.txt");
        assert!(terms(&path).unwrap().is_empty());
        std::fs::write(&path, "# template only\n\n").unwrap();
        assert!(terms(&path).unwrap().is_empty());
        std::fs::write(&path, "# template\nKubernetes\nNew York\n").unwrap();
        assert_eq!(terms(&path).unwrap(), vec!["Kubernetes", "New York"]);
    }

    #[test]
    fn the_shipped_template_parses_to_an_empty_vocabulary() {
        // Every line of the starter file must be a comment — otherwise
        // a fresh install biases toward words from the instructions.
        assert!(
            parse_terms(TEMPLATE).is_empty(),
            "template must contain no active terms: {:?}",
            parse_terms(TEMPLATE)
        );
    }

    #[test]
    fn a_utf8_bom_does_not_turn_the_first_comment_into_a_term() {
        // Several editors prepend a BOM. Glued to the leading `#` it
        // stops looking like a comment and biases toward the instructions.
        let with_bom = format!("\u{feff}{TEMPLATE}");
        assert!(
            parse_terms(&with_bom).is_empty(),
            "BOM leaked into terms: {:?}",
            parse_terms(&with_bom)
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
