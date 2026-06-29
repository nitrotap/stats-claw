//! Quality gates: every `unsafe` usage in the workspace's library
//! sources must carry a `// SAFETY:` justification within the lines immediately
//! preceding it, so that each escape from Rust's safety guarantees is deliberate,
//! documented, and reviewable.
//!
//! The codebase forbids `unsafe` outright (`unsafe_code = "forbid"` in the
//! workspace lints) and currently contains none, so this gate passes. Its purpose
//! is to *lock the invariant*: should the forbid lint ever be relaxed to `allow`
//! for a narrowly-scoped block, this test still requires the justification.
//!
//! The scanner walks `crates/*/src/**/*.rs` from the workspace root, skips
//! `@generated` files (machine-emitted, not hand-maintained), and reports every
//! offending `file:line` so a failure points straight at the missing rationale.

use std::path::{Path, PathBuf};

/// How many lines above an `unsafe` usage may carry its `// SAFETY:` comment.
///
/// A justification on any of the immediately preceding lines (skipping blank
/// lines and attributes) counts; three lines covers a doc-style block such as
/// `// SAFETY: <reason spanning>` / `// <continuation>` directly above the item.
const SAFETY_LOOKBACK: usize = 3;

/// Resolves this crate's root directory (its manifest directory).
///
/// # Returns
///
/// The absolute path to the package root, taken from `CARGO_MANIFEST_DIR`; the
/// library sources live under its `src/` subdirectory.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Collects every `.rs` file under `root` (recursively) into `out`.
///
/// # Arguments
///
/// * `root` — the directory to walk; non-directories and unreadable dirs are
///   silently skipped (a missing crate `src/` is not an error here).
/// * `out` — accumulator the discovered paths are pushed onto.
fn rust_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Returns whether `content` is a machine-generated file, identified by an
/// `@generated` marker in its first few lines.
///
/// # Arguments
///
/// * `content` — the full file text.
fn is_generated(content: &str) -> bool {
    content.lines().take(5).any(|l| l.contains("@generated"))
}

/// Returns whether `line` uses the `unsafe` keyword as a token.
///
/// Tokenizes on whitespace and the punctuation that can abut `unsafe`
/// (`{`, `(`, `;`) so that identifiers merely containing the substring (e.g.
/// `unsafemath`) do not match.
///
/// # Arguments
///
/// * `line` — a single source line, comments already stripped by the caller.
fn uses_unsafe_keyword(line: &str) -> bool {
    line.split(|c: char| c.is_whitespace() || c == '{' || c == '(' || c == ';')
        .any(|tok| tok == "unsafe")
}

/// Returns whether a `// SAFETY:` justification appears on `line`.
///
/// # Arguments
///
/// * `line` — a single raw source line (comments intact).
fn is_safety_comment(line: &str) -> bool {
    line.trim_start().starts_with("// SAFETY:")
}

/// Scans `source` and returns the 1-based line numbers of every `unsafe` usage
/// that lacks a `// SAFETY:` justification within the preceding [`SAFETY_LOOKBACK`]
/// lines.
///
/// Line `//`-comments are stripped before the `unsafe`-keyword test so that prose
/// mentioning the word does not trip the scan; the `// SAFETY:` lookback, by
/// contrast, inspects the raw lines (the justification *is* a comment).
///
/// # Arguments
///
/// * `source` — the full text of one Rust source file.
///
/// # Returns
///
/// The offending 1-based line numbers, in ascending order. Empty means the file
/// is clean.
fn unjustified_unsafe_lines(source: &str) -> Vec<usize> {
    let raw: Vec<&str> = source.lines().collect();
    let mut offenders = Vec::new();
    for (idx, raw_line) in raw.iter().enumerate() {
        // Strip a trailing `//` line comment before testing for the keyword, so
        // `// describes an unsafe operation` is not flagged as a usage.
        let code = match raw_line.find("//") {
            Some(pos) => raw_line.get(..pos).unwrap_or(raw_line),
            None => raw_line,
        };
        if !uses_unsafe_keyword(code) {
            continue;
        }
        let justified = (1..=SAFETY_LOOKBACK)
            .filter_map(|back| idx.checked_sub(back))
            .filter_map(|i| raw.get(i))
            .any(|prev| is_safety_comment(prev));
        if !justified {
            offenders.push(idx + 1);
        }
    }
    offenders
}

#[test]
fn justified_unsafe_passes() {
    let src = "// SAFETY: the pointer is non-null and properly aligned here.\n\
               let x = unsafe { *p };\n";
    assert!(
        unjustified_unsafe_lines(src).is_empty(),
        "a justified unsafe usage must be accepted, got offenders {:?}",
        unjustified_unsafe_lines(src)
    );
}

#[test]
fn unjustified_unsafe_is_reported_with_location() {
    let src = "fn f() {\n    let x = unsafe { *p };\n}\n";
    let offenders = unjustified_unsafe_lines(src);
    assert_eq!(
        offenders,
        vec![2],
        "an unjustified unsafe on line 2 must be reported, got {offenders:?}"
    );
}

#[test]
fn no_unsafe_passes() {
    let src = "fn safe() -> i32 {\n    let n = 1 + 2;\n    n\n}\n";
    assert!(
        unjustified_unsafe_lines(src).is_empty(),
        "source with no unsafe must pass, got {:?}",
        unjustified_unsafe_lines(src)
    );
}

#[test]
fn prose_mentioning_unsafe_in_a_comment_is_not_flagged() {
    let src = "// This routine avoids any unsafe pointer arithmetic.\nlet n = 1;\n";
    assert!(
        unjustified_unsafe_lines(src).is_empty(),
        "the word 'unsafe' inside a comment must not be flagged, got {:?}",
        unjustified_unsafe_lines(src)
    );
}

#[test]
fn library_sources_have_no_unjustified_unsafe() {
    let root = crate_root();
    let src = root.join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(
        !files.is_empty(),
        "expected to find library sources under {}",
        src.display()
    );

    let mut offenders = Vec::new();
    for file in &files {
        let content = std::fs::read_to_string(file).unwrap_or_default();
        if is_generated(&content) {
            continue;
        }
        for line in unjustified_unsafe_lines(&content) {
            let shown = file.strip_prefix(&root).unwrap_or(file);
            offenders.push(format!("  {}:{line}", shown.display()));
        }
    }

    assert!(
        offenders.is_empty(),
        "unsafe usages without a `// SAFETY:` justification (within {SAFETY_LOOKBACK} preceding lines):\n{}",
        offenders.join("\n")
    );
}
