//! Style tests — deterministic, zero-dependency guards on the source tree.
//!
//! These enforce mechanical house style that clippy does not cover:
//!   * `max_500_loc_per_file`   — no source file exceeds 500 lines
//!   * `max_10_files_per_folder` — no directory holds more than 10 files
//!   * `no_as_casts`            — no `as` casts in library/binary code
//!
//! Scope: the Rust source roots (`src/` and `tests/`). `@generated` files are
//! exempt from the line-count and cast checks — machine-emitted code is not
//! hand-maintained. The `as`-cast check runs over `src/` only (production code);
//! test helpers may cast freely.
//!
//! NOTE: this file is protected by the `protect_paths` Claude Code hook, so the
//! assistant cannot weaken these checks by editing them.

use std::fs;
use std::path::{Path, PathBuf};

const MAX_LOC: usize = 500;
const MAX_FILES_PER_FOLDER: usize = 10;

/// Source roots scanned, relative to the crate root.
const SCAN_ROOTS: &[&str] = &["src", "tests"];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Collect every `.rs` file under `root` (recursively).
fn rust_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
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

/// Every directory at or below `root` (inclusive).
fn directories(root: &Path, out: &mut Vec<PathBuf>) {
    if !root.is_dir() {
        return;
    }
    out.push(root.to_path_buf());
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            directories(&path, out);
        }
    }
}

/// A file is "generated" if its first few lines carry an `@generated` marker.
fn is_generated(content: &str) -> bool {
    content.lines().take(5).any(|l| l.contains("@generated"))
}

fn rel(path: &Path) -> String {
    path.strip_prefix(crate_root())
        .unwrap_or(path)
        .display()
        .to_string()
}

#[test]
fn max_500_loc_per_file() {
    let root = crate_root();
    let mut files = Vec::new();
    for r in SCAN_ROOTS {
        rust_files(&root.join(r), &mut files);
    }

    let mut offenders = Vec::new();
    for file in &files {
        let content = fs::read_to_string(file).unwrap_or_default();
        if is_generated(&content) {
            continue;
        }
        let lines = content.lines().count();
        if lines > MAX_LOC {
            offenders.push(format!("  {} has {lines} lines (max {MAX_LOC})", rel(file)));
        }
    }

    assert!(
        offenders.is_empty(),
        "files exceed the {MAX_LOC}-line limit:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn max_10_files_per_folder() {
    let root = crate_root();
    let mut dirs = Vec::new();
    for r in SCAN_ROOTS {
        directories(&root.join(r), &mut dirs);
    }

    let mut offenders = Vec::new();
    for dir in &dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        let count = entries.flatten().filter(|e| e.path().is_file()).count();
        if count > MAX_FILES_PER_FOLDER {
            offenders.push(format!(
                "  {} holds {count} files (max {MAX_FILES_PER_FOLDER})",
                rel(dir)
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "folders exceed the {MAX_FILES_PER_FOLDER}-file limit:\n{}",
        offenders.join("\n")
    );
}

/// Remove `//` line comments and `/* */` block comments so prose like
/// "treat it as a value" never trips the cast scan. Conservative: it does not
/// track string literals, which is fine — the source has no `as`-in-string.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut in_block = false;
    let mut in_line = false;
    while let Some(&b) = bytes.get(i) {
        let next = bytes.get(i + 1).copied().unwrap_or(0);
        if in_line {
            if b == b'\n' {
                in_line = false;
                out.push('\n');
            }
            i += 1;
        } else if in_block {
            if b == b'*' && next == b'/' {
                in_block = false;
                i += 2;
            } else {
                if b == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
        } else if b == b'/' && next == b'/' {
            in_line = true;
            i += 2;
        } else if b == b'/' && next == b'*' {
            in_block = true;
            i += 2;
        } else {
            out.push(char_from(b));
            i += 1;
        }
    }
    out
}

/// Drop `<.. as ..>` qualified-path segments (e.g. `<Self as Trait>::f`) which
/// use the `as` keyword but are not casts.
fn strip_qualified_paths(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut depth = 0u32;
    for ch in line.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

fn char_from(b: u8) -> char {
    char::from(b)
}

#[test]
fn no_as_casts() {
    let root = crate_root();
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files); // production code only

    let mut offenders = Vec::new();
    for file in &files {
        let content = fs::read_to_string(file).unwrap_or_default();
        if is_generated(&content) {
            continue;
        }
        let cleaned = strip_comments(&content);
        for (idx, raw_line) in cleaned.lines().enumerate() {
            let trimmed = raw_line.trim_start();
            if trimmed.starts_with("use ") || trimmed.starts_with("pub use ") {
                continue; // `use foo as bar;` renames are not casts
            }
            let line = strip_qualified_paths(raw_line);
            let is_cast = line
                .split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == ',')
                .any(|tok| tok == "as");
            if is_cast {
                offenders.push(format!("  {}:{}", rel(file), idx + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "`as` casts are banned in src/ — found at:\n{}\nUse `From`/`TryFrom`/typed \
         constructors instead.",
        offenders.join("\n")
    );
}
