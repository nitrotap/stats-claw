//! Shared test harness: golden-fixture loading, NaN-aware tolerance assertions,
//! and identifiability helpers (ARI, sign alignment, KS statistic).
//!
//! Every family equivalence suite loads committed JSON from
//! `reference/golden/<name>.json` and compares Rust output against it. Python is
//! never invoked at test time — the fixtures are the offline source of truth.
//!
//! Resolution: fixtures sit in `reference/` (a crate-root sibling of `src/` and
//! `tests/`), so the protected `style.rs` folder scan never counts them. The
//! loader joins `CARGO_MANIFEST_DIR` with `reference/golden`.

// This module is `#[path]`-included by each integration-test binary, so Rust
// compiles a private copy per binary. `unreachable_pub` therefore fires on the
// shared `pub` surface, and `dead_code` fires for helpers a given binary does
// not call — both are false positives for a shared test harness. Allowed
// deliberately rather than degrading the cross-file `pub` API.
#![allow(unreachable_pub, dead_code)]

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

/// Failure modes when loading or shaping a golden fixture.
///
/// Returned (never panicked) so test bodies surface a clean error via `?` and
/// stay clear of the crate's `unwrap_used`/`panic` lint gate.
#[derive(Debug)]
pub enum HarnessError {
    /// The fixture file could not be read from disk; carries its display path.
    Read(String),
    /// The fixture text was not valid JSON; carries the parser message.
    Parse(String),
    /// A fixture key was missing or had the wrong JSON shape; carries the key.
    Shape(&'static str),
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(p) => write!(f, "could not read fixture: {p}"),
            Self::Parse(m) => write!(f, "could not parse fixture: {m}"),
            Self::Shape(k) => write!(f, "fixture key has unexpected shape: {k}"),
        }
    }
}

impl std::error::Error for HarnessError {}

/// Loads and parses the golden fixture named `name` (without the `.json` suffix).
///
/// # Arguments
///
/// * `name` — fixture basename; resolved to `reference/golden/<name>.json`.
///
/// # Returns
///
/// The parsed JSON document.
///
/// # Errors
///
/// Returns [`HarnessError::Read`] if the file cannot be read, or
/// [`HarnessError::Parse`] if its contents are not valid JSON.
pub fn load(name: &str) -> Result<serde_json::Value, HarnessError> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("reference/golden")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| HarnessError::Read(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&raw).map_err(|e| HarnessError::Parse(e.to_string()))
}

/// Extracts the array at `key` as a `Vec<f64>`.
///
/// # Arguments
///
/// * `v` — a fixture document.
/// * `key` — the top-level key whose value must be a JSON array of numbers.
///
/// # Returns
///
/// The array's elements as `f64`.
///
/// # Errors
///
/// Returns [`HarnessError::Shape`] if `key` is absent, not an array, or holds a
/// non-numeric element.
pub fn f64s(v: &serde_json::Value, key: &'static str) -> Result<Vec<f64>, HarnessError> {
    let arr = v
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape(key))?;
    arr.iter()
        .map(|x| x.as_f64().ok_or(HarnessError::Shape(key)))
        .collect()
}

/// Extracts the scalar `f64` at top-level `key`.
///
/// # Arguments
///
/// * `v` — a fixture document.
/// * `key` — a top-level key whose value must be a JSON number.
///
/// # Errors
///
/// Returns [`HarnessError::Shape`] if `key` is absent or not numeric.
pub fn scalar(v: &serde_json::Value, key: &'static str) -> Result<f64, HarnessError> {
    v.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or(HarnessError::Shape(key))
}

/// Extracts the scalar `f64` at a runtime-computed `key`, returning `None` when
/// absent or non-numeric.
///
/// Lets a test build the key dynamically (e.g. `format!("p_{suffix}")`) without
/// the `&'static str` lifetime that [`scalar`] requires.
#[must_use]
pub fn opt(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(serde_json::Value::as_f64)
}

/// Extracts the scalar `f64` at `key` within a nested object `obj`.
///
/// # Arguments
///
/// * `obj` — a JSON object (e.g. a sub-block of a fixture).
/// * `key` — a key whose value must be a JSON number.
///
/// # Errors
///
/// Returns [`HarnessError::Shape`] if `key` is absent or not numeric.
pub fn field(obj: &serde_json::Value, key: &'static str) -> Result<f64, HarnessError> {
    obj.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or(HarnessError::Shape(key))
}

/// Extracts an unsigned integer at top-level `key`.
///
/// # Errors
///
/// Returns [`HarnessError::Shape`] if `key` is absent or not a `u64`.
pub fn u64_at(v: &serde_json::Value, key: &'static str) -> Result<u64, HarnessError> {
    v.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or(HarnessError::Shape(key))
}

/// Extracts a `usize` at top-level `key`.
///
/// # Errors
///
/// Returns [`HarnessError::Shape`] if `key` is absent, not a `u64`, or exceeds
/// `usize` range.
pub fn usize_at(v: &serde_json::Value, key: &'static str) -> Result<usize, HarnessError> {
    usize::try_from(u64_at(v, key)?).map_err(|_| HarnessError::Shape(key))
}

/// Extracts a contingency `table` (array of numeric rows) at top-level `key`.
///
/// # Arguments
///
/// * `v` — a fixture document.
/// * `key` — a top-level key whose value must be a JSON array of numeric arrays.
///
/// # Errors
///
/// Returns [`HarnessError::Shape`] if `key` is absent, not an array of arrays, or
/// holds a non-numeric element.
pub fn matrix(v: &serde_json::Value, key: &'static str) -> Result<Vec<Vec<f64>>, HarnessError> {
    let rows = v
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape(key))?;
    rows.iter()
        .map(|row| {
            row.as_array()
                .ok_or(HarnessError::Shape(key))?
                .iter()
                .map(|x| x.as_f64().ok_or(HarnessError::Shape(key)))
                .collect()
        })
        .collect()
}

/// Asserts `actual` matches `expected` within `|a-e| <= atol + rtol*|e|`.
///
/// Two NaNs are treated as equal so fixtures recording undefined quantities
/// (scipy's NaN) pass.
///
/// # Panics
///
/// Panics (failing the test) when the values differ by more than the tolerance.
pub fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    if actual.is_nan() && expected.is_nan() {
        return;
    }
    let diff = (actual - expected).abs();
    let tol = rtol.mul_add(expected.abs(), atol);
    assert!(
        diff <= tol,
        "not within tolerance: actual={actual}, expected={expected}, \
         diff={diff}, atol={atol}, rtol={rtol}"
    );
}

/// Asserts two slices match elementwise within tolerance (NaN-aware), failing on
/// any length or value mismatch.
///
/// # Panics
///
/// Panics (failing the test) on a length mismatch or any element outside the
/// `atol + rtol*|e|` band.
pub fn assert_vec_close(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a.is_nan() && e.is_nan() {
            continue;
        }
        let diff = (a - e).abs();
        let tol = rtol.mul_add(e.abs(), atol);
        assert!(
            diff <= tol,
            "index {i}: actual={a}, expected={e}, diff={diff}, atol={atol}, rtol={rtol}"
        );
    }
}

/// Computes `C(x, 2) = x(x-1)/2` as an `f64`, the pair count used by the ARI.
fn comb2(x: u64) -> f64 {
    let x = pairs_input(x);
    x * (x - 1.0) / 2.0
}

/// Widens a count to `f64` losslessly for the combinatorial sums (counts here are
/// far below `2^53`, so the conversion is exact).
fn pairs_input(x: u64) -> f64 {
    let lo = u32::try_from(x & 0xFFFF_FFFF).unwrap_or(0);
    let hi = u32::try_from(x >> 32).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Adjusted Rand Index between two label vectors (Hubert & Arabie 1985).
///
/// Returns `1.0` for identical partitions and is invariant to cluster relabeling,
/// making it the agreement score the clustering suites assert against.
///
/// # Arguments
///
/// * `a` — first label assignment.
/// * `b` — second label assignment; must be the same length as `a`.
///
/// # Returns
///
/// The adjusted Rand index in `(-∞, 1.0]` (`1.0` = perfect agreement).
///
/// # Panics
///
/// Panics (failing the test) if `a` and `b` differ in length.
#[must_use]
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len(), "label vectors differ in length");
    let mut contingency: HashMap<(usize, usize), u64> = HashMap::new();
    let mut row: HashMap<usize, u64> = HashMap::new();
    let mut col: HashMap<usize, u64> = HashMap::new();
    for (&ai, &bi) in a.iter().zip(b) {
        *contingency.entry((ai, bi)).or_default() += 1;
        *row.entry(ai).or_default() += 1;
        *col.entry(bi).or_default() += 1;
    }
    let sum_ij: f64 = contingency.values().map(|&v| comb2(v)).sum();
    let sum_a: f64 = row.values().map(|&v| comb2(v)).sum();
    let sum_b: f64 = col.values().map(|&v| comb2(v)).sum();
    let total = comb2(count_to_u64(a.len()));
    let expected = sum_a * sum_b / total;
    let max = 0.5 * (sum_a + sum_b);
    if (max - expected).abs() < 1e-15 {
        return 1.0;
    }
    (sum_ij - expected) / (max - expected)
}

/// Widens a `usize` length to `u64` without an `as` cast.
fn count_to_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// Flips the sign of `actual` in place when doing so increases its alignment with
/// `reference` (decomposition components are sign-ambiguous, so the suites align
/// before comparing).
pub fn align_sign(actual: &mut [f64], reference: &[f64]) {
    let dot: f64 = actual.iter().zip(reference).map(|(a, r)| a * r).sum();
    if dot < 0.0 {
        for a in actual.iter_mut() {
            *a = -*a;
        }
    }
}

/// Computes the one-sample Kolmogorov–Smirnov statistic `sup|F_n − F|`.
///
/// # Arguments
///
/// * `sorted_sample` — the sample in ascending order.
/// * `cdf` — the reference CDF evaluated pointwise.
///
/// # Returns
///
/// The supremum distance between the empirical and reference CDFs.
pub fn ks_statistic(sorted_sample: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
    let n = sorted_sample.len();
    if n == 0 {
        return 0.0;
    }
    let n_f = count_to_u64(n);
    let n_f = pairs_input(n_f);
    let mut d = 0.0_f64;
    for (i, &x) in sorted_sample.iter().enumerate() {
        let f = cdf(x);
        let i_f = pairs_input(count_to_u64(i));
        let d_plus = (i_f + 1.0) / n_f - f;
        let d_minus = f - i_f / n_f;
        d = d.max(d_plus).max(d_minus);
    }
    d
}

/// Returns the 5%-level KS critical value `1.36 / sqrt(n)` for sample size `n`.
#[must_use]
pub fn ks_critical_05(n: usize) -> f64 {
    1.36 / pairs_input(count_to_u64(n)).sqrt()
}
