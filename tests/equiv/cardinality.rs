//! Accuracy suite for the cardinality-estimation (`HyperLogLog`) family.
//!
//! IMPORTANT — this is an **exact-ground-truth accuracy bound**, NOT a library
//! bit-match. `HyperLogLog` has no canonical scipy / numpy / scikit-learn reference
//! to diff against the way the regression family diffs against
//! `sklearn.LinearRegression`. The fixture instead records the **exact** distinct
//! count of each test stream (computed offline with Python's built-in `set` — the
//! unarguable ground truth), and this suite asserts that the `HyperLogLog` estimate
//! lands inside `HyperLogLog`'s **theoretical** relative standard-error band,
//! `1.04 / √m` with `m = 2^precision`. There is no claim of reproducing another
//! library's bits — only that the estimator meets its own published accuracy
//! guarantee against the true count.
//!
//! Cross-language stream reproduction: storing the 1k / 10k / 100k-element streams
//! as JSON would bloat the fixture, so each stream is defined by a small,
//! fully-documented 64-bit linear congruential generator (LCG) whose multiplier,
//! increment, seed, and element-space modulus the fixture carries. This suite
//! regenerates the *identical* stream from those parameters, so the multiset the
//! Rust estimator sees is byte-for-byte the one Python counted exactly.

use crate::common;
use crate::common::HarnessError;
use stats_claw::algorithms::cardinality::HyperLogLog;

/// Modulus of the 64-bit LCG state (`2^64`), applied via `wrapping_*` arithmetic.
///
/// The Python generator computes `mod 2^64`; `u64` wrapping multiply/add is the
/// exact same arithmetic, so both sides produce identical states.
const fn lcg_next(state: u64, mult: u64, inc: u64) -> u64 {
    state.wrapping_mul(mult).wrapping_add(inc)
}

/// Regenerates the documented LCG stream of `length` elements modulo `modulus`.
///
/// Mirrors `gen_cardinality.py::_lcg_stream` exactly: advance the state, then take
/// `state % modulus` as each element, so the Rust estimator sees the same multiset
/// (and hence the same exact distinct count) Python recorded.
fn lcg_stream(seed: u64, mult: u64, inc: u64, modulus: u64, length: usize) -> Vec<u64> {
    let mut state = seed;
    let mut out = Vec::with_capacity(length);
    for _ in 0..length {
        state = lcg_next(state, mult, inc);
        out.push(state % modulus);
    }
    out
}

/// Reads a `u64` field from the fixture, mapping a shape error into the harness type.
fn u64_field(fx: &serde_json::Value, key: &'static str) -> Result<u64, HarnessError> {
    fx.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or(HarnessError::Shape(key))
}

/// The `HyperLogLog` estimate lands within the theoretical error band of the exact
/// distinct count for every fixture stream.
///
/// The accuracy gate is `|estimate − exact| / exact ≤ error_bound_multiple ×
/// standard_error`, where `standard_error = 1.04 / √m` is `HyperLogLog`'s own
/// published relative standard error and `error_bound_multiple` (3.0) is the fixed
/// safety factor recorded in the fixture. This is the estimator meeting its
/// theoretical guarantee against the true count — not a library equivalence.
#[test]
fn hyperloglog_estimate_within_theoretical_error_bound() -> Result<(), HarnessError> {
    let fx = common::load("cardinality")?;

    let mult = u64_field(&fx, "lcg_multiplier")?;
    let inc = u64_field(&fx, "lcg_increment")?;
    let seed = u64_field(&fx, "seed")?;
    let modulus = u64_field(&fx, "element_modulus")?;
    let precision = u8::try_from(common::u64_at(&fx, "precision")?)
        .map_err(|_| HarnessError::Shape("precision"))?;
    let standard_error = common::scalar(&fx, "standard_error")?;
    let bound_multiple = common::scalar(&fx, "error_bound_multiple")?;
    let allowed_rel = bound_multiple * standard_error;

    let cases = fx
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape("cases"))?;
    assert!(!cases.is_empty(), "fixture must carry at least one case");

    for case in cases {
        let length = usize::try_from(
            case.get("length")
                .and_then(serde_json::Value::as_u64)
                .ok_or(HarnessError::Shape("length"))?,
        )
        .map_err(|_| HarnessError::Shape("length"))?;
        let exact = common::field(case, "exact_distinct")?;

        let stream = lcg_stream(seed, mult, inc, modulus, length);
        let hll = HyperLogLog::from_u64_iter(precision, stream)
            .map_err(|e| HarnessError::Parse(format!("HyperLogLog build failed: {e}")))?;
        let estimate = hll.estimate();

        let rel = (estimate - exact).abs() / exact;
        assert!(
            rel <= allowed_rel,
            "stream length {length}: estimate {estimate} vs exact {exact} \
             -> relative error {rel} exceeded {bound_multiple}x standard error \
             {standard_error} (bound {allowed_rel})"
        );
    }
    Ok(())
}

/// Sanity guard: the Rust LCG reproduces a stream whose own exact distinct count
/// (computed here with a `HashSet`) matches the fixture's Python `set` ground truth.
///
/// This proves the cross-language stream regeneration is faithful — if the Rust LCG
/// diverged from Python's, the exact counts would differ and the accuracy test above
/// would be meaningless.
#[test]
fn rust_lcg_reproduces_pythons_exact_distinct_count() -> Result<(), HarnessError> {
    use std::collections::HashSet;
    let fx = common::load("cardinality")?;
    let mult = u64_field(&fx, "lcg_multiplier")?;
    let inc = u64_field(&fx, "lcg_increment")?;
    let seed = u64_field(&fx, "seed")?;
    let modulus = u64_field(&fx, "element_modulus")?;

    let cases = fx
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape("cases"))?;
    for case in cases {
        let length = usize::try_from(
            case.get("length")
                .and_then(serde_json::Value::as_u64)
                .ok_or(HarnessError::Shape("length"))?,
        )
        .map_err(|_| HarnessError::Shape("length"))?;
        let exact = common::field(case, "exact_distinct")?;

        let stream = lcg_stream(seed, mult, inc, modulus, length);
        let distinct = stream.iter().collect::<HashSet<_>>().len();
        let distinct_f = f64::from(u32::try_from(distinct).unwrap_or(u32::MAX));
        assert!(
            (distinct_f - exact).abs() < 0.5,
            "stream length {length}: Rust exact distinct {distinct} != Python {exact}"
        );
    }
    Ok(())
}
