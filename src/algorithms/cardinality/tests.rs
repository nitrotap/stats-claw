//! Unit tests for the `HyperLogLog` cardinality base block, added one behaviour at a
//! time.
//!
//! These exercise the deterministic hash, register update, the estimator's
//! accuracy on known multisets, and every typed error path against hand-computed
//! or theoretically-bounded values. The exact-distinct-count accuracy fixtures
//! live in `tests/equiv/cardinality.rs`.

use super::*;

/// The finalizer hash is deterministic and injective on a representative key range.
#[test]
fn hash_is_deterministic_and_injective() {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for k in 0_u64..10_000 {
        let h = hash64(k);
        assert_eq!(h, hash64(k), "hash must be deterministic for key {k}");
        assert!(seen.insert(h), "hash collided on a distinct key {k}");
    }
}

/// A `HyperLogLog` is created for any precision in the supported `[4, 18]` range,
/// allocating `m = 2^p` zeroed registers.
#[test]
fn new_allocates_two_to_the_p_registers() -> Result<(), CardinalityError> {
    let hll = HyperLogLog::new(10)?;
    assert_eq!(
        hll.register_count(),
        1 << 10,
        "precision 10 -> 1024 registers"
    );
    assert_eq!(hll.precision(), 10, "precision must round-trip");
    Ok(())
}

/// Precisions outside the supported `[4, 18]` range are rejected.
#[test]
fn new_rejects_out_of_range_precision() {
    assert_eq!(
        HyperLogLog::new(3).err(),
        Some(CardinalityError::InvalidPrecision),
        "precision 3 must be rejected"
    );
    assert_eq!(
        HyperLogLog::new(19).err(),
        Some(CardinalityError::InvalidPrecision),
        "precision 19 must be rejected"
    );
}

/// Adding one element sets exactly one register, and re-adding the same element
/// (the register update is an idempotent maximum) changes nothing.
#[test]
fn add_sets_one_register_and_is_idempotent() -> Result<(), CardinalityError> {
    let mut hll = HyperLogLog::new(10)?;
    hll.add(42);
    let nonzero = hll.nonzero_register_count();
    assert_eq!(nonzero, 1, "one element must touch exactly one register");
    // Re-adding the identical element is an idempotent max: no register changes.
    hll.add(42);
    assert_eq!(
        hll.nonzero_register_count(),
        1,
        "re-adding the same element must not change any register"
    );
    Ok(())
}

/// An estimator with no elements added estimates a cardinality of zero.
#[test]
fn empty_estimator_counts_zero() -> Result<(), CardinalityError> {
    let hll = HyperLogLog::new(12)?;
    let est = hll.estimate();
    assert!(est.abs() < 1e-9, "empty estimate should be 0, was {est}");
    Ok(())
}

/// A moderate number of distinct elements is estimated within a few standard
/// errors of the exact count (the small-range linear-counting regime then the raw
/// estimate).
#[test]
fn estimates_distinct_count_within_error_bound() -> Result<(), CardinalityError> {
    let p = 14_u8;
    let n = 50_000_u64;
    let mut hll = HyperLogLog::new(p)?;
    for x in 0..n {
        hll.add(x);
    }
    let est = hll.estimate();
    let exact = 50_000.0;
    // Standard error 1.04/sqrt(m); allow 3x as a deterministic safety band.
    let m = f64::from(1_u32 << p);
    let std_err = 1.04 / m.sqrt();
    let rel = (est - exact).abs() / exact;
    assert!(
        rel <= 3.0 * std_err,
        "relative error {rel} exceeded 3x standard error {std_err} (est {est})"
    );
    Ok(())
}

/// A small distinct count (most registers still empty) is estimated accurately via
/// the small-range linear-counting correction.
#[test]
fn small_cardinality_uses_linear_counting() -> Result<(), CardinalityError> {
    let p = 14_u8;
    let n = 200_u64;
    let hll = HyperLogLog::from_u64_iter(p, 0..n)?;
    let est = hll.estimate();
    let exact = 200.0;
    let m = f64::from(1_u32 << p);
    let std_err = 1.04 / m.sqrt();
    let rel = (est - exact).abs() / exact;
    // Linear counting is far tighter than 3x SE in this regime; assert the band.
    assert!(
        rel <= 3.0 * std_err,
        "small-range relative error {rel} exceeded 3x standard error {std_err} (est {est})"
    );
    Ok(())
}

/// Building from an iterator yields the same registers (hence the same estimate) as
/// adding each element one by one.
#[test]
fn from_u64_iter_matches_incremental_add() -> Result<(), CardinalityError> {
    let p = 12_u8;
    let data: Vec<u64> = (0..5_000).collect();

    let built = HyperLogLog::from_u64_iter(p, data.iter().copied())?;

    let mut incremental = HyperLogLog::new(p)?;
    for &x in &data {
        incremental.add(x);
    }
    assert!(
        (built.estimate() - incremental.estimate()).abs() < 1e-9,
        "iterator build must match incremental add: {} vs {}",
        built.estimate(),
        incremental.estimate()
    );
    Ok(())
}

/// The estimate is deterministic: the same precision and multiset (in any order)
/// give bit-identical estimates.
#[test]
fn estimate_is_order_independent_and_deterministic() -> Result<(), CardinalityError> {
    let p = 12_u8;
    // Same multiset (0..3_000) fed forward and reversed.
    let a = HyperLogLog::from_u64_iter(p, 0..3_000)?;
    let b = HyperLogLog::from_u64_iter(p, (0..3_000).rev())?;
    // The register update is an order-independent max, so estimates match exactly.
    assert_eq!(
        a.estimate().to_bits(),
        b.estimate().to_bits(),
        "estimate must be identical regardless of insertion order"
    );
    Ok(())
}
