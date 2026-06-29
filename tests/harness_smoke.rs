//! Smoke tests for the equivalence harness: tolerance assertions, golden-fixture
//! loading, and the identifiability helpers. These verify the harness itself
//! before any family suite depends on it.

mod common;

use common::HarnessError;

/// `assert_close` accepts a value inside the absolute tolerance band.
#[test]
fn assert_close_respects_tolerance() {
    common::assert_close(1.0 + 1e-11, 1.0, 1e-10, 0.0);
}

/// Two NaNs compare equal (scipy emits NaN for undefined moments; the harness
/// must treat matching NaNs as a pass rather than a failure).
#[test]
fn assert_close_treats_dual_nan_as_equal() {
    common::assert_close(f64::NAN, f64::NAN, 0.0, 0.0);
}

/// The loader reads a committed fixture and exposes its scalars and arrays.
#[test]
fn load_reads_committed_smoke_fixture() -> Result<(), HarnessError> {
    let fx = common::load("harness_smoke")?;
    let scalar = fx
        .get("scalar")
        .and_then(serde_json::Value::as_f64)
        .ok_or(HarnessError::Shape("scalar"))?;
    common::assert_close(scalar, 1.5, 0.0, 0.0);
    let values = common::f64s(&fx, "values")?;
    common::assert_vec_close(&values, &[0.0, 1.0, 2.5], 0.0, 0.0);
    Ok(())
}

/// ARI is exactly 1.0 when a partition is compared with itself.
#[test]
fn ari_is_one_for_identical_partitions() {
    let p = [0usize, 0, 1, 1, 2];
    assert!(
        (common::adjusted_rand_index(&p, &p) - 1.0).abs() < 1e-12,
        "ARI of a partition with itself must be 1.0"
    );
}

/// ARI is invariant to cluster relabeling: the same partition under a label
/// permutation still scores 1.0.
#[test]
fn ari_is_permutation_invariant() {
    let a = [0usize, 0, 1, 1];
    let b = [1usize, 1, 0, 0];
    assert!(
        (common::adjusted_rand_index(&a, &b) - 1.0).abs() < 1e-12,
        "ARI must ignore which integer labels a cluster"
    );
}

/// ARI is below 1.0 (and near 0) for an unrelated partition.
#[test]
fn ari_drops_for_disagreeing_partitions() {
    let a = [0usize, 0, 0, 1, 1, 1];
    let b = [0usize, 1, 0, 1, 0, 1];
    let ari = common::adjusted_rand_index(&a, &b);
    assert!(
        ari < 0.5,
        "disagreeing partitions should score well below 1.0, got {ari}"
    );
}

/// `align_sign` flips a negatively-correlated vector to point with the reference.
#[test]
fn align_sign_flips_anti_aligned_vector() {
    let mut actual = [-1.0, -2.0, -3.0];
    let reference = [1.0, 2.0, 3.0];
    common::align_sign(&mut actual, &reference);
    common::assert_vec_close(&actual, &[1.0, 2.0, 3.0], 0.0, 0.0);
}

/// `align_sign` leaves an already-aligned vector unchanged.
#[test]
fn align_sign_preserves_aligned_vector() {
    let mut actual = [1.0, 2.0, 3.0];
    let reference = [1.0, 2.0, 3.0];
    common::align_sign(&mut actual, &reference);
    common::assert_vec_close(&actual, &[1.0, 2.0, 3.0], 0.0, 0.0);
}

/// The KS statistic of a sample against its own empirical step CDF is small;
/// here we check the closed form against a hand-computed value.
#[test]
fn ks_statistic_matches_hand_computation() {
    // Sample {0.0, 0.5, 1.0} vs the uniform[0,1] CDF F(x)=x.
    // At x=0.0: D+ = 1/3 - 0 = 1/3, D- = 0 - 0 = 0.
    // The supremum over the three points is 1/3.
    let sample = [0.0, 0.5, 1.0];
    let d = common::ks_statistic(&sample, |x| x);
    common::assert_close(d, 1.0 / 3.0, 1e-12, 0.0);
}

/// The 5% KS critical value is `1.36 / sqrt(n)`.
#[test]
fn ks_critical_05_is_known_constant() {
    common::assert_close(common::ks_critical_05(100), 0.136, 1e-12, 0.0);
}
