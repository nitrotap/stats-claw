//! Equivalence and reproducibility suite for the resampling module.
//!
//! Reproducibility tests assert that seeded draws are byte-identical across runs
//! and seed-sensitive (AC-5 Story 5.1). The interval test asserts our percentile
//! CI of a seeded median bootstrap matches the committed `scipy.stats.bootstrap`
//! reference within Monte-Carlo relative error (AC-5 Story 5.2, QA-DIST-084).

use crate::common;

use stats_claw::resampling::{
    bootstrap_indices, bootstrap_statistic, kfold_indices, percentile_ci, permutation,
};
use stats_claw::rng::SplitMix64;

/// Sample median of `xs` (linear interpolation between the two central order
/// statistics for even lengths), matching `numpy.median` used by the reference.
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let mid = n / 2;
    if n % 2 == 1 {
        s.get(mid).copied().unwrap_or(f64::NAN)
    } else {
        let a = s.get(mid - 1).copied().unwrap_or(f64::NAN);
        let b = s.get(mid).copied().unwrap_or(f64::NAN);
        0.5 * (a + b)
    }
}

#[test]
fn bootstrap_is_reproducible() {
    let a = bootstrap_indices(50, 100, &mut SplitMix64::new(1));
    let b = bootstrap_indices(50, 100, &mut SplitMix64::new(1));
    assert_eq!(
        a, b,
        "identical seeds must yield identical resample collections"
    );
}

#[test]
fn bootstrap_is_seed_sensitive() {
    let a = bootstrap_indices(50, 100, &mut SplitMix64::new(1));
    let c = bootstrap_indices(50, 100, &mut SplitMix64::new(2));
    assert_ne!(
        a, c,
        "distinct seeds must drive distinct resample collections"
    );
}

#[test]
fn kfold_is_a_partition() {
    let folds = kfold_indices(20, 5, &mut SplitMix64::new(2));
    let mut seen = vec![0u32; 20];
    for (_, test) in &folds {
        for &i in test {
            if let Some(c) = seen.get_mut(i) {
                *c += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "every observation must appear in exactly one test fold, got counts {seen:?}"
    );
}

#[test]
fn permutation_is_reproducible() {
    let a = permutation(100, &mut SplitMix64::new(5));
    let b = permutation(100, &mut SplitMix64::new(5));
    assert_eq!(a, b, "identical seeds must yield identical permutations");
}

#[test]
fn permutation_is_seed_sensitive() {
    let a = permutation(100, &mut SplitMix64::new(5));
    let c = permutation(100, &mut SplitMix64::new(6));
    assert_ne!(a, c, "distinct seeds must drive distinct permutations");
}

#[test]
fn kfold_is_reproducible() {
    let a = kfold_indices(20, 5, &mut SplitMix64::new(2));
    let b = kfold_indices(20, 5, &mut SplitMix64::new(2));
    assert_eq!(a, b, "identical seeds must yield identical CV splits");
}

#[test]
fn percentile_ci_brackets_center() -> Result<(), stats_claw::error::Error> {
    let xs: Vec<f64> = (0..1000).map(f64::from).collect();
    let (lo, hi) = percentile_ci(&xs, 0.05)?;
    assert!(
        lo < 500.0 && hi > 500.0,
        "90% CI must bracket the center: lo={lo}, hi={hi}"
    );
    Ok(())
}

#[test]
fn percentile_ci_rejects_empty_input() {
    assert_eq!(
        percentile_ci(&[], 0.05),
        Err(stats_claw::error::Error::EmptyInput),
        "empty samples must be a typed EmptyInput error, not a panic"
    );
}

#[test]
fn kfold_train_and_test_are_complementary() {
    let n = 20;
    let folds = kfold_indices(n, 5, &mut SplitMix64::new(3));
    for (train, test) in &folds {
        assert_eq!(
            train.len() + test.len(),
            n,
            "train and test sizes must sum to n for each fold"
        );
    }
}

#[test]
fn median_bootstrap_ci_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = common::load("resamp_median_ci")?;
    let data = common::f64s(&fixture, "data")?;
    let alpha = fixture
        .get("alpha")
        .and_then(serde_json::Value::as_f64)
        .ok_or("alpha missing")?;
    let ref_low = fixture
        .get("ci_low")
        .and_then(serde_json::Value::as_f64)
        .ok_or("ci_low missing")?;
    let ref_high = fixture
        .get("ci_high")
        .and_then(serde_json::Value::as_f64)
        .ok_or("ci_high missing")?;

    // Our own seeded bootstrap distribution of the median, then its percentile CI.
    let stats = bootstrap_statistic(&data, 20_000, &mut SplitMix64::new(12345), median)?;
    let (lo, hi) = percentile_ci(&stats, alpha)?;

    // Within Monte-Carlo relative error of the scipy reference (rel <= 1e-2 per
    // the build plan / QA-DIST-084 within-MC convention). Our RNG stream differs
    // from scipy's, so only the bootstrap distribution converges, not the bounds
    // exactly; the tolerance absorbs the residual MC gap at large B.
    common::assert_close(lo, ref_low, 0.0, 1e-2);
    common::assert_close(hi, ref_high, 0.0, 1e-2);
    Ok(())
}
