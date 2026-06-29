//! AC-5 deferred interval items now satisfiable with the categorical track's
//! fixtures: Cramér's V bootstrap CI (QA-CAT-010), delta bootstrap CI
//! (QA-CAT-095), a coverage-rate simulation, and a Beta-posterior credible
//! interval (AC-5 Story 5.2).
//!
//! Provenance note: the `test_cramers_boot` / `test_boot_delta` fixtures are NOT
//! independent scipy references — their bounds are produced by stats-claw's OWN
//! `SplitMix64` seeded resample, replayed in Python so the committed numbers match
//! the Rust path bit-for-bit (see their `_provenance.library`). The two tests over
//! them are therefore **determinism / regression** checks (same seed + same scheme
//! ⇒ identical draws), not equivalence-to-scipy checks. Genuine scipy/analytic
//! equivalence is covered separately: the percentile-CI *coverage* simulation
//! below, the Beta credible interval against `scipy.stats.beta.ppf`
//! (`test_beta_credible`), and the resampling/categorical modules' own
//! Monte-Carlo-error tests.

use crate::common;
use stats_claw::resampling::{
    beta_credible_interval, bootstrap_statistic, coverage_rate, percentile_ci,
};
use stats_claw::rng::SplitMix64;
use stats_claw::tests_stat::categorical::cramers_v_bootstrap_ci;

/// Mean of a slice, used as the bootstrap statistic for the delta interval.
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = u32::try_from(xs.len()).unwrap_or(u32::MAX);
    xs.iter().sum::<f64>() / f64::from(n)
}

#[test]
fn cramers_v_bootstrap_ci_is_deterministic_regression() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_cramers_boot")?;
    let table = common::matrix(&fx, "table")?;
    let seed = common::u64_at(&fx, "seed")?;
    let b = common::usize_at(&fx, "b")?;
    let alpha = common::scalar(&fx, "alpha")?;
    let (lo, hi) = cramers_v_bootstrap_ci(&table, b, alpha, &mut SplitMix64::new(seed))?;
    // Regression / determinism check: the fixture bounds were produced by stats-claw's
    // OWN SplitMix64 resample (not scipy), so the same seed + same observation-
    // resampling scheme reproduces identical draws and the bounds match exactly.
    // This pins the resampling pipeline against drift; it is NOT a scipy reference.
    common::assert_close(lo, common::scalar(&fx, "ci_low")?, 1e-9, 1e-9);
    common::assert_close(hi, common::scalar(&fx, "ci_high")?, 1e-9, 1e-9);
    Ok(())
}

#[test]
fn delta_bootstrap_ci_is_deterministic_regression() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_boot_delta")?;
    let before = common::f64s(&fx, "before")?;
    let after = common::f64s(&fx, "after")?;
    let delta: Vec<f64> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
    let seed = common::u64_at(&fx, "seed")?;
    let b = common::usize_at(&fx, "b")?;
    let alpha = common::scalar(&fx, "alpha")?;

    let stats = bootstrap_statistic(&delta, b, &mut SplitMix64::new(seed), mean)?;
    let (lo, hi) = percentile_ci(&stats, alpha)?;
    // Regression / determinism check against stats-claw's own seeded resample (see the
    // module-level provenance note): same seed + scheme ⇒ identical draws. Not a
    // scipy reference.
    common::assert_close(lo, common::scalar(&fx, "ci_low")?, 1e-9, 1e-9);
    common::assert_close(hi, common::scalar(&fx, "ci_high")?, 1e-9, 1e-9);
    Ok(())
}

#[test]
fn percentile_ci_coverage_is_near_nominal() {
    // Over many seeded replications, a 95% percentile CI for the mean of a known
    // normal should cover the true mean about 95% of the time.
    let rate = coverage_rate(0.0, 1.0, 40, 400, 0.05, &mut SplitMix64::new(2024));
    common::assert_close(rate, 0.95, 0.07, 0.0);
}

/// C5 — Zero-variance sample returns a documented degenerate point interval.
///
/// When all bootstrap statistic values are identical (zero variance), `percentile_ci`
/// returns `Ok((value, value))`: a degenerate interval whose lower and upper bounds
/// are equal. This is a valid, documented edge-case — the interval contains only
/// the single observed value. No panic occurs.
///
/// # Errors
///
/// Returns an error if `percentile_ci` unexpectedly errors on the zero-variance input.
#[test]
fn percentile_ci_zero_variance_returns_degenerate_interval() -> Result<(), stats_claw::error::Error>
{
    let zero_var: Vec<f64> = vec![5.0; 200];
    let (lo, hi) = percentile_ci(&zero_var, 0.05)?;
    assert!(
        (lo - 5.0).abs() < 1e-12,
        "degenerate lower bound must equal the constant value, got {lo}"
    );
    assert!(
        (hi - 5.0).abs() < 1e-12,
        "degenerate upper bound must equal the constant value, got {hi}"
    );
    assert!(
        lo <= hi,
        "lower bound {lo} must not exceed upper bound {hi} for a degenerate interval"
    );
    Ok(())
}

/// C5 — A single-element sample returns a documented degenerate point interval.
///
/// With one observation, both the `alpha/2` and `1-alpha/2` ranks map to the
/// same element. `percentile_ci` returns `Ok((x, x))` — a degenerate interval.
/// This is too small to form a stable interval in practice (bootstrapping a
/// single value yields all-identical resamples), but the function succeeds
/// without panicking and emits a documented result.
///
/// # Errors
///
/// Returns an error if `percentile_ci` unexpectedly errors on the single-element input.
#[test]
fn percentile_ci_single_element_returns_degenerate_interval() -> Result<(), stats_claw::error::Error>
{
    let single = &[42.0_f64];
    let (lo, hi) = percentile_ci(single, 0.05)?;
    assert!(
        (lo - 42.0).abs() < 1e-12,
        "degenerate lower bound must equal the sole element, got {lo}"
    );
    assert!(
        (hi - 42.0).abs() < 1e-12,
        "degenerate upper bound must equal the sole element, got {hi}"
    );
    assert!(
        lo <= hi,
        "lower bound {lo} must not exceed upper bound {hi}"
    );
    Ok(())
}

/// C5 — A small-but-structured sample (n=3) returns the extreme-element interval.
///
/// `percentile_ci` consumes the bootstrap *statistic distribution*, not raw
/// observations: the slice passed to it is a vector of per-resample statistic
/// values, so "n=3" here means three bootstrap statistic draws, which is far too
/// small for a stable interval in practice. With alpha=0.05, the 2.5th percentile
/// rank maps to index 0 and the 97.5th percentile rank maps to index 2, so the
/// function returns the min and max of the three values — the full observed range.
/// No panic occurs and the interval is well-ordered (lo ≤ hi).
///
/// # Errors
///
/// Returns an error if `percentile_ci` unexpectedly errors on this input.
#[test]
fn percentile_ci_small_structured_sample_returns_full_range() -> Result<(), stats_claw::error::Error>
{
    // Three bootstrap-statistic values: a small-but-non-degenerate distribution.
    // With alpha=0.05 and n=3, floor_rank(0.025, 3)=0 and floor_rank(0.975, 3)=2,
    // so the interval spans [min, max] of the sorted slice.
    let stats = &[1.0_f64, 2.0, 3.0];
    let (lo, hi) = percentile_ci(stats, 0.05)?;
    assert!(
        (lo - 1.0).abs() < 1e-12,
        "lower bound must be the minimum element (1.0), got {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-12,
        "upper bound must be the maximum element (3.0), got {hi}"
    );
    assert!(
        lo <= hi,
        "lower bound {lo} must not exceed upper bound {hi}"
    );
    Ok(())
}

#[test]
fn beta_credible_interval_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_beta_credible")?;
    let alpha0 = common::scalar(&fx, "alpha0")?;
    let beta0 = common::scalar(&fx, "beta0")?;
    let successes = common::scalar(&fx, "successes")?;
    let trials = common::scalar(&fx, "trials")?;
    let alpha = common::scalar(&fx, "alpha")?;
    let (lo, hi) = beta_credible_interval(alpha0, beta0, successes, trials, alpha)?;
    common::assert_close(lo, common::scalar(&fx, "ci_low")?, 1e-9, 1e-9);
    common::assert_close(hi, common::scalar(&fx, "ci_high")?, 1e-9, 1e-9);
    Ok(())
}
