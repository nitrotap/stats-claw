//! Goodness-of-fit equivalence: Kolmogorov–Smirnov (one/two-sample),
//! Anderson–Darling, and Shapiro–Wilk against scipy fixtures.

use crate::common;
use stats_claw::tests_stat::goodness_of_fit::{
    anderson_darling, anderson_normal_critical_values, ks_one_sample, ks_one_sample_mode,
    ks_two_sample, ks_two_sample_mode, shapiro_wilk,
};
use stats_claw::tests_stat::{Alternative, Mode};

const STAT_RTOL: f64 = 1e-8;
/// Exact KS p-values come from finite-`n` algorithms, so they match scipy's
/// `method="exact"` to floating-point noise.
const P_EXACT_ATOL: f64 = 1e-8;

#[test]
fn ks_one_sample_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_ks")?;
    let sample = common::f64s(&fx, "sample")?;
    // Against the standard normal, as in scipy.stats.kstest(x, "norm").
    let r = ks_one_sample(&sample, 0.0, 1.0)?;
    common::assert_close(
        r.statistic,
        common::scalar(&fx, "one_statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(r.p_value, common::scalar(&fx, "one_p")?, 1e-3, 1e-3);
    Ok(())
}

#[test]
fn anderson_darling_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_anderson")?;
    let sample = common::f64s(&fx, "sample")?;
    let r = anderson_darling(&sample)?;
    common::assert_close(r.statistic, common::scalar(&fx, "statistic")?, 0.0, 1e-8);

    let refs = common::f64s(&fx, "critical_values")?;
    let got = anderson_normal_critical_values(sample.len());
    common::assert_vec_close(&got, &refs, 1e-3, 0.0);
    Ok(())
}

#[test]
fn shapiro_wilk_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_shapiro")?;
    let sample = common::f64s(&fx, "sample")?;
    let r = shapiro_wilk(&sample)?;
    // W statistic from Royston's algorithm matches to ~1e-7; the p-value is the
    // Royston normalizing transform (asymptotic band).
    common::assert_close(r.statistic, common::scalar(&fx, "statistic")?, 0.0, 1e-6);
    common::assert_close(r.p_value, common::scalar(&fx, "p_value")?, 1e-4, 1e-3);
    Ok(())
}

#[test]
fn ks_two_sample_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_ks")?;
    let a = common::f64s(&fx, "sample")?;
    let b = common::f64s(&fx, "sample_b")?;
    let r = ks_two_sample(&a, &b)?;
    common::assert_close(
        r.statistic,
        common::scalar(&fx, "two_statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(r.p_value, common::scalar(&fx, "two_p")?, 1e-3, 1e-3);
    Ok(())
}

#[test]
fn ks_one_sample_exact_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_ks_exact")?;
    let sample = common::f64s(&fx, "sample")?;
    for (alt, suffix) in [
        (Alternative::TwoSided, "two_sided"),
        (Alternative::Less, "less"),
        (Alternative::Greater, "greater"),
    ] {
        let r = ks_one_sample_mode(&sample, 0.0, 1.0, alt, Mode::Exact)?;
        let d_ref = common::opt(&fx, &format!("one_statistic_{suffix}")).ok_or("D")?;
        let p_ref = common::opt(&fx, &format!("one_p_{suffix}")).ok_or("p")?;
        common::assert_close(r.statistic, d_ref, 0.0, STAT_RTOL);
        common::assert_close(r.p_value, p_ref, P_EXACT_ATOL, 0.0);
    }
    Ok(())
}

#[test]
fn ks_two_sample_exact_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_ks_exact")?;
    let a = common::f64s(&fx, "sample")?;
    let b = common::f64s(&fx, "sample_b")?;
    let r = ks_two_sample_mode(&a, &b, Mode::Exact)?;
    common::assert_close(
        r.statistic,
        common::scalar(&fx, "two_statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(r.p_value, common::scalar(&fx, "two_p")?, P_EXACT_ATOL, 0.0);
    Ok(())
}
