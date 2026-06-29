//! Log-space / extreme-p-value equivalence (AC-2 Story 2.2, P5).
//!
//! Asserts the distribution `logsf`/`logcdf` and the test-level `log_p_value`
//! match `scipy.stats` at extreme inputs where the linear path underflows to
//! `0.0`. Fixtures: `test_logsf.json` (per-distribution `<dist>.logsf/logcdf`)
//! and `test_extreme_p.json` (a one-sample t with a huge effect).

use crate::common;
use stats_claw::distributions::LogCdf;
use stats_claw::distributions::{
    ChiSquaredDistribution, FDistribution, NormalDistribution, TDistribution,
};
use stats_claw::tests_stat::{parametric::t_test_1samp, Alternative};

/// A boxed test error, so helpers can `?` on both fixture and assertion failures.
type TestError = Box<dyn std::error::Error>;

/// Relative tolerance on a log value of magnitude `≥ 1`. A log value near `0`
/// (a non-tail point) has unbounded relative sensitivity to a tiny absolute
/// error, so for `|log| < LOG_NEAR_ZERO` we assert absolute tolerance instead.
const LOG_RTOL: f64 = 1e-9;
/// Absolute tolerance applied where the log value is near zero.
const LOG_ATOL: f64 = 1e-9;
/// Magnitude below which a log value is treated as "near zero" (use `LOG_ATOL`).
const LOG_NEAR_ZERO: f64 = 1.0;

/// Asserts `got` matches the scipy reference `want` on a log scale: relative for
/// magnitudes `≥ 1`, absolute for near-zero (non-tail) values.
fn assert_log_close(got: f64, want: f64, label: &str) {
    let (atol, rtol) = if want.abs() < LOG_NEAR_ZERO {
        (LOG_ATOL, 0.0)
    } else {
        (0.0, LOG_RTOL)
    };
    common::assert_close(got, want, atol, rtol);
    assert!(got.is_finite(), "{label}: log value was not finite ({got})");
}

/// Checks `logsf`/`logcdf` of `dist` against the fixture grid for one
/// distribution sub-block, zipping the parallel `x` / `logsf` / `logcdf` arrays
/// so no slice indexing is needed.
fn check_dist(dist: &impl LogCdf, block: &serde_json::Value, name: &str) -> Result<(), TestError> {
    let xs = common::f64s(block, "x")?;
    let sf = common::f64s(block, "logsf")?;
    let cdf = common::f64s(block, "logcdf")?;
    for ((&x, &want_sf), &want_cdf) in xs.iter().zip(&sf).zip(&cdf) {
        assert_log_close(dist.logsf(x), want_sf, &format!("{name} logsf"));
        assert_log_close(dist.logcdf(x), want_cdf, &format!("{name} logcdf"));
    }
    Ok(())
}

/// Reads a small integer degrees-of-freedom field (stored as a JSON integer) as
/// the `i64` the generated distribution structs carry — no `f64 → int` cast.
fn df_of(obj: &serde_json::Value, key: &'static str) -> Result<i64, TestError> {
    obj.get(key)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| format!("fixture missing integer df field {key}").into())
}

/// Asserts the four distributions' `logsf`/`logcdf` reproduce scipy at every
/// fixture grid point, including the deep tail where the linear `1 - cdf` is 0.0.
#[test]
fn logsf_logcdf_match_scipy() -> Result<(), TestError> {
    let fx = common::load("test_logsf")?;

    let normal = fx.get("normal").ok_or("normal")?;
    let n = NormalDistribution {
        mean: 0.0,
        standard_deviation: 1.0,
        ..Default::default()
    };
    check_dist(&n, normal, "normal")?;

    let t = fx.get("t").ok_or("t")?;
    let t_dist = TDistribution {
        degrees_of_freedom: df_of(t, "df")?,
        ..Default::default()
    };
    check_dist(&t_dist, t, "t")?;

    let chi2 = fx.get("chi2").ok_or("chi2")?;
    let chi2_dist = ChiSquaredDistribution {
        degrees_of_freedom: df_of(chi2, "df")?,
        ..Default::default()
    };
    check_dist(&chi2_dist, chi2, "chi2")?;

    let f = fx.get("f").ok_or("f")?;
    let f_dist = FDistribution {
        numerator_df: df_of(f, "d1")?,
        denominator_df: df_of(f, "d2")?,
        ..Default::default()
    };
    check_dist(&f_dist, f, "f")?;
    Ok(())
}

/// The one-sample t-test's `log_p_value` matches scipy at an extreme effect, for
/// all three alternatives, while the linear two-sided p-value has underflowed to
/// exactly `0.0` — the failure the log-space path fixes.
#[test]
fn extreme_log_p_matches_scipy() -> Result<(), TestError> {
    let fx = common::load("test_extreme_p")?;
    let sample = common::f64s(&fx, "sample")?;
    let popmean = common::scalar(&fx, "popmean")?;

    let two = t_test_1samp(&sample, popmean, Alternative::TwoSided)?;
    let log_two = two.log_p_value.ok_or("no log p (two-sided)")?;
    assert_log_close(
        log_two,
        common::scalar(&fx, "log_p_two_sided")?,
        "t-test two-sided log p",
    );

    let greater = t_test_1samp(&sample, popmean, Alternative::Greater)?;
    assert_log_close(
        greater.log_p_value.ok_or("no log p (greater)")?,
        common::scalar(&fx, "log_p_greater")?,
        "t-test greater log p",
    );

    let less = t_test_1samp(&sample, popmean, Alternative::Less)?;
    assert_log_close(
        less.log_p_value.ok_or("no log p (less)")?,
        common::scalar(&fx, "log_p_less")?,
        "t-test less log p",
    );

    // The linear two-sided p-value has underflowed to exactly 0.0 (stats-claw's
    // `1 - cdf(|t|)` rounds to 0 once `cdf → 1`), while `log_two` above is finite
    // and scipy-accurate — exactly the failure the log-space path fixes.
    assert_eq!(
        two.p_value.to_bits(),
        0.0_f64.to_bits(),
        "linear two-sided p should underflow to 0.0, was {}",
        two.p_value
    );
    assert!(
        log_two.is_finite() && log_two < -40.0,
        "log p should be finite and far negative, was {log_two}"
    );
    Ok(())
}
