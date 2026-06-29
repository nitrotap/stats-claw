//! Correlation-test equivalence: Pearson, Spearman, and Kendall against scipy
//! fixtures, across all three alternatives.

use crate::common;
use stats_claw::tests_stat::correlation::{kendall, pearson, spearman};
use stats_claw::tests_stat::{Alternative, TestResult};

const STAT_RTOL: f64 = 1e-8;
const P_ATOL: f64 = 1e-6;

type CorrFn = fn(&[f64], &[f64], Alternative) -> Result<TestResult, stats_claw::error::Error>;

/// Asserts one correlation routine matches its r/ρ/τ statistic and all three
/// alternative p-values.
fn check(
    fx: &serde_json::Value,
    x: &[f64],
    y: &[f64],
    name: &str,
    stat_key: &str,
    run: CorrFn,
) -> Result<(), Box<dyn std::error::Error>> {
    let stat_ref = common::opt(fx, stat_key).ok_or("stat")?;
    let two = run(x, y, Alternative::TwoSided)?;
    common::assert_close(two.statistic, stat_ref, 0.0, STAT_RTOL);
    for (alt, suffix) in [
        (Alternative::TwoSided, "two_sided"),
        (Alternative::Less, "less"),
        (Alternative::Greater, "greater"),
    ] {
        let p_ref = common::opt(fx, &format!("{name}_p_{suffix}")).ok_or("p")?;
        common::assert_close(run(x, y, alt)?.p_value, p_ref, P_ATOL, 0.0);
    }
    Ok(())
}

#[test]
fn correlations_match_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_correlation")?;
    let x = common::f64s(&fx, "x")?;
    let y = common::f64s(&fx, "y")?;
    check(&fx, &x, &y, "pearson", "pearson_r", pearson)?;
    check(&fx, &x, &y, "spearman", "spearman_r", spearman)?;
    check(&fx, &x, &y, "kendall", "kendall_tau", kendall)?;
    Ok(())
}

#[test]
fn pearson_rejects_zero_variance() {
    let constant = [2.0, 2.0, 2.0, 2.0];
    let y = [1.0, 2.0, 3.0, 4.0];
    assert!(
        pearson(&constant, &y, Alternative::TwoSided).is_err(),
        "constant input must be a typed error for Pearson"
    );
}
