//! Nonparametric-test equivalence: Mann–Whitney U, Kruskal–Wallis H, Wilcoxon
//! signed-rank, and Friedman against scipy fixtures.

use crate::common;
use stats_claw::tests_stat::nonparametric::{
    friedman, kruskal_wallis, mann_whitney_u, mann_whitney_u_mode, wilcoxon_signed_rank,
    wilcoxon_signed_rank_mode,
};
use stats_claw::tests_stat::{Alternative, Mode};

const STAT_RTOL: f64 = 1e-8;
const P_ATOL: f64 = 1e-6;
/// Exact p-values are rational, so they must match scipy to floating-point noise.
const P_EXACT_ATOL: f64 = 1e-8;

/// Loads the `groups` field (array of numeric arrays) from a fixture.
fn groups_of(fx: &serde_json::Value) -> Vec<Vec<f64>> {
    common::matrix(fx, "groups").unwrap_or_default()
}

#[test]
fn mann_whitney_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_mann_whitney")?;
    let a = common::f64s(&fx, "a")?;
    let b = common::f64s(&fx, "b")?;

    for (alt, suffix) in [
        (Alternative::TwoSided, "two_sided"),
        (Alternative::Less, "less"),
        (Alternative::Greater, "greater"),
    ] {
        let r = mann_whitney_u(&a, &b, alt, true)?;
        let u_ref = common::opt(&fx, &format!("u_{suffix}")).ok_or("u")?;
        let p_ref = common::opt(&fx, &format!("p_{suffix}")).ok_or("p")?;
        common::assert_close(r.statistic, u_ref, 0.0, STAT_RTOL);
        common::assert_close(r.p_value, p_ref, P_ATOL, 0.0);
    }
    Ok(())
}

#[test]
fn wilcoxon_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_wilcoxon")?;
    let a = common::f64s(&fx, "a")?;
    let b = common::f64s(&fx, "b")?;
    for (alt, suffix) in [
        (Alternative::TwoSided, "two_sided"),
        (Alternative::Less, "less"),
        (Alternative::Greater, "greater"),
    ] {
        let r = wilcoxon_signed_rank(&a, &b, alt, true)?;
        let w_ref = common::opt(&fx, &format!("w_{suffix}")).ok_or("w")?;
        let p_ref = common::opt(&fx, &format!("p_{suffix}")).ok_or("p")?;
        common::assert_close(r.statistic, w_ref, 0.0, STAT_RTOL);
        common::assert_close(r.p_value, p_ref, P_ATOL, 0.0);
    }
    Ok(())
}

#[test]
fn friedman_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_friedman")?;
    let measurements = common::matrix(&fx, "measurements")?;
    let views: Vec<&[f64]> = measurements.iter().map(Vec::as_slice).collect();
    let r = friedman(&views)?;
    common::assert_close(
        r.statistic,
        common::scalar(&fx, "statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(r.p_value, common::scalar(&fx, "p_value")?, P_ATOL, 0.0);
    common::assert_close(r.df.ok_or("df")?, common::scalar(&fx, "df")?, 0.0, 0.0);
    common::assert_close(
        r.effect_size.ok_or("W")?,
        common::scalar(&fx, "kendalls_w")?,
        0.0,
        1e-8,
    );
    Ok(())
}

#[test]
fn kruskal_wallis_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_kruskal")?;
    let groups = groups_of(&fx);
    let views: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();
    let r = kruskal_wallis(&views)?;
    common::assert_close(
        r.statistic,
        common::scalar(&fx, "statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(r.p_value, common::scalar(&fx, "p_value")?, P_ATOL, 0.0);
    common::assert_close(r.df.ok_or("df")?, common::scalar(&fx, "df")?, 0.0, 0.0);
    Ok(())
}

#[test]
fn mann_whitney_exact_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_mann_whitney_exact")?;
    let a = common::f64s(&fx, "a")?;
    let b = common::f64s(&fx, "b")?;
    for (alt, suffix) in [
        (Alternative::TwoSided, "two_sided"),
        (Alternative::Less, "less"),
        (Alternative::Greater, "greater"),
    ] {
        let r = mann_whitney_u_mode(&a, &b, alt, true, Mode::Exact)?;
        let u_ref = common::opt(&fx, &format!("u_{suffix}")).ok_or("u")?;
        let p_ref = common::opt(&fx, &format!("p_{suffix}")).ok_or("p")?;
        common::assert_close(r.statistic, u_ref, 0.0, STAT_RTOL);
        common::assert_close(r.p_value, p_ref, P_EXACT_ATOL, 0.0);
    }
    Ok(())
}

#[test]
fn wilcoxon_exact_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_wilcoxon_exact")?;
    let a = common::f64s(&fx, "a")?;
    let b = common::f64s(&fx, "b")?;
    for (alt, suffix) in [
        (Alternative::TwoSided, "two_sided"),
        (Alternative::Less, "less"),
        (Alternative::Greater, "greater"),
    ] {
        let r = wilcoxon_signed_rank_mode(&a, &b, alt, true, Mode::Exact)?;
        let w_ref = common::opt(&fx, &format!("w_{suffix}")).ok_or("w")?;
        let p_ref = common::opt(&fx, &format!("p_{suffix}")).ok_or("p")?;
        common::assert_close(r.statistic, w_ref, 0.0, STAT_RTOL);
        common::assert_close(r.p_value, p_ref, P_EXACT_ATOL, 0.0);
    }
    Ok(())
}
