//! Categorical-test equivalence: chi-squared independence, Cramér's V, Fisher
//! exact, `McNemar`, and Cochran's Q against scipy/statsmodels fixtures.

use crate::common;
use stats_claw::tests_stat::categorical::{
    chi_squared_independence, cochrans_q, cramers_v, fisher_exact, mcnemar,
};
use stats_claw::tests_stat::Alternative;

#[test]
fn chi2_independence_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_chi2_independence")?;
    let table = common::matrix(&fx, "table")?;
    let r = chi_squared_independence(&table)?;
    common::assert_close(r.statistic, common::scalar(&fx, "statistic")?, 0.0, 1e-8);
    common::assert_close(r.p_value, common::scalar(&fx, "p_value")?, 1e-8, 0.0);
    common::assert_close(
        r.df.ok_or("df missing")?,
        common::scalar(&fx, "df")?,
        0.0,
        0.0,
    );
    Ok(())
}

#[test]
fn cramers_v_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_cramers_v")?;
    let table = common::matrix(&fx, "table")?;
    let v = cramers_v(&table)?;
    common::assert_close(v, common::scalar(&fx, "cramers_v")?, 0.0, 1e-8);
    Ok(())
}

#[test]
fn cramers_v_is_zero_for_independent_table() -> Result<(), Box<dyn std::error::Error>> {
    // Rows proportional to one another → perfect independence → V = 0.
    let table = vec![vec![10.0, 20.0], vec![20.0, 40.0]];
    let v = cramers_v(&table)?;
    common::assert_close(v, 0.0, 1e-12, 0.0);
    Ok(())
}

#[test]
fn fisher_exact_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_fisher_exact")?;
    let table = common::matrix(&fx, "table")?;
    let two = fisher_exact(&table, Alternative::TwoSided)?;
    common::assert_close(two.p_value, common::scalar(&fx, "p_two_sided")?, 1e-8, 0.0);
    common::assert_close(
        two.effect_size.ok_or("odds ratio")?,
        common::scalar(&fx, "odds_ratio")?,
        0.0,
        1e-8,
    );
    common::assert_close(
        fisher_exact(&table, Alternative::Less)?.p_value,
        common::scalar(&fx, "p_less")?,
        1e-8,
        0.0,
    );
    common::assert_close(
        fisher_exact(&table, Alternative::Greater)?.p_value,
        common::scalar(&fx, "p_greater")?,
        1e-8,
        0.0,
    );
    Ok(())
}

#[test]
fn mcnemar_matches_statsmodels() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_mcnemar")?;
    let large = common::matrix(&fx, "large")?;
    let small = common::matrix(&fx, "small")?;

    let asym = mcnemar(&large, false, false)?;
    common::assert_close(
        asym.statistic,
        common::scalar(&fx, "stat_large")?,
        0.0,
        1e-8,
    );
    common::assert_close(asym.p_value, common::scalar(&fx, "p_large")?, 1e-6, 0.0);

    let corrected = mcnemar(&large, false, true)?;
    common::assert_close(
        corrected.statistic,
        common::scalar(&fx, "stat_large_cc")?,
        0.0,
        1e-8,
    );
    common::assert_close(
        corrected.p_value,
        common::scalar(&fx, "p_large_cc")?,
        1e-6,
        0.0,
    );

    let exact = mcnemar(&small, true, false)?;
    common::assert_close(
        exact.p_value,
        common::scalar(&fx, "p_small_exact")?,
        1e-8,
        0.0,
    );
    Ok(())
}

#[test]
fn mcnemar_zero_discordant_is_p_one() -> Result<(), Box<dyn std::error::Error>> {
    // b = c = 0 → statistic undefined → p = 1, no division by zero.
    let table = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
    let r = mcnemar(&table, false, false)?;
    common::assert_close(r.p_value, 1.0, 0.0, 0.0);
    Ok(())
}

#[test]
fn cochrans_q_matches_statsmodels() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_cochran_q")?;
    let data = common::matrix(&fx, "data")?;
    let rows: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let r = cochrans_q(&rows)?;
    common::assert_close(r.statistic, common::scalar(&fx, "statistic")?, 0.0, 1e-8);
    common::assert_close(r.p_value, common::scalar(&fx, "p_value")?, 1e-6, 0.0);
    common::assert_close(r.df.ok_or("df")?, common::scalar(&fx, "df")?, 0.0, 0.0);
    Ok(())
}

#[test]
fn cochrans_q_all_equal_rows_is_zero() -> Result<(), Box<dyn std::error::Error>> {
    // Every subject responds identically across treatments → Q = 0, p = 1.
    let data = [vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0]];
    let rows: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let r = cochrans_q(&rows)?;
    common::assert_close(r.statistic, 0.0, 0.0, 0.0);
    common::assert_close(r.p_value, 1.0, 0.0, 0.0);
    Ok(())
}

#[test]
fn chi2_independence_rejects_degenerate_table() {
    // A single-row table has no association to test → typed error, not a panic.
    let single_row = vec![vec![1.0, 2.0, 3.0]];
    assert!(
        chi_squared_independence(&single_row).is_err(),
        "single-row table must be a typed error"
    );
    assert!(
        chi_squared_independence(&[]).is_err(),
        "empty table must be a typed error"
    );
}
