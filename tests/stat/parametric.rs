//! Parametric-test equivalence: t-tests (1-sample, independent, paired, Welch),
//! one-way ANOVA, and the Levene/Bartlett variance tests against scipy fixtures.

use crate::common;
use stats_claw::tests_stat::parametric::{
    bartlett, levene, one_way_anova, t_test_1samp, t_test_ind, t_test_paired, t_test_welch,
};
use stats_claw::tests_stat::Alternative;

/// Extracts the `groups` field (array of numeric arrays) from a fixture.
fn groups_of(fx: &serde_json::Value) -> Vec<Vec<f64>> {
    common::matrix(fx, "groups").unwrap_or_default()
}

const STAT_RTOL: f64 = 1e-8;
// p-values flow through the framework Student's t / F CDFs (regularized
// incomplete beta), whose deep-tail precision is the asymptotic 1e-6 band of the
// build-plan tolerance table, not the 1e-8 reserved for closed-form quantities.
const P_ATOL: f64 = 1e-6;

type TResult = Result<stats_claw::tests_stat::TestResult, stats_claw::error::Error>;

/// Asserts a t-test block (statistic, df, and all three alternatives) matches.
fn check_t_block(
    block: &serde_json::Value,
    run: impl Fn(Alternative) -> TResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let two = run(Alternative::TwoSided)?;
    common::assert_close(
        two.statistic,
        common::field(block, "statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(two.df.ok_or("df")?, common::field(block, "df")?, 0.0, 1e-8);
    common::assert_close(
        two.p_value,
        common::field(block, "p_two_sided")?,
        P_ATOL,
        0.0,
    );
    common::assert_close(
        run(Alternative::Less)?.p_value,
        common::field(block, "p_less")?,
        P_ATOL,
        0.0,
    );
    common::assert_close(
        run(Alternative::Greater)?.p_value,
        common::field(block, "p_greater")?,
        P_ATOL,
        0.0,
    );
    Ok(())
}

#[test]
fn t_tests_match_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_t_test")?;
    let a = common::f64s(&fx, "a")?;
    let b = common::f64s(&fx, "b")?;

    let one = fx.get("one_sample").ok_or("one_sample")?;
    let popmean = common::field(one, "popmean")?;
    check_t_block(one, |alt| t_test_1samp(&a, popmean, alt))?;

    let ind = fx.get("independent").ok_or("independent")?;
    check_t_block(ind, |alt| t_test_ind(&a, &b, alt))?;

    let paired = fx.get("paired").ok_or("paired")?;
    check_t_block(paired, |alt| t_test_paired(&a, &b, alt))?;

    let welch = fx.get("welch").ok_or("welch")?;
    check_t_block(welch, |alt| t_test_welch(&a, &b, alt))?;
    Ok(())
}

#[test]
fn one_way_anova_matches_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_anova_oneway")?;
    let groups = groups_of(&fx);
    let views: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();
    let r = one_way_anova(&views)?;
    common::assert_close(
        r.statistic,
        common::scalar(&fx, "statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(r.p_value, common::scalar(&fx, "p_value")?, P_ATOL, 0.0);
    common::assert_close(
        r.df.ok_or("df")?,
        common::scalar(&fx, "df_between")?,
        0.0,
        1e-8,
    );
    common::assert_close(
        r.effect_size.ok_or("eta^2")?,
        common::scalar(&fx, "eta_squared")?,
        0.0,
        1e-8,
    );
    Ok(())
}

#[test]
fn levene_and_bartlett_match_scipy() -> Result<(), Box<dyn std::error::Error>> {
    let fx = common::load("test_variance")?;
    let groups = groups_of(&fx);
    let views: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();

    let lev = levene(&views)?;
    common::assert_close(
        lev.statistic,
        common::scalar(&fx, "levene_statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(lev.p_value, common::scalar(&fx, "levene_p")?, P_ATOL, 0.0);
    common::assert_close(
        lev.df.ok_or("levene df")?,
        common::scalar(&fx, "levene_df_between")?,
        0.0,
        1e-8,
    );

    let bar = bartlett(&views)?;
    common::assert_close(
        bar.statistic,
        common::scalar(&fx, "bartlett_statistic")?,
        0.0,
        STAT_RTOL,
    );
    common::assert_close(bar.p_value, common::scalar(&fx, "bartlett_p")?, P_ATOL, 0.0);
    common::assert_close(
        bar.df.ok_or("bartlett df")?,
        common::scalar(&fx, "bartlett_df")?,
        0.0,
        1e-8,
    );
    Ok(())
}

#[test]
fn variance_tests_reject_constant_input() {
    // Every group constant → zero variance → typed error, not NaN.
    let a = [2.0, 2.0, 2.0];
    let b = [5.0, 5.0, 5.0];
    assert!(
        bartlett(&[&a[..], &b[..]]).is_err(),
        "all-constant groups must be a typed error for Bartlett"
    );
}

#[test]
fn t_test_rejects_zero_variance() {
    // A constant one-sample input has no variance → typed error, not NaN.
    let constant = [3.0, 3.0, 3.0, 3.0];
    assert!(
        t_test_1samp(&constant, 0.0, Alternative::TwoSided).is_err(),
        "zero-variance sample must be a typed error"
    );
}
