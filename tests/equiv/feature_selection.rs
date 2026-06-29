//! Equivalence suite for the feature-selection family.
//!
//! Loads a committed golden fixture and asserts the stats-claw selectors reproduce
//! the reference quantities they pin: the per-feature **ANOVA F-scores and
//! p-values** from `sklearn.feature_selection.f_classif` and the per-feature
//! **population variances** from `sklearn.feature_selection.VarianceThreshold`
//! (`ddof = 0`). Python never runs here — the fixture is the offline source of
//! truth.

use crate::common;
use crate::common::HarnessError;
use stats_claw::algorithms::feature_selection::{
    FeatureSelectionError, anova_f_pvalues, anova_f_scores, variance_threshold,
};

/// Tolerances for the equivalence comparisons.
///
/// The Rust ANOVA F path reuses the framework's one-way ANOVA, the same
/// computation `f_classif` performs, so the per-feature **F-scores** agree to the
/// `1e-9` gate (achieved max-abs diff ~`8.9e-15` on this fixture — essentially
/// machine precision). The **p-values** route through the framework F distribution
/// (regularized incomplete beta), whose deep-tail precision is the asymptotic
/// `1e-6` band documented for every F p-value in the build-plan tolerance table
/// (achieved ~`5.4e-17` here, since this fixture's p-values sit near `1e-4`, well
/// inside the beta tail's accurate regime). The **variances** use the same
/// `ddof = 0` divisor as `VarianceThreshold` and agree to ~`4.4e-16` (machine
/// precision); they are checked at the `1e-9` gate.
const F_ATOL: f64 = 1e-9;
const F_RTOL: f64 = 1e-9;
const P_ATOL: f64 = 1e-6;
const VAR_ATOL: f64 = 1e-9;
const VAR_RTOL: f64 = 1e-9;

/// Maps a borrowed [`FeatureSelectionError`] into the harness error type so tests
/// use `?` on a selection result.
fn select_err(e: &FeatureSelectionError) -> HarnessError {
    HarnessError::Parse(format!("selector failed: {e}"))
}

/// Reads the fixture's labelled design matrix `x` and integer class `labels`.
///
/// The matrix is stored row-major (one inner array per sample); the labels are
/// stored as integers and widened from the `f64s` reader back to `usize` for the
/// class-grouped F-test.
fn load_dataset() -> Result<(Vec<Vec<f64>>, Vec<usize>), HarnessError> {
    let fx = common::load("feature_selection")?;
    let x = common::matrix(&fx, "x")?;
    let labels_f = common::f64s(&fx, "labels")?;
    let labels: Vec<usize> = labels_f
        .iter()
        .map(|&v| usize::try_from(label_to_u64(v)).unwrap_or(0))
        .collect();
    Ok((x, labels))
}

/// Rounds a non-negative integer-valued `f64` label to `u64` without an `as` cast.
///
/// Labels in the fixture are small non-negative integers stored as JSON numbers, so
/// rounding to the nearest integer and reading the exact integer value recovers
/// them losslessly while satisfying the no-`as` lint.
fn label_to_u64(v: f64) -> u64 {
    let rounded = v.round();
    if rounded <= 0.0 {
        return 0;
    }
    // Binary-search the integer whose exact f64 widening equals `rounded`.
    let (mut lo, mut hi) = (0u64, 1_000_000u64);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if u64_to_f64(mid) <= rounded {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Widens a small `u64` to `f64` without an `as` cast (values are tiny labels).
fn u64_to_f64(n: u64) -> f64 {
    let hi = u32::try_from(n >> 32).unwrap_or(0);
    let lo = u32::try_from(n & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// The stats-claw ANOVA F-scores reproduce `sklearn.feature_selection.f_classif`.
#[test]
fn anova_f_scores_agree_with_sklearn() -> Result<(), HarnessError> {
    let fx = common::load("feature_selection")?;
    let expected = common::f64s(&fx, "f_scores")?;
    let (x, labels) = load_dataset()?;

    let scores = anova_f_scores(&x, &labels).map_err(|e| select_err(&e))?;
    common::assert_vec_close(&scores, &expected, F_ATOL, F_RTOL);
    Ok(())
}

/// The stats-claw ANOVA F p-values reproduce `f_classif`'s p-values within the F
/// distribution's documented asymptotic tail band.
#[test]
fn anova_f_pvalues_agree_with_sklearn() -> Result<(), HarnessError> {
    let fx = common::load("feature_selection")?;
    let expected = common::f64s(&fx, "p_values")?;
    let (x, labels) = load_dataset()?;

    let pvalues = anova_f_pvalues(&x, &labels).map_err(|e| select_err(&e))?;
    common::assert_vec_close(&pvalues, &expected, P_ATOL, 0.0);
    Ok(())
}

/// The stats-claw variance scores reproduce `VarianceThreshold.variances_`
/// (population variance, `ddof = 0`).
#[test]
fn variances_agree_with_sklearn() -> Result<(), HarnessError> {
    let fx = common::load("feature_selection")?;
    let expected = common::f64s(&fx, "variances")?;
    let (x, _labels) = load_dataset()?;

    // The threshold is irrelevant to the *variances* (the scores); 0.0 keeps every
    // non-constant feature and exercises the same per-feature variance arithmetic.
    let sel = variance_threshold(&x, 0.0).map_err(|e| select_err(&e))?;
    common::assert_vec_close(sel.scores(), &expected, VAR_ATOL, VAR_RTOL);
    Ok(())
}
