//! Unit tests for the [`super`] regression estimators.
//!
//! Split into its own file (included via `#[path]`) so `mod.rs` stays inside the
//! project's 500-line `style.rs` cap while the tests still compile under
//! `cargo test` with access to the module's private items.

use super::{RegressionError, ols, ridge};

/// OLS recovers the exact coefficients of a noise-free linear relationship.
#[test]
fn ols_recovers_exact_line() -> Result<(), RegressionError> {
    // y = 2 + 3*x, sampled at four points (no noise).
    let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![2.0, 5.0, 8.0, 11.0];
    let model = ols(&x, &y)?;
    assert!(
        (model.intercept() - 2.0).abs() < 1e-9,
        "intercept was {}",
        model.intercept()
    );
    let slope = model.coefficients().first().copied().unwrap_or(f64::NAN);
    assert!((slope - 3.0).abs() < 1e-9, "slope was {slope}");
    Ok(())
}

/// OLS recovers exact coefficients for a two-predictor noise-free plane.
#[test]
fn ols_recovers_multivariate_plane() -> Result<(), RegressionError> {
    // y = 1 + 2*x₁ − 0.5*x₂ on four non-collinear points.
    let x = vec![
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 5.0],
    ];
    let y: Vec<f64> = x
        .iter()
        .map(|r| {
            let x1 = r.first().copied().unwrap_or(0.0);
            let x2 = r.get(1).copied().unwrap_or(0.0);
            (-0.5_f64).mul_add(x2, 2.0_f64.mul_add(x1, 1.0))
        })
        .collect();
    let model = ols(&x, &y)?;
    assert!(
        (model.intercept() - 1.0).abs() < 1e-9,
        "intercept was {}",
        model.intercept()
    );
    let b1 = model.coefficients().first().copied().unwrap_or(f64::NAN);
    let b2 = model.coefficients().get(1).copied().unwrap_or(f64::NAN);
    assert!((b1 - 2.0).abs() < 1e-9, "b1 was {b1}");
    assert!((b2 + 0.5).abs() < 1e-9, "b2 was {b2}");
    Ok(())
}

/// `predict` reconstructs the noise-free targets exactly from the fitted model.
#[test]
fn predict_matches_targets_on_exact_fit() -> Result<(), RegressionError> {
    let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![2.0, 5.0, 8.0, 11.0];
    let model = ols(&x, &y)?;
    for (row, &target) in x.iter().zip(&y) {
        let pred = model.predict(row);
        assert!((pred - target).abs() < 1e-9, "pred {pred} vs {target}");
    }
    Ok(())
}

/// `r_squared` is 1.0 for a perfect fit.
#[test]
fn r_squared_is_one_for_perfect_fit() -> Result<(), RegressionError> {
    let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![2.0, 5.0, 8.0, 11.0];
    let model = ols(&x, &y)?;
    let r2 = model.r_squared(&x, &y);
    assert!((r2 - 1.0).abs() < 1e-9, "R² was {r2}");
    Ok(())
}

/// `r_squared` matches the closed-form `1 − SS_res/SS_tot` on a noisy fit.
#[test]
fn r_squared_matches_closed_form_on_noisy_fit() -> Result<(), RegressionError> {
    // Hand-computable case: points (1,1),(2,3),(3,2),(4,5). The OLS line is
    // ŷ = 0 + 1.1·x (slope 5.5/5 = 1.1, intercept 0). Predictions are
    // 1.1, 2.2, 3.3, 4.4, so SS_res = 0.01+0.64+1.69+0.36 = 2.70 and
    // SS_tot = Σ(y−2.75)² = 8.75, giving R² = 1 − 2.70/8.75.
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y = vec![1.0, 3.0, 2.0, 5.0];
    let model = ols(&x, &y)?;
    let r2 = model.r_squared(&x, &y);
    let want = 1.0 - 2.70 / 8.75;
    assert!((r2 - want).abs() < 1e-9, "R² was {r2}, want {want}");
    Ok(())
}

/// Ridge shrinks the slope toward zero as the penalty grows, with the intercept
/// approaching the target mean (the penalty never touches the intercept).
#[test]
fn ridge_shrinks_slope_toward_zero() -> Result<(), RegressionError> {
    let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![2.0, 5.0, 8.0, 11.0];
    let ols_slope = ols(&x, &y)?
        .coefficients()
        .first()
        .copied()
        .unwrap_or(f64::NAN);
    let small = ridge(&x, &y, 1.0)?;
    let large = ridge(&x, &y, 1000.0)?;
    let small_slope = small.coefficients().first().copied().unwrap_or(f64::NAN);
    let large_slope = large.coefficients().first().copied().unwrap_or(f64::NAN);
    assert!(
        small_slope < ols_slope,
        "ridge slope {small_slope} !< OLS {ols_slope}"
    );
    assert!(
        large_slope < small_slope,
        "heavier penalty must shrink more"
    );
    assert!(large_slope > 0.0, "slope crossed zero: {large_slope}");
    // The intercept always satisfies β₀ = ȳ − β₁·x̄ (here ȳ=6.5, x̄=1.5).
    let recovered = large_slope.mul_add(1.5, large.intercept());
    assert!((recovered - 6.5).abs() < 1e-9, "β₀+β₁·x̄ was {recovered}");
    assert!(
        large.intercept() > small.intercept(),
        "a heavier penalty must push the intercept toward the target mean"
    );
    Ok(())
}

/// An empty design matrix is a clean [`RegressionError::EmptyInput`].
#[test]
fn empty_input_is_an_error() {
    assert_eq!(ols(&[], &[]), Err(RegressionError::EmptyInput));
}

/// Mismatched row counts surface a [`RegressionError::RowMismatch`].
#[test]
fn row_mismatch_is_an_error() {
    let x = vec![vec![1.0], vec![2.0]];
    let y = vec![1.0];
    assert_eq!(
        ols(&x, &y),
        Err(RegressionError::RowMismatch {
            rows_x: 2,
            rows_y: 1
        })
    );
}

/// Rows of unequal width surface a [`RegressionError::RaggedRows`].
#[test]
fn ragged_rows_are_an_error() {
    let x = vec![vec![1.0, 2.0], vec![3.0]];
    let y = vec![1.0, 2.0];
    assert_eq!(ols(&x, &y), Err(RegressionError::RaggedRows));
}

/// A negative penalty is rejected as [`RegressionError::InvalidPenalty`].
#[test]
fn negative_penalty_is_an_error() {
    let x = vec![vec![1.0], vec![2.0]];
    let y = vec![1.0, 2.0];
    assert_eq!(ridge(&x, &y, -1.0), Err(RegressionError::InvalidPenalty));
}

/// Perfectly collinear predictors make the OLS normal equations singular.
#[test]
fn collinear_predictors_are_singular_under_ols() {
    // Column 2 is exactly 2× column 1, so XᵀX is rank-deficient.
    let x = vec![
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
        vec![4.0, 8.0],
    ];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(ols(&x, &y), Err(RegressionError::Singular));
}

/// Ridge regularises the same collinear system into a unique solution.
#[test]
fn ridge_resolves_collinear_predictors() -> Result<(), RegressionError> {
    let x = vec![
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
        vec![4.0, 8.0],
    ];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let model = ridge(&x, &y, 0.5)?;
    assert_eq!(model.coefficients().len(), 2, "two slopes expected");
    // Predictions remain finite and close to the targets.
    let r2 = model.r_squared(&x, &y);
    assert!(r2 > 0.9, "ridge R² on collinear data was {r2}");
    Ok(())
}

/// A deterministic fit: identical inputs yield byte-identical coefficients.
#[test]
fn fit_is_deterministic() -> Result<(), RegressionError> {
    let x = vec![
        vec![0.3, 1.2],
        vec![1.7, 0.4],
        vec![2.1, 3.3],
        vec![3.9, 2.0],
    ];
    let y = vec![1.5, 2.2, 4.8, 5.1];
    let a = ridge(&x, &y, 0.7)?;
    let b = ridge(&x, &y, 0.7)?;
    assert_eq!(a, b, "ridge fit must be deterministic");
    Ok(())
}
