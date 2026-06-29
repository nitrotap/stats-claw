//! Equivalence suite for the linear-regression family (OLS + ridge).
//!
//! Each test loads a committed `scikit-learn` golden fixture and asserts the
//! stats-claw estimator agrees on the fitted intercept, every coefficient, the
//! in-sample R², and the held-out grid predictions, all within a stated
//! tolerance. Python never runs here — the fixtures are the offline source of
//! truth.

use crate::common;
use crate::common::HarnessError;
use serde_json::Value;
use stats_claw::algorithms::regression::{RegressionError, ols, ridge};

/// Relative/absolute tolerance for the regression equivalence comparisons.
///
/// OLS and ridge both reduce to the same dense normal-equations solve `scikit`
/// uses (its dense `cholesky` path), so the agreement is tight. The gate is set
/// at `1e-9`; the actual achieved max-abs difference across intercept,
/// coefficients, and R² is ~`4.5e-13` for OLS and ~`4.3e-14` for ridge
/// (essentially machine precision against `scikit-learn` 1.9.0).
const ATOL: f64 = 1e-9;
const RTOL: f64 = 1e-9;

/// Parses the `data` key of a fixture as a row-major design matrix.
fn data_matrix(fx: &Value, key: &'static str) -> Result<Vec<Vec<f64>>, HarnessError> {
    fx.get(key)
        .and_then(Value::as_array)
        .ok_or(HarnessError::Shape(key))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or(HarnessError::Shape(key))?
                .iter()
                .map(|v| v.as_f64().ok_or(HarnessError::Shape(key)))
                .collect()
        })
        .collect()
}

/// Maps a borrowed [`RegressionError`] into the harness error type so tests use
/// `?` on a fit result.
fn fit_err(e: &RegressionError) -> HarnessError {
    HarnessError::Parse(format!("regression fit failed: {e}"))
}

/// Asserts the fitted model agrees with the fixture on intercept, coefficients,
/// R², and grid predictions.
fn assert_agrees(
    model: &stats_claw::algorithms::regression::LinearModel,
    fx: &Value,
    x: &[Vec<f64>],
    y: &[f64],
) -> Result<(), HarnessError> {
    common::assert_close(
        model.intercept(),
        common::scalar(fx, "intercept")?,
        ATOL,
        RTOL,
    );
    common::assert_vec_close(
        model.coefficients(),
        &common::f64s(fx, "coefficients")?,
        ATOL,
        RTOL,
    );
    common::assert_close(
        model.r_squared(x, y),
        common::scalar(fx, "r_squared")?,
        ATOL,
        RTOL,
    );
    let grid = data_matrix(fx, "grid")?;
    let expected = common::f64s(fx, "grid_predictions")?;
    let got: Vec<f64> = grid.iter().map(|row| model.predict(row)).collect();
    common::assert_vec_close(&got, &expected, ATOL, RTOL);
    Ok(())
}

#[test]
fn ols_agrees_with_sklearn_linear_regression() -> Result<(), HarnessError> {
    let fx = common::load("regression_ols")?;
    let x = data_matrix(&fx, "data")?;
    let y = common::f64s(&fx, "target")?;
    let model = ols(&x, &y).map_err(|e| fit_err(&e))?;
    assert_agrees(&model, &fx, &x, &y)
}

#[test]
fn ridge_agrees_with_sklearn_ridge() -> Result<(), HarnessError> {
    let fx = common::load("regression_ridge")?;
    let x = data_matrix(&fx, "data")?;
    let y = common::f64s(&fx, "target")?;
    let alpha = common::scalar(&fx, "alpha")?;
    let model = ridge(&x, &y, alpha).map_err(|e| fit_err(&e))?;
    assert_agrees(&model, &fx, &x, &y)
}
