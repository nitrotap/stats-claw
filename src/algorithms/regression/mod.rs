//! Linear regression: ordinary least squares (OLS) and ridge (L2) regression.
//!
//! Both estimators fit a coefficient vector `β` and an intercept by the normal
//! equations. OLS minimises `‖y − Xβ‖²`; ridge adds an L2 penalty `λ‖β‖²` on the
//! slopes (the intercept is left unpenalised, matching `scikit-learn`'s `Ridge`).
//! The fit centres the design and target, solves `(X_cᵀX_c + λI)β = X_cᵀy_c` for
//! the centred slopes by Gaussian elimination with partial pivoting, and recovers
//! the intercept from the column means. This base block backs `RegressionMethod`.

mod linalg;

use linalg::{centered, column_means, mean, solve};

/// A fitted linear model: an intercept plus one slope per predictor column.
///
/// Fields are private so the fitted parameters stay an invariant pair (the
/// intercept and slopes always come from the same fit); read them through
/// [`LinearModel::intercept`] and [`LinearModel::coefficients`], and evaluate the
/// model with [`LinearModel::predict`] / [`LinearModel::r_squared`].
#[derive(Debug, Clone, PartialEq)]
pub struct LinearModel {
    /// The fitted intercept term (`β₀`); the model's prediction at the origin.
    intercept: f64,
    /// One fitted slope coefficient per predictor column, in input column order.
    coefficients: Vec<f64>,
}

impl LinearModel {
    /// Returns the fitted intercept term (`β₀`).
    #[must_use]
    pub const fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Returns the fitted slope coefficients, one per predictor column in input
    /// column order.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Predicts the target for a single observation `row`.
    ///
    /// Computes `β₀ + Σ_j β_j x_j`. Coordinates of `row` beyond the fitted slope
    /// count are ignored, and missing coordinates contribute zero, so a `row` whose
    /// length differs from the training width still yields a finite prediction
    /// (callers are responsible for passing rows of the fitted dimension).
    ///
    /// # Arguments
    ///
    /// * `row` — the predictor values for one observation, in fitted column order.
    ///
    /// # Returns
    ///
    /// The model's scalar prediction `ŷ` for `row`.
    #[must_use]
    pub fn predict(&self, row: &[f64]) -> f64 {
        self.coefficients
            .iter()
            .zip(row)
            .fold(self.intercept, |acc, (&beta, &x)| beta.mul_add(x, acc))
    }

    /// Computes the coefficient of determination `R²` of the model on `x`/`y`.
    ///
    /// `R² = 1 − SS_res / SS_tot`, where `SS_res = Σ(yᵢ − ŷᵢ)²` and
    /// `SS_tot = Σ(yᵢ − ȳ)²`. Matches `scikit-learn`'s `score`: when `SS_tot` is
    /// zero (a constant target) the result is `1.0` if the residuals also vanish,
    /// else `0.0`. `R²` can be negative when the model fits worse than the mean.
    ///
    /// # Arguments
    ///
    /// * `x` — design matrix to score against, one inner slice per observation.
    /// * `y` — observed targets, one per row of `x`.
    ///
    /// # Returns
    ///
    /// The `R²` score in `(−∞, 1.0]`. An empty `y` yields `0.0`.
    #[must_use]
    pub fn r_squared(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let y_mean = mean(y);
        let mut ss_res = 0.0_f64;
        let mut ss_tot = 0.0_f64;
        for (row, &yi) in x.iter().zip(y) {
            let resid = yi - self.predict(row);
            ss_res = resid.mul_add(resid, ss_res);
            let dev = yi - y_mean;
            ss_tot = dev.mul_add(dev, ss_tot);
        }
        if ss_tot <= 0.0 {
            return if ss_res <= 0.0 { 1.0 } else { 0.0 };
        }
        1.0 - ss_res / ss_tot
    }
}

/// Errors that prevent a linear model from being fitted.
///
/// Returned (never panicked) so callers stay clear of the crate's `unwrap`/`panic`
/// lint gate and can surface a clean diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegressionError {
    /// `x` and `y` disagreed on the number of observations (rows).
    RowMismatch {
        /// Number of rows in the design matrix `x`.
        rows_x: usize,
        /// Number of entries in the target vector `y`.
        rows_y: usize,
    },
    /// The design matrix had no rows, so no model can be fitted.
    EmptyInput,
    /// Rows of `x` did not all share the same number of columns.
    RaggedRows,
    /// The (penalised) normal-equations matrix was singular to working precision,
    /// so the coefficients are not uniquely determined.
    Singular,
    /// The ridge penalty `λ` was negative (or non-finite), which is not a valid L2
    /// regularisation strength.
    InvalidPenalty,
}

impl std::fmt::Display for RegressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RowMismatch { rows_x, rows_y } => {
                write!(f, "row count mismatch: x has {rows_x}, y has {rows_y}")
            }
            Self::EmptyInput => write!(f, "design matrix has no observations"),
            Self::RaggedRows => write!(f, "design matrix rows have unequal length"),
            Self::Singular => write!(f, "normal-equations matrix is singular"),
            Self::InvalidPenalty => {
                write!(f, "ridge penalty must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for RegressionError {}

/// Fits an ordinary-least-squares linear model to `x` and `y`.
///
/// Equivalent to `scikit-learn`'s `LinearRegression` with `fit_intercept=True`: the
/// model includes an unpenalised intercept and minimises the residual sum of
/// squares `‖y − Xβ − β₀‖²`. Internally this is ridge with a zero penalty.
///
/// # Arguments
///
/// * `x` — design matrix as one inner slice per observation; every row must have
///   the same number of predictor columns.
/// * `y` — target values, one per row of `x`.
///
/// # Returns
///
/// A [`LinearModel`] with the fitted intercept and per-column slopes.
///
/// # Errors
///
/// Returns [`RegressionError::EmptyInput`] for no rows,
/// [`RegressionError::RowMismatch`] if `x` and `y` disagree on the row count,
/// [`RegressionError::RaggedRows`] if rows differ in width, or
/// [`RegressionError::Singular`] if the normal-equations matrix is not invertible
/// (e.g. perfectly collinear predictors).
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::regression::ols;
///
/// // y = 2 + 3*x₁ exactly: the fit recovers intercept 2 and slope 3.
/// let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
/// let y = vec![2.0, 5.0, 8.0, 11.0];
/// let model = ols(&x, &y)?;
/// assert!((model.intercept() - 2.0).abs() < 1e-9, "intercept {}", model.intercept());
/// let slope = model.coefficients().first().copied().unwrap_or(f64::NAN);
/// assert!((slope - 3.0).abs() < 1e-9, "slope {slope}");
/// # Ok::<(), stats_claw::algorithms::regression::RegressionError>(())
/// ```
pub fn ols(x: &[Vec<f64>], y: &[f64]) -> Result<LinearModel, RegressionError> {
    ridge(x, y, 0.0)
}

/// Fits a ridge (L2-regularised) linear model to `x` and `y` with penalty `lambda`.
///
/// Equivalent to `scikit-learn`'s `Ridge(alpha=lambda, fit_intercept=True)`: the
/// design and target are centred, the penalised normal equations
/// `(X_cᵀX_c + λI)β = X_cᵀy_c` are solved for the centred slopes (so the penalty
/// shrinks the slopes but never the intercept), and the intercept is recovered from
/// the column means. With `lambda == 0.0` this reduces to [`ols`].
///
/// # Arguments
///
/// * `x` — design matrix, one inner slice per observation; rows must share width.
/// * `y` — target values, one per row of `x`.
/// * `lambda` — L2 penalty strength `λ ≥ 0`; larger values shrink the slopes more.
///
/// # Returns
///
/// A [`LinearModel`] with the fitted intercept and shrunk per-column slopes.
///
/// # Errors
///
/// Returns [`RegressionError::InvalidPenalty`] if `lambda` is negative or
/// non-finite, plus the same shape/singularity errors as [`ols`].
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::regression::ridge;
///
/// // A positive penalty shrinks the slope toward zero relative to OLS.
/// let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
/// let y = vec![2.0, 5.0, 8.0, 11.0];
/// let model = ridge(&x, &y, 1.0)?;
/// let slope = model.coefficients().first().copied().unwrap_or(f64::NAN);
/// assert!(slope < 3.0 && slope > 0.0, "shrunk slope {slope}");
/// # Ok::<(), stats_claw::algorithms::regression::RegressionError>(())
/// ```
pub fn ridge(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Result<LinearModel, RegressionError> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(RegressionError::InvalidPenalty);
    }
    let n_rows = x.len();
    if n_rows == 0 {
        return Err(RegressionError::EmptyInput);
    }
    if y.len() != n_rows {
        return Err(RegressionError::RowMismatch {
            rows_x: n_rows,
            rows_y: y.len(),
        });
    }
    let n_cols = x.first().map_or(0, Vec::len);
    if x.iter().any(|row| row.len() != n_cols) {
        return Err(RegressionError::RaggedRows);
    }
    // A model with no predictors is just the target mean as the intercept.
    if n_cols == 0 {
        return Ok(LinearModel {
            intercept: mean(y),
            coefficients: Vec::new(),
        });
    }

    let col_means = column_means(x, n_cols);
    let y_mean = mean(y);

    // Penalised normal equations on the centred data:
    //   A = X_cᵀX_c + λI   (size n_cols × n_cols, symmetric)
    //   b = X_cᵀy_c
    let mut a = vec![vec![0.0_f64; n_cols]; n_cols];
    let mut b = vec![0.0_f64; n_cols];
    for (row, &yi) in x.iter().zip(y) {
        let yc = yi - y_mean;
        for j in 0..n_cols {
            let xj = centered(row, &col_means, j);
            if let Some(bj) = b.get_mut(j) {
                *bj = xj.mul_add(yc, *bj);
            }
            if let Some(a_row) = a.get_mut(j) {
                for k in 0..n_cols {
                    let xk = centered(row, &col_means, k);
                    if let Some(a_jk) = a_row.get_mut(k) {
                        *a_jk = xj.mul_add(xk, *a_jk);
                    }
                }
            }
        }
    }
    for (j, a_row) in a.iter_mut().enumerate() {
        if let Some(a_jj) = a_row.get_mut(j) {
            *a_jj += lambda;
        }
    }

    let coefficients = solve(a, b)?;
    // Recover the intercept: β₀ = ȳ − Σ_j β_j x̄_j.
    let slope_dot_mean: f64 = coefficients
        .iter()
        .zip(&col_means)
        .map(|(&beta, &mean_j)| beta * mean_j)
        .sum();
    Ok(LinearModel {
        intercept: y_mean - slope_dot_mean,
        coefficients,
    })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
