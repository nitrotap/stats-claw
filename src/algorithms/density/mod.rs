//! Kernel density estimation (KDE) with a Gaussian kernel.
//!
//! Estimates a continuous probability density from a one-dimensional sample by
//! summing a Gaussian bump over every observation. The kernel bandwidth follows
//! Scott's rule with the same covariance factor `scipy.stats.gaussian_kde` uses
//! (`factor = n^(−1/5)` in one dimension, `bandwidth² = factor² · var(data, ddof=1)`),
//! so the estimate reproduces scipy's `gaussian_kde` to machine precision. This
//! base block backs the `DensityEstimation` computational method.

mod kernel;

use kernel::{count_to_f64, sample_variance};

/// `2π`, the constant in the Gaussian normalising factor `1/√(2π h²)`.
const TWO_PI: f64 = std::f64::consts::TAU;

/// A fitted Gaussian kernel density estimator over a one-dimensional sample.
///
/// Holds the observations and the squared bandwidth (kernel variance) derived
/// from them, so density queries reuse a single bandwidth computed once at fit
/// time. Fields are private to keep the sample and its derived bandwidth an
/// invariant pair; query the estimate with [`GaussianKde::density`] /
/// [`GaussianKde::density_at`].
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianKde {
    /// The fitted one-dimensional observations.
    data: Vec<f64>,
    /// The kernel variance `h²` (squared bandwidth) shared by every kernel.
    variance: f64,
}

/// Errors that prevent a Gaussian KDE from being fitted.
///
/// Returned (never panicked) so callers stay clear of the crate's `unwrap`/`panic`
/// lint gate and can surface a clean diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DensityError {
    /// The sample was empty, so no density can be estimated.
    EmptyInput,
    /// The sample had only one observation, so its variance — and hence Scott's
    /// bandwidth — is undefined (degenerate).
    SinglePoint,
    /// Every observation was identical (zero variance), so the bandwidth collapses
    /// to zero and the estimate is a degenerate spike.
    ZeroVariance,
    /// An observation was non-finite (`NaN` or `±∞`), which has no valid density.
    NonFinite,
}

impl std::fmt::Display for DensityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "sample has no observations"),
            Self::SinglePoint => write!(f, "sample needs at least two observations"),
            Self::ZeroVariance => write!(f, "sample has zero variance"),
            Self::NonFinite => write!(f, "sample contains a non-finite value"),
        }
    }
}

impl std::error::Error for DensityError {}

impl GaussianKde {
    /// The kernel variance `h²` (squared bandwidth) shared by every kernel.
    #[must_use]
    pub const fn variance(&self) -> f64 {
        self.variance
    }

    /// The kernel bandwidth `h` (standard deviation of each Gaussian bump).
    #[must_use]
    pub fn bandwidth(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Evaluates the estimated density at a single query point `x`.
    ///
    /// # Arguments
    ///
    /// * `x` — the point at which to evaluate the density.
    ///
    /// # Returns
    ///
    /// The estimated density `f(x) ≥ 0`.
    #[must_use]
    pub fn density_at(&self, x: f64) -> f64 {
        // f(x) = (1/n) Σ_i N(x; xᵢ, h²); the 1/√(2π h²) normaliser is shared.
        let n = count_to_f64(self.data.len());
        let norm = 1.0 / (TWO_PI * self.variance).sqrt();
        let inv_two_var = 1.0 / (2.0 * self.variance);
        let sum: f64 = self
            .data
            .iter()
            .map(|&xi| {
                let d = x - xi;
                (-d * d * inv_two_var).exp()
            })
            .sum();
        norm * sum / n
    }

    /// Evaluates the estimated density at every query point in `xs`.
    ///
    /// # Arguments
    ///
    /// * `xs` — the points at which to evaluate the density.
    ///
    /// # Returns
    ///
    /// One estimated density per query point, in the input order.
    #[must_use]
    pub fn density(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.density_at(x)).collect()
    }
}

/// Fits a Gaussian kernel density estimator to the one-dimensional `data`.
///
/// # Arguments
///
/// * `data` — the observations to estimate a density from; needs at least two
///   distinct finite values.
///
/// # Returns
///
/// A fitted [`GaussianKde`].
///
/// # Errors
///
/// Returns [`DensityError::EmptyInput`], [`DensityError::SinglePoint`],
/// [`DensityError::ZeroVariance`], or [`DensityError::NonFinite`].
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::density::gaussian_kde;
///
/// // Matches `scipy.stats.gaussian_kde([2, 4, 6]).evaluate([4.0])`.
/// let kde = gaussian_kde(&[2.0, 4.0, 6.0])?;
/// let f4 = kde.density_at(4.0);
/// assert!((f4 - 0.159_078_110_463_200_72).abs() < 1e-9, "f(4) was {f4}");
/// # Ok::<(), stats_claw::algorithms::density::DensityError>(())
/// ```
pub fn gaussian_kde(data: &[f64]) -> Result<GaussianKde, DensityError> {
    if data.is_empty() {
        return Err(DensityError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(DensityError::NonFinite);
    }
    if data.len() < 2 {
        return Err(DensityError::SinglePoint);
    }
    let var = sample_variance(data);
    if var <= 0.0 {
        return Err(DensityError::ZeroVariance);
    }
    // Scott's rule (scipy's `gaussian_kde` default in 1-D): the covariance factor
    // is `n^(−1/(d+4)) = n^(−1/5)`, and the kernel variance is `factor² · var`.
    let n = count_to_f64(data.len());
    let factor = n.powf(-1.0 / 5.0);
    let variance = factor * factor * var;
    Ok(GaussianKde {
        data: data.to_vec(),
        variance,
    })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
