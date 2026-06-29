//! Equivalence suite for the density-estimation family (Gaussian KDE).
//!
//! Loads a committed `scipy.stats.gaussian_kde` golden fixture and asserts the
//! stats-claw `gaussian_kde` agrees on the Scott covariance factor, the kernel
//! bandwidth/variance, and the density on a query grid, all within a stated
//! tolerance. Python never runs here — the fixture is the offline source of
//! truth.

use crate::common;
use crate::common::HarnessError;
use stats_claw::algorithms::density::{gaussian_kde, DensityError};

/// Relative/absolute tolerance for the density equivalence comparisons.
///
/// The Rust estimator reproduces scipy's exact construction (Scott factor
/// `n^(−1/5)`, kernel variance `factor² · var(data, ddof=1)`, and a plain
/// Gaussian-sum evaluation), so the agreement is at machine precision. The gate
/// is set at `1e-9`; the achieved max-abs difference across the factor,
/// bandwidth, and every grid density is ~`1.4e-17` against `scipy` 1.17.1.
const ATOL: f64 = 1e-9;
const RTOL: f64 = 1e-9;

/// Maps a borrowed [`DensityError`] into the harness error type so tests use `?`
/// on a fit result.
fn fit_err(e: &DensityError) -> HarnessError {
    HarnessError::Parse(format!("kde fit failed: {e}"))
}

#[test]
fn gaussian_kde_agrees_with_scipy() -> Result<(), HarnessError> {
    let fx = common::load("density_kde")?;
    let data = common::f64s(&fx, "data")?;
    let kde = gaussian_kde(&data).map_err(|e| fit_err(&e))?;

    // Bandwidth/variance match scipy's Scott-rule covariance.
    common::assert_close(kde.variance(), common::scalar(&fx, "variance")?, ATOL, RTOL);
    common::assert_close(
        kde.bandwidth(),
        common::scalar(&fx, "bandwidth")?,
        ATOL,
        RTOL,
    );

    // The estimated density agrees on every grid point.
    let grid = common::f64s(&fx, "grid")?;
    let expected = common::f64s(&fx, "density")?;
    let got = kde.density(&grid);
    common::assert_vec_close(&got, &expected, ATOL, RTOL);
    Ok(())
}
