//! Unit tests for the [`super`] Gaussian kernel density estimator.
//!
//! Split into its own file (included via `#[path]`) so `mod.rs` stays inside the
//! project's 500-line `style.rs` cap while the tests still compile under
//! `cargo test` with access to the module's private items.

use super::{gaussian_kde, DensityError};

/// A fitted KDE reproduces a hand-computed Gaussian-sum density at a query point.
#[test]
fn density_matches_hand_value() -> Result<(), DensityError> {
    // data 2,4,6: var(ddof=1)=4, n=3, factor=3^(-1/5), h²=factor²·4.
    // f(4) = (1/3)·Σ N(4; xᵢ, h²) = 0.15907811046320072 (hand-computed).
    let kde = gaussian_kde(&[2.0, 4.0, 6.0])?;
    let f4 = kde.density_at(4.0);
    assert!(
        (f4 - 0.159_078_110_463_200_72).abs() < 1e-12,
        "f(4) was {f4}"
    );
    Ok(())
}

/// The fitted bandwidth follows Scott's rule: `h = n^(−1/5) · √var(ddof=1)`.
#[test]
fn bandwidth_follows_scotts_rule() -> Result<(), DensityError> {
    // data 2,4,6: var=4, n=3, factor=3^(−1/5), h=factor·2 = 1.6054831235204614.
    let kde = gaussian_kde(&[2.0, 4.0, 6.0])?;
    let h = kde.bandwidth();
    assert!((h - 1.605_483_123_520_461_4).abs() < 1e-12, "h was {h}");
    // `variance()` is the square of the bandwidth.
    let var = kde.variance();
    assert!(
        (var - h * h).abs() < 1e-12,
        "variance/bandwidth mismatch: {var}"
    );
    Ok(())
}

/// An empty sample is rejected with [`DensityError::EmptyInput`].
#[test]
fn empty_sample_is_rejected() {
    assert_eq!(gaussian_kde(&[]), Err(DensityError::EmptyInput));
}

/// A single-observation sample is rejected (Scott's bandwidth is undefined).
#[test]
fn single_point_is_rejected() {
    assert_eq!(gaussian_kde(&[3.0]), Err(DensityError::SinglePoint));
}

/// A zero-variance sample (all identical) is rejected, not silently spiked.
#[test]
fn zero_variance_is_rejected() {
    assert_eq!(
        gaussian_kde(&[5.0, 5.0, 5.0]),
        Err(DensityError::ZeroVariance)
    );
}

/// A non-finite observation is rejected before any arithmetic.
#[test]
fn non_finite_is_rejected() {
    assert_eq!(
        gaussian_kde(&[1.0, f64::NAN, 3.0]),
        Err(DensityError::NonFinite)
    );
    assert_eq!(
        gaussian_kde(&[1.0, f64::INFINITY, 3.0]),
        Err(DensityError::NonFinite)
    );
}

/// `density` over a slice equals `density_at` evaluated per element, in order.
#[test]
fn density_slice_matches_pointwise() -> Result<(), DensityError> {
    let kde = gaussian_kde(&[1.0, 2.0, 2.5, 3.0, 7.0, 8.0])?;
    let xs = [2.0, 5.0, 7.5];
    let batch = kde.density(&xs);
    assert_eq!(batch.len(), xs.len(), "batch length");
    for (i, &x) in xs.iter().enumerate() {
        let one = kde.density_at(x);
        let many = batch.get(i).copied().unwrap_or(f64::NAN);
        assert!((one - many).abs() < 1e-15, "index {i}: {one} vs {many}");
    }
    Ok(())
}

/// The estimated density integrates to ~1 over a wide grid (it is a valid pdf).
#[test]
fn density_integrates_to_one() -> Result<(), DensityError> {
    let kde = gaussian_kde(&[1.0, 2.0, 2.5, 3.0, 7.0, 8.0])?;
    // Trapezoidal rule over [-20, 30] with a fine step; the Gaussian tails beyond
    // this window are negligible for this sample.
    let lo = -20.0_f64;
    let hi = 30.0_f64;
    let steps = 50_000_usize;
    let step = (hi - lo) / f64::from(u32::try_from(steps).unwrap_or(0));
    let mut area = 0.0_f64;
    let mut prev = kde.density_at(lo);
    for i in 1..=steps {
        let x = step.mul_add(f64::from(u32::try_from(i).unwrap_or(0)), lo);
        let cur = kde.density_at(x);
        area += 0.5 * (prev + cur) * step;
        prev = cur;
    }
    assert!((area - 1.0).abs() < 1e-4, "integral was {area}");
    Ok(())
}
