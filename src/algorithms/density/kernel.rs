//! Numerical helpers for the Gaussian KDE: count widening and sample variance.
//!
//! Kept separate from the estimator API so `mod.rs` stays inside the project's
//! 500-line `style.rs` cap and the `as`-free count widening is reusable.

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Sample sizes here are far below `2^53`, so splitting into 32-bit halves and
/// recombining reproduces the value exactly while satisfying the `style.rs`
/// no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64` (exact for the sample sizes this crate handles).
pub(super) fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Returns the unbiased (`ddof = 1`) sample variance of `values`.
///
/// Matches `numpy.var(..., ddof=1)` — the divisor is `n − 1` — which is the
/// covariance `scipy.stats.gaussian_kde` scales by its Scott factor. Returns
/// `0.0` for fewer than two values (the caller treats that as a degenerate fit).
///
/// # Arguments
///
/// * `values` — the observations whose spread to measure.
///
/// # Returns
///
/// `Σ (vᵢ − v̄)² / (n − 1)`, or `0.0` when `values` has fewer than two entries.
pub(super) fn sample_variance(values: &[f64]) -> f64 {
    let n = count_to_f64(values.len());
    if n < 2.0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    let ss: f64 = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum();
    ss / (n - 1.0)
}

/// Kani proof harness for the Gaussian-KDE sample variance.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy.
#[cfg(kani)]
mod verification {
    use super::sample_variance;

    /// Magnitude bound on each observation: it keeps the squared deviations finite so
    /// the accumulation cannot diverge into `∞`/`NaN`, isolating the non-negativity
    /// argument (mirrors the moments and distribution-layer proofs).
    const MAX_ABS: f64 = 1e6;

    /// Proves [`sample_variance`] is panic-/overflow-free and non-negative for a
    /// symbolic three-observation sample with bounded finite values.
    ///
    /// The unbiased (`ddof = 1`) variance is a sum of squared deviations over the
    /// positive divisor `n − 1` (here `2`), so within the finite `|v| ≤ 1e6` box it is
    /// provably finite and `≥ 0` under `f64` rounding — the property `gaussian_kde`'s
    /// `var > 0` degeneracy check and Scott bandwidth rely on.
    #[kani::proof]
    fn density_sample_variance_non_negative() {
        let v0: f64 = kani::any();
        let v1: f64 = kani::any();
        let v2: f64 = kani::any();
        for v in [v0, v1, v2] {
            kani::assume(v.is_finite());
            kani::assume(v.abs() <= MAX_ABS);
        }
        let var = sample_variance(&[v0, v1, v2]);
        assert!(
            var.is_finite(),
            "sample_variance produced a non-finite value"
        );
        assert!(var >= 0.0, "sample_variance produced a negative value");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The widening reproduces a representative count exactly.
    #[test]
    fn count_widens_exactly() {
        assert!((count_to_f64(2048) - 2048.0).abs() < 1e-12, "widening");
    }

    /// The unbiased variance matches a hand-computed value.
    #[test]
    fn variance_matches_hand_value() {
        // values 2,4,6: mean 4, SS = 4+0+4 = 8, /(3-1) = 4.
        let v = sample_variance(&[2.0, 4.0, 6.0]);
        assert!((v - 4.0).abs() < 1e-12, "variance was {v}");
    }

    /// Fewer than two values yields a zero (degenerate) variance.
    #[test]
    fn variance_of_singleton_is_zero() {
        assert!(sample_variance(&[3.0]).abs() < 1e-12, "singleton variance");
    }
}
