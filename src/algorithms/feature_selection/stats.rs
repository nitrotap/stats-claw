//! Numeric helpers for the feature-selection base block: `as`-free count widening
//! and the per-feature population variance.
//!
//! Kept separate from the selector API so `mod.rs` stays inside the project's
//! 500-line `style.rs` cap. The variance matches the convention the equivalence
//! suite pins: `sklearn.feature_selection.VarianceThreshold` uses the **population**
//! variance (`ddof = 0`, divisor `n`), so the variances the selector computes
//! reproduce sklearn exactly.

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Sample/feature counts here are far below `2^53`, so splitting into 32-bit
/// halves and recombining reproduces the value exactly while satisfying the
/// `style.rs` no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64` (exact for the sizes this crate handles).
pub(super) fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Returns the arithmetic mean of `values`.
///
/// # Arguments
///
/// * `values` — the observations to average; must be non-empty (the caller
///   rejects an empty sample before calling).
///
/// # Returns
///
/// `Σ vᵢ / n`, or `0.0` for an empty slice (a degenerate value the caller never
/// reaches because it validates emptiness first).
pub(super) fn mean(values: &[f64]) -> f64 {
    let n = count_to_f64(values.len());
    if n == 0.0 {
        return 0.0;
    }
    values.iter().sum::<f64>() / n
}

/// Returns the population variance (`ddof = 0`) of `values`.
///
/// Matches `sklearn.feature_selection.VarianceThreshold`, which computes
/// `var = Σ (vᵢ − v̄)² / n` (divisor `n`, not `n − 1`), so the per-feature
/// variances the selector produces reproduce sklearn exactly.
///
/// # Arguments
///
/// * `values` — the observations whose spread to measure; must be non-empty.
///
/// # Returns
///
/// `Σ (vᵢ − v̄)² / n`, or `0.0` for an empty slice.
pub(super) fn population_variance(values: &[f64]) -> f64 {
    let n = count_to_f64(values.len());
    if n == 0.0 {
        return 0.0;
    }
    let m = mean(values);
    let ss: f64 = values
        .iter()
        .map(|&v| {
            let d = v - m;
            d * d
        })
        .sum();
    ss / n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The widening reproduces a representative count exactly.
    #[test]
    fn count_widens_exactly() {
        assert!((count_to_f64(4096) - 4096.0).abs() < 1e-12, "widening");
    }

    /// The mean matches a hand-computed value.
    #[test]
    fn mean_matches_hand_value() {
        let m = mean(&[2.0, 4.0, 6.0]);
        assert!((m - 4.0).abs() < 1e-12, "mean was {m}");
    }

    /// Population variance uses divisor `n` (ddof = 0), matching sklearn's
    /// `VarianceThreshold`.
    #[test]
    fn population_variance_uses_divisor_n() {
        // values 2,4,6: mean 4, SS = 8, /3 = 2.6666...
        let v = population_variance(&[2.0, 4.0, 6.0]);
        assert!(
            (v - 2.666_666_666_666_666_5).abs() < 1e-12,
            "variance was {v}"
        );
    }
}
