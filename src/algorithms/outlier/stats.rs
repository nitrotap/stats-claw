//! Numeric helpers for the outlier detectors: count widening, population/sample
//! moments, the numpy-`linear` percentile, the median, and the MAD.
//!
//! Kept separate from the detector API so `mod.rs` stays inside the project's
//! 500-line `style.rs` cap and the `as`-free count widening is reusable. Each
//! statistic matches the reference library convention the equivalence suite pins:
//! `scipy.stats.zscore` (population std, `ddof = 0`) and `numpy.percentile` with
//! its default `'linear'` interpolation.

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

/// Returns the population standard deviation (`ddof = 0`) of `values`.
///
/// Matches `scipy.stats.zscore`'s default scaling — the divisor is `n`, not
/// `n − 1` — so the z-scores the detector produces reproduce scipy exactly.
///
/// # Arguments
///
/// * `values` — the observations whose spread to measure; must be non-empty.
///
/// # Returns
///
/// `√(Σ (vᵢ − v̄)² / n)`, or `0.0` for an empty slice.
pub(super) fn population_std(values: &[f64]) -> f64 {
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
    (ss / n).sqrt()
}

/// Returns the `q`-quantile of `values` using numpy's default `'linear'`
/// interpolation.
///
/// Reproduces `numpy.percentile(values, 100·q, method='linear')`: the data is
/// sorted, the fractional index `h = q·(n − 1)` is formed, and the result is the
/// linear interpolation between the bracketing order statistics. This is the
/// convention the IQR/Tukey fences pin against the numpy reference.
///
/// # Arguments
///
/// * `sorted` — the observations **already sorted ascending**.
/// * `q` — the quantile in `[0, 1]` (e.g. `0.25` for the first quartile).
///
/// # Returns
///
/// The interpolated quantile, or `0.0` for an empty slice (unreachable: the
/// caller validates non-emptiness first).
pub(super) fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    // numpy 'linear': virtual index h = q*(n-1), interpolate between floor/ceil.
    let h = q * (count_to_f64(n) - 1.0);
    let lo_idx = floor_to_usize(h);
    let frac = h - count_to_f64(lo_idx);
    let lo = sorted.get(lo_idx).copied().unwrap_or(0.0);
    let hi = sorted.get(lo_idx + 1).copied().unwrap_or(lo);
    (hi - lo).mul_add(frac, lo)
}

/// Floors a non-negative percentile virtual index to a `usize` without an `as`
/// cast.
///
/// The input is the numpy virtual index `h = q·(n − 1)`, so it is non-negative
/// and bounded by `n − 1 < 2^32` for every sample this crate handles. The floor
/// is found by a fixed 32-step binary search over bit positions, entirely in
/// integer space, so no narrowing `as` cast is needed.
///
/// # Arguments
///
/// * `x` — a non-negative, finite value below `2^32`.
///
/// # Returns
///
/// `⌊x⌋` as a `usize`.
fn floor_to_usize(x: f64) -> usize {
    if x <= 0.0 {
        return 0;
    }
    let mut acc: u32 = 0;
    let mut bit: u32 = 1 << 31;
    while bit > 0 {
        let candidate = acc | bit;
        if f64::from(candidate) <= x {
            acc = candidate;
        }
        bit >>= 1;
    }
    usize::try_from(acc).unwrap_or(0)
}

/// Returns the median of `values` (numpy's `'linear'` 50th percentile).
///
/// # Arguments
///
/// * `values` — the observations; must be non-empty.
///
/// # Returns
///
/// The median, sorting a local copy so the caller's slice is untouched.
pub(super) fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_linear(&sorted, 0.5)
}

/// Returns the median absolute deviation (MAD) of `values` about their median.
///
/// `MAD = median(|vᵢ − median(v)|)`. Used by the modified (MAD-based) z-score,
/// which is robust to the outliers a mean/std z-score is itself distorted by.
///
/// # Arguments
///
/// * `values` — the observations; must be non-empty.
///
/// # Returns
///
/// The MAD (`≥ 0`).
pub(super) fn median_absolute_deviation(values: &[f64]) -> f64 {
    let med = median(values);
    let deviations: Vec<f64> = values.iter().map(|&v| (v - med).abs()).collect();
    median(&deviations)
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

    /// Population std uses divisor `n` (ddof = 0), matching scipy `zscore`.
    #[test]
    fn population_std_uses_divisor_n() {
        // values 2,4,6: mean 4, SS = 8, /3 = 2.6667, sqrt = 1.632993...
        let s = population_std(&[2.0, 4.0, 6.0]);
        assert!((s - 1.632_993_161_855_452_2).abs() < 1e-12, "std was {s}");
    }

    /// The linear percentile reproduces numpy's quartiles on a small sample.
    #[test]
    fn percentile_matches_numpy_linear() {
        // np.percentile([1,2,3,4,100],[25,75]) == [2, 4].
        let sorted = [1.0, 2.0, 3.0, 4.0, 100.0];
        let q1 = percentile_linear(&sorted, 0.25);
        let q3 = percentile_linear(&sorted, 0.75);
        assert!((q1 - 2.0).abs() < 1e-12, "q1 was {q1}");
        assert!((q3 - 4.0).abs() < 1e-12, "q3 was {q3}");
    }

    /// The percentile interpolates between order statistics (even-length sample).
    #[test]
    fn percentile_interpolates_between_points() {
        // np.percentile([1,2,3,4],25) = 1.75 (h = 0.25*3 = 0.75).
        let sorted = [1.0, 2.0, 3.0, 4.0];
        let q1 = percentile_linear(&sorted, 0.25);
        assert!((q1 - 1.75).abs() < 1e-12, "q1 was {q1}");
    }

    /// The median matches a hand value for an even-length sample.
    #[test]
    fn median_of_even_sample() {
        let m = median(&[1.0, 2.0, 3.0, 4.0]);
        assert!((m - 2.5).abs() < 1e-12, "median was {m}");
    }

    /// The MAD matches a hand-computed value.
    #[test]
    fn mad_matches_hand_value() {
        // values 1,2,3,4,5: median 3, |dev| = 2,1,0,1,2 -> sorted 0,1,1,2,2,
        // median of deviations = 1.
        let mad = median_absolute_deviation(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((mad - 1.0).abs() < 1e-12, "mad was {mad}");
    }
}
