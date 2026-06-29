//! Unsupervised learning algorithms.
//!
//! This module defines the shared algorithm primitives — the squared-Euclidean
//! distance helper and a small `Matrix` view over row-major data — and houses the
//! clustering family in the [`clustering`] subgroup folder. Every clustering
//! routine consumes `&[Vec<f64>]` (one inner `Vec` per observation) and returns a
//! label vector that the equivalence suite compares to `scikit-learn` by adjusted
//! Rand index.
//!
//! Decomposition/embedding (PCA, factor analysis, ICA, t-SNE, UMAP, LLE) live in
//! the [`decomposition`] subgroup; they compare to their `scikit-learn` references
//! under the sign/order/stochastic-aware standards documented there. PELT
//! change-point detection lives in [`change_point`] and compares to `ruptures` for
//! exact breakpoint equality. Gaussian kernel density estimation lives in
//! [`density`] and compares to `scipy.stats.gaussian_kde` (Scott's-rule bandwidth)
//! to machine precision. Univariate outlier / anomaly detection (z-score, IQR
//! Tukey fence, modified MAD-based z-score) lives in [`outlier`] and compares to
//! `scipy.stats.zscore` and `numpy.percentile` to machine precision. Univariate
//! feature selection (variance threshold + ANOVA F-test `f_classif` score) lives
//! in [`feature_selection`] and compares to
//! `sklearn.feature_selection.VarianceThreshold` and
//! `sklearn.feature_selection.f_classif` (F-scores to ~`1e-9`, p-values to the F
//! distribution's asymptotic `1e-6` tail band, variances exact). `HyperLogLog`
//! distinct-count (cardinality) estimation lives in [`cardinality`]; it has no
//! canonical reference library, so it is checked against the **exact** distinct
//! count (a `HashSet` ground truth) landing inside a small multiple of
//! `HyperLogLog`'s `≈ 1.04 / √m` theoretical standard error.

pub mod association;
pub mod cardinality;
pub mod change_point;
pub mod clustering;
pub mod decomposition;
pub mod density;
pub mod feature_selection;
pub mod outlier;
pub mod regression;

/// Computes the squared Euclidean distance between two equal-length points.
///
/// The square root is omitted: clustering routines compare and accumulate
/// distances where the monotonic squared form is both faster and more accurate
/// (it avoids a `sqrt` round-trip), and inertia is defined as a sum of squared
/// distances.
///
/// # Arguments
///
/// * `a` — first point.
/// * `b` — second point; must have the same length as `a`. Extra coordinates in
///   the longer slice are ignored (the zip stops at the shorter length), so
///   callers are responsible for passing equal-dimension points.
///
/// # Returns
///
/// `Σ (aᵢ − bᵢ)²`, always `≥ 0` for finite inputs.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::euclidean_sq;
///
/// // (0,0) to (3,4): 9 + 16 = 25.
/// assert!((euclidean_sq(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn euclidean_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Computes the elementwise mean (centroid) of a non-empty set of points.
///
/// # Arguments
///
/// * `points` — the points to average; each must share the dimension `dim`.
/// * `dim` — the dimensionality of every point.
///
/// # Returns
///
/// The centroid as a length-`dim` vector, or a zero vector when `points` is empty
/// (callers that must distinguish an empty cluster check the count themselves).
#[must_use]
pub fn centroid(points: &[&[f64]], dim: usize) -> Vec<f64> {
    let mut sum = vec![0.0_f64; dim];
    for &p in points {
        for (s, &v) in sum.iter_mut().zip(p) {
            *s += v;
        }
    }
    let n = count_to_f64(points.len());
    if n > 0.0 {
        for s in &mut sum {
            *s /= n;
        }
    }
    sum
}

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Counts here are sample/cluster sizes far below `2^53`, so splitting into
/// 32-bit halves and recombining reproduces the value exactly while satisfying the
/// `style.rs` no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64` (exact for the sample sizes this crate handles).
#[must_use]
pub fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_sq_is_zero_for_identical_points() {
        assert!(euclidean_sq(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).abs() < 1e-12);
    }

    #[test]
    fn euclidean_sq_matches_pythagoras() {
        assert!((euclidean_sq(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn centroid_averages_coordinates() {
        let first = [0.0_f64, 0.0];
        let second = [2.0_f64, 4.0];
        let mean = centroid(&[&first, &second], 2);
        assert_eq!(mean.len(), 2, "centroid dim");
        let mean_x = mean.first().copied().unwrap_or(f64::NAN);
        let mean_y = mean.get(1).copied().unwrap_or(f64::NAN);
        assert!((mean_x - 1.0).abs() < 1e-12, "x was {mean_x}");
        assert!((mean_y - 2.0).abs() < 1e-12, "y was {mean_y}");
    }

    #[test]
    fn count_to_f64_widens_exactly() {
        assert!((count_to_f64(150) - 150.0).abs() < 1e-12);
    }
}
