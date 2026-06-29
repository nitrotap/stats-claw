//! Shared rank-assignment helpers for the rank-based tests.
//!
//! Several nonparametric and correlation tests (Mann–Whitney, Kruskal–Wallis,
//! Wilcoxon, Friedman, Spearman, Kendall) rank pooled or paired data and apply
//! the mid-rank convention to ties, matching `scipy.stats`. This module centralizes
//! that logic so the convention is identical across tests.

/// Assigns average (mid-)ranks to `values`, the convention scipy uses for ties.
///
/// Ranks are 1-based; a run of `m` tied values that would occupy ranks `r‥r+m`
/// each receives their average `r + (m−1)/2`. The returned vector is in the same
/// order as `values`.
///
/// # Arguments
///
/// * `values` — the data to rank; order is preserved in the output.
///
/// # Returns
///
/// A vector of mid-ranks aligned with `values`.
pub(super) fn mid_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let a = values.get(i).copied().unwrap_or(f64::NAN);
        let b = values.get(j).copied().unwrap_or(f64::NAN);
        a.total_cmp(&b)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        let vi = order.get(i).and_then(|&k| values.get(k)).copied();
        while j < n {
            let vj = order.get(j).and_then(|&k| values.get(k)).copied();
            if vi
                .zip(vj)
                .is_some_and(|(a, b)| (a - b).abs() < f64::EPSILON)
            {
                j += 1;
            } else {
                break;
            }
        }
        // Ranks i+1 ‥ j (1-based); average them.
        let first = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let last = u32::try_from(j).unwrap_or(u32::MAX);
        let avg = f64::midpoint(f64::from(first), f64::from(last));
        for slot in order.get(i..j).into_iter().flatten() {
            if let Some(r) = ranks.get_mut(*slot) {
                *r = avg;
            }
        }
        i = j;
    }
    ranks
}

/// Computes the tie-correction sum `Σ (tⱼ³ − tⱼ)` over the tie-group sizes `tⱼ` of
/// `values`, used by the Kruskal–Wallis and Mann–Whitney variance corrections.
///
/// # Arguments
///
/// * `values` — the pooled data.
///
/// # Returns
///
/// `Σ (t³ − t)` over each maximal run of equal values.
pub(super) fn tie_correction(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut total = 0.0;
    let mut i = 0;
    let n = sorted.len();
    while i < n {
        let mut j = i + 1;
        while j < n
            && sorted
                .get(i)
                .zip(sorted.get(j))
                .is_some_and(|(a, b)| (a - b).abs() < f64::EPSILON)
        {
            j += 1;
        }
        let count = u32::try_from(j - i).unwrap_or(u32::MAX);
        let t = f64::from(count);
        total += t.mul_add(t * t, -t);
        i = j;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct values get sequential 1-based ranks.
    #[test]
    fn distinct_values_rank_sequentially() {
        let r = mid_ranks(&[30.0, 10.0, 20.0]);
        assert_eq!(r, vec![3.0, 1.0, 2.0], "ranks were {r:?}");
    }

    /// Tied values share their average rank.
    #[test]
    fn ties_share_average_rank() {
        let r = mid_ranks(&[10.0, 20.0, 20.0, 40.0]);
        assert_eq!(r, vec![1.0, 2.5, 2.5, 4.0], "ranks were {r:?}");
    }

    /// The tie correction sums `t^3 − t` over tie groups.
    #[test]
    fn tie_correction_counts_groups() {
        // one group of 2 → 2^3 - 2 = 6
        let c = tie_correction(&[10.0, 20.0, 20.0, 40.0]);
        assert!((c - 6.0).abs() < 1e-12, "correction was {c}");
    }
}
