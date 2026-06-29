//! Exact null distribution for Kendall's tau with no ties.
//!
//! Under the no-ties null the number of discordant pairs `D` over `n` items has
//! the same distribution as the number of inversions of a random permutation,
//! whose counts `cₙ(k)` satisfy the classic recurrence
//! `cₙ(k) = Σ_{j=0..n−1} cₙ₋₁(k − j)`. scipy uses this exact distribution by
//! default for small, untied samples, so we match it rather than the normal
//! approximation.

use crate::tests_stat::Alternative;

/// Exact Kendall p-value for `n` untied pairs with `discordant` discordant pairs.
///
/// `tau`'s sign selects the tail: the lower tail counts permutations with at most
/// `discordant` discordant pairs (strong positive association), the upper tail at
/// least that many. The two-sided p-value is `2·min(lower, upper)` clamped to 1.
///
/// # Arguments
///
/// * `n` — number of pairs (`≥ 2`).
/// * `discordant` — observed discordant-pair count (a non-negative integer value).
/// * `tau` — the observed tau-b (only its sign is used here).
/// * `alternative` — which tail(s) contribute.
///
/// # Returns
///
/// The exact p-value in `[0, 1]`.
pub(super) fn kendall_two_sided_exact(
    n: usize,
    discordant: f64,
    tau: f64,
    alternative: Alternative,
) -> f64 {
    let max_inv = n * (n - 1) / 2;
    let counts = inversion_counts(n, max_inv);
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 1.0;
    }
    let d = discordant.round();
    let d_idx = clamp_index(d, max_inv);

    // Lower tail: permutations with ≤ d discordant pairs (favors positive tau).
    let lower: f64 = counts.iter().take(d_idx + 1).sum::<f64>() / total;
    // Upper tail: permutations with ≥ d discordant pairs.
    let upper: f64 = counts.iter().skip(d_idx).sum::<f64>() / total;

    match alternative {
        Alternative::Greater => lower.clamp(0.0, 1.0),
        Alternative::Less => upper.clamp(0.0, 1.0),
        Alternative::TwoSided => {
            // Choose the tail in the observed direction, then double.
            let tail = if tau >= 0.0 { lower } else { upper };
            (2.0 * tail).clamp(0.0, 1.0)
        }
    }
}

/// Clamps a discordant count to a valid index in `0..=max_inv`.
fn clamp_index(d: f64, max_inv: usize) -> usize {
    if d <= 0.0 {
        return 0;
    }
    let mut idx = 0usize;
    while idx < max_inv && f64_eq_or_less(idx + 1, d) {
        idx += 1;
    }
    idx
}

/// Returns whether `count` (as `f64`) is `≤ d`, without an `as` cast.
fn f64_eq_or_less(count: usize, d: f64) -> bool {
    let lo = u32::try_from(count).unwrap_or(u32::MAX);
    f64::from(lo) <= d
}

/// Counts permutations of `n` items by their number of inversions `0..=max_inv`.
///
/// Returns a vector `c` where `c[k]` is the number of length-`n` permutations with
/// exactly `k` inversions, via the sliding-window recurrence.
fn inversion_counts(n: usize, max_inv: usize) -> Vec<f64> {
    let mut counts = vec![0.0; max_inv + 1];
    if let Some(first) = counts.first_mut() {
        *first = 1.0;
    }
    let mut current_max = 0usize;
    for i in 1..n {
        let new_max = (current_max + i).min(max_inv);
        let prefix = prefix_sums(&counts, new_max);
        let mut next = vec![0.0; max_inv + 1];
        for (k, slot) in next.iter_mut().enumerate().take(new_max + 1) {
            // Window of width i+1 (j = 0..=i): Σ counts[k-i ..= k].
            let hi = prefix.get(k).copied().unwrap_or(0.0);
            let lo = k
                .checked_sub(i + 1)
                .and_then(|idx| prefix.get(idx))
                .copied()
                .unwrap_or(0.0);
            *slot = hi - lo;
        }
        counts = next;
        current_max = new_max;
    }
    counts
}

/// Inclusive prefix sums of `counts[0..=upto]` (`prefix[k] = Σ_{j≤k} counts[j]`).
fn prefix_sums(counts: &[f64], upto: usize) -> Vec<f64> {
    let mut prefix = vec![0.0; upto + 1];
    let mut running = 0.0;
    for (k, slot) in prefix.iter_mut().enumerate() {
        running += counts.get(k).copied().unwrap_or(0.0);
        *slot = running;
    }
    prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For 4 items the inversion counts are the known sequence 1,3,5,6,5,3,1.
    #[test]
    fn inversion_counts_for_four() {
        let c = inversion_counts(4, 6);
        assert_eq!(c, vec![1.0, 3.0, 5.0, 6.0, 5.0, 3.0, 1.0], "counts {c:?}");
    }

    /// Perfect concordance (D = 0) gives one-sided p = 1/n! for n = 4.
    #[test]
    fn perfect_concordance_p_is_reciprocal_factorial() {
        // 4! = 24 → one-sided p = 1/24.
        let p = kendall_two_sided_exact(4, 0.0, 1.0, Alternative::Greater);
        assert!((p - 1.0 / 24.0).abs() < 1e-12, "p was {p}");
    }
}
