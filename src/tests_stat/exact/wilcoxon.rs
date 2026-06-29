//! Exact Wilcoxon signed-rank-sum null distribution and its tail probabilities.
//!
//! With no zero differences and no ties, the absolute differences receive the
//! ranks `1..=n`, and under the null each rank independently carries a `+` or `−`
//! sign with probability `½`. The null distribution of the positive-rank sum
//! `W₊` is therefore the coefficient list of `∏_{j=1}^{n} (1 + q^j)` over the
//! `2^n` equally likely sign patterns. This module builds that list by
//! convolution and reads off the exact tail probabilities, matching
//! `scipy.stats.wilcoxon(method="exact")`.

use crate::tests_stat::Alternative;

/// Counts of sign patterns giving each positive-rank sum `W₊`, for `n` nonzero,
/// untied differences.
///
/// Index `w` of the returned vector holds the number of the `2^n` sign patterns
/// whose positive ranks sum to `w`, for `w` in `0..=n(n+1)/2`. Built by
/// convolving in each rank `j = 1..=n` (multiplying the running polynomial by
/// `(1 + q^j)`).
///
/// # Arguments
///
/// * `count` — the number of nonzero, untied differences (signed-rank count).
///
/// # Returns
///
/// A vector of length `count(count+1)/2 + 1` of sign-pattern counts as `f64`;
/// the entries sum to `2^count`.
#[must_use]
pub fn rank_sum_distribution(count: usize) -> Vec<f64> {
    let max_w = count * (count + 1) / 2;
    let mut counts = vec![0.0_f64; max_w + 1];
    if let Some(first) = counts.first_mut() {
        *first = 1.0;
    }
    for j in 1..=count {
        // Multiply by (1 + q^j): each existing sum w also appears at w + j.
        for w in (j..=max_w).rev() {
            let from = counts.get(w - j).copied().unwrap_or(0.0);
            if let Some(slot) = counts.get_mut(w) {
                *slot += from;
            }
        }
    }
    counts
}

/// Exact two-sided / one-sided p-value for an observed positive-rank sum `r_plus`.
///
/// Uses the exact null distribution from [`rank_sum_distribution`]:
/// `greater = P(W₊ ≥ r_plus)`, `less = P(W₊ ≤ r_plus)`, and
/// `two-sided = min(2·min(less, greater), 1)`, matching
/// `scipy.stats.wilcoxon(method="exact")`.
///
/// # Arguments
///
/// * `r_plus` — the observed positive-rank sum (integer-valued `f64`, no ties).
/// * `count` — the number of nonzero, untied differences.
/// * `alternative` — the tested direction of the median difference.
///
/// # Returns
///
/// The exact p-value in `[0, 1]`.
#[must_use]
pub fn p_value(r_plus: f64, count: usize, alternative: Alternative) -> f64 {
    let dist = rank_sum_distribution(count);
    let total: f64 = dist.iter().sum();
    let w_idx = r_plus.round();
    let le: f64 = dist
        .iter()
        .enumerate()
        .filter(|(w, _)| f64_from_index(*w) <= w_idx)
        .map(|(_, &c)| c)
        .sum();
    let ge: f64 = dist
        .iter()
        .enumerate()
        .filter(|(w, _)| f64_from_index(*w) >= w_idx)
        .map(|(_, &c)| c)
        .sum();
    let p_less = le / total;
    let p_greater = ge / total;
    match alternative {
        Alternative::Less => p_less.clamp(0.0, 1.0),
        Alternative::Greater => p_greater.clamp(0.0, 1.0),
        Alternative::TwoSided => (2.0 * p_less.min(p_greater)).clamp(0.0, 1.0),
    }
}

/// Widens a `usize` index to `f64` without an `as` cast (indices are tiny).
fn f64_from_index(i: usize) -> f64 {
    u32::try_from(i).map_or(f64::INFINITY, f64::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The signed-rank-sum distribution sums to `2^n` and is symmetric about
    /// `n(n+1)/4`.
    #[test]
    fn distribution_sums_to_two_pow_n() {
        let n = 6;
        let dist = rank_sum_distribution(n);
        let total: f64 = dist.iter().sum();
        assert!((total - 64.0).abs() < 1e-9, "total was {total}");
        let max_w = n * (n + 1) / 2; // 21
        assert_eq!(dist.len(), max_w + 1, "len was {}", dist.len());
        for w in 0..=(max_w / 2) {
            let lo = dist.get(w).copied().unwrap_or(f64::NAN);
            let hi = dist.get(max_w - w).copied().unwrap_or(f64::NAN);
            assert!((lo - hi).abs() < 1e-9, "asymmetry at {w}: {lo} vs {hi}");
        }
    }

    /// Tail probabilities for W₊ = 8 (n = 6) match the hand-enumerated values.
    #[test]
    fn tails_match_enumeration() {
        // From a full 2^6 enumeration: P(W+>=8)=0.71875, P(W+<=8)=0.34375.
        let greater = p_value(8.0, 6, Alternative::Greater);
        let less = p_value(8.0, 6, Alternative::Less);
        let two = p_value(8.0, 6, Alternative::TwoSided);
        assert!((greater - 0.71875).abs() < 1e-12, "greater was {greater}");
        assert!((less - 0.34375).abs() < 1e-12, "less was {less}");
        assert!((two - 0.6875).abs() < 1e-12, "two was {two}");
    }
}
