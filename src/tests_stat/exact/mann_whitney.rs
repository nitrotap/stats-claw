//! Exact Mann–Whitney U null distribution and its tail probabilities.
//!
//! For two independent samples of sizes `n1` and `n2` with **no ties**, the null
//! distribution of `U` is the number of distinct rank arrangements yielding each
//! value of `U`, normalised by the total `C(n1+n2, n1)`. Those arrangement counts
//! are exactly the coefficients of the Gaussian binomial coefficient
//! `[n1+n2 choose n1]_q = ∏_{i=1}^{n1} (1 − q^{n2+i}) / (1 − q^i)`, which this
//! module builds by exact polynomial multiplication and series division (counts
//! stay integral and well below `2^53`, so `f64` is exact).

use crate::tests_stat::Alternative;

/// Counts of rank arrangements giving each `U` value, for samples of sizes
/// `n1` and `n2` with no ties.
///
/// Index `u` of the returned vector holds the number of arrangements with
/// statistic `U = u`, for `u` in `0..=n1*n2`; the entries sum to `C(n1+n2, n1)`.
/// Computed as the coefficient list of the Gaussian binomial coefficient by
/// repeatedly multiplying by `(1 − q^{n2+i})` and dividing by `(1 − q^i)`.
///
/// # Arguments
///
/// * `n1` — size of the first sample (the one whose `U1` is reported).
/// * `n2` — size of the second sample.
///
/// # Returns
///
/// A vector of length `n1*n2 + 1` of arrangement counts as `f64`.
#[must_use]
pub fn u_distribution(n1: usize, n2: usize) -> Vec<f64> {
    let max_u = n1 * n2;
    // poly accumulates the Gaussian binomial coefficient as a dense coefficient
    // list. Start at the constant polynomial 1.
    let mut poly = vec![0.0_f64; max_u + 1];
    if let Some(first) = poly.first_mut() {
        *first = 1.0;
    }
    for i in 1..=n1 {
        // Numerator step: multiply by (1 − q^{n2+i}), descending so each
        // coefficient reads its un-updated lower neighbour.
        let shift = n2 + i;
        for u in (0..=max_u).rev() {
            let sub = u.checked_sub(shift).and_then(|k| poly.get(k)).copied();
            if let (Some(s), Some(slot)) = (sub, poly.get_mut(u)) {
                *slot -= s;
            }
        }
        // Denominator step: divide by (1 − q^i) via the series recurrence
        // res[u] = num[u] + res[u − i].
        for u in i..=max_u {
            let prev = poly.get(u - i).copied().unwrap_or(0.0);
            if let Some(slot) = poly.get_mut(u) {
                *slot += prev;
            }
        }
    }
    poly
}

/// Exact two-sided / one-sided p-value for an observed Mann–Whitney `u1`.
///
/// Uses the exact null distribution from [`u_distribution`]:
/// `less = P(U1 ≤ u1)`, `greater = P(U1 ≥ u1)`, and
/// `two-sided = min(2·min(less, greater), 1)`, matching
/// `scipy.stats.mannwhitneyu(method="exact")`.
///
/// # Arguments
///
/// * `u1` — the observed `U1` statistic (an integer-valued `f64`, no ties).
/// * `n1`, `n2` — the two sample sizes.
/// * `alternative` — the tested direction.
///
/// # Returns
///
/// The exact p-value in `[0, 1]`.
#[must_use]
pub fn p_value(u1: f64, n1: usize, n2: usize, alternative: Alternative) -> f64 {
    let dist = u_distribution(n1, n2);
    let total: f64 = dist.iter().sum();
    // With no ties U is integer-valued, so rounding gives a safe lookup index.
    let u_idx = u1.round();
    let le: f64 = dist
        .iter()
        .enumerate()
        .filter(|(u, _)| f64_from_index(*u) <= u_idx)
        .map(|(_, &c)| c)
        .sum();
    let ge: f64 = dist
        .iter()
        .enumerate()
        .filter(|(u, _)| f64_from_index(*u) >= u_idx)
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

    /// The exact U distribution sums to `C(n1+n2, n1)`, has the right length, and
    /// is symmetric about `n1*n2/2`.
    #[test]
    fn distribution_sums_to_choose() {
        let dist = u_distribution(3, 4);
        let total: f64 = dist.iter().sum();
        // C(7,3) = 35.
        assert!((total - 35.0).abs() < 1e-9, "total was {total}");
        assert_eq!(dist.len(), 13, "len was {}", dist.len());
        // Known coefficient list of [7 choose 3]_q.
        let expected = [
            1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0,
        ];
        for (u, (&got, &want)) in dist.iter().zip(&expected).enumerate() {
            assert!((got - want).abs() < 1e-9, "coeff {u}: {got} vs {want}");
        }
    }

    /// Tail probabilities for U1 = 0 match the hand-computed exact values.
    #[test]
    fn tail_at_zero_matches_reference() {
        // n1=3, n2=4: P(U1<=0) = 1/35, P(U1>=0) = 1.
        let less = p_value(0.0, 3, 4, Alternative::Less);
        let greater = p_value(0.0, 3, 4, Alternative::Greater);
        let two = p_value(0.0, 3, 4, Alternative::TwoSided);
        assert!((less - 1.0 / 35.0).abs() < 1e-12, "less was {less}");
        assert!((greater - 1.0).abs() < 1e-12, "greater was {greater}");
        assert!((two - 2.0 / 35.0).abs() < 1e-12, "two was {two}");
    }
}
