//! PELT (Pruned Exact Linear Time) change-point detection with an L2 cost.
//!
//! PELT (Killick, Fearnhead & Eckley 2012) finds the segmentation of a signal that
//! minimizes the total within-segment cost plus a per-change-point penalty, exactly
//! (no approximation) while pruning candidate change points that can never become
//! optimal. With the L2 (least-squares) cost, a segment's cost is its sum of
//! squared deviations from the segment mean, `Σxᵢ² − (Σxᵢ)²/len`, computed in O(1)
//! from prefix sums.
//!
//! The breakpoint convention matches `ruptures.Pelt(model="l2").predict(pen)`: the
//! returned indices are the *end* index of each segment in increasing order, so the
//! final signal length `n` is always the last element and the implicit segment start
//! `0` is never listed.

use crate::algorithms::decomposition::count_to_f64;

/// Detects change points in `signal` by penalized least-squares PELT.
///
/// # Arguments
///
/// * `signal` — the 1-D series to segment. An empty signal yields an empty result.
/// * `penalty` — the per-change-point penalty `β ≥ 0`; larger values yield fewer
///   change points. (A negative penalty is treated as `0`.)
/// * `min_size` — the minimum number of points in any segment (`ruptures`' `min_size`;
///   clamped to at least `1`).
///
/// # Returns
///
/// The segment-end indices in increasing order, ending in `signal.len()`, matching
/// `ruptures`' breakpoint convention.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::change_point::pelt_l2;
///
/// // A clean step from 0 to 10 at index 3 (length 6).
/// let signal = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
/// let bkps = pelt_l2(&signal, 1.0, 1);
/// assert_eq!(bkps, vec![3, 6], "breakpoints were {bkps:?}");
/// ```
#[must_use]
pub fn pelt_l2(signal: &[f64], penalty: f64, min_size: usize) -> Vec<usize> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let min_size = min_size.max(1);
    let beta = penalty.max(0.0);
    let (prefix, prefix_sq) = prefix_sums(signal);

    // `best_cost[t]` = minimal penalized cost of segmenting `signal[0..t]`.
    let mut best_cost = vec![f64::INFINITY; n + 1];
    if let Some(slot) = best_cost.first_mut() {
        *slot = -beta;
    }
    // `last_change[t]` = the change point preceding the final segment ending at `t`.
    let mut last_change = vec![0_usize; n + 1];
    // Pruned set of candidate previous change points.
    let mut candidates: Vec<usize> = vec![0];

    for end in 1..=n {
        if end < min_size {
            continue;
        }
        let mut best = f64::INFINITY;
        let mut best_start = 0_usize;
        for &start in &candidates {
            if end - start < min_size {
                continue;
            }
            let prior = best_cost.get(start).copied().unwrap_or(f64::INFINITY);
            if !prior.is_finite() {
                continue;
            }
            let cost = prior + segment_cost(&prefix, &prefix_sq, start, end) + beta;
            if cost < best {
                best = cost;
                best_start = start;
            }
        }
        if let Some(slot) = best_cost.get_mut(end) {
            *slot = best;
        }
        if let Some(slot) = last_change.get_mut(end) {
            *slot = best_start;
        }
        prune(
            &mut candidates,
            &best_cost,
            &prefix,
            &prefix_sq,
            end,
            beta,
            min_size,
        );
    }

    backtrack(&last_change, n)
}

/// Computes the inclusive prefix sums of `signal` and its squares.
///
/// Returns `(prefix, prefix_sq)` of length `n + 1` where `prefix[i] = Σ signal[..i]`,
/// enabling O(1) segment sums.
fn prefix_sums(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let mut prefix = vec![0.0_f64; n + 1];
    let mut prefix_sq = vec![0.0_f64; n + 1];
    let mut running = 0.0_f64;
    let mut running_sq = 0.0_f64;
    for (i, &x) in signal.iter().enumerate() {
        running += x;
        running_sq = x.mul_add(x, running_sq);
        if let Some(slot) = prefix.get_mut(i + 1) {
            *slot = running;
        }
        if let Some(slot) = prefix_sq.get_mut(i + 1) {
            *slot = running_sq;
        }
    }
    (prefix, prefix_sq)
}

/// Least-squares cost of the segment `signal[start..end]`.
///
/// Equal to `Σ xᵢ² − (Σ xᵢ)² / len`, the sum of squared deviations from the segment
/// mean, which `ruptures`' `CostL2` minimizes.
fn segment_cost(prefix: &[f64], prefix_sq: &[f64], start: usize, end: usize) -> f64 {
    let len = count_to_f64(end - start);
    if len <= 0.0 {
        return 0.0;
    }
    let sum = prefix.get(end).copied().unwrap_or(0.0) - prefix.get(start).copied().unwrap_or(0.0);
    let sum_sq =
        prefix_sq.get(end).copied().unwrap_or(0.0) - prefix_sq.get(start).copied().unwrap_or(0.0);
    sum_sq - (sum * sum) / len
}

/// Prunes candidate change points that can never beat the current best.
///
/// A candidate `start` is discarded when `best_cost[start] + cost(start..end)` already
/// exceeds `best_cost[end]` (the PELT pruning inequality, with `K = 0` for the L2
/// cost): such a `start` cannot be optimal for any later end either. New candidate
/// `end` is admitted once a full `min_size` segment can follow it.
fn prune(
    candidates: &mut Vec<usize>,
    best_cost: &[f64],
    prefix: &[f64],
    prefix_sq: &[f64],
    end: usize,
    beta: f64,
    min_size: usize,
) {
    let reference = best_cost.get(end).copied().unwrap_or(f64::INFINITY);
    candidates.retain(|&start| {
        let prior = best_cost.get(start).copied().unwrap_or(f64::INFINITY);
        if !prior.is_finite() {
            return false;
        }
        prior + segment_cost(prefix, prefix_sq, start, end) <= reference + beta
    });
    if end + min_size <= prefix.len().saturating_sub(1) || end >= min_size {
        candidates.push(end);
    }
}

/// Rebuilds the increasing breakpoint list from the `last_change` back-pointers.
///
/// Walks from the signal length `n` back to `0` collecting each segment end, then
/// reverses so the result is increasing and ends in `n` (the `ruptures` convention).
fn backtrack(last_change: &[usize], n: usize) -> Vec<usize> {
    let mut breakpoints = Vec::new();
    let mut end = n;
    while end > 0 {
        breakpoints.push(end);
        let prev = last_change.get(end).copied().unwrap_or(0);
        if prev >= end {
            break;
        }
        end = prev;
    }
    breakpoints.reverse();
    breakpoints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_step_is_detected() {
        let signal = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let bkps = pelt_l2(&signal, 1.0, 1);
        assert_eq!(bkps, vec![3, 6], "breakpoints were {bkps:?}");
    }

    #[test]
    fn constant_signal_has_no_interior_change_point() {
        let signal = [4.0, 4.0, 4.0, 4.0, 4.0];
        let bkps = pelt_l2(&signal, 5.0, 1);
        assert_eq!(bkps, vec![5], "breakpoints were {bkps:?}");
    }

    #[test]
    fn empty_signal_is_empty() {
        assert!(pelt_l2(&[], 1.0, 1).is_empty());
    }

    #[test]
    fn high_penalty_suppresses_weak_change_points() {
        // A tiny bump that a large penalty should refuse to split on.
        let signal = [0.0, 0.0, 0.1, 0.0, 0.0];
        let bkps = pelt_l2(&signal, 100.0, 1);
        assert_eq!(bkps, vec![5], "breakpoints were {bkps:?}");
    }
}
