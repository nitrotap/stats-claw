//! Percentile confidence intervals and the bootstrap-statistic helper they
//! consume.
//!
//! [`percentile_ci`] is validated for reference equivalence against
//! `scipy.stats.bootstrap` golden fixtures downstream.

use super::index::floor_rank;
use super::schemes::bootstrap_indices;
use crate::distributions::NormalDistribution;
use crate::distributions::Sample;
use crate::error::{Error, Result};
use crate::rng::SplitMix64;

/// Computes the two-sided percentile confidence interval of `samples`.
///
/// Sorts a copy of `samples` by total order (NaN-safe) and returns the
/// `alpha/2` and `1 - alpha/2` percentile values by nearest-rank: the lower bound
/// is the element at rank `floor((alpha/2) * n)` and the upper bound the element
/// at rank `floor((1 - alpha/2) * n)`, both clamped to `0..n`. This is the
/// percentile-method interval used for the bootstrap distribution of a statistic.
///
/// # Arguments
///
/// * `samples` — the (e.g. bootstrap) statistic values; order does not matter.
/// * `alpha` — the total tail mass; e.g. `0.05` for a 95% interval. Expected in
///   `[0, 1]`.
///
/// # Returns
///
/// `(lower, upper)` percentile bounds drawn from `samples`.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] when `samples` is empty (no interval is defined).
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::percentile_ci;
///
/// let xs: Vec<f64> = (0..1000).map(f64::from).collect();
/// let (lo, hi) = percentile_ci(&xs, 0.05)?;
/// assert!(lo < 500.0 && hi > 500.0);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn percentile_ci(samples: &[f64], alpha: f64) -> Result<(f64, f64)> {
    if samples.is_empty() {
        return Err(Error::EmptyInput);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let lo_rank = floor_rank(alpha / 2.0, n).min(n - 1);
    let hi_rank = floor_rank(1.0 - alpha / 2.0, n).min(n - 1);
    let lo = sorted.get(lo_rank).copied().ok_or(Error::EmptyInput)?;
    let hi = sorted.get(hi_rank).copied().ok_or(Error::EmptyInput)?;
    Ok((lo, hi))
}

/// Computes the bootstrap distribution of a statistic over `data`.
///
/// Draws `b` with-replacement resamples of `data` (via [`bootstrap_indices`]),
/// evaluates `statistic` on each resampled view, and returns the `b` results.
/// Pairing this with [`percentile_ci`] yields a percentile bootstrap interval.
///
/// # Arguments
///
/// * `data` — the observed sample.
/// * `b` — number of bootstrap resamples.
/// * `rng` — the deterministic generator driving the resampling.
/// * `statistic` — maps a resampled view of the data to a scalar (e.g. the
///   median).
///
/// # Returns
///
/// A vector of `b` statistic values, one per resample.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] when `data` is empty.
pub fn bootstrap_statistic<F>(
    data: &[f64],
    b: usize,
    rng: &mut SplitMix64,
    statistic: F,
) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    if data.is_empty() {
        return Err(Error::EmptyInput);
    }
    let n = data.len();
    let draws = bootstrap_indices(n, b, rng);
    let mut out = Vec::with_capacity(b);
    for indices in &draws {
        let view: Vec<f64> = indices
            .iter()
            .map(|&i| data.get(i).copied().unwrap_or(f64::NAN))
            .collect();
        out.push(statistic(&view));
    }
    Ok(out)
}

/// Estimates the empirical coverage of a percentile interval for the mean of a
/// known normal, over seeded Monte-Carlo replications.
///
/// Each replication draws `n` variates from `Normal(mean, std_dev)`, forms the
/// `1 − alpha` percentile interval of the sample, and checks whether it brackets
/// the true `mean`. The returned rate should approach the nominal `1 − alpha` as
/// `replications` grows.
///
/// # Arguments
///
/// * `mean`, `std_dev` — the true sampling distribution (`std_dev > 0`).
/// * `n` — observations per replication.
/// * `replications` — number of seeded replications.
/// * `alpha` — total tail mass of each interval (e.g. `0.05`).
/// * `rng` — the deterministic generator driving every draw.
///
/// # Returns
///
/// The fraction of replications whose interval contained `mean`, in `[0, 1]`.
#[must_use]
pub fn coverage_rate(
    mean: f64,
    std_dev: f64,
    n: usize,
    replications: usize,
    alpha: f64,
    rng: &mut SplitMix64,
) -> f64 {
    let dist = NormalDistribution {
        mean,
        standard_deviation: std_dev,
        ..Default::default()
    };
    let mut covered = 0u32;
    for _ in 0..replications {
        let sample: Vec<f64> = (0..n).map(|_| dist.sample(rng)).collect();
        if let Ok((lo, hi)) = percentile_ci(&sample, alpha)
            && lo <= mean
            && mean <= hi
        {
            covered += 1;
        }
    }
    let reps = u32::try_from(replications).unwrap_or(u32::MAX).max(1);
    f64::from(covered) / f64::from(reps)
}
