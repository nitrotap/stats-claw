//! Cramér's V effect size for a contingency table and its bootstrap confidence
//! interval.

use super::chi_squared_test::chi_squared_stat;
use crate::error::{Error, Result};
use crate::resampling::{bootstrap_indices, percentile_ci};
use crate::rng::SplitMix64;
use crate::tests_stat::parametric::floor_to_i64;

/// Computes Cramér's V, the chi-squared-based association strength of a
/// contingency `table`, matching `scipy.stats.contingency.association(method=
/// "cramer")`.
///
/// `V = √(χ² / (n · (min(r, c) − 1)))` where `χ²` is the Pearson statistic, `n`
/// the table total, and `r`, `c` the row and column counts. It lies in `[0, 1]`:
/// `0` for perfect independence, `1` for perfect association.
///
/// # Arguments
///
/// * `table` — an r×c contingency table of non-negative observed counts.
///
/// # Returns
///
/// Cramér's V in `[0, 1]`.
///
/// # Errors
///
/// Propagates the validation errors of [`chi_squared_stat`]: empty, ragged,
/// sub-2×2, or invalid-count tables.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::categorical::cramers_v;
///
/// // Rows proportional → independence → V = 0.
/// let table = vec![vec![10.0, 20.0], vec![20.0, 40.0]];
/// assert!(cramers_v(&table)? < 1e-12);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn cramers_v(table: &[Vec<f64>]) -> Result<f64> {
    let (statistic, _, rows, cols) = chi_squared_stat(table)?;
    let total: f64 = table.iter().flat_map(|row| row.iter()).sum();
    let min_dim = min_dim_minus_one(rows, cols);
    if total <= 0.0 || min_dim <= 0.0 {
        return Ok(0.0);
    }
    Ok((statistic / (total * min_dim)).sqrt())
}

/// Returns `min(rows, cols) − 1` as an `f64` without an `as` cast.
fn min_dim_minus_one(rows: usize, cols: usize) -> f64 {
    let m = rows.min(cols).saturating_sub(1);
    let lo = u32::try_from(m).unwrap_or(u32::MAX);
    f64::from(lo)
}

/// Percentile bootstrap confidence interval for Cramér's V.
///
/// Expands `table` into one cell-index observation per count (row-major), draws
/// `b` with-replacement resamples of that observation list from `rng` (via
/// [`bootstrap_indices`]), rebuilds each resampled table, recomputes Cramér's V,
/// and returns the two-sided percentile interval at level `alpha`. Using the
/// framework's deterministic RNG makes the interval byte-reproducible for a fixed
/// seed and reference-matching at that seed.
///
/// # Arguments
///
/// * `table` — an r×c contingency table of non-negative integer-valued counts.
/// * `b` — number of bootstrap resamples.
/// * `alpha` — total tail mass (e.g. `0.05` for a 95% interval).
/// * `rng` — the deterministic generator driving the resampling.
///
/// # Returns
///
/// `(lower, upper)` percentile bounds of the bootstrap Cramér's V distribution.
///
/// # Errors
///
/// Propagates [`cramers_v`]'s validation errors and returns [`Error::EmptyInput`]
/// when the table holds no observations.
pub fn cramers_v_bootstrap_ci(
    table: &[Vec<f64>],
    b: usize,
    alpha: f64,
    rng: &mut SplitMix64,
) -> Result<(f64, f64)> {
    let (_, _, rows, cols) = chi_squared_stat(table)?;
    let observations = flatten_to_cells(table, cols);
    if observations.is_empty() {
        return Err(Error::EmptyInput);
    }
    let draws = bootstrap_indices(observations.len(), b, rng);
    let mut values = Vec::with_capacity(b);
    for indices in &draws {
        let resampled = rebuild_table(&observations, indices, rows, cols);
        values.push(cramers_v(&resampled).unwrap_or(0.0));
    }
    percentile_ci(&values, alpha)
}

/// Expands a contingency table into one row-major cell index per observed count.
fn flatten_to_cells(table: &[Vec<f64>], cols: usize) -> Vec<usize> {
    let mut obs = Vec::new();
    for (i, row) in table.iter().enumerate() {
        for (j, &count) in row.iter().enumerate() {
            let reps = floor_to_i64(count.max(0.0).round());
            for _ in 0..reps {
                obs.push(i * cols + j);
            }
        }
    }
    obs
}

/// Rebuilds an r×c table from resampled cell indices.
fn rebuild_table(
    observations: &[usize],
    indices: &[usize],
    rows: usize,
    cols: usize,
) -> Vec<Vec<f64>> {
    let mut table = vec![vec![0.0; cols]; rows];
    for &idx in indices {
        if let Some(&cell) = observations.get(idx) {
            let r = cell / cols;
            let c = cell % cols;
            if let Some(slot) = table.get_mut(r).and_then(|row| row.get_mut(c)) {
                *slot += 1.0;
            }
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cramér's V of a maximally associated 2×2 table (a diagonal) is 1.
    #[test]
    fn perfect_association_is_one() -> Result<()> {
        let table = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let v = cramers_v(&table)?;
        assert!((v - 1.0).abs() < 1e-12, "V was {v}");
        Ok(())
    }
}
