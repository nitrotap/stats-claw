//! Pearson chi-squared test of independence for an r×c contingency table.

use crate::error::{Error, Result};
use crate::tests_stat::TestResult;

/// Computes the Pearson chi-squared statistic and its degrees of freedom for a
/// contingency `table`, validating shape and finiteness.
///
/// Shared by [`chi_squared_independence`] and the Cramér's V effect size, which
/// both need the χ² statistic and `(rows, cols)` shape.
///
/// # Arguments
///
/// * `table` — an r×c contingency table of non-negative counts; every row must
///   have the same width.
///
/// # Returns
///
/// `(statistic, df, rows, cols)`.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] for an empty/zero-width table,
/// [`Error::DegenerateInput`] for a table smaller than 2×2 or with ragged rows,
/// and [`Error::InvalidInput`] for a negative, NaN, or infinite count.
pub(crate) fn chi_squared_stat(table: &[Vec<f64>]) -> Result<(f64, f64, usize, usize)> {
    let rows = table.len();
    if rows == 0 || table.first().is_none_or(Vec::is_empty) {
        return Err(Error::EmptyInput);
    }
    let cols = table.first().map_or(0, Vec::len);
    if rows < 2 || cols < 2 {
        return Err(Error::DegenerateInput(
            "need at least a 2x2 table".to_owned(),
        ));
    }
    for row in table {
        if row.len() != cols {
            return Err(Error::DegenerateInput(
                "ragged contingency table".to_owned(),
            ));
        }
        for &count in row {
            if !count.is_finite() || count < 0.0 {
                return Err(Error::InvalidInput(
                    "counts must be finite and non-negative".to_owned(),
                ));
            }
        }
    }

    let row_sums: Vec<f64> = table.iter().map(|row| row.iter().sum()).collect();
    let mut col_sums = vec![0.0; cols];
    for row in table {
        for (j, &count) in row.iter().enumerate() {
            if let Some(sum) = col_sums.get_mut(j) {
                *sum += count;
            }
        }
    }
    let total: f64 = row_sums.iter().sum();
    if total <= 0.0 {
        return Err(Error::DegenerateInput("table total is zero".to_owned()));
    }

    let mut stat = 0.0;
    for (i, row) in table.iter().enumerate() {
        let row_i = row_sums.get(i).copied().unwrap_or(0.0);
        for (j, &observed) in row.iter().enumerate() {
            let col_j = col_sums.get(j).copied().unwrap_or(0.0);
            let expected = row_i * col_j / total;
            if expected > 0.0 {
                let diff = observed - expected;
                stat += diff * diff / expected;
            }
        }
    }
    let df = pairs_df(rows, cols);
    Ok((stat, df, rows, cols))
}

/// Returns `(rows-1)*(cols-1)` as an `f64` without an `as` cast.
fn pairs_df(rows: usize, cols: usize) -> f64 {
    let dof = (rows - 1) * (cols - 1);
    let lo = u32::try_from(dof).unwrap_or(u32::MAX);
    f64::from(lo)
}

/// Pearson chi-squared test of independence (no continuity correction), matching
/// `scipy.stats.chi2_contingency(table, correction=False)`.
///
/// The statistic is `Σ (O − E)² / E` with expected counts `E_ij = rowᵢ·colⱼ / n`,
/// and the p-value is the upper-tail probability `P(χ²_df ≥ statistic)` evaluated
/// through [`crate::distributions::ChiSquaredDistribution`]'s CDF.
///
/// # Arguments
///
/// * `table` — an r×c contingency table of non-negative observed counts.
///
/// # Returns
///
/// A [`TestResult`] with the χ² statistic, the upper-tail p-value, `df =
/// (r−1)(c−1)`, and `effect_size = None` (Cramér's V is reported separately).
///
/// # Errors
///
/// Propagates the validation errors of `chi_squared_stat`: empty, ragged,
/// sub-2×2, or invalid-count tables.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::categorical::chi_squared_independence;
///
/// let table = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
/// let r = chi_squared_independence(&table)?;
/// assert_eq!(r.df, Some(1.0));
/// assert!((0.0..=1.0).contains(&r.p_value));
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn chi_squared_independence(table: &[Vec<f64>]) -> Result<TestResult> {
    let (statistic, df, r, c) = chi_squared_stat(table)?;
    let int_df = i64::try_from((r - 1) * (c - 1)).unwrap_or(i64::MAX);
    let p_value = crate::tests_stat::chi_squared_upper_tail(statistic, int_df);
    let log_p_value = crate::tests_stat::chi_squared_upper_log_tail(statistic, int_df);
    Ok(TestResult {
        statistic,
        p_value,
        log_p_value: Some(log_p_value),
        df: Some(df),
        effect_size: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 table's degrees of freedom is 1 and the statistic is non-negative.
    #[test]
    fn two_by_two_has_one_df() -> Result<()> {
        let table = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let r = chi_squared_independence(&table)?;
        assert_eq!(r.df, Some(1.0), "2x2 table must have 1 df");
        assert!(r.statistic >= 0.0, "statistic was {}", r.statistic);
        Ok(())
    }

    /// Negative counts are rejected before any computation.
    #[test]
    fn negative_counts_are_invalid() {
        let table = vec![vec![-1.0, 2.0], vec![3.0, 4.0]];
        assert!(matches!(
            chi_squared_independence(&table),
            Err(Error::InvalidInput(_))
        ));
    }
}
