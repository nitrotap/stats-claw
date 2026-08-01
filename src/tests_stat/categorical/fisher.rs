//! Fisher's exact test for a 2×2 contingency table (hypergeometric).

use crate::error::{Error, Result};
use crate::special::ln_gamma;
use crate::tests_stat::parametric::floor_to_i64;
use crate::tests_stat::{Alternative, TestResult};

/// Fisher's exact test on a 2×2 contingency `table`, matching
/// `scipy.stats.fisher_exact(table, alternative=…)`.
///
/// Conditions on the table margins so the count in the top-left cell follows a
/// hypergeometric distribution, then sums exact tail probabilities. The reported
/// effect size is the sample odds ratio `ad / bc` (scipy's `statistic`). The
/// two-sided p-value sums every table at least as unlikely as the observed one.
///
/// # Arguments
///
/// * `table` — a 2×2 table `[[a, b], [c, d]]` of non-negative integer counts.
/// * `alternative` — `Greater`/`Less` test the odds ratio against 1.
///
/// # Returns
///
/// A [`TestResult`] with `statistic = odds ratio`, the exact p-value, no df, and
/// `effect_size = Some(odds ratio)`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for a non-2×2 table or a negative/non-finite
/// count.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::{categorical::fisher_exact, Alternative};
///
/// // A balanced 2×2 table has an odds ratio of 1.
/// let table = vec![vec![5.0, 5.0], vec![5.0, 5.0]];
/// let r = fisher_exact(&table, Alternative::TwoSided)?;
/// assert!((r.statistic - 1.0).abs() < 1e-12, "odds ratio was {}", r.statistic);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn fisher_exact(table: &[Vec<f64>], alternative: Alternative) -> Result<TestResult> {
    let cell = cells(table)?;
    let observed = cell.a;
    let row1 = cell.a + cell.b;
    let row2 = cell.c + cell.d;
    let col1 = cell.a + cell.c;

    // Integer support of the top-left cell given the margins.
    let lo = floor_to_i64((col1 - row2).max(0.0));
    let hi = floor_to_i64(col1.min(row1));
    let p_observed = log_hypergeom(observed, row1, row2, col1).exp();
    // scipy includes every table whose probability is ≤ p_observed·(1 + 1e-7).
    let threshold = p_observed * (1.0 + 1e-7);

    let mut p_two = 0.0;
    let mut p_less = 0.0;
    let mut p_greater = 0.0;
    for step in lo..=hi {
        let k = f64::from(i32::try_from(step).unwrap_or(i32::MAX));
        let pk = log_hypergeom(k, row1, row2, col1).exp();
        if k <= observed + 0.5 {
            p_less += pk;
        }
        if k >= observed - 0.5 {
            p_greater += pk;
        }
        if pk <= threshold {
            p_two += pk;
        }
    }

    let odds_ratio = odds(cell.a, cell.b, cell.c, cell.d);
    let p_value = match alternative {
        Alternative::Less => p_less.clamp(0.0, 1.0),
        Alternative::Greater => p_greater.clamp(0.0, 1.0),
        Alternative::TwoSided => p_two.clamp(0.0, 1.0),
    };
    Ok(TestResult {
        statistic: odds_ratio,
        p_value,
        // Fisher's exact hypergeometric p; no continuous-tail log-sf path.
        log_p_value: None,
        df: None,
        effect_size: Some(odds_ratio),
    })
}

/// The four cells of a 2×2 table, named for the standard `[[a, b], [c, d]]`
/// layout.
#[derive(Debug, Clone, Copy)]
struct Cells {
    /// Top-left count.
    a: f64,
    /// Top-right count.
    b: f64,
    /// Bottom-left count.
    c: f64,
    /// Bottom-right count.
    d: f64,
}

/// Extracts and validates the four cells of a 2×2 table.
fn cells(table: &[Vec<f64>]) -> Result<Cells> {
    let bad = || Error::InvalidInput("Fisher exact requires a 2x2 table".to_owned());
    if table.len() != 2 {
        return Err(bad());
    }
    let row0 = table.first().ok_or_else(bad)?;
    let row1 = table.get(1).ok_or_else(bad)?;
    if row0.len() != 2 || row1.len() != 2 {
        return Err(bad());
    }
    let cell = Cells {
        a: *row0.first().ok_or_else(bad)?,
        b: *row0.get(1).ok_or_else(bad)?,
        c: *row1.first().ok_or_else(bad)?,
        d: *row1.get(1).ok_or_else(bad)?,
    };
    for value in [cell.a, cell.b, cell.c, cell.d] {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::InvalidInput(
                "counts must be finite and non-negative".to_owned(),
            ));
        }
    }
    Ok(cell)
}

/// Sample odds ratio `ad / bc`, with scipy's `+∞` for a zero `bc` denominator.
fn odds(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let denom = b * c;
    if denom == 0.0 {
        if a * d == 0.0 {
            f64::NAN
        } else {
            f64::INFINITY
        }
    } else {
        a * d / denom
    }
}

/// Log hypergeometric probability of `k` successes given the table margins.
fn log_hypergeom(k: f64, row1: f64, row2: f64, col1: f64) -> f64 {
    log_choose(row1, k) + log_choose(row2, col1 - k) - log_choose(row1 + row2, col1)
}

/// Log binomial coefficient `ln C(n, k)` via `ln_gamma` (`−∞` outside support).
fn log_choose(n: f64, k: f64) -> f64 {
    if k < 0.0 || k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// Kani formal-verification harness for Fisher's exact table-validation layer.
///
/// Compiled only under `cargo kani`. A one-row table trips the `table.len() != 2`
/// guard in the private `cells` extractor, so [`fisher_exact`] returns `Err` before
/// the hypergeometric enumeration; the symbolic cell values are never inspected.
#[cfg(kani)]
mod verification {
    use super::fisher_exact;
    use crate::error::Error;
    use crate::tests_stat::Alternative;

    /// Proves Fisher's exact test rejects a non-2×2 table via `Err` without
    /// panicking, for arbitrary symbolic counts.
    #[kani::proof]
    // Live path returns Err before any loop; the unwind bound caps the dead
    // transcendental-tail branches CBMC unwinds during model construction.
    #[kani::unwind(2)]
    fn fisher_rejects_non_2x2() {
        let table: [Vec<f64>; 1] = [vec![kani::any::<f64>(), kani::any::<f64>()]];
        assert!(
            matches!(
                fisher_exact(&table, Alternative::TwoSided),
                Err(Error::InvalidInput(_))
            ),
            "non-2x2 table must reject with InvalidInput"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A balanced table has an odds ratio of 1.
    #[test]
    fn balanced_table_odds_ratio_is_one() -> Result<()> {
        let table = vec![vec![5.0, 5.0], vec![5.0, 5.0]];
        let r = fisher_exact(&table, Alternative::TwoSided)?;
        assert!(
            (r.statistic - 1.0).abs() < 1e-12,
            "odds ratio was {}",
            r.statistic
        );
        Ok(())
    }

    /// A non-2×2 table is rejected.
    #[test]
    fn non_2x2_is_invalid() {
        let table = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert!(matches!(
            fisher_exact(&table, Alternative::TwoSided),
            Err(Error::InvalidInput(_))
        ));
    }
}
