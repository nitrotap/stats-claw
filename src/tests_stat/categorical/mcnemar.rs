//! The `McNemar` test for paired binary data (asymptotic, corrected, and exact).

use crate::error::{Error, Result};
use crate::special::ln_gamma;
use crate::tests_stat::parametric::floor_to_i64;
use crate::tests_stat::{TestResult, chi_squared_upper_log_tail, chi_squared_upper_tail};

/// The `McNemar` test on a 2×2 paired table, matching
/// `statsmodels.stats.contingency_tables.mcnemar(table, exact, correction)`.
///
/// Uses only the discordant off-diagonal counts `b = table[0][1]` and
/// `c = table[1][0]`. The asymptotic statistic is `(b − c)² / (b + c)`, chi-squared
/// with one degree of freedom; with `correction` it becomes `(|b − c| − 1)² /
/// (b + c)`. The exact path reports a two-sided binomial p-value (no statistic).
///
/// # Arguments
///
/// * `table` — a 2×2 paired-outcome table.
/// * `exact` — use the exact binomial p-value (for small `b + c`).
/// * `correction` — apply the continuity correction (asymptotic path only).
///
/// # Returns
///
/// A [`TestResult`] with the `McNemar` statistic (`NaN` on the exact path), the
/// p-value, `df = Some(1.0)` for the asymptotic path (`None` for exact), and no
/// effect size. When `b + c = 0` the statistic is undefined and the p-value is 1.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for a non-2×2 table or a negative/non-finite
/// count.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::categorical::mcnemar;
///
/// // Equal discordant counts give a zero statistic and p = 1.
/// let table = vec![vec![5.0, 7.0], vec![7.0, 5.0]];
/// let r = mcnemar(&table, false, false)?;
/// assert!(r.statistic.abs() < 1e-12, "statistic was {}", r.statistic);
/// assert!((r.p_value - 1.0).abs() < 1e-9, "p was {}", r.p_value);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn mcnemar(table: &[Vec<f64>], exact: bool, correction: bool) -> Result<TestResult> {
    let (b, c) = discordant(table)?;
    let total = b + c;
    if total == 0.0 {
        return Ok(TestResult {
            statistic: f64::NAN,
            p_value: 1.0,
            log_p_value: if exact { None } else { Some(0.0) },
            df: if exact { None } else { Some(1.0) },
            effect_size: None,
        });
    }

    if exact {
        let p_value = exact_binomial_two_sided(b, c);
        return Ok(TestResult {
            statistic: f64::NAN,
            p_value,
            // Exact binomial p; no continuous-tail log-sf path.
            log_p_value: None,
            df: None,
            effect_size: None,
        });
    }

    let diff = (b - c).abs();
    let numerator = if correction {
        let adj = (diff - 1.0).max(0.0);
        adj * adj
    } else {
        diff * diff
    };
    let statistic = numerator / total;
    let p_value = chi_squared_upper_tail(statistic, 1);
    Ok(TestResult {
        statistic,
        p_value,
        log_p_value: Some(chi_squared_upper_log_tail(statistic, 1)),
        df: Some(1.0),
        effect_size: None,
    })
}

/// Extracts the off-diagonal discordant counts `(b, c)` of a 2×2 table.
fn discordant(table: &[Vec<f64>]) -> Result<(f64, f64)> {
    let bad = || Error::InvalidInput("McNemar requires a 2x2 table".to_owned());
    if table.len() != 2 {
        return Err(bad());
    }
    let row0 = table.first().ok_or_else(bad)?;
    let row1 = table.get(1).ok_or_else(bad)?;
    if row0.len() != 2 || row1.len() != 2 {
        return Err(bad());
    }
    let b = *row0.get(1).ok_or_else(bad)?;
    let c = *row1.first().ok_or_else(bad)?;
    if !b.is_finite() || !c.is_finite() || b < 0.0 || c < 0.0 {
        return Err(Error::InvalidInput(
            "counts must be finite and non-negative".to_owned(),
        ));
    }
    Ok((b, c))
}

/// Two-sided exact binomial p-value of `b` successes in `b + c` trials at `p = ½`.
///
/// Matches statsmodels' exact `McNemar`: `2 · Σ_{k ≤ min(b, c)} C(n, k) (½)ⁿ`,
/// clamped to 1.
fn exact_binomial_two_sided(b: f64, c: f64) -> f64 {
    let n = b + c;
    let smaller = floor_to_i64(b.min(c));
    let n_i = floor_to_i64(n);
    let ln_half_n = n * 0.5f64.ln();
    let mut tail = 0.0;
    for k in 0..=smaller {
        tail += (log_choose(n_i, k) + ln_half_n).exp();
    }
    (2.0 * tail).min(1.0)
}

/// Log binomial coefficient `ln C(n, k)` for integer arguments.
fn log_choose(n: i64, k: i64) -> f64 {
    if k < 0 || k > n {
        return f64::NEG_INFINITY;
    }
    let nf = f64::from(i32::try_from(n).unwrap_or(i32::MAX));
    let kf = f64::from(i32::try_from(k).unwrap_or(i32::MAX));
    ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equal discordant counts give a zero asymptotic statistic and p = 1.
    #[test]
    fn equal_discordant_is_zero_statistic() -> Result<()> {
        let table = vec![vec![5.0, 7.0], vec![7.0, 5.0]];
        let r = mcnemar(&table, false, false)?;
        assert!(r.statistic.abs() < 1e-12, "statistic was {}", r.statistic);
        assert!((r.p_value - 1.0).abs() < 1e-9, "p was {}", r.p_value);
        Ok(())
    }

    /// A non-2×2 table is rejected.
    #[test]
    fn non_2x2_is_invalid() {
        let table = vec![vec![1.0, 2.0, 3.0]];
        assert!(matches!(
            mcnemar(&table, false, false),
            Err(Error::InvalidInput(_))
        ));
    }
}
