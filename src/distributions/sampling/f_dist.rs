//! F-distribution numerics, for the [`FDistribution`].
//!
//! Equivalent to `scipy.stats.f(numerator_df, denominator_df)`. The density uses
//! `ln_beta`, the CDF is the regularized incomplete beta `betai`, the quantile
//! inverts it by bisection, and sampling forms the ratio of two scaled
//! chi-squared draws, `(χ²_{d1}/d1) / (χ²_{d2}/d2)`.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Cdf, Moments};
//! use stats_claw::distributions::FDistribution;
//!
//! let d = FDistribution { numerator_df: 6, denominator_df: 12, ..Default::default() };
//! // Mean is d2 / (d2 - 2) = 12 / 10 = 1.2.
//! assert!(matches!(d.mean(), Some(m) if (m - 1.2).abs() < 1e-12));
//! // CDF saturates toward 1 in the right tail.
//! assert!(d.cdf(1_000.0) > 1.0 - 1e-9);
//! ```

use super::super::{Cdf, LogCdf, Moments, Pdf, Quantile, Sample, bisection_quantile, count_to_f64};
use crate::distributions::FDistribution;
use crate::rng::SplitMix64;
use crate::special::{betai, ln_beta, ln_betai_lower, ln_betai_upper};

impl FDistribution {
    /// Numerator degrees of freedom as a float (`d1`).
    fn d1(&self) -> f64 {
        count_to_f64(self.numerator_df)
    }
    /// Denominator degrees of freedom as a float (`d2`).
    fn d2(&self) -> f64 {
        count_to_f64(self.denominator_df)
    }
}

impl Pdf for FDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (d1, d2) = (self.d1(), self.d2());
        // log f(x) = ½d1·ln(d1·x) + ½d2·ln d2 − ½(d1+d2)·ln(d1·x+d2) − ln B − ln x
        let log_num = (0.5 * d1).mul_add((d1 * x).ln(), 0.5 * d2 * d2.ln());
        let log_den =
            (0.5 * (d1 + d2)).mul_add(d1.mul_add(x, d2).ln(), ln_beta(0.5 * d1, 0.5 * d2));
        (log_num - log_den - x.ln()).exp()
    }
}

impl Cdf for FDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (d1, d2) = (self.d1(), self.d2());
        let d1x = d1 * x;
        betai(0.5 * d1, 0.5 * d2, d1x / (d1x + d2))
    }
}

impl LogCdf for FDistribution {
    fn logsf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (d1, d2) = (self.d1(), self.d2());
        let d1x = d1 * x;
        // sf = 1 − I_w(d1/2, d2/2); the upper log tail keeps the small side fast.
        ln_betai_upper(0.5 * d1, 0.5 * d2, d1x / (d1x + d2))
    }
    fn logcdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let (d1, d2) = (self.d1(), self.d2());
        let d1x = d1 * x;
        ln_betai_lower(0.5 * d1, 0.5 * d2, d1x / (d1x + d2))
    }
}

impl Quantile for FDistribution {
    fn quantile(&self, p: f64) -> f64 {
        // Heavy right tail; a wide bracket covers the near-1 grid point.
        bisection_quantile(p, 0.0, 1.0e7, |x| self.cdf(x))
    }
}

impl Moments for FDistribution {
    fn mean(&self) -> Option<f64> {
        let d2 = self.d2();
        if self.denominator_df > 2 {
            Some(d2 / (d2 - 2.0))
        } else {
            None
        }
    }
    fn variance(&self) -> Option<f64> {
        if self.denominator_df <= 4 {
            return None;
        }
        let (d1, d2) = (self.d1(), self.d2());
        let numerator = 2.0 * d2 * d2 * (d1 + d2 - 2.0);
        let denominator = d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0);
        Some(numerator / denominator)
    }
}

impl Sample for FDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        let num = chi_squared_over_df(self.numerator_df, rng);
        let den = chi_squared_over_df(self.denominator_df, rng);
        num / den
    }
}

/// Draws `χ²_df / df` — a scaled chi-squared with `df` degrees of freedom.
fn chi_squared_over_df(df: i64, rng: &mut SplitMix64) -> f64 {
    let mut sum = 0.0;
    for _ in 0..df.max(1) {
        let z = rng.standard_normal();
        sum = z.mul_add(z, sum);
    }
    sum / count_to_f64(df.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mean is `d2 / (d2 − 2)` when the denominator df exceeds 2.
    #[test]
    fn mean_for_large_denominator() {
        let d = FDistribution {
            numerator_df: 6,
            denominator_df: 12,
            ..Default::default()
        };
        assert!(matches!(d.mean(), Some(m) if (m - 12.0 / 10.0).abs() < 1e-12));
    }

    /// The CDF saturates toward 1 in the upper tail.
    #[test]
    fn cdf_saturates() {
        let d = FDistribution {
            numerator_df: 6,
            denominator_df: 12,
            ..Default::default()
        };
        assert!(d.cdf(1e6) > 1.0 - 1e-9);
    }

    /// `logsf` agrees with the log of the linear survival in the body and matches
    /// scipy `f.logsf` deep in the tail (d1 = 3, d2 = 10, x = 1e4).
    #[test]
    fn logsf_matches_scipy_and_stays_finite() {
        let d = FDistribution {
            numerator_df: 3,
            denominator_df: 10,
            ..Default::default()
        };
        let body = d.logsf(1.0);
        assert!(
            ((body - (1.0 - d.cdf(1.0)).ln()) / body.abs()).abs() < 1e-9,
            "logsf body was {body}"
        );
        let tail = d.logsf(1e4);
        let want = -39.037_790_534_655_89; // scipy.stats.f.logsf(1e4, 3, 10)
        assert!(tail.is_finite(), "logsf(1e4) was {tail}");
        assert!(
            ((tail - want) / want).abs() < 1e-9,
            "logsf(1e4) = {tail}, want {want}"
        );
    }
}
