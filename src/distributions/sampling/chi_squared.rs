//! Chi-squared distribution numerics, for the [`ChiSquaredDistribution`] parameter struct.
//!
//! Equivalent to `scipy.stats.chi2(degrees_of_freedom)`: the distribution of a
//! sum of `k` squared standard normals. The density is closed form, the CDF is
//! the regularized lower incomplete gamma `gamma_p`, the quantile inverts it by
//! bisection, and sampling sums squared `standard_normal` draws.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Cdf, Moments};
//! use stats_claw::distributions::ChiSquaredDistribution;
//!
//! let d = ChiSquaredDistribution { degrees_of_freedom: 5, ..Default::default() };
//! // Mean equals the degrees of freedom.
//! assert_eq!(d.mean(), Some(5.0));
//! // CDF at 0 is 0; CDF at a very large value is ≈ 1.
//! assert!(d.cdf(0.0).abs() < 1e-12);
//! assert!(d.cdf(1_000.0) > 1.0 - 1e-9);
//! ```

use super::super::{bisection_quantile, count_to_f64, Cdf, LogCdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::ChiSquaredDistribution;
use crate::rng::SplitMix64;
use crate::special::{gamma_p, ln_gamma, ln_gamma_p, ln_gamma_q};

impl ChiSquaredDistribution {
    /// Degrees of freedom as a float (`k`).
    fn k(&self) -> f64 {
        count_to_f64(self.degrees_of_freedom)
    }
}

impl Pdf for ChiSquaredDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            // Density is 0 for k>2, 0.5 for k=2, and diverges for k<2; scipy's
            // grid starts at 0 so report the k≥2 closed-form limit.
            return if (self.k() - 2.0).abs() < 1e-12 {
                0.5
            } else {
                0.0
            };
        }
        let half_k = 0.5 * self.k();
        let log_norm = half_k.mul_add(2.0_f64.ln(), ln_gamma(half_k));
        let tail = (-0.5f64).mul_add(x, -log_norm);
        let ln_density = (half_k - 1.0).mul_add(x.ln(), tail);
        ln_density.exp()
    }
}

impl Cdf for ChiSquaredDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gamma_p(0.5 * self.k(), 0.5 * x)
        }
    }
}

impl LogCdf for ChiSquaredDistribution {
    fn logsf(&self, x: f64) -> f64 {
        // sf(x) = Q(k/2, x/2); its log stays finite where Q underflows.
        if x <= 0.0 {
            0.0
        } else {
            ln_gamma_q(0.5 * self.k(), 0.5 * x)
        }
    }
    fn logcdf(&self, x: f64) -> f64 {
        // cdf(x) = P(k/2, x/2).
        if x <= 0.0 {
            f64::NEG_INFINITY
        } else {
            ln_gamma_p(0.5 * self.k(), 0.5 * x)
        }
    }
}

impl Quantile for ChiSquaredDistribution {
    fn quantile(&self, p: f64) -> f64 {
        let k = self.k();
        let hi = 20.0f64.mul_add((2.0 * k).sqrt(), k).max(1.0);
        bisection_quantile(p, 0.0, hi, |x| self.cdf(x))
    }
}

impl Moments for ChiSquaredDistribution {
    fn mean(&self) -> Option<f64> {
        Some(self.k())
    }
    fn variance(&self) -> Option<f64> {
        Some(2.0 * self.k())
    }
}

impl Sample for ChiSquaredDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        let mut sum = 0.0;
        for _ in 0..self.degrees_of_freedom.max(0) {
            let z = rng.standard_normal();
            sum = z.mul_add(z, sum);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mean equals the degrees of freedom and variance twice that.
    #[test]
    fn moments_are_df_and_twice_df() {
        let d = ChiSquaredDistribution {
            degrees_of_freedom: 5,
            ..Default::default()
        };
        assert_eq!(d.mean(), Some(5.0));
        assert_eq!(d.variance(), Some(10.0));
    }

    /// The CDF is monotone and saturates toward 1 in the upper tail.
    #[test]
    fn cdf_saturates() {
        let d = ChiSquaredDistribution {
            degrees_of_freedom: 3,
            ..Default::default()
        };
        assert!(d.cdf(0.0).abs() < 1e-12);
        assert!(d.cdf(1e6) > 1.0 - 1e-9);
    }

    /// `logsf` agrees with the log of the linear survival in the body and matches
    /// scipy `chi2.logsf` deep in the tail (df = 3, x = 200).
    #[test]
    fn logsf_matches_scipy_and_stays_finite() {
        let d = ChiSquaredDistribution {
            degrees_of_freedom: 3,
            ..Default::default()
        };
        let body = d.logsf(2.0);
        assert!(
            ((body - (1.0 - d.cdf(2.0)).ln()) / body.abs()).abs() < 1e-9,
            "logsf body was {body}"
        );
        let tail = d.logsf(200.0);
        let want = -97.571_669_639_663_45; // scipy.stats.chi2.logsf(200, 3)
        assert!(tail.is_finite(), "logsf(200) was {tail}");
        assert!(
            ((tail - want) / want).abs() < 1e-9,
            "logsf(200) = {tail}, want {want}"
        );
    }
}
