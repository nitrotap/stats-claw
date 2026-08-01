//! Log-normal distribution numerics, for the
//! [`LogNormalDistribution`].
//!
//! Equivalent to `scipy.stats.lognorm(s = std_log_value, scale =
//! exp(mean_log_value))`: a variable whose logarithm is normal with mean
//! `mean_log_value` and standard deviation `std_log_value`. The CDF and quantile
//! delegate to a standard [`NormalDistribution`] on the log scale, reusing its
//! erf-based machinery; sampling exponentiates a normal draw.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::Quantile;
//! use stats_claw::distributions::LogNormalDistribution;
//!
//! let d = LogNormalDistribution { mean_log_value: 0.0, std_log_value: 1.0, ..Default::default() };
//! // Median is exp(mean_log_value) = exp(0) = 1.
//! assert!((d.quantile(0.5) - 1.0).abs() < 1e-9, "median was {}", d.quantile(0.5));
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::LogNormalDistribution;
use crate::distributions::NormalDistribution;
use crate::rng::SplitMix64;
use std::f64::consts::PI;

impl LogNormalDistribution {
    /// The unit normal on the log scale this distribution is built from.
    fn log_normal(&self) -> NormalDistribution {
        NormalDistribution {
            mean: self.mean_log_value,
            standard_deviation: self.std_log_value,
            ..Default::default()
        }
    }
}

impl Pdf for LogNormalDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (mu, sigma) = (self.mean_log_value, self.std_log_value);
        let z = (x.ln() - mu) / sigma;
        (-0.5 * z * z).exp() / (x * sigma * (2.0 * PI).sqrt())
    }
}

impl Cdf for LogNormalDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            self.log_normal().cdf(x.ln())
        }
    }
}

impl Quantile for LogNormalDistribution {
    fn quantile(&self, p: f64) -> f64 {
        self.log_normal().quantile(p).exp()
    }
}

impl Moments for LogNormalDistribution {
    fn mean(&self) -> Option<f64> {
        let (mu, sigma) = (self.mean_log_value, self.std_log_value);
        Some((0.5 * sigma).mul_add(sigma, mu).exp())
    }
    fn variance(&self) -> Option<f64> {
        let (mu, sigma) = (self.mean_log_value, self.std_log_value);
        let s2 = sigma * sigma;
        Some(s2.exp_m1() * 2.0f64.mul_add(mu, s2).exp())
    }
}

impl Sample for LogNormalDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.std_log_value
            .mul_add(rng.standard_normal(), self.mean_log_value)
            .exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The median is `exp(mean_log_value)`.
    #[test]
    fn median_is_exp_mu() {
        let d = LogNormalDistribution {
            mean_log_value: 0.3,
            std_log_value: 0.6,
            ..Default::default()
        };
        assert!((d.quantile(0.5) - 0.3_f64.exp()).abs() < 1e-9);
    }

    /// The mean is `exp(μ + σ²/2)`.
    #[test]
    fn mean_matches_closed_form() {
        let d = LogNormalDistribution {
            mean_log_value: 0.0,
            std_log_value: 1.0,
            ..Default::default()
        };
        // exp(0.5) ≈ 1.648721
        assert!(matches!(d.mean(), Some(m) if (m - 1.648_721_271).abs() < 1e-6));
    }
}
