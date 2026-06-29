//! Exponential distribution numerics, for the [`ExponentialDistribution`] parameter struct.
//!
//! Equivalent to `scipy.stats.expon(scale = 1 / rate_parameter)`: the waiting
//! time of a Poisson process with rate `λ = rate_parameter`. Every function is
//! closed form; sampling is the inverse-CDF transform of a single uniform draw.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Cdf, Pdf};
//! use stats_claw::distributions::ExponentialDistribution;
//!
//! let d = ExponentialDistribution { rate_parameter: 2.0, ..Default::default() };
//! // Density at origin equals the rate.
//! assert!((d.pdf(0.0) - 2.0).abs() < 1e-12, "pdf was {}", d.pdf(0.0));
//! // CDF at the mean (0.5) is 1 - e^{-1} ≈ 0.6321.
//! assert!((d.cdf(0.5) - (1.0 - (-1.0f64).exp())).abs() < 1e-12);
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::ExponentialDistribution;
use crate::rng::SplitMix64;

impl Pdf for ExponentialDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate_parameter * (-self.rate_parameter * x).exp()
        }
    }
}

impl Cdf for ExponentialDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            (-self.rate_parameter * x).exp().mul_add(-1.0, 1.0)
        }
    }
}

impl Quantile for ExponentialDistribution {
    fn quantile(&self, p: f64) -> f64 {
        -(1.0 - p).ln() / self.rate_parameter
    }
}

impl Moments for ExponentialDistribution {
    fn mean(&self) -> Option<f64> {
        Some(1.0 / self.rate_parameter)
    }
    fn variance(&self) -> Option<f64> {
        Some(1.0 / (self.rate_parameter * self.rate_parameter))
    }
}

impl Sample for ExponentialDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.quantile(rng.next_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The density at the origin equals the rate.
    #[test]
    fn density_at_origin_is_rate() {
        let d = ExponentialDistribution {
            rate_parameter: 2.0,
            ..Default::default()
        };
        assert!((d.pdf(0.0) - 2.0).abs() < 1e-12, "was {}", d.pdf(0.0));
    }

    /// The mean is the reciprocal rate.
    #[test]
    fn mean_is_reciprocal_rate() {
        let d = ExponentialDistribution {
            rate_parameter: 4.0,
            ..Default::default()
        };
        assert_eq!(d.mean(), Some(0.25));
    }
}
