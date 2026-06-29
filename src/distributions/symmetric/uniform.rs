//! Continuous uniform distribution numerics, for the [`UniformDistribution`] parameter struct.
//!
//! Equivalent to `scipy.stats.uniform(loc=lower_bound, scale=upper_bound −
//! lower_bound)`: constant density on `[lower_bound, upper_bound]`, zero outside.
//! Every function is closed form; sampling is an affine map of a single uniform
//! draw.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Cdf, Pdf};
//! use stats_claw::distributions::UniformDistribution;
//!
//! let d = UniformDistribution { lower_bound: 0.0, upper_bound: 4.0, ..Default::default() };
//! // Density on [0, 4] is 1/4 = 0.25.
//! assert!((d.pdf(2.0) - 0.25).abs() < 1e-12, "pdf was {}", d.pdf(2.0));
//! // CDF at the midpoint is 0.5.
//! assert!((d.cdf(2.0) - 0.5).abs() < 1e-12);
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::UniformDistribution;
use crate::rng::SplitMix64;

impl UniformDistribution {
    /// Width of the support, `upper_bound − lower_bound`.
    fn width(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }
}

impl Pdf for UniformDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.lower_bound || x > self.upper_bound {
            0.0
        } else {
            1.0 / self.width()
        }
    }
}

impl Cdf for UniformDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.lower_bound {
            0.0
        } else if x >= self.upper_bound {
            1.0
        } else {
            (x - self.lower_bound) / self.width()
        }
    }
}

impl Quantile for UniformDistribution {
    fn quantile(&self, p: f64) -> f64 {
        self.width().mul_add(p, self.lower_bound)
    }
}

impl Moments for UniformDistribution {
    fn mean(&self) -> Option<f64> {
        Some(0.5 * (self.lower_bound + self.upper_bound))
    }
    fn variance(&self) -> Option<f64> {
        let w = self.width();
        Some(w * w / 12.0)
    }
}

impl Sample for UniformDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.width().mul_add(rng.next_f64(), self.lower_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Density is the reciprocal width inside the support and zero outside.
    #[test]
    fn density_is_reciprocal_width_inside() {
        let d = UniformDistribution {
            lower_bound: 0.0,
            upper_bound: 4.0,
            ..Default::default()
        };
        assert!((d.pdf(2.0) - 0.25).abs() < 1e-12);
        assert!(d.pdf(-1.0).abs() < 1e-12, "outside support must be 0");
    }

    /// The midpoint is the mean; variance is `width²/12`.
    #[test]
    fn moments_are_closed_form() {
        let d = UniformDistribution {
            lower_bound: -2.0,
            upper_bound: 3.0,
            ..Default::default()
        };
        assert_eq!(d.mean(), Some(0.5));
        assert!(matches!(d.variance(), Some(v) if (v - 25.0 / 12.0).abs() < 1e-12));
    }
}
