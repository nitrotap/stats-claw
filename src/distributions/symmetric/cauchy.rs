//! Cauchy (Lorentz) distribution numerics, for the [`CauchyDistribution`] parameter struct.
//!
//! Equivalent to `scipy.stats.cauchy(loc=location, scale=scale)`: a symmetric,
//! heavy-tailed distribution with no finite moments. The CDF/quantile are the
//! closed-form arctangent/tangent pair, and sampling is the inverse-CDF transform
//! of a single uniform draw.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::Pdf;
//! use stats_claw::distributions::CauchyDistribution;
//! use std::f64::consts::PI;
//!
//! let d = CauchyDistribution { location: 0.0, scale: 1.0, ..Default::default() };
//! // The standard Cauchy peaks at 1/π at the location.
//! assert!((d.pdf(0.0) - 1.0 / PI).abs() < 1e-12, "peak was {}", d.pdf(0.0));
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::CauchyDistribution;
use crate::rng::SplitMix64;
use std::f64::consts::PI;

impl Pdf for CauchyDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        1.0 / (PI * self.scale * z.mul_add(z, 1.0))
    }
}

impl Cdf for CauchyDistribution {
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        0.5 + z.atan() / PI
    }
}

impl Quantile for CauchyDistribution {
    fn quantile(&self, p: f64) -> f64 {
        self.scale.mul_add((PI * (p - 0.5)).tan(), self.location)
    }
}

impl Moments for CauchyDistribution {
    /// Undefined — the Cauchy integral for the mean does not converge.
    fn mean(&self) -> Option<f64> {
        None
    }
    /// Undefined — the Cauchy second moment does not converge.
    fn variance(&self) -> Option<f64> {
        None
    }
}

impl Sample for CauchyDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.quantile(rng.next_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard Cauchy peaks at `1/π` at the location.
    #[test]
    fn peak_density_at_location() {
        let d = CauchyDistribution {
            location: 0.0,
            scale: 1.0,
            ..Default::default()
        };
        assert!(
            (d.pdf(0.0) - 1.0 / PI).abs() < 1e-12,
            "peak was {}",
            d.pdf(0.0)
        );
    }

    /// Both moments are undefined regardless of parameters.
    #[test]
    fn moments_are_none() {
        let d = CauchyDistribution {
            location: 3.0,
            scale: 2.0,
            ..Default::default()
        };
        assert!(d.mean().is_none() && d.variance().is_none());
    }
}
