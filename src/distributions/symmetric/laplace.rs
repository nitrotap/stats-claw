//! Laplace (double-exponential) distribution numerics, for the
//! [`LaplaceDistribution`].
//!
//! Equivalent to `scipy.stats.laplace(loc=location, scale=scale)`: a symmetric
//! exponential decay about `location` with rate `1/scale`. The CDF and quantile
//! are piecewise closed forms split at the median, and sampling uses the
//! inverse-CDF transform of a single uniform draw.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Pdf, Quantile};
//! use stats_claw::distributions::LaplaceDistribution;
//!
//! let d = LaplaceDistribution { location: 0.0, scale: 1.0, ..Default::default() };
//! // Peak density of Laplace(0, 1) is 1/(2b) = 0.5.
//! assert!((d.pdf(0.0) - 0.5).abs() < 1e-12, "peak was {}", d.pdf(0.0));
//! // The median equals the location.
//! assert!((d.quantile(0.5) - 0.0).abs() < 1e-12);
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::LaplaceDistribution;
use crate::rng::SplitMix64;

impl Pdf for LaplaceDistribution {
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.location).abs() / self.scale).exp() / (2.0 * self.scale)
    }
}

impl Cdf for LaplaceDistribution {
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            (-z).exp().mul_add(-0.5, 1.0)
        }
    }
}

impl Quantile for LaplaceDistribution {
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.5 {
            self.scale.mul_add((2.0 * p).ln(), self.location)
        } else {
            let upper_tail = 2.0f64.mul_add(-p, 2.0);
            self.scale.mul_add(-upper_tail.ln(), self.location)
        }
    }
}

impl Moments for LaplaceDistribution {
    fn mean(&self) -> Option<f64> {
        Some(self.location)
    }
    fn variance(&self) -> Option<f64> {
        Some(2.0 * self.scale * self.scale)
    }
}

impl Sample for LaplaceDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.quantile(rng.next_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The density peaks at `1/(2b)` at the location.
    #[test]
    fn peak_density_at_location() {
        let d = LaplaceDistribution {
            location: 0.0,
            scale: 1.0,
            ..Default::default()
        };
        assert!((d.pdf(0.0) - 0.5).abs() < 1e-12, "peak was {}", d.pdf(0.0));
    }

    /// The median equals the location.
    #[test]
    fn median_is_location() {
        let d = LaplaceDistribution {
            location: 3.0,
            scale: 2.0,
            ..Default::default()
        };
        assert!((d.quantile(0.5) - 3.0).abs() < 1e-12);
    }
}
