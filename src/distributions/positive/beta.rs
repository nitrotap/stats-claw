//! Beta distribution numerics, for the [`BetaDistribution`].
//!
//! Equivalent to `scipy.stats.beta(alpha_parameter, beta_parameter)` on the
//! support `[0, 1]`. The density uses `ln_beta` for stability, the CDF is the
//! regularized incomplete beta `betai`, the quantile inverts it by bracketed
//! bisection, and sampling forms `X / (X + Y)` from two unit-scale gamma draws.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Cdf, Pdf};
//! use stats_claw::distributions::BetaDistribution;
//!
//! // Beta(1, 1) is the standard uniform.
//! let d = BetaDistribution { alpha_parameter: 1.0, beta_parameter: 1.0, ..Default::default() };
//! assert!((d.pdf(0.3) - 1.0).abs() < 1e-12, "pdf was {}", d.pdf(0.3));
//! assert!((d.cdf(0.4) - 0.4).abs() < 1e-12);
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample, bisection_quantile};
use super::gamma::marsaglia_tsang;
use crate::distributions::BetaDistribution;
use crate::rng::SplitMix64;
use crate::special::{betai, ln_beta};

impl Pdf for BetaDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        let (a, b) = (self.alpha_parameter, self.beta_parameter);
        let ln_density = (a - 1.0).mul_add(x.ln(), (b - 1.0) * (1.0 - x).ln()) - ln_beta(a, b);
        ln_density.exp()
    }
}

impl Cdf for BetaDistribution {
    fn cdf(&self, x: f64) -> f64 {
        betai(self.alpha_parameter, self.beta_parameter, x)
    }
}

impl Quantile for BetaDistribution {
    fn quantile(&self, p: f64) -> f64 {
        bisection_quantile(p, 0.0, 1.0, |x| self.cdf(x))
    }
}

impl Moments for BetaDistribution {
    fn mean(&self) -> Option<f64> {
        let (a, b) = (self.alpha_parameter, self.beta_parameter);
        Some(a / (a + b))
    }
    fn variance(&self) -> Option<f64> {
        let (a, b) = (self.alpha_parameter, self.beta_parameter);
        let s = a + b;
        let denom = s * s * (s + 1.0);
        Some((a * b) / denom)
    }
}

impl Sample for BetaDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        let x = marsaglia_tsang(self.alpha_parameter, rng);
        let y = marsaglia_tsang(self.beta_parameter, rng);
        x / (x + y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Beta(1, 1) is the standard uniform: density 1 everywhere on [0, 1].
    #[test]
    fn beta_one_one_is_uniform() {
        let d = BetaDistribution {
            alpha_parameter: 1.0,
            beta_parameter: 1.0,
            ..Default::default()
        };
        assert!((d.pdf(0.3) - 1.0).abs() < 1e-12, "was {}", d.pdf(0.3));
        assert!((d.cdf(0.4) - 0.4).abs() < 1e-12);
    }

    /// The mean is `a / (a + b)`.
    #[test]
    fn mean_is_a_over_a_plus_b() {
        let d = BetaDistribution {
            alpha_parameter: 2.0,
            beta_parameter: 5.0,
            ..Default::default()
        };
        assert!(matches!(d.mean(), Some(m) if (m - 2.0 / 7.0).abs() < 1e-12));
    }
}
