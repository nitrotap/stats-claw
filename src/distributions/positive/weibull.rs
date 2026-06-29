//! Weibull distribution numerics, for the [`WeibullDistribution`] parameter struct.
//!
//! Equivalent to `scipy.stats.weibull_min(c = shape_parameter, scale =
//! scale_parameter)`. The density, CDF, and quantile are closed forms; the
//! moments use the gamma function via `exp(ln_gamma(..))`, and sampling is the
//! inverse-CDF transform.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::Cdf;
//! use stats_claw::distributions::WeibullDistribution;
//!
//! // With shape=1 the Weibull is Exponential(1/scale).
//! let d = WeibullDistribution { shape_parameter: 1.0, scale_parameter: 2.0, ..Default::default() };
//! // cdf(2) = 1 - exp(-1) ≈ 0.6321.
//! assert!((d.cdf(2.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-12);
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::WeibullDistribution;
use crate::rng::SplitMix64;
use crate::special::ln_gamma;

impl WeibullDistribution {
    /// `(x / scale)` raised to the shape, the recurring `(x/λ)^k` term.
    fn scaled_pow(&self, x: f64) -> f64 {
        (x / self.scale_parameter).powf(self.shape_parameter)
    }
}

impl Pdf for WeibullDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let (k, lambda) = (self.shape_parameter, self.scale_parameter);
        let z = x / lambda;
        (k / lambda) * z.powf(k - 1.0) * (-self.scaled_pow(x)).exp()
    }
}

impl Cdf for WeibullDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            (-self.scaled_pow(x)).exp().mul_add(-1.0, 1.0)
        }
    }
}

impl Quantile for WeibullDistribution {
    fn quantile(&self, p: f64) -> f64 {
        self.scale_parameter * (-(1.0 - p).ln()).powf(1.0 / self.shape_parameter)
    }
}

impl Moments for WeibullDistribution {
    fn mean(&self) -> Option<f64> {
        Some(self.scale_parameter * gamma(1.0 + 1.0 / self.shape_parameter))
    }
    fn variance(&self) -> Option<f64> {
        let g1 = gamma(1.0 + 1.0 / self.shape_parameter);
        let g2 = gamma(1.0 + 2.0 / self.shape_parameter);
        Some(self.scale_parameter * self.scale_parameter * g2.mul_add(1.0, -(g1 * g1)))
    }
}

impl Sample for WeibullDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.quantile(rng.next_f64())
    }
}

/// Gamma function `Γ(x)` for `x > 0`, via `exp(ln_gamma(x))`.
fn gamma(x: f64) -> f64 {
    ln_gamma(x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With shape 1 the Weibull reduces to an exponential with rate `1/scale`.
    #[test]
    fn shape_one_is_exponential() {
        let d = WeibullDistribution {
            shape_parameter: 1.0,
            scale_parameter: 2.0,
            ..Default::default()
        };
        // exponential cdf at x=2 with rate 0.5 is 1 - e^{-1}
        assert!((d.cdf(2.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-12);
    }

    /// The mean is `scale · Γ(1 + 1/shape)`.
    #[test]
    fn mean_matches_gamma_form() {
        let d = WeibullDistribution {
            shape_parameter: 2.0,
            scale_parameter: 1.0,
            ..Default::default()
        };
        // Γ(1.5) = √π/2 ≈ 0.886227
        assert!(matches!(d.mean(), Some(m) if (m - 0.886_226_925).abs() < 1e-6));
    }
}
