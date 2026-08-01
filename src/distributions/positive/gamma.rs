//! Gamma distribution numerics, for the [`GammaDistribution`].
//!
//! Equivalent to `scipy.stats.gamma(shape_parameter, scale = scale_parameter)`.
//! The density is closed form; the CDF is the regularized lower incomplete gamma
//! `gamma_p`; the quantile inverts the CDF by bracketed bisection; and sampling
//! uses the Marsaglia–Tsang method (with the shape<1 boosting trick).
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Cdf, Moments};
//! use stats_claw::distributions::GammaDistribution;
//!
//! let d = GammaDistribution { shape_parameter: 2.0, scale_parameter: 3.0, ..Default::default() };
//! // Mean is shape * scale = 6.
//! assert_eq!(d.mean(), Some(6.0));
//! // With shape=1 the Gamma is exponential; cdf(1) with rate 0.5 is 1 - e^{-0.5}.
//! let exp = GammaDistribution { shape_parameter: 1.0, scale_parameter: 2.0, ..Default::default() };
//! assert!((exp.cdf(2.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-10);
//! ```

use super::super::{Cdf, Moments, Pdf, Quantile, Sample, bisection_quantile};
use crate::distributions::GammaDistribution;
use crate::rng::SplitMix64;
use crate::special::{gamma_p, ln_gamma};

impl Pdf for GammaDistribution {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let (k, theta) = (self.shape_parameter, self.scale_parameter);
        // log-space for stability: (k-1)·ln x − x/θ − k·ln θ − ln Γ(k)
        let log_norm = k.mul_add(theta.ln(), ln_gamma(k));
        let ln_density = (k - 1.0).mul_add(x.ln(), -(x / theta) - log_norm);
        ln_density.exp()
    }
}

impl Cdf for GammaDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gamma_p(self.shape_parameter, x / self.scale_parameter)
        }
    }
}

impl Quantile for GammaDistribution {
    fn quantile(&self, p: f64) -> f64 {
        let mean = self.shape_parameter * self.scale_parameter;
        let sd = self.scale_parameter * self.shape_parameter.sqrt();
        let hi = 20.0f64.mul_add(sd, mean).max(self.scale_parameter);
        bisection_quantile(p, 0.0, hi, |x| self.cdf(x))
    }
}

impl Moments for GammaDistribution {
    fn mean(&self) -> Option<f64> {
        Some(self.shape_parameter * self.scale_parameter)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.shape_parameter * self.scale_parameter * self.scale_parameter)
    }
}

impl Sample for GammaDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.scale_parameter * marsaglia_tsang(self.shape_parameter, rng)
    }
}

/// Draws a standard Gamma(`shape`, 1) variate via Marsaglia–Tsang (2000).
///
/// For `shape ≥ 1` it uses the squeeze acceptance directly; for `shape < 1` it
/// draws at `shape + 1` and rescales by `U^(1/shape)`, which is exact. Shared
/// with the Beta sampler (`X/(X+Y)` of two unit-scale gammas).
///
/// # Arguments
///
/// * `shape` — the gamma shape parameter (`> 0`).
/// * `rng` — the deterministic generator to draw from.
///
/// # Returns
///
/// One Gamma(`shape`, 1) variate.
pub(super) fn marsaglia_tsang(shape: f64, rng: &mut SplitMix64) -> f64 {
    if shape < 1.0 {
        let boosted = marsaglia_tsang(shape + 1.0, rng);
        return boosted * rng.next_f64().powf(1.0 / shape);
    }
    let depth = shape - 1.0 / 3.0;
    let coeff = 1.0 / (9.0 * depth).sqrt();
    loop {
        let normal = rng.standard_normal();
        let base = coeff.mul_add(normal, 1.0);
        if base <= 0.0 {
            continue;
        }
        let cubed = base * base * base;
        let uniform = rng.next_f64();
        let normal_sq = normal * normal;
        // Marsaglia–Tsang acceptance in log space.
        let accept = 0.5f64.mul_add(normal_sq, depth.mul_add(-cubed, depth) + depth * cubed.ln());
        if uniform.ln() < accept {
            return depth * cubed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With shape 1 the Gamma reduces to an exponential with rate `1/scale`.
    #[test]
    fn shape_one_cdf_is_exponential() {
        let d = GammaDistribution {
            shape_parameter: 1.0,
            scale_parameter: 2.0,
            ..Default::default()
        };
        assert!((d.cdf(2.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-10);
    }

    /// The mean is `shape · scale`.
    #[test]
    fn mean_is_shape_times_scale() {
        let d = GammaDistribution {
            shape_parameter: 2.5,
            scale_parameter: 1.5,
            ..Default::default()
        };
        assert_eq!(d.mean(), Some(3.75));
    }
}
