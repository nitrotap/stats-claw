//! Student's t-distribution numerics, for the [`TDistribution`] parameter struct.
//!
//! Equivalent to `scipy.stats.t(degrees_of_freedom)`. The density uses `ln_gamma`
//! for stability, the CDF is expressed through the regularized incomplete beta
//! `betai` (mirrored for negative `t`), the quantile inverts the CDF by
//! bisection, and sampling forms `Z / sqrt(χ²_ν / ν)`.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Cdf, Pdf};
//! use stats_claw::distributions::TDistribution;
//!
//! let d = TDistribution { degrees_of_freedom: 7, ..Default::default() };
//! // The t distribution is symmetric: density at +t equals density at -t.
//! assert!((d.pdf(1.3) - d.pdf(-1.3)).abs() < 1e-12);
//! // CDF at 0 is exactly 0.5.
//! assert!((d.cdf(0.0) - 0.5).abs() < 1e-12);
//! ```

use super::super::{bisection_quantile, count_to_f64, Cdf, LogCdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::TDistribution;
use crate::rng::SplitMix64;
use crate::special::{betai, ln_betai_lower, ln_gamma};
use std::f64::consts::{LN_2, PI};

impl TDistribution {
    /// Degrees of freedom as a float (`ν`).
    fn nu(&self) -> f64 {
        count_to_f64(self.degrees_of_freedom)
    }
}

impl Pdf for TDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let nu = self.nu();
        let gamma_ratio = ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu);
        let log_norm = 0.5f64.mul_add(-(nu * PI).ln(), gamma_ratio);
        let ln_kernel = -0.5 * (nu + 1.0) * x.mul_add(x / nu, 1.0).ln();
        (log_norm + ln_kernel).exp()
    }
}

impl Cdf for TDistribution {
    fn cdf(&self, x: f64) -> f64 {
        let nu = self.nu();
        // Iₓ with x = ν/(ν+t²) gives the two-tailed mass; halve and mirror.
        let ib = betai(0.5 * nu, 0.5, nu / x.mul_add(x, nu));
        if x >= 0.0 {
            0.5f64.mul_add(-ib, 1.0)
        } else {
            0.5 * ib
        }
    }
}

impl LogCdf for TDistribution {
    fn logsf(&self, x: f64) -> f64 {
        // Symmetric about 0: P(T > x) = P(T < −x), so logsf(x) = logcdf(−x).
        self.logcdf(-x)
    }
    fn logcdf(&self, x: f64) -> f64 {
        let nu = self.nu();
        // ln of the two-tailed beta mass: ln I_w(ν/2, ½) with w = ν/(ν+x²).
        let ln_ib = ln_betai_lower(0.5 * nu, 0.5, nu / x.mul_add(x, nu));
        if x <= 0.0 {
            // Left tail: cdf = ½·I_w; ln cdf = ln ½ + ln I_w (finite as x → −∞).
            ln_ib - LN_2
        } else {
            // Right of center: cdf = 1 − ½·I_w ≈ 1; ln_1p keeps it accurate.
            (-0.5 * ln_ib.exp()).ln_1p()
        }
    }
}

impl Quantile for TDistribution {
    fn quantile(&self, p: f64) -> f64 {
        // The t-distribution is symmetric about 0, so anchor the median exactly
        // and mirror the lower half onto the upper. Solving only on `[0, hi]`
        // keeps the bracket tight and the median exact (bisection on a symmetric
        // `[-hi, hi]` would leave an O(1e-8) residual at p = 0.5).
        if (p - 0.5).abs() < f64::EPSILON {
            return 0.0;
        }
        if p < 0.5 {
            return -self.quantile(1.0 - p);
        }
        bisection_quantile(p, 0.0, 1.0e6, |x| self.cdf(x))
    }
}

impl Moments for TDistribution {
    fn mean(&self) -> Option<f64> {
        if self.degrees_of_freedom > 1 {
            Some(0.0)
        } else {
            None
        }
    }
    fn variance(&self) -> Option<f64> {
        let nu = self.nu();
        if self.degrees_of_freedom > 2 {
            Some(nu / (nu - 2.0))
        } else {
            None
        }
    }
}

impl Sample for TDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        let z = rng.standard_normal();
        let mut chi2 = 0.0;
        for _ in 0..self.degrees_of_freedom.max(1) {
            let g = rng.standard_normal();
            chi2 = g.mul_add(g, chi2);
        }
        z / (chi2 / self.nu()).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The density is symmetric about zero.
    #[test]
    fn density_is_symmetric() {
        let d = TDistribution {
            degrees_of_freedom: 7,
            ..Default::default()
        };
        assert!((d.pdf(1.3) - d.pdf(-1.3)).abs() < 1e-12);
    }

    /// The CDF at zero is one half.
    #[test]
    fn cdf_at_zero_is_half() {
        let d = TDistribution {
            degrees_of_freedom: 4,
            ..Default::default()
        };
        assert!((d.cdf(0.0) - 0.5).abs() < 1e-12);
    }

    /// `logsf` agrees with the log of the linear survival in the body and matches
    /// scipy `t.logsf` deep in the tail (df = 5, t = 50, where 1 − cdf underflows
    /// past linear precision).
    #[test]
    fn logsf_matches_scipy_and_stays_finite() {
        let d = TDistribution {
            degrees_of_freedom: 5,
            ..Default::default()
        };
        let body = d.logsf(1.0);
        assert!(
            ((body - (1.0 - d.cdf(1.0)).ln()) / body.abs()).abs() < 1e-9,
            "logsf body was {body}"
        );
        let tail = d.logsf(50.0);
        let want = -17.314_140_361_404_83; // scipy.stats.t.logsf(50, 5)
        assert!(tail.is_finite(), "logsf(50) was {tail}");
        assert!(
            ((tail - want) / want).abs() < 1e-9,
            "logsf(50) = {tail}, want {want}"
        );
    }
}
