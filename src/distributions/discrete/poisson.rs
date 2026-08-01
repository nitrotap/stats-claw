//! Poisson distribution numerics, for the
//! [`PoissonDistribution`].
//!
//! Equivalent to `scipy.stats.poisson(rate_parameter)`. The PMF uses `ln_gamma`
//! for the factorial, the CDF is the regularized upper incomplete gamma
//! `gamma_q(k+1, λ)`, the quantile is the smallest `k` with `cdf(k) ≥ p`, and
//! sampling uses Knuth's multiplicative method.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Moments, Pmf};
//! use stats_claw::distributions::PoissonDistribution;
//!
//! let d = PoissonDistribution { rate_parameter: 3.0, ..Default::default() };
//! // Mean and variance both equal the rate.
//! assert_eq!(d.mean(), Some(3.0));
//! assert_eq!(d.variance(), Some(3.0));
//! // PMF is positive at a support point.
//! assert!(d.pmf(3) > 0.0, "pmf at mode was {}", d.pmf(3));
//! ```

use super::super::{Cdf, Moments, Pmf, Quantile, Sample, count_to_f64};
use crate::distributions::PoissonDistribution;
use crate::rng::SplitMix64;
use crate::special::{gamma_q, ln_gamma};

/// Hard cap on the support scan for the quantile: far beyond the tail of any
/// realistic Poisson rate, so the loop terminates without a float comparison.
const MAX_SUPPORT: i64 = 1_000_000;

impl Pmf for PoissonDistribution {
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        let lambda = self.rate_parameter;
        let kf = count_to_f64(k);
        // log pmf = k·ln λ − λ − ln Γ(k+1)
        let log_pmf = kf.mul_add(lambda.ln(), -lambda) - ln_gamma(kf + 1.0);
        log_pmf.exp()
    }
}

impl Cdf for PoissonDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let k = x.floor();
        // P(X ≤ k) = Q(k+1, λ) (regularized upper incomplete gamma).
        gamma_q(k + 1.0, self.rate_parameter)
    }
}

impl Quantile for PoissonDistribution {
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        // Scan the support in order, returning the first `k` whose cumulative
        // mass reaches `p`. The hard cap guards against a runaway loop for
        // pathological inputs; the Poisson tail is exhausted long before it for
        // any realistic rate.
        let mut cumulative = 0.0;
        let mut last = 0i64;
        for k in 0..=MAX_SUPPORT {
            last = k;
            cumulative += self.pmf(k);
            if cumulative >= p - 1e-12 {
                return count_to_f64(k);
            }
        }
        count_to_f64(last)
    }
}

impl Moments for PoissonDistribution {
    fn mean(&self) -> Option<f64> {
        Some(self.rate_parameter)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.rate_parameter)
    }
}

impl Sample for PoissonDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        let threshold = (-self.rate_parameter).exp();
        let mut product = rng.next_f64();
        let mut count = 0.0;
        // Knuth's method: multiply uniforms until the product drops below e^{−λ};
        // a `loop` keeps the float comparison out of a `while` head.
        loop {
            if product <= threshold {
                return count;
            }
            product *= rng.next_f64();
            count += 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean and variance both equal the rate.
    #[test]
    fn mean_and_variance_equal_rate() {
        let d = PoissonDistribution {
            rate_parameter: 4.0,
            ..Default::default()
        };
        assert_eq!(d.mean(), Some(4.0));
        assert_eq!(d.variance(), Some(4.0));
    }

    /// The PMF sums to ~1 over a wide support window.
    #[test]
    fn pmf_sums_to_one() {
        let d = PoissonDistribution {
            rate_parameter: 4.0,
            ..Default::default()
        };
        let total: f64 = (0..60).map(|k| d.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-12, "sum was {total}");
    }
}
