//! Binomial distribution numerics, for the [`BinomialDistribution`] parameter struct.
//!
//! Equivalent to `scipy.stats.binom(number_of_trials, success_probability)`. The
//! PMF uses `ln_gamma` for the binomial coefficient, the CDF is expressed through
//! the regularized incomplete beta `betai`, the quantile is the smallest `k` with
//! `cdf(k) ≥ p`, and sampling counts `n` Bernoulli trials.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::{Moments, Pmf};
//! use stats_claw::distributions::BinomialDistribution;
//!
//! let d = BinomialDistribution { number_of_trials: 10, success_probability: 0.5, ..Default::default() };
//! // Mean is n*p = 5; PMF at the mode (5) is the largest.
//! assert_eq!(d.mean(), Some(5.0));
//! assert!(d.pmf(5) > d.pmf(3), "mode pmf should exceed flank");
//! ```

use super::super::{Cdf, Moments, Pmf, Quantile, Sample, count_to_f64};
use crate::distributions::BinomialDistribution;
use crate::rng::SplitMix64;
use crate::special::{betai, ln_gamma};

impl BinomialDistribution {
    /// Number of trials as a float (`n`).
    fn n(&self) -> f64 {
        count_to_f64(self.number_of_trials)
    }
}

impl Pmf for BinomialDistribution {
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 || k > self.number_of_trials {
            return 0.0;
        }
        let (n, p) = (self.n(), self.success_probability);
        let kf = count_to_f64(k);
        // log C(n,k) = ln Γ(n+1) − ln Γ(k+1) − ln Γ(n−k+1)
        let log_choose = ln_gamma(n + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(n - kf + 1.0);
        let log_pmf = (n - kf).mul_add((1.0 - p).ln(), kf.mul_add(p.ln(), log_choose));
        log_pmf.exp()
    }
}

impl Cdf for BinomialDistribution {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x >= self.n() {
            return 1.0;
        }
        let k = x.floor();
        let p = self.success_probability;
        // P(X ≤ k) = I_{1−p}(n−k, k+1)
        betai(self.n() - k, k + 1.0, 1.0 - p)
    }
}

impl Quantile for BinomialDistribution {
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return self.n();
        }
        let mut cumulative = 0.0;
        for k in 0..=self.number_of_trials {
            cumulative += self.pmf(k);
            if cumulative >= p - 1e-12 {
                return count_to_f64(k);
            }
        }
        self.n()
    }
}

impl Moments for BinomialDistribution {
    fn mean(&self) -> Option<f64> {
        Some(self.n() * self.success_probability)
    }
    fn variance(&self) -> Option<f64> {
        let p = self.success_probability;
        Some(self.n() * p * (1.0 - p))
    }
}

impl Sample for BinomialDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        let mut successes = 0.0;
        for _ in 0..self.number_of_trials.max(0) {
            if rng.next_f64() < self.success_probability {
                successes += 1.0;
            }
        }
        successes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mean is `n·p` and the variance `n·p·(1−p)`.
    #[test]
    fn moments_are_np_and_npq() {
        let d = BinomialDistribution {
            number_of_trials: 20,
            success_probability: 0.5,
            ..Default::default()
        };
        assert_eq!(d.mean(), Some(10.0));
        assert_eq!(d.variance(), Some(5.0));
    }

    /// The PMF sums to one over the full support.
    #[test]
    fn pmf_sums_to_one() {
        let d = BinomialDistribution {
            number_of_trials: 12,
            success_probability: 0.3,
            ..Default::default()
        };
        let total: f64 = (0..=12).map(|k| d.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-12, "sum was {total}");
    }
}
