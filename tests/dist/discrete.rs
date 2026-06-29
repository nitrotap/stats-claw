//! Equivalence tests for the discrete distributions (binomial, poisson) against
//! their scipy golden fixtures: pmf/cdf/ppf over the integer support, moment
//! agreement, and a seeded chi-square goodness-of-fit of the sampler.

use super::{check_discrete_grid, check_moments, check_sampling_chi2};
use stats_claw::distributions::{BinomialDistribution, PoissonDistribution};

fn binomial() -> BinomialDistribution {
    BinomialDistribution {
        number_of_trials: 20,
        success_probability: 0.35,
        ..Default::default()
    }
}

const BINOMIAL_SUPPORT: [i64; 21] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
];

#[test]
fn binomial_pmf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_discrete_grid("dist_binomial", &binomial(), &BINOMIAL_SUPPORT)
}

#[test]
fn binomial_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_binomial", &binomial())
}

#[test]
fn binomial_sampling_reproducible_and_fits_pmf() {
    check_sampling_chi2(&binomial(), 41, &BINOMIAL_SUPPORT);
}

fn poisson() -> PoissonDistribution {
    PoissonDistribution {
        rate_parameter: 4.0,
        ..Default::default()
    }
}

const POISSON_SUPPORT: [i64; 25] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

#[test]
fn poisson_pmf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_discrete_grid("dist_poisson", &poisson(), &POISSON_SUPPORT)
}

#[test]
fn poisson_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_poisson", &poisson())
}

#[test]
fn poisson_sampling_reproducible_and_fits_pmf() {
    check_sampling_chi2(&poisson(), 43, &POISSON_SUPPORT);
}
