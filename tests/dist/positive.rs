//! Equivalence tests for the positive-support distributions (exponential,
//! gamma, weibull, lognormal, beta) against their scipy golden fixtures.

use super::{check_continuous_grid, check_moments, check_round_trip, check_sampling_ks};
use stats_claw::distributions::{
    BetaDistribution, ExponentialDistribution, GammaDistribution, LogNormalDistribution,
    WeibullDistribution,
};

const PS: &[f64] = &[1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1e-6];

fn exponential() -> ExponentialDistribution {
    ExponentialDistribution {
        rate_parameter: 0.75,
        ..Default::default()
    }
}

#[test]
fn exponential_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_exponential", &exponential())
}

#[test]
fn exponential_cdf_quantile_round_trips() {
    check_round_trip(&exponential(), PS);
}

#[test]
fn exponential_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_exponential", &exponential())
}

#[test]
fn exponential_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&exponential(), 11);
}

fn weibull() -> WeibullDistribution {
    WeibullDistribution {
        shape_parameter: 1.8,
        scale_parameter: 2.5,
        ..Default::default()
    }
}

#[test]
fn weibull_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_weibull", &weibull())
}

#[test]
fn weibull_cdf_quantile_round_trips() {
    check_round_trip(&weibull(), PS);
}

#[test]
fn weibull_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_weibull", &weibull())
}

#[test]
fn weibull_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&weibull(), 13);
}

fn lognormal() -> LogNormalDistribution {
    LogNormalDistribution {
        mean_log_value: 0.3,
        std_log_value: 0.6,
        ..Default::default()
    }
}

#[test]
fn lognormal_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_lognormal", &lognormal())
}

#[test]
fn lognormal_cdf_quantile_round_trips() {
    check_round_trip(&lognormal(), PS);
}

#[test]
fn lognormal_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_lognormal", &lognormal())
}

#[test]
fn lognormal_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&lognormal(), 17);
}

fn gamma() -> GammaDistribution {
    GammaDistribution {
        shape_parameter: 2.5,
        scale_parameter: 1.5,
        ..Default::default()
    }
}

#[test]
fn gamma_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_gamma", &gamma())
}

#[test]
fn gamma_cdf_quantile_round_trips() {
    check_round_trip(&gamma(), PS);
}

#[test]
fn gamma_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_gamma", &gamma())
}

#[test]
fn gamma_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&gamma(), 19);
}

fn beta() -> BetaDistribution {
    BetaDistribution {
        alpha_parameter: 2.0,
        beta_parameter: 5.0,
        ..Default::default()
    }
}

#[test]
fn beta_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_beta", &beta())
}

#[test]
fn beta_cdf_quantile_round_trips() {
    check_round_trip(&beta(), PS);
}

#[test]
fn beta_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_beta", &beta())
}

#[test]
fn beta_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&beta(), 23);
}
