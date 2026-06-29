//! Equivalence tests for the sampling-distribution family (chi-squared,
//! Student's t, F) against their scipy golden fixtures, including the
//! undefined-moment cases (T variance for df ≤ 2, F moments for low denominator
//! df).

use super::{check_continuous_grid, check_moments, check_round_trip, check_sampling_ks};
use stats_claw::distributions::Moments;
use stats_claw::distributions::{ChiSquaredDistribution, FDistribution, TDistribution};

const PS: &[f64] = &[1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1e-6];

fn chi_squared() -> ChiSquaredDistribution {
    ChiSquaredDistribution {
        degrees_of_freedom: 5,
        ..Default::default()
    }
}

#[test]
fn chi_squared_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_chi_squared", &chi_squared())
}

#[test]
fn chi_squared_cdf_quantile_round_trips() {
    check_round_trip(&chi_squared(), PS);
}

#[test]
fn chi_squared_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_chi_squared", &chi_squared())
}

#[test]
fn chi_squared_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&chi_squared(), 29);
}

fn students_t() -> TDistribution {
    TDistribution {
        degrees_of_freedom: 7,
        ..Default::default()
    }
}

#[test]
fn students_t_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_students_t", &students_t())
}

#[test]
fn students_t_cdf_quantile_round_trips() {
    check_round_trip(&students_t(), PS);
}

#[test]
fn students_t_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_students_t", &students_t())
}

#[test]
fn students_t_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&students_t(), 31);
}

/// T-distribution variance is undefined for df ≤ 2 and the mean for df ≤ 1.
#[test]
fn students_t_low_df_moments_are_undefined() {
    let df1 = TDistribution {
        degrees_of_freedom: 1,
        ..Default::default()
    };
    assert_eq!(df1.mean(), None, "mean undefined for df=1");
    assert_eq!(df1.variance(), None, "variance undefined for df=1");
    let df2 = TDistribution {
        degrees_of_freedom: 2,
        ..Default::default()
    };
    assert_eq!(df2.mean(), Some(0.0), "mean defined for df=2");
    assert_eq!(df2.variance(), None, "variance undefined for df=2");
}

fn f_dist() -> FDistribution {
    FDistribution {
        numerator_df: 6,
        denominator_df: 12,
        ..Default::default()
    }
}

#[test]
fn f_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_f", &f_dist())
}

#[test]
fn f_cdf_quantile_round_trips() {
    check_round_trip(&f_dist(), PS);
}

#[test]
fn f_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_f", &f_dist())
}

#[test]
fn f_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&f_dist(), 37);
}

/// F-distribution mean is undefined for denominator df ≤ 2 and variance for ≤ 4.
#[test]
fn f_low_denominator_df_moments_are_undefined() {
    let d2_2 = FDistribution {
        numerator_df: 4,
        denominator_df: 2,
        ..Default::default()
    };
    assert_eq!(d2_2.mean(), None, "mean undefined for d2=2");
    assert_eq!(d2_2.variance(), None, "variance undefined for d2=2");
    let d2_4 = FDistribution {
        numerator_df: 4,
        denominator_df: 4,
        ..Default::default()
    };
    assert!(d2_4.mean().is_some(), "mean defined for d2=4");
    assert_eq!(d2_4.variance(), None, "variance undefined for d2=4");
}
