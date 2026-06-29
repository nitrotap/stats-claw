//! Equivalence tests for the symmetric distributions (normal, laplace, cauchy,
//! uniform) against their scipy golden fixtures.

use super::{check_continuous_grid, check_moments, check_round_trip, check_sampling_ks};
use stats_claw::distributions::{
    CauchyDistribution, LaplaceDistribution, NormalDistribution, UniformDistribution,
};
use stats_claw::rng::SplitMix64;

const PS: &[f64] = &[1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1e-6];

fn normal() -> NormalDistribution {
    NormalDistribution {
        mean: 1.5,
        standard_deviation: 2.0,
        ..Default::default()
    }
}

#[test]
fn normal_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_normal", &normal())
}

#[test]
fn normal_cdf_quantile_round_trips() {
    check_round_trip(&normal(), PS);
}

#[test]
fn normal_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_normal", &normal())
}

#[test]
fn normal_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&normal(), 123);
}

/// The native-SIMD batch ziggurat (`sample_batch`) is reproducible on the active
/// CPU path for a fixed seed, and a large fill is distributed as N(mean, σ²): its
/// empirical mean/variance converge and it fits the normal CDF under a 1% KS check.
/// This proves the equidistribution / ziggurat-correctness of whichever
/// runtime-dispatched kernel (NEON / AVX2 / scalar) this machine actually runs.
#[test]
fn normal_sample_batch_reproducible_and_fits_cdf() {
    let dist = normal();
    let n = 50_000usize;
    let fill = |seed: u64| {
        let mut rng = SplitMix64::new(seed);
        let mut out = vec![0.0; n];
        dist.sample_batch(&mut rng, &mut out);
        out
    };
    let mut xs = fill(2024);
    assert_eq!(
        xs,
        fill(2024),
        "sample_batch not reproducible on this CPU path"
    );

    let count = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let mean = xs.iter().sum::<f64>() / count;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / count;
    assert!(
        (mean - dist.mean).abs() < 0.05,
        "mean {mean} not near {}",
        dist.mean
    );
    let want_var = dist.standard_deviation * dist.standard_deviation;
    assert!(
        (var - want_var).abs() < 0.2,
        "variance {var} not near {want_var}"
    );

    xs.sort_by(f64::total_cmp);
    let n_f = count;
    let inv_scale = 1.0 / (dist.standard_deviation * std::f64::consts::SQRT_2);
    let cdf = |x: f64| 0.5 * (1.0 + erf((x - dist.mean) * inv_scale));
    let mut ks = 0.0_f64;
    for (i, &x) in xs.iter().enumerate() {
        let f = cdf(x);
        let i_f = f64::from(u32::try_from(i).unwrap_or(u32::MAX));
        ks = ks.max((i_f + 1.0) / n_f - f).max(f - i_f / n_f);
    }
    let crit = 1.63 / n_f.sqrt();
    assert!(ks < crit, "batch-sample KS={ks} exceeds 1% critical {crit}");
}

/// The error function `erf`, re-derived locally for the KS reference so the
/// integration test stays independent of the crate's internal `special` module.
fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7) — ample for a KS sanity bound.
    let t = 1.0 / x.abs().mul_add(0.327_591_1, 1.0);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

fn laplace() -> LaplaceDistribution {
    LaplaceDistribution {
        location: -0.5,
        scale: 1.5,
        ..Default::default()
    }
}

#[test]
fn laplace_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_laplace", &laplace())
}

#[test]
fn laplace_cdf_quantile_round_trips() {
    check_round_trip(&laplace(), PS);
}

#[test]
fn laplace_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_laplace", &laplace())
}

#[test]
fn laplace_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&laplace(), 7);
}

fn cauchy() -> CauchyDistribution {
    CauchyDistribution {
        location: 0.0,
        scale: 1.0,
        ..Default::default()
    }
}

#[test]
fn cauchy_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_cauchy", &cauchy())
}

#[test]
fn cauchy_cdf_quantile_round_trips() {
    check_round_trip(&cauchy(), PS);
}

/// Cauchy has no finite mean or variance; both report `None`, matching scipy NaN.
#[test]
fn cauchy_moments_are_undefined() -> Result<(), super::HarnessError> {
    check_moments("dist_cauchy", &cauchy())
}

/// Heavy-tail: moment convergence is skipped; only the KS fit is required.
#[test]
fn cauchy_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&cauchy(), 99);
}

fn uniform() -> UniformDistribution {
    UniformDistribution {
        lower_bound: -2.0,
        upper_bound: 3.0,
        ..Default::default()
    }
}

#[test]
fn uniform_pdf_cdf_ppf_match_scipy() -> Result<(), super::HarnessError> {
    check_continuous_grid("dist_uniform", &uniform())
}

#[test]
fn uniform_cdf_quantile_round_trips() {
    check_round_trip(&uniform(), PS);
}

#[test]
fn uniform_moments_match() -> Result<(), super::HarnessError> {
    check_moments("dist_uniform", &uniform())
}

#[test]
fn uniform_sampling_reproducible_and_fits_cdf() {
    check_sampling_ks(&uniform(), 5);
}
