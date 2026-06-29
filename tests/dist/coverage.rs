//! P4 test-coverage hardening: moment-convergence (C4a) and tail-stress (C4b)
//! tests for all 14 distributions.
//!
//! Addresses the NOT-IMPLEMENTED/partial notes in chapter 01 (Story 1.4
//! moment-convergence and Story 1.1 tail-stress guard). Helpers are in
//! `dist/mod.rs`; parameter-sweep tests (C4c) live in `dist/param_sweep.rs`.
//!
//! **Moment-convergence tolerances (C4a):** `n = 100_000`. Absolute tolerances
//! are chosen at approximately 8–10× the standard error of the empirical moment
//! estimator so any correct sampler passes without false-positive failures from
//! random seed variation.
//!
//! **Tail-stress grids (C4b):** grids span far-tail values chosen so a broken
//! implementation would overflow to NaN/±∞ but a correct one saturates cleanly
//! to 0 or 1.

use super::{check_moment_convergence, check_pmf_tail_stress, check_tail_stress};
use stats_claw::distributions::{
    BetaDistribution, BinomialDistribution, CauchyDistribution, ChiSquaredDistribution,
    ExponentialDistribution, FDistribution, GammaDistribution, LaplaceDistribution,
    LogNormalDistribution, NormalDistribution, PoissonDistribution, TDistribution,
    UniformDistribution, WeibullDistribution,
};

// ─── C4a — Moment-convergence ────────────────────────────────────────────────

#[test]
fn normal_moment_convergence() {
    // Normal(μ=1.5, σ=2.0): mean=1.5, var=4.0.
    // σ_mean = σ/√n = 2/316.2 ≈ 0.0063; gate ~8σ at 0.05.
    check_moment_convergence(
        &NormalDistribution {
            mean: 1.5,
            standard_deviation: 2.0,
            ..Default::default()
        },
        123,
        0.05,
        0.0, // mean atol/rtol
        0.15,
        0.0, // var atol/rtol
    );
}

#[test]
fn laplace_moment_convergence() {
    // Laplace(loc=-0.5, scale=1.5): mean=-0.5, var=4.5.
    check_moment_convergence(
        &LaplaceDistribution {
            location: -0.5,
            scale: 1.5,
            ..Default::default()
        },
        7,
        0.07,
        0.0,
        0.20,
        0.0,
    );
}

#[test]
fn uniform_moment_convergence() {
    // Uniform(lower=-2, upper=3): mean=0.5, var=25/12≈2.083.
    check_moment_convergence(
        &UniformDistribution {
            lower_bound: -2.0,
            upper_bound: 3.0,
            ..Default::default()
        },
        5,
        0.04,
        0.0,
        0.10,
        0.0,
    );
}

#[test]
fn exponential_moment_convergence() {
    // Exponential(rate=0.75): mean=1.333, var=1.778.
    check_moment_convergence(
        &ExponentialDistribution {
            rate_parameter: 0.75,
            ..Default::default()
        },
        11,
        0.05,
        0.0,
        0.15,
        0.0,
    );
}

#[test]
fn weibull_moment_convergence() {
    // Weibull(shape=1.8, scale=2.5): mean≈2.197, var≈1.507.
    check_moment_convergence(
        &WeibullDistribution {
            shape_parameter: 1.8,
            scale_parameter: 2.5,
            ..Default::default()
        },
        13,
        0.05,
        0.0,
        0.15,
        0.0,
    );
}

#[test]
fn lognormal_moment_convergence() {
    // LogNormal(μ_log=0.3, σ_log=0.6): heavy right tail, use rtol.
    check_moment_convergence(
        &LogNormalDistribution {
            mean_log_value: 0.3,
            std_log_value: 0.6,
            ..Default::default()
        },
        17,
        0.10,
        0.10,
        0.30,
        0.20,
    );
}

#[test]
fn gamma_moment_convergence() {
    // Gamma(shape=2.5, scale=1.5): mean=3.75, var=5.625.
    check_moment_convergence(
        &GammaDistribution {
            shape_parameter: 2.5,
            scale_parameter: 1.5,
            ..Default::default()
        },
        19,
        0.07,
        0.0,
        0.25,
        0.0,
    );
}

#[test]
fn beta_moment_convergence() {
    // Beta(α=2, β=5): mean≈0.286, var≈0.034.
    check_moment_convergence(
        &BetaDistribution {
            alpha_parameter: 2.0,
            beta_parameter: 5.0,
            ..Default::default()
        },
        23,
        0.01,
        0.0,
        0.005,
        0.0,
    );
}

#[test]
fn chi_squared_moment_convergence() {
    // ChiSquared(df=5): mean=5, var=10.
    check_moment_convergence(
        &ChiSquaredDistribution {
            degrees_of_freedom: 5,
            ..Default::default()
        },
        29,
        0.07,
        0.0,
        0.40,
        0.0,
    );
}

#[test]
fn students_t_moment_convergence() {
    // T(df=7): mean=0, var=1.4.
    check_moment_convergence(
        &TDistribution {
            degrees_of_freedom: 7,
            ..Default::default()
        },
        31,
        0.03,
        0.0,
        0.15,
        0.0,
    );
}

#[test]
fn f_moment_convergence() {
    // F(d1=6, d2=12): mean=1.2, var≈0.432.
    check_moment_convergence(
        &FDistribution {
            numerator_df: 6,
            denominator_df: 12,
            ..Default::default()
        },
        37,
        0.05,
        0.0,
        0.15,
        0.0,
    );
}

#[test]
fn binomial_moment_convergence() {
    // Binomial(n=20, p=0.35): mean=7, var=4.55.
    check_moment_convergence(
        &BinomialDistribution {
            number_of_trials: 20,
            success_probability: 0.35,
            ..Default::default()
        },
        41,
        0.07,
        0.0,
        0.20,
        0.0,
    );
}

#[test]
fn poisson_moment_convergence() {
    // Poisson(λ=4): mean=4, var=4.
    check_moment_convergence(
        &PoissonDistribution {
            rate_parameter: 4.0,
            ..Default::default()
        },
        43,
        0.07,
        0.0,
        0.20,
        0.0,
    );
}

/// Cauchy has no finite moments; `check_moment_convergence` short-circuits via
/// the `None` guard and performs no assertion — this test documents the skip.
#[test]
fn cauchy_moment_convergence_skipped_no_finite_moments() {
    check_moment_convergence(
        &CauchyDistribution {
            location: 0.0,
            scale: 1.0,
            ..Default::default()
        },
        99,
        0.0,
        0.0,
        0.0,
        0.0,
    );
}

// ─── C4b — Tail-stress guard ─────────────────────────────────────────────────

/// Grid spanning both far tails of ℝ, including extreme-but-finite values.
const REAL_LINE_GRID: &[f64] = &[
    -1e10, -1e6, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1e6, 1e10,
];

/// Grid for distributions on (0, ∞): very small positives and very large values.
const POSITIVE_GRID: &[f64] = &[1e-300, 1e-10, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e6, 1e15];

/// Grid for distributions on [0, 1]: boundary and near-boundary values.
const UNIT_GRID: &[f64] = &[0.0, 1e-15, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-15, 1.0];

#[test]
fn normal_tail_stress() {
    check_tail_stress(
        &NormalDistribution {
            mean: 1.5,
            standard_deviation: 2.0,
            ..Default::default()
        },
        REAL_LINE_GRID,
    );
}

#[test]
fn laplace_tail_stress() {
    check_tail_stress(
        &LaplaceDistribution {
            location: -0.5,
            scale: 1.5,
            ..Default::default()
        },
        REAL_LINE_GRID,
    );
}

#[test]
fn cauchy_tail_stress() {
    check_tail_stress(
        &CauchyDistribution {
            location: 0.0,
            scale: 1.0,
            ..Default::default()
        },
        REAL_LINE_GRID,
    );
}

#[test]
fn uniform_tail_stress() {
    // pdf is 0 outside [lower, upper]; cdf saturates to 0/1. Both valid.
    check_tail_stress(
        &UniformDistribution {
            lower_bound: -2.0,
            upper_bound: 3.0,
            ..Default::default()
        },
        REAL_LINE_GRID,
    );
}

#[test]
fn exponential_tail_stress() {
    check_tail_stress(
        &ExponentialDistribution {
            rate_parameter: 0.75,
            ..Default::default()
        },
        POSITIVE_GRID,
    );
}

#[test]
fn weibull_tail_stress() {
    check_tail_stress(
        &WeibullDistribution {
            shape_parameter: 1.8,
            scale_parameter: 2.5,
            ..Default::default()
        },
        POSITIVE_GRID,
    );
}

#[test]
fn lognormal_tail_stress() {
    check_tail_stress(
        &LogNormalDistribution {
            mean_log_value: 0.3,
            std_log_value: 0.6,
            ..Default::default()
        },
        POSITIVE_GRID,
    );
}

#[test]
fn gamma_tail_stress() {
    check_tail_stress(
        &GammaDistribution {
            shape_parameter: 2.5,
            scale_parameter: 1.5,
            ..Default::default()
        },
        POSITIVE_GRID,
    );
}

#[test]
fn beta_tail_stress() {
    check_tail_stress(
        &BetaDistribution {
            alpha_parameter: 2.0,
            beta_parameter: 5.0,
            ..Default::default()
        },
        UNIT_GRID,
    );
}

#[test]
fn chi_squared_tail_stress() {
    check_tail_stress(
        &ChiSquaredDistribution {
            degrees_of_freedom: 5,
            ..Default::default()
        },
        POSITIVE_GRID,
    );
}

#[test]
fn students_t_tail_stress() {
    check_tail_stress(
        &TDistribution {
            degrees_of_freedom: 7,
            ..Default::default()
        },
        REAL_LINE_GRID,
    );
}

#[test]
fn f_tail_stress() {
    check_tail_stress(
        &FDistribution {
            numerator_df: 6,
            denominator_df: 12,
            ..Default::default()
        },
        POSITIVE_GRID,
    );
}

// ─── C4b — Discrete PMF tail-stress ─────────────────────────────────────────
//
// Binomial and Poisson implement `Pmf + Cdf`, not `Pdf + Cdf`, so
// `check_tail_stress` (bounded on `Pdf`) cannot cover them. The parallel
// `check_pmf_tail_stress` helper asserts the same invariants via the `Pmf`
// trait: pmf ≥ 0 ∧ finite; cdf ∈ [0,1] ∧ finite; no NaN/overflow across a
// wide integer grid including large k (cdf saturation toward 1) and k = 0.

/// Integer k-grid for discrete tail-stress: k=0 (boundary), body, and very
/// large k (beyond the support, where pmf must be 0 and cdf must be 1).
const DISCRETE_GRID: &[i64] = &[0, 1, 5, 10, 50, 100, 500, 1_000, 10_000, 1_000_000];

#[test]
fn binomial_pmf_tail_stress() {
    // Binomial(n=20, p=0.35): support is [0, 20]. Grid extends far beyond n to
    // verify pmf = 0 and cdf = 1 for k > n.
    check_pmf_tail_stress(
        &BinomialDistribution {
            number_of_trials: 20,
            success_probability: 0.35,
            ..Default::default()
        },
        DISCRETE_GRID,
    );
}

#[test]
fn poisson_pmf_tail_stress() {
    // Poisson(λ=4): unbounded support; large k is deep in the right tail where
    // the pmf decays toward 0 and cdf saturates toward 1.
    check_pmf_tail_stress(
        &PoissonDistribution {
            rate_parameter: 4.0,
            ..Default::default()
        },
        DISCRETE_GRID,
    );
}
