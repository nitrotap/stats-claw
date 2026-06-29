//! Shared helpers for the per-subgroup distribution equivalence tests.
//!
//! These wrap the repeated four-check structure (pdf/cdf/ppf vs scipy,
//! round-trip, moments, seeded sampling) so each subgroup file stays focused and
//! under the 200-line cap. Helpers borrow the parent binary's `common` harness
//! via `super::common`.
//!
//! P4 additions (C4a/C4b/C4c):
//! * [`check_moment_convergence`] — empirical mean/variance converge to analytic
//!   `mean()`/`variance()` on a large seeded draw.
//! * [`check_tail_stress`] — pdf ≥ 0, cdf ∈ [0,1], no NaN/±∞ across an
//!   extreme-spanning grid.
//! * [`check_param_sweep`] — pdf/cdf/quantile/moments over a slice of parameter
//!   sets supplied by the caller.

// `#[path]`-included per test binary, so the shared `pub` helpers look
// unreachable and some are unused in any single binary — both false positives
// for a cross-file test harness, mirroring the `common` module's allow.
#![allow(unreachable_pub, dead_code)]

use super::common::{self, HarnessError};
use stats_claw::distributions::{Cdf, Moments, Pdf, Pmf, Quantile, Sample};
use stats_claw::rng::SplitMix64;

pub mod coverage;
pub mod discrete;
pub mod param_sweep;
pub mod positive;
pub mod sampling;
pub mod symmetric;

/// Asserts a continuous distribution's `pdf`, `cdf`, and `quantile` match the
/// scipy fixture `name` over its stored grid, within AC-1 tolerances.
///
/// # Arguments
///
/// * `name` — golden fixture basename.
/// * `dist` — the distribution under test (implements pdf/cdf/quantile).
///
/// # Errors
///
/// Returns [`HarnessError`] if the fixture cannot be loaded or shaped.
pub fn check_continuous_grid<D>(name: &str, dist: &D) -> Result<(), HarnessError>
where
    D: Pdf + Cdf + Quantile,
{
    let fx = common::load(name)?;
    let xs = common::f64s(&fx, "x")?;
    let pdf: Vec<f64> = xs.iter().map(|&x| dist.pdf(x)).collect();
    let cdf: Vec<f64> = xs.iter().map(|&x| dist.cdf(x)).collect();
    common::assert_vec_close(&pdf, &common::f64s(&fx, "pdf")?, 1e-10, 1e-9);
    common::assert_vec_close(&cdf, &common::f64s(&fx, "cdf")?, 1e-10, 1e-9);
    assert_non_decreasing(&cdf);
    let ppf: Vec<f64> = common::f64s(&fx, "p")?
        .iter()
        .map(|&p| dist.quantile(p))
        .collect();
    common::assert_vec_close(&ppf, &common::f64s(&fx, "ppf")?, 1e-9, 1e-9);
    Ok(())
}

/// Asserts `cdf(quantile(p)) == p` within tolerance over a probability grid.
pub fn check_round_trip<D>(dist: &D, ps: &[f64])
where
    D: Cdf + Quantile,
{
    for &p in ps {
        common::assert_close(dist.cdf(dist.quantile(p)), p, 1e-9, 1e-9);
    }
}

/// Asserts `mean()`/`variance()` match the fixture's stored values.
///
/// `None` is required to coincide with a missing or NaN fixture entry (scipy
/// reports NaN for undefined moments).
///
/// # Errors
///
/// Returns [`HarnessError`] if the fixture cannot be loaded.
pub fn check_moments<D>(name: &str, dist: &D) -> Result<(), HarnessError>
where
    D: Moments,
{
    let fx = common::load(name)?;
    assert_moment(
        dist.mean(),
        fx.get("mean").and_then(serde_json::Value::as_f64),
    );
    assert_moment(
        dist.variance(),
        fx.get("variance").and_then(serde_json::Value::as_f64),
    );
    Ok(())
}

/// Asserts a continuous sampler is reproducible under a fixed seed and that a
/// large draw fits its own `cdf` under a Kolmogorov–Smirnov check.
pub fn check_sampling_ks<D>(dist: &D, seed: u64)
where
    D: Cdf + Sample,
{
    let draw = |s: u64| {
        let mut r = SplitMix64::new(s);
        (0..20_000)
            .map(|_| dist.sample(&mut r))
            .collect::<Vec<f64>>()
    };
    let a = draw(seed);
    assert_eq!(a, draw(seed), "sampler not reproducible for seed {seed}");
    let mut sorted = a;
    sorted.sort_by(f64::total_cmp);
    let ks = common::ks_statistic(&sorted, |x| dist.cdf(x));
    // Use the 1% Kolmogorov critical value (1.63/√n) rather than the 5% one: this
    // is a sampler-correctness sanity check, not an inferential test, so the
    // looser bound avoids the ~5% false-rejection rate that a boundary seed can
    // hit while still flagging any genuinely wrong sampler (whose KS is far
    // larger). A broken sampler fails either threshold by a wide margin.
    let n = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    let crit = 1.63 / f64::from(n).sqrt();
    assert!(ks < crit, "KS={ks} exceeds 1% critical {crit}");
}

/// Asserts a discrete distribution's `pmf` and `cdf` match the scipy fixture
/// `name` over the integer `support`, and that `quantile` reproduces scipy's
/// step-function `ppf` over the probability grid.
///
/// # Errors
///
/// Returns [`HarnessError`] if the fixture cannot be loaded or shaped.
pub fn check_discrete_grid<D>(name: &str, dist: &D, support: &[i64]) -> Result<(), HarnessError>
where
    D: Pmf + Cdf + Quantile,
{
    let fx = common::load(name)?;
    let pmf: Vec<f64> = support.iter().map(|&k| dist.pmf(k)).collect();
    let cdf: Vec<f64> = support.iter().map(|&k| dist.cdf(int_to_f64(k))).collect();
    common::assert_vec_close(&pmf, &common::f64s(&fx, "pmf")?, 1e-10, 1e-9);
    common::assert_vec_close(&cdf, &common::f64s(&fx, "cdf")?, 1e-10, 1e-9);
    assert_non_decreasing(&cdf);
    let ppf: Vec<f64> = common::f64s(&fx, "p")?
        .iter()
        .map(|&p| dist.quantile(p))
        .collect();
    // scipy's discrete ppf returns integer-valued steps; match exactly.
    common::assert_vec_close(&ppf, &common::f64s(&fx, "ppf")?, 0.0, 0.0);
    Ok(())
}

/// Asserts a discrete sampler is reproducible and that its empirical counts agree
/// with the theoretical `pmf` under a chi-square goodness-of-fit check over the
/// given integer `support`.
pub fn check_sampling_chi2<D>(dist: &D, seed: u64, support: &[i64])
where
    D: Pmf + Sample,
{
    let count = 40_000usize;
    let draw = |s: u64| {
        let mut r = SplitMix64::new(s);
        (0..count)
            .map(|_| dist.sample(&mut r))
            .collect::<Vec<f64>>()
    };
    let a = draw(seed);
    assert_eq!(a, draw(seed), "discrete sampler not reproducible");
    let total = usize_to_f64(count);
    let mut chi2 = 0.0;
    let mut bins = 0u32;
    for &k in support {
        let observed = a.iter().filter(|&&x| same_int(x, k)).count();
        let expected = dist.pmf(k) * total;
        if expected >= 5.0 {
            bins += 1;
            let diff = usize_to_f64(observed) - expected;
            chi2 += diff * diff / expected;
        }
    }
    // Loose ceiling well above the χ² upper-tail critical value for this many
    // pooled bins; a correct sampler sits far below it, a broken one far above.
    let ceiling = 5.0 * f64::from(bins);
    assert!(chi2 < ceiling, "chi2 GoF={chi2} exceeds ceiling {ceiling}");
}

fn assert_moment(actual: Option<f64>, expected: Option<f64>) {
    // A `None` from us must coincide with a missing/NaN fixture entry (scipy
    // reports NaN for undefined moments); a finite value must match within
    // tolerance. Both sides are normalised to "undefined" so a single
    // comparison covers every case without a bare `panic!`.
    let is_undefined = |m: Option<f64>| m.is_none_or(f64::is_nan);
    match (actual, expected) {
        (Some(a), Some(e)) if !a.is_nan() && !e.is_nan() => {
            common::assert_close(a, e, 1e-12, 1e-12);
        }
        (a, e) => assert!(
            is_undefined(a) && is_undefined(e),
            "moment definedness mismatch: actual={a:?}, expected={e:?}"
        ),
    }
}

fn assert_non_decreasing(values: &[f64]) {
    for pair in values.windows(2) {
        if let [lo, hi] = pair {
            assert!(*hi >= *lo - 1e-12, "cdf not monotone: {lo} then {hi}");
        }
    }
}

/// Lossless `i64`→`f64` for small support points (`|k| ≤ i32::MAX`).
fn int_to_f64(k: i64) -> f64 {
    f64::from(i32::try_from(k).unwrap_or(i32::MAX))
}

/// Lossless `usize`→`f64` for sample counts (`≤ u32::MAX`).
fn usize_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// Whether a sampled `f64` equals integer `k` (samplers emit exact integers).
fn same_int(x: f64, k: i64) -> bool {
    (x - int_to_f64(k)).abs() < 0.5
}

/// Asserts that a large seeded sample's empirical mean and variance converge to
/// `dist.mean()` and `dist.variance()` within documented sampling tolerances.
///
/// Draws `n = 100_000` variates from a fixed seed and checks that the empirical
/// moments lie within `atol + rtol * |expected|` of the theoretical values.
/// Call sites choose tolerances appropriate for the distribution's skewness and
/// tail behaviour; a `3σ / √n` bound is a sound default for light-tailed families.
///
/// # Arguments
///
/// * `dist` — the distribution under test.
/// * `seed` — reproducible RNG seed.
/// * `mu_abs`, `mu_rel` — absolute and relative tolerance for the mean.
/// * `sigma_abs`, `sigma_rel` — absolute and relative tolerance for the variance.
///
/// # Panics
///
/// Panics (failing the test) when the empirical moments fall outside tolerance.
pub fn check_moment_convergence<D>(
    dist: &D,
    seed: u64,
    mu_abs: f64,
    mu_rel: f64,
    sigma_abs: f64,
    sigma_rel: f64,
) where
    D: Moments + Sample,
{
    // Only applicable when moments are defined.
    let Some(true_mean) = dist.mean() else { return };
    let Some(true_var) = dist.variance() else {
        return;
    };

    let n = 100_000usize;
    let mut rng = SplitMix64::new(seed);
    let samples: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

    let n_f = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let emp_mean = samples.iter().sum::<f64>() / n_f;
    let emp_var = samples
        .iter()
        .map(|&x| (x - emp_mean) * (x - emp_mean))
        .sum::<f64>()
        / (n_f - 1.0);

    let tol_mean = mu_rel.mul_add(true_mean.abs(), mu_abs);
    let tol_var = sigma_rel.mul_add(true_var.abs(), sigma_abs);

    assert!(
        (emp_mean - true_mean).abs() <= tol_mean,
        "moment-convergence (mean): empirical={emp_mean}, analytic={true_mean}, \
         diff={}, tol={tol_mean}",
        (emp_mean - true_mean).abs()
    );
    assert!(
        (emp_var - true_var).abs() <= tol_var,
        "moment-convergence (variance): empirical={emp_var}, analytic={true_var}, \
         diff={}, tol={tol_var}",
        (emp_var - true_var).abs()
    );
}

/// Asserts that a discrete distribution's `pmf` and `cdf` stay within valid
/// ranges across a wide integer grid including boundary (k=0), body, and large k
/// values where pmf decays to 0 and cdf saturates toward 1 (C4b discrete
/// tail-stress guard).
///
/// Parallel to [`check_tail_stress`] for the `Pmf + Cdf` trait boundary: because
/// `Pmf` and `Pdf` are separate traits, discrete distributions (Binomial, Poisson)
/// cannot be covered by `check_tail_stress` and need this dedicated helper.
///
/// For every integer `k` in `ks`:
/// - `pmf(k) ≥ 0` and is finite (no NaN, no ±∞ overflow).
/// - `cdf(k as f64) ∈ [0, 1]` and is finite.
///
/// # Arguments
///
/// * `dist` — the distribution under test (implements `Pmf` and `Cdf`).
/// * `ks` — the integer evaluation grid, which should span k=0, the central body,
///   and very large k values beyond the effective support.
///
/// # Panics
///
/// Panics (failing the test) on any violation.
pub fn check_pmf_tail_stress<D>(dist: &D, ks: &[i64])
where
    D: Pmf + Cdf,
{
    for &k in ks {
        let p = dist.pmf(k);
        let c = dist.cdf(int_to_f64(k));
        assert!(
            p.is_finite() && p >= 0.0,
            "pmf({k}) = {p} is not finite and non-negative"
        );
        assert!(
            c.is_finite() && (0.0..=1.0).contains(&c),
            "cdf({k}) = {c} is not in [0, 1]"
        );
    }
}

/// Asserts that pdf and cdf stay within valid ranges across an extreme-spanning
/// grid including far-tail and boundary inputs (C4b tail-stress guard).
///
/// For every point in `xs`:
/// - `pdf(x) ≥ 0` and is finite (no NaN, no ±∞ overflow).
/// - `cdf(x) ∈ [0, 1]` and is finite.
///
/// # Arguments
///
/// * `dist` — the distribution under test (implements `Pdf` and `Cdf`).
/// * `xs` — the evaluation grid, which should span the far tails and contain
///   extreme-but-finite inputs (e.g. ±1e300 for distributions on ℝ).
///
/// # Panics
///
/// Panics (failing the test) on any violation.
pub fn check_tail_stress<D>(dist: &D, xs: &[f64])
where
    D: Pdf + Cdf,
{
    for &x in xs {
        let p = dist.pdf(x);
        let c = dist.cdf(x);
        assert!(
            p.is_finite() && p >= 0.0,
            "pdf({x}) = {p} is not finite and non-negative"
        );
        assert!(
            c.is_finite() && (0.0..=1.0).contains(&c),
            "cdf({x}) = {c} is not in [0, 1]"
        );
    }
}

/// Asserts that pdf/cdf/quantile/moments are consistent over a grid of
/// parameterizations (C4c parameter-sweep).
///
/// Each element of `param_cases` is a `(dist, ps)` pair: the helper asserts pdf
/// ≥ 0 + cdf monotone at the same fixed x-grid as the fixture, and
/// `cdf(quantile(p)) ≈ p` at the given probability grid, and that any defined
/// moment is finite. This sweeps the parameter space rather than a single
/// fixed point.
///
/// The x-grid and the probability grid `ps` are caller-supplied so each family
/// can choose representative values for its support.
///
/// # Arguments
///
/// * `param_cases` — slice of `(dist, note)` pairs with different parameters;
///   `note` is included in assertion messages.
/// * `xs` — x-values for pdf/cdf sanity checks.
/// * `ps` — probability values for round-trip checks.
///
/// # Panics
///
/// Panics (failing the test) on any violation across all parameterizations.
pub fn check_param_sweep<D>(param_cases: &[(D, &str)], xs: &[f64], ps: &[f64])
where
    D: Pdf + Cdf + Quantile + Moments,
{
    for (dist, note) in param_cases {
        // pdf ≥ 0 and finite.
        let pdfs: Vec<f64> = xs.iter().map(|&x| dist.pdf(x)).collect();
        for (&x, &p) in xs.iter().zip(&pdfs) {
            assert!(p.is_finite() && p >= 0.0, "[{note}] pdf({x}) = {p} invalid");
        }
        // cdf in [0,1] and monotone.
        let cdfs: Vec<f64> = xs.iter().map(|&x| dist.cdf(x)).collect();
        for (&x, &c) in xs.iter().zip(&cdfs) {
            assert!(
                c.is_finite() && (0.0..=1.0).contains(&c),
                "[{note}] cdf({x}) = {c} out of [0,1]"
            );
        }
        assert_non_decreasing(&cdfs);
        // round-trip at interior ps.
        for &p in ps {
            let rt = dist.cdf(dist.quantile(p));
            assert!(
                (rt - p).abs() < 1e-6,
                "[{note}] cdf(quantile({p})) = {rt}, expected {p}"
            );
        }
        // defined moments are finite.
        if let Some(m) = dist.mean() {
            assert!(m.is_finite(), "[{note}] mean() = {m} is not finite");
        }
        if let Some(v) = dist.variance() {
            assert!(
                v.is_finite() && v >= 0.0,
                "[{note}] variance() = {v} invalid"
            );
        }
    }
}
