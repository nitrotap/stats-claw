//! Kolmogorov–Smirnov one- and two-sample tests.

use super::kolmogorov_sf;
use crate::distributions::Cdf;
use crate::distributions::NormalDistribution;
use crate::error::{Error, Result};
use crate::tests_stat::exact::ks as exact;
use crate::tests_stat::parametric::len_f64;
use crate::tests_stat::{Alternative, Mode, TestResult};

/// Default `Auto` threshold: the exact KS distributions are used when the
/// governing sample size is below this, otherwise the asymptotic Kolmogorov tail.
/// At `n = 100` the Marsaglia–Tsang–Wang matrix stays small and its `n`th power
/// is fast, while the asymptotic tail is already accurate there.
const EXACT_THRESHOLD: usize = 100;

/// One-sample Kolmogorov–Smirnov test of `sample` against a normal reference,
/// matching `scipy.stats.kstest(sample, "norm", method="asymp")`.
///
/// Computes `D = supₓ |F_n(x) − Φ(x)|` against the normal CDF with the given
/// `mean`/`std_dev` (use `0`/`1` for the standard normal) and takes the p-value
/// from the asymptotic Kolmogorov distribution.
///
/// # Arguments
///
/// * `sample` — the observations (non-empty).
/// * `mean`, `std_dev` — the reference normal's parameters (`std_dev > 0`).
///
/// # Returns
///
/// A [`TestResult`] with `statistic = D`, the asymptotic p-value, no df, no effect
/// size.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] for an empty sample and [`Error::InvalidInput`]
/// for a non-positive `std_dev`.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::goodness_of_fit::ks_one_sample;
///
/// // Testing a standard-normal-looking sample: D should be small.
/// let sample = [-1.0_f64, -0.5, 0.0, 0.5, 1.0];
/// let r = ks_one_sample(&sample, 0.0, 1.0)?;
/// assert!(r.statistic >= 0.0, "D was {}", r.statistic);
/// assert!((0.0..=1.0).contains(&r.p_value));
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn ks_one_sample(sample: &[f64], mean: f64, std_dev: f64) -> Result<TestResult> {
    if sample.is_empty() {
        return Err(Error::EmptyInput);
    }
    if std_dev <= 0.0 {
        return Err(Error::InvalidInput("std_dev must be positive".to_owned()));
    }
    let dist = NormalDistribution {
        mean,
        standard_deviation: std_dev,
        ..Default::default()
    };
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = len_f64(sorted.len());
    let mut d = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let cdf = dist.cdf(x);
        let upper = len_f64(i + 1) / n - cdf;
        let lower = cdf - len_f64(i) / n;
        d = d.max(upper).max(lower);
    }
    let p_value = kolmogorov_sf(n.sqrt() * d);
    Ok(TestResult {
        statistic: d,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// Two-sample Kolmogorov–Smirnov test, matching
/// `scipy.stats.ks_2samp(a, b, method="asymp")`.
///
/// Computes `D = supₓ |F_a(x) − F_b(x)|` over the pooled support and takes the
/// p-value from the asymptotic Kolmogorov distribution scaled by the effective
/// sample size `√(nₐ·n_b / (nₐ + n_b))`.
///
/// # Arguments
///
/// * `a`, `b` — the two samples (each non-empty).
///
/// # Returns
///
/// A [`TestResult`] with `statistic = D`, the asymptotic p-value, no df, no effect
/// size.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] if either sample is empty.
pub fn ks_two_sample(first: &[f64], second: &[f64]) -> Result<TestResult> {
    if first.is_empty() || second.is_empty() {
        return Err(Error::EmptyInput);
    }
    let mut sorted_a = first.to_vec();
    let mut sorted_b = second.to_vec();
    sorted_a.sort_by(f64::total_cmp);
    sorted_b.sort_by(f64::total_cmp);
    let na = len_f64(sorted_a.len());
    let nb = len_f64(sorted_b.len());

    // Merge the sorted samples, tracking each empirical CDF at every step.
    let mut idx_a = 0;
    let mut idx_b = 0;
    let mut cdf_a = 0.0;
    let mut cdf_b = 0.0;
    let mut d = 0.0_f64;
    while idx_a < sorted_a.len() && idx_b < sorted_b.len() {
        let va = sorted_a.get(idx_a).copied().unwrap_or(f64::NAN);
        let vb = sorted_b.get(idx_b).copied().unwrap_or(f64::NAN);
        if va <= vb {
            idx_a += 1;
            cdf_a = len_f64(idx_a) / na;
        }
        if vb <= va {
            idx_b += 1;
            cdf_b = len_f64(idx_b) / nb;
        }
        d = d.max((cdf_a - cdf_b).abs());
    }
    let en = (na * nb / (na + nb)).sqrt();
    let p_value = kolmogorov_sf(en * d);
    Ok(TestResult {
        statistic: d,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// One-sample KS test with an explicit exact / asymptotic / auto mode selector,
/// matching `scipy.stats.ks_1samp(sample, cdf, alternative, method)`.
///
/// The directional statistic follows scipy: [`Alternative::TwoSided`] uses
/// `D = sup|F_n − Φ|`, [`Alternative::Greater`] uses `D⁺ = sup(F_n − Φ)`, and
/// [`Alternative::Less`] uses `D⁻ = sup(Φ − F_n)`. In [`Mode::Exact`] the p-value
/// is the exact finite-`n` distribution
/// ([`crate::tests_stat::exact::ks`]); [`Mode::Asymptotic`] uses the asymptotic
/// Kolmogorov tail (two-sided only); [`Mode::Auto`] picks exact below
/// [`EXACT_THRESHOLD`].
///
/// # Arguments
///
/// * `sample` — the observations (non-empty).
/// * `mean`, `std_dev` — the reference normal's parameters (`std_dev > 0`).
/// * `alternative` — the tested direction.
/// * `mode` — the p-value computation mode.
///
/// # Returns
///
/// A [`TestResult`] with `statistic = D` (directional), the mode's p-value, no df,
/// no effect size.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] for an empty sample, [`Error::InvalidInput`] for
/// a non-positive `std_dev`, and [`Error::InvalidInput`] when a one-sided
/// alternative is combined with the asymptotic mode (one-sided KS has no exact
/// asymptotic tail here — use [`Mode::Exact`]).
pub fn ks_one_sample_mode(
    sample: &[f64],
    mean: f64,
    std_dev: f64,
    alternative: Alternative,
    mode: Mode,
) -> Result<TestResult> {
    if sample.is_empty() {
        return Err(Error::EmptyInput);
    }
    if std_dev <= 0.0 {
        return Err(Error::InvalidInput("std_dev must be positive".to_owned()));
    }
    let dist = NormalDistribution {
        mean,
        standard_deviation: std_dev,
        ..Default::default()
    };
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let n = len_f64(count);
    let mut d_plus = 0.0_f64;
    let mut d_minus = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let cdf = dist.cdf(x);
        d_plus = d_plus.max(len_f64(i + 1) / n - cdf);
        d_minus = d_minus.max(cdf - len_f64(i) / n);
    }
    let d = match alternative {
        Alternative::TwoSided => d_plus.max(d_minus),
        Alternative::Greater => d_plus,
        Alternative::Less => d_minus,
    };

    let resolved = mode.resolve(count, EXACT_THRESHOLD);
    let p_value = match (resolved, alternative) {
        (Mode::Exact, _) => exact::one_sample_p(d, count, alternative),
        (_, Alternative::TwoSided) => kolmogorov_sf(n.sqrt() * d),
        (_, Alternative::Less | Alternative::Greater) => {
            return Err(Error::InvalidInput(
                "one-sided KS supports only exact mode here".to_owned(),
            ));
        }
    };
    Ok(TestResult {
        statistic: d,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// Two-sample KS test with an explicit exact / asymptotic / auto mode selector,
/// matching `scipy.stats.ks_2samp(a, b, method)` for the two-sided alternative.
///
/// In [`Mode::Exact`] the p-value comes from the exact lattice-path count
/// ([`crate::tests_stat::exact::ks`]); [`Mode::Asymptotic`] uses the asymptotic
/// Kolmogorov tail (as [`ks_two_sample`]); [`Mode::Auto`] picks exact when the
/// smaller sample is below [`EXACT_THRESHOLD`].
///
/// # Arguments
///
/// * `first`, `second` — the two samples (each non-empty).
/// * `mode` — the p-value computation mode.
///
/// # Returns
///
/// A [`TestResult`] with `statistic = D`, the mode's two-sided p-value, no df, no
/// effect size.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] if either sample is empty.
pub fn ks_two_sample_mode(first: &[f64], second: &[f64], mode: Mode) -> Result<TestResult> {
    let base = ks_two_sample(first, second)?;
    let n1 = first.len();
    let n2 = second.len();
    let resolved = mode.resolve(n1.min(n2), EXACT_THRESHOLD);
    if matches!(resolved, Mode::Asymptotic) {
        return Ok(base);
    }
    let p_value = exact::two_sample_p(base.statistic, n1, n2);
    Ok(TestResult { p_value, ..base })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty sample is a typed error.
    #[test]
    fn empty_one_sample_is_error() {
        assert!(matches!(
            ks_one_sample(&[], 0.0, 1.0),
            Err(Error::EmptyInput)
        ));
    }

    /// Identical samples have a zero two-sample statistic.
    #[test]
    fn identical_two_samples_have_zero_d() -> Result<()> {
        let a = [1.0, 2.0, 3.0, 4.0];
        let r = ks_two_sample(&a, &a)?;
        assert!(r.statistic.abs() < 1e-12, "D was {}", r.statistic);
        Ok(())
    }
}
