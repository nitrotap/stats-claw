//! Goodness-of-fit tests (KS, Anderson-Darling, Shapiro-Wilk).
//!
//! Houses the Kolmogorov–Smirnov one- and two-sample tests, the Anderson–Darling
//! normality test, and the Shapiro–Wilk normality test. The KS p-values use the
//! asymptotic Kolmogorov distribution (matching scipy's `method="asymp"`).

pub mod anderson;
pub mod ks_test;
pub mod shapiro;

pub use anderson::{anderson_darling, anderson_normal_critical_values};
pub use ks_test::{ks_one_sample, ks_one_sample_mode, ks_two_sample, ks_two_sample_mode};
pub use shapiro::shapiro_wilk;

/// Asymptotic Kolmogorov survival function `Q(x) = 2·Σ_{j≥1} (−1)^{j−1} e^{−2j²x²}`.
///
/// This is `scipy.stats.kstwobign.sf`, the large-sample null distribution both KS
/// tests evaluate at the scaled statistic.
///
/// # Arguments
///
/// * `x` — the scaled KS statistic (`≥ 0`); `x = 0` yields `1.0`.
///
/// # Returns
///
/// The upper-tail probability in `[0, 1]`.
pub(crate) fn kolmogorov_sf(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut sign = 1.0_f64;
    for j in 1..=100 {
        let jf = f64::from(j);
        let term = (-2.0 * jf * jf * x * x).exp();
        sum = sign.mul_add(term, sum);
        sign = -sign;
        if term < 1e-15 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}
