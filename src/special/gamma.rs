//! Gamma-family special functions: log-gamma (Lanczos) and the regularized
//! incomplete gamma `P`/`Q` (series expansion + Lentz continued fraction).
//!
//! Accuracy target is ~1e-12, validated against scipy golden fixtures by the
//! distribution suites. All routines avoid `as` casts (the `style.rs` guard
//! forbids them in `src/`): loop indices are carried as `f64` accumulators.

use std::f64::consts::PI;

/// `g` parameter of the Lanczos approximation (matched to [`LANCZOS`]).
const G: f64 = 7.0;

/// Lanczos coefficients for `g = 7`, `n = 9` (accurate to ~1e-15).
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Convergence floor and iteration cap shared by the series/continued fraction.
const TINY: f64 = 1e-300;
const REL_EPS: f64 = 1e-16;
const MAX_ITERS: usize = 1000;

/// Computes the natural log of the gamma function via the Lanczos approximation.
///
/// Values below `0.5` are mapped through the reflection formula
/// `Γ(x)Γ(1−x) = π / sin(πx)` so the approximation only runs on its accurate
/// right half-line.
///
/// # Arguments
///
/// * `x` — the argument; finite and not a non-positive integer (where the gamma
///   function has poles).
///
/// # Returns
///
/// `ln Γ(x)`.
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = first_lanczos();
    let t = x + G + 0.5;
    let mut k = 0.0_f64;
    for &c in LANCZOS.iter().skip(1) {
        k += 1.0;
        a += c / (x + k);
    }
    (x + 0.5).mul_add(t.ln(), 0.5 * (2.0 * PI).ln()) - t + a.ln()
}

/// Returns the leading Lanczos coefficient (helper keeps `ln_gamma` index-free).
fn first_lanczos() -> f64 {
    LANCZOS.first().copied().unwrap_or(0.0)
}

/// Computes the natural log of the beta function `B(a, b)`.
///
/// # Arguments
///
/// * `a`, `b` — beta parameters; both must be positive.
///
/// # Returns
///
/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a+b)`.
#[must_use]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Computes the natural log of the binomial coefficient `C(n, k)`.
///
/// Uses `ln C(n, k) = ln Γ(n+1) − ln Γ(k+1) − ln Γ(n−k+1)` so large coefficients
/// (e.g. `C(n1+n2, n1)` in the exact KS two-sample count) never overflow before
/// being exponentiated.
///
/// # Arguments
///
/// * `n` — the population size.
/// * `k` — the number chosen; values `k > n` yield `−∞` (`C = 0`).
///
/// # Returns
///
/// `ln C(n, k)`, or `f64::NEG_INFINITY` when `k > n`.
#[must_use]
pub fn ln_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let nf = usize_to_f64(n);
    let kf = usize_to_f64(k);
    ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0)
}

/// Widens a `usize` to `f64` losslessly without an `as` cast (counts here stay
/// far below `2^53`).
fn usize_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Computes the regularized lower incomplete gamma `P(a, x)`.
///
/// # Arguments
///
/// * `a` — shape parameter; must be `> 0`.
/// * `x` — evaluation point; must be `>= 0`.
///
/// # Returns
///
/// `P(a, x)` in `[0, 1]`, or `NaN` if `a <= 0` or `x < 0`.
#[must_use]
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_cf(a, x)
    }
}

/// Computes the regularized upper incomplete gamma `Q(a, x) = 1 − P(a, x)`.
///
/// # Arguments
///
/// * `a` — shape parameter; must be `> 0`.
/// * `x` — evaluation point; must be `>= 0`.
///
/// # Returns
///
/// `Q(a, x)` in `[0, 1]`, or `NaN` if `a <= 0` or `x < 0`.
#[must_use]
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

/// Evaluates `P(a, x)` by its series expansion (converges fast for `x < a + 1`).
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAX_ITERS {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * REL_EPS {
            break;
        }
    }
    sum * (a.mul_add(x.ln(), -x) - ln_gamma(a)).exp()
}

/// Evaluates `Q(a, x)` by Lentz's continued fraction (for `x >= a + 1`).
// Single-char names (`b`, `c`, `d`, `h`, `an`) are the canonical Lentz /
// Numerical-Recipes notation; renaming them would obscure the algorithm.
#[allow(clippy::many_single_char_names)]
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i = 0.0_f64;
    for _ in 0..MAX_ITERS {
        i += 1.0;
        let an = -i * (i - a);
        b += 2.0;
        d = an.mul_add(d, b);
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < REL_EPS {
            break;
        }
    }
    (a.mul_add(x.ln(), -x) - ln_gamma(a)).exp() * h
}

/// Computes `ln Q(a, x)`, the natural log of the regularized upper incomplete
/// gamma, without ever forming the underflowing linear value.
///
/// This is the `scipy.stats.chi2.logsf` / `scipy.stats.gamma.logsf` building
/// block: in the upper tail (`x ≥ a + 1`) the continued-fraction form is
/// `exp(a·ln x − x − lnΓ(a)) · h`, so its log is `a·ln x − x − lnΓ(a) + ln h`,
/// finite where `Q(a, x)` itself has underflowed to `0.0`. On the lower side
/// (`x < a + 1`) it falls back to `ln(1 − P)` via [`f64::ln_1p`], which is well
/// conditioned because `P` is bounded away from `1` there.
///
/// # Arguments
///
/// * `a` — shape parameter; must be `> 0`.
/// * `x` — evaluation point; must be `>= 0`.
///
/// # Returns
///
/// `ln Q(a, x) ∈ (−∞, 0]`, or `NaN` if `a <= 0` or `x < 0`. `x = 0` yields `0`.
#[must_use]
pub fn ln_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Q = 1 − P with P comfortably below 1 here; ln_1p(−P) is accurate.
        (-gamma_p_series(a, x)).ln_1p()
    } else {
        ln_gamma_q_cf(a, x)
    }
}

/// Computes `ln P(a, x)`, the natural log of the regularized lower incomplete
/// gamma, the complement of [`ln_gamma_q`].
///
/// On the lower side (`x < a + 1`) the series form is `S · exp(a·ln x − x −
/// lnΓ(a))`, whose log is `ln S + a·ln x − x − lnΓ(a)` — finite as `x → 0` where
/// `P` underflows. On the upper side it uses `ln(1 − Q)` via [`f64::ln_1p`].
///
/// # Arguments
///
/// * `a` — shape parameter; must be `> 0`.
/// * `x` — evaluation point; must be `>= 0`.
///
/// # Returns
///
/// `ln P(a, x) ∈ (−∞, 0]`, or `NaN` if `a <= 0` or `x < 0`. `x = 0` yields `−∞`.
#[must_use]
pub fn ln_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < a + 1.0 {
        ln_gamma_p_series(a, x)
    } else {
        // P = 1 − Q with Q comfortably below 1 here; ln_1p(−Q) is accurate.
        (-gamma_q_cf(a, x)).ln_1p()
    }
}

/// Evaluates `ln P(a, x)` from the series expansion (the log of [`gamma_p_series`]
/// without exponentiating the prefactor).
fn ln_gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAX_ITERS {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * REL_EPS {
            break;
        }
    }
    sum.ln() + a.mul_add(x.ln(), -x) - ln_gamma(a)
}

/// Evaluates `ln Q(a, x)` from Lentz's continued fraction (the log of
/// [`gamma_q_cf`] without exponentiating the prefactor).
// Single-char names mirror the canonical Lentz / Numerical-Recipes notation.
#[allow(clippy::many_single_char_names)]
fn ln_gamma_q_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i = 0.0_f64;
    for _ in 0..MAX_ITERS {
        i += 1.0;
        let an = -i * (i - a);
        b += 2.0;
        d = an.mul_add(d, b);
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < REL_EPS {
            break;
        }
    }
    a.mul_add(x.ln(), -x) - ln_gamma(a) + h.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ln_gamma_q` agrees with `gamma_q().ln()` on both sides of the `x = a+1`
    /// branch where the linear value has not underflowed.
    #[test]
    fn ln_gamma_q_matches_log_of_linear() {
        // (a, x): first is upper-side (x >= a+1), second is lower-side.
        for &(a, x) in &[(1.5, 3.0), (2.5, 1.0)] {
            let want = gamma_q(a, x).ln();
            let got = ln_gamma_q(a, x);
            assert!(
                ((got - want) / want.abs().max(1.0)).abs() < 1e-12,
                "ln_gamma_q({a},{x}) = {got}, want {want}"
            );
        }
    }

    /// `ln_gamma_p` agrees with `gamma_p().ln()` on both sides of the branch.
    #[test]
    fn ln_gamma_p_matches_log_of_linear() {
        for &(a, x) in &[(1.5, 3.0), (2.5, 1.0)] {
            let want = gamma_p(a, x).ln();
            let got = ln_gamma_p(a, x);
            assert!(
                ((got - want) / want.abs().max(1.0)).abs() < 1e-12,
                "ln_gamma_p({a},{x}) = {got}, want {want}"
            );
        }
    }

    /// `ln_gamma_q` stays finite deep in the tail where `gamma_q` underflows, and
    /// matches the scipy `gamma.logsf` reference (≤ 1e-9 relative).
    #[test]
    fn ln_gamma_q_finite_in_deep_tail() {
        let got = ln_gamma_q(1.5, 400.0);
        let want = -396.882_237_824_145_1; // scipy.stats.gamma.logsf(400, 1.5)
        assert!(got.is_finite(), "ln_gamma_q(1.5, 400) was {got}");
        assert!(
            ((got - want) / want).abs() < 1e-9,
            "ln_gamma_q(1.5, 400) rel error too large: got {got}, want {want}"
        );
    }
}
