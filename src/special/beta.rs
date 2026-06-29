//! Regularized incomplete beta function `I_x(a, b)`.
//!
//! Uses the standard prefactor times Lentz's continued fraction, switching to the
//! reflection `I_x(a,b) = 1 − I_{1-x}(b,a)` on the slowly-converging side. The
//! loop index is carried as an `f64` accumulator to avoid `as` casts (forbidden
//! in `src/` by `style.rs`). Accuracy target ~1e-12.

use super::ln_gamma;

/// Convergence floor and iteration cap for the continued fraction.
const TINY: f64 = 1e-300;
const REL_EPS: f64 = 1e-16;
const MAX_ITERS: usize = 1000;

/// Computes the regularized incomplete beta `I_x(a, b)`.
///
/// # Arguments
///
/// * `a`, `b` — beta parameters; both must be `> 0`.
/// * `x` — evaluation point; clamped to `[0, 1]` (returns the boundary value
///   outside it).
///
/// # Returns
///
/// `I_x(a, b)` in `[0, 1]`.
#[must_use]
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let log_prefactor = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let bt = b
        .mul_add((1.0 - x).ln(), a.mul_add(x.ln(), log_prefactor))
        .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Computes `ln I_x(a, b)`, the natural log of the regularized lower incomplete
/// beta, without forming the underflowing linear value.
///
/// This is the log-space tail used by the Student's-t and F `logsf`/`logcdf`
/// paths. The continued-fraction form `I_x(a,b) = bt · cf / a` (on its
/// fast-converging side) has log `ln bt + ln(cf/a)`, finite where `I_x` itself
/// underflows to `0.0`. On the reflected side it returns `ln(1 − I_{1-x}(b,a))`
/// via [`f64::ln_1p`], whose argument `I_{1-x}(b,a)` is the small (fast-side)
/// quantity, so no catastrophic cancellation occurs.
///
/// # Arguments
///
/// * `a`, `b` — beta parameters; both must be `> 0`.
/// * `x` — evaluation point; `x <= 0` yields `−∞` (`I = 0`), `x >= 1` yields `0`.
///
/// # Returns
///
/// `ln I_x(a, b) ∈ (−∞, 0]`.
#[must_use]
pub fn ln_betai_lower(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return 0.0;
    }
    if x < (a + 1.0) / (a + b + 2.0) {
        // Fast side: I_x(a,b) = bt · cf / a is the small quantity; log directly.
        ln_bt(a, b, x) + (betacf(a, b, x) / a).ln()
    } else {
        // Reflected side: compute the small upper tail ln(1 − I) = ln I_{1-x}(b,a),
        // then ln I = ln(1 − exp(ln_upper)) via ln_1p on the small exponential.
        let ln_upper = ln_bt(b, a, 1.0 - x) + (betacf(b, a, 1.0 - x) / b).ln();
        (-ln_upper.exp()).ln_1p()
    }
}

/// Computes `ln(1 − I_x(a, b))`, the natural log of the regularized *upper*
/// incomplete beta, via the reflection `1 − I_x(a,b) = I_{1-x}(b,a)`.
///
/// Routing through [`ln_betai_lower`] on the reflected arguments keeps the small
/// (underflowing) tail on the fast-converging side, so the upper tail stays
/// finite and accurate where `1 − I_x` would underflow in linear space.
///
/// # Arguments
///
/// * `a`, `b` — beta parameters; both must be `> 0`.
/// * `x` — evaluation point; `x <= 0` yields `0`, `x >= 1` yields `−∞`.
///
/// # Returns
///
/// `ln(1 − I_x(a, b)) ∈ (−∞, 0]`.
#[must_use]
pub fn ln_betai_upper(a: f64, b: f64, x: f64) -> f64 {
    ln_betai_lower(b, a, 1.0 - x)
}

/// Computes `ln bt = ln Γ(a+b) − ln Γ(a) − ln Γ(b) + a·ln x + b·ln(1−x)`, the log
/// of the incomplete-beta prefactor shared by both log tails.
///
/// # Arguments
///
/// * `a`, `b` — beta parameters (both `> 0`).
/// * `x` — evaluation point in `(0, 1)`.
///
/// # Returns
///
/// The log prefactor `ln bt`.
fn ln_bt(a: f64, b: f64, x: f64) -> f64 {
    let log_prefactor = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    b.mul_add((1.0 - x).ln(), a.mul_add(x.ln(), log_prefactor))
}

/// Evaluates the continued fraction for the incomplete beta (Lentz's method).
// Single-char names (`c`, `d`, `h`, `m`) are the canonical Numerical-Recipes
// notation for this continued fraction; renaming them would obscure the math.
#[allow(clippy::many_single_char_names)]
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = lentz_guard(1.0 - qab * x / qap);
    d = 1.0 / d;
    let mut h = d;
    let mut m = 0.0_f64;
    for _ in 0..MAX_ITERS {
        m += 1.0;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = lentz_guard(aa.mul_add(d, 1.0));
        c = lentz_guard(1.0 + aa / c);
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = lentz_guard(aa.mul_add(d, 1.0));
        c = lentz_guard(1.0 + aa / c);
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < REL_EPS {
            break;
        }
    }
    h
}

/// Replaces a near-zero denominator with [`TINY`] to keep Lentz's method stable.
fn lentz_guard(v: f64) -> f64 {
    if v.abs() < TINY { TINY } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ln_betai_lower` agrees with `betai().ln()` on both the fast and reflected
    /// sides of the continued-fraction switch where the linear value is healthy.
    #[test]
    fn ln_betai_lower_matches_log_of_linear() {
        // (a, b, x): fast side (small x), then reflected side (large x).
        for &(a, b, x) in &[(2.0, 3.0, 0.3), (0.5, 2.5, 0.9)] {
            let want = betai(a, b, x).ln();
            let got = ln_betai_lower(a, b, x);
            assert!(
                ((got - want) / want.abs().max(1.0)).abs() < 1e-10,
                "ln_betai_lower({a},{b},{x}) = {got}, want {want}"
            );
        }
    }

    /// `ln_betai_upper` agrees with `(1 - betai()).ln()` in the body.
    #[test]
    fn ln_betai_upper_matches_log_of_complement() {
        for &(a, b, x) in &[(2.0, 3.0, 0.3), (1.5, 0.5, 0.001)] {
            let want = (1.0 - betai(a, b, x)).ln();
            let got = ln_betai_upper(a, b, x);
            assert!(
                ((got - want) / want.abs().max(1.0)).abs() < 1e-10,
                "ln_betai_upper({a},{b},{x}) = {got}, want {want}"
            );
        }
    }

    /// The lower log tail stays finite where the linear `betai` underflows: the
    /// Student's-t upper tail at t = 50, ν = 5 maps to a tiny `I_x(2.5, 0.5)`.
    #[test]
    fn ln_betai_lower_finite_in_deep_tail() {
        let x = 5.0 / 50.0_f64.mul_add(50.0, 5.0); // ν / (ν + t²)
        let got = ln_betai_lower(2.5, 0.5, x);
        // scipy: log(betainc(2.5, 0.5, x)); the t logsf later adds ln(0.5).
        let want = -16.620_993_180_844_884;
        assert!(got.is_finite(), "ln_betai_lower deep tail was {got}");
        assert!(
            ((got - want) / want).abs() < 1e-9,
            "ln_betai_lower deep tail rel error: got {got}, want {want}"
        );
    }
}
