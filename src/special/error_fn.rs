//! Fast, accurate error functions `erf`/`erfc` via W. J. Cody's rational
//! approximation (Cody, *Math. Comp.* 23, 1969; the algorithm used by most libm
//! implementations).
//!
//! This replaces the previous `erf = P(½, x²)` route through the regularized
//! incomplete-gamma series, which paid a full series/continued-fraction expansion
//! per call and made the Normal `cdf` ~18× slower than vectorized scipy. The
//! rational approximation here costs a handful of fused multiply-adds plus one
//! `exp`, while staying accurate to ~1e-16 relative — comfortably inside the
//! distribution tolerance (pdf/cdf abs ≤ 1e-10).
//!
//! Three regions, each with its own minimax rational form:
//!
//! * `|x| < 0.46875` — direct rational for `erf(x)` (a tiny argument, no `exp`).
//! * `0.46875 ≤ |x| < 4` — rational for `erfc(x)` scaled by `exp(−x²)`.
//! * `|x| ≥ 4` — asymptotic rational for `erfc(x)` scaled by `exp(−x²)`.
//!
//! Computing `erfc` directly in the tail (rather than `1 − erf`) preserves
//! relative accuracy where `erf(x) → 1` and catastrophic cancellation would
//! otherwise destroy the small `erfc` value.

/// Threshold below which the direct `erf` rational form is used (Cody `THRESH`).
const SMALL: f64 = 0.468_75;
/// Threshold above which the asymptotic `erfc` form is used.
const LARGE: f64 = 4.0;
/// Magnitude beyond which `erfc` underflows to `0` (Cody `XBIG`); also catches
/// `+∞`, so the `exp(−ax²)` split never forms `∞ − ∞`.
const XBIG: f64 = 26.543;
/// `1/√π` (Cody `SQRPI`), the leading coefficient of the asymptotic expansion.
const INV_SQRT_PI: f64 = 5.641_895_835_477_563e-1;

// Coefficients below are W. J. Cody's CALERF double-precision constants
// (netlib `specfun/erf`), transcribed verbatim from the Fortran DATA statements.

/// Numerator coefficients (Cody array `A`) for `erf` on `|x| ≤ 0.46875`.
const A_ERF: [f64; 5] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_6e2,
    3.774_852_376_853_02e2,
    3.209_377_589_138_469_4e3,
    1.857_777_061_846_031_5e-1,
];
/// Denominator coefficients (Cody array `B`) for `erf` on `|x| ≤ 0.46875`.
const B_ERF: [f64; 4] = [
    2.360_129_095_234_412_2e1,
    2.440_246_379_344_441_7e2,
    1.282_616_526_077_372_3e3,
    2.844_236_833_439_171e3,
];

/// Numerator coefficients (Cody array `C`) for `erfc` on `0.46875 < |x| ≤ 4`.
const C_ERFC: [f64; 9] = [
    5.641_884_969_886_701e-1,
    8.883_149_794_388_377,
    6.611_919_063_714_163e1,
    2.986_351_381_974_001e2,
    8.819_522_212_417_69e2,
    1.712_047_612_634_070_7e3,
    2.051_078_377_826_071_6e3,
    1.230_339_354_797_997_2e3,
    2.153_115_354_744_038_3e-8,
];
/// Denominator coefficients (Cody array `D`) for `erfc` on `0.46875 < |x| ≤ 4`.
const D_ERFC: [f64; 8] = [
    1.574_492_611_070_983_5e1,
    1.176_939_508_913_125e2,
    5.371_811_018_620_099e2,
    1.621_389_574_566_690_3e3,
    3.290_799_235_733_459_7e3,
    4.362_619_090_143_247e3,
    3.439_367_674_143_721_6e3,
    1.230_339_354_803_749_5e3,
];

/// Numerator coefficients (Cody array `P`) for the asymptotic `erfc` on `|x| > 4`.
const P_ASYMP: [f64; 6] = [
    3.053_266_349_612_323_6e-1,
    3.603_448_999_498_044_5e-1,
    1.257_817_261_112_292_6e-1,
    1.608_378_514_874_227_5e-2,
    6.587_491_615_298_378e-4,
    1.631_538_713_730_209_7e-2,
];
/// Denominator coefficients (Cody array `Q`) for the asymptotic `erfc` on `|x| > 4`.
const Q_ASYMP: [f64; 5] = [
    2.568_520_192_289_822,
    1.872_952_849_923_460_4,
    5.279_051_029_514_285e-1,
    6.051_834_131_244_132e-2,
    2.335_204_976_268_691_8e-3,
];

/// Computes the error function `erf(x)` via Cody's rational approximation.
///
/// Accurate to ~1e-16 relative over the whole real line; odd by construction.
///
/// # Arguments
///
/// * `x` — the argument; any `f64`. `NaN` propagates; `±∞` map to `±1`.
///
/// # Returns
///
/// `erf(x)` in `(-1, 1)`.
///
/// # Examples
///
/// ```
/// use stats_claw::special::erf;
/// assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12);
/// assert!((erf(-0.7) + erf(0.7)).abs() < 1e-15);
/// ```
#[must_use]
pub fn erf(x: f64) -> f64 {
    let ax = x.abs();
    if ax < SMALL {
        return erf_small(x);
    }
    let c = erfc_ge_half(ax);
    if x < 0.0 { c - 1.0 } else { 1.0 - c }
}

/// Computes the complementary error function `erfc(x) = 1 − erf(x)`.
///
/// Evaluated directly (not as `1 − erf`) for `|x| ≥ 0.5`, preserving relative
/// accuracy in the tail where `erf(x) → ±1`.
///
/// # Arguments
///
/// * `x` — the argument; any `f64`. `NaN` propagates; `+∞ → 0`, `−∞ → 2`.
///
/// # Returns
///
/// `erfc(x)` in `(0, 2)`.
///
/// # Examples
///
/// ```
/// use stats_claw::special::erfc;
/// // Deep tail stays accurate: erfc(5) ≈ 1.537e-12.
/// assert!((erfc(5.0) - 1.537_459_794_428_035e-12).abs() < 1e-24);
/// ```
#[must_use]
pub fn erfc(x: f64) -> f64 {
    let ax = x.abs();
    if ax < SMALL {
        return 1.0 - erf_small(x);
    }
    let c = erfc_ge_half(ax);
    if x < 0.0 { 2.0 - c } else { c }
}

/// Computes the natural log of the complementary error function `ln(erfc(x))`.
///
/// The linear [`erfc`] underflows to `0.0` once `x ≳ 27`, after which its log is
/// `-∞`. This routine instead takes the log analytically: in the mid/tail Cody
/// regions `erfc(ax) = exp(−ax²) · R(ax)`, so `ln erfc(ax) = −ax² + ln R(ax)` is
/// evaluated without ever forming the underflowing product. This is the
/// `scipy.stats.norm.logsf` building block (`norm.logsf(x) = ln erfc(x/√2) − ln 2`)
/// and stays finite and accurate (~1e-12 relative) over the whole real line.
///
/// # Arguments
///
/// * `x` — the argument; any `f64`. `NaN` propagates; `+∞ → −∞`, `−∞ → ln 2`.
///
/// # Returns
///
/// `ln(erfc(x)) ∈ (−∞, ln 2]`.
///
/// # Examples
///
/// ```
/// use stats_claw::special::ln_erfc;
/// // Body agrees with the log of the linear erfc.
/// assert!((ln_erfc(1.0) - (-1.849_605_509_933_248)).abs() < 1e-12);
/// // Deep tail stays finite where erfc(40) has underflowed to 0.
/// assert!(ln_erfc(40.0).is_finite());
/// ```
#[must_use]
pub fn ln_erfc(x: f64) -> f64 {
    let ax = x.abs();
    if ax < SMALL {
        // No cancellation or underflow near 0; the direct log is accurate.
        return (1.0 - erf_small(x)).ln();
    }
    if x < 0.0 {
        // erfc(x) = 2 − erfc(|x|) ∈ [1, 2); never underflows, direct log is fine.
        return (2.0 - erfc_ge_half(ax)).ln();
    }
    ln_erfc_ge_half(ax)
}

/// Computes `ln(erfc(ax))` for `ax = |x| ≥ 0.46875` (Cody regions 2 and 3),
/// folding the `exp(−ax²)` weight into the log analytically so the deep tail
/// stays finite.
///
/// # Arguments
///
/// * `ax` — the non-negative magnitude `|x|`, at least `SMALL`.
///
/// # Returns
///
/// `ln(erfc(ax))`, finite for all finite `ax`.
fn ln_erfc_ge_half(ax: f64) -> f64 {
    if ax.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let log_weight = ln_exp_weight(ax);
    let ratio = if ax < LARGE {
        cody_ratio(ax, &C_ERFC, &D_ERFC)
    } else {
        let z = 1.0 / (ax * ax);
        let r = z * cody_ratio(z, &P_ASYMP, &Q_ASYMP);
        (INV_SQRT_PI - r) / ax
    };
    log_weight + ratio.ln()
}

/// Computes `ln(exp(−ax²)) = −ax²` via Cody's 16-bit split so the dominant term is
/// formed without rounding `ax²` first — the log counterpart of [`exp_weight`].
///
/// # Arguments
///
/// * `ax` — a non-negative magnitude.
///
/// # Returns
///
/// `−ax²`, evaluated as `−t² − (ax−t)(ax+t)` with `t` the 16-bit truncation of `ax`.
fn ln_exp_weight(ax: f64) -> f64 {
    let t = trunc_16(ax);
    let d = ax - t;
    (-d).mul_add(ax + t, -t * t)
}

/// Direct `erf` rational form for `|x| < 0.5` (Cody region 1).
///
/// # Arguments
///
/// * `x` — argument with `|x| < 0.5`.
///
/// # Returns
///
/// `erf(x)`.
fn erf_small(x: f64) -> f64 {
    let z = x * x;
    x * cody_ratio(z, &A_ERF, &B_ERF)
}

/// Evaluates one Cody rational `R(v) = N(v) / D(v)` at `v`.
///
/// Mirrors the CALERF Horner recurrence exactly: the numerator is seeded with
/// its **last** coefficient (`nums[len-1] · v`), the remaining numerator
/// coefficients (all but the last two) are folded `(acc + c) · v`, and the
/// final numerator coefficient (`nums[len-2]`) is added without a trailing
/// multiply; the denominator runs the same recurrence seeded from `v` over all
/// but its last coefficient, which is added last. The denominator array has
/// exactly one fewer element than the numerator array, so `dens` pairs with the
/// folded numerator coefficients.
///
/// # Arguments
///
/// * `v` — the evaluation point (`x²` in region 1, `|x|` in region 2,
///   `1/x²` in region 3).
/// * `nums` — numerator coefficients in CALERF order (seed last).
/// * `dens` — denominator coefficients in CALERF order; `dens.len() + 1 == nums.len()`.
///
/// # Returns
///
/// `N(v) / D(v)`.
fn cody_ratio(v: f64, nums: &[f64], dens: &[f64]) -> f64 {
    let empty: &[f64] = &[];
    let (seed, rest) = nums.split_last().map_or((0.0, empty), |(s, r)| (*s, r));
    let (num_last, num_mid) = rest.split_last().map_or((0.0, empty), |(l, m)| (*l, m));
    let (den_last, den_mid) = dens.split_last().map_or((0.0, empty), |(l, m)| (*l, m));

    let mut xnum = seed * v;
    for &c in num_mid {
        xnum = (xnum + c) * v;
    }
    let mut xden = v;
    for &c in den_mid {
        xden = (xden + c) * v;
    }
    (xnum + num_last) / (xden + den_last)
}

/// Computes `erfc(ax)` for `ax = |x| ≥ 0.5` via region 2 or 3.
///
/// # Arguments
///
/// * `ax` — the non-negative magnitude `|x|`, at least `0.5`.
///
/// # Returns
///
/// `erfc(ax)` in `(0, 1]`.
fn erfc_ge_half(ax: f64) -> f64 {
    if ax >= XBIG {
        // erfc has underflowed to 0; this also catches +∞, where the exp split
        // would otherwise form ∞ − ∞ = NaN. (NaN fails this test and every
        // branch below, so it propagates through `erfc_tail` unchanged.)
        return 0.0;
    }
    if ax < LARGE {
        erfc_mid(ax)
    } else {
        erfc_tail(ax)
    }
}

/// `erfc` rational form for `0.46875 < ax < 4` (Cody region 2).
fn erfc_mid(ax: f64) -> f64 {
    exp_weight(ax) * cody_ratio(ax, &C_ERFC, &D_ERFC)
}

/// Asymptotic `erfc` rational form for `ax ≥ 4` (Cody region 3).
fn erfc_tail(ax: f64) -> f64 {
    let z = 1.0 / (ax * ax);
    let r = z * cody_ratio(z, &P_ASYMP, &Q_ASYMP);
    let scaled = (INV_SQRT_PI - r) / ax;
    exp_weight(ax) * scaled
}

/// Computes `exp(−ax²)` via Cody's 16-bit-split trick, which keeps the dominant
/// part of the exponent exactly representable so the `erfc` tail avoids losing
/// low-order bits to rounding of `ax²`.
///
/// `exp(−ax²) = exp(−t²) · exp(−(ax−t)(ax+t))` where `t` is `ax` truncated to a
/// multiple of `1/16`.
///
/// # Arguments
///
/// * `ax` — a non-negative magnitude.
///
/// # Returns
///
/// `exp(−ax²)`.
fn exp_weight(ax: f64) -> f64 {
    let t = trunc_16(ax);
    let d = ax - t;
    (-t * t).exp() * (-d * (ax + t)).exp()
}

/// Truncates `ax` to 16 binary places (multiple of `1/16`), giving a value whose
/// square has an exactly representable leading part — Cody's trick to evaluate
/// `exp(-ax²)` without losing low-order bits to rounding.
///
/// # Arguments
///
/// * `ax` — a non-negative magnitude.
///
/// # Returns
///
/// `floor(16·ax)/16`, the 16-bit-truncated value `≤ ax`.
fn trunc_16(ax: f64) -> f64 {
    (ax * 16.0).floor() / 16.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `erf(1)` matches the high-precision reference to 1e-15.
    #[test]
    fn erf_one_high_precision() {
        assert!(
            (erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-15,
            "erf(1) was {}",
            erf(1.0)
        );
    }

    /// `erf` is odd to machine precision.
    #[test]
    fn erf_is_odd() {
        for &x in &[0.3, 0.7, 1.5, 3.0, 6.0] {
            assert!((erf(-x) + erf(x)).abs() < 1e-15, "erf not odd at {x}");
        }
    }

    /// `erf(0.5)` (region boundary) matches the reference.
    #[test]
    fn erf_half_boundary() {
        assert!(
            (erf(0.5) - 0.520_499_877_813_046_5).abs() < 1e-15,
            "erf(0.5) was {}",
            erf(0.5)
        );
    }

    /// `erf(2)` (region-2 interior) matches the reference.
    #[test]
    fn erf_two() {
        assert!(
            (erf(2.0) - 0.995_322_265_018_952_7).abs() < 1e-15,
            "erf(2) was {}",
            erf(2.0)
        );
    }

    /// `erfc` in the deep tail keeps relative accuracy (the whole point of the
    /// direct `erfc` form): `erfc(5) ≈ 1.5375e-12`.
    #[test]
    fn erfc_deep_tail() {
        let got = erfc(5.0);
        let want = 1.537_459_794_428_035e-12;
        assert!(
            ((got - want) / want).abs() < 1e-12,
            "erfc(5) rel error too large: got {got}, want {want}"
        );
    }

    /// `erfc(4)` (region 2→3 boundary) matches the reference.
    #[test]
    fn erfc_large_boundary() {
        let got = erfc(4.0);
        let want = 1.541_725_790_028_002e-8;
        assert!(
            ((got - want) / want).abs() < 1e-12,
            "erfc(4) rel error too large: got {got}, want {want}"
        );
    }

    /// `erfc(x) = 1 − erf(x)` holds for moderate `x` (where no cancellation).
    #[test]
    fn erfc_complements_erf() {
        for &x in &[-0.3, 0.0, 0.4, 1.0] {
            assert!(
                (erfc(x) - (1.0 - erf(x))).abs() < 1e-15,
                "erfc != 1-erf at {x}"
            );
        }
    }

    /// Endpoints: `erf(±∞) = ±1`, `erfc(+∞)=0`, `erfc(−∞)=2`.
    #[test]
    fn infinities() {
        assert!((erf(f64::INFINITY) - 1.0).abs() < 1e-15);
        assert!((erf(f64::NEG_INFINITY) + 1.0).abs() < 1e-15);
        assert!(erfc(f64::INFINITY).abs() < 1e-300);
        assert!((erfc(f64::NEG_INFINITY) - 2.0).abs() < 1e-15);
    }

    /// `ln_erfc` matches `erfc(x).ln()` where `erfc` has not underflowed.
    #[test]
    fn ln_erfc_matches_log_of_erfc_in_body() {
        for &x in &[-1.0, 0.0, 0.5, 1.0, 2.0, 4.0] {
            let want = erfc(x).ln();
            let got = ln_erfc(x);
            assert!(
                ((got - want) / want.abs().max(1.0)).abs() < 1e-12,
                "ln_erfc({x}) = {got}, want {want}"
            );
        }
    }

    /// `ln_erfc` stays finite and accurate deep in the tail where `erfc` itself
    /// underflows to `0.0` (the whole point of the log-space path).
    #[test]
    fn ln_erfc_finite_in_deep_tail() {
        // Precondition: the linear erfc has underflowed to exactly 0.0 at x = 40,
        // so its log would be −∞; bit-compare avoids the float_cmp lint.
        assert_eq!(erfc(40.0).to_bits(), 0.0_f64.to_bits());
        let got = ln_erfc(40.0);
        // scipy: special.log_ndtr(-40*sqrt(2)) + ln(2) = ln(erfc(40)).
        let want = -1_604.261_556_653_274_3;
        assert!(got.is_finite(), "ln_erfc(40) was {got}");
        assert!(
            ((got - want) / want).abs() < 1e-9,
            "ln_erfc(40) rel error too large: got {got}, want {want}"
        );
    }
}
