//! Native-SIMD batch hot paths for the Normal distribution.
//!
//! The scalar `pdf`/`cdf` loops are at an architectural disadvantage against
//! `scipy.stats.norm`, which dispatches vectorized SIMD C ufuncs over the whole
//! array. This module closes that gap with hand-written `core::arch` intrinsics —
//! std-only, *not* a third-party dependency, so the zero-runtime-dependency NFR
//! still holds — selected at **runtime** for both Linux target arches:
//!
//! * `aarch64` + `neon` — two `f64` lanes per vector (`float64x2_t`).
//! * `x86_64` + `avx2`/`fma` — four `f64` lanes per vector (`__m256d`).
//! * any other target, or a CPU lacking the feature — the scalar fallback
//!   (`f64::exp` for the pdf; the Cody-`erf` path for the cdf).
//!
//! Only the `pdf` `exp` loop is genuinely SIMD-friendly and is the path that
//! benefits most; the `cdf` keeps the branchy scalar Cody `erf` in the lane loop
//! (vectorizing `erf` to ≤1e-10 is not worth the accuracy risk) but still
//! benefits from the dense batch layout. Sampling is left scalar (a cached
//! Box–Muller generator cannot match numpy's vectorized ziggurat).
//!
//! ## Accuracy
//!
//! The vectorized `exp` uses Cephes-style range reduction
//! `exp(x) = 2^k · exp(r)`, `k = round(x / ln 2)`, `r = x − k·ln 2`, with a
//! polynomial on the reduced `r ∈ [−ln2/2, ln2/2]` and the `2^k` factor built by
//! writing `k` into the IEEE-754 exponent field. It is accurate to ≤ 1e-12
//! relative, so the Normal `pdf` stays well inside the distribution
//! tolerance (abs ≤ 1e-10). The scalar fallback ([`simd_exp_scalar`]) delegates
//! to `f64::exp`, so the kernel tests validate the vectorized `exp` directly
//! against the standard library to ≤ 1e-12.
//!
//! ## Safety
//!
//! `unsafe_code` is `deny` workspace-wide; this module narrowly relaxes it to
//! `allow` (below) because `core::arch` intrinsics are `unsafe` by definition.
//! Every `unsafe` item carries a `// SAFETY:` line (enforced by `tests/gates.rs`):
//! each `#[target_feature]` kernel is only ever called from
//! [`normal_pdf_into`] *after* the matching `is_*_feature_detected!` check, so the
//! instructions are always available on the running CPU; all vector loads/stores
//! operate on `chunks_exact`-sized windows that are in-bounds by construction.

// SAFETY: see the module-level "Safety" section — this allow is the deliberate,
// narrowly-scoped relaxation of the workspace `unsafe_code = "deny"` for the
// `core::arch` SIMD kernels; every `unsafe` item below is justified inline.
#![allow(unsafe_code)]

use crate::distributions::NormalDistribution;

/// `ln(2)`, used in the range reduction `r = x − k·ln2`.
const LN2: f64 = std::f64::consts::LN_2;
/// `1 / ln(2)`, used to compute the integer power-of-two exponent `k`.
const LOG2_E: f64 = std::f64::consts::LOG2_E;
/// `1/√(2π)`, the standard-normal density normalization constant.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Taylor coefficients `1/n!` for `exp(r)` on the reduced range
/// `r ∈ [−ln2/2, ln2/2]`, highest degree first down to `1/2!`. With `|r| ≤ 0.347`
/// a degree-11 series is accurate to ≤ 1e-13. The series is evaluated by Horner
/// to `r²·poly + r + 1`, so these are the coefficients for the `r²·poly` part
/// (i.e. `1/11! … 1/2!`); the `+ r + 1` head is added explicitly.
const EXP_TAYLOR: [f64; 10] = [
    1.0 / 39_916_800.0, // 1/11!
    1.0 / 3_628_800.0,  // 1/10!
    1.0 / 362_880.0,    // 1/9!
    1.0 / 40_320.0,     // 1/8!
    1.0 / 5_040.0,      // 1/7!
    1.0 / 720.0,        // 1/6!
    1.0 / 120.0,        // 1/5!
    1.0 / 24.0,         // 1/4!
    1.0 / 6.0,          // 1/3!
    1.0 / 2.0,          // 1/2!
];
/// IEEE-754 double exponent bias.
const EXP_BIAS: i64 = 1023;
/// Bit position of the IEEE-754 double exponent field (also the `vshlq`/`slli`
/// const-generic shift amount, hence `i32`).
const EXP_SHIFT: i32 = 52;

/// Evaluates `exp(x)` for the scalar fallback path.
///
/// This is the correctness fallback used when no SIMD feature is available and
/// the per-lane correctness oracle the kernel tests compare against. It delegates
/// to the standard-library `f64::exp` rather than re-deriving the Cephes
/// reduction in scalar code: the scalar path is **not** the performance-critical
/// hot loop (the NEON/AVX2 kernels carry the batch workload), so its only job is
/// to be exactly
/// correct and `as`-free. The SIMD lanes implement the vectorized Cephes `exp`
/// ([`EXP_TAYLOR`]) directly and are validated against this oracle to ≤ 1e-12 by
/// the kernel tests.
///
/// # Arguments
///
/// * `x` — any finite `f64`; for the Normal pdf the argument is `−½z² ≤ 0`.
///
/// # Returns
///
/// `e^x`.
fn simd_exp_scalar(x: f64) -> f64 {
    x.exp()
}

/// Fills `out[i]` with the Normal `pdf` of `xs[i]`, using the fastest available
/// runtime path (NEON / AVX2 / scalar).
///
/// # Arguments
///
/// * `dist` — the Normal distribution whose density is evaluated.
/// * `xs` — the evaluation grid.
/// * `out` — output buffer; only the leading `min(xs.len(), out.len())` entries
///   are written if the lengths differ.
pub(super) fn normal_pdf_into(dist: &NormalDistribution, xs: &[f64], out: &mut [f64]) {
    let inv_sigma = 1.0 / dist.standard_deviation;
    let norm = INV_SQRT_2PI * inv_sigma;
    let mean = dist.mean;

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: `neon` is confirmed present on this CPU by the runtime check
            // immediately above, which is `pdf_neon`'s only safety precondition.
            unsafe {
                pdf_neon(mean, inv_sigma, norm, xs, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: `avx2` and `fma` are confirmed present by the runtime checks
            // immediately above, which is `pdf_avx2`'s only safety precondition.
            unsafe {
                pdf_avx2(mean, inv_sigma, norm, xs, out);
            }
            return;
        }
    }
    pdf_scalar(mean, inv_sigma, norm, xs, out);
}

/// Scalar fallback for [`normal_pdf_into`]; also the per-lane correctness oracle.
///
/// # Arguments
///
/// * `mean` — the distribution mean.
/// * `inv_sigma` — `1/σ`, hoisted out of the loop.
/// * `norm` — `1/(σ√(2π))`, the density scale, hoisted out of the loop.
/// * `xs` — the evaluation grid.
/// * `out` — output buffer written in lockstep with `xs`.
fn pdf_scalar(mean: f64, inv_sigma: f64, norm: f64, xs: &[f64], out: &mut [f64]) {
    for (o, &x) in out.iter_mut().zip(xs) {
        let z = (x - mean) * inv_sigma;
        *o = simd_exp_scalar(-0.5 * z * z) * norm;
    }
}

/// Fills `out[i]` with the Normal `cdf` of `xs[i]`.
///
/// The `erf` evaluation stays scalar (Cody rational, kept for its ≤1e-16 tail
/// accuracy), so this is a batch-layout convenience rather than a vectorized
/// kernel; the vectorized hot path is `pdf`, not `cdf`.
///
/// # Arguments
///
/// * `dist` — the Normal distribution whose CDF is evaluated.
/// * `xs` — the evaluation grid.
/// * `out` — output buffer; written in lockstep with `xs`.
pub(super) fn normal_cdf_into(dist: &NormalDistribution, xs: &[f64], out: &mut [f64]) {
    let inv_scale = 1.0 / (dist.standard_deviation * std::f64::consts::SQRT_2);
    let mean = dist.mean;
    for (o, &x) in out.iter_mut().zip(xs) {
        let z = (x - mean) * inv_scale;
        *o = 0.5 * (1.0 + crate::special::erf(z));
    }
}

/// NEON (`aarch64`) two-lane `f64` Normal-pdf kernel.
///
/// Processes the grid two elements at a time with a vectorized Cephes `exp`,
/// then handles the `< 2`-element tail with the scalar path.
///
/// # Arguments
///
/// * `mean`, `inv_sigma`, `norm` — the hoisted constants (see [`pdf_scalar`]).
/// * `xs` — the evaluation grid.
/// * `out` — output buffer written in lockstep with `xs`.
///
/// # Safety
///
/// The caller must ensure the `neon` target feature is available on the running
/// CPU (checked in [`normal_pdf_into`]). All vector loads/stores are over 2-wide
/// windows bounded by `n - (n % 2)` and are therefore in-bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: callers (only `normal_pdf_into`) guarantee `neon` is available; the
// body's loads/stores stay within `n - (n % 2)`-bounded 2-wide windows.
unsafe fn pdf_neon(mean: f64, inv_sigma: f64, norm: f64, xs: &[f64], out: &mut [f64]) {
    use std::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};

    let lanes = 2;
    let n = xs.len().min(out.len());
    let body = n - (n % lanes);

    let vmean = vdupq_n_f64(mean);
    let vinv = vdupq_n_f64(inv_sigma);
    let vnorm = vdupq_n_f64(norm);
    let vneg_half = vdupq_n_f64(-0.5);

    let mut i = 0;
    while i < body {
        // SAFETY: `i + 2 <= body <= n <= xs.len()`, so the 2-wide load/store are
        // in-bounds; `neon` is guaranteed by the caller's contract.
        let vx = vld1q_f64(xs.as_ptr().add(i));
        let z = vmulq_f64(vsubq_f64(vx, vmean), vinv);
        let arg = vmulq_f64(vmulq_f64(z, z), vneg_half);
        let e = exp_neon(arg);
        let res = vmulq_f64(e, vnorm);
        vst1q_f64(out.as_mut_ptr().add(i), res);
        i += lanes;
    }

    if body < n {
        pdf_scalar(
            mean,
            inv_sigma,
            norm,
            xs.get(body..n).unwrap_or(&[]),
            out.get_mut(body..n).unwrap_or(&mut []),
        );
    }
}

/// Vectorized Cephes `exp` for a NEON `float64x2_t`.
///
/// # Arguments
///
/// * `x` — two packed `f64` exponents.
///
/// # Returns
///
/// The packed `e^x`.
///
/// # Safety
///
/// Requires the `neon` target feature (guaranteed by the calling kernel). Does
/// no memory access — register math only.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: callers guarantee `neon`; the body does register-only math, no memory
// access, so it is sound whenever the feature is present.
unsafe fn exp_neon(x: std::arch::aarch64::float64x2_t) -> std::arch::aarch64::float64x2_t {
    use std::arch::aarch64::{
        vaddq_f64, vaddq_s64, vcvtnq_s64_f64, vcvtq_f64_s64, vdupq_n_f64, vdupq_n_s64, vfmaq_f64,
        vmulq_f64, vreinterpretq_f64_s64, vshlq_n_s64, vsubq_f64,
    };

    let log2e = vdupq_n_f64(LOG2_E);
    let neg_ln2 = vdupq_n_f64(-LN2);

    // k = round(x / ln2) as integer lanes; kf its float.
    let ki = vcvtnq_s64_f64(vmulq_f64(x, log2e));
    let kf = vcvtq_f64_s64(ki);
    // r = x - k*ln2  (fused multiply-add: x + kf*(-ln2)).
    let r = vfmaq_f64(x, kf, neg_ln2);

    // Horner over the Taylor coefficients; exp(r) = 1 + r + r²·p.
    let mut p = vdupq_n_f64(EXP_TAYLOR[0]);
    for &c in &EXP_TAYLOR[1..] {
        p = vfmaq_f64(vdupq_n_f64(c), p, r);
    }
    let r2 = vmulq_f64(r, r);
    let _ = vsubq_f64; // import kept for symmetry with the scalar derivation
    let exp_r = vaddq_f64(vfmaq_f64(r, r2, p), vdupq_n_f64(1.0));

    // 2^k: (k + 1023) << 52 reinterpreted as f64, then multiply.
    let biased = vaddq_s64(ki, vdupq_n_s64(EXP_BIAS));
    let pow2 = vreinterpretq_f64_s64(vshlq_n_s64::<EXP_SHIFT>(biased));
    vmulq_f64(exp_r, pow2)
}

/// AVX2 (`x86_64`) four-lane `f64` Normal-pdf kernel.
///
/// # Arguments
///
/// * `mean`, `inv_sigma`, `norm` — the hoisted constants (see [`pdf_scalar`]).
/// * `xs` — the evaluation grid.
/// * `out` — output buffer written in lockstep with `xs`.
///
/// # Safety
///
/// The caller must ensure `avx2` and `fma` are available (checked in
/// [`normal_pdf_into`]). All vector loads/stores are over 4-wide windows bounded
/// by `n - (n % 4)` and are in-bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: callers (only `normal_pdf_into`) guarantee `avx2`+`fma`; the body's
// loads/stores stay within `n - (n % 4)`-bounded 4-wide windows.
unsafe fn pdf_avx2(mean: f64, inv_sigma: f64, norm: f64, xs: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::{
        _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };

    let lanes = 4;
    let n = xs.len().min(out.len());
    let body = n - (n % lanes);

    let vmean = _mm256_set1_pd(mean);
    let vinv = _mm256_set1_pd(inv_sigma);
    let vnorm = _mm256_set1_pd(norm);
    let vneg_half = _mm256_set1_pd(-0.5);

    let mut i = 0;
    while i < body {
        // SAFETY: `i + 4 <= body <= n <= xs.len()`, so the 4-wide unaligned
        // load/store are in-bounds; `avx2`+`fma` guaranteed by the caller.
        unsafe {
            let vx = _mm256_loadu_pd(xs.as_ptr().add(i));
            let z = _mm256_mul_pd(_mm256_sub_pd(vx, vmean), vinv);
            let arg = _mm256_mul_pd(_mm256_mul_pd(z, z), vneg_half);
            let e = exp_avx2(arg);
            let res = _mm256_mul_pd(e, vnorm);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), res);
        }
        i += lanes;
    }

    if body < n {
        pdf_scalar(
            mean,
            inv_sigma,
            norm,
            xs.get(body..n).unwrap_or(&[]),
            out.get_mut(body..n).unwrap_or(&mut []),
        );
    }
}

/// Vectorized Cephes `exp` for an AVX2 `__m256d` (four packed `f64`).
///
/// # Arguments
///
/// * `x` — four packed `f64` exponents.
///
/// # Returns
///
/// The packed `e^x`.
///
/// # Safety
///
/// Requires `avx2`+`fma` (guaranteed by the calling kernel). Register math only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: callers guarantee `avx2`+`fma`; the body does register-only math, no
// memory access, so it is sound whenever the features are present.
unsafe fn exp_avx2(x: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{
        _mm256_add_epi64, _mm256_add_pd, _mm256_castsi256_pd, _mm256_cvtepi32_epi64,
        _mm256_cvtpd_epi32, _mm256_fmadd_pd, _mm256_mul_pd, _mm256_round_pd, _mm256_set1_epi64x,
        _mm256_set1_pd, _mm256_slli_epi64,
    };
    // Round-to-nearest, suppress exceptions (`_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC`).
    const ROUND_NEAREST: i32 = 0x00;

    let log2e = _mm256_set1_pd(LOG2_E);
    let neg_ln2 = _mm256_set1_pd(-LN2);

    // kf = round(x / ln2) as f64; r = x - kf*ln2 (fused).
    let kf = _mm256_round_pd::<ROUND_NEAREST>(_mm256_mul_pd(x, log2e));
    let r = _mm256_fmadd_pd(kf, neg_ln2, x);

    let mut p = _mm256_set1_pd(EXP_TAYLOR[0]);
    for &c in &EXP_TAYLOR[1..] {
        p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(c));
    }
    let r2 = _mm256_mul_pd(r, r);
    let exp_r = _mm256_add_pd(_mm256_fmadd_pd(r2, p, r), _mm256_set1_pd(1.0));

    // 2^k: widen kf→i32→i64, bias by 1023, shift into the exponent field.
    let ki32 = _mm256_cvtpd_epi32(kf);
    let ki64 = _mm256_cvtepi32_epi64(ki32);
    let biased = _mm256_add_epi64(ki64, _mm256_set1_epi64x(EXP_BIAS));
    let pow2 = _mm256_castsi256_pd(_mm256_slli_epi64::<EXP_SHIFT>(biased));
    _mm256_mul_pd(exp_r, pow2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Pdf;

    /// The batch path (NEON/AVX2 vectorized `exp` when available, else scalar)
    /// matches an **independent** standard-library Gaussian-density reference
    /// `1/(σ√2π)·exp(−z²/2)` to ≤ 1e-12, so the hand-written vectorized `exp` is
    /// validated against `f64::exp`, not just against the crate's own scalar path.
    #[test]
    fn batch_pdf_matches_stdlib_gaussian() {
        let dist = NormalDistribution {
            mean: 0.0,
            standard_deviation: 1.0,
            ..Default::default()
        };
        // 251 (prime) exercises the SIMD tail on both 2-wide and 4-wide kernels.
        let xs: Vec<f64> = (0..251).map(|i| f64::from(i).mul_add(0.04, -5.0)).collect();
        let mut out = vec![0.0; xs.len()];
        normal_pdf_into(&dist, &xs, &mut out);
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        for (o, &x) in out.iter().zip(&xs) {
            let want = inv_sqrt_2pi * (-0.5 * x * x).exp();
            let rel = ((o - want) / want).abs();
            assert!(
                rel < 1e-12,
                "pdf({x}) rel error {rel}: got {o}, want {want}"
            );
        }
    }

    /// Batch `normal_pdf_into` equals the per-element scalar `pdf`, including a
    /// non-lane-multiple length so the tail handling is exercised.
    #[test]
    fn batch_pdf_matches_scalar_pdf() {
        let dist = NormalDistribution {
            mean: 0.3,
            standard_deviation: 1.7,
            ..Default::default()
        };
        // 103 is prime — not a multiple of 2 (NEON) or 4 (AVX2): exercises the tail.
        let xs: Vec<f64> = (0..103).map(|i| f64::from(i).mul_add(0.1, -5.0)).collect();
        let mut out = vec![0.0; xs.len()];
        normal_pdf_into(&dist, &xs, &mut out);
        for (o, &x) in out.iter().zip(&xs) {
            let want = dist.pdf(x);
            assert!(
                (o - want).abs() < 1e-12,
                "batch pdf {o} != scalar pdf {want} at x={x}"
            );
        }
    }

    /// Batch `normal_cdf_into` equals the per-element scalar `cdf`.
    #[test]
    fn batch_cdf_matches_scalar_cdf() {
        use crate::distributions::Cdf;
        let dist = NormalDistribution {
            mean: -0.5,
            standard_deviation: 2.0,
            ..Default::default()
        };
        let xs: Vec<f64> = (0..97).map(|i| f64::from(i).mul_add(0.1, -5.0)).collect();
        let mut out = vec![0.0; xs.len()];
        normal_cdf_into(&dist, &xs, &mut out);
        for (o, &x) in out.iter().zip(&xs) {
            let want = dist.cdf(x);
            assert!(
                (o - want).abs() < 1e-12,
                "batch cdf {o} != scalar cdf {want} at x={x}"
            );
        }
    }
}
