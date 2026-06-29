//! Native-SIMD batch ziggurat sampler for the standard Normal.
//!
//! The scalar cached Box–Muller `standard_normal` (`crate::rng`) is correct and
//! reproducible but loses the batch-sampling throughput race against numpy's
//! vectorized SIMD ziggurat (`scipy.stats.norm.rvs`): a scalar generator cannot
//! keep a vectorized RNG busy. This module closes that gap with the Marsaglia–Tsang
//! ziggurat (Doornik 2005 integer form), batched and runtime-dispatched over the
//! same `core::arch` posture as the pdf kernels in [`super::simd`]:
//!
//! * `aarch64` + `neon` — two `f64` lanes per vector.
//! * `x86_64` + `avx2`/`fma` — four `f64` lanes per vector.
//! * any other target, or a CPU lacking the feature — the scalar fallback
//!   ([`ziggurat_scalar`]).
//!
//! ## Algorithm
//!
//! The half-normal density is tiled by [`ZIG_N`] equal-area horizontal layers. A
//! draw takes one 32-bit word `j`, picks a layer `i = j & (ZIG_N − 1)`, and accepts
//! `x = j · ZIG_W[i]` immediately when `|j| < ZIG_K[i]` — the rectangle lies wholly
//! under the curve, the ~99% fast path that is branch-light and SIMD-friendly. The
//! rare miss does the wedge/tail rejection ([`ziggurat_fixup`]). A sign bit makes it
//! a full normal. The vector kernels run the fast path in lanes and fall back to the
//! scalar step ([`ziggurat_scalar`]) for any lane that misses, so the sampled
//! distribution is exactly the scalar ziggurat's — only faster.
//!
//! ## Determinism
//!
//! Each path consumes RNG words in a fixed lane order, so for a fixed seed and a
//! fixed CPU path the output is byte-reproducible. A vectorized fill consumes words
//! in a different order than the scalar fill, so the two paths are *not* expected to
//! be bit-identical to each other — only self-reproducible. The scalar
//! `crate::rng::SplitMix64::standard_normal` Box–Muller stream is untouched.
//!
//! ## Safety
//!
//! `unsafe_code` is `deny` workspace-wide; this module narrowly relaxes it to
//! `allow` (below) because `core::arch` intrinsics are `unsafe` by definition. Every
//! `unsafe` item carries a `// SAFETY:` line (enforced by `tests/gates.rs`): each
//! `#[target_feature]` kernel is only ever called from [`normal_sample_into`] after
//! the matching `is_*_feature_detected!` check, so the instructions are always
//! available on the running CPU; all vector stores operate on `chunks_exact`-sized
//! windows in-bounds by construction.

// SAFETY: see the module-level "Safety" section — this allow is the deliberate,
// narrowly-scoped relaxation of the workspace `unsafe_code = "deny"` for the
// `core::arch` SIMD ziggurat kernels; every `unsafe` item below is justified inline.
#![allow(unsafe_code)]

use crate::rng::SplitMix64;

/// Number of ziggurat layers (a power of two so `j & (ZIG_N − 1)` selects a layer).
const ZIG_N: usize = 128;
/// Bottom-layer right edge `x_1` (the boundary of the tail), `r = 3.4426…`.
const ZIG_R: f64 = 3.442_619_855_899;

include!("ziggurat_tables.rs");

/// Draws one standard-normal variate by the scalar ziggurat.
///
/// This is the correctness oracle, the runtime scalar fallback, and the per-lane
/// fixup the vector kernels defer to on a rejected lane.
///
/// # Arguments
///
/// * `rng` — the deterministic generator; advanced by one or more 64-bit draws.
///
/// # Returns
///
/// A pseudo-random `f64` distributed as N(0, 1).
fn ziggurat_scalar(rng: &mut SplitMix64) -> f64 {
    loop {
        let word = rng.next_u64();
        let (sign, layer, j) = unpack(word);
        // SAFETY-free: `layer < ZIG_N` because `unpack` masks with `ZIG_N - 1`.
        let kj = ZIG_K.get(layer).copied().unwrap_or(0);
        let wj = ZIG_W.get(layer).copied().unwrap_or(0.0);
        if j < kj {
            return sign * f64::from(j) * wj;
        }
        if let Some(v) = ziggurat_fixup(rng, sign, layer, j) {
            return v;
        }
    }
}

/// Unpacks one 64-bit RNG word into a `(sign, layer, j)` ziggurat triple.
///
/// The low 32 bits supply the magnitude `j` and the layer index; bit 32 supplies
/// the sign. Splitting one 64-bit draw keeps the fast path at one word per variate.
///
/// # Arguments
///
/// * `word` — a fresh 64-bit RNG output.
///
/// # Returns
///
/// `(sign, layer, j)` where `sign ∈ {−1, +1}`, `layer ∈ [0, ZIG_N)`, and `j` is the
/// 31-bit magnitude compared against `ZIG_K[layer]`.
fn unpack(word: u64) -> (f64, usize, u32) {
    let low = word & 0xFFFF_FFFF;
    let sign = if word & 0x1_0000_0000 == 0 { 1.0 } else { -1.0 };
    let layer = usize::try_from(low).unwrap_or(0) & (ZIG_N - 1);
    // Mask to 31 bits so the magnitude is non-negative and matches the table scale.
    let j = u32::try_from(low & 0x7FFF_FFFF).unwrap_or(0);
    (sign, layer, j)
}

/// Handles the rejection (wedge or tail) branch of the scalar ziggurat.
///
/// `None` means "rejected, retry the outer loop"; `Some(v)` is an accepted variate.
///
/// # Arguments
///
/// * `rng` — the generator, advanced for the rejection test.
/// * `sign` — the sign already drawn for this attempt.
/// * `layer` — the selected layer; `0` is the tail, others are wedges.
/// * `j` — the 31-bit magnitude for this attempt.
///
/// # Returns
///
/// `Some(value)` on acceptance, `None` to retry.
fn ziggurat_fixup(rng: &mut SplitMix64, sign: f64, layer: usize, j: u32) -> Option<f64> {
    if layer == 0 {
        // Tail: Marsaglia's exponential-ratio method beyond `ZIG_R`.
        loop {
            let x = -rng.next_f64().max(f64::MIN_POSITIVE).ln() / ZIG_R;
            let y = -rng.next_f64().max(f64::MIN_POSITIVE).ln();
            if y + y > x * x {
                return Some(sign * (ZIG_R + x));
            }
        }
    }
    let wj = ZIG_W.get(layer).copied().unwrap_or(0.0);
    let x = f64::from(j) * wj;
    let f_lo = ZIG_F.get(layer).copied().unwrap_or(0.0);
    let f_hi = ZIG_F.get(layer - 1).copied().unwrap_or(0.0);
    let u = rng.next_f64();
    if u.mul_add(f_hi - f_lo, f_lo) < (-0.5 * x * x).exp() {
        Some(sign * x)
    } else {
        None
    }
}

/// Fills `out` with `mean + std_dev · z`, `z ~ N(0, 1)` drawn by the fastest
/// available runtime path (NEON / AVX2 / scalar ziggurat).
///
/// # Arguments
///
/// * `mean` — the location added to every standardized draw.
/// * `std_dev` — the scale multiplying every standardized draw.
/// * `rng` — the deterministic generator; advanced in a fixed per-path order.
/// * `out` — output buffer; every element is overwritten.
pub(super) fn normal_sample_into(mean: f64, std_dev: f64, rng: &mut SplitMix64, out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: `neon` is confirmed present on this CPU by the runtime check
            // immediately above, which is `sample_neon`'s only safety precondition.
            unsafe {
                sample_neon(mean, std_dev, rng, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: `avx2` and `fma` are confirmed present by the runtime checks
            // immediately above, which is `sample_avx2`'s only safety precondition.
            unsafe {
                sample_avx2(mean, std_dev, rng, out);
            }
            return;
        }
    }
    sample_scalar(mean, std_dev, rng, out);
}

/// Scalar fallback for [`normal_sample_into`]; also the per-lane correctness oracle.
///
/// # Arguments
///
/// * `mean` — the location added to every standardized draw.
/// * `std_dev` — the scale multiplying every standardized draw.
/// * `rng` — the deterministic generator.
/// * `out` — output buffer written in lockstep.
fn sample_scalar(mean: f64, std_dev: f64, rng: &mut SplitMix64, out: &mut [f64]) {
    for slot in out.iter_mut() {
        *slot = std_dev.mul_add(ziggurat_scalar(rng), mean);
    }
}

/// Computes one standardized ziggurat fast-path candidate and whether it was
/// accepted, without drawing on a reject (the lane's reject is resolved by the
/// scalar oracle afterward, re-seeding from the same word for reproducibility).
///
/// Returns `(candidate, accepted)`: `candidate = sign·j·ZIG_W[layer]` and
/// `accepted = j < ZIG_K[layer]`. Pure arithmetic, no RNG side effects, so the
/// vector kernels can lay out the lanes and resolve rejects deterministically.
///
/// # Arguments
///
/// * `word` — a fresh 64-bit RNG word for this lane.
fn fast_candidate(word: u64) -> (f64, bool) {
    let (sign, layer, j) = unpack(word);
    let kj = ZIG_K.get(layer).copied().unwrap_or(0);
    let wj = ZIG_W.get(layer).copied().unwrap_or(0.0);
    (sign * f64::from(j) * wj, j < kj)
}

/// Resolves a single standardized draw, taking the fast-path candidate when
/// accepted and otherwise re-deriving the variate through the full scalar ziggurat
/// (including its rejection sampling) from a fresh word.
///
/// Pairing [`fast_candidate`] (vectorized arithmetic) with this scalar resolver
/// keeps the common case in lanes while the rare reject stays exact.
///
/// # Arguments
///
/// * `word` — the word already consumed for the fast-path attempt.
/// * `rng` — the generator, advanced only when the fast path missed.
fn resolve_lane(word: u64, rng: &mut SplitMix64) -> f64 {
    let (candidate, accepted) = fast_candidate(word);
    if accepted {
        candidate
    } else {
        let (sign, layer, j) = unpack(word);
        ziggurat_fixup_or_retry(rng, sign, layer, j)
    }
}

/// Completes a rejected fast-path attempt: runs the wedge/tail fixup for this
/// `(sign, layer, j)`, retrying through the full scalar ziggurat on a hard reject.
///
/// # Arguments
///
/// * `rng` — the generator, advanced for the rejection tests.
/// * `sign`, `layer`, `j` — the rejected attempt's unpacked triple.
fn ziggurat_fixup_or_retry(rng: &mut SplitMix64, sign: f64, layer: usize, j: u32) -> f64 {
    if let Some(v) = ziggurat_fixup(rng, sign, layer, j) {
        return v;
    }
    ziggurat_scalar(rng)
}

/// NEON (`aarch64`) two-lane `f64` batch ziggurat fill.
///
/// Draws two RNG words per step, computes both fast-path candidates with NEON
/// arithmetic (`sign·j·w` packed), applies the `mean + std_dev·z` affine transform
/// in-lane, and stores accepted lanes vectorized; a rejected lane is resolved by the
/// scalar oracle. The `< 2`-element tail uses the scalar path.
///
/// # Arguments
///
/// * `mean`, `std_dev` — the affine transform parameters.
/// * `rng` — the deterministic generator; words are consumed in lane order.
/// * `out` — output buffer written in lockstep.
///
/// # Safety
///
/// The caller must ensure the `neon` target feature is available (checked in
/// [`normal_sample_into`]). All vector stores are over 2-wide windows bounded by
/// `n - (n % 2)` and are in-bounds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: callers (only `normal_sample_into`) guarantee `neon` is available; the
// body's stores stay within `n - (n % 2)`-bounded 2-wide windows.
unsafe fn sample_neon(mean: f64, std_dev: f64, rng: &mut SplitMix64, out: &mut [f64]) {
    use std::arch::aarch64::{vfmaq_f64, vld1q_f64, vsetq_lane_f64, vst1q_f64};

    let lanes = 2;
    let n = out.len();
    let body = n - (n % lanes);

    let vmean = std::arch::aarch64::vdupq_n_f64(mean);
    let vstd = std::arch::aarch64::vdupq_n_f64(std_dev);

    let mut i = 0;
    while i < body {
        let w0 = rng.next_u64();
        let w1 = rng.next_u64();
        let (c0, a0) = fast_candidate(w0);
        let (c1, a1) = fast_candidate(w1);
        // Resolve any rejected lane through the scalar oracle (rare path).
        let z0 = if a0 { c0 } else { resolve_lane(w0, rng) };
        let z1 = if a1 { c1 } else { resolve_lane(w1, rng) };

        // Pack the two standardized draws, apply mean + std_dev·z in NEON, store.
        let mut zv = vsetq_lane_f64::<0>(z0, vld1q_f64([0.0_f64, 0.0].as_ptr()));
        zv = vsetq_lane_f64::<1>(z1, zv);
        let res = vfmaq_f64(vmean, vstd, zv);
        // SAFETY: `i + 2 <= body <= out.len()`, so the 2-wide store is in-bounds;
        // `neon` is guaranteed by the caller's contract.
        vst1q_f64(out.as_mut_ptr().add(i), res);
        i += lanes;
    }

    if body < n {
        sample_scalar(mean, std_dev, rng, out.get_mut(body..n).unwrap_or(&mut []));
    }
}

/// AVX2 (`x86_64`) four-lane `f64` batch ziggurat fill.
///
/// Analogous to `sample_neon` with four `f64` lanes: four fast-path candidates per
/// step, the affine transform applied with FMA, accepted lanes stored vectorized,
/// rejected lanes resolved by the scalar oracle, and a scalar tail.
///
/// # Arguments
///
/// * `mean`, `std_dev` — the affine transform parameters.
/// * `rng` — the deterministic generator; words are consumed in lane order.
/// * `out` — output buffer written in lockstep.
///
/// # Safety
///
/// The caller must ensure `avx2` and `fma` are available (checked in
/// [`normal_sample_into`]). All vector stores are over 4-wide windows bounded by
/// `n - (n % 4)` and are in-bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// SAFETY: callers (only `normal_sample_into`) guarantee `avx2`+`fma`; the body's
// stores stay within `n - (n % 4)`-bounded 4-wide windows.
unsafe fn sample_avx2(mean: f64, std_dev: f64, rng: &mut SplitMix64, out: &mut [f64]) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

    let lanes = 4;
    let n = out.len();
    let body = n - (n % lanes);

    let vmean = _mm256_set1_pd(mean);
    let vstd = _mm256_set1_pd(std_dev);

    let mut i = 0;
    while i < body {
        let words = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let mut zs = [0.0_f64; 4];
        for (slot, &word) in zs.iter_mut().zip(words.iter()) {
            let (cand, ok) = fast_candidate(word);
            *slot = if ok { cand } else { resolve_lane(word, rng) };
        }
        // SAFETY: `zs` is a 4-element stack array, so the 4-wide unaligned load
        // reads exactly its bounds; `avx2` is guaranteed by the caller.
        let zv = _mm256_loadu_pd(zs.as_ptr());
        let res = _mm256_fmadd_pd(vstd, zv, vmean);
        // SAFETY: `i + 4 <= body <= out.len()`, so the 4-wide unaligned store is
        // in-bounds; `avx2`+`fma` guaranteed by the caller.
        _mm256_storeu_pd(out.as_mut_ptr().add(i), res);
        i += lanes;
    }

    if body < n {
        sample_scalar(mean, std_dev, rng, out.get_mut(body..n).unwrap_or(&mut []));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `2^31`, the scale relating the 32-bit comparison form to the `f64` edges:
    /// the layer right edge is `ZIG_W[i] · 2^31`.
    const TWO_POW_31: f64 = 2_147_483_648.0;

    /// The committed layer tables satisfy the ziggurat construction invariant:
    /// `ZIG_F[i] = exp(−½·(ZIG_W[i]·2^31)²)` for every interior layer, the top
    /// layer has density `1`, and the bottom edge is exactly `ZIG_R`. A typo in any
    /// literal breaks one of these, so the constants are checked, not trusted.
    #[test]
    fn table_construction_invariant_holds() {
        let top = ZIG_F.first().copied().unwrap_or(0.0);
        assert!(
            (top - 1.0).abs() < 1e-15,
            "top-layer density must be 1, was {top}"
        );
        for (i, (&w, &fi)) in ZIG_W.iter().zip(ZIG_F.iter()).enumerate().skip(1) {
            let edge = w * TWO_POW_31;
            let want = (-0.5 * edge * edge).exp();
            assert!(
                (fi - want).abs() < 1e-12,
                "layer {i}: ZIG_F={fi} != f(edge)={want}"
            );
        }
        let bottom_edge = ZIG_W.last().copied().unwrap_or(0.0) * TWO_POW_31;
        assert!(
            (bottom_edge - ZIG_R).abs() < 1e-9,
            "bottom edge {bottom_edge} must equal R={ZIG_R}"
        );
    }

    /// The standard normal CDF `½(1 + erf(x/√2))`, the KS reference.
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + crate::special::erf(x / std::f64::consts::SQRT_2))
    }

    /// One-sample Kolmogorov–Smirnov statistic `sup|F_n − F|` over an ascending
    /// `sorted` sample against reference `cdf`.
    fn ks_statistic(sorted: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
        let n = f64::from(u32::try_from(sorted.len()).unwrap_or(u32::MAX));
        let mut d = 0.0_f64;
        for (i, &x) in sorted.iter().enumerate() {
            let f = cdf(x);
            let i_f = f64::from(u32::try_from(i).unwrap_or(u32::MAX));
            d = d.max((i_f + 1.0) / n - f).max(f - i_f / n);
        }
        d
    }

    /// A large scalar ziggurat sample is distributed as N(0, 1): its empirical
    /// mean/variance converge to 0/1 and it fits the normal CDF under a
    /// Kolmogorov–Smirnov check below the 1% critical value. This is the
    /// equidistribution / ziggurat-correctness proof for the scalar oracle that the
    /// SIMD lanes defer to.
    #[test]
    fn scalar_ziggurat_fits_standard_normal() {
        let mut rng = SplitMix64::new(20_240_628);
        let n = 100_000usize;
        let mut xs: Vec<f64> = (0..n).map(|_| ziggurat_scalar(&mut rng)).collect();
        let count = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
        let mean = xs.iter().sum::<f64>() / count;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / count;
        assert!(mean.abs() < 0.02, "empirical mean {mean} not near 0");
        assert!(
            (var - 1.0).abs() < 0.03,
            "empirical variance {var} not near 1"
        );

        xs.sort_by(f64::total_cmp);
        let ks = ks_statistic(&xs, normal_cdf);
        let crit = 1.63 / count.sqrt();
        assert!(ks < crit, "KS={ks} exceeds 1% critical {crit}");
    }

    /// `normal_sample_into` is reproducible under a fixed seed and applies the
    /// `mean + std_dev·z` affine transform: filling the buffer twice from an
    /// identically-seeded generator yields byte-identical results, and the
    /// transformed sample's empirical mean/variance track `mean`/`std_dev²`. A
    /// non-lane-multiple length (103, prime) exercises the SIMD tail handling.
    #[test]
    fn batch_fill_is_reproducible_and_affine() {
        let (mean, std_dev) = (1.5, 2.0);
        let fill = |seed: u64, len: usize| {
            let mut rng = SplitMix64::new(seed);
            let mut out = vec![0.0; len];
            normal_sample_into(mean, std_dev, &mut rng, &mut out);
            out
        };
        let a = fill(424_242, 103);
        assert_eq!(a, fill(424_242, 103), "batch fill not reproducible");

        let big = fill(99, 80_000);
        let count = f64::from(u32::try_from(big.len()).unwrap_or(u32::MAX));
        let emp_mean = big.iter().sum::<f64>() / count;
        let emp_var = big
            .iter()
            .map(|x| (x - emp_mean) * (x - emp_mean))
            .sum::<f64>()
            / count;
        assert!(
            (emp_mean - mean).abs() < 0.05,
            "mean {emp_mean} not near {mean}"
        );
        let want_var = std_dev * std_dev;
        assert!(
            (emp_var - want_var).abs() < 0.15,
            "variance {emp_var} not near {want_var}"
        );
    }
}
