//! Normal (Gaussian) distribution numerics, for the
//! [`NormalDistribution`].
//!
//! Equivalent to `scipy.stats.norm(loc=mean, scale=standard_deviation)`: the
//! density is the closed-form Gaussian, the CDF is `½(1 + erf(z))`, and the
//! quantile inverts it via a Newton-refined inverse error function. Sampling uses
//! the RNG's cached Box–Muller `standard_normal`.
//!
//! # Examples
//!
//! ```
//! use stats_claw::distributions::Pdf;
//! use stats_claw::distributions::NormalDistribution;
//!
//! let n = NormalDistribution { mean: 0.0, standard_deviation: 1.0, ..Default::default() };
//! // Peak density of the standard normal is 1/sqrt(2π) ≈ 0.398_942.
//! assert!((n.pdf(0.0) - 0.398_942_28).abs() < 1e-6, "peak was {}", n.pdf(0.0));
//! ```

use super::super::{Cdf, LogCdf, Moments, Pdf, Quantile, Sample};
use crate::distributions::NormalDistribution;
use crate::rng::SplitMix64;
use crate::special::{erf, ln_erfc};
use std::f64::consts::{LN_2, PI, SQRT_2};

/// `1/√(2π)`, the standard-normal density's normalization constant, precomputed
/// so the batch `pdf` hot path needs no `sqrt` per element.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

impl NormalDistribution {
    /// Evaluates the density at every point of `xs`, writing the results into
    /// `out`, using the fastest available native-SIMD path (NEON / AVX2 / scalar).
    ///
    /// This is the dense batch hot path the perf gate measures (`pdf` is the
    /// gated metric). It is numerically identical to calling [`Pdf::pdf`] per
    /// element (the SIMD `exp` agrees with the scalar `exp` to ≤ 1e-12 relative),
    /// so callers needing a single value should still use [`Pdf::pdf`].
    ///
    /// # Arguments
    ///
    /// * `xs` — the evaluation grid.
    /// * `out` — the output buffer; should have the same length as `xs`. If the
    ///   lengths differ, only the leading `min(xs.len(), out.len())` entries are
    ///   written.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::distributions::Pdf;
    /// use stats_claw::distributions::NormalDistribution;
    ///
    /// let n = NormalDistribution { mean: 0.0, standard_deviation: 1.0, ..Default::default() };
    /// let xs = [-1.0, 0.0, 1.0];
    /// let mut out = [0.0; 3];
    /// n.pdf_batch(&xs, &mut out);
    /// assert!((out[1] - n.pdf(0.0)).abs() < 1e-12);
    /// ```
    pub fn pdf_batch(&self, xs: &[f64], out: &mut [f64]) {
        super::super::simd::normal_pdf_into(self, xs, out);
    }

    /// Evaluates the CDF at every point of `xs`, writing the results into `out`.
    ///
    /// Numerically identical to calling [`Cdf::cdf`] per element; provided as the
    /// batch-layout counterpart of [`Self::pdf_batch`]. The `erf` evaluation stays
    /// scalar (Cody rational), so this is a convenience over the dense grid rather
    /// than a vectorized kernel.
    ///
    /// # Arguments
    ///
    /// * `xs` — the evaluation grid.
    /// * `out` — the output buffer; same-length contract as [`Self::pdf_batch`].
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::distributions::Cdf;
    /// use stats_claw::distributions::NormalDistribution;
    ///
    /// let n = NormalDistribution { mean: 0.0, standard_deviation: 1.0, ..Default::default() };
    /// let xs = [0.0];
    /// let mut out = [0.0; 1];
    /// n.cdf_batch(&xs, &mut out);
    /// assert!((out[0] - 0.5).abs() < 1e-12);
    /// ```
    pub fn cdf_batch(&self, xs: &[f64], out: &mut [f64]) {
        super::super::simd::normal_cdf_into(self, xs, out);
    }

    /// Draws `out.len()` variates into `out` using the native-SIMD batch ziggurat
    /// (NEON / AVX2 / scalar), the dense batch sampling hot path.
    ///
    /// This is the batch counterpart of [`Sample::sample`]. Where `sample` draws one
    /// variate by cached Box–Muller (and is the path the equivalence/GoF fixtures
    /// pin), `sample_batch` fills a whole buffer by the Marsaglia–Tsang ziggurat,
    /// vectorized per CPU — the workload that races numpy's `norm.rvs`. Both are
    /// statistically N(mean, σ²); they do **not** produce the same stream for a given
    /// seed (different algorithms consume the RNG differently), but each is itself
    /// reproducible for a fixed seed and CPU path.
    ///
    /// # Arguments
    ///
    /// * `rng` — the deterministic generator; advanced in a fixed per-path order so
    ///   a fixed seed yields a reproducible buffer on a given CPU.
    /// * `out` — the output buffer; every element is overwritten with a draw.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::distributions::NormalDistribution;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// let n = NormalDistribution { mean: 0.0, standard_deviation: 1.0, ..Default::default() };
    /// let mut rng = SplitMix64::new(7);
    /// let mut out = [0.0; 4];
    /// n.sample_batch(&mut rng, &mut out);
    /// // Same seed, same CPU path → identical buffer.
    /// let mut rng2 = SplitMix64::new(7);
    /// let mut out2 = [0.0; 4];
    /// n.sample_batch(&mut rng2, &mut out2);
    /// assert_eq!(out, out2);
    /// ```
    pub fn sample_batch(&self, rng: &mut SplitMix64, out: &mut [f64]) {
        super::super::ziggurat::normal_sample_into(self.mean, self.standard_deviation, rng, out);
    }
}

impl Pdf for NormalDistribution {
    fn pdf(&self, x: f64) -> f64 {
        // Hoist the reciprocal scale and normalization out of the transcendental:
        // one `mul`, one `exp`, two `mul` — no per-call `sqrt` or division.
        let inv_sigma = 1.0 / self.standard_deviation;
        let z = (x - self.mean) * inv_sigma;
        (-0.5 * z * z).exp() * (INV_SQRT_2PI * inv_sigma)
    }
}

impl Cdf for NormalDistribution {
    fn cdf(&self, x: f64) -> f64 {
        // Fold the 1/(σ√2) scale into a single reciprocal-multiply so the hot
        // path is one `mul` plus the `erf` rational eval — no per-call division.
        let inv_scale = 1.0 / (self.standard_deviation * SQRT_2);
        let z = (x - self.mean) * inv_scale;
        0.5 * (1.0 + erf(z))
    }
}

impl LogCdf for NormalDistribution {
    fn logsf(&self, x: f64) -> f64 {
        // sf(x) = erfc(z)/2 with z = (x-μ)/(σ√2); ln sf = ln erfc(z) − ln 2.
        let z = (x - self.mean) / (self.standard_deviation * SQRT_2);
        ln_erfc(z) - LN_2
    }
    fn logcdf(&self, x: f64) -> f64 {
        // cdf(x) = sf(2μ − x) by symmetry; equivalently ln erfc(−z) − ln 2.
        let z = (x - self.mean) / (self.standard_deviation * SQRT_2);
        ln_erfc(-z) - LN_2
    }
}

impl Quantile for NormalDistribution {
    fn quantile(&self, p: f64) -> f64 {
        let erf_target = 2.0f64.mul_add(p, -1.0);
        self.standard_deviation
            .mul_add(SQRT_2 * inv_erf(erf_target), self.mean)
    }
}

impl Moments for NormalDistribution {
    fn mean(&self) -> Option<f64> {
        Some(self.mean)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.standard_deviation * self.standard_deviation)
    }
}

impl Sample for NormalDistribution {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        self.standard_deviation
            .mul_add(rng.standard_normal(), self.mean)
    }
}

/// Inverse error function via a rational approximation refined by Newton steps on
/// `erf`.
///
/// The seed is Giles' polynomial approximation; three Newton iterations using the
/// exact derivative `erf'(x) = 2/√π · e^(−x²)` then drive the residual to ~1e-15
/// over the open interval `(-1, 1)`.
///
/// # Arguments
///
/// * `y` — the target erf value in `[-1, 1]`; the closed endpoints map to `±∞`.
///
/// # Returns
///
/// The `x` with `erf(x) = y`.
fn inv_erf(y: f64) -> f64 {
    if y <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if y >= 1.0 {
        return f64::INFINITY;
    }
    let w = -((1.0 - y) * (1.0 + y)).ln();
    let mut x = if w < 5.0 {
        let w = w - 2.5;
        let mut p: f64 = 2.810_226_36e-08;
        for c in [
            3.432_739_39e-07,
            -3.523_387_7e-06,
            -4.391_506_54e-06,
            0.000_218_580_87,
            -0.001_253_725_03,
            -0.004_177_681_64,
            0.246_640_727,
            1.501_409_41,
        ] {
            p = p.mul_add(w, c);
        }
        p * y
    } else {
        let w = w.sqrt() - 3.0;
        let mut p: f64 = -0.000_200_214_257;
        for c in [
            0.000_100_950_558,
            0.001_349_343_22,
            -0.003_673_428_44,
            0.005_739_507_73,
            -0.007_622_461_3,
            0.009_438_870_47,
            1.001_674_06,
            2.832_976_82,
        ] {
            p = p.mul_add(w, c);
        }
        p * y
    };
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..3 {
        let err = erf(x) - y;
        x -= err / (two_over_sqrt_pi * (-x * x).exp());
    }
    x
}

/// Kani formal-verification harnesses for the Normal distribution (tier 2).
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy.
#[cfg(kani)]
mod verification {
    use super::{NormalDistribution, Pdf};

    /// Builds a `NormalDistribution` with symbolic `mean`/`standard_deviation` and
    /// empty string metadata (unused by the density).
    ///
    /// # Arguments
    ///
    /// * `mean` — the symbolic location parameter.
    /// * `standard_deviation` — the symbolic scale parameter.
    ///
    /// # Returns
    ///
    /// A distribution carrying the two symbolic numeric parameters.
    fn make(mean: f64, standard_deviation: f64) -> NormalDistribution {
        NormalDistribution {
            mean,
            standard_deviation,
            variance: standard_deviation * standard_deviation,
            parameterization: String::new(),
            distribution_name: String::new(),
            description: String::new(),
        }
    }

    /// Proves the Normal density evaluation is panic-/overflow-free for every valid
    /// parameterization and every finite, magnitude-bounded evaluation point.
    ///
    /// The mathematical invariant `0 ≤ pdf(x) ≤ 1/(σ√2π)` holds because the density
    /// is `exp(−½z²) · (INV_SQRT_2PI / σ)` with `exp ∈ (0, 1]` for the non-positive
    /// exponent, `INV_SQRT_2PI > 0`, and `σ > 0`. Kani cannot *prove* that bound: it
    /// models `exp` with a sound over-approximation that does not encode
    /// `exp(t) ∈ (0, 1]` for `t ≤ 0`, so a value assertion fails spuriously
    /// (confirmed: `p >= 0.0` and the peak bound both report counterexamples where
    /// the modelled `exp` returns an out-of-range value). Per the tier-2 guidance
    /// this harness therefore verifies the tractable, non-transcendental property —
    /// that evaluating the density never panics, overflows, or hits UB over the
    /// whole valid parameter/argument box.
    #[kani::proof]
    fn normal_pdf_no_panic() {
        let mean: f64 = kani::any();
        let sigma: f64 = kani::any();
        let x: f64 = kani::any();
        kani::assume(mean.is_finite() && mean.abs() <= 1e6);
        kani::assume(x.is_finite() && x.abs() <= 1e6);
        kani::assume(sigma.is_finite() && sigma >= 1e-3 && sigma <= 1e6);
        let dist = make(mean, sigma);
        let _ = dist.pdf(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard normal peaks at `1/√(2π) ≈ 0.398_942` at the mean.
    #[test]
    fn standard_normal_peak_density() {
        let n = NormalDistribution {
            mean: 0.0,
            standard_deviation: 1.0,
            ..Default::default()
        };
        assert!(
            (n.pdf(0.0) - 0.398_942_28).abs() < 1e-6,
            "peak density was {}",
            n.pdf(0.0)
        );
    }

    /// The shared behaviour traits still let us use it polymorphically.
    #[test]
    fn density_decreases_away_from_mean() {
        fn density_at(d: &impl Pdf, x: f64) -> f64 {
            d.pdf(x)
        }
        let n = NormalDistribution {
            mean: 2.0,
            standard_deviation: 0.5,
            ..Default::default()
        };
        assert!(
            density_at(&n, 2.0) > density_at(&n, 3.0),
            "density should fall off from the mean"
        );
    }

    /// `logsf`/`logcdf` agree with the log of the linear `cdf` in the body and
    /// stay finite (matching scipy) in the deep tail where `1 - cdf` underflows.
    #[test]
    fn logsf_matches_scipy_and_stays_finite() {
        let n = NormalDistribution {
            mean: 0.0,
            standard_deviation: 1.0,
            ..Default::default()
        };
        // Body: logsf(0.5) == ln(1 - cdf(0.5)).
        let body = n.logsf(0.5);
        assert!(
            ((body - (1.0 - n.cdf(0.5)).ln()) / body.abs()).abs() < 1e-10,
            "logsf body was {body}"
        );
        // Deep tail: linear sf underflows but logsf matches scipy norm.logsf(40).
        let tail = n.logsf(40.0);
        let want = -804.608_442_013_753_9;
        assert!(tail.is_finite(), "logsf(40) was {tail}");
        assert!(
            ((tail - want) / want).abs() < 1e-9,
            "logsf(40) = {tail}, want {want}"
        );
    }
}
