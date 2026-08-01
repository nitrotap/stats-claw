//! Unit tests for the Monte-Carlo resampling numerics (`super`).
//!
//! Split into this submodule file so `monte_carlo.rs` stays within the
//! 500-line `tests/style.rs` cap; the resampling directory is at its 10-file
//! limit, so a subdirectory module holds the extracted tests.
//!
//! # What may and may not be asserted bit-for-bit
//!
//! Two classes of value can be pinned to exact bits here and stay true on every
//! target:
//!
//! * anything computed from [`SplitMix64::next_u64`] or
//!   [`SplitMix64::next_f64`] — the state mixing is integer arithmetic and the
//!   uniform draw is one exact division by `2^53`, so the whole chain, the
//!   summation included, is bit-identical everywhere; and
//! * two values produced by one binary in one run (the same-seed, delegation,
//!   and `run`-equivalence checks), which is what determinism actually means.
//!
//! Anything derived from [`SplitMix64::standard_normal`] must **not** be
//! pinned. Box–Muller evaluates `f64::ln`, `f64::sin`, and `f64::cos`, which
//! Rust delegates to the platform math library; those kernels are accurate to
//! well under an ulp but are not correctly rounded, and different
//! implementations round differently. Measured on this module's own `seed = 7`,
//! `n_sims = 100_000` normal run: `15_900` of the `100_000` draws differ by 1–2
//! ulp between aarch64 and `x86_64`, moving the mean of the run by 26 ulp, and
//! 100 draws differ between an unoptimised and an optimised build of the same
//! source on the same machine. Note that it is the draws that move, not the
//! summation: the sum is a fixed left fold, and the uniform estimate below is
//! bit-identical across both targets and both optimisation levels.
//!
//! So the normal-sampler tests assert against the *theoretical* value within a
//! tolerance derived from that statistic's own standard error, and the
//! bit-exact regression pins sit on the uniform stream, where bit-exactness is
//! a guarantee this crate can actually keep. Please do not "tighten" those
//! tolerances back into pinned bit patterns: the pinned numbers would be true
//! only on the machine and build that recorded them.

use super::*;

/// Replicate count shared by the seeded pins and the theory tests.
const N_SIMS: usize = 100_000;

/// Bit-exact pin on the seeded *uniform* Monte-Carlo mean (`seed = 7`,
/// `n_sims = 100_000`, `sim = next_f64`).
///
/// Portable by construction — see this module's header — and verified identical
/// on aarch64 and `x86_64` in both optimised and unoptimised builds. This is the
/// determinism regression guard for [`monte_carlo_estimate`]; it replaces a pin
/// on the standard-normal mean, which held only on the machine that recorded it.
const PIN_UNIFORM_MEAN_BITS: u64 = 4_602_673_648_210_023_252;
/// Bit-exact pin on the matching seeded uniform standard error, portable for the
/// same reason as [`PIN_UNIFORM_MEAN_BITS`].
const PIN_UNIFORM_SE_BITS: u64 = 4_561_567_819_190_864_006;
/// Bit-exact pin on the seeded uniform upper-tail p-value of `0.975`
/// (`seed = 7`, `n_sims = 100_000`).
///
/// The p-value is `(b + 1) / (n + 1)`: an integer count of comparisons against
/// portable draws, widened exactly and divided once, so it is exact on every
/// target.
const PIN_UNIFORM_PVALUE_BITS: u64 = 4_583_024_316_151_914_253;

/// The estimate exposes its summaries through accessors, matching the
/// `CvScores` / `JackknifeEstimate` encapsulation style.
#[test]
fn exposes_summaries_via_accessors() -> Result<()> {
    let est = monte_carlo_estimate(4, &mut SplitMix64::new(1), |_r| 2.0)?;
    assert!(
        (est.mean() - 2.0).abs() < 1e-12,
        "mean() was {}",
        est.mean()
    );
    assert!(
        est.std_error().abs() < 1e-12,
        "std_error() was {}",
        est.std_error()
    );
    assert_eq!(
        est.n_simulations(),
        4,
        "n_simulations() must report the count"
    );
    Ok(())
}

/// Fewer than two simulations leaves the standard error undefined.
#[test]
fn too_few_simulations_is_insufficient_data() {
    let mut rng = SplitMix64::new(1);
    let result = monte_carlo_estimate(1, &mut rng, SplitMix64::next_f64);
    assert!(
        matches!(result, Err(Error::InsufficientData)),
        "expected InsufficientData, got {result:?}"
    );
}

/// Identically seeded generators reproduce the estimate bit-for-bit.
///
/// Both runs happen inside this binary, so this is a determinism check, not a
/// cross-target claim, and the exact comparison is sound.
#[test]
fn same_seed_reproduces_estimate() -> Result<()> {
    let a = monte_carlo_estimate(
        2_000,
        &mut SplitMix64::new(2024),
        SplitMix64::standard_normal,
    )?;
    let b = monte_carlo_estimate(
        2_000,
        &mut SplitMix64::new(2024),
        SplitMix64::standard_normal,
    )?;
    assert_eq!(
        a.mean().to_bits(),
        b.mean().to_bits(),
        "mean stream diverged"
    );
    assert_eq!(
        a.std_error().to_bits(),
        b.std_error().to_bits(),
        "std_error stream diverged"
    );
    assert_eq!(
        a.n_simulations(),
        b.n_simulations(),
        "n_simulations diverged"
    );
    Ok(())
}

/// A large standard-normal simulation recovers a mean and a standard error
/// statistically indistinguishable from theory.
///
/// Both gates are four standard errors of the statistic under test — a fixed
/// margin that depends only on `n`, never on the target:
///
/// * the mean of `n` unit-variance draws has standard error `1 / sqrt(n)`, so
///   the mean must land within `4 / sqrt(n)` of the true `0`; and
/// * the sample standard deviation has *relative* standard error
///   `1 / sqrt(2 (n - 1))`, so the reported standard error must land within
///   `4 / sqrt(2 (n - 1))` relative of the theoretical `1 / sqrt(n)`.
///
/// These replace bit-exact pins on the same two numbers. The pins asserted that
/// the platform math library rounds `ln`/`sin`/`cos` identically on every
/// target, which it does not; see this module's header for the measurement, and
/// [`seeded_uniform_estimate_is_pinned_bit_for_bit`] for the regression pin that
/// took over their job.
#[test]
fn standard_normal_mean_matches_theory() -> Result<()> {
    let est = monte_carlo_estimate(N_SIMS, &mut SplitMix64::new(7), SplitMix64::standard_normal)?;
    let n = count_to_f64(N_SIMS);
    let theoretical_se = 1.0 / n.sqrt();
    assert!(
        est.mean().abs() < 4.0 * theoretical_se,
        "mean {} not within 4 theoretical SE ({}) of 0",
        est.mean(),
        4.0 * theoretical_se
    );
    let se_relative_error = (est.std_error() / theoretical_se - 1.0).abs();
    let se_relative_tol = 4.0 / (2.0 * (n - 1.0)).sqrt();
    assert!(
        se_relative_error < se_relative_tol,
        "std_error {} is {se_relative_error} relative from the theoretical \
         {theoretical_se}, past the {se_relative_tol} bound",
        est.std_error()
    );
    Ok(())
}

/// The seeded *uniform* estimate is pinned bit-for-bit.
///
/// This is the determinism regression guard for [`monte_carlo_estimate`], sited
/// on the one stream whose bits are portable: a change to the generator, to the
/// mean/variance reduction, or to the estimator's arithmetic moves these bits,
/// while a change of target or optimisation level does not. The accompanying
/// theory check keeps the pins honest — they encode the true `Uniform[0, 1)`
/// moments, not a captured wrong answer.
#[test]
fn seeded_uniform_estimate_is_pinned_bit_for_bit() -> Result<()> {
    let est = monte_carlo_estimate(N_SIMS, &mut SplitMix64::new(7), SplitMix64::next_f64)?;
    assert_eq!(
        est.mean().to_bits(),
        PIN_UNIFORM_MEAN_BITS,
        "seeded uniform mean drifted"
    );
    assert_eq!(
        est.std_error().to_bits(),
        PIN_UNIFORM_SE_BITS,
        "seeded uniform std_error drifted"
    );
    // E[U] = 0.5 and sd(U) = 1 / sqrt(12), so the mean must sit within four of
    // its own standard errors of 0.5.
    assert!(
        (est.mean() - 0.5).abs() < 4.0 * est.std_error(),
        "uniform mean {} not within 4 SE ({}) of 0.5",
        est.mean(),
        4.0 * est.std_error()
    );
    Ok(())
}

/// The add-one correction bounds the p-value to `(0, 1]`, and an observation
/// more extreme than every null draw yields the floor `1 / (n + 1)`.
#[test]
fn p_value_is_bounded_and_hits_the_floor() -> Result<()> {
    let n = 10_000;
    // No standard-normal draw reaches 1e9, so b = 0 and p = 1 / (n + 1).
    let p = monte_carlo_p_value(
        1e9,
        n,
        &mut SplitMix64::new(11),
        SplitMix64::standard_normal,
        Alternative::Greater,
    )?;
    assert!(p > 0.0 && p <= 1.0, "p escaped (0, 1]: {p}");
    let expected = 1.0 / (count_to_f64(n) + 1.0);
    assert!(
        (p - expected).abs() < 1e-12,
        "floor p was {p}, expected {expected}"
    );
    Ok(())
}

/// The upper-tail p-value of `1.96` against a standard-normal null recovers the
/// textbook `P(Z >= 1.96)`.
///
/// The gate is four binomial standard errors, `sqrt(p (1 - p) / n)` — about
/// `4.9e-4` here, so `4 SE ~ 2.0e-3`. It is not a bit pin: the count behind `p`
/// is taken over standard-normal draws, whose last ulp is platform math-library
/// dependent (see this module's header). The portable counterpart is
/// [`seeded_uniform_p_value_is_pinned_bit_for_bit`].
#[test]
fn upper_tail_p_value_matches_normal_survival() -> Result<()> {
    // Reference: scipy.stats.norm.sf(1.96) == 0.024997895148220435.
    const THEORETICAL: f64 = 0.024_997_895_148_220_435;
    let p = monte_carlo_p_value(
        1.96,
        N_SIMS,
        &mut SplitMix64::new(7),
        SplitMix64::standard_normal,
        Alternative::Greater,
    )?;
    let se = (THEORETICAL * (1.0 - THEORETICAL) / count_to_f64(N_SIMS)).sqrt();
    assert!(
        (p - THEORETICAL).abs() < 4.0 * se,
        "MC p {p} is further than 4 SE ({}) from theoretical {THEORETICAL}",
        4.0 * se
    );
    Ok(())
}

/// The seeded *uniform* upper-tail p-value is pinned bit-for-bit.
///
/// `P(U >= 0.975) = 0.025` exactly, the comparisons are against portable draws,
/// and `(b + 1) / (n + 1)` is one exact division, so this pin holds on every
/// target — it is the regression guard for [`monte_carlo_p_value`] that the
/// standard-normal pin could not be.
#[test]
fn seeded_uniform_p_value_is_pinned_bit_for_bit() -> Result<()> {
    const THEORETICAL: f64 = 0.025;
    let p = monte_carlo_p_value(
        0.975,
        N_SIMS,
        &mut SplitMix64::new(7),
        SplitMix64::next_f64,
        Alternative::Greater,
    )?;
    assert_eq!(
        p.to_bits(),
        PIN_UNIFORM_PVALUE_BITS,
        "seeded uniform p-value drifted"
    );
    let se = (THEORETICAL * (1.0 - THEORETICAL) / count_to_f64(N_SIMS)).sqrt();
    assert!(
        (p - THEORETICAL).abs() < 4.0 * se,
        "uniform upper-tail p {p} is further than 4 SE ({}) from {THEORETICAL}",
        4.0 * se
    );
    Ok(())
}

/// The inherent method on the scheme delegates bit-for-bit to the
/// free function.
///
/// Both calls run in this binary, so the exact comparison tests delegation, not
/// cross-target reproducibility.
#[test]
fn inherent_estimate_delegates() -> Result<()> {
    let scheme = MonteCarloResampling::default();
    let via_method =
        scheme.estimate(1_000, &mut SplitMix64::new(3), SplitMix64::standard_normal)?;
    let via_free =
        monte_carlo_estimate(1_000, &mut SplitMix64::new(3), SplitMix64::standard_normal)?;
    assert_eq!(
        via_method.mean().to_bits(),
        via_free.mean().to_bits(),
        "delegation changed the mean"
    );
    assert_eq!(
        via_method.std_error().to_bits(),
        via_free.std_error().to_bits(),
        "delegation changed the std_error"
    );
    Ok(())
}

/// `run` consumes the scheme's `number_of_iterations` and `random_seed`, so it
/// must reproduce `monte_carlo_estimate` called with the equivalent count and
/// seed bit-for-bit (`random_seed` 7 reinterprets to `7u64`).
///
/// As above, both calls run in this binary: the exact comparison is a wiring
/// check, not a claim about other targets.
#[test]
fn run_matches_estimate_with_equivalent_seed() -> Result<()> {
    let scheme = MonteCarloResampling {
        number_of_iterations: 1_000,
        random_seed: 7,
        ..Default::default()
    };
    let via_run = scheme.run(SplitMix64::standard_normal)?;
    let via_free =
        monte_carlo_estimate(1_000, &mut SplitMix64::new(7), SplitMix64::standard_normal)?;
    assert_eq!(
        via_run.mean().to_bits(),
        via_free.mean().to_bits(),
        "run() diverged from monte_carlo_estimate mean"
    );
    assert_eq!(
        via_run.std_error().to_bits(),
        via_free.std_error().to_bits(),
        "run() diverged from monte_carlo_estimate std_error"
    );
    assert_eq!(
        via_run.n_simulations(),
        1_000,
        "run() must report the configured iteration count"
    );
    Ok(())
}

/// A non-positive `number_of_iterations` is a bad configuration and must be
/// rejected with `InvalidInput` (mirroring `CrossValidation::run`).
#[test]
fn run_rejects_non_positive_iterations() {
    let scheme = MonteCarloResampling {
        number_of_iterations: 0,
        random_seed: 1,
        ..Default::default()
    };
    let result = scheme.run(SplitMix64::next_f64);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "non-positive number_of_iterations must be InvalidInput, got {result:?}"
    );
}
