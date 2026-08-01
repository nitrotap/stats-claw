//! Monte-Carlo resampling numerics: simulation-based expectation estimates and
//! the Phipson–Smyth add-one Monte-Carlo p-value, for the
//! [`MonteCarloResampling`] scheme.
//!
//! Both estimators draw from the deterministic [`SplitMix64`]
//! PRNG, so a fixed seed reproduces a result bit-for-bit within a given build
//! and target. Reproducing it bit-for-bit *across* targets additionally needs a
//! portable simulation closure: [`SplitMix64::next_f64`] qualifies, whereas
//! [`SplitMix64::standard_normal`] inherits the platform math library's last ulp
//! and so must be compared with a tolerance. The
//! p-value uses the `(b + 1) / (n + 1)` correction of Phipson &
//! Smyth (2010), which keeps the estimate strictly positive and never reports an
//! impossible zero p-value.
//!
//! # Examples
//!
//! ```
//! use stats_claw::resampling::monte_carlo_estimate;
//! use stats_claw::rng::SplitMix64;
//!
//! // Estimate E[U] for U ~ Uniform[0, 1); the true mean is 0.5.
//! let est = monte_carlo_estimate(10_000, &mut SplitMix64::new(1), |r| r.next_f64())?;
//! assert!((est.mean() - 0.5).abs() < 4.0 * est.std_error(), "mean was {}", est.mean());
//! # Ok::<(), stats_claw::error::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::numeric::count_to_f64;
use crate::resampling::MonteCarloResampling;
use crate::rng::SplitMix64;
use crate::tests_stat::Alternative;

/// The outcome of a Monte-Carlo expectation estimate.
///
/// Bundles the sample mean of the simulated draws with its standard error and the
/// number of simulations that produced it, so a caller can build a confidence
/// band (`mean ± z · std_error`) without re-deriving the sample size. The fields
/// are private; read them through the [`mean`](Self::mean),
/// [`std_error`](Self::std_error), and [`n_simulations`](Self::n_simulations)
/// accessors, matching the `CvScores` / `JackknifeEstimate` style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonteCarloEstimate {
    /// The sample mean of the `n_simulations` simulated values, i.e. the estimate
    /// of `E[f]`.
    mean: f64,
    /// The standard error of the mean: `sd(sims, ddof=1) / sqrt(n_simulations)`.
    std_error: f64,
    /// The number of simulations averaged into `mean` (always `>= 2`).
    n_simulations: usize,
}

impl MonteCarloEstimate {
    /// Returns the Monte-Carlo estimate of `E[f]`: the sample mean of the draws.
    ///
    /// # Returns
    ///
    /// The mean of the `n_simulations` simulated values.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::monte_carlo_estimate;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// let est = monte_carlo_estimate(10_000, &mut SplitMix64::new(1), |r| r.next_f64())?;
    /// assert!((est.mean() - 0.5).abs() < 4.0 * est.std_error(), "mean was {}", est.mean());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the standard error of the mean: `sd(sims, ddof=1) / sqrt(n_sims)`.
    ///
    /// # Returns
    ///
    /// The standard error of [`mean`](Self::mean); `0.0` when every draw was
    /// identical.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::monte_carlo_estimate;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// // A constant simulation has zero spread, hence zero standard error.
    /// let est = monte_carlo_estimate(8, &mut SplitMix64::new(1), |_r| 2.0)?;
    /// assert!(est.std_error().abs() < 1e-12, "std_error was {}", est.std_error());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn std_error(&self) -> f64 {
        self.std_error
    }

    /// Returns the number of simulation replicates averaged into [`mean`](Self::mean).
    ///
    /// # Returns
    ///
    /// The simulation count `n_sims` (always `>= 2`).
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::monte_carlo_estimate;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// let est = monte_carlo_estimate(5_000, &mut SplitMix64::new(1), |r| r.next_f64())?;
    /// assert_eq!(est.n_simulations(), 5_000);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn n_simulations(&self) -> usize {
        self.n_simulations
    }
}

/// Estimates `E[f]` by Monte-Carlo simulation.
///
/// Runs `sim` `n_sims` times against `rng`, then returns the sample mean of the
/// draws together with its standard error `sd(sims, ddof=1) / sqrt(n_sims)`. The
/// draws share one deterministic generator, so a fixed seed reproduces the
/// estimate bit-for-bit within a given build and target — and on every target
/// too, provided `sim` is itself portable (see the module header).
///
/// # Arguments
///
/// * `n_sims` — number of simulation replicates; must be `>= 2` so the
///   `ddof = 1` variance is defined.
/// * `rng` — the deterministic generator threaded through every replicate.
/// * `sim` — the simulation closure; each call may advance `rng` and returns one
///   realised value of `f`.
///
/// # Returns
///
/// A [`MonteCarloEstimate`] holding the mean, its standard error, and `n_sims`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] when `n_sims < 2` (the sample standard
/// error is undefined for fewer than two draws).
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::monte_carlo_estimate;
/// use stats_claw::rng::SplitMix64;
///
/// let est = monte_carlo_estimate(50_000, &mut SplitMix64::new(3), |r| r.standard_normal())?;
/// assert!(est.mean().abs() < 4.0 * est.std_error(), "standard-normal mean was {}", est.mean());
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn monte_carlo_estimate(
    n_sims: usize,
    rng: &mut SplitMix64,
    mut sim: impl FnMut(&mut SplitMix64) -> f64,
) -> Result<MonteCarloEstimate> {
    if n_sims < 2 {
        return Err(Error::InsufficientData);
    }
    let mut sims = Vec::with_capacity(n_sims);
    for _ in 0..n_sims {
        sims.push(sim(rng));
    }
    let n = count_to_f64(n_sims);
    // Shared two-pass mean and ddof=1 variance; `n_sims >= 2` is guaranteed
    // above, so the variance denominator is defined.
    let mean = crate::numeric::mean(&sims);
    let variance = crate::numeric::sample_variance(&sims);
    let std_error = variance.sqrt() / n.sqrt();
    Ok(MonteCarloEstimate {
        mean,
        std_error,
        n_simulations: n_sims,
    })
}

/// Computes a Monte-Carlo p-value with the Phipson–Smyth add-one correction.
///
/// Draws `n_sims` statistics from the null distribution via `null_sim`, counts how
/// many `b` are at least as extreme as `observed` under `alternative`, and returns
/// `(b + 1) / (n_sims + 1)`. The `+1` in both terms is the Phipson & Smyth (2010)
/// correction: it treats `observed` itself as one draw of the null, so the p-value
/// is never an impossible zero and stays in `(0, 1]`.
///
/// The extremeness rule per `alternative` is:
/// * [`Alternative::Greater`] — `sim >= observed` (upper tail).
/// * [`Alternative::Less`] — `sim <= observed` (lower tail).
/// * [`Alternative::TwoSided`] — `sim.abs() >= observed.abs()`. This plain
///   magnitude rule assumes the null statistic is already centred on zero; any
///   centring of `observed`/`sim` is the caller's responsibility.
///
/// # Arguments
///
/// * `observed` — the statistic actually observed on the real data.
/// * `n_sims` — number of null replicates to simulate; must be `>= 1`.
/// * `rng` — the deterministic generator threaded through every replicate.
/// * `null_sim` — draws one statistic from the null distribution.
/// * `alternative` — which tail(s) define "at least as extreme".
///
/// # Returns
///
/// The corrected p-value in `(0, 1]`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] when `n_sims == 0` (no null draws).
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::monte_carlo_p_value;
/// use stats_claw::rng::SplitMix64;
/// use stats_claw::tests_stat::Alternative;
///
/// // An observation past every plausible standard-normal draw hits the floor.
/// let p = monte_carlo_p_value(
///     10.0,
///     1_000,
///     &mut SplitMix64::new(1),
///     |r| r.standard_normal(),
///     Alternative::Greater,
/// )?;
/// assert!((p - 1.0 / 1_001.0).abs() < 1e-12, "p was {p}");
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn monte_carlo_p_value(
    observed: f64,
    n_sims: usize,
    rng: &mut SplitMix64,
    mut null_sim: impl FnMut(&mut SplitMix64) -> f64,
    alternative: Alternative,
) -> Result<f64> {
    if n_sims == 0 {
        return Err(Error::InsufficientData);
    }
    let abs_observed = observed.abs();
    let mut at_least_as_extreme = 0usize;
    for _ in 0..n_sims {
        let sim = null_sim(rng);
        let extreme = match alternative {
            Alternative::Greater => sim >= observed,
            Alternative::Less => sim <= observed,
            Alternative::TwoSided => sim.abs() >= abs_observed,
        };
        if extreme {
            at_least_as_extreme += 1;
        }
    }
    let b = count_to_f64(at_least_as_extreme);
    let n = count_to_f64(n_sims);
    Ok((b + 1.0) / (n + 1.0))
}

impl MonteCarloResampling {
    /// Estimates `E[f]` by simulation, attaching the numeric to the
    /// scheme type.
    ///
    /// This is the framework's inherent-impl entry point: it delegates verbatim to
    /// the free [`monte_carlo_estimate`] using the explicit `n_sims`, generator,
    /// and simulation closure. The scheme's configured fields (its
    /// `number_of_iterations` / `random_seed`) are intentionally not consulted —
    /// the explicit arguments take precedence so a caller keeps full control of
    /// the run.
    ///
    /// # Arguments
    ///
    /// * `n_sims` — number of simulation replicates; must be `>= 2`.
    /// * `rng` — the deterministic generator threaded through every replicate.
    /// * `sim` — the simulation closure returning one realised value of `f`.
    ///
    /// # Returns
    ///
    /// A [`MonteCarloEstimate`] holding the mean, its standard error, and `n_sims`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] when `n_sims < 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::MonteCarloResampling;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// let scheme = MonteCarloResampling::default();
    /// let est = scheme.estimate(10_000, &mut SplitMix64::new(5), |r| r.next_f64())?;
    /// assert!((est.mean() - 0.5).abs() < 4.0 * est.std_error(), "mean was {}", est.mean());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    // The delegation ignores `self`'s configured fields by design (explicit
    // arguments win); the method exists to hang the numeric off the scheme type
    // per the framework's inherent-impl pattern.
    #[allow(clippy::unused_self)]
    pub fn estimate(
        &self,
        n_sims: usize,
        rng: &mut SplitMix64,
        sim: impl FnMut(&mut SplitMix64) -> f64,
    ) -> Result<MonteCarloEstimate> {
        monte_carlo_estimate(n_sims, rng, sim)
    }

    /// Runs a Monte-Carlo estimate using this scheme's own configuration.
    ///
    /// Reads the replicate count from
    /// [`number_of_iterations`](Self::number_of_iterations) and seeds the
    /// deterministic PRNG from [`random_seed`](Self::random_seed), then delegates
    /// to [`monte_carlo_estimate`]. The `i64` seed is reinterpreted to `u64`
    /// bit-for-bit via [`i64::cast_unsigned`] (not a numeric `as` cast, which the
    /// `style.rs` guard bans), so a positive seed maps to the same magnitude —
    /// mirroring [`CrossValidation::run`](crate::resampling::CrossValidation::run).
    /// Unlike [`estimate`](Self::estimate) — which takes an explicit count and
    /// generator and ignores these fields — `run` *consumes* the scheme's
    /// configured fields, so the parameter struct is itself executable against the
    /// numerics.
    ///
    /// # Arguments
    ///
    /// * `sim` — the simulation closure; each call may advance the seeded
    ///   generator and returns one realised value of `f`.
    ///
    /// # Returns
    ///
    /// A [`MonteCarloEstimate`] holding the mean, its standard error, and the
    /// configured iteration count.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidInput`] when `number_of_iterations` is non-positive or
    ///   unrepresentable as a `usize` count.
    /// * [`Error::InsufficientData`] when the configured count is below two (the
    ///   `ddof = 1` standard error is then undefined), propagated from
    ///   [`monte_carlo_estimate`].
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::MonteCarloResampling;
    ///
    /// let scheme = MonteCarloResampling {
    ///     number_of_iterations: 10_000,
    ///     random_seed: 5,
    ///     ..Default::default()
    /// };
    /// // Estimating E[U] for U ~ Uniform[0, 1): the true mean is 0.5.
    /// let est = scheme.run(|r| r.next_f64())?;
    /// assert!((est.mean() - 0.5).abs() < 4.0 * est.std_error(), "mean was {}", est.mean());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn run(&self, sim: impl FnMut(&mut SplitMix64) -> f64) -> Result<MonteCarloEstimate> {
        if self.number_of_iterations <= 0 {
            return Err(Error::InvalidInput(
                "number_of_iterations must be positive".to_owned(),
            ));
        }
        let n_sims = usize::try_from(self.number_of_iterations).map_err(|_| {
            Error::InvalidInput("number_of_iterations exceeds the usize count range".to_owned())
        })?;
        let mut rng = SplitMix64::new(self.random_seed.cast_unsigned());
        monte_carlo_estimate(n_sims, &mut rng, sim)
    }
}

/// Kani formal-verification harnesses for the Monte-Carlo estimators.
///
/// These prove the input-validation paths (over symbolic replicate counts) and the
/// Phipson–Smyth p-value bound (over a symbolic observed statistic and symbolic
/// finite null draws), rather than the sampled runs the `#[cfg(test)]` suite uses.
/// The simulation/null closures are supplied by the caller, so the transcendental
/// kernels a real caller might use never enter these proofs — the harnesses verify
/// the layer's own control flow and arithmetic. Compiled only under `cargo kani`
/// (behind `#[cfg(kani)]`); invisible to normal build/test/clippy. Run e.g. with
/// `cargo kani -Z stubbing -p stats-claw --harness resampling_mc_p_value_bounded`.
#[cfg(kani)]
mod verification;

#[cfg(test)]
mod tests;
