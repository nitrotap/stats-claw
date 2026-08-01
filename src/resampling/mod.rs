//! Resampling-based inference: seeded bootstrap, permutation, and CV splits, plus
//! the percentile confidence-interval estimator built on top of them.
//!
//! Every sampler draws from the deterministic [`SplitMix64`](crate::rng::SplitMix64)
//! PRNG, so results are bit-for-bit reproducible under a fixed seed and identical
//! across platforms. The percentile interval ([`percentile_ci`])
//! is validated for reference equivalence against
//! `scipy.stats.bootstrap` golden fixtures downstream.
//!
//! The module is split into focused submodules under the file-size cap:
//! [`schemes`] holds the seeded resampling schemes (bootstrap, permutation,
//! k-fold splits); [`intervals`] the percentile interval and bootstrap-statistic
//! helper; [`bayesian`] the Beta credible interval; [`cross_validation`] the
//! k-fold CV evaluator and shared [`CvScores`] type; [`loocv`] the leave-one-out
//! CV evaluator (returning the same [`CvScores`]); [`stratified`] the
//! class-balanced k-fold splitter; [`monte_carlo`] the simulation-based
//! expectation estimate and Phipson–Smyth p-value; [`jackknife`] the
//! deterministic leave-one-out bias/standard-error estimator; and the private
//! `index` module the cast-free index arithmetic they share.
//!
//! The scheme types here follow the crate-wide field-consumption rule: `CrossValidation::run`,
//! `LeaveOneOutCrossValidation::run`, and `MonteCarloResampling::run` consume
//! their struct's own configured fields (fold count / iteration count and
//! `random_seed`), whereas `MonteCarloResampling::estimate` takes an explicit
//! count and generator and so ignores those overlapping fields by design.

pub mod types;
pub use types::*;
pub mod bayesian;
pub mod cross_validation;
mod index;
pub mod intervals;
pub mod jackknife;
pub mod loocv;
pub mod monte_carlo;
pub mod schemes;
pub mod stratified;

pub use bayesian::beta_credible_interval;
pub use cross_validation::{CvScores, cross_validate};
pub use intervals::{bootstrap_statistic, coverage_rate, percentile_ci};
pub use jackknife::{JackknifeEstimate, jackknife_indices, jackknife_statistic};
pub use loocv::{loo_cross_validate, loo_indices};
pub use monte_carlo::{MonteCarloEstimate, monte_carlo_estimate, monte_carlo_p_value};
pub use schemes::{bootstrap_indices, kfold_indices, permutation};
pub use stratified::stratified_kfold_indices;
