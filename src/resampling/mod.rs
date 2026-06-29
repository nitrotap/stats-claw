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
//! [`schemes`] holds the seeded resampling schemes, [`intervals`] the percentile
//! interval and bootstrap-statistic helper, and the private `index` module the
//! cast-free index arithmetic both share.

pub mod bayesian;
mod index;
pub mod intervals;
pub mod schemes;

pub use bayesian::beta_credible_interval;
pub use intervals::{bootstrap_statistic, coverage_rate, percentile_ci};
pub use schemes::{bootstrap_indices, kfold_indices, permutation};
