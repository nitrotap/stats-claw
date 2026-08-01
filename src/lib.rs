//! stats-claw — in-process, zero-dependency statistical computing for Rust.
//!
//! Probability distributions, hypothesis tests, resampling, optimizers,
//! streaming summaries, likelihood estimation, and supporting algorithms.
//! Each distribution is a plain parameter struct over which the behaviour
//! traits (`Pdf`/`Cdf`/`Quantile`/`Sample`/`Moments`) are implemented by hand.

pub mod algorithms;
pub mod distributions;
pub mod error;
pub mod likelihood;
/// Framework-internal shared numeric primitives (count widening, mean, variance).
mod numeric;
pub mod optimizers;
pub mod resampling;
pub mod rng;
pub mod special;
pub mod streaming;
pub mod tests_stat;
