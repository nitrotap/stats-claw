//! stats-claw — data science on the hot path.
//!
//! An in-process, zero-runtime-dependency statistical computing library for
//! Rust: probability distributions, hypothesis tests, resampling, optimizers,
//! streaming summaries, and supporting algorithms. The numerics are validated
//! against committed `scipy.stats` golden fixtures within documented tolerances.
//!
//! Each distribution is a plain parameter struct (in [`distributions`]) over
//! which the behaviour traits (`Pdf`/`Cdf`/`Quantile`/`Sample`/`Moments`/
//! `LogCdf`) are implemented by hand.

pub mod algorithms;
pub mod distributions;
pub mod error;
pub mod optimizers;
pub mod resampling;
pub mod rng;
pub mod special;
pub mod streaming;
pub mod tests_stat;
