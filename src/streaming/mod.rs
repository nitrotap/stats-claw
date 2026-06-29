//! Online (streaming) estimators that run in bounded memory.
//!
//! Each estimator consumes one value at a time through `update` and reports its
//! current estimate through an accessor, holding only a fixed-size summary of the
//! stream seen so far. State size is a compile-time constant — independent of how
//! many values have been consumed — so an arbitrarily long stream is processed
//! without memory growth.
//!
//! * [`RunningMoments`] — Welford's online mean and variance.
//! * [`P2Quantile`] — the P² streaming-quantile estimator (Jain & Chlamtac 1985).

mod moments;
mod p2;

pub use moments::RunningMoments;
pub use p2::P2Quantile;

/// Converts a stream count to `f64` for use as a divisor.
///
/// Shared by the streaming estimators so the count→float conversion is done one
/// way: through a lossless `u32` window, never an `as` cast (banned in `src/`).
///
/// # Arguments
///
/// * `n` — a stream count. Counts beyond `u32::MAX` clamp to `u32::MAX`, far past
///   any realistic stream where `f64` already loses integer precision (`> 2^53`).
///
/// # Returns
///
/// `n` represented as an `f64`.
pub(crate) fn count_to_f64(n: u64) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// Converts a 1-based, possibly-negative marker position to `f64`.
///
/// Shared by [`P2Quantile`]'s parabolic/linear updates. Goes through a lossless
/// `u32` magnitude window so the conversion never uses an `as` cast.
///
/// # Arguments
///
/// * `n` — a marker position; its magnitude clamps to `u32::MAX` (far beyond the
///   five P² markers' realistic positions).
///
/// # Returns
///
/// `n` as an `f64`, sign preserved.
pub(crate) fn position_as_f64(n: i64) -> f64 {
    let mag = f64::from(u32::try_from(n.unsigned_abs()).unwrap_or(u32::MAX));
    if n < 0 {
        -mag
    } else {
        mag
    }
}
