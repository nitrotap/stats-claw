//! Numeric helpers for the association-rule base block: `as`-free count widening.
//!
//! Kept separate from the mining API so `mod.rs` stays inside the project's
//! 500-line `style.rs` cap. Supports, confidences, and lifts are exact rational
//! ratios of integer transaction counts, so the only numeric primitive needed here
//! is a lossless `usize → f64` widening for the ratio denominators.

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Transaction and itemset counts here are far below `2^53`, so splitting into
/// 32-bit halves and recombining reproduces the value exactly while satisfying the
/// `style.rs` no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64` (exact for the counts this crate handles).
pub(super) fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The widening reproduces a representative count exactly.
    #[test]
    fn count_widens_exactly() {
        assert!((count_to_f64(4096) - 4096.0).abs() < 1e-12, "widening");
    }

    /// The widening reproduces zero exactly (empty transaction set guard).
    #[test]
    fn count_widens_zero() {
        assert!(count_to_f64(0).abs() < 1e-12, "zero widening");
    }
}
