//! The `HyperLogLog` estimator core: the `α_m` bias constant, the harmonic-mean raw
//! estimate, and the small-range / large-range corrections.
//!
//! Split out of `mod.rs` to keep the public API file inside the project's 500-line
//! `style.rs` cap. The functions here are pure over the register array, so the whole
//! estimate is a deterministic function of `(precision, registers)`.

/// `2^32`, the hash-space size used by the original paper's large-range correction.
const TWO_POW_32: f64 = 4_294_967_296.0;
/// `2^32` again, named for the large-range branch's readability.
const HASH_SPACE_32: f64 = TWO_POW_32;

/// Returns the `HyperLogLog` bias-correction constant `α_m` for `m` registers.
///
/// The raw harmonic-mean estimate is biased; `α_m` (Flajolet et al. 2007, Fig. 3)
/// removes the leading-order bias. The paper gives closed forms for the small
/// register counts and the limit `α_m → 0.7213 / (1 + 1.079/m)` for `m ≥ 128`.
///
/// # Arguments
///
/// * `m` — the register count `2^precision` (always a power of two `≥ 16` here).
///
/// # Returns
///
/// The multiplicative bias-correction constant for that register count.
pub(super) fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        // m >= 128: the asymptotic form the paper specifies.
        _ => 0.7213 / (1.0 + 1.079 / count_to_f64(m)),
    }
}

/// Estimates the distinct count from the register array.
///
/// Implements the original three-regime estimator:
///
/// 1. the **raw** estimate `E = α_m · m² / Σ_j 2^(−register_j)` (harmonic mean of
///    `2^register`);
/// 2. the **small-range** correction: when `E ≤ 2.5·m` and `z` registers are still
///    zero, linear counting `m · ln(m / z)` is used instead (it is far more accurate
///    when many registers are empty);
/// 3. the **large-range** correction near the `2^32` hash ceiling. Because this
///    crate hashes to 64 bits, `E` never approaches `2^32` in practice, so this
///    branch is effectively inert — it is included for completeness and documented
///    as such.
///
/// # Arguments
///
/// * `registers` — the `m` register values (each `1 + leading-zero run`, or `0`).
///
/// # Returns
///
/// The estimated number of distinct elements, `≥ 0`. An all-zero register array
/// yields `0` exactly (linear counting with `z == m`).
pub(super) fn estimate_from_registers(registers: &[u8]) -> f64 {
    let m = count_to_f64(registers.len());
    if m <= 0.0 {
        return 0.0;
    }

    // One pass over the registers: accumulate both the harmonic-mean denominator
    // `Σ 2^(-register_j)` and the empty-register count the small-range regime needs.
    let mut sum = 0.0_f64;
    let mut zeros = 0_usize;
    for &r in registers {
        sum += 2.0_f64.powi(-i32::from(r));
        if r == 0 {
            zeros += 1;
        }
    }
    let raw = alpha_m(registers.len()) * m * m / sum;

    // Small-range regime: prefer linear counting while registers remain empty.
    if raw <= 2.5 * m && zeros > 0 {
        let z = count_to_f64(zeros);
        return m * (m / z).ln();
    }

    // Large-range regime: correct toward the 2^32 hash ceiling. Inert at 64-bit
    // hash width (raw never approaches 2^32 here), kept for completeness.
    if raw > HASH_SPACE_32 / 30.0 {
        return -HASH_SPACE_32 * (1.0 - raw / HASH_SPACE_32).ln();
    }

    raw
}

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Register counts and zero counts here are far below `2^53`, so splitting into
/// 32-bit halves and recombining reproduces the value exactly while satisfying the
/// `style.rs` no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64` (exact for the counts this module handles).
pub(super) fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(TWO_POW_32, f64::from(lo))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `α_m` matches the paper's tabulated small-`m` constants.
    #[test]
    fn alpha_m_matches_paper_constants() {
        assert!((alpha_m(16) - 0.673).abs() < 1e-12, "alpha_16");
        assert!((alpha_m(32) - 0.697).abs() < 1e-12, "alpha_32");
        assert!((alpha_m(64) - 0.709).abs() < 1e-12, "alpha_64");
    }

    /// `α_m` for large `m` uses the asymptotic form.
    #[test]
    fn alpha_m_uses_asymptotic_form_for_large_m() {
        let got = alpha_m(1024);
        let want = 0.7213 / (1.0 + 1.079 / 1024.0);
        assert!((got - want).abs() < 1e-12, "alpha_1024 was {got}");
    }

    /// An all-zero register array estimates exactly zero distinct elements.
    #[test]
    fn all_zero_registers_estimate_zero() {
        let regs = vec![0_u8; 1024];
        let est = estimate_from_registers(&regs);
        assert!(est.abs() < 1e-9, "all-zero estimate should be 0, was {est}");
    }

    /// The count widening reproduces a representative value exactly.
    #[test]
    fn count_widens_exactly() {
        assert!((count_to_f64(16_384) - 16_384.0).abs() < 1e-12, "widening");
    }
}
