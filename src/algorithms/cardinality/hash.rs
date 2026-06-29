//! Deterministic 64-bit finalizer hash for `HyperLogLog` register addressing.
//!
//! Kept separate from the estimator so `mod.rs` stays inside the project's
//! 500-line `style.rs` cap. `HyperLogLog` needs a hash that scatters inputs as
//! uniformly as possible across the 64-bit space, because both the register index
//! (the leading `p` bits) and the leading-zero count (the trailing bits) are read
//! straight off the hash word; any clustering biases the estimate. A documented,
//! fixed finalizer also makes the estimator deterministic for a given input
//! multiset, which the equivalence suite relies on.
//!
//! The finalizer is the `SplitMix64` output stage (Steele, Lea & Flood 2014) — the
//! same avalanche mix the framework's [`crate::rng::SplitMix64`] uses — applied to
//! the raw `u64` key. It is a bijection on `u64`, so distinct inputs map to distinct
//! hashes (no collisions are introduced by the hash itself), and it passes the
//! avalanche property `HyperLogLog` assumes of its hash.

/// First avalanche multiplier from the `SplitMix64` finalizer (Steele et al. 2014).
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second avalanche multiplier from the `SplitMix64` finalizer (Steele et al. 2014).
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// Hashes a 64-bit key to a well-scattered 64-bit value via the `SplitMix64`
/// finalizer.
///
/// The finalizer xor-shifts and multiplies the key twice, then xor-shifts once
/// more, giving each input bit roughly equal influence over every output bit (the
/// avalanche property `HyperLogLog`'s accuracy guarantee assumes). It is a bijection
/// on `u64`: it introduces no collisions of its own, so two inputs collide only if
/// the caller's keys were already equal.
///
/// # Arguments
///
/// * `key` — the 64-bit value to hash (e.g. an integer element, or a key derived
///   from bytes by the caller).
///
/// # Returns
///
/// The scattered 64-bit hash of `key`.
///
/// # Notes
///
/// Internal helper exercised through the public estimator; its determinism and
/// injectivity are unit-tested in `tests.rs`. No doctest here because the function
/// is private to the family module and not reachable from a doc example.
pub(super) const fn hash64(key: u64) -> u64 {
    let mut z = key;
    z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
    z ^ (z >> 31)
}
