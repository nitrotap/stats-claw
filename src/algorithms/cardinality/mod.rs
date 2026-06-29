//! `HyperLogLog` distinct-count (cardinality) estimation.
//!
//! `HyperLogLog` (Flajolet, Fusy, Gandouet & Meunier, 2007) estimates the number of
//! **distinct** elements in a stream using a fixed, tiny amount of memory — `m`
//! one-byte registers — rather than the `O(distinct)` memory an exact `HashSet`
//! count needs. Each element is hashed to a 64-bit word; the leading `p` bits pick
//! one of `m = 2^p` registers, and the register keeps the maximum, over all
//! elements routed to it, of `1 + (leading zeros of the remaining bits)`. Many
//! distinct elements push some register's run of leading zeros high, and the
//! harmonic mean of `2^register` across the registers estimates the cardinality.
//!
//! The estimator implemented here is the original 2007 form with its three regimes:
//!
//! * the **raw** estimate `E = α_m · m² / Σ_j 2^(−register_j)`;
//! * a **small-range** correction — when `E ≤ 2.5·m` and some registers are still
//!   zero, linear counting `m · ln(m / zeros)` is more accurate;
//! * a **large-range** correction near the 32-bit hash ceiling. This crate hashes
//!   to a full 64-bit word, so that ceiling is astronomically far away and the
//!   correction never fires in practice; it is implemented for completeness and
//!   documented as inert at 64-bit width.
//!
//! ## Accuracy
//!
//! `HyperLogLog` is an *approximate* counter: its relative standard error is
//! `≈ 1.04 / √m`, so precision `p` trades memory (`m = 2^p` bytes) for accuracy.
//! At `p = 14` (16 KiB) the standard error is about `0.81 %`. There is **no**
//! canonical library bit-match to assert against — the equivalence suite instead
//! pins the estimate against the **exact** distinct count (a ground truth computed
//! with a `HashSet`) and checks it lands inside a small multiple of HLL's
//! theoretical standard error.
//!
//! ## Determinism
//!
//! For a given precision and input multiset the estimate is fully deterministic:
//! the hash ([`hash`]) is a fixed bijective finalizer, register updates are
//! order-independent maxima, and the estimator is a pure function of the registers.
//!
//! This base block backs the `CardinalityEstimation` computational method.

mod estimate;
mod hash;

use estimate::estimate_from_registers;
use hash::hash64;

/// Number of bits in the hash word the register index and run length are read from.
const HASH_BITS: u32 = 64;

/// Smallest supported `HyperLogLog` precision (register-index bit width).
///
/// Below `p = 4` the bias-correction constant `α_m` and the small-range regime are
/// not well defined, so the original paper treats `m = 16` as the floor.
const MIN_PRECISION: u8 = 4;
/// Largest supported `HyperLogLog` precision.
///
/// `p = 18` is `m = 262_144` registers (256 KiB); beyond this the memory cost
/// outweighs the accuracy gain for this crate's use, so it is the supported ceiling.
const MAX_PRECISION: u8 = 18;

/// Errors that prevent constructing or updating a `HyperLogLog` estimator.
///
/// Returned (never panicked) so callers stay clear of the crate's `unwrap`/`panic`
/// lint gate and can surface a clean diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardinalityError {
    /// The requested precision was outside the supported `[4, 18]` range, so the
    /// register count `2^p` and the `α_m` correction would be ill-defined.
    InvalidPrecision,
}

impl std::fmt::Display for CardinalityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPrecision => {
                write!(
                    f,
                    "precision must lie in [{MIN_PRECISION}, {MAX_PRECISION}]"
                )
            }
        }
    }
}

impl std::error::Error for CardinalityError {}

/// A `HyperLogLog` distinct-count estimator over `m = 2^precision` byte registers.
///
/// Construct with [`HyperLogLog::new`], feed elements with [`HyperLogLog::add`]
/// (or build directly with [`HyperLogLog::from_u64_iter`]), then read the estimated
/// cardinality with [`HyperLogLog::estimate`]. The struct holds only the precision
/// and the register array, so it is cheap to clone and fully deterministic.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    /// Register-index bit width `p`; the register count is `2^p`.
    precision: u8,
    /// The `2^precision` registers, each holding the running maximum `1 + leading
    /// zeros` for the hashes routed to it. One byte each (max run length `< 64`).
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// Creates an empty estimator with `2^precision` zeroed registers.
    ///
    /// # Arguments
    ///
    /// * `precision` — the register-index bit width `p`; must lie in
    ///   `[4, 18]`. The register count is `m = 2^p` and the relative standard error
    ///   is `≈ 1.04 / √m`.
    ///
    /// # Returns
    ///
    /// An estimator whose estimate is `0` until elements are added.
    ///
    /// # Errors
    ///
    /// Returns [`CardinalityError::InvalidPrecision`] if `precision` is outside
    /// `[4, 18]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::cardinality::HyperLogLog;
    ///
    /// let mut hll = HyperLogLog::new(14)?;
    /// for x in 0_u64..1000 {
    ///     hll.add(x);
    /// }
    /// // 1000 distinct elements, estimated within HLL's ~0.8% standard error at p=14.
    /// let est = hll.estimate();
    /// assert!((est - 1000.0).abs() / 1000.0 < 0.05, "estimate was {est}");
    /// # Ok::<(), stats_claw::algorithms::cardinality::CardinalityError>(())
    /// ```
    pub fn new(precision: u8) -> Result<Self, CardinalityError> {
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&precision) {
            return Err(CardinalityError::InvalidPrecision);
        }
        let m = 1_usize << precision;
        Ok(Self {
            precision,
            registers: vec![0_u8; m],
        })
    }

    /// Builds an estimator at `precision` from an iterator of `u64` elements.
    ///
    /// Equivalent to [`HyperLogLog::new`] followed by [`HyperLogLog::add`] for each
    /// element, but reads as a single expression for the common "estimate the
    /// distinct count of this stream" call. Because the register update is an
    /// order-independent maximum, the result is independent of the iteration order.
    ///
    /// # Arguments
    ///
    /// * `precision` — the register-index bit width `p`; must lie in `[4, 18]`.
    /// * `values` — the elements to record; duplicates count once.
    ///
    /// # Returns
    ///
    /// An estimator populated with every element of `values`.
    ///
    /// # Errors
    ///
    /// Returns [`CardinalityError::InvalidPrecision`] if `precision` is outside
    /// `[4, 18]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::cardinality::HyperLogLog;
    ///
    /// // 200 distinct values, each seen twice — the distinct count is 200.
    /// let stream = (0_u64..200).chain(0_u64..200);
    /// let hll = HyperLogLog::from_u64_iter(12, stream)?;
    /// let est = hll.estimate();
    /// assert!((est - 200.0).abs() / 200.0 < 0.1, "estimate was {est}");
    /// # Ok::<(), stats_claw::algorithms::cardinality::CardinalityError>(())
    /// ```
    pub fn from_u64_iter<I>(precision: u8, values: I) -> Result<Self, CardinalityError>
    where
        I: IntoIterator<Item = u64>,
    {
        let mut hll = Self::new(precision)?;
        for value in values {
            hll.add(value);
        }
        Ok(hll)
    }

    /// The precision `p` (register-index bit width) this estimator was built with.
    #[must_use]
    pub const fn precision(&self) -> u8 {
        self.precision
    }

    /// The register count `m = 2^precision`.
    #[must_use]
    pub const fn register_count(&self) -> usize {
        self.registers.len()
    }

    /// Adds a 64-bit element to the estimator, updating one register.
    ///
    /// The element is scattered by the fixed finalizer ([`hash::hash64`]); the
    /// leading `precision` bits of the hash select the register, and the register is
    /// raised to `max(current, 1 + leading_zeros(remaining bits))`. The update is a
    /// maximum, so it is idempotent in the element (re-adding the same value never
    /// changes any register) and order-independent across distinct elements — which
    /// is what makes the estimate deterministic for a given input multiset.
    ///
    /// Non-`u64` elements are added by first reducing them to a `u64` key; integer
    /// streams pass their value directly.
    ///
    /// # Arguments
    ///
    /// * `value` — the element to record. Equal values are treated as one distinct
    ///   element; the count is over the *set* of values seen.
    pub fn add(&mut self, value: u64) {
        let hash = hash64(value);
        let p = u32::from(self.precision);
        // The leading `p` bits address the register: shift the rest away.
        let index = usize::try_from(hash >> (HASH_BITS - p)).unwrap_or(0);
        // The remaining `64 - p` bits carry the run-length signal. Shift the index
        // bits out (filling with zeros from the left); the rank is `1 + leading
        // zeros` of that tail, capped so a hash whose tail is all zeros still gives
        // a defined rank of `64 - p + 1`.
        let tail = hash << p;
        let rank = leading_zero_rank(tail, p);
        if let Some(slot) = self.registers.get_mut(index) {
            if rank > *slot {
                *slot = rank;
            }
        }
    }

    /// Estimates the number of distinct elements added so far.
    ///
    /// A pure function of the register array: the harmonic-mean raw estimate with
    /// the small-range (linear-counting) and large-range corrections applied as the
    /// register state warrants (see [`estimate`]). An estimator with nothing added
    /// returns `0` exactly.
    ///
    /// The result is an *approximation* with relative standard error `≈ 1.04 / √m`
    /// (`m = 2^precision`); it is not the exact count, and callers needing the true
    /// distinct count must use an exact structure instead.
    ///
    /// # Returns
    ///
    /// The estimated distinct-element count, `≥ 0`.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        estimate_from_registers(&self.registers)
    }
}

/// Computes the `HyperLogLog` register rank `1 + leading_zeros` for a hash tail.
///
/// `tail` is the hash with its `p` index bits already shifted out to the left, so
/// its own leading zeros count the run of zeros in the original `64 - p` tail bits.
/// When `tail` is entirely zero (`leading_zeros == 64`) the run spans the whole
/// `64 - p`-bit tail, giving the maximal rank `64 - p + 1`; the `min` clamps to that
/// so a register never exceeds the representable maximum.
///
/// # Arguments
///
/// * `tail` — the hash left-shifted by `p` (index bits removed).
/// * `p` — the precision; the tail carries `64 - p` meaningful bits.
///
/// # Returns
///
/// The register rank, in `1..=(64 - p + 1)`, as a `u8` (always `< 64`).
fn leading_zero_rank(tail: u64, p: u32) -> u8 {
    let tail_bits = HASH_BITS - p;
    // `leading_zeros` counts from the full 64-bit word; the tail's own run of zeros
    // is the same value because the index bits were shifted out as zeros only when
    // the tail itself led with zeros. Cap at the tail width so an all-zero tail maps
    // to the maximal rank rather than counting the shifted-in index zeros.
    let zeros = tail.leading_zeros().min(tail_bits);
    // rank = 1 + run length; `zeros + 1 <= 64 - p + 1 <= 61`, fits a u8.
    u8::try_from(zeros + 1).unwrap_or(u8::MAX)
}

#[cfg(test)]
impl HyperLogLog {
    /// Counts how many registers are non-zero (test introspection only).
    ///
    /// Lets unit tests assert that a single `add` touches exactly one register and
    /// that the idempotent maximum leaves the register population unchanged.
    fn nonzero_register_count(&self) -> usize {
        self.registers.iter().filter(|&&r| r != 0).count()
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
