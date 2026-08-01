// Kani proof harnesses for the ziggurat index derivation, `include!`d into the
// `#[cfg(kani)] mod verification` block of `ziggurat.rs` (kept in a separate file
// so `ziggurat.rs` stays within the 500-line `tests/style.rs` limit). `super`
// here refers to the `ziggurat` module. Compiled only under `cargo kani`.

use super::{SplitMix64, ZIG_F, ZIG_K, ZIG_N, ZIG_W, fast_candidate, unpack, ziggurat_fixup};

/// `2^31`: the 31-bit magnitude ceiling `unpack` masks the drawn word down to.
const TWO_POW_31: u32 = 0x8000_0000;

/// Proves [`unpack`] maps *every* 64-bit word to a `(sign, layer, j)` triple
/// whose `layer ∈ [0, ZIG_N)` and `j < 2^31`, with `sign = ±1`. This is the
/// root of the table-bounds guarantee: `layer = low & (ZIG_N − 1)` cannot
/// exceed `ZIG_N − 1` for any input.
#[kani::proof]
fn ziggurat_unpack_indices_in_range() {
    let word: u64 = kani::any();
    let (sign, layer, j) = unpack(word);
    assert!(layer < ZIG_N, "layer {layer} escaped [0, ZIG_N)");
    assert!(j < TWO_POW_31, "magnitude j escaped the 31-bit range");
    assert!(sign == 1.0 || sign == -1.0, "sign was neither +1 nor -1");
}

/// Proves the layer from any symbolic word indexes all three ziggurat tables
/// in bounds (`ZIG_K`, `ZIG_W`, `ZIG_F` each have `ZIG_N` entries), and that
/// `layer − 1` — the upper-edge density index used by the wedge fixup — is in
/// bounds whenever `layer > 0`.
#[kani::proof]
fn ziggurat_table_indices_in_bounds() {
    let word: u64 = kani::any();
    let (_, layer, _) = unpack(word);
    assert!(ZIG_K.get(layer).is_some(), "ZIG_K index out of bounds");
    assert!(ZIG_W.get(layer).is_some(), "ZIG_W index out of bounds");
    assert!(ZIG_F.get(layer).is_some(), "ZIG_F index out of bounds");
    if layer > 0 {
        assert!(
            ZIG_F.get(layer - 1).is_some(),
            "ZIG_F[layer-1] index out of bounds"
        );
    }
}

/// Proves [`fast_candidate`] — the vectorized fast-path arithmetic the SIMD
/// lanes run — is panic-free for every symbolic word and produces a finite
/// candidate (the table entries are finite and `j` is bounded).
#[kani::proof]
fn ziggurat_fast_candidate_finite() {
    let word: u64 = kani::any();
    let (candidate, _accepted) = fast_candidate(word);
    assert!(candidate.is_finite(), "fast-path candidate was non-finite");
}

/// Proves the wedge branch of [`ziggurat_fixup`] (`layer ∈ [1, ZIG_N)`) is
/// panic-/overflow-free for a symbolic generator state and any in-range
/// `(layer, j)`. The tail branch (`layer == 0`) is an unbounded
/// rejection-sampling loop and is deliberately excluded here (see the report);
/// this harness verifies the bounded wedge path the fast-path miss falls into.
#[kani::proof]
#[kani::unwind(2)]
fn ziggurat_wedge_fixup_no_panic() {
    let state: u64 = kani::any();
    let mut rng = SplitMix64::new(state);
    let layer: usize = kani::any();
    kani::assume(layer >= 1);
    kani::assume(layer < ZIG_N);
    let j: u32 = kani::any();
    kani::assume(j < TWO_POW_31);
    let sign: f64 = if kani::any() { 1.0 } else { -1.0 };
    let _ = ziggurat_fixup(&mut rng, sign, layer, j);
}
