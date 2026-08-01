// Kani proof harnesses for the feature-selection base block, `include!`d into the
// `#[cfg(kani)] mod verification` block of `mod.rs` (kept in a separate file so
// `mod.rs` stays within the 500-line `tests/style.rs` limit). `super` here refers
// to the `feature_selection` module. Compiled only under `cargo kani`.

use super::{FeatureSelectionError, top_k_mask, validate_matrix};

/// Proves [`validate_matrix`] never panics and classifies a symbolic rectangular
/// `2×2` feature matrix correctly.
///
/// Every entry is fully symbolic (`NaN`/`±∞` permitted), so the finiteness scan is
/// exercised over all combinations. A rectangular two-by-two matrix is never empty,
/// has features, and is never ragged, so the only reachable outcomes are `Ok((2,2))`
/// (all four finite) and `Err(NonFinite)`; the harness pins both and rules the
/// shape-error variants unreachable for this shape.
#[kani::proof]
fn fs_validate_matrix_rejects_non_finite() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    let c: f64 = kani::any();
    let d: f64 = kani::any();
    let x = vec![vec![a, b], vec![c, d]];
    let all_finite = a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite();
    match validate_matrix(&x) {
        Ok((rows, cols)) => {
            assert!(all_finite, "Ok returned for a non-finite matrix");
            assert!(rows == 2 && cols == 2, "shape was not the expected 2x2");
        }
        Err(FeatureSelectionError::NonFinite) => {
            assert!(!all_finite, "NonFinite returned for an all-finite matrix");
        }
        Err(_) => assert!(false, "validate_matrix returned an unreachable variant"),
    }
}

/// Proves [`top_k_mask`] never panics and flags exactly `min(k, n)` features for a
/// symbolic three-feature score vector and a symbolic `k`.
///
/// The scores are symbolic finite values and `k` ranges over `0..=4` (which spans
/// every distinct `min(k, 3)` outcome: `0, 1, 2, 3`, plus the saturating `k > n`
/// case at `k = 4`). Because the ranked index list holds each of the `n` column
/// indices once, taking `min(k, n)` of them and flagging each yields exactly
/// `min(k, n)` distinct `true` entries regardless of how ties in the scores resolve.
/// Constraining `k` to its behaviour-distinct range (rather than all `2^64` values)
/// and the scores to finite (rather than branching the comparator on `NaN`) keeps
/// CBMC within memory; the `#[kani::unwind(6)]` unrolls the three-element insertion
/// sort and the mask loops.
#[kani::proof]
#[kani::unwind(6)]
fn fs_top_k_mask_selects_min_k_n() {
    let s0: f64 = kani::any();
    let s1: f64 = kani::any();
    let s2: f64 = kani::any();
    for s in [s0, s1, s2] {
        kani::assume(s.is_finite());
    }
    let k: usize = kani::any();
    kani::assume(k <= 4);
    let scores = [s0, s1, s2];
    let mask = top_k_mask(&scores, k);
    let selected = mask.iter().filter(|&&flag| flag).count();
    assert!(selected == k.min(3), "selected count was not min(k, n)");
}
