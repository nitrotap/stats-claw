// Kani proof harnesses for the two-way ANOVA input-validation layer, `include!`d
// into the `#[cfg(kani)] mod verification` block of `two_way_anova.rs` (kept in a
// separate file so `two_way_anova.rs` stays within the 500-line `tests/style.rs`
// limit). `super` here refers to the `two_way_anova` module. Compiled only under
// `cargo kani`.

use super::two_way_anova;
use crate::error::Error;

/// Proves two-way ANOVA rejects an empty grid via `Err` without panicking. The
/// leading `cells.first()` guard returns [`Error::InsufficientData`] before any
/// marginal-sum or F-tail arithmetic.
#[kani::proof]
// Live path returns Err before any loop; the unwind bound caps the dead
// transcendental-tail branches CBMC unwinds during model construction.
#[kani::unwind(2)]
fn two_way_anova_rejects_empty_grid() {
    let cells: [Vec<Vec<f64>>; 0] = [];
    assert!(
        matches!(two_way_anova(&cells), Err(Error::InsufficientData)),
        "empty two-way grid must reject with InsufficientData"
    );
}

/// Proves two-way ANOVA rejects a grid with a single factor-A level via `Err`
/// without panicking, for an arbitrary symbolic replicate value. The `a < 2` guard
/// returns [`Error::InsufficientData`] before the interior.
#[kani::proof]
// Live path returns Err before any loop; the unwind bound caps the dead
// transcendental-tail branches CBMC unwinds during model construction.
#[kani::unwind(2)]
fn two_way_anova_rejects_single_level() {
    let cells: [Vec<Vec<f64>>; 1] = [vec![vec![kani::any::<f64>()]]];
    assert!(
        matches!(two_way_anova(&cells), Err(Error::InsufficientData)),
        "single factor-A level must reject with InsufficientData"
    );
}
