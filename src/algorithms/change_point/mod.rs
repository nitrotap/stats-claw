//! Change-point detection algorithms.
//!
//! The single member is PELT (Pruned Exact Linear Time), which finds the
//! penalized-optimal segmentation of a signal. It lives in the `pelt` submodule
//! and is re-exported here so callers name `change_point::pelt_l2` without the
//! submodule path.
//!
//! Change points are discrete indices, so the equivalence suite compares stats-claw's
//! breakpoints to `ruptures` for exact set equality — there is no tolerance for an
//! off-by-one segment boundary.

mod pelt;

pub use pelt::pelt_l2;
