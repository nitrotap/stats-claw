//! Per-family statistical-test equivalence modules.
//!
//! Each submodule loads its committed golden fixtures and asserts the matching
//! `stats_claw::tests_stat` function reproduces the scipy/statsmodels reference.

mod categorical;
mod correlation;
mod goodness_of_fit;
mod intervals;
mod logspace;
mod nonparametric;
mod parametric;
