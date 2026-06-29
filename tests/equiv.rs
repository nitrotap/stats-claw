//! Consolidated equivalence-suite entry point (AC-1 through AC-5).
//!
//! A single integration-test binary that pulls in every family's equivalence
//! suite as a submodule, so the protected `tests/` root stays within the
//! ten-file folder limit. Behaviour is identical to the former per-family
//! `*_equiv.rs` binaries — every test still runs and asserts the same fixtures.
//!
//! ## Module layout
//!
//! * [`common`] — the shared golden-fixture harness, hoisted to the crate root
//!   so `stat/*` (`use crate::common`) and `dist/mod.rs` (`super::common`)
//!   resolve against the same copy.
//! * [`dist`] — the distribution equivalence suite (pdf/pmf/cdf/ppf vs scipy,
//!   round-trip, moments, seeded sampling).
//! * [`stat`] — the statistical-test equivalence suite (statistic, p-value, df,
//!   effect size, edge cases, categorical bootstrap CIs).
//! * [`optimizers`], [`resampling`], [`algorithms`], [`algorithms_decomp`] — the
//!   remaining family suites, one submodule each.

#[path = "common/mod.rs"]
mod common;

#[path = "dist/mod.rs"]
mod dist;

#[path = "stat/mod.rs"]
mod stat;

#[path = "equiv/algorithms.rs"]
mod algorithms;
#[path = "equiv/algorithms_decomp.rs"]
mod algorithms_decomp;
#[path = "equiv/association.rs"]
mod association;
#[path = "equiv/cardinality.rs"]
mod cardinality;
#[path = "equiv/density.rs"]
mod density;
#[path = "equiv/feature_selection.rs"]
mod feature_selection;
#[path = "equiv/optimizers.rs"]
mod optimizers;
#[path = "equiv/outlier.rs"]
mod outlier;
#[path = "equiv/regression.rs"]
mod regression;
#[path = "equiv/resampling.rs"]
mod resampling;
