//! Exact (combinatorial) null distributions for the rank and goodness-of-fit
//! tests.
//!
//! Where the asymptotic paths in [`crate::tests_stat::nonparametric`] and
//! [`crate::tests_stat::goodness_of_fit`] use a normal or Kolmogorov
//! approximation, this module enumerates the exact null distribution so small
//! samples reproduce `scipy.stats`' `method="exact"` p-values to floating-point
//! precision. Each submodule owns one test's exact math:
//!
//! * [`mann_whitney`] — the exact Mann–Whitney U distribution.
//! * [`wilcoxon`] — the exact Wilcoxon signed-rank-sum distribution.
//! * [`ks`] — the exact one- and two-sample Kolmogorov–Smirnov distributions.
//!
//! These are low-level building blocks; most callers should use the `*_mode`
//! entry points on each test (e.g.
//! [`crate::tests_stat::nonparametric::mann_whitney_u_mode`]), which select
//! between this module and the asymptotic path via [`crate::tests_stat::Mode`].

pub mod ks;
pub mod mann_whitney;
pub mod wilcoxon;
