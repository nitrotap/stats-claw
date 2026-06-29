//! Categorical hypothesis tests over contingency tables.
//!
//! Houses the Pearson chi-squared test of independence (the worked PATTERN) and,
//! as the track lands them, the Fisher exact test, the `McNemar` test, the
//! Cochran Q test, and the Cramér's V effect size with its bootstrap interval.
//! Each test computes its asymptotic p-value through the framework's null
//! distribution ([`crate::distributions::ChiSquaredDistribution`]) rather than a
//! re-derived series.

pub mod chi_squared_test;
pub mod cochran;
pub mod effect_size;
pub mod fisher;
pub mod mcnemar;

pub use chi_squared_test::chi_squared_independence;
pub use cochran::cochrans_q;
pub use effect_size::{cramers_v, cramers_v_bootstrap_ci};
pub use fisher::fisher_exact;
pub use mcnemar::mcnemar;
