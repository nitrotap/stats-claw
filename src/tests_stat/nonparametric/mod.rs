//! Nonparametric rank-based hypothesis tests.
//!
//! Houses Mann–Whitney U, Kruskal–Wallis H, the Wilcoxon signed-rank test, and
//! the Friedman test. Each ranks its inputs with the shared mid-rank convention
//! ([`crate::tests_stat::ranks`]) and takes its p-value from the matching
//! framework null distribution (normal approximation or chi-squared).

pub mod friedman;
pub mod kruskal;
pub mod mann_whitney;
pub mod wilcoxon;

pub use friedman::friedman;
pub use kruskal::kruskal_wallis;
pub use mann_whitney::{mann_whitney_u, mann_whitney_u_mode};
pub use wilcoxon::{wilcoxon_signed_rank, wilcoxon_signed_rank_mode};

use crate::distributions::Cdf;
use crate::distributions::NormalDistribution;

/// Standard-normal CDF `P(Z ≤ z)` via the framework normal distribution.
///
/// Rank tests using a normal approximation (Mann–Whitney, Wilcoxon) take their
/// p-value from this helper so they share the framework [`NormalDistribution`]
/// CDF.
///
/// # Arguments
///
/// * `z` — the standardized statistic.
///
/// # Returns
///
/// The lower-tail probability in `[0, 1]`.
pub(crate) fn normal_cdf(z: f64) -> f64 {
    let standard = NormalDistribution {
        mean: 0.0,
        standard_deviation: 1.0,
        ..Default::default()
    };
    standard.cdf(z).clamp(0.0, 1.0)
}
