//! Correlation tests (Pearson, Spearman, Kendall).
//!
//! Each routine reports the correlation coefficient as its statistic and a
//! p-value for the chosen [`crate::tests_stat::Alternative`]. Pearson and Spearman
//! take the p-value from the framework Student's t distribution
//! (`t = r·√((n−2)/(1−r²))`); Kendall uses the exact null distribution when the
//! data has no ties (matching scipy's default), falling back to the tie-corrected
//! normal approximation.

pub mod coefficients;
mod kendall_exact;

pub use coefficients::{kendall, pearson, spearman};
