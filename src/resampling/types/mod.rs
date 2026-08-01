//! Plain-data parameter structs for the library's statistical constructs.
//!
//! Each struct carries only the parameters (and any descriptive string fields)
//! that identify a construct; all numerics live in the behaviour traits and
//! inherent methods implemented for these types in the sibling modules. The
//! structs derive `Default` so callers build them with struct-update syntax and
//! set only the fields they care about.
//!
//! This file is produced mechanically by the `carve` tool from the source
//! project; edit the carve inputs rather than this file.

/// Partition-based scheme for estimating generalization performance.
#[derive(Debug, Clone, Default)]
pub struct CrossValidation {
    /// Number of folds.
    pub number_of_folds: i64,
    /// Whether folds are stratified.
    pub stratified: bool,
    /// Random seed.
    pub random_seed: i64,
    /// Unique name identifying a resampling scheme.
    pub scheme_name: String,
    /// Free-text description.
    pub description: String,
}

/// Leave-one-out resampling for bias and variance estimation.
#[derive(Debug, Clone, Default)]
pub struct JackknifeResampling {
    /// Unique name identifying a resampling scheme.
    pub scheme_name: String,
    /// Free-text description.
    pub description: String,
}

/// Cross-validation using a single observation per validation fold.
#[derive(Debug, Clone, Default)]
pub struct LeaveOneOutCrossValidation {
    /// Unique name identifying a resampling scheme.
    pub scheme_name: String,
    /// Free-text description.
    pub description: String,
}

/// Repeated random train-validation splits.
#[derive(Debug, Clone, Default)]
pub struct MonteCarloResampling {
    /// Number of iterations.
    pub number_of_iterations: i64,
    /// Training fraction.
    pub train_size: f64,
    /// Random seed.
    pub random_seed: i64,
    /// Unique name identifying a resampling scheme.
    pub scheme_name: String,
    /// Free-text description.
    pub description: String,
}

/// Cross-validation preserving class proportions across folds.
#[derive(Debug, Clone, Default)]
pub struct StratifiedCrossValidation {
    /// Number of folds.
    pub number_of_folds: i64,
    /// Random seed.
    pub random_seed: i64,
    /// Unique name identifying a resampling scheme.
    pub scheme_name: String,
    /// Free-text description.
    pub description: String,
}
