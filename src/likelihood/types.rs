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

/// Likelihood for success counts under a binomial model.
#[derive(Debug, Clone, Default)]
pub struct BinomialLikelihood {
    /// Number of trials.
    pub number_of_trials: i64,
    /// Unique name identifying a likelihood function.
    pub function_name: String,
    /// Free-text description.
    pub description: String,
    /// Mathematical formula.
    pub formula: String,
}

/// Likelihood for categorical outcomes.
#[derive(Debug, Clone, Default)]
pub struct CategoricalLikelihood {
    /// Number of categories.
    pub number_of_categories: i64,
    /// Unique name identifying a likelihood function.
    pub function_name: String,
    /// Free-text description.
    pub description: String,
    /// Mathematical formula.
    pub formula: String,
}

/// Likelihood for waiting-time data under an exponential model.
#[derive(Debug, Clone, Default)]
pub struct ExponentialLikelihood {
    /// Unique name identifying a likelihood function.
    pub function_name: String,
    /// Free-text description.
    pub description: String,
    /// Mathematical formula.
    pub formula: String,
}

/// General maximum-likelihood estimation procedure.
#[derive(Debug, Clone, Default)]
pub struct MaximumLikelihood {
    /// Estimation method.
    pub estimation_method: String,
    /// Convergence tolerance.
    pub convergence_tolerance: f64,
    /// Unique name identifying a likelihood function.
    pub function_name: String,
    /// Free-text description.
    pub description: String,
    /// Mathematical formula.
    pub formula: String,
}

/// Likelihood for data under a normal model.
#[derive(Debug, Clone, Default)]
pub struct NormalLikelihood {
    /// Whether the mean is parameterized.
    pub mean_parameterized: bool,
    /// Whether the variance is parameterized.
    pub variance_parameterized: bool,
    /// Unique name identifying a likelihood function.
    pub function_name: String,
    /// Free-text description.
    pub description: String,
    /// Mathematical formula.
    pub formula: String,
}

/// Likelihood for count data under a Poisson model.
#[derive(Debug, Clone, Default)]
pub struct PoissonLikelihood {
    /// Unique name identifying a likelihood function.
    pub function_name: String,
    /// Free-text description.
    pub description: String,
    /// Mathematical formula.
    pub formula: String,
}
