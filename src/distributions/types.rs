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

/// Beta distribution on a bounded support with two shape parameters.
#[derive(Debug, Clone, Default)]
pub struct BetaDistribution {
    /// Alpha shape parameter.
    pub alpha_parameter: f64,
    /// Beta shape parameter.
    pub beta_parameter: f64,
    /// Lower bound of distribution support.
    pub support_lower_bound: f64,
    /// Upper bound of distribution support.
    pub support_upper_bound: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Distribution of successes over a fixed number of Bernoulli trials.
#[derive(Debug, Clone, Default)]
pub struct BinomialDistribution {
    /// Number of trials.
    pub number_of_trials: i64,
    /// Probability of success per trial.
    pub success_probability: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Heavy-tailed distribution with location and scale and undefined mean.
#[derive(Debug, Clone, Default)]
pub struct CauchyDistribution {
    /// Location parameter.
    pub location: f64,
    /// Scale parameter.
    pub scale: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Chi-squared distribution parameterized by degrees of freedom.
#[derive(Debug, Clone, Default)]
pub struct ChiSquaredDistribution {
    /// Degrees of freedom.
    pub degrees_of_freedom: i64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Distribution of waiting times at a constant rate.
#[derive(Debug, Clone, Default)]
pub struct ExponentialDistribution {
    /// Rate parameter.
    pub rate_parameter: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// F distribution parameterized by numerator and denominator degrees of
#[derive(Debug, Clone, Default)]
pub struct FDistribution {
    /// Numerator degrees of freedom.
    pub numerator_df: i64,
    /// Denominator degrees of freedom.
    pub denominator_df: i64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Gamma distribution with shape and scale parameters.
#[derive(Debug, Clone, Default)]
pub struct GammaDistribution {
    /// Shape parameter.
    pub shape_parameter: f64,
    /// Scale parameter.
    pub scale_parameter: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Double-exponential distribution with location and scale.
#[derive(Debug, Clone, Default)]
pub struct LaplaceDistribution {
    /// Location parameter.
    pub location: f64,
    /// Scale parameter.
    pub scale: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Distribution whose logarithm is normally distributed.
#[derive(Debug, Clone, Default)]
pub struct LogNormalDistribution {
    /// Mean of the log values.
    pub mean_log_value: f64,
    /// Standard deviation of the log values.
    pub std_log_value: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Gaussian distribution parameterized by mean and standard deviation.
#[derive(Debug, Clone, Default)]
pub struct NormalDistribution {
    /// Mean parameter of a distribution.
    pub mean: f64,
    /// Standard deviation parameter.
    pub standard_deviation: f64,
    /// Variance parameter.
    pub variance: f64,
    /// Named parameterization scheme.
    pub parameterization: String,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Distribution of event counts at a fixed rate.
#[derive(Debug, Clone, Default)]
pub struct PoissonDistribution {
    /// Rate parameter.
    pub rate_parameter: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Student t distribution parameterized by degrees of freedom.
#[derive(Debug, Clone, Default)]
pub struct TDistribution {
    /// Degrees of freedom.
    pub degrees_of_freedom: i64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Uniform distribution over a bounded interval.
#[derive(Debug, Clone, Default)]
pub struct UniformDistribution {
    /// Lower bound value.
    pub lower_bound: f64,
    /// Upper bound value.
    pub upper_bound: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}

/// Weibull distribution with shape and scale parameters.
#[derive(Debug, Clone, Default)]
pub struct WeibullDistribution {
    /// Shape parameter.
    pub shape_parameter: f64,
    /// Scale parameter.
    pub scale_parameter: f64,
    /// Unique name identifying a distribution.
    pub distribution_name: String,
    /// Free-text description.
    pub description: String,
}
