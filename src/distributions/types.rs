//! Plain-data parameter structs for the supported probability distributions.
//!
//! Each struct carries only the parameters (and two descriptive string fields)
//! that identify a distribution; all numerics live in the behaviour traits
//! (`Pdf`/`Pmf`/`Cdf`/`Quantile`/`Moments`/`Sample`/`LogCdf`) implemented for
//! these types in the sibling subgroup modules. The structs derive `Default`
//! so callers can build them with struct-update syntax and only set the fields
//! they care about:
//!
//! ```
//! use stats_claw::distributions::NormalDistribution;
//!
//! let n = NormalDistribution { mean: 0.0, standard_deviation: 1.0, ..Default::default() };
//! assert_eq!(n.mean, 0.0);
//! ```
//!
//! The `distribution_name` and `description` fields are optional human-readable
//! labels; the numerics never read them, so they default to the empty string.

/// Parameters of a Beta distribution on a finite interval.
///
/// The Beta family models a random proportion or probability. Its density is
/// shaped by two positive parameters and supported on `[support_lower_bound,
/// support_upper_bound]` (the standard Beta uses `[0, 1]`).
#[derive(Debug, Clone, Default)]
pub struct BetaDistribution {
    /// First shape parameter `α`; must be `> 0`. Larger values push mass toward
    /// the upper bound.
    pub alpha_parameter: f64,
    /// Second shape parameter `β`; must be `> 0`. Larger values push mass toward
    /// the lower bound.
    pub beta_parameter: f64,
    /// Inclusive lower bound of the support; `0.0` for the standard Beta.
    pub support_lower_bound: f64,
    /// Inclusive upper bound of the support; `1.0` for the standard Beta.
    pub support_upper_bound: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Binomial distribution.
///
/// Models the number of successes in `number_of_trials` independent Bernoulli
/// trials, each succeeding with probability `success_probability`.
#[derive(Debug, Clone, Default)]
pub struct BinomialDistribution {
    /// Number of independent trials `n`; must be `≥ 0`.
    pub number_of_trials: i64,
    /// Per-trial success probability `p`; must lie in `[0, 1]`.
    pub success_probability: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Cauchy distribution.
///
/// A heavy-tailed, symmetric distribution with undefined mean and variance,
/// centred at `location` with half-width-at-half-maximum `scale`.
#[derive(Debug, Clone, Default)]
pub struct CauchyDistribution {
    /// Location parameter `x₀`; the median and mode of the distribution.
    pub location: f64,
    /// Scale parameter `γ`; must be `> 0`. The half-width at half-maximum.
    pub scale: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a chi-squared distribution.
///
/// The distribution of a sum of `degrees_of_freedom` squared standard normals;
/// the reference distribution for variance and goodness-of-fit tests.
#[derive(Debug, Clone, Default)]
pub struct ChiSquaredDistribution {
    /// Degrees of freedom `k`; must be `≥ 1`.
    pub degrees_of_freedom: i64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of an Exponential distribution.
///
/// Models the waiting time between events in a Poisson process; supported on
/// `[0, ∞)` and parameterised by its rate.
#[derive(Debug, Clone, Default)]
pub struct ExponentialDistribution {
    /// Rate parameter `λ`; must be `> 0`. The mean is `1/λ`.
    pub rate_parameter: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of an F distribution.
///
/// The ratio of two independent chi-squared variates each divided by their
/// degrees of freedom; the reference distribution for ANOVA and variance-ratio
/// tests.
#[derive(Debug, Clone, Default)]
pub struct FDistribution {
    /// Numerator degrees of freedom `d₁`; must be `≥ 1`.
    pub numerator_df: i64,
    /// Denominator degrees of freedom `d₂`; must be `≥ 1`.
    pub denominator_df: i64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Gamma distribution.
///
/// A flexible positive-support family generalising the exponential and
/// chi-squared distributions, in the shape/scale parameterisation.
#[derive(Debug, Clone, Default)]
pub struct GammaDistribution {
    /// Shape parameter `k`; must be `> 0`.
    pub shape_parameter: f64,
    /// Scale parameter `θ`; must be `> 0`. The mean is `k·θ`.
    pub scale_parameter: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Laplace (double-exponential) distribution.
///
/// A symmetric distribution with a sharp peak at `location` and exponentially
/// decaying tails governed by `scale`.
#[derive(Debug, Clone, Default)]
pub struct LaplaceDistribution {
    /// Location parameter `μ`; the median, mode, and mean.
    pub location: f64,
    /// Scale parameter `b`; must be `> 0`. The variance is `2·b²`.
    pub scale: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a log-normal distribution.
///
/// The distribution of a variable whose natural logarithm is normally
/// distributed; supported on `(0, ∞)` and parameterised on the log scale.
#[derive(Debug, Clone, Default)]
pub struct LogNormalDistribution {
    /// Mean `μ` of the underlying normal on the log scale.
    pub mean_log_value: f64,
    /// Standard deviation `σ` of the underlying normal on the log scale; `> 0`.
    pub std_log_value: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Normal (Gaussian) distribution.
///
/// The canonical bell-curve, fully specified by its `mean` and
/// `standard_deviation`; `variance` is carried as a convenience field.
#[derive(Debug, Clone, Default)]
pub struct NormalDistribution {
    /// Mean `μ`; the centre of the distribution.
    pub mean: f64,
    /// Standard deviation `σ`; must be `> 0`.
    pub standard_deviation: f64,
    /// Variance `σ²`; a convenience field, not required by the numerics.
    pub variance: f64,
    /// Optional parameterization label; unused by the numerics, defaults to empty.
    pub parameterization: String,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Poisson distribution.
///
/// Models the count of events occurring in a fixed interval at a constant
/// average rate; supported on the non-negative integers.
#[derive(Debug, Clone, Default)]
pub struct PoissonDistribution {
    /// Rate parameter `λ`; must be `> 0`. Both the mean and the variance.
    pub rate_parameter: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Student's t distribution.
///
/// A symmetric, heavier-tailed analogue of the normal used for inference on
/// means from small samples; converges to the normal as the degrees of freedom
/// grow.
#[derive(Debug, Clone, Default)]
pub struct TDistribution {
    /// Degrees of freedom `ν`; must be `≥ 1`.
    pub degrees_of_freedom: i64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a continuous Uniform distribution.
///
/// Constant density over the half-open interval `[lower_bound, upper_bound)`
/// and zero outside it.
#[derive(Debug, Clone, Default)]
pub struct UniformDistribution {
    /// Inclusive lower bound `a` of the support.
    pub lower_bound: f64,
    /// Exclusive upper bound `b` of the support; must satisfy `b > a`.
    pub upper_bound: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}

/// Parameters of a Weibull distribution.
///
/// A positive-support family widely used in reliability and survival analysis,
/// in the shape/scale parameterisation.
#[derive(Debug, Clone, Default)]
pub struct WeibullDistribution {
    /// Shape parameter `k`; must be `> 0`. Controls the hazard-rate trend.
    pub shape_parameter: f64,
    /// Scale parameter `λ`; must be `> 0`. The characteristic life.
    pub scale_parameter: f64,
    /// Optional human-readable name; unused by the numerics, defaults to empty.
    pub distribution_name: String,
    /// Optional human-readable description; unused by the numerics, defaults to empty.
    pub description: String,
}
