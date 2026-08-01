//! Univariate outlier / anomaly detection: z-score, IQR (Tukey fence), and
//! modified (MAD-based) z-score detectors.
//!
//! Each detector scores a one-dimensional sample and flags the points that fall
//! outside a chosen rule, returning both the per-point scores and the boolean
//! outlier mask so a caller can threshold or inspect either. The numerics pin the
//! reference-library conventions the equivalence suite checks:
//!
//! * the **z-score** uses the population standard deviation (`ddof = 0`), matching
//!   `scipy.stats.zscore`;
//! * the **IQR / Tukey-fence** quartiles use `numpy.percentile`'s default
//!   `'linear'` interpolation, so the fences `Q1 − k·IQR .. Q3 + k·IQR` agree with
//!   numpy exactly;
//! * the **modified z-score** uses the median and the median absolute deviation
//!   (MAD) with the conventional `0.6745` consistency constant (Iglewicz & Hoaglin).
//!
//! These base blocks back both the `OutlierDetection` and `AnomalyDetection`
//! computational methods.

mod stats;

use stats::{mean, median, median_absolute_deviation, percentile_linear, population_std};

/// The consistency constant `Φ⁻¹(0.75) ≈ 0.6745` scaling the MAD so the modified
/// z-score is comparable to a standard z-score for normally distributed data
/// (Iglewicz & Hoaglin, 1993).
const MAD_CONSISTENCY: f64 = 0.674_5;

/// The result of running a detector over a sample: the per-point scores and the
/// boolean outlier mask, aligned to the input order.
///
/// Returned by every detector so a caller can threshold, rank, or simply read off
/// which observations were flagged. The two vectors always share the input
/// length; `mask[i]` is the flag for the point with score `scores[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Per-point detector score, in input order. The score's meaning depends on
    /// the detector: a signed z-score, a signed modified z-score, or the signed
    /// distance past the nearer Tukey fence (`0.0` inside the fences).
    scores: Vec<f64>,
    /// Per-point outlier flag, in input order: `true` where the point was flagged.
    mask: Vec<bool>,
}

impl Detection {
    /// The per-point detector scores, in input order.
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// The per-point outlier mask, in input order (`true` = flagged).
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// The number of points flagged as outliers.
    #[must_use]
    pub fn outlier_count(&self) -> usize {
        self.mask.iter().filter(|&&flagged| flagged).count()
    }
}

/// Errors that prevent a detector from scoring a sample.
///
/// Returned (never panicked) so callers stay clear of the crate's `unwrap`/`panic`
/// lint gate and can surface a clean diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutlierError {
    /// The sample was empty, so nothing can be scored.
    EmptyInput,
    /// An observation was non-finite (`NaN` or `±∞`), which has no valid score.
    NonFinite,
    /// The requested threshold (or fence multiplier) was not a finite, positive
    /// value, so the flagging rule would be ill-defined.
    InvalidThreshold,
    /// The sample's spread collapsed to zero (a constant sample for the z-score,
    /// or a zero MAD for the modified z-score), so a scale-normalised score is
    /// undefined.
    ZeroSpread,
}

impl std::fmt::Display for OutlierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "sample has no observations"),
            Self::NonFinite => write!(f, "sample contains a non-finite value"),
            Self::InvalidThreshold => write!(f, "threshold must be finite and positive"),
            Self::ZeroSpread => write!(f, "sample has zero spread (constant or zero MAD)"),
        }
    }
}

impl std::error::Error for OutlierError {}

/// Validates a sample is non-empty and finite, and a threshold is finite-positive.
///
/// Shared front-door check so every detector rejects the same degenerate inputs
/// before any arithmetic runs.
///
/// # Arguments
///
/// * `data` — the sample to validate.
/// * `threshold` — the flagging threshold / fence multiplier to validate.
///
/// # Errors
///
/// Returns [`OutlierError::EmptyInput`], [`OutlierError::NonFinite`], or
/// [`OutlierError::InvalidThreshold`].
fn validate(data: &[f64], threshold: f64) -> Result<(), OutlierError> {
    if data.is_empty() {
        return Err(OutlierError::EmptyInput);
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(OutlierError::NonFinite);
    }
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(OutlierError::InvalidThreshold);
    }
    Ok(())
}

/// Flags outliers by the standard z-score `z = (x − mean) / std` (`|z| > threshold`).
///
/// Uses the population standard deviation (`ddof = 0`), so the returned scores
/// reproduce `scipy.stats.zscore` exactly; the mask flags every point whose
/// absolute z-score strictly exceeds `threshold` (3.0 is the textbook default).
///
/// # Arguments
///
/// * `data` — the one-dimensional sample to score; must be non-empty and finite.
/// * `threshold` — the absolute z-score above which a point is flagged; must be
///   finite and `> 0`.
///
/// # Returns
///
/// A [`Detection`] whose `scores` are the signed z-scores and whose `mask` flags
/// `|z| > threshold`.
///
/// # Errors
///
/// Returns [`OutlierError::EmptyInput`], [`OutlierError::NonFinite`],
/// [`OutlierError::InvalidThreshold`], or [`OutlierError::ZeroSpread`] (constant
/// sample).
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::outlier::zscore_detect;
///
/// // One point sits far from the rest; with threshold 1.5 it is flagged.
/// // (With only five points the maximal |z| caps near 2.0, so 1.5 isolates it.)
/// let det = zscore_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 1.5)?;
/// assert_eq!(det.mask(), &[false, false, false, false, true]);
/// // The flagged point's z-score reproduces scipy.stats.zscore.
/// let z = det.scores().last().copied().unwrap_or(f64::NAN);
/// assert!((z - 1.999_342_861_818_962_6).abs() < 1e-9, "z was {z}");
/// # Ok::<(), stats_claw::algorithms::outlier::OutlierError>(())
/// ```
pub fn zscore_detect(data: &[f64], threshold: f64) -> Result<Detection, OutlierError> {
    validate(data, threshold)?;
    let std = population_std(data);
    if std <= 0.0 {
        return Err(OutlierError::ZeroSpread);
    }
    let m = mean(data);
    let scores: Vec<f64> = data.iter().map(|&x| (x - m) / std).collect();
    let mask: Vec<bool> = scores.iter().map(|z| z.abs() > threshold).collect();
    Ok(Detection { scores, mask })
}

/// Flags outliers by the Tukey fence `Q1 − k·IQR .. Q3 + k·IQR` (the IQR rule).
///
/// The quartiles use `numpy.percentile`'s default `'linear'` interpolation, so the
/// fences agree with a numpy reference exactly. A point is flagged when it falls
/// strictly outside the closed fence interval; its score is the signed distance to
/// the nearer fence (`0.0` inside the fences, positive above the upper fence,
/// negative below the lower fence).
///
/// # Arguments
///
/// * `data` — the one-dimensional sample to score; must be non-empty and finite.
/// * `k` — the fence multiplier (`1.5` is Tukey's classic value, `3.0` flags only
///   "far" outliers); must be finite and `> 0`.
///
/// # Returns
///
/// A [`Detection`] whose `scores` are the signed distances past the nearer fence
/// and whose `mask` flags points outside the fences.
///
/// # Errors
///
/// Returns [`OutlierError::EmptyInput`], [`OutlierError::NonFinite`], or
/// [`OutlierError::InvalidThreshold`] (non-finite or non-positive `k`).
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::outlier::iqr_detect;
///
/// // The extreme value lies far above the upper Tukey fence (k = 1.5).
/// let det = iqr_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 1.5)?;
/// assert_eq!(det.mask(), &[false, false, false, false, true]);
/// assert_eq!(det.outlier_count(), 1);
/// # Ok::<(), stats_claw::algorithms::outlier::OutlierError>(())
/// ```
pub fn iqr_detect(data: &[f64], k: f64) -> Result<Detection, OutlierError> {
    validate(data, k)?;
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile_linear(&sorted, 0.25);
    let q3 = percentile_linear(&sorted, 0.75);
    let iqr = q3 - q1;
    let lower = k.mul_add(-iqr, q1);
    let upper = k.mul_add(iqr, q3);
    let scores: Vec<f64> = data
        .iter()
        .map(|&x| {
            if x > upper {
                x - upper
            } else if x < lower {
                x - lower
            } else {
                0.0
            }
        })
        .collect();
    let mask: Vec<bool> = data.iter().map(|&x| x < lower || x > upper).collect();
    Ok(Detection { scores, mask })
}

/// Flags outliers by the modified (MAD-based) z-score `M = 0.6745·(x − med)/MAD`.
///
/// Robust alternative to the mean/std z-score: it uses the median and the median
/// absolute deviation, which a few extreme points cannot inflate, so the very
/// outliers being sought do not mask themselves. A point is flagged when
/// `|M| > threshold` (Iglewicz & Hoaglin recommend `3.5`).
///
/// # Arguments
///
/// * `data` — the one-dimensional sample to score; must be non-empty and finite.
/// * `threshold` — the absolute modified-z above which a point is flagged; must be
///   finite and `> 0`.
///
/// # Returns
///
/// A [`Detection`] whose `scores` are the signed modified z-scores and whose
/// `mask` flags `|M| > threshold`.
///
/// # Errors
///
/// Returns [`OutlierError::EmptyInput`], [`OutlierError::NonFinite`],
/// [`OutlierError::InvalidThreshold`], or [`OutlierError::ZeroSpread`] (zero MAD,
/// i.e. more than half the sample shares the median).
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::outlier::modified_zscore_detect;
///
/// // The MAD-based score flags the extreme point at the 3.5 threshold.
/// let det = modified_zscore_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 3.5)?;
/// assert_eq!(det.mask(), &[false, false, false, false, true]);
/// # Ok::<(), stats_claw::algorithms::outlier::OutlierError>(())
/// ```
pub fn modified_zscore_detect(data: &[f64], threshold: f64) -> Result<Detection, OutlierError> {
    validate(data, threshold)?;
    let mad = median_absolute_deviation(data);
    if mad <= 0.0 {
        return Err(OutlierError::ZeroSpread);
    }
    let med = median(data);
    let scale = MAD_CONSISTENCY / mad;
    let scores: Vec<f64> = data.iter().map(|&x| (x - med) * scale).collect();
    let mask: Vec<bool> = scores.iter().map(|m| m.abs() > threshold).collect();
    Ok(Detection { scores, mask })
}

/// Kani proof harness for the shared outlier-detector input validation.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy.
#[cfg(kani)]
mod verification {
    use super::{OutlierError, validate};

    /// Proves the shared [`validate`] front door never panics and returns exactly
    /// the right classification for a symbolic two-observation sample and a symbolic
    /// threshold.
    ///
    /// Both observations and the threshold are fully symbolic (`NaN`/`±∞`
    /// permitted), so this covers every finiteness combination the detectors can be
    /// handed. `Ok` is proved to imply both observations are finite *and* the
    /// threshold is finite and positive — i.e. no degenerate input slips past the
    /// guard into the arithmetic.
    #[kani::proof]
    fn outlier_validate_rejects_degenerate() {
        let x0: f64 = kani::any();
        let x1: f64 = kani::any();
        let threshold: f64 = kani::any();
        let data = [x0, x1];
        match validate(&data, threshold) {
            Ok(()) => {
                assert!(
                    x0.is_finite() && x1.is_finite(),
                    "Ok passed a non-finite sample"
                );
                assert!(
                    threshold.is_finite() && threshold > 0.0,
                    "Ok passed a non-positive threshold"
                );
            }
            Err(OutlierError::NonFinite) => {
                assert!(
                    !x0.is_finite() || !x1.is_finite(),
                    "NonFinite on a finite sample"
                );
            }
            Err(OutlierError::InvalidThreshold) => {
                assert!(
                    !threshold.is_finite() || threshold <= 0.0,
                    "InvalidThreshold on a valid threshold"
                );
            }
            Err(OutlierError::EmptyInput | OutlierError::ZeroSpread) => {
                // A two-element sample is never empty, and `validate` never inspects
                // spread; reaching either here would be a control-flow bug.
                assert!(false, "validate returned an unreachable variant");
            }
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
