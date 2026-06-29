//! Univariate feature selection: variance-threshold filtering and the ANOVA
//! F-test (`f_classif`) univariate score.
//!
//! Both selectors operate on a feature matrix `x` given as one inner `Vec<f64>`
//! per **sample** (row), with `x[i][j]` the value of feature `j` for sample `i`;
//! every row must share the same feature count. They return a per-feature result
//! plus a boolean selection mask so a caller can threshold, rank, or inspect the
//! scores directly. The numerics pin the reference-library conventions the
//! equivalence suite checks:
//!
//! * **variance threshold** drops features whose **population** variance
//!   (`ddof = 0`) is `≤ threshold`, keeping `variance > threshold`, matching
//!   `sklearn.feature_selection.VarianceThreshold`;
//! * the **ANOVA F-test** computes, per feature, the one-way ANOVA F-statistic
//!   across the labelled classes and its upper-tail p-value, reproducing
//!   `sklearn.feature_selection.f_classif` (which is itself a per-feature one-way
//!   ANOVA). The selection keeps the top-`k` features by F-score.
//!
//! This base block backs the `FeatureSelection` computational method.

mod stats;

use crate::tests_stat::parametric::one_way_anova;
use stats::population_variance;

/// Errors that prevent a selector from scoring a feature matrix.
///
/// Returned (never panicked) so callers stay clear of the crate's `unwrap`/`panic`
/// lint gate and can surface a clean diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureSelectionError {
    /// The feature matrix had no samples (no rows), so nothing can be scored.
    EmptyInput,
    /// The matrix had rows but no feature columns, so there is nothing to select.
    NoFeatures,
    /// The rows had differing feature counts (a ragged matrix), so columns are
    /// undefined.
    RaggedRows,
    /// A matrix entry was non-finite (`NaN` or `±∞`), which has no valid score.
    NonFinite,
    /// The supplied threshold was not finite, so the keep rule would be ill-defined.
    InvalidThreshold,
    /// The label vector's length did not match the number of samples (rows).
    LabelLengthMismatch,
    /// Fewer than two distinct classes were present, so a between-class F-test is
    /// undefined.
    TooFewClasses,
    /// A class had fewer observations than the F-test requires, or the
    /// within-class variation was zero, so the ANOVA F-statistic is undefined.
    DegenerateClasses,
}

impl std::fmt::Display for FeatureSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "feature matrix has no samples"),
            Self::NoFeatures => write!(f, "feature matrix has no feature columns"),
            Self::RaggedRows => write!(f, "feature matrix rows have differing lengths"),
            Self::NonFinite => write!(f, "feature matrix contains a non-finite value"),
            Self::InvalidThreshold => write!(f, "variance threshold must be finite"),
            Self::LabelLengthMismatch => {
                write!(f, "label count does not match the number of samples")
            }
            Self::TooFewClasses => write!(f, "fewer than two distinct classes present"),
            Self::DegenerateClasses => {
                write!(f, "a class is too small or has zero within-class variation")
            }
        }
    }
}

impl std::error::Error for FeatureSelectionError {}

/// The result of running a univariate selector over a feature matrix: the
/// per-feature scores and the boolean selection mask, aligned to column order.
///
/// The `scores` vector holds one value per feature column (a variance, or an
/// ANOVA F-statistic), and `mask[j]` flags whether feature `j` was selected.
/// The two vectors always share the feature count.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    /// Per-feature score, in column order. The score's meaning depends on the
    /// selector: the population variance, or the ANOVA F-statistic.
    scores: Vec<f64>,
    /// Per-feature selection flag, in column order: `true` where the feature was
    /// kept.
    mask: Vec<bool>,
}

impl Selection {
    /// The per-feature scores, in column order.
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// The per-feature selection mask, in column order (`true` = kept).
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// The number of features selected.
    #[must_use]
    pub fn selected_count(&self) -> usize {
        self.mask.iter().filter(|&&kept| kept).count()
    }

    /// The zero-based indices of the selected features, in ascending column order.
    #[must_use]
    pub fn selected_indices(&self) -> Vec<usize> {
        self.mask
            .iter()
            .enumerate()
            .filter_map(|(j, &kept)| kept.then_some(j))
            .collect()
    }
}

/// Validates a feature matrix is non-empty, rectangular, and finite, returning the
/// `(n_samples, n_features)` shape.
///
/// Shared front-door check so every selector rejects the same degenerate inputs
/// before any arithmetic runs.
///
/// # Arguments
///
/// * `x` — the feature matrix, one inner slice per sample.
///
/// # Returns
///
/// `(n_samples, n_features)` on success.
///
/// # Errors
///
/// Returns [`FeatureSelectionError::EmptyInput`],
/// [`FeatureSelectionError::NoFeatures`], [`FeatureSelectionError::RaggedRows`],
/// or [`FeatureSelectionError::NonFinite`].
fn validate_matrix(x: &[Vec<f64>]) -> Result<(usize, usize), FeatureSelectionError> {
    let n_samples = x.len();
    if n_samples == 0 {
        return Err(FeatureSelectionError::EmptyInput);
    }
    let n_features = x.first().map_or(0, Vec::len);
    if n_features == 0 {
        return Err(FeatureSelectionError::NoFeatures);
    }
    if x.iter().any(|row| row.len() != n_features) {
        return Err(FeatureSelectionError::RaggedRows);
    }
    if x.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
        return Err(FeatureSelectionError::NonFinite);
    }
    Ok((n_samples, n_features))
}

/// Extracts column `j` of a validated feature matrix as an owned vector.
///
/// The matrix is row-major (one inner slice per sample), so this gathers the
/// `j`-th entry of every row into the feature's observation vector. `j` must be a
/// valid column index (the caller iterates `0..n_features`).
///
/// # Arguments
///
/// * `x` — the validated, rectangular feature matrix.
/// * `j` — the zero-based column index to extract.
///
/// # Returns
///
/// The values of feature `j` across all samples, in sample order.
fn column(x: &[Vec<f64>], j: usize) -> Vec<f64> {
    x.iter()
        .map(|row| row.get(j).copied().unwrap_or(0.0))
        .collect()
}

/// Selects features whose population variance exceeds `threshold`.
///
/// Computes each feature's population variance (`ddof = 0`, divisor `n`) and keeps
/// the feature when `variance > threshold`, exactly as
/// `sklearn.feature_selection.VarianceThreshold` does (it drops features whose
/// variance is `≤ threshold`). With `threshold = 0.0` this removes only the
/// zero-variance (constant) features.
///
/// # Arguments
///
/// * `x` — the feature matrix, one inner `Vec<f64>` per sample; must be non-empty,
///   rectangular, and finite.
/// * `threshold` — the variance below or equal to which a feature is dropped; must
///   be finite (`0.0` keeps every non-constant feature).
///
/// # Returns
///
/// A [`Selection`] whose `scores` are the per-feature population variances and
/// whose `mask` flags `variance > threshold`.
///
/// # Errors
///
/// Returns [`FeatureSelectionError::EmptyInput`],
/// [`FeatureSelectionError::NoFeatures`], [`FeatureSelectionError::RaggedRows`],
/// [`FeatureSelectionError::NonFinite`], or
/// [`FeatureSelectionError::InvalidThreshold`].
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::feature_selection::variance_threshold;
///
/// // Feature 0 is constant (variance 0); features 1 and 2 vary.
/// let x = vec![
///     vec![0.0, 1.0, 2.0],
///     vec![0.0, 4.0, 3.0],
///     vec![0.0, 7.0, 10.0],
/// ];
/// let sel = variance_threshold(&x, 1.0)?;
/// // Variance of feature 0 is 0 (≤ 1, dropped); features 1, 2 exceed 1 (kept).
/// assert_eq!(sel.mask(), &[false, true, true]);
/// assert_eq!(sel.selected_indices(), vec![1, 2]);
/// # Ok::<(), stats_claw::algorithms::feature_selection::FeatureSelectionError>(())
/// ```
pub fn variance_threshold(
    x: &[Vec<f64>],
    threshold: f64,
) -> Result<Selection, FeatureSelectionError> {
    let (_, n_features) = validate_matrix(x)?;
    if !threshold.is_finite() {
        return Err(FeatureSelectionError::InvalidThreshold);
    }
    let scores: Vec<f64> = (0..n_features)
        .map(|j| population_variance(&column(x, j)))
        .collect();
    let mask: Vec<bool> = scores.iter().map(|&v| v > threshold).collect();
    Ok(Selection { scores, mask })
}

/// Scores each feature by the ANOVA F-test across the labelled classes
/// (`f_classif`), keeping the top-`k` by F-score.
///
/// For every feature column, the samples are partitioned by their integer class
/// label and a one-way ANOVA F-statistic `F = MS_between / MS_within` is formed;
/// this is precisely what `sklearn.feature_selection.f_classif` computes (a
/// per-feature one-way ANOVA). The companion p-values are the F upper-tail
/// probabilities and are returned by [`anova_f_pvalues`]. The selection mask keeps
/// the `k` features with the largest F-scores (ties broken by lower column index),
/// mirroring `SelectKBest(f_classif, k)`.
///
/// # Arguments
///
/// * `x` — the feature matrix, one inner `Vec<f64>` per sample; must be non-empty,
///   rectangular, and finite.
/// * `labels` — the integer class label of each sample; its length must equal the
///   number of rows, and at least two distinct labels must be present.
/// * `k` — how many top-scoring features to select. `0` selects none; a `k` at or
///   above the feature count selects all.
///
/// # Returns
///
/// A [`Selection`] whose `scores` are the per-feature ANOVA F-statistics (in
/// column order) and whose `mask` flags the top-`k`.
///
/// # Errors
///
/// Returns [`FeatureSelectionError::EmptyInput`],
/// [`FeatureSelectionError::NoFeatures`], [`FeatureSelectionError::RaggedRows`],
/// [`FeatureSelectionError::NonFinite`],
/// [`FeatureSelectionError::LabelLengthMismatch`],
/// [`FeatureSelectionError::TooFewClasses`], or
/// [`FeatureSelectionError::DegenerateClasses`] (a class too small or with zero
/// within-class variation, so the F-statistic is undefined).
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::feature_selection::anova_f_select;
///
/// // Two classes; feature 1 separates them cleanly, feature 0 does not.
/// let x = vec![
///     vec![1.0, 0.0],
///     vec![1.1, 0.1],
///     vec![0.9, 9.0],
///     vec![1.0, 9.1],
/// ];
/// let labels = [0usize, 0, 1, 1];
/// let sel = anova_f_select(&x, &labels, 1)?;
/// // The cleanly-separating feature 1 has the larger F-score and is selected.
/// assert_eq!(sel.selected_indices(), vec![1]);
/// # Ok::<(), stats_claw::algorithms::feature_selection::FeatureSelectionError>(())
/// ```
pub fn anova_f_select(
    x: &[Vec<f64>],
    labels: &[usize],
    k: usize,
) -> Result<Selection, FeatureSelectionError> {
    let scores = anova_f_scores(x, labels)?;
    let mask = top_k_mask(&scores, k);
    Ok(Selection { scores, mask })
}

/// Computes the per-feature ANOVA F-statistics (`f_classif`), in column order.
///
/// The scoring half of [`anova_f_select`] without a selection rule: useful when a
/// caller wants the raw F-scores to rank or threshold itself.
///
/// # Arguments
///
/// * `x` — the feature matrix, one inner `Vec<f64>` per sample.
/// * `labels` — the integer class label of each sample.
///
/// # Returns
///
/// The per-feature ANOVA F-statistics in column order.
///
/// # Errors
///
/// The same conditions as [`anova_f_select`].
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::feature_selection::anova_f_scores;
///
/// let x = vec![vec![0.0], vec![0.1], vec![9.0], vec![9.1]];
/// let labels = [0usize, 0, 1, 1];
/// let f = anova_f_scores(&x, &labels)?;
/// assert_eq!(f.len(), 1);
/// assert!(f[0] > 0.0, "F was {}", f[0]);
/// # Ok::<(), stats_claw::algorithms::feature_selection::FeatureSelectionError>(())
/// ```
pub fn anova_f_scores(x: &[Vec<f64>], labels: &[usize]) -> Result<Vec<f64>, FeatureSelectionError> {
    let (scores, _) = anova_f_scores_and_pvalues(x, labels)?;
    Ok(scores)
}

/// Computes the per-feature ANOVA F-test p-values (`f_classif`), in column order.
///
/// The upper-tail probabilities companion to [`anova_f_scores`]: `p_j` is the
/// probability that an `F(k − 1, N − k)` variate exceeds feature `j`'s observed
/// F-statistic, reproducing the p-values `sklearn.feature_selection.f_classif`
/// returns.
///
/// # Arguments
///
/// * `x` — the feature matrix, one inner `Vec<f64>` per sample.
/// * `labels` — the integer class label of each sample.
///
/// # Returns
///
/// The per-feature p-values in column order, each in `[0, 1]`.
///
/// # Errors
///
/// The same conditions as [`anova_f_select`].
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::feature_selection::anova_f_pvalues;
///
/// let x = vec![vec![0.0], vec![0.1], vec![9.0], vec![9.1]];
/// let labels = [0usize, 0, 1, 1];
/// let p = anova_f_pvalues(&x, &labels)?;
/// assert_eq!(p.len(), 1);
/// assert!((0.0..=1.0).contains(&p[0]), "p was {}", p[0]);
/// # Ok::<(), stats_claw::algorithms::feature_selection::FeatureSelectionError>(())
/// ```
pub fn anova_f_pvalues(
    x: &[Vec<f64>],
    labels: &[usize],
) -> Result<Vec<f64>, FeatureSelectionError> {
    let (_, pvalues) = anova_f_scores_and_pvalues(x, labels)?;
    Ok(pvalues)
}

/// Computes both the per-feature ANOVA F-statistics and their p-values in one pass.
///
/// The shared engine behind [`anova_f_scores`] and [`anova_f_pvalues`]: it
/// validates the matrix and labels, groups the samples by class once, and for each
/// feature runs the framework's one-way ANOVA (which supplies both the F-statistic
/// and the F upper-tail p-value), so the two public accessors never re-do the work
/// or diverge.
///
/// # Arguments
///
/// * `x` — the feature matrix, one inner `Vec<f64>` per sample.
/// * `labels` — the integer class label of each sample.
///
/// # Returns
///
/// `(scores, pvalues)`, each a per-feature vector in column order.
///
/// # Errors
///
/// The same conditions as [`anova_f_select`].
fn anova_f_scores_and_pvalues(
    x: &[Vec<f64>],
    labels: &[usize],
) -> Result<(Vec<f64>, Vec<f64>), FeatureSelectionError> {
    let (n_samples, n_features) = validate_matrix(x)?;
    if labels.len() != n_samples {
        return Err(FeatureSelectionError::LabelLengthMismatch);
    }
    let class_order = distinct_in_order(labels);
    if class_order.len() < 2 {
        return Err(FeatureSelectionError::TooFewClasses);
    }

    let mut scores = Vec::with_capacity(n_features);
    let mut pvalues = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let col = column(x, j);
        // Partition this feature's values by class, in first-seen class order.
        let groups: Vec<Vec<f64>> = class_order
            .iter()
            .map(|&c| {
                labels
                    .iter()
                    .zip(&col)
                    .filter_map(|(&label, &v)| (label == c).then_some(v))
                    .collect()
            })
            .collect();
        let views: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();
        let result = one_way_anova(&views).map_err(|_| FeatureSelectionError::DegenerateClasses)?;
        scores.push(result.statistic);
        pvalues.push(result.p_value);
    }
    Ok((scores, pvalues))
}

/// Returns the distinct values of `labels` in first-seen order.
///
/// Class order is deterministic (first appearance) so the grouped F-test is
/// reproducible; the F-statistic is invariant to class ordering, so this only
/// fixes a stable iteration order.
///
/// # Arguments
///
/// * `labels` — the per-sample class labels.
///
/// # Returns
///
/// Each distinct label once, in the order it first appears.
fn distinct_in_order(labels: &[usize]) -> Vec<usize> {
    let mut seen: Vec<usize> = Vec::new();
    for &label in labels {
        if !seen.contains(&label) {
            seen.push(label);
        }
    }
    seen
}

/// Builds a top-`k` selection mask from per-feature `scores`.
///
/// Flags the `k` features with the largest scores, breaking ties toward the lower
/// column index (a stable, deterministic rule matching `SelectKBest`'s behaviour
/// of keeping the earlier feature on ties). A `k` of `0` selects none; a `k` at or
/// above the feature count selects all.
///
/// # Arguments
///
/// * `scores` — the per-feature scores, in column order.
/// * `k` — how many top features to flag.
///
/// # Returns
///
/// A boolean mask aligned to `scores`, with exactly `min(k, len)` entries `true`.
fn top_k_mask(scores: &[f64], k: usize) -> Vec<bool> {
    let n = scores.len();
    let keep = k.min(n);
    // Order indices by descending score, ties by ascending index (stable rule).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let sa = scores.get(a).copied().unwrap_or(f64::NEG_INFINITY);
        let sb = scores.get(b).copied().unwrap_or(f64::NEG_INFINITY);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut mask = vec![false; n];
    for &idx in order.iter().take(keep) {
        if let Some(flag) = mask.get_mut(idx) {
            *flag = true;
        }
    }
    mask
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
