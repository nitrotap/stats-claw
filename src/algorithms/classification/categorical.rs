//! Categorical Naive Bayes, matching `sklearn.naive_bayes.CategoricalNB`.

use super::{argmax, classification_result_from, normalize_log, sorted_unique, validate_dims};
use crate::algorithms::classification::ClassificationResult;
use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};

/// Fitted Categorical Naive Bayes model (Laplace/Lidstone smoothing `alpha`).
///
/// Holds the sorted class labels, per-class log priors, per-feature category
/// cardinalities, and the per-feature per-class per-category log-probabilities.
/// Construct one with [`categorical_nb_fit`].
#[derive(Debug, Clone)]
pub struct CategoricalNbModel {
    /// Sorted, de-duplicated class labels, in scoring/column order.
    classes: Vec<usize>,
    /// `ln P(class)` per class, in `classes` order.
    log_priors: Vec<f64>,
    /// `log_probs[j][c][t]` = `ln P(featureⱼ = t | classₖ)`, `k` indexing
    /// `classes`.
    log_probs: Vec<Vec<Vec<f64>>>,
    /// Per-feature category cardinality (`max index + 1` seen in training).
    cardinalities: Vec<usize>,
    /// Number of features expected in every input row.
    n_features: usize,
}

/// Fits a Categorical Naive Bayes model to discrete (integer-coded) features.
///
/// Reproduces `sklearn.naive_bayes.CategoricalNB`: per-feature per-class category
/// probabilities with additive (Lidstone) smoothing `alpha`, class-frequency
/// priors, and per-feature cardinality = `max training index + 1`. Scoring is
/// done in log space.
///
/// # Arguments
///
/// * `x` — training design matrix of non-negative integer category codes, one
///   inner `Vec` per observation; every row must share a length.
/// * `y` — one class label per observation.
/// * `alpha` — additive smoothing strength (`0.0` = none, `1.0` = Laplace);
///   must be `≥ 0`.
///
/// # Returns
///
/// The fitted [`CategoricalNbModel`].
///
/// # Errors
///
/// * [`Error::EmptyInput`] if `x` or `y` is empty.
/// * [`Error::InvalidInput`] on an `x`/`y` length mismatch, zero features,
///   ragged rows, or `alpha < 0`.
/// * [`Error::InsufficientData`] if fewer than two distinct classes are present.
///
/// # Notes
///
/// With `alpha = 0.0` (no smoothing) a category that never co-occurred with a
/// given class in training has an exact zero probability, so that class's
/// `ln P(category | class)` is `−∞`; if this happens for every class at scoring
/// time the joint log-likelihoods are all `−∞` and the normalized log posteriors
/// become `NaN` — the same degeneracy `sklearn.naive_bayes.CategoricalNB` exhibits
/// at `alpha = 0`. Use a positive `alpha` (e.g. `1.0`) to avoid it.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::classification::naive_bayes::categorical_nb_fit;
///
/// let x = vec![vec![0, 1], vec![0, 0], vec![2, 2], vec![2, 0]];
/// let y = vec![0, 0, 1, 1];
/// let model = categorical_nb_fit(&x, &y, 1.0)?;
/// assert_eq!(model.predict(&[vec![0, 1]])?, vec![0]);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn categorical_nb_fit(x: &[Vec<usize>], y: &[usize], alpha: f64) -> Result<CategoricalNbModel> {
    if alpha < 0.0 {
        return Err(Error::InvalidInput("alpha must be >= 0".to_owned()));
    }
    let n_features = validate_dims(x, y)?;
    let classes = sorted_unique(y);
    if classes.len() < 2 {
        return Err(Error::InsufficientData);
    }
    let n_total = count_to_f64(y.len());

    let mut cardinalities = vec![0_usize; n_features];
    for row in x {
        for (card, &value) in cardinalities.iter_mut().zip(row.iter()) {
            *card = (*card).max(value + 1);
        }
    }

    let mut counts: Vec<Vec<Vec<f64>>> = cardinalities
        .iter()
        .map(|&card| vec![vec![0.0_f64; card]; classes.len()])
        .collect();
    let mut class_totals = vec![0.0_f64; classes.len()];

    for (row, &label) in x.iter().zip(y) {
        let class_idx = classes.iter().position(|&c| c == label).unwrap_or(0);
        if let Some(total) = class_totals.get_mut(class_idx) {
            *total += 1.0;
        }
        for (feature, &value) in counts.iter_mut().zip(row.iter()) {
            if let Some(cell) = feature.get_mut(class_idx).and_then(|c| c.get_mut(value)) {
                *cell += 1.0;
            }
        }
    }

    let log_priors: Vec<f64> = class_totals
        .iter()
        .map(|&total| (total / n_total).ln())
        .collect();

    let log_probs: Vec<Vec<Vec<f64>>> = counts
        .iter()
        .zip(cardinalities.iter())
        .map(|(feature, &card)| {
            let card_f = count_to_f64(card);
            feature
                .iter()
                .zip(class_totals.iter())
                .map(|(class_counts, &total)| {
                    let denom = alpha.mul_add(card_f, total);
                    class_counts
                        .iter()
                        .map(|&count| ((count + alpha) / denom).ln())
                        .collect()
                })
                .collect()
        })
        .collect();

    Ok(CategoricalNbModel {
        classes,
        log_priors,
        log_probs,
        cardinalities,
        n_features,
    })
}

impl CategoricalNbModel {
    /// Returns the sorted class labels, in [`Self::predict_log_proba`] column
    /// order.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Computes `ln P(class) + Σⱼ ln P(xⱼ | class)` for every sample and class.
    fn joint_log_likelihoods(&self, x: &[Vec<usize>]) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::with_capacity(x.len());
        for row in x {
            if row.len() != self.n_features {
                return Err(Error::InvalidInput(
                    "sample feature count differs from the fitted model".to_owned(),
                ));
            }
            for (&value, &card) in row.iter().zip(self.cardinalities.iter()) {
                if value >= card {
                    return Err(Error::InvalidInput(
                        "category index exceeds the trained cardinality".to_owned(),
                    ));
                }
            }
            let mut scores = self.log_priors.clone();
            for (feature, &value) in self.log_probs.iter().zip(row.iter()) {
                for (score, class_probs) in scores.iter_mut().zip(feature.iter()) {
                    *score += class_probs.get(value).copied().unwrap_or(f64::NEG_INFINITY);
                }
            }
            out.push(scores);
        }
        Ok(out)
    }

    /// Predicts the class label for every sample in `x` (ties break low).
    ///
    /// # Arguments
    ///
    /// * `x` — rows to classify; each must have the fitted feature count and only
    ///   category indices seen in training.
    ///
    /// # Returns
    ///
    /// One predicted class label per input row, in input order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] on a feature-count mismatch or an out-of-range
    /// category index.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::classification::naive_bayes::categorical_nb_fit;
    ///
    /// let x = vec![vec![0], vec![0], vec![1], vec![1]];
    /// let y = vec![0, 0, 1, 1];
    /// let model = categorical_nb_fit(&x, &y, 1.0)?;
    /// assert_eq!(model.predict(&[vec![1]])?, vec![1]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn predict(&self, x: &[Vec<usize>]) -> Result<Vec<usize>> {
        let joints = self.joint_log_likelihoods(x)?;
        Ok(joints
            .iter()
            .map(|row| self.classes.get(argmax(row)).copied().unwrap_or(0))
            .collect())
    }

    /// Predicts the normalized log posteriors for every sample in `x`.
    ///
    /// Each returned row has one entry per class (in [`Self::classes`] order)
    /// whose exponentials sum to 1.
    ///
    /// # Arguments
    ///
    /// * `x` — rows to score; same category/shape contract as [`Self::predict`].
    ///
    /// # Returns
    ///
    /// One log-posterior row per input row.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] on a feature-count mismatch or an out-of-range
    /// category index.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::classification::naive_bayes::categorical_nb_fit;
    ///
    /// let x = vec![vec![0], vec![0], vec![1], vec![1]];
    /// let y = vec![0, 0, 1, 1];
    /// let model = categorical_nb_fit(&x, &y, 1.0)?;
    /// let lp = model.predict_log_proba(&[vec![0]])?;
    /// let total: f64 = lp[0].iter().map(|v| v.exp()).sum();
    /// assert!((total - 1.0).abs() < 1e-12, "posteriors summed to {total}");
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn predict_log_proba(&self, x: &[Vec<usize>]) -> Result<Vec<Vec<f64>>> {
        let joints = self.joint_log_likelihoods(x)?;
        Ok(joints.iter().map(|row| normalize_log(row)).collect())
    }

    /// Builds a populated [`ClassificationResult`] from predictions on
    /// `(x, y_true)`: accuracy plus macro-averaged precision / recall / F1.
    ///
    /// # Arguments
    ///
    /// * `x` — evaluation design matrix.
    /// * `y_true` — the true class labels, one per row of `x`.
    ///
    /// # Returns
    ///
    /// A [`ClassificationResult`] scored on `(x, y_true)`.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidInput`] on a length mismatch, feature-count mismatch, or
    ///   out-of-range category index.
    /// * [`Error::EmptyInput`] if `x` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::classification::naive_bayes::categorical_nb_fit;
    ///
    /// let x = vec![vec![0], vec![0], vec![1], vec![1]];
    /// let y = vec![0, 0, 1, 1];
    /// let model = categorical_nb_fit(&x, &y, 1.0)?;
    /// let result = model.classification_result(&x, &y)?;
    /// assert!((result.accuracy - 1.0).abs() < 1e-12, "accuracy {}", result.accuracy);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn classification_result(
        &self,
        x: &[Vec<usize>],
        y_true: &[usize],
    ) -> Result<ClassificationResult> {
        let predictions = self.predict(x)?;
        classification_result_from(
            &self.classes,
            &predictions,
            y_true,
            "Categorical Naive Bayes",
        )
    }
}
