//! Gaussian Naive Bayes, matching `sklearn.naive_bayes.GaussianNB`.

use super::{argmax, classification_result_from, normalize_log, sorted_unique, validate_dims};
use crate::algorithms::classification::ClassificationResult;
use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};

/// `ln(2π)`, precomputed for the per-feature Gaussian log density.
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// `scikit-learn`'s default `GaussianNB` variance-smoothing coefficient.
const VAR_SMOOTHING: f64 = 1e-9;

/// Fitted Gaussian Naive Bayes model.
///
/// Holds the sorted class labels, the per-class log priors, and the per-class
/// per-feature Gaussian mean and (smoothed) variance. Construct one with
/// [`gaussian_nb_fit`].
#[derive(Debug, Clone)]
pub struct GaussianNbModel {
    /// Sorted, de-duplicated class labels, in scoring/column order.
    classes: Vec<usize>,
    /// `ln P(class)` per class, in `classes` order.
    log_priors: Vec<f64>,
    /// Per-class per-feature MLE means (`means[c][j]`).
    means: Vec<Vec<f64>>,
    /// Per-class per-feature smoothed variances (`variances[c][j]`).
    variances: Vec<Vec<f64>>,
    /// Number of features expected in every input row.
    n_features: usize,
}

/// Returns each feature's biased (`/n`) variance over all rows of `x`.
fn feature_variances(x: &[Vec<f64>], n_features: usize) -> Vec<f64> {
    let n = count_to_f64(x.len());
    let mut mean = vec![0.0_f64; n_features];
    for row in x {
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0_f64; n_features];
    for row in x {
        for ((acc, &v), &m) in var.iter_mut().zip(row.iter()).zip(mean.iter()) {
            let diff = v - m;
            *acc = diff.mul_add(diff, *acc);
        }
    }
    for acc in &mut var {
        *acc /= n;
    }
    var
}

/// Fits a Gaussian Naive Bayes model to continuous features.
///
/// Reproduces `sklearn.naive_bayes.GaussianNB`: per-class per-feature means and
/// biased (`/n`) variance MLEs, a shared variance floor
/// `var_smoothing = 1e-9 · max feature variance` added to every variance, and
/// class-frequency priors. Scoring (see [`GaussianNbModel::predict`]) is done
/// entirely in log space.
///
/// # Arguments
///
/// * `x` — training design matrix, one inner `Vec` of feature values per
///   observation; every row must share a length.
/// * `y` — one class label per observation.
///
/// # Returns
///
/// The fitted [`GaussianNbModel`].
///
/// # Errors
///
/// * [`Error::EmptyInput`] if `x` or `y` is empty.
/// * [`Error::InvalidInput`] on an `x`/`y` length mismatch, zero features, or
///   ragged rows.
/// * [`Error::InsufficientData`] if fewer than two distinct classes are present.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::classification::naive_bayes::gaussian_nb_fit;
///
/// let x = vec![vec![0.0], vec![0.5], vec![9.0], vec![9.5]];
/// let y = vec![0, 0, 1, 1];
/// let model = gaussian_nb_fit(&x, &y)?;
/// assert_eq!(model.predict(&[vec![0.1], vec![9.2]])?, vec![0, 1]);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn gaussian_nb_fit(x: &[Vec<f64>], y: &[usize]) -> Result<GaussianNbModel> {
    let n_features = validate_dims(x, y)?;
    let classes = sorted_unique(y);
    if classes.len() < 2 {
        return Err(Error::InsufficientData);
    }
    let n_total = count_to_f64(y.len());
    let global_var = feature_variances(x, n_features);
    let max_var = global_var.iter().copied().fold(0.0_f64, f64::max);
    let epsilon = VAR_SMOOTHING * max_var;

    let mut log_priors = Vec::with_capacity(classes.len());
    let mut means = Vec::with_capacity(classes.len());
    let mut variances = Vec::with_capacity(classes.len());
    for &cls in &classes {
        let rows: Vec<&Vec<f64>> = x
            .iter()
            .zip(y)
            .filter_map(|(row, &label)| (label == cls).then_some(row))
            .collect();
        let n_c = count_to_f64(rows.len());
        let mut mean = vec![0.0_f64; n_features];
        for row in &rows {
            for (m, &v) in mean.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n_c;
        }
        let mut var = vec![0.0_f64; n_features];
        for row in &rows {
            for ((acc, &v), &m) in var.iter_mut().zip(row.iter()).zip(mean.iter()) {
                let diff = v - m;
                *acc = diff.mul_add(diff, *acc);
            }
        }
        for acc in &mut var {
            *acc = (*acc / n_c).max(0.0) + epsilon;
        }
        log_priors.push((n_c / n_total).ln());
        means.push(mean);
        variances.push(var);
    }
    Ok(GaussianNbModel {
        classes,
        log_priors,
        means,
        variances,
        n_features,
    })
}

impl GaussianNbModel {
    /// Returns the sorted class labels, in [`Self::predict_log_proba`] column
    /// order.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Computes `ln P(class) + Σⱼ ln N(xⱼ; μ, σ²)` for every sample and class.
    fn joint_log_likelihoods(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::with_capacity(x.len());
        for row in x {
            if row.len() != self.n_features {
                return Err(Error::InvalidInput(
                    "sample feature count differs from the fitted model".to_owned(),
                ));
            }
            let mut scores = Vec::with_capacity(self.classes.len());
            for ((&log_prior, mean), var) in self
                .log_priors
                .iter()
                .zip(self.means.iter())
                .zip(self.variances.iter())
            {
                let mut ll = log_prior;
                for ((&value, &m), &v) in row.iter().zip(mean.iter()).zip(var.iter()) {
                    let diff = value - m;
                    let quad = diff.mul_add(diff, 0.0) / (2.0 * v);
                    ll += (-0.5_f64).mul_add(LN_2PI + v.ln(), -quad);
                }
                scores.push(ll);
            }
            out.push(scores);
        }
        Ok(out)
    }

    /// Predicts the class label for every sample in `x` (ties break low).
    ///
    /// # Arguments
    ///
    /// * `x` — rows to classify; each must have the fitted feature count.
    ///
    /// # Returns
    ///
    /// One predicted class label per input row, in input order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if any row's length differs from the fitted
    /// feature count.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::classification::naive_bayes::gaussian_nb_fit;
    ///
    /// let x = vec![vec![0.0], vec![0.5], vec![9.0], vec![9.5]];
    /// let y = vec![0, 0, 1, 1];
    /// let model = gaussian_nb_fit(&x, &y)?;
    /// assert_eq!(model.predict(&[vec![9.1]])?, vec![1]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<usize>> {
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
    /// * `x` — rows to score; each must have the fitted feature count.
    ///
    /// # Returns
    ///
    /// One log-posterior row per input row.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if any row's length differs from the fitted
    /// feature count.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::classification::naive_bayes::gaussian_nb_fit;
    ///
    /// let x = vec![vec![0.0], vec![0.5], vec![9.0], vec![9.5]];
    /// let y = vec![0, 0, 1, 1];
    /// let model = gaussian_nb_fit(&x, &y)?;
    /// let lp = model.predict_log_proba(&[vec![0.1]])?;
    /// let total: f64 = lp[0].iter().map(|v| v.exp()).sum();
    /// assert!((total - 1.0).abs() < 1e-12, "posteriors summed to {total}");
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn predict_log_proba(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
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
    /// * [`Error::InvalidInput`] on an `x`/`y_true` length mismatch or a row
    ///   whose feature count differs from the fitted model.
    /// * [`Error::EmptyInput`] if `x` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::algorithms::classification::naive_bayes::gaussian_nb_fit;
    ///
    /// let x = vec![vec![0.0], vec![0.5], vec![9.0], vec![9.5]];
    /// let y = vec![0, 0, 1, 1];
    /// let model = gaussian_nb_fit(&x, &y)?;
    /// let result = model.classification_result(&x, &y)?;
    /// assert!((result.accuracy - 1.0).abs() < 1e-12, "accuracy {}", result.accuracy);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn classification_result(
        &self,
        x: &[Vec<f64>],
        y_true: &[usize],
    ) -> Result<ClassificationResult> {
        let predictions = self.predict(x)?;
        classification_result_from(&self.classes, &predictions, y_true, "Gaussian Naive Bayes")
    }
}
