//! Supervised classification algorithms.
//!
//! Each fitted model reports its scores through the shared
//! [`ClassificationResult`] parameter struct.
//!
//! The first classifier is deterministic, closed-form **Naive Bayes** in two
//! variants — Gaussian (continuous features) and Categorical (discrete features)
//! — living in [`naive_bayes`]. Both reproduce `scikit-learn`'s
//! `sklearn.naive_bayes.GaussianNB` / `CategoricalNB` semantics exactly: all
//! scoring is done in log space (no underflow), priors are the training class
//! frequencies, and `argmax` ties break toward the lower class label. Each fitted
//! model can emit a populated `ClassificationResult` via its `classification_result`
//! method, computing accuracy plus macro-averaged precision / recall / F1 from
//! predictions against the true labels.
//!
//! This module also houses the input-validation, log-space, and metric helpers
//! shared by every classifier in the family; they are module-private and reached
//! from the submodules.

pub mod types;
pub use types::*;
pub mod naive_bayes;

mod categorical;
mod gaussian;

use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};

/// Validates a design matrix `x` against its label vector `y` and returns the
/// shared feature count.
///
/// # Arguments
///
/// * `x` — one inner `Vec` per observation; every row must share a length.
/// * `y` — one class label per observation.
///
/// # Returns
///
/// The number of features (columns) common to every row.
///
/// # Errors
///
/// * [`Error::EmptyInput`] if `x` or `y` is empty.
/// * [`Error::InvalidInput`] if `x` and `y` differ in length, if there are zero
///   features, or if the rows are ragged.
fn validate_dims<T>(x: &[Vec<T>], y: &[usize]) -> Result<usize> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::EmptyInput);
    }
    if x.len() != y.len() {
        return Err(Error::InvalidInput(
            "x and y must have the same number of rows".to_owned(),
        ));
    }
    let n_features = x.first().map_or(0, Vec::len);
    if n_features == 0 {
        return Err(Error::InvalidInput("x has zero features".to_owned()));
    }
    if x.iter().any(|row| row.len() != n_features) {
        return Err(Error::InvalidInput(
            "all rows must have the same feature count".to_owned(),
        ));
    }
    Ok(n_features)
}

/// Returns the sorted, de-duplicated class labels present in `y`.
fn sorted_unique(y: &[usize]) -> Vec<usize> {
    let mut classes = y.to_vec();
    classes.sort_unstable();
    classes.dedup();
    classes
}

/// Returns the log-sum-exp of `values`, the numerically stable `ln Σ exp(vᵢ)`.
///
/// Used to normalize joint log-likelihoods into log posteriors without leaving
/// log space. Returns [`f64::NEG_INFINITY`] for an empty slice.
fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Normalizes a row of joint log-likelihoods into log posteriors (subtracting
/// the log-sum-exp) so their exponentials sum to 1.
fn normalize_log(row: &[f64]) -> Vec<f64> {
    let lse = log_sum_exp(row);
    row.iter().map(|&v| v - lse).collect()
}

/// Returns the index of the maximum entry of `row`, breaking ties toward the
/// lowest index (so, with sorted class labels, toward the lower label).
fn argmax(row: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &val) in row.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

/// Builds a [`ClassificationResult`] from a set of predictions and true labels.
///
/// Computes accuracy plus macro-averaged precision / recall / F1 over `classes`
/// (the unweighted mean of the per-class metrics, matching `sklearn.metrics`
/// with `average="macro"`; a per-class metric with a zero denominator
/// contributes `0.0`), then fills the descriptive string fields.
///
/// # Arguments
///
/// * `classes` — the sorted class labels the model can emit.
/// * `predictions` — one predicted label per sample.
/// * `y_true` — one true label per sample; must match `predictions` in length.
/// * `method` — a human-readable method name for the descriptive fields.
///
/// # Errors
///
/// * [`Error::EmptyInput`] if there are no predictions.
/// * [`Error::InvalidInput`] if `predictions` and `y_true` differ in length.
fn classification_result_from(
    classes: &[usize],
    predictions: &[usize],
    y_true: &[usize],
    method: &str,
) -> Result<ClassificationResult> {
    if predictions.is_empty() || y_true.is_empty() {
        return Err(Error::EmptyInput);
    }
    if predictions.len() != y_true.len() {
        return Err(Error::InvalidInput(
            "predictions and y_true must have the same length".to_owned(),
        ));
    }
    let n = count_to_f64(predictions.len());
    let correct = predictions
        .iter()
        .zip(y_true)
        .filter(|(p, t)| p == t)
        .count();
    let accuracy = count_to_f64(correct) / n;

    let n_classes = count_to_f64(classes.len());
    let mut precision_sum = 0.0_f64;
    let mut recall_sum = 0.0_f64;
    let mut f1_sum = 0.0_f64;
    for &cls in classes {
        let mut true_pos = 0.0_f64;
        let mut pred_pos = 0.0_f64;
        let mut actual_pos = 0.0_f64;
        for (&p, &t) in predictions.iter().zip(y_true) {
            if p == cls {
                pred_pos += 1.0;
            }
            if t == cls {
                actual_pos += 1.0;
            }
            if p == cls && t == cls {
                true_pos += 1.0;
            }
        }
        let precision = if pred_pos > 0.0 {
            true_pos / pred_pos
        } else {
            0.0
        };
        let recall = if actual_pos > 0.0 {
            true_pos / actual_pos
        } else {
            0.0
        };
        let denom = precision + recall;
        let f1 = if denom > 0.0 {
            2.0 * precision * recall / denom
        } else {
            0.0
        };
        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
    }

    Ok(ClassificationResult {
        accuracy,
        precision: precision_sum / n_classes,
        recall: recall_sum / n_classes,
        f1_score: f1_sum / n_classes,
        result_id: format!("{method} classification"),
        timestamp: String::new(),
        description: format!(
            "{method} scored on {} samples across {} classes",
            predictions.len(),
            classes.len()
        ),
    })
}

/// Kani proof harnesses for the shared classification helpers.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy.
#[cfg(kani)]
mod verification {
    use super::{Error, argmax, validate_dims};

    /// Proves [`validate_dims`] never panics and returns the feature count for a
    /// symbolic `2×2` design matrix paired with a two-element label vector.
    ///
    /// The matrix entries are fully symbolic `f64`, and a rectangular two-by-two
    /// matrix with a matching label length is never empty, never ragged, and has a
    /// non-zero feature count, so the sole reachable outcome is `Ok(2)`; any `Err`
    /// here would be a control-flow bug, which the harness rules out.
    #[kani::proof]
    fn class_validate_dims_ok() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        let c: f64 = kani::any();
        let d: f64 = kani::any();
        let x = vec![vec![a, b], vec![c, d]];
        let y = [0_usize, 1];
        match validate_dims(&x, &y) {
            Ok(features) => assert!(features == 2, "feature count was not 2"),
            Err(Error::EmptyInput | Error::InvalidInput(_)) => {
                assert!(
                    false,
                    "a rectangular 2x2 matrix with 2 labels must validate"
                );
            }
            Err(_) => assert!(false, "validate_dims returned an unexpected error"),
        }
    }

    /// Proves [`argmax`] never panics and returns an in-bounds index for a symbolic
    /// three-element score row.
    ///
    /// The scores are fully symbolic (including `NaN`, for which the strict `>`
    /// comparison is always false, leaving the running best index unchanged), so the
    /// returned index is provably `< 3` for every combination — the guarantee the
    /// class-label lookup that follows `argmax` depends on. The `#[kani::unwind(4)]`
    /// unrolls the fixed three-element scan.
    #[kani::proof]
    #[kani::unwind(4)]
    fn class_argmax_in_bounds() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        let c: f64 = kani::any();
        let idx = argmax(&[a, b, c]);
        assert!(idx < 3, "argmax index escaped the row");
    }
}
