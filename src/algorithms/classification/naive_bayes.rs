//! Naive Bayes classifiers, for the shared classification parameter structs.
//!
//! Two deterministic, closed-form variants reproduce `scikit-learn` exactly:
//!
//! * [`GaussianNbModel`] / [`gaussian_nb_fit`] — continuous features modelled
//!   per class by an independent Gaussian, matching
//!   `sklearn.naive_bayes.GaussianNB` (biased `/n` variance MLE plus the
//!   `var_smoothing = 1e-9 · max feature variance` floor, class-frequency
//!   priors).
//! * [`CategoricalNbModel`] / [`categorical_nb_fit`] — discrete features
//!   modelled per class by a categorical distribution with additive (Lidstone)
//!   smoothing `alpha`, matching `sklearn.naive_bayes.CategoricalNB`.
//!
//! All scoring is done in log space to avoid underflow, and `argmax` ties break
//! toward the lower class label. Each model can emit a populated
//! [`ClassificationResult`](crate::algorithms::classification::ClassificationResult) via its
//! `classification_result` method. The two variants are implemented in the
//! sibling `gaussian` / `categorical` modules and re-exported here (each variant
//! is fully documented but too large to share one file under the 500-line
//! per-file cap).
//!
//! # Examples
//!
//! ```
//! use stats_claw::algorithms::classification::naive_bayes::gaussian_nb_fit;
//!
//! let x = vec![vec![1.0, 2.0], vec![1.5, 1.8], vec![8.0, 8.0], vec![9.0, 7.5]];
//! let y = vec![0, 0, 1, 1];
//! let model = gaussian_nb_fit(&x, &y)?;
//! assert_eq!(model.predict(&[vec![1.2, 1.9]])?, vec![0]);
//! # Ok::<(), stats_claw::error::Error>(())
//! ```

pub use super::categorical::{CategoricalNbModel, categorical_nb_fit};
pub use super::gaussian::{GaussianNbModel, gaussian_nb_fit};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
