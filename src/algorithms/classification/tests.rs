//! Unit tests for the Naive Bayes classifiers.
//!
//! Golden fixtures for the Gaussian and Categorical predictions and normalized
//! log posteriors were produced with `scikit-learn` 1.9.0 and are compared to
//! `1e-6`. Generating snippet (run under a `numpy`/`sklearn` Python):
//!
//! ```python
//! import numpy as np
//! from sklearn.naive_bayes import GaussianNB, CategoricalNB
//! Xg = [[1.0,2.0],[1.5,1.8],[2.0,2.2],[0.5,1.0],
//!       [8.0,8.0],[9.0,7.5],[7.5,8.5],[8.5,9.0]]
//! yg = [0,0,0,0,1,1,1,1]
//! gnb = GaussianNB().fit(np.array(Xg), np.array(yg))
//! Xt = [[1.2,1.9],[8.2,8.1],[4.0,5.0]]
//! print(gnb.predict(np.array(Xt)))            # [0 1 0]
//! print(gnb.predict_log_proba(np.array(Xt)))  # rows embedded below
//! Xc = [[0,1],[0,0],[1,1],[0,1],[2,2],[2,0],[1,2],[2,2]]
//! yc = [0,0,0,0,1,1,1,1]
//! cnb = CategoricalNB(alpha=1.0).fit(np.array(Xc), np.array(yc))
//! Xct = [[0,1],[2,2],[1,0]]
//! print(cnb.predict(np.array(Xct)))           # [0 1 0]
//! print(cnb.predict_log_proba(np.array(Xct))) # rows embedded below
//! ```

use super::{categorical_nb_fit, gaussian_nb_fit};
use crate::error::Error;

/// Returns whether two matrices of floats agree elementwise within `tol`.
fn rows_close(got: &[Vec<f64>], want: &[Vec<f64>], tol: f64) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.len() == w.len() && g.iter().zip(w).all(|(a, b)| (a - b).abs() < tol))
}

/// The Gaussian training design matrix from the fixture snippet.
fn gaussian_training() -> (Vec<Vec<f64>>, Vec<usize>) {
    let x = vec![
        vec![1.0, 2.0],
        vec![1.5, 1.8],
        vec![2.0, 2.2],
        vec![0.5, 1.0],
        vec![8.0, 8.0],
        vec![9.0, 7.5],
        vec![7.5, 8.5],
        vec![8.5, 9.0],
    ];
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1];
    (x, y)
}

/// The Categorical training design matrix from the fixture snippet.
fn categorical_training() -> (Vec<Vec<usize>>, Vec<usize>) {
    let x = vec![
        vec![0, 1],
        vec![0, 0],
        vec![1, 1],
        vec![0, 1],
        vec![2, 2],
        vec![2, 0],
        vec![1, 2],
        vec![2, 2],
    ];
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1];
    (x, y)
}

#[test]
fn gaussian_fit_rejects_empty_x() {
    let y: Vec<usize> = vec![];
    let got = gaussian_nb_fit(&[], &y);
    assert!(
        matches!(got, Err(Error::EmptyInput)),
        "empty x should be EmptyInput, got {got:?}"
    );
}

#[test]
fn gaussian_fit_rejects_length_mismatch() {
    let x = vec![vec![1.0], vec![2.0]];
    let y = vec![0];
    let got = gaussian_nb_fit(&x, &y);
    assert!(
        matches!(got, Err(Error::InvalidInput(_))),
        "length mismatch should be InvalidInput, got {got:?}"
    );
}

#[test]
fn gaussian_fit_rejects_zero_features() {
    let x = vec![vec![], vec![]];
    let y = vec![0, 1];
    let got = gaussian_nb_fit(&x, &y);
    assert!(
        matches!(got, Err(Error::InvalidInput(_))),
        "zero features should be InvalidInput, got {got:?}"
    );
}

#[test]
fn gaussian_fit_rejects_single_class() {
    let x = vec![vec![1.0], vec![2.0]];
    let y = vec![0, 0];
    let got = gaussian_nb_fit(&x, &y);
    assert!(
        matches!(got, Err(Error::InsufficientData)),
        "single class should be InsufficientData, got {got:?}"
    );
}

#[test]
fn categorical_fit_rejects_negative_alpha() {
    let (x, y) = categorical_training();
    let got = categorical_nb_fit(&x, &y, -0.5);
    assert!(
        matches!(got, Err(Error::InvalidInput(_))),
        "negative alpha should be InvalidInput, got {got:?}"
    );
}

#[test]
fn gaussian_matches_sklearn_golden() -> Result<(), Error> {
    let (x, y) = gaussian_training();
    let model = gaussian_nb_fit(&x, &y)?;
    let test = vec![vec![1.2, 1.9], vec![8.2, 8.1], vec![4.0, 5.0]];
    assert_eq!(model.predict(&test)?, vec![0, 1, 0], "gaussian predict");
    let want = vec![
        vec![0.0, -144.186_513_900_147_33],
        vec![-174.201_905_060_181_9, 0.0],
        vec![-0.000_213_251_968_709_471_38, -8.453_142_763_802_39],
    ];
    let got = model.predict_log_proba(&test)?;
    assert!(
        rows_close(&got, &want, 1e-6),
        "gaussian log_proba mismatch: {got:?}"
    );
    Ok(())
}

#[test]
fn categorical_matches_sklearn_golden() -> Result<(), Error> {
    let (x, y) = categorical_training();
    let model = categorical_nb_fit(&x, &y, 1.0)?;
    let test = vec![vec![0, 1], vec![2, 2], vec![1, 0]];
    assert_eq!(model.predict(&test)?, vec![0, 1, 0], "categorical predict");
    let want = vec![
        vec![-0.060_624_621_816_434_8, -2.833_213_344_056_215_7],
        vec![-2.833_213_344_056_215_7, -0.060_624_621_816_434_8],
        vec![-0.693_147_180_559_945_4, -0.693_147_180_559_945_4],
    ];
    let got = model.predict_log_proba(&test)?;
    assert!(
        rows_close(&got, &want, 1e-6),
        "categorical log_proba mismatch: {got:?}"
    );
    Ok(())
}

#[test]
fn categorical_tie_breaks_to_lower_label() -> Result<(), Error> {
    let (x, y) = categorical_training();
    let model = categorical_nb_fit(&x, &y, 1.0)?;
    // Sample [1, 0] scores identically for both classes; the argmax must pick 0.
    let got = model.predict(&[vec![1, 0]])?;
    assert_eq!(got, vec![0], "tie must break toward the lower label");
    let lp = model.predict_log_proba(&[vec![1, 0]])?;
    let row = lp.first().cloned().unwrap_or_default();
    let a = row.first().copied().unwrap_or(f64::NAN);
    let b = row.get(1).copied().unwrap_or(f64::NAN);
    assert!(
        (a - b).abs() < 1e-12,
        "tie posteriors should match: {a} vs {b}"
    );
    Ok(())
}

#[test]
fn gaussian_fit_is_deterministic() -> Result<(), Error> {
    let (x, y) = gaussian_training();
    let first = gaussian_nb_fit(&x, &y)?;
    let second = gaussian_nb_fit(&x, &y)?;
    let test = vec![vec![1.2, 1.9], vec![8.2, 8.1]];
    assert_eq!(first.predict(&test)?, second.predict(&test)?, "predict");
    let lp_first = first.predict_log_proba(&test)?;
    let lp_second = second.predict_log_proba(&test)?;
    let bits_match = lp_first
        .iter()
        .zip(&lp_second)
        .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(bits_match, "log_proba should be bit-identical across fits");
    Ok(())
}

#[test]
fn categorical_rejects_unknown_category() -> Result<(), Error> {
    let (x, y) = categorical_training();
    let model = categorical_nb_fit(&x, &y, 1.0)?;
    // Feature 0's trained cardinality is 3 (indices 0,1,2); index 3 is unseen.
    let got = model.predict(&[vec![3, 0]]);
    assert!(
        matches!(got, Err(Error::InvalidInput(_))),
        "unknown category should be InvalidInput, got {got:?}"
    );
    Ok(())
}

#[test]
fn gaussian_predict_rejects_feature_mismatch() -> Result<(), Error> {
    let (x, y) = gaussian_training();
    let model = gaussian_nb_fit(&x, &y)?;
    let got = model.predict(&[vec![1.0]]);
    assert!(
        matches!(got, Err(Error::InvalidInput(_))),
        "feature-count mismatch should be InvalidInput, got {got:?}"
    );
    Ok(())
}

#[test]
fn gaussian_classification_result_is_perfect_on_training() -> Result<(), Error> {
    let (x, y) = gaussian_training();
    let model = gaussian_nb_fit(&x, &y)?;
    let result = model.classification_result(&x, &y)?;
    assert!(
        (result.accuracy - 1.0).abs() < 1e-12,
        "accuracy {}",
        result.accuracy
    );
    assert!(
        (result.precision - 1.0).abs() < 1e-12,
        "precision {}",
        result.precision
    );
    assert!(
        (result.recall - 1.0).abs() < 1e-12,
        "recall {}",
        result.recall
    );
    assert!(
        (result.f1_score - 1.0).abs() < 1e-12,
        "f1 {}",
        result.f1_score
    );
    assert!(
        !result.result_id.is_empty(),
        "result_id should be populated"
    );
    assert!(
        !result.description.is_empty(),
        "description should be populated"
    );
    Ok(())
}

#[test]
fn classification_result_scores_a_wrong_prediction() -> Result<(), Error> {
    let (x, y) = gaussian_training();
    let model = gaussian_nb_fit(&x, &y)?;
    // A class-0 point labelled as class 1: the single prediction is wrong.
    let eval = vec![vec![1.2, 1.9]];
    let y_true = vec![1];
    let result = model.classification_result(&eval, &y_true)?;
    assert!(
        (result.accuracy - 0.0).abs() < 1e-12,
        "accuracy {}",
        result.accuracy
    );
    assert!(
        (result.f1_score - 0.0).abs() < 1e-12,
        "f1 {}",
        result.f1_score
    );
    Ok(())
}

#[test]
fn macro_metrics_match_sklearn_on_imperfect_multiclass() -> Result<(), Error> {
    // Macro precision / recall / F1 over an imperfect 3-class confusion, verified
    // against sklearn.metrics (average="macro", zero_division=0). Generating snippet:
    //   from sklearn.metrics import (precision_score, recall_score,
    //                                f1_score, accuracy_score)
    //   y_true = [0, 0, 0, 1, 1, 2]
    //   y_pred = [0, 0, 1, 1, 2, 2]
    //   precision_score(y_true, y_pred, average="macro", zero_division=0) -> 0.6666666666666666
    //   recall_score(y_true, y_pred, average="macro", zero_division=0)    -> 0.7222222222222222
    //   f1_score(y_true, y_pred, average="macro", zero_division=0)        -> 0.6555555555555556
    //   accuracy_score(y_true, y_pred)                                    -> 0.6666666666666666
    let classes = [0_usize, 1, 2];
    let y_pred = [0_usize, 0, 1, 1, 2, 2];
    let y_true = [0_usize, 0, 0, 1, 1, 2];
    let result =
        super::super::classification_result_from(&classes, &y_pred, &y_true, "Golden Multiclass")?;
    assert!(
        (result.accuracy - 0.666_666_666_666_666_6).abs() < 1e-10,
        "accuracy was {}",
        result.accuracy
    );
    assert!(
        (result.precision - 0.666_666_666_666_666_6).abs() < 1e-10,
        "macro precision was {}",
        result.precision
    );
    assert!(
        (result.recall - 0.722_222_222_222_222_2).abs() < 1e-10,
        "macro recall was {}",
        result.recall
    );
    assert!(
        (result.f1_score - 0.655_555_555_555_555_6).abs() < 1e-10,
        "macro f1 was {}",
        result.f1_score
    );
    Ok(())
}

#[test]
fn categorical_classification_result_is_perfect_on_training() -> Result<(), Error> {
    let (x, y) = categorical_training();
    let model = categorical_nb_fit(&x, &y, 1.0)?;
    let result = model.classification_result(&x, &y)?;
    assert!(
        (result.accuracy - 1.0).abs() < 1e-12,
        "accuracy {}",
        result.accuracy
    );
    assert!(
        !result.description.is_empty(),
        "description should be populated"
    );
    Ok(())
}
