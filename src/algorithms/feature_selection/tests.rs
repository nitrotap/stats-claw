//! Unit tests for the feature-selection base block, added one behaviour at a time.
//!
//! These exercise the variance-threshold filter and the ANOVA F-test selector
//! against hand-computed values and every typed error path; the cross-library
//! equivalence (vs `sklearn`) lives in `tests/equiv/feature_selection.rs`.

use super::*;

/// Variance threshold keeps `variance > threshold` and drops the constant feature,
/// reproducing `sklearn.feature_selection.VarianceThreshold`'s population variance.
#[test]
fn variance_threshold_drops_constant_feature() -> Result<(), FeatureSelectionError> {
    // Feature 0 is constant (var 0); feature 1 var = 6; feature 2 var = 12.6667.
    let x = vec![
        vec![0.0, 1.0, 2.0],
        vec![0.0, 4.0, 3.0],
        vec![0.0, 7.0, 10.0],
    ];
    let sel = variance_threshold(&x, 1.0)?;
    assert_eq!(
        sel.mask(),
        &[false, true, true],
        "mask was {:?}",
        sel.mask()
    );
    let scores = sel.scores();
    assert!(
        scores.first().copied().unwrap_or(f64::NAN).abs() < 1e-12,
        "feature 0 variance was {:?}",
        scores.first()
    );
    let f1 = scores.get(1).copied().unwrap_or(f64::NAN);
    assert!((f1 - 6.0).abs() < 1e-12, "feature 1 variance was {f1}");
    Ok(())
}

/// The ANOVA F-score reproduces a hand-derived per-feature one-way ANOVA F.
#[test]
fn anova_f_scores_match_hand_value() -> Result<(), FeatureSelectionError> {
    // One feature, three clean groups (a, b, c). scipy.f_oneway gives F = 61.
    let x = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![5.0],
        vec![6.0],
        vec![7.0],
        vec![10.0],
        vec![11.0],
        vec![12.0],
    ];
    let labels = [0usize, 0, 0, 1, 1, 1, 2, 2, 2];
    let f = anova_f_scores(&x, &labels)?;
    assert_eq!(f.len(), 1, "one feature expected, got {}", f.len());
    let f0 = f.first().copied().unwrap_or(f64::NAN);
    assert!((f0 - 61.0).abs() < 1e-9, "F was {f0}");
    Ok(())
}

/// `anova_f_select` keeps the top-`k` features by F-score (highest first).
#[test]
fn anova_f_select_keeps_top_k() -> Result<(), FeatureSelectionError> {
    // Feature 0 separates classes cleanly (large F); feature 1 overlaps (small F).
    let x = vec![
        vec![0.0, 1.0],
        vec![0.1, 0.9],
        vec![9.0, 1.0],
        vec![9.1, 1.1],
    ];
    let labels = [0usize, 0, 1, 1];
    let sel = anova_f_select(&x, &labels, 1)?;
    assert_eq!(sel.selected_count(), 1, "exactly one feature kept");
    assert_eq!(
        sel.selected_indices(),
        vec![0],
        "the separating feature 0 must win, got {:?}",
        sel.selected_indices()
    );
    Ok(())
}

/// `k` at or above the feature count selects every feature; `k = 0` selects none.
#[test]
fn anova_f_select_k_bounds() -> Result<(), FeatureSelectionError> {
    let x = vec![
        vec![0.0, 1.0],
        vec![0.1, 0.9],
        vec![9.0, 5.0],
        vec![9.1, 5.2],
    ];
    let labels = [0usize, 0, 1, 1];
    let all = anova_f_select(&x, &labels, 9)?;
    assert_eq!(all.selected_count(), 2, "k beyond width selects all");
    let none = anova_f_select(&x, &labels, 0)?;
    assert_eq!(none.selected_count(), 0, "k = 0 selects none");
    Ok(())
}

/// The p-values are the F upper-tail probabilities in `[0, 1]` and order with F.
#[test]
fn anova_f_pvalues_track_scores() -> Result<(), FeatureSelectionError> {
    let x = vec![
        vec![0.0, 1.0],
        vec![0.1, 0.9],
        vec![9.0, 1.0],
        vec![9.1, 1.1],
    ];
    let labels = [0usize, 0, 1, 1];
    let f = anova_f_scores(&x, &labels)?;
    let p = anova_f_pvalues(&x, &labels)?;
    assert_eq!(p.len(), 2, "one p-value per feature");
    for (i, &pi) in p.iter().enumerate() {
        assert!((0.0..=1.0).contains(&pi), "p[{i}] out of range: {pi}");
    }
    // The higher-F feature (0) must have the smaller (more significant) p-value.
    let (f0, f1) = (
        f.first().copied().unwrap_or(0.0),
        f.get(1).copied().unwrap_or(0.0),
    );
    let (p0, p1) = (
        p.first().copied().unwrap_or(1.0),
        p.get(1).copied().unwrap_or(1.0),
    );
    assert!(f0 > f1, "feature 0 should have larger F: {f0} vs {f1}");
    assert!(p0 < p1, "feature 0 should have smaller p: {p0} vs {p1}");
    Ok(())
}

/// An empty matrix is rejected with [`FeatureSelectionError::EmptyInput`].
#[test]
fn empty_matrix_is_rejected() {
    let x: Vec<Vec<f64>> = Vec::new();
    assert_eq!(
        variance_threshold(&x, 0.0),
        Err(FeatureSelectionError::EmptyInput)
    );
    assert_eq!(
        anova_f_scores(&x, &[0usize]),
        Err(FeatureSelectionError::EmptyInput)
    );
}

/// A matrix with rows but no columns is rejected with `NoFeatures`.
#[test]
fn no_features_is_rejected() {
    let x = vec![Vec::new(), Vec::new()];
    assert_eq!(
        variance_threshold(&x, 0.0),
        Err(FeatureSelectionError::NoFeatures)
    );
}

/// Rows of differing length are rejected with `RaggedRows`.
#[test]
fn ragged_rows_are_rejected() {
    let x = vec![vec![1.0, 2.0], vec![3.0]];
    assert_eq!(
        variance_threshold(&x, 0.0),
        Err(FeatureSelectionError::RaggedRows)
    );
}

/// A non-finite entry is rejected with `NonFinite`.
#[test]
fn non_finite_is_rejected() {
    let x = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
    assert_eq!(
        variance_threshold(&x, 0.0),
        Err(FeatureSelectionError::NonFinite)
    );
}

/// A non-finite variance threshold is rejected with `InvalidThreshold`.
#[test]
fn non_finite_threshold_is_rejected() {
    let x = vec![vec![1.0], vec![2.0]];
    assert_eq!(
        variance_threshold(&x, f64::NAN),
        Err(FeatureSelectionError::InvalidThreshold)
    );
}

/// A label vector whose length differs from the row count is rejected.
#[test]
fn label_length_mismatch_is_rejected() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0]];
    assert_eq!(
        anova_f_scores(&x, &[0usize, 1]),
        Err(FeatureSelectionError::LabelLengthMismatch)
    );
}

/// Fewer than two distinct classes is rejected with `TooFewClasses`.
#[test]
fn too_few_classes_is_rejected() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0]];
    assert_eq!(
        anova_f_scores(&x, &[0usize, 0, 0]),
        Err(FeatureSelectionError::TooFewClasses)
    );
}

/// Zero within-class variation (constant feature) is rejected with
/// `DegenerateClasses`, mapped from the underlying ANOVA's degenerate-input error.
#[test]
fn degenerate_classes_is_rejected() {
    // Both classes are constant at the same value: zero within-class variation.
    let x = vec![vec![5.0], vec![5.0], vec![5.0], vec![5.0]];
    let labels = [0usize, 0, 1, 1];
    assert_eq!(
        anova_f_scores(&x, &labels),
        Err(FeatureSelectionError::DegenerateClasses)
    );
}
