//! Unit tests for the [`super`] outlier detectors (z-score, IQR, modified-z).
//!
//! Split into its own file (included via `#[path]`) so `mod.rs` stays inside the
//! project's 500-line `style.rs` cap while the tests still compile under
//! `cargo test` with access to the module's private items.

use super::{
    Detection, MAD_CONSISTENCY, OutlierError, iqr_detect, modified_zscore_detect, zscore_detect,
};

/// The flagged score in a [`Detection`], or `NaN` if absent (test-only reader).
fn score_at(det: &Detection, i: usize) -> f64 {
    det.scores().get(i).copied().unwrap_or(f64::NAN)
}

/// The z-score detector reproduces `scipy.stats.zscore` (population std).
#[test]
fn zscore_matches_scipy() -> Result<(), OutlierError> {
    // scipy.stats.zscore([1,2,3,4,100])[4] = 1.9993428618189626.
    let det = zscore_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 2.0)?;
    let z = score_at(&det, 4);
    assert!((z - 1.999_342_861_818_962_6).abs() < 1e-12, "z was {z}");
    Ok(())
}

/// The z-score detector flags only the extreme point at threshold 1.5.
///
/// With five points the maximal |z| caps near 2.0 (scipy gives ~1.999 here), so
/// a 1.5 threshold flags the single extreme point while the rest sit near 0.5.
#[test]
fn zscore_flags_extreme_point() -> Result<(), OutlierError> {
    let det = zscore_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 1.5)?;
    assert_eq!(
        det.mask(),
        &[false, false, false, false, true],
        "mask {:?}",
        det.mask()
    );
    assert_eq!(det.outlier_count(), 1, "count {}", det.outlier_count());
    Ok(())
}

/// A constant sample has zero std, so the z-score is undefined and rejected.
#[test]
fn zscore_rejects_constant_sample() {
    assert_eq!(
        zscore_detect(&[5.0, 5.0, 5.0], 3.0),
        Err(OutlierError::ZeroSpread)
    );
}

/// The IQR detector reproduces numpy's `'linear'` Tukey fences and flags outside.
#[test]
fn iqr_matches_numpy_fences() -> Result<(), OutlierError> {
    // np.percentile([1,2,3,4,100],[25,75]) = [2,4]; IQR=2; fences -1 .. 7.
    let det = iqr_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 1.5)?;
    assert_eq!(
        det.mask(),
        &[false, false, false, false, true],
        "mask {:?}",
        det.mask()
    );
    // The extreme point's score is its distance past the upper fence: 100 - 7 = 93.
    let s = score_at(&det, 4);
    assert!((s - 93.0).abs() < 1e-12, "score was {s}");
    Ok(())
}

/// Points inside the Tukey fences score exactly zero.
#[test]
fn iqr_inside_points_score_zero() -> Result<(), OutlierError> {
    let det = iqr_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 1.5)?;
    for i in 0..4 {
        let s = score_at(&det, i);
        assert!(s.abs() < 1e-12, "index {i} score {s}");
    }
    Ok(())
}

/// A point below the lower fence is flagged with a negative score.
#[test]
fn iqr_flags_low_outlier_negative() -> Result<(), OutlierError> {
    // np.percentile([-100,10,11,12,13],[25,75]) -> low value below lower fence.
    let det = iqr_detect(&[-100.0, 10.0, 11.0, 12.0, 13.0], 1.5)?;
    assert!(det.mask().first().copied().unwrap_or(false), "low flagged");
    let s = score_at(&det, 0);
    assert!(s < 0.0, "low score should be negative, was {s}");
    Ok(())
}

/// The modified z-score uses the median/MAD and the `0.6745` constant.
#[test]
fn modified_zscore_uses_mad() -> Result<(), OutlierError> {
    // data 1,2,3,4,100: median 3, MAD 1; modified-z of 100 = 0.6745*(97)/1.
    let det = modified_zscore_detect(&[1.0, 2.0, 3.0, 4.0, 100.0], 3.5)?;
    let m = score_at(&det, 4);
    let want = MAD_CONSISTENCY * 97.0;
    assert!((m - want).abs() < 1e-12, "modified-z was {m}, want {want}");
    assert!(
        det.mask().last().copied().unwrap_or(false),
        "extreme flagged"
    );
    Ok(())
}

/// A zero-MAD sample (majority share the median) is rejected, not divided by zero.
#[test]
fn modified_zscore_rejects_zero_mad() {
    // 1,1,1,1,9: median 1, deviations 0,0,0,0,8 -> MAD = median = 0.
    assert_eq!(
        modified_zscore_detect(&[1.0, 1.0, 1.0, 1.0, 9.0], 3.5),
        Err(OutlierError::ZeroSpread)
    );
}

/// Every detector rejects an empty sample with [`OutlierError::EmptyInput`].
#[test]
fn empty_sample_is_rejected() {
    assert_eq!(zscore_detect(&[], 3.0), Err(OutlierError::EmptyInput));
    assert_eq!(iqr_detect(&[], 1.5), Err(OutlierError::EmptyInput));
    assert_eq!(
        modified_zscore_detect(&[], 3.5),
        Err(OutlierError::EmptyInput)
    );
}

/// Every detector rejects a non-finite observation before any arithmetic.
#[test]
fn non_finite_is_rejected() {
    let bad = [1.0, f64::NAN, 3.0];
    assert_eq!(zscore_detect(&bad, 3.0), Err(OutlierError::NonFinite));
    assert_eq!(iqr_detect(&bad, 1.5), Err(OutlierError::NonFinite));
    let inf = [1.0, f64::INFINITY, 3.0];
    assert_eq!(
        modified_zscore_detect(&inf, 3.5),
        Err(OutlierError::NonFinite)
    );
}

/// Every detector rejects a non-finite or non-positive threshold / multiplier.
#[test]
fn invalid_threshold_is_rejected() {
    let data = [1.0, 2.0, 3.0, 4.0, 100.0];
    assert_eq!(
        zscore_detect(&data, 0.0),
        Err(OutlierError::InvalidThreshold)
    );
    assert_eq!(iqr_detect(&data, -1.5), Err(OutlierError::InvalidThreshold));
    assert_eq!(
        modified_zscore_detect(&data, f64::NAN),
        Err(OutlierError::InvalidThreshold)
    );
}

/// Scores and mask always share the input length, in input order.
#[test]
fn scores_and_mask_align_to_input() -> Result<(), OutlierError> {
    let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 100.0];
    let det = zscore_detect(&data, 3.0)?;
    assert_eq!(det.scores().len(), data.len(), "scores length");
    assert_eq!(det.mask().len(), data.len(), "mask length");
    Ok(())
}
