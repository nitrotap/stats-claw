//! Equivalence suite for the outlier / anomaly-detection family.
//!
//! Loads a committed golden fixture and asserts the stats-claw detectors reproduce
//! the reference quantities they pin: the per-point **z-scores** from
//! `scipy.stats.zscore` (population std, `ddof=0`) and the **IQR / Tukey fences**
//! from `numpy.percentile`'s default `'linear'` interpolation. Python never runs
//! here — the fixture is the offline source of truth.

use crate::common;
use crate::common::HarnessError;
use stats_claw::algorithms::outlier::{iqr_detect, zscore_detect, OutlierError};

/// Relative/absolute tolerance for the equivalence comparisons.
///
/// The Rust z-score detector uses the population std (`ddof=0`) exactly as
/// `scipy.stats.zscore` does — the same `(x − mean)/std` arithmetic — so on this
/// fixture the per-point scores agree to `0.0` (machine precision); the gate is
/// `1e-9`. The IQR detector's quartiles use numpy's default `'linear'`
/// interpolation, so the reconstructed Tukey fences agree with numpy **exactly**
/// (diff `0.0`).
const ATOL: f64 = 1e-9;
const RTOL: f64 = 1e-9;

/// Maps a borrowed [`OutlierError`] into the harness error type so tests use `?`
/// on a detection result.
fn detect_err(e: &OutlierError) -> HarnessError {
    HarnessError::Parse(format!("detector failed: {e}"))
}

/// The stats-claw z-score detector reproduces `scipy.stats.zscore` per point.
#[test]
fn zscore_agrees_with_scipy() -> Result<(), HarnessError> {
    let fx = common::load("outlier_detect")?;
    let data = common::f64s(&fx, "data")?;
    let expected = common::f64s(&fx, "zscores")?;

    // Threshold is irrelevant to the *scores*; any finite-positive value works.
    let det = zscore_detect(&data, 3.0).map_err(|e| detect_err(&e))?;
    common::assert_vec_close(det.scores(), &expected, ATOL, RTOL);
    Ok(())
}

/// The stats-claw IQR detector's fences match numpy's `'linear'` Tukey fences.
///
/// The `Detection` exposes the signed distance past the nearer fence, so for the
/// two flagged extremes the fence is recovered as `x − score` and compared to the
/// numpy reference. Inside points score exactly zero, confirming they fall within
/// the same fences.
#[test]
fn iqr_fences_agree_with_numpy() -> Result<(), HarnessError> {
    let fx = common::load("outlier_detect")?;
    let data = common::f64s(&fx, "data")?;
    let k = common::scalar(&fx, "k")?;
    let lower = common::scalar(&fx, "lower_fence")?;
    let upper = common::scalar(&fx, "upper_fence")?;

    let det = iqr_detect(&data, k).map_err(|e| detect_err(&e))?;
    let scores = det.scores();
    let mask = det.mask();

    // Reconstruct each fence from a flagged extreme and compare to numpy.
    // High extreme: score = x - upper_fence (x above upper). Low extreme:
    // score = x - lower_fence (x below lower).
    for (i, &x) in data.iter().enumerate() {
        let score = scores.get(i).copied().unwrap_or(f64::NAN);
        let flagged = mask.get(i).copied().unwrap_or(false);
        if flagged && x > upper {
            common::assert_close(x - score, upper, ATOL, RTOL);
        } else if flagged && x < lower {
            common::assert_close(x - score, lower, ATOL, RTOL);
        } else {
            // Inside the fences: the score must be exactly zero.
            common::assert_close(score, 0.0, ATOL, RTOL);
        }
    }

    // Sanity: exactly the two seeded extremes (60 and -25) are flagged.
    assert_eq!(det.outlier_count(), 2, "expected two flagged extremes");
    Ok(())
}
