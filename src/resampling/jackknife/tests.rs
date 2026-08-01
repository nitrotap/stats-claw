//! Unit tests for the deterministic jackknife numerics (`super`).
//!
//! Split into this submodule file so `jackknife.rs` stays within the 500-line
//! `tests/style.rs` cap; the resampling directory is at its 10-file limit, so a
//! subdirectory module holds the extracted tests.

use super::*;

#[test]
fn indices_error_when_n_below_two() {
    assert_eq!(
        jackknife_indices(1),
        Err(Error::InsufficientData),
        "n < 2 must be rejected as insufficient data"
    );
}

#[test]
fn indices_leave_one_out_for_three() -> Result<()> {
    assert_eq!(
        jackknife_indices(3)?,
        vec![vec![1, 2], vec![0, 2], vec![0, 1]],
        "each set must omit exactly its own index, in order"
    );
    Ok(())
}

/// Fixed textbook sample reused across the statistic golden tests.
const DATA: [f64; 8] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

/// Arithmetic mean of `s`, computed cast-free.
fn mean(s: &[f64]) -> f64 {
    s.iter().sum::<f64>() / count_to_f64(s.len())
}

/// Plug-in (maximum-likelihood, divide-by-`n`) variance of `s`; a biased
/// estimator whose bias the jackknife should recover.
fn plugin_var(s: &[f64]) -> f64 {
    let m = mean(s);
    s.iter()
        .map(|&v| {
            let deviation = v - m;
            deviation * deviation
        })
        .sum::<f64>()
        / count_to_f64(s.len())
}

#[test]
fn mean_std_error_matches_classic_identity() -> Result<()> {
    // python: import numpy as np; d=np.array([2,4,4,4,5,5,7,9.])
    //         np.std(d, ddof=1)/np.sqrt(len(d))  -> 0.7559289460184544
    let expected = 0.755_928_946_018_454_4;
    let est = jackknife_statistic(&DATA, mean)?;
    assert!(
        (est.std_error() - expected).abs() < 1e-12,
        "jackknife SE of the mean was {}, expected {expected}",
        est.std_error()
    );
    Ok(())
}

#[test]
fn mean_bias_is_zero() -> Result<()> {
    // The leave-one-out means average back to the grand mean, so the
    // jackknife bias of the mean is identically zero (to rounding).
    let est = jackknife_statistic(&DATA, mean)?;
    assert!(
        est.bias().abs() < 1e-12,
        "jackknife bias of the mean was {}, expected 0",
        est.bias()
    );
    Ok(())
}

#[test]
fn variance_bias_correction_recovers_unbiased() -> Result<()> {
    // python: import numpy as np; d=np.array([2,4,4,4,5,5,7,9.]); n=len(d)
    //   plugin=lambda a: np.mean((a-a.mean())**2)
    //   est=plugin(d); reps=[plugin(np.delete(d,i)) for i in range(n)]
    //   bias=(n-1)*(np.mean(reps)-est)  -> -0.57142857142856895
    //   est-bias                        ->  4.5714285714285694
    //   np.var(d, ddof=1)               ->  4.5714285714285712
    let est = jackknife_statistic(&DATA, plugin_var)?;
    assert!(
        (est.bias() - (-0.571_428_571_428_569)).abs() < 1e-9,
        "jackknife bias of the plug-in variance was {}",
        est.bias()
    );
    let corrected = est.estimate() - est.bias();
    let unbiased = 4.571_428_571_428_571;
    assert!(
        (corrected - unbiased).abs() < 1e-9,
        "bias-corrected variance was {corrected}, expected {unbiased}"
    );
    Ok(())
}

#[test]
fn accessors_expose_estimate_and_replicates() -> Result<()> {
    let est = jackknife_statistic(&DATA, mean)?;
    assert!(
        (est.estimate() - 5.0).abs() < 1e-12,
        "estimate accessor was {}",
        est.estimate()
    );
    assert!(est.bias().abs() < 1e-12, "bias accessor was {}", est.bias());
    assert!(
        est.std_error() > 0.0,
        "std_error accessor was {}",
        est.std_error()
    );
    assert_eq!(
        est.replicates().len(),
        DATA.len(),
        "replicates accessor must return one value per observation"
    );
    Ok(())
}

#[test]
fn inherent_estimate_delegates_to_free_function() -> Result<()> {
    let scheme = JackknifeResampling::default();
    let via_scheme = scheme.estimate(&DATA, mean)?;
    let direct = jackknife_statistic(&DATA, mean)?;
    assert_eq!(
        via_scheme, direct,
        "the inherent estimate must delegate to jackknife_statistic"
    );
    Ok(())
}

/// D4 boundary: at `n = 2` the jackknife SE of the mean equals the classic
/// `sd(ddof=1) / sqrt(2)`, and its bias is identically zero.
///
/// For `data = [a, b]` the leave-one-out means are `b` and `a`, so the SE
/// reduces to `|b - a| / 2` and `sd(ddof=1)/sqrt(2)` reduces to the same
/// `|b - a| / 2`; here `|4 - 1| / 2 = 1.5`.
#[test]
fn n_two_mean_se_equals_sd_over_sqrt_two_and_bias_is_zero() -> Result<()> {
    let data = [1.0_f64, 4.0];
    let est = jackknife_statistic(&data, mean)?;
    // data = [1.0, 4.0]: mean 2.5, both deviations ±1.5. With n - 1 = 1 the
    // ddof=1 variance is 1.5² + 1.5², so sd(ddof=1) = sqrt(that), SE = sd/√2.
    let dev = 1.5_f64;
    let var_ddof1 = dev.mul_add(dev, dev * dev);
    let expected_se = var_ddof1.sqrt() / 2.0_f64.sqrt();
    assert!(
        (est.std_error() - expected_se).abs() < 1e-12,
        "n=2 SE was {}, expected sd(ddof=1)/sqrt(2) = {expected_se}",
        est.std_error()
    );
    assert!(
        (est.std_error() - 1.5).abs() < 1e-12,
        "n=2 SE was {}, expected closed-form 1.5",
        est.std_error()
    );
    assert!(
        est.bias().abs() < 1e-12,
        "n=2 jackknife bias of the mean was {}, expected 0",
        est.bias()
    );
    Ok(())
}

/// D4 boundary: `jackknife_indices(2)` yields each singleton complement in order.
#[test]
fn indices_leave_one_out_for_two() -> Result<()> {
    assert_eq!(
        jackknife_indices(2)?,
        vec![vec![1], vec![0]],
        "n=2 leave-one-out sets must be [[1], [0]]"
    );
    Ok(())
}
