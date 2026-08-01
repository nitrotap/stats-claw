//! Tests for the binomial maximum-likelihood numerics.
//!
//! Golden values are produced with `scipy.stats.binom`; each fixture records the
//! exact generating snippet. Data used throughout: `[3, 5, 2, 4, 6]` with
//! `n_trials = 10`, whose closed-form estimate is `p̂ = 20 / 50 = 0.4`.

use super::*;
use crate::likelihood::{LogLikelihood, MleFit, fit_mle};

/// Builds the shared 10-trial model used by most tests.
fn model() -> BinomialLikelihood {
    BinomialLikelihood {
        number_of_trials: 10,
        ..Default::default()
    }
}

/// The canonical five-observation sample; `p̂ = 0.4`.
const DATA: [f64; 5] = [3.0, 5.0, 2.0, 4.0, 6.0];

/// Whether `x` is exactly `−∞` (avoids a `float_cmp` on `== NEG_INFINITY`).
fn is_neg_inf(x: f64) -> bool {
    x.is_infinite() && x.is_sign_negative()
}

/// Reads the fit's single fitted parameter without slice indexing (denied by the
/// lint gate); returns `NaN` on the impossible empty-parameter case.
fn first_param(fit: &MleFit) -> f64 {
    fit.params().first().copied().unwrap_or(f64::NAN)
}

#[test]
fn fit_empty_data_is_insufficient() {
    assert!(
        matches!(model().fit(&[]), Err(Error::InsufficientData)),
        "empty data should be InsufficientData"
    );
}

#[test]
fn fit_observation_above_n_trials_is_invalid() {
    // 11 > n_trials = 10.
    assert!(
        matches!(model().fit(&[3.0, 11.0]), Err(Error::InvalidInput(_))),
        "x > n should be InvalidInput"
    );
}

#[test]
fn fit_non_integer_observation_is_invalid() {
    assert!(
        matches!(model().fit(&[3.0, 2.5]), Err(Error::InvalidInput(_))),
        "non-integer x should be InvalidInput"
    );
}

#[test]
fn fit_all_zero_observations_is_degenerate() {
    assert!(
        matches!(
            model().fit(&[0.0, 0.0, 0.0]),
            Err(Error::DegenerateInput(_))
        ),
        "all-zero data (p_hat = 0) should be DegenerateInput"
    );
}

#[test]
fn fit_all_at_n_trials_is_degenerate() {
    assert!(
        matches!(model().fit(&[10.0, 10.0]), Err(Error::DegenerateInput(_))),
        "all-at-n data (p_hat = 1) should be DegenerateInput"
    );
}

#[test]
fn fit_non_positive_n_trials_is_invalid() {
    let bad = BinomialLikelihood {
        number_of_trials: 0,
        ..Default::default()
    };
    assert!(
        matches!(bad.fit(&DATA), Err(Error::InvalidInput(_))),
        "n_trials = 0 should be InvalidInput"
    );
}

#[test]
fn log_likelihood_matches_scipy_golden() {
    // python3:
    //   from scipy.stats import binom
    //   binom.logpmf([3,5,2,4,6], 10, 0.3).sum()  # -9.961906023181967
    let expected = -9.961_906_023_181_967_f64;
    let got = model().log_likelihood(&[0.3], &DATA);
    let rel = (got - expected).abs() / expected.abs();
    assert!(
        rel < 1e-10,
        "logL was {got}, expected {expected} (rel {rel})"
    );
}

#[test]
fn log_likelihood_outside_unit_interval_is_neg_infinity() {
    let m = model();
    assert!(
        is_neg_inf(m.log_likelihood(&[0.0], &DATA)),
        "p = 0 must be -inf"
    );
    assert!(
        is_neg_inf(m.log_likelihood(&[1.0], &DATA)),
        "p = 1 must be -inf"
    );
    assert!(
        is_neg_inf(m.log_likelihood(&[1.5], &DATA)),
        "p = 1.5 must be -inf"
    );
    assert!(
        is_neg_inf(m.log_likelihood(&[-0.1], &DATA)),
        "p = -0.1 must be -inf"
    );
}

#[test]
fn log_likelihood_rejects_invalid_observations() {
    let m = model();
    assert!(
        is_neg_inf(m.log_likelihood(&[0.4], &[3.0, 11.0])),
        "x > n must be -inf"
    );
    assert!(
        is_neg_inf(m.log_likelihood(&[0.4], &[3.0, 2.5])),
        "non-integer x must be -inf"
    );
    assert!(
        is_neg_inf(m.log_likelihood(&[0.4], &[-1.0])),
        "negative x must be -inf"
    );
}

#[test]
fn log_likelihood_non_finite_observation_is_neg_inf() {
    // Pins the D3 contract: a non-finite observation gives −∞. The
    // `success_count` guard already rejects NaN/±∞ (they are not finite counts);
    // this locks the behavior in place.
    let m = model();
    assert!(
        is_neg_inf(m.log_likelihood(&[0.4], &[3.0, f64::NAN])),
        "NaN observation must be -inf"
    );
    assert!(
        is_neg_inf(m.log_likelihood(&[0.4], &[3.0, f64::INFINITY])),
        "+inf observation must be -inf"
    );
}

#[test]
fn n_params_is_one() {
    assert_eq!(model().n_params(), 1, "binomial has one free parameter");
}

#[test]
fn fit_recovers_closed_form_p_hat() -> Result<()> {
    let fit = model().fit(&DATA)?;
    // p̂ = 20 / (5·10) = 0.4 exactly.
    let p_hat = first_param(&fit);
    assert!((p_hat - 0.4).abs() < 1e-12, "p_hat was {p_hat}");
    Ok(())
}

#[test]
fn fit_reports_converged_zero_iterations() -> Result<()> {
    let fit = model().fit(&DATA)?;
    assert!(fit.converged(), "closed-form fit must report converged");
    assert_eq!(fit.iterations(), 0, "closed-form fit does 0 iterations");
    Ok(())
}

#[test]
fn fit_log_likelihood_matches_scipy_at_p_hat() -> Result<()> {
    // python3: binom.logpmf([3,5,2,4,6], 10, 0.4).sum()  # -8.832784968964091
    let expected = -8.832_784_968_964_091_f64;
    let fit = model().fit(&DATA)?;
    let got = fit.log_likelihood();
    let rel = (got - expected).abs() / expected.abs();
    assert!(
        rel < 1e-10,
        "fit logL was {got}, expected {expected} (rel {rel})"
    );
    Ok(())
}

#[test]
fn fit_aic_bic_identities() -> Result<()> {
    let fit = model().fit(&DATA)?;
    let ll = fit.log_likelihood();
    let neg_two_ll = -2.0 * ll;
    // k = 1 parameter, n = 5 observations: aic = 2k − 2ℓ, bic = k·ln(n) − 2ℓ.
    let aic = fit.aic();
    let bic = fit.bic();
    assert!(
        (aic - (2.0 + neg_two_ll)).abs() < 1e-12,
        "aic identity failed: {aic}"
    );
    assert!(
        (bic - (5.0_f64.ln() + neg_two_ll)).abs() < 1e-12,
        "bic identity failed: {bic}"
    );
    // Cross-check against the scipy-derived values.
    // python3: aic = 2*1 - 2*ll = 19.665569937928183; bic = ln(5) - 2*ll = 19.275007850362282
    assert!(
        (aic - 19.665_569_937_928_183).abs() < 1e-9,
        "aic golden mismatch: {aic}"
    );
    assert!(
        (bic - 19.275_007_850_362_282).abs() < 1e-9,
        "bic golden mismatch: {bic}"
    );
    Ok(())
}

#[test]
fn fit_mle_from_perturbed_init_recovers_p_hat() -> Result<()> {
    // The free optimizer, started away from p̂ = 0.4, must recover it. The
    // penalty in the objective keeps trial steps that leave (0,1) finite.
    let m = model();
    let fit = fit_mle(&m, &DATA, &[0.25], 1e-10)?;
    let p_hat = first_param(&fit);
    assert!((p_hat - 0.4).abs() < 1e-5, "optimizer p_hat was {p_hat}");
    Ok(())
}

#[test]
fn boundary_observations_are_valid_when_interior_p_hat() -> Result<()> {
    // A sample containing both 0 and n is fine as long as p̂ stays interior.
    let m = model();
    let data = [0.0, 10.0, 5.0];
    let fit = m.fit(&data)?;
    // p̂ = 15 / 30 = 0.5.
    let p_hat = first_param(&fit);
    assert!((p_hat - 0.5).abs() < 1e-12, "p_hat was {p_hat}");
    // log-likelihood must be finite (0 and n contribute finite terms).
    assert!(
        m.log_likelihood(&[0.5], &data).is_finite(),
        "logL must be finite"
    );
    Ok(())
}
