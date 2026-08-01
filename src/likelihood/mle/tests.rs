//! Unit tests for the generic MLE framework in [`super`], kept in a separate
//! file so `mle.rs` stays under the 500-line style cap.

use super::*;

/// Gaussian log-likelihood test model with parameters `[μ, σ]`:
/// `ℓ = −n/2·ln(2πσ²) − Σ(xᵢ−μ)²/(2σ²)`, with `σ ≤ 0` mapped to `−∞`.
struct GaussianLl;

impl LogLikelihood for GaussianLl {
    fn n_params(&self) -> usize {
        2
    }

    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        let mu = *params.first().unwrap_or(&0.0);
        let sigma = *params.get(1).unwrap_or(&0.0);
        if sigma <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let n = count_to_f64(data.len());
        let var = sigma * sigma;
        let sse: f64 = data.iter().map(|x| (x - mu) * (x - mu)).sum();
        let two_pi = 2.0 * std::f64::consts::PI;
        (-n / 2.0).mul_add((two_pi * var).ln(), -(sse / (2.0 * var)))
    }
}

/// Textbook sample with sample mean 5 and MLE standard deviation 2
/// (Σ(x−5)² = 32, /8 = 4, √4 = 2).
const DATA: [f64; 8] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

/// Reads parameter `i` from a fit, mapping an out-of-range index to an error
/// so tests can use `?` instead of an index that could panic.
fn param(fit: &MleFit, i: usize) -> Result<f64> {
    fit.params()
        .get(i)
        .copied()
        .ok_or_else(|| Error::InvalidInput(format!("missing parameter {i}")))
}

#[test]
fn fit_mle_rejects_bad_input() {
    assert!(
        matches!(
            fit_mle(&GaussianLl, &[], &[5.0, 2.0], 1e-8),
            Err(Error::InsufficientData)
        ),
        "empty data should be InsufficientData"
    );
    assert!(
        matches!(
            fit_mle(&GaussianLl, &DATA, &[5.0], 1e-8),
            Err(Error::InvalidInput(_))
        ),
        "wrong init length should be InvalidInput"
    );
    assert!(
        matches!(
            fit_mle(&GaussianLl, &DATA, &[5.0, 2.0], 0.0),
            Err(Error::InvalidInput(_))
        ),
        "non-positive tolerance should be InvalidInput"
    );
}

#[test]
fn fit_mle_rejects_domain_invalid_init() {
    // init with sigma = -1.0 is outside the Gaussian domain (ℓ = −∞ there). The
    // penalized objective would be flat around it, so the numerical gradient is
    // zero and lbfgs would "converge" at the invalid point — must be rejected.
    assert!(
        matches!(
            fit_mle(&GaussianLl, &DATA, &[0.0, -1.0], 1e-8),
            Err(Error::InvalidInput(_))
        ),
        "domain-invalid init should be InvalidInput"
    );
}

#[test]
fn fit_mle_from_near_boundary_init_reaches_interior() -> Result<()> {
    // A valid start on the low-σ side (σ₀ = 0.7, well below the MLE σ̂ = 2) must
    // still climb to the interior optimum, not stall near the boundary. This is
    // the smallest σ₀ inside this unconstrained L-BFGS's basin for the fixture;
    // starts far closer to the boundary (σ₀ ≲ 0.6) exceed the basin and are a
    // known conditioning limit of the generic optimizer — scale-parameter models
    // handle that downstream via a log-σ reparametrization.
    let fit = fit_mle(&GaussianLl, &DATA, &[4.0, 0.7], 1e-8)?;
    let mu = param(&fit, 0)?;
    let sigma = param(&fit, 1)?;
    assert!((mu - 5.0).abs() < 1e-5, "mu_hat was {mu}");
    assert!((sigma - 2.0).abs() < 1e-5, "sigma_hat was {sigma}");
    assert!(sigma > 0.0, "sigma left valid domain: {sigma}");
    assert!(fit.converged(), "expected convergence");
    assert!(fit.log_likelihood().is_finite(), "ll not finite");
    Ok(())
}

#[test]
fn log_likelihood_matches_scipy() {
    // python3: from scipy import stats; import numpy as np
    //   d = np.array([2.,4.,4.,4.,5.,5.,7.,9.])
    //   stats.norm.logpdf(d, loc=5.0, scale=2.0).sum() -> -16.896685710116945
    //   stats.norm.logpdf(d, loc=3.0, scale=1.5).sum() -> -24.817451352724916
    let at_opt = GaussianLl.log_likelihood(&[5.0, 2.0], &DATA);
    assert!(
        (at_opt - (-16.896_685_710_116_945)).abs() < 1e-9,
        "ll at (5,2) was {at_opt}"
    );
    let off = GaussianLl.log_likelihood(&[3.0, 1.5], &DATA);
    assert!(
        (off - (-24.817_451_352_724_916)).abs() < 1e-9,
        "ll at (3,1.5) was {off}"
    );
}

#[test]
fn fit_mle_recovers_estimates_and_exposes_diagnostics() -> Result<()> {
    let fit = fit_mle(&GaussianLl, &DATA, &[4.0, 1.0], 1e-8)?;
    let mu = param(&fit, 0)?;
    let sigma = param(&fit, 1)?;
    // Closed-form MLE: mean = 5, sd (÷n) = 2; σ must stay in the valid domain.
    assert!((mu - 5.0).abs() < 1e-5, "mu_hat was {mu}");
    assert!((sigma - 2.0).abs() < 1e-5, "sigma_hat was {sigma}");
    assert!(sigma > 0.0, "sigma left valid domain: {sigma}");
    assert_eq!(fit.params().len(), 2, "params len");
    assert!(fit.converged(), "expected convergence");
    assert!(fit.iterations() >= 1, "iterations {}", fit.iterations());
    assert!(fit.log_likelihood().is_finite(), "ll not finite");
    Ok(())
}

#[test]
fn aic_bic_satisfy_their_identities() -> Result<()> {
    // Values are implied by `log_likelihood_matches_scipy` plus recovery:
    // aic = 2·2 − 2·(−16.8966857) = 37.79337; bic = 2·ln 8 − 2ℓ = 37.95225.
    let fit = fit_mle(&GaussianLl, &DATA, &[4.0, 1.0], 1e-8)?;
    let ll = fit.log_likelihood();
    let (k, n) = (2.0_f64, 8.0_f64);
    assert!(
        (fit.aic() - 2.0f64.mul_add(k, -2.0 * ll)).abs() < 1e-12,
        "aic identity broken: {}",
        fit.aic()
    );
    assert!(
        (fit.bic() - k.mul_add(n.ln(), -2.0 * ll)).abs() < 1e-12,
        "bic identity broken: {}",
        fit.bic()
    );
    Ok(())
}

#[test]
fn from_closed_form_matches_fit_mle_criteria() -> Result<()> {
    // A closed-form result built from a fit's params/logL must report the
    // same AIC/BIC as the fit itself (same k, n, ℓ through one formula).
    let fit = fit_mle(&GaussianLl, &DATA, &[4.0, 1.0], 1e-8)?;
    let closed = MleFit::from_closed_form(fit.params().to_vec(), fit.log_likelihood(), DATA.len());
    assert!(
        (closed.aic() - fit.aic()).abs() < 1e-12,
        "aic mismatch: {} vs {}",
        closed.aic(),
        fit.aic()
    );
    assert!(
        (closed.bic() - fit.bic()).abs() < 1e-12,
        "bic mismatch: {} vs {}",
        closed.bic(),
        fit.bic()
    );
    assert!(closed.converged(), "closed form is exact, so converged");
    assert_eq!(closed.iterations(), 0, "closed form does no iterations");
    Ok(())
}

#[test]
fn maximum_likelihood_honors_convergence_tolerance() -> Result<()> {
    use crate::likelihood::MaximumLikelihood;
    // A very loose tolerance stops almost immediately, far from the optimum.
    let loose = MaximumLikelihood {
        convergence_tolerance: 1e6,
        ..Default::default()
    };
    let loose_fit = loose.fit(&GaussianLl, &DATA, &[4.0, 1.0])?;
    let loose_mu = param(&loose_fit, 0)?;
    // A tight tolerance recovers the estimate.
    let tight = MaximumLikelihood {
        convergence_tolerance: 1e-10,
        ..Default::default()
    };
    let tight_fit = tight.fit(&GaussianLl, &DATA, &[4.0, 1.0])?;
    let tight_mu = param(&tight_fit, 0)?;
    assert!((tight_mu - 5.0).abs() < 1e-5, "tight mu was {tight_mu}");
    assert!(
        (loose_mu - tight_mu).abs() > 1e-3,
        "loose ({loose_mu}) and tight ({tight_mu}) tolerances behaved identically"
    );
    // The default (0.0) tolerance falls back to 1e-8 and still recovers.
    let defaulted = MaximumLikelihood::default();
    let default_fit = defaulted.fit(&GaussianLl, &DATA, &[4.0, 1.0])?;
    let default_mu = param(&default_fit, 0)?;
    assert!(
        (default_mu - 5.0).abs() < 1e-5,
        "default mu was {default_mu}"
    );
    Ok(())
}
