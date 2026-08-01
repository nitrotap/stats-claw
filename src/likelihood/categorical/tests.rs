//! Tests for the categorical likelihood.
//!
//! Golden fixtures are produced with `python3` (numpy/scipy); each fixture
//! embeds the exact snippet that produced its expected constant.

use super::{CategoricalLikelihoodModel, CategoricalLogOdds, log_likelihood_core};
use crate::error::{Error, Result};
use crate::likelihood::CategoricalLikelihood;
use crate::likelihood::{LogLikelihood, MleFit, fit_mle};

/// Reads parameter `j` from a fit, mapping an out-of-range index to an error so
/// tests can use `?` instead of a slice index that could panic.
fn param(fit: &MleFit, j: usize) -> Result<f64> {
    fit.params()
        .get(j)
        .copied()
        .ok_or_else(|| Error::InvalidInput(format!("missing parameter {j}")))
}

/// `fit` on empty data reports [`Error::InsufficientData`].
#[test]
fn fit_empty_is_insufficient_data() {
    assert!(
        matches!(
            CategoricalLikelihood::default().fit(3, &[]),
            Err(Error::InsufficientData)
        ),
        "empty data should be InsufficientData"
    );
}

/// `fit` rejects an out-of-range category index with [`Error::InvalidInput`].
#[test]
fn fit_index_out_of_range_is_invalid_input() {
    // 3 is not a valid index for k = 3 (valid indices are 0, 1, 2).
    assert!(
        matches!(
            CategoricalLikelihood::default().fit(3, &[0.0, 3.0]),
            Err(Error::InvalidInput(_))
        ),
        "out-of-range index should be InvalidInput"
    );
}

/// `fit` rejects a non-integer datum with [`Error::InvalidInput`].
#[test]
fn fit_non_integer_is_invalid_input() {
    assert!(
        matches!(
            CategoricalLikelihood::default().fit(3, &[0.0, 1.5]),
            Err(Error::InvalidInput(_))
        ),
        "non-integer datum should be InvalidInput"
    );
}

/// The trait log-likelihood matches a hand/scipy golden value to 1e-10 relative.
///
/// ```python
/// import math
/// data = [0,0,1,2,2,2]; p = [0.2, 0.3, 0.5]
/// counts = [data.count(j) for j in range(3)]           # [2, 1, 3]
/// ll = sum(c*math.log(pj) for c, pj in zip(counts, p)) # -6.502290170873972
/// ```
#[test]
fn trait_log_likelihood_matches_golden() {
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    let data = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
    let ll = model.log_likelihood(&[0.2, 0.3, 0.5], &data);
    let expected = -6.502_290_170_873_972;
    assert!(
        ((ll - expected) / expected).abs() < 1e-10,
        "ll was {ll}, expected {expected}"
    );
}

/// A non-positive probability on a non-empty category yields `NEG_INFINITY`.
#[test]
fn trait_zero_prob_on_nonempty_category_is_neg_inf() {
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    // Category 0 is observed but assigned probability 0.
    let ll = model.log_likelihood(&[0.0, 0.4, 0.6], &[0.0, 1.0, 2.0]);
    assert!(ll.is_infinite() && ll < 0.0, "ll was {ll}");
}

/// A parameter vector of the wrong length yields `NEG_INFINITY`.
#[test]
fn trait_wrong_params_length_is_neg_inf() {
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    let ll = model.log_likelihood(&[0.5, 0.5], &[0.0, 1.0, 2.0]);
    assert!(ll.is_infinite() && ll < 0.0, "ll was {ll}");
}

/// A parameter vector that does not sum to 1 yields `NEG_INFINITY`.
#[test]
fn trait_unnormalized_params_is_neg_inf() {
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    let ll = model.log_likelihood(&[0.3, 0.3, 0.3], &[0.0, 1.0, 2.0]);
    assert!(ll.is_infinite() && ll < 0.0, "ll was {ll}");
}

/// A zero-probability on an *empty* category is fine (0·ln0 ≔ 0 convention).
#[test]
fn trait_zero_prob_on_empty_category_is_finite() {
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    // Category 2 has probability 0 but is never observed.
    let ll = model.log_likelihood(&[0.5, 0.5, 0.0], &[0.0, 0.0, 1.0, 1.0]);
    // Expected 4·ln0.5 = -2.772588722239781.
    let expected = -2.772_588_722_239_781;
    assert!(ll.is_finite(), "ll was {ll}");
    assert!(
        ((ll - expected) / expected).abs() < 1e-10,
        "ll was {ll}, expected {expected}"
    );
}

/// Pins the D3 contract: a non-finite category index yields `NEG_INFINITY`. The
/// `is_valid_index` guard already rejects NaN/±∞ (neither is a finite index);
/// this locks the behavior against future refactors.
#[test]
fn trait_non_finite_observation_is_neg_inf() {
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    let ll_nan = model.log_likelihood(&[0.2, 0.3, 0.5], &[0.0, f64::NAN, 2.0]);
    assert!(ll_nan.is_infinite() && ll_nan < 0.0, "NaN index: {ll_nan}");
    let ll_inf = model.log_likelihood(&[0.2, 0.3, 0.5], &[0.0, f64::INFINITY, 2.0]);
    assert!(ll_inf.is_infinite() && ll_inf < 0.0, "+inf index: {ll_inf}");
}

/// `log_likelihood_core` reused by the trait and the inherent method agree.
#[test]
fn core_and_trait_agree() {
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    let params = [0.2, 0.3, 0.5];
    let data = [0.0, 1.0, 2.0, 2.0];
    let via_trait = model.log_likelihood(&params, &data);
    let via_core = log_likelihood_core(3, &params, &data);
    assert!(
        (via_trait - via_core).abs() < 1e-15,
        "core disagreed with trait"
    );
}

/// `fit` recovers the empirical frequencies to 1e-12 and its logL/AIC/BIC.
///
/// ```python
/// import math
/// data = [0,0,1,2,2,2]; k = 3; n = len(data)          # n = 6
/// counts = [data.count(j) for j in range(k)]          # [2, 1, 3]
/// phat = [c/n for c in counts]                         # [1/3, 1/6, 1/2]
/// ll = sum(c*math.log(pj) for c, pj in zip(counts, phat) if c > 0)  # -6.068425588244111
/// aic = 2*k - 2*ll                                     # 18.13685117648822
/// bic = k*math.log(n) - 2*ll                           # 17.512129584172385
/// ```
#[test]
fn fit_recovers_empirical_frequencies() -> Result<()> {
    let model = CategoricalLikelihood::default();
    let fit = model.fit(3, &[0.0, 0.0, 1.0, 2.0, 2.0, 2.0])?;
    let expected = [1.0 / 3.0, 1.0 / 6.0, 0.5];
    for (j, &want) in expected.iter().enumerate() {
        let got = param(&fit, j)?;
        assert!((got - want).abs() < 1e-12, "p[{j}] was {got}, want {want}");
    }
    assert!(fit.converged(), "closed-form fit should report converged");
    assert_eq!(fit.iterations(), 0, "closed form does no iterations");

    let ll = fit.log_likelihood();
    let ll_want = -6.068_425_588_244_111;
    assert!(((ll - ll_want) / ll_want).abs() < 1e-12, "ll was {ll}");
    // AIC/BIC identity: aic = 2k − 2ℓ, bic = k·ln(n) − 2ℓ with k = 3, n = 6.
    assert!(
        (fit.aic() - 18.136_851_176_488_22).abs() < 1e-10,
        "aic was {}",
        fit.aic()
    );
    assert!(
        (fit.bic() - 17.512_129_584_172_385).abs() < 1e-10,
        "bic was {}",
        fit.bic()
    );
    Ok(())
}

/// `fit` allows an unobserved category, assigning it `p̂ = 0`, and the fitted
/// log-likelihood stays finite (0·ln0 ≔ 0).
///
/// ```python
/// import math
/// data = [0,0,1,1]; k = 3; n = 4
/// counts = [data.count(j) for j in range(k)]          # [2, 2, 0]
/// phat = [c/n for c in counts]                         # [0.5, 0.5, 0.0]
/// ll = sum(c*math.log(pj) for c, pj in zip(counts, phat) if c > 0)  # -2.772588722239781
/// ```
#[test]
fn fit_allows_zero_count_category() -> Result<()> {
    let model = CategoricalLikelihood::default();
    let fit = model.fit(3, &[0.0, 0.0, 1.0, 1.0])?;
    let expected = [0.5, 0.5, 0.0];
    for (j, &want) in expected.iter().enumerate() {
        let got = param(&fit, j)?;
        assert!((got - want).abs() < 1e-12, "p[{j}] was {got}, want {want}");
    }
    let ll = fit.log_likelihood();
    let ll_want = -2.772_588_722_239_781;
    assert!(ll.is_finite(), "fitted ll was {ll}");
    assert!(((ll - ll_want) / ll_want).abs() < 1e-12, "ll was {ll}");
    Ok(())
}

/// `from_probabilities` rejects a non-interior or unnormalized simplex point.
#[test]
fn from_probabilities_rejects_bad_input() {
    // Empty vector: no reference category.
    assert!(
        matches!(
            CategoricalLogOdds::from_probabilities(&[]),
            Err(Error::InvalidInput(_))
        ),
        "empty p should be InvalidInput"
    );
    // Reference probability p0 = 0 (boundary): log-odds is undefined.
    assert!(
        matches!(
            CategoricalLogOdds::from_probabilities(&[0.0, 0.5, 0.5]),
            Err(Error::InvalidInput(_))
        ),
        "p0 = 0 should be InvalidInput"
    );
    // A non-reference zero entry is also outside the open simplex.
    assert!(
        matches!(
            CategoricalLogOdds::from_probabilities(&[0.5, 0.0, 0.5]),
            Err(Error::InvalidInput(_))
        ),
        "an interior zero should be InvalidInput"
    );
    // Does not sum to 1.
    assert!(
        matches!(
            CategoricalLogOdds::from_probabilities(&[0.3, 0.3, 0.3]),
            Err(Error::InvalidInput(_))
        ),
        "unnormalized p should be InvalidInput"
    );
}

/// `probabilities ∘ from_probabilities` is the identity on the open simplex.
#[test]
fn probabilities_from_probabilities_round_trip() -> Result<()> {
    let p = [0.2, 0.3, 0.5];
    let z = CategoricalLogOdds::from_probabilities(&p)?;
    let model = CategoricalLogOdds { n_categories: 3 };
    let recovered = model.probabilities(&z)?;
    for (j, &want) in p.iter().enumerate() {
        let got = *recovered
            .get(j)
            .ok_or_else(|| Error::InvalidInput(format!("missing p{j}")))?;
        assert!((got - want).abs() < 1e-12, "p[{j}] was {got}, want {want}");
    }
    Ok(())
}

/// A non-finite logit puts the log-odds model outside its domain (`−∞`), and a
/// wrong-length parameter vector likewise — mirroring the simplex model's guards.
#[test]
fn logodds_invalid_params_are_neg_inf() {
    let model = CategoricalLogOdds { n_categories: 3 };
    let data = [0.0, 1.0, 2.0];
    let ll_nan = model.log_likelihood(&[f64::NAN, 0.0], &data);
    assert!(ll_nan.is_infinite() && ll_nan < 0.0, "NaN logit: {ll_nan}");
    let ll_len = model.log_likelihood(&[0.0], &data);
    assert!(
        ll_len.is_infinite() && ll_len < 0.0,
        "wrong length: {ll_len}"
    );
}

/// `fit_mle` on the unconstrained log-odds model recovers the closed-form
/// empirical frequencies `p̂ⱼ = countⱼ / n` after softmax — the consistency
/// cross-check the constrained simplex parameterization could not support.
///
/// Fixture: `[0,0,1,2,2,2]` over `k = 3` (counts `[2,1,3]`, all `> 0`) has
/// `p̂ = [1/3, 1/6, 1/2]`. Started from the uniform logits `[0, 0]`, the free
/// optimizer must land on logits whose softmax matches `p̂` to `1e-4`.
#[test]
fn logodds_fit_mle_recovers_empirical_frequencies() -> Result<()> {
    let model = CategoricalLogOdds { n_categories: 3 };
    let data = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
    let fit = fit_mle(&model, &data, &[0.0, 0.0], 1e-8)?;
    assert!(
        fit.converged(),
        "fit_mle should converge, iters {}",
        fit.iterations()
    );
    let p = model.probabilities(fit.params())?;
    let expected = [1.0 / 3.0, 1.0 / 6.0, 0.5];
    for (j, &want) in expected.iter().enumerate() {
        let got = *p
            .get(j)
            .ok_or_else(|| Error::InvalidInput(format!("missing p{j}")))?;
        assert!((got - want).abs() <= 1e-4, "p[{j}] was {got}, want {want}");
    }
    Ok(())
}

/// The log-odds model's log-likelihood equals the simplex model's on the same
/// `(p, data)`, once `p` is mapped to logits via [`CategoricalLogOdds::from_probabilities`].
/// This is the correctness anchor for the softmax reparametrization.
#[test]
fn logodds_log_likelihood_matches_categorical_model() -> Result<()> {
    let p = [0.2, 0.3, 0.5];
    let data = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
    let z = CategoricalLogOdds::from_probabilities(&p)?;
    let logodds = CategoricalLogOdds { n_categories: 3 };
    let model = CategoricalLikelihoodModel { n_categories: 3 };
    let ll_logodds = logodds.log_likelihood(&z, &data);
    let ll_model = model.log_likelihood(&p, &data);
    assert!(
        (ll_logodds - ll_model).abs() < 1e-12,
        "logodds ll {ll_logodds} vs simplex model ll {ll_model}"
    );
    Ok(())
}
