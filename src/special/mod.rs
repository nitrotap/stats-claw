//! Special mathematical functions underpinning the distribution numerics.
//!
//! This module hosts the error functions (`erf`/`erfc`), gamma, and beta
//! families that distribution pdfs, cdfs, and quantiles depend on. It exists as
//! a directory module so each function family can live in its own focused file
//! (e.g. `gamma.rs`, `beta.rs`) and be re-exported here.
//!
//! Consumers name only `special::<fn>` (never the submodule path); the flat
//! re-export surface below is the public contract.

mod beta;
mod error_fn;
mod gamma;

pub use beta::{betai, ln_betai_lower, ln_betai_upper};
pub use error_fn::{erf, erfc, ln_erfc};
pub use gamma::{gamma_p, gamma_q, ln_beta, ln_choose, ln_gamma, ln_gamma_p, ln_gamma_q};

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn ln_gamma_half_is_half_ln_pi() {
        // Γ(1/2) = √π  ⇒  ln Γ(1/2) = ½ ln π.
        assert!(
            PI.ln().mul_add(-0.5, ln_gamma(0.5)).abs() < 1e-12,
            "ln_gamma(0.5) should equal 0.5*ln(pi)"
        );
    }

    #[test]
    fn ln_gamma_integers_match_factorials() {
        // Γ(n) = (n-1)!  ⇒  ln Γ(5) = ln 24.
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn erf_one() {
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12);
    }

    #[test]
    fn erf_is_odd() {
        assert!((erf(-0.7) + erf(0.7)).abs() < 1e-12, "erf must be odd");
        assert!((erfc(0.3) - (1.0 - erf(0.3))).abs() < 1e-12);
    }

    #[test]
    fn gamma_p_complements_gamma_q() {
        assert!((gamma_p(2.5, 3.0) + gamma_q(2.5, 3.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn betai_is_one_half_at_symmetric_midpoint() {
        // I_{0.5}(3, 3) = 0.5 by symmetry.
        assert!((betai(3.0, 3.0, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ln_beta_matches_gamma_definition() {
        // B(2, 3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12.
        assert!((ln_beta(2.0, 3.0).exp() - 1.0 / 12.0).abs() < 1e-12);
    }
}
