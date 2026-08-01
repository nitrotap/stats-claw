use super::{Alternative, Error, SplitMix64, monte_carlo_estimate, monte_carlo_p_value};

/// Proves [`monte_carlo_estimate`] rejects every symbolic `n_sims < 2` with
/// [`Error::InsufficientData`] and never panics — the `ddof = 1` standard error
/// is undefined below two draws, so the guard fires before any simulation.
///
/// The `#[kani::unwind(2)]` bound caps the (unreachable-on-feasible-paths)
/// simulation loop: with `n_sims < 2` the guard returns before it, and CBMC
/// discharges the over-unwinding of the infeasible `n_sims >= 2` branch
/// vacuously.
#[kani::proof]
#[kani::unwind(2)]
fn resampling_mc_estimate_rejects_small_n() {
    let n_sims: usize = kani::any();
    kani::assume(n_sims < 2);
    let state: u64 = kani::any();
    let mut rng = SplitMix64::new(state);
    let result = monte_carlo_estimate(n_sims, &mut rng, |_r| 0.0);
    assert!(
        matches!(result, Err(Error::InsufficientData)),
        "n_sims < 2 must be rejected with InsufficientData"
    );
}

/// Proves [`monte_carlo_p_value`] rejects `n_sims == 0` with
/// [`Error::InsufficientData`] and never panics — there are no null draws to
/// count against.
#[kani::proof]
fn resampling_mc_p_value_rejects_zero_n() {
    let observed: f64 = kani::any();
    let state: u64 = kani::any();
    let mut rng = SplitMix64::new(state);
    let result = monte_carlo_p_value(observed, 0, &mut rng, |_r| 0.0, Alternative::Greater);
    assert!(
        matches!(result, Err(Error::InsufficientData)),
        "n_sims == 0 must be rejected with InsufficientData"
    );
}

/// Proves the Phipson–Smyth add-one correction keeps the p-value in `(0, 1]`
/// for a symbolic observed statistic and symbolic finite null draws. With
/// `n_sims = 2` the extreme count `b` lies in `0..=2`, so `p = (b + 1)/(n + 1)`
/// is bounded strictly above zero and never exceeds one — the estimator can
/// never report an impossible zero p-value. The null closure returns an
/// arbitrary finite `f64`, standing in for any caller's null simulator.
#[kani::proof]
#[kani::unwind(4)]
fn resampling_mc_p_value_bounded() {
    let observed: f64 = kani::any();
    kani::assume(observed.is_finite());
    let state: u64 = kani::any();
    let mut rng = SplitMix64::new(state);
    let result = monte_carlo_p_value(
        observed,
        2,
        &mut rng,
        |_r| {
            let x: f64 = kani::any();
            kani::assume(x.is_finite());
            x
        },
        Alternative::Greater,
    );
    assert!(result.is_ok(), "n_sims >= 1 must produce a p-value");
    if let Ok(p) = result {
        assert!(
            p > 0.0,
            "Phipson–Smyth p-value must be strictly positive: {p}"
        );
        assert!(p <= 1.0, "p-value must not exceed one: {p}");
    }
}
