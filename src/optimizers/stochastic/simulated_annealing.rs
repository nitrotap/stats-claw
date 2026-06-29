//! Simulated annealing over a bounded box (seeded, reproducible).
//!
//! The stats-claw counterpart of `scipy.optimize.dual_annealing`.

use super::{clamp, scale};
use crate::optimizers::{ConvergenceStatus, Objective, OptimizeResult};
use crate::rng::SplitMix64;

/// Minimizes `obj` over the box `bounds` by simulated annealing.
///
/// Starts from the box center, proposes Gaussian perturbations scaled by the box
/// width and the current temperature, and accepts worse moves with the
/// Metropolis probability `exp(−Δ/T)`. The temperature cools geometrically
/// (`T ← 0.95·T`) each iteration. The best point ever seen is returned. Status is
/// always [`ConvergenceStatus::MaxIterReached`] because annealing runs its full
/// schedule by design.
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `bounds` — per-coordinate `(lo, hi)` box bounds; its length is the
///   dimension.
/// * `max_iter` — the number of annealing steps (the cooling schedule length).
/// * `rng` — the seeded PRNG; the same seed reproduces the run exactly.
///
/// # Returns
///
/// An [`OptimizeResult`] reporting the best point found, its value, the iteration
/// count, and the status.
///
/// # Examples
///
/// ```
/// use stats_claw::optimizers::stochastic::simulated_annealing;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::rng::SplitMix64;
///
/// let obj = Quadratic::new(vec![3.0, -2.0]);
/// let bounds = vec![(0.0, 6.0), (-4.0, 0.0)];
/// let mut rng = SplitMix64::new(7);
/// let r = simulated_annealing(&obj, &bounds, 10_000, &mut rng);
/// // The objective should be substantially reduced from the center.
/// assert!(r.fx < 5.0, "f(x) was {}", r.fx);
/// ```
#[must_use]
pub fn simulated_annealing(
    obj: &impl Objective,
    bounds: &[(f64, f64)],
    max_iter: usize,
    rng: &mut SplitMix64,
) -> OptimizeResult {
    let mut current: Vec<f64> = bounds.iter().map(|&(lo, hi)| scale(0.5, lo, hi)).collect();
    let mut current_f = obj.value(&current);
    let mut best = current.clone();
    let mut best_f = current_f;
    let mut temp = 1.0_f64;
    let mut iterations = 0;
    // Geometric cooling to a small floor across the whole budget so late
    // proposals are fine enough to refine the optimum to a tight tolerance.
    let budget = max_iter.max(1);
    let cooling = 1e-4_f64.powf(1.0 / f64::from(u32::try_from(budget).unwrap_or(u32::MAX)));
    for step in 0..max_iter {
        iterations = step + 1;
        let proposal: Vec<f64> = current
            .iter()
            .zip(bounds)
            .map(|(&xi, &(lo, hi))| {
                let width = hi - lo;
                clamp(
                    rng.standard_normal().mul_add(temp * width * 0.1, xi),
                    lo,
                    hi,
                )
            })
            .collect();
        let proposal_f = obj.value(&proposal);
        let delta = proposal_f - current_f;
        if delta < 0.0 || rng.next_f64() < (-delta / temp.max(1e-12)).exp() {
            current = proposal;
            current_f = proposal_f;
        }
        if current_f < best_f {
            best_f = current_f;
            best.clone_from(&current);
        }
        temp *= cooling;
    }
    OptimizeResult {
        x: best,
        fx: best_f,
        iterations,
        status: ConvergenceStatus::MaxIterReached,
    }
}
