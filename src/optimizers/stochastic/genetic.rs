//! Differential-evolution genetic optimizer over a bounded box (seeded).
//!
//! The stats-claw counterpart of `scipy.optimize.differential_evolution`. Uses the
//! classic `rand/1/bin` strategy: each candidate is mutated by a scaled
//! difference of two other members and recombined by binomial crossover.

use super::{clamp, scale};
use crate::optimizers::{ConvergenceStatus, Objective, OptimizeResult};
use crate::rng::SplitMix64;

/// Differential weight `F` applied to the difference vector.
const F: f64 = 0.8;
/// Crossover probability `CR` for binomial recombination.
const CR: f64 = 0.9;
/// Population size (number of candidate vectors per generation).
const POP: usize = 15;

/// Minimizes `obj` over the box `bounds` by differential evolution.
///
/// Initializes a seeded uniform population in the box, then for `generations`
/// rounds mutates each member with `a + F·(b − c)` (three distinct random
/// members), applies binomial crossover against the target, clamps to the box,
/// and keeps the trial if it improves the objective. The best member is
/// returned. Status is always [`ConvergenceStatus::MaxIterReached`] (the loop
/// runs its full generation budget by design).
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `bounds` — per-coordinate `(lo, hi)` box bounds.
/// * `generations` — the number of generations to evolve.
/// * `rng` — the seeded PRNG; the same seed reproduces the run exactly.
///
/// # Returns
///
/// An [`OptimizeResult`] reporting the best member found, its value, the
/// generation count, and the status.
///
/// # Examples
///
/// ```
/// use stats_claw::optimizers::stochastic::genetic;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::rng::SplitMix64;
///
/// let obj = Quadratic::new(vec![3.0, -2.0]);
/// let bounds = vec![(0.0, 5.0), (-5.0, 0.0)];
/// let mut rng = SplitMix64::new(99);
/// let r = genetic(&obj, &bounds, 100, &mut rng);
/// // The objective should be close to its minimum of 0.
/// assert!(r.fx < 1.0, "f(x) was {}", r.fx);
/// ```
#[must_use]
pub fn genetic(
    obj: &impl Objective,
    bounds: &[(f64, f64)],
    generations: usize,
    rng: &mut SplitMix64,
) -> OptimizeResult {
    let dim = bounds.len();
    let mut pop: Vec<Vec<f64>> = (0..POP)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| scale(rng.next_f64(), lo, hi))
                .collect()
        })
        .collect();
    let mut fitness: Vec<f64> = pop.iter().map(|c| obj.value(c)).collect();
    let mut iterations = 0;
    for gen in 0..generations {
        iterations = gen + 1;
        for i in 0..POP {
            let (a, b, c) = three_distinct(i, rng);
            let trial = mutate_crossover(&pop, i, a, b, c, dim, bounds, rng);
            let trial_f = obj.value(&trial);
            if trial_f < *fitness.get(i).unwrap_or(&f64::INFINITY) {
                if let Some(slot) = pop.get_mut(i) {
                    *slot = trial;
                }
                if let Some(slot) = fitness.get_mut(i) {
                    *slot = trial_f;
                }
            }
        }
    }
    let (best_idx, &best_f) = fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((0, &f64::INFINITY));
    let best = pop.get(best_idx).cloned().unwrap_or_default();
    OptimizeResult {
        x: best,
        fx: best_f,
        iterations,
        status: ConvergenceStatus::MaxIterReached,
    }
}

/// Builds the mutated, crossed-over trial vector for population member `target`.
#[allow(clippy::too_many_arguments)] // a focused internal helper; splitting would obscure the DE update
fn mutate_crossover(
    pop: &[Vec<f64>],
    target: usize,
    a: usize,
    b: usize,
    c: usize,
    dim: usize,
    bounds: &[(f64, f64)],
    rng: &mut SplitMix64,
) -> Vec<f64> {
    let xa = pop.get(a).cloned().unwrap_or_default();
    let xb = pop.get(b).cloned().unwrap_or_default();
    let xc = pop.get(c).cloned().unwrap_or_default();
    let xt = pop.get(target).cloned().unwrap_or_default();
    let forced = bounded_index(rng, dim.max(1));
    (0..dim)
        .map(|j| {
            let (lo, hi) = *bounds.get(j).unwrap_or(&(0.0, 1.0));
            let donor = F.mul_add(
                xb.get(j).unwrap_or(&0.0) - xc.get(j).unwrap_or(&0.0),
                *xa.get(j).unwrap_or(&0.0),
            );
            if j == forced || rng.next_f64() < CR {
                clamp(donor, lo, hi)
            } else {
                *xt.get(j).unwrap_or(&0.0)
            }
        })
        .collect()
}

/// Reduces a 64-bit draw to an index in `[0, modulus)` without an `as` cast.
fn bounded_index(rng: &mut SplitMix64, modulus: usize) -> usize {
    let m = u64::try_from(modulus).unwrap_or(1).max(1);
    let r = rng.next_u64() % m;
    usize::try_from(r).unwrap_or(0)
}

/// Draws three population indices distinct from each other and from `exclude`.
fn three_distinct(exclude: usize, rng: &mut SplitMix64) -> (usize, usize, usize) {
    let pick = |rng: &mut SplitMix64| bounded_index(rng, POP);
    let a = loop {
        let v = pick(rng);
        if v != exclude {
            break v;
        }
    };
    let b = loop {
        let v = pick(rng);
        if v != exclude && v != a {
            break v;
        }
    };
    let c = loop {
        let v = pick(rng);
        if v != exclude && v != a && v != b {
            break v;
        }
    };
    (a, b, c)
}
