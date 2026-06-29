//! Gaussian mixture model fit by expectation–maximisation.
//!
//! A GMM models the data as a weighted sum of `k` multivariate Gaussians and is fit
//! by EM: the E-step computes each point's responsibility (posterior probability) for
//! every component using the [`NormalDistribution`](crate::distributions::NormalDistribution)
//! family's Gaussian density generalised to full covariance; the M-step re-estimates
//! the mixture weights, means, and covariances from those responsibilities. Iteration
//! continues until the average log-likelihood stops improving by more than `tol`.
//! Initialisation reuses the framework's k-means (which itself uses the deterministic
//! [`SplitMix64`](crate::rng::SplitMix64) RNG), matching `scikit-learn`'s
//! `init_params="kmeans"`.
//!
//! The Bayesian and Akaike information criteria are identifiable (free of label
//! permutation), so the equivalence suite compares them to `sklearn.mixture.
//! GaussianMixture` directly, alongside an adjusted-Rand-index check on the hard
//! assignments and a verification that every responsibility row is a probability
//! distribution.

use crate::algorithms::clustering::kmeans;
use crate::algorithms::decomposition::{at, count_to_f64, jacobi_eigen, put, symmetric_inverse};

/// Diagonal regularisation added to each covariance, matching `scikit-learn`'s
/// `reg_covar` default; keeps covariances positive-definite.
const REG_COVAR: f64 = 1e-6;

/// Outcome of a Gaussian-mixture EM fit.
#[derive(Debug, Clone)]
pub struct GmmResult {
    /// Hard cluster assignment (arg-max responsibility) per point, in input order.
    pub labels: Vec<usize>,
    /// Soft assignments: `responsibilities[i][c]` is point `i`'s posterior for
    /// component `c`; each row sums to one.
    pub responsibilities: Vec<Vec<f64>>,
    /// Bayesian information criterion `−2·logL + p·ln(n)` (lower is better).
    pub bic: f64,
    /// Akaike information criterion `−2·logL + 2p` (lower is better).
    pub aic: f64,
}

/// Fits a `k`-component Gaussian mixture to `data` by EM.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension. Empty
///   input yields an empty result.
/// * `k` — number of mixture components, clamped to the number of points.
/// * `max_iter` — maximum EM iterations.
/// * `tol` — convergence tolerance on the per-sample average log-likelihood change.
/// * `seed` — RNG seed for the k-means initialisation (determinism contract).
///
/// # Returns
///
/// A [`GmmResult`] with hard/soft assignments and the BIC/AIC information criteria.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::gmm_em;
///
/// let data = vec![
///     vec![0.0, 0.0],
///     vec![0.2, 0.1],
///     vec![10.0, 10.0],
///     vec![10.1, 9.9],
/// ];
/// let r = gmm_em(&data, 2, 100, 1e-3, 42);
/// // The two tight pairs fall into different components.
/// assert_ne!(r.labels.first(), r.labels.get(2));
/// ```
#[must_use]
pub fn gmm_em(data: &[Vec<f64>], k: usize, max_iter: usize, tol: f64, seed: u64) -> GmmResult {
    let n = data.len();
    let dim = data.first().map_or(0, Vec::len);
    let k = k.min(n);
    if n == 0 || dim == 0 || k == 0 {
        return GmmResult {
            labels: Vec::new(),
            responsibilities: Vec::new(),
            bic: 0.0,
            aic: 0.0,
        };
    }
    let mut params = initialise(data, k, dim, seed);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut responsibilities = vec![vec![0.0_f64; k]; n];

    for _ in 0..max_iter {
        let avg_ll = e_step(data, &params, &mut responsibilities, k, dim);
        m_step(data, &responsibilities, &mut params, k, dim);
        let converged = (avg_ll - prev_ll).abs() < tol;
        prev_ll = avg_ll;
        if converged {
            break;
        }
    }
    // Final E-step so responsibilities and the log-likelihood reflect the last M-step.
    let avg_ll = e_step(data, &params, &mut responsibilities, k, dim);

    let labels = hard_labels(&responsibilities);
    let total_ll = avg_ll * count_to_f64(n);
    let free = free_parameters(k, dim);
    let bic = free.mul_add(count_to_f64(n).ln(), -2.0 * total_ll);
    let aic = 2.0_f64.mul_add(free, -2.0 * total_ll);
    GmmResult {
        labels,
        responsibilities,
        bic,
        aic,
    }
}

/// The fitted mixture parameters.
struct Params {
    /// Mixture weights `π_c`, summing to one.
    weights: Vec<f64>,
    /// Component means, one length-`dim` vector per component.
    means: Vec<Vec<f64>>,
    /// Component covariances, each a row-major `dim×dim` buffer.
    covariances: Vec<Vec<f64>>,
}

/// Initialises the mixture from a k-means partition: uniform weights, the cluster
/// centroids as means, and per-cluster empirical covariances.
fn initialise(data: &[Vec<f64>], k: usize, dim: usize, seed: u64) -> Params {
    let assignment = kmeans(data, k, 100, seed);
    let mut weights = vec![0.0_f64; k];
    let mut means = assignment.centers.clone();
    means.resize(k, vec![0.0_f64; dim]);
    let mut covariances = vec![diagonal(dim, 1.0); k];

    for cluster in 0..k {
        let members: Vec<&Vec<f64>> = data
            .iter()
            .zip(&assignment.labels)
            .filter(|&(_, &l)| l == cluster)
            .map(|(p, _)| p)
            .collect();
        let count = count_to_f64(members.len());
        if let Some(slot) = weights.get_mut(cluster) {
            *slot = if data.is_empty() {
                0.0
            } else {
                count / count_to_f64(data.len())
            };
        }
        if let (Some(mean), Some(cov)) = (means.get(cluster), covariances.get_mut(cluster)) {
            *cov = empirical_covariance(&members, mean, dim);
        }
    }
    Params {
        weights,
        means,
        covariances,
    }
}

/// E-step: fills `responsibilities` with each point's posterior over components and
/// returns the average per-sample log-likelihood.
fn e_step(
    data: &[Vec<f64>],
    params: &Params,
    responsibilities: &mut [Vec<f64>],
    k: usize,
    dim: usize,
) -> f64 {
    let log_consts: Vec<(Vec<f64>, f64)> =
        (0..k).map(|c| gaussian_log_const(params, c, dim)).collect();
    let mut total = 0.0_f64;
    for (point, resp) in data.iter().zip(responsibilities.iter_mut()) {
        // log(π_c) + log N(x | μ_c, Σ_c) per component.
        let log_weighted: Vec<f64> = (0..k)
            .map(|c| {
                let weight = params.weights.get(c).copied().unwrap_or(0.0).max(1e-300);
                weight.ln() + gaussian_log_density(point, params, &log_consts, c, dim)
            })
            .collect();
        let max = log_weighted
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_weighted.iter().map(|&v| (v - max).exp()).sum();
        let log_norm = max + sum_exp.ln();
        total += log_norm;
        for (slot, &lw) in resp.iter_mut().zip(&log_weighted) {
            *slot = (lw - log_norm).exp();
        }
    }
    if data.is_empty() {
        0.0
    } else {
        total / count_to_f64(data.len())
    }
}

/// M-step: re-estimates weights, means, and covariances from `responsibilities`.
fn m_step(
    data: &[Vec<f64>],
    responsibilities: &[Vec<f64>],
    params: &mut Params,
    k: usize,
    dim: usize,
) {
    let n = count_to_f64(data.len());
    for c in 0..k {
        let nk: f64 = responsibilities
            .iter()
            .map(|r| r.get(c).copied().unwrap_or(0.0))
            .sum::<f64>()
            .max(1e-300);
        // Weight.
        if let Some(w) = params.weights.get_mut(c) {
            *w = nk / n;
        }
        // Mean.
        let mut mean = vec![0.0_f64; dim];
        for (point, resp) in data.iter().zip(responsibilities) {
            let r = resp.get(c).copied().unwrap_or(0.0);
            for (m, &x) in mean.iter_mut().zip(point) {
                *m = r.mul_add(x, *m);
            }
        }
        for m in &mut mean {
            *m /= nk;
        }
        // Covariance with reg_covar on the diagonal.
        let mut cov = vec![0.0_f64; dim * dim];
        for (point, resp) in data.iter().zip(responsibilities) {
            let r = resp.get(c).copied().unwrap_or(0.0);
            for a in 0..dim {
                let da = point.get(a).copied().unwrap_or(0.0) - mean.get(a).copied().unwrap_or(0.0);
                for b in 0..dim {
                    let db =
                        point.get(b).copied().unwrap_or(0.0) - mean.get(b).copied().unwrap_or(0.0);
                    let cur = at(&cov, dim, a, b);
                    put(&mut cov, dim, a, b, (r * da).mul_add(db, cur));
                }
            }
        }
        for a in 0..dim {
            for b in 0..dim {
                let cur = at(&cov, dim, a, b) / nk;
                put(
                    &mut cov,
                    dim,
                    a,
                    b,
                    cur + if a == b { REG_COVAR } else { 0.0 },
                );
            }
        }
        if let Some(slot) = params.means.get_mut(c) {
            *slot = mean;
        }
        if let Some(slot) = params.covariances.get_mut(c) {
            *slot = cov;
        }
    }
}

/// Precomputes component `c`'s precision matrix and the log-normaliser
/// `−½(d·ln(2π) + ln|Σ|)` of its Gaussian density.
fn gaussian_log_const(params: &Params, c: usize, dim: usize) -> (Vec<f64>, f64) {
    let cov = params
        .covariances
        .get(c)
        .cloned()
        .unwrap_or_else(|| diagonal(dim, 1.0));
    let precision = symmetric_inverse(&cov, dim);
    let (values, _) = jacobi_eigen(&cov, dim);
    let log_det: f64 = values.iter().map(|&v| v.max(1e-300).ln()).sum();
    let constant = -0.5 * count_to_f64(dim).mul_add((2.0 * std::f64::consts::PI).ln(), log_det);
    (precision, constant)
}

/// Evaluates the multivariate Gaussian log-density of `point` under component `c`.
fn gaussian_log_density(
    point: &[f64],
    params: &Params,
    log_consts: &[(Vec<f64>, f64)],
    c: usize,
    dim: usize,
) -> f64 {
    let Some((precision, constant)) = log_consts.get(c) else {
        return f64::NEG_INFINITY;
    };
    let mean = params.means.get(c);
    let diff: Vec<f64> = (0..dim)
        .map(|i| {
            point.get(i).copied().unwrap_or(0.0)
                - mean.and_then(|m| m.get(i)).copied().unwrap_or(0.0)
        })
        .collect();
    // Mahalanobis term diffᵀ · precision · diff.
    let mut maha = 0.0_f64;
    for a in 0..dim {
        for b in 0..dim {
            maha = (diff.get(a).copied().unwrap_or(0.0) * at(precision, dim, a, b))
                .mul_add(diff.get(b).copied().unwrap_or(0.0), maha);
        }
    }
    constant - 0.5 * maha
}

/// Returns the arg-max-responsibility hard label for each point.
fn hard_labels(responsibilities: &[Vec<f64>]) -> Vec<usize> {
    responsibilities
        .iter()
        .map(|row| {
            let mut best = 0_usize;
            let mut best_v = f64::NEG_INFINITY;
            for (c, &r) in row.iter().enumerate() {
                if r > best_v {
                    best_v = r;
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// Empirical covariance of `members` about `mean` (`n` denominator), with a tiny
/// diagonal ridge so a singleton cluster stays invertible.
fn empirical_covariance(members: &[&Vec<f64>], mean: &[f64], dim: usize) -> Vec<f64> {
    let mut cov = vec![0.0_f64; dim * dim];
    for &point in members {
        for a in 0..dim {
            let da = point.get(a).copied().unwrap_or(0.0) - mean.get(a).copied().unwrap_or(0.0);
            for b in 0..dim {
                let db = point.get(b).copied().unwrap_or(0.0) - mean.get(b).copied().unwrap_or(0.0);
                let cur = at(&cov, dim, a, b);
                put(&mut cov, dim, a, b, da.mul_add(db, cur));
            }
        }
    }
    let count = count_to_f64(members.len()).max(1.0);
    for a in 0..dim {
        for b in 0..dim {
            let cur = at(&cov, dim, a, b) / count;
            put(
                &mut cov,
                dim,
                a,
                b,
                cur + if a == b { REG_COVAR } else { 0.0 },
            );
        }
    }
    cov
}

/// Builds a `dim×dim` diagonal matrix with `value` on the diagonal.
fn diagonal(dim: usize, value: f64) -> Vec<f64> {
    let mut m = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        put(&mut m, dim, i, i, value);
    }
    m
}

/// Free-parameter count for a full-covariance `k`-component mixture in `dim`
/// dimensions: covariances + means + weights, matching scikit-learn's `_n_parameters`.
fn free_parameters(k: usize, dim: usize) -> f64 {
    let cov = count_to_f64(k) * count_to_f64(dim) * (count_to_f64(dim) + 1.0) / 2.0;
    let means = count_to_f64(dim) * count_to_f64(k);
    cov + means + count_to_f64(k) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated 2-D clusters.
    fn two_clusters() -> Vec<Vec<f64>> {
        let mut data = Vec::new();
        for i in 0..10 {
            let jitter = count_to_f64(i) * 0.05;
            data.push(vec![jitter, jitter]);
            data.push(vec![10.0 + jitter, 10.0 - jitter]);
        }
        data
    }

    #[test]
    fn separates_two_clusters() {
        let data = two_clusters();
        let r = gmm_em(&data, 2, 200, 1e-3, 42);
        assert_eq!(r.labels.len(), data.len(), "label count");
        // The first point of each interleaved pair differs in component.
        assert_ne!(r.labels.first(), r.labels.get(1), "adjacent pair merged");
    }

    #[test]
    fn responsibilities_sum_to_one() {
        let data = two_clusters();
        let r = gmm_em(&data, 2, 200, 1e-3, 7);
        for (i, row) in r.responsibilities.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row {i} summed to {sum}");
        }
    }

    #[test]
    fn empty_input_is_empty() {
        let r = gmm_em(&[], 3, 10, 1e-3, 0);
        assert!(r.labels.is_empty(), "labels not empty");
        assert!((r.bic - 0.0).abs() < 1e-12, "bic was {}", r.bic);
    }
}
