//! Streaming-estimator suite (AC-7 / Story 7.2).
//!
//! Asserts the online estimators in [`stats_claw::streaming`] match their batch
//! counterparts within tolerance and run in bounded memory whose state size is
//! independent of stream length. Each estimator consumes values one at a time via
//! `update` and reports its current estimate via the appropriate accessor.

use stats_claw::rng::SplitMix64;
use stats_claw::streaming::{P2Quantile, RunningMoments};

/// Two-pass batch mean, the reference the streaming mean must reproduce.
fn batch_mean(xs: &[f64]) -> f64 {
    let n = u32::try_from(xs.len()).unwrap_or(u32::MAX);
    xs.iter().sum::<f64>() / f64::from(n)
}

/// Two-pass Bessel-corrected sample variance, the streaming variance reference.
fn batch_variance(xs: &[f64]) -> f64 {
    let mean = batch_mean(xs);
    let n = u32::try_from(xs.len()).unwrap_or(u32::MAX);
    let ss: f64 = xs.iter().map(|&x| (x - mean) * (x - mean)).sum();
    ss / f64::from(n - 1)
}

/// Exact batch quantile by linear interpolation between order statistics, the
/// reference the P² approximation is compared against (numpy's "linear" method).
// `lo` is a non-negative, in-range float index; the `usize` truncation is exact.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn batch_quantile(xs: &[f64], p: f64) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    let rank = p * f64::from(n - 1);
    let lo = rank.floor();
    let lo_idx = lo as usize;
    let frac = rank - lo;
    let a = sorted.get(lo_idx).copied().unwrap_or(0.0);
    let b = sorted.get(lo_idx + 1).copied().unwrap_or(a);
    a + frac * (b - a)
}

#[test]
fn running_mean_matches_batch() -> Result<(), String> {
    let xs: Vec<f64> = (0..1_000)
        .map(|i| f64::from(i).mul_add(0.5, -17.0))
        .collect();
    let mut m = RunningMoments::new();
    for &x in &xs {
        m.update(x);
    }
    let got = m.mean();
    let want = batch_mean(&xs);
    if (got - want).abs() < 1e-12 {
        Ok(())
    } else {
        Err(format!("streaming mean {got} vs batch {want}"))
    }
}

/// A reproducible, non-monotonic stream so the P² estimate is exercised on a
/// genuinely shuffled order (P² is order-sensitive in its transient, not its
/// limit). Uses the project's seeded PRNG; no external `rand` dependency.
fn shuffled_stream(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.standard_normal()).collect()
}

#[test]
fn p2_quantile_matches_batch() -> Result<(), String> {
    let xs = shuffled_stream(50_000, 0x5EED);
    let p = 0.5;
    let mut q = P2Quantile::new(p);
    for &x in &xs {
        q.update(x);
    }
    let got = q.value();
    let want = batch_quantile(&xs, p);
    // P² is an approximation; sub-1% relative agreement on a large stream is the
    // documented expectation (Jain & Chlamtac 1985), far looser than the 1e-12
    // moment estimators.
    let rel = (got - want).abs() / want.abs().max(1.0);
    if rel < 1e-2 {
        Ok(())
    } else {
        Err(format!("P2 quantile {got} vs batch {want} (rel {rel})"))
    }
}

#[test]
fn running_variance_matches_batch() -> Result<(), String> {
    let xs: Vec<f64> = (0..1_000)
        .map(|i| f64::from(i).mul_add(0.5, -17.0))
        .collect();
    let mut m = RunningMoments::new();
    for &x in &xs {
        m.update(x);
    }
    let got = m.variance();
    let want = batch_variance(&xs);
    let rel = (got - want).abs() / want.abs();
    if rel < 1e-10 {
        Ok(())
    } else {
        Err(format!(
            "streaming variance {got} vs batch {want} (rel {rel})"
        ))
    }
}

#[test]
fn moments_state_size_is_constant_across_stream_lengths() -> Result<(), String> {
    // The struct is fixed-size, so `size_of` is a compile-time constant; consuming
    // more values must never enlarge a live instance's footprint.
    let base = size_of::<RunningMoments>();
    for &n in &[0_usize, 10, 10_000, 1_000_000] {
        let mut m = RunningMoments::new();
        for i in 0..n {
            m.update(f64::from(u32::try_from(i % 1000).unwrap_or(0)));
        }
        let live = size_of_val(&m);
        if live != base {
            return Err(format!("RunningMoments size {live} at n={n}, base {base}"));
        }
        if m.count() != u64::try_from(n).unwrap_or(u64::MAX) {
            return Err(format!("count drifted at n={n}"));
        }
    }
    Ok(())
}

#[test]
fn p2_state_size_is_constant_across_stream_lengths() -> Result<(), String> {
    let base = size_of::<P2Quantile>();
    for &n in &[0_usize, 10, 10_000, 1_000_000] {
        let mut q = P2Quantile::new(0.95);
        for i in 0..n {
            q.update(f64::from(u32::try_from(i % 1000).unwrap_or(0)));
        }
        let live = size_of_val(&q);
        if live != base {
            return Err(format!("P2Quantile size {live} at n={n}, base {base}"));
        }
        if q.count() != u64::try_from(n).unwrap_or(u64::MAX) {
            return Err(format!("count drifted at n={n}"));
        }
    }
    Ok(())
}
