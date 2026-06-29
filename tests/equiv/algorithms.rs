//! Equivalence suite for the clustering algorithm family.
//!
//! Each test loads a committed `scikit-learn` golden fixture and asserts the
//! stats-claw implementation agrees by adjusted Rand index (≥ 0.99), that the
//! identifiable scalars (inertia, discovered cluster count, core-sample count)
//! match, and that a fixed seed yields identical output on repeated runs.
//!
//! Python never runs here — the fixtures are the offline source of truth.

use crate::common;
use crate::common::HarnessError;
use serde_json::Value;
use stats_claw::algorithms::clustering::{
    affinity_propagation, agglomerative, cluster_count, dbscan, kmeans, mean_shift, spectral,
    Linkage,
};

/// Parses the `data` key of a fixture as a row-major point matrix.
fn data_matrix(fx: &Value) -> Result<Vec<Vec<f64>>, HarnessError> {
    fx.get("data")
        .and_then(Value::as_array)
        .ok_or(HarnessError::Shape("data"))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or(HarnessError::Shape("data row"))?
                .iter()
                .map(|v| v.as_f64().ok_or(HarnessError::Shape("data value")))
                .collect()
        })
        .collect()
}

/// Parses an integer label array, mapping `scikit-learn`'s `-1` noise label onto
/// the stats-claw [`stats_claw::algorithms::clustering::NOISE`] sentinel.
fn labels(fx: &Value, key: &'static str) -> Result<Vec<usize>, HarnessError> {
    fx.get(key)
        .and_then(Value::as_array)
        .ok_or(HarnessError::Shape(key))?
        .iter()
        .map(|v| {
            let raw = v.as_i64().ok_or(HarnessError::Shape(key))?;
            Ok(usize::try_from(raw).unwrap_or(stats_claw::algorithms::clustering::NOISE))
        })
        .collect()
}

/// Reads a top-level `f64` scalar from a fixture.
fn scalar(fx: &Value, key: &'static str) -> Result<f64, HarnessError> {
    fx.get(key)
        .and_then(Value::as_f64)
        .ok_or(HarnessError::Shape(key))
}

/// Reads a top-level unsigned scalar from a fixture.
fn usize_scalar(fx: &Value, key: &'static str) -> Result<usize, HarnessError> {
    let raw = fx
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(HarnessError::Shape(key))?;
    Ok(usize::try_from(raw).unwrap_or(0))
}

#[test]
fn kmeans_agrees_with_sklearn_by_ari() -> Result<(), HarnessError> {
    let fx = common::load("algo_kmeans_blobs")?;
    let data = data_matrix(&fx)?;
    let reference = labels(&fx, "labels")?;
    let result = kmeans(&data, 3, 300, 42);
    let ari = common::adjusted_rand_index(&result.labels, &reference);
    assert!(ari >= 0.99, "k-means ARI was {ari}");
    common::assert_close(result.inertia, scalar(&fx, "inertia")?, 0.0, 1e-6);
    assert_eq!(cluster_count(&result.labels), 3, "discovered cluster count");
    Ok(())
}

#[test]
fn kmeans_is_deterministic_for_fixed_seed() -> Result<(), HarnessError> {
    let fx = common::load("algo_kmeans_blobs")?;
    let data = data_matrix(&fx)?;
    assert_eq!(
        kmeans(&data, 3, 300, 7).labels,
        kmeans(&data, 3, 300, 7).labels
    );
    Ok(())
}

#[test]
fn dbscan_agrees_with_sklearn_by_ari() -> Result<(), HarnessError> {
    let fx = common::load("algo_dbscan_blobs")?;
    let data = data_matrix(&fx)?;
    let reference = labels(&fx, "labels")?;
    let eps = scalar(&fx, "eps")?;
    let min_samples = usize_scalar(&fx, "min_samples")?;
    let result = dbscan(&data, eps, min_samples);
    let ari = common::adjusted_rand_index(&result.labels, &reference);
    assert!(ari >= 0.99, "DBSCAN ARI was {ari}");
    assert_eq!(
        result.core_samples.len(),
        usize_scalar(&fx, "core_sample_count")?,
        "core-sample count"
    );
    assert_eq!(cluster_count(&result.labels), 3, "discovered cluster count");
    Ok(())
}

#[test]
fn dbscan_is_deterministic_for_fixed_inputs() -> Result<(), HarnessError> {
    let fx = common::load("algo_dbscan_blobs")?;
    let data = data_matrix(&fx)?;
    let eps = scalar(&fx, "eps")?;
    let min_samples = usize_scalar(&fx, "min_samples")?;
    assert_eq!(
        dbscan(&data, eps, min_samples).labels,
        dbscan(&data, eps, min_samples).labels
    );
    Ok(())
}

#[test]
fn hierarchical_agrees_with_sklearn_by_ari() -> Result<(), HarnessError> {
    let fx = common::load("algo_hierarchical_blobs")?;
    let data = data_matrix(&fx)?;
    let reference = labels(&fx, "labels")?;
    let result = agglomerative(&data, 3, Linkage::Ward);
    let ari = common::adjusted_rand_index(&result, &reference);
    assert!(ari >= 0.99, "agglomerative ARI was {ari}");
    assert_eq!(cluster_count(&result), 3, "discovered cluster count");
    Ok(())
}

#[test]
fn hierarchical_is_deterministic() -> Result<(), HarnessError> {
    let fx = common::load("algo_hierarchical_blobs")?;
    let data = data_matrix(&fx)?;
    assert_eq!(
        agglomerative(&data, 3, Linkage::Ward),
        agglomerative(&data, 3, Linkage::Ward)
    );
    Ok(())
}

#[test]
fn mean_shift_agrees_with_sklearn_by_ari() -> Result<(), HarnessError> {
    let fx = common::load("algo_mean_shift_blobs")?;
    let data = data_matrix(&fx)?;
    let reference = labels(&fx, "labels")?;
    let bandwidth = scalar(&fx, "bandwidth")?;
    let result = mean_shift(&data, bandwidth);
    let ari = common::adjusted_rand_index(&result.labels, &reference);
    assert!(ari >= 0.99, "mean-shift ARI was {ari}");
    assert_eq!(
        result.n_clusters,
        usize_scalar(&fx, "n_clusters")?,
        "discovered cluster count"
    );
    Ok(())
}

#[test]
fn mean_shift_is_deterministic() -> Result<(), HarnessError> {
    let fx = common::load("algo_mean_shift_blobs")?;
    let data = data_matrix(&fx)?;
    let bandwidth = scalar(&fx, "bandwidth")?;
    assert_eq!(
        mean_shift(&data, bandwidth).labels,
        mean_shift(&data, bandwidth).labels
    );
    Ok(())
}

#[test]
fn affinity_agrees_with_sklearn_by_ari() -> Result<(), HarnessError> {
    let fx = common::load("algo_affinity_blobs")?;
    let data = data_matrix(&fx)?;
    let reference = labels(&fx, "labels")?;
    let damping = scalar(&fx, "damping")?;
    let preference = scalar(&fx, "preference")?;
    let result = affinity_propagation(&data, damping, preference, 200);
    let ari = common::adjusted_rand_index(&result, &reference);
    assert!(ari >= 0.99, "affinity ARI was {ari}");
    assert_eq!(
        cluster_count(&result),
        usize_scalar(&fx, "n_clusters")?,
        "discovered cluster count"
    );
    Ok(())
}

#[test]
fn affinity_is_deterministic() -> Result<(), HarnessError> {
    let fx = common::load("algo_affinity_blobs")?;
    let data = data_matrix(&fx)?;
    let damping = scalar(&fx, "damping")?;
    let preference = scalar(&fx, "preference")?;
    assert_eq!(
        affinity_propagation(&data, damping, preference, 200),
        affinity_propagation(&data, damping, preference, 200)
    );
    Ok(())
}

#[test]
fn spectral_agrees_with_sklearn_by_ari() -> Result<(), HarnessError> {
    let fx = common::load("algo_spectral_blobs")?;
    let data = data_matrix(&fx)?;
    let reference = labels(&fx, "labels")?;
    let gamma = scalar(&fx, "gamma")?;
    let result = spectral(&data, 3, gamma, 42);
    let ari = common::adjusted_rand_index(&result, &reference);
    assert!(ari >= 0.99, "spectral ARI was {ari}");
    assert_eq!(cluster_count(&result), 3, "discovered cluster count");
    Ok(())
}

#[test]
fn spectral_is_deterministic_for_fixed_seed() -> Result<(), HarnessError> {
    let fx = common::load("algo_spectral_blobs")?;
    let data = data_matrix(&fx)?;
    let gamma = scalar(&fx, "gamma")?;
    assert_eq!(spectral(&data, 3, gamma, 7), spectral(&data, 3, gamma, 7));
    Ok(())
}
