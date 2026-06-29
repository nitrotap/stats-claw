//! Perf-gate checker: deserializes a results record and enforces the documented
//! targets. Each `check_*` returns `Err` with a descriptive report on a miss, so
//! both the real recorded results and synthetic rows can drive the gate.

// `#[path]`-included by the `perf` binary, so its `pub` surface looks unreachable
// and some helpers look unused to a single compilation unit — both false
// positives for a `#[path]` test helper, mirroring the `common` harness.
#![allow(dead_code, unreachable_pub)]

use std::collections::HashMap;

use serde::Deserialize;

/// The unified per-family throughput target: every family's gated batch metric
/// must beat its Python baseline by at least this factor.
///
/// Set to **2.0×**, applied consistently across every family record — the gate,
/// the results README, and the throughput benchmarks all use this value.
pub const TARGET_FACTOR: f64 = 2.0;

/// One family's recorded performance results (the shape of a `results/*.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct PerfRecord {
    /// Family name (e.g. `distributions`, `optimizers`), named in any report.
    pub family: String,
    /// Dataset identifier, named in any shortfall report.
    pub dataset: String,
    /// The gated batch-throughput metric and its target.
    pub gate: GateMetric,
    /// The recorded hot-path latency and its sub-millisecond target.
    pub latency: LatencyMetric,
    /// The streaming bounded-memory record.
    pub streaming: StreamingRecord,
}

/// The single gated batch-throughput metric.
#[derive(Debug, Clone, Deserialize)]
pub struct GateMetric {
    /// Name of the gated metric (e.g. `normal_pdf_batch_throughput`).
    pub metric: String,
    /// Measured factor of Rust throughput over the baseline.
    pub factor: f64,
    /// Minimum acceptable factor.
    pub target_factor: f64,
}

/// The recorded hot-path latency at the documented percentile.
#[derive(Debug, Clone, Deserialize)]
pub struct LatencyMetric {
    /// Measured 99th-percentile single-call latency, in milliseconds.
    pub p99_ms: f64,
    /// The sub-millisecond target the measured latency must meet.
    pub target_ms: f64,
}

/// The streaming bounded-memory record: per-estimator state growth plus the
/// auditable out-of-scope list.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamingRecord {
    /// Permitted state-size growth across stream lengths, in bytes (0 = bounded).
    pub bound_bytes_growth_target: i64,
    /// In-scope streaming estimators keyed by name.
    pub estimators: HashMap<String, EstimatorRecord>,
    /// Estimators deliberately recorded as out of scope, keyed by name.
    pub out_of_scope: HashMap<String, String>,
}

/// One streaming estimator's recorded state size and growth.
#[derive(Debug, Clone, Deserialize)]
pub struct EstimatorRecord {
    /// Fixed state size in bytes.
    pub state_bytes: u64,
    /// Observed growth in bytes across the measured stream lengths.
    pub growth_bytes: i64,
    /// `in_scope` or otherwise — recorded so coverage is auditable.
    pub scope: String,
}

/// Fails if the gated batch factor is below [`TARGET_FACTOR`], naming the
/// family, dataset, and the shortfall.
///
/// The bar is the single workspace-wide [`TARGET_FACTOR`] (2.0×), not the
/// per-record `target_factor` field: a record cannot ship a weaker self-declared
/// bar. A record whose `target_factor` disagrees with [`TARGET_FACTOR`] is itself
/// a failure, so the published target stays consistent across every family.
///
/// # Errors
///
/// Returns `Err` with a shortfall report if the measured factor is below
/// [`TARGET_FACTOR`], or if the record's declared `target_factor` does not equal
/// [`TARGET_FACTOR`].
pub fn check_factor(record: &PerfRecord) -> Result<(), String> {
    if (record.gate.target_factor - TARGET_FACTOR).abs() > 1e-9 {
        return Err(format!(
            "perf gate FAILED for family `{}`: declared target {:.3} != unified target {TARGET_FACTOR:.3}; every family must publish the same bar",
            record.family, record.gate.target_factor
        ));
    }
    if record.gate.factor + 1e-9 < TARGET_FACTOR {
        let shortfall = TARGET_FACTOR - record.gate.factor;
        return Err(format!(
            "perf gate FAILED for family `{}` on dataset `{}` metric `{}`: factor {:.3} < target {TARGET_FACTOR:.3} (shortfall {:.3})",
            record.family, record.dataset, record.gate.metric, record.gate.factor, shortfall
        ));
    }
    Ok(())
}

/// Fails if any required published field is missing or non-positive, ensuring
/// the recorded results are complete and auditable.
pub fn check_published_completeness(record: &PerfRecord) -> Result<(), String> {
    if record.dataset.is_empty() {
        return Err("published results missing dataset".to_owned());
    }
    if record.gate.metric.is_empty() {
        return Err("published results missing gated metric name".to_owned());
    }
    if !(record.gate.factor > 0.0 && record.gate.target_factor > 0.0) {
        return Err("published factor and target must be positive".to_owned());
    }
    if !(record.latency.p99_ms > 0.0 && record.latency.target_ms > 0.0) {
        return Err("published latency and target must be positive".to_owned());
    }
    Ok(())
}

/// Fails if the hot-path latency exceeds its sub-millisecond target, naming the
/// measured value and the target.
pub fn check_latency(record: &PerfRecord) -> Result<(), String> {
    if record.latency.p99_ms > record.latency.target_ms + 1e-12 {
        return Err(format!(
            "perf gate FAILED: measured p99 latency {:.6} ms exceeds target {:.6} ms",
            record.latency.p99_ms, record.latency.target_ms
        ));
    }
    Ok(())
}

/// Fails if any in-scope streaming estimator's state grows beyond the documented
/// bound, reporting the growth trend.
pub fn check_streaming_bounded(record: &PerfRecord) -> Result<(), String> {
    for (name, est) in &record.streaming.estimators {
        if est.growth_bytes > record.streaming.bound_bytes_growth_target {
            return Err(format!(
                "perf gate FAILED: streaming estimator `{name}` state growth {} bytes exceeds bound {} bytes (growth trend is not flat)",
                est.growth_bytes, record.streaming.bound_bytes_growth_target
            ));
        }
    }
    Ok(())
}

/// Fails unless streaming coverage is auditable: either the family records at
/// least one in-scope estimator, or — for a batch-only family with no online
/// formulation — it explicitly records why under `out_of_scope`.
///
/// # Errors
///
/// Returns `Err` if a family records neither in-scope estimators nor an
/// `out_of_scope` rationale, or if any recorded estimator is not `in_scope`.
pub fn check_streaming_coverage(record: &PerfRecord) -> Result<(), String> {
    if record.streaming.estimators.is_empty() {
        if record.streaming.out_of_scope.is_empty() {
            return Err(format!(
                "family `{}` records no streaming estimators and no out_of_scope rationale; coverage is not auditable",
                record.family
            ));
        }
        return Ok(());
    }
    for (name, est) in &record.streaming.estimators {
        if est.scope != "in_scope" {
            return Err(format!(
                "estimator `{name}` is neither in_scope nor listed out_of_scope"
            ));
        }
    }
    Ok(())
}

impl PerfRecord {
    /// A synthetic record whose factor falls below target — drives the gate-fail
    /// path for [`check_factor`].
    pub fn synthetic_below_target() -> Self {
        Self {
            family: "synthetic_family".to_owned(),
            dataset: "synthetic_dataset".to_owned(),
            gate: GateMetric {
                metric: "synthetic_metric".to_owned(),
                factor: 0.5,
                target_factor: TARGET_FACTOR,
            },
            latency: LatencyMetric {
                p99_ms: 0.01,
                target_ms: 1.0,
            },
            streaming: Self::flat_streaming(),
        }
    }

    /// A synthetic record whose gated factor is exactly `factor`, with everything
    /// else bounded and in-target — drives the factor checks at the 2.0× boundary.
    ///
    /// # Arguments
    ///
    /// * `factor` — the measured throughput factor to plant in the gated metric.
    pub fn synthetic_with_factor(factor: f64) -> Self {
        let mut r = Self::synthetic_below_target();
        r.gate.factor = factor;
        r
    }

    /// A synthetic record whose latency exceeds target — drives [`check_latency`].
    pub fn synthetic_over_latency() -> Self {
        let mut r = Self::synthetic_below_target();
        r.gate.factor = 3.0;
        r.latency = LatencyMetric {
            p99_ms: 5.0,
            target_ms: 1.0,
        };
        r
    }

    /// A synthetic record whose streaming state grows — drives
    /// [`check_streaming_bounded`].
    pub fn synthetic_growing_streaming() -> Self {
        let mut r = Self::synthetic_below_target();
        r.gate.factor = 3.0;
        let mut estimators = HashMap::new();
        estimators.insert(
            "LeakyEstimator".to_owned(),
            EstimatorRecord {
                state_bytes: 1024,
                growth_bytes: 4096,
                scope: "in_scope".to_owned(),
            },
        );
        r.streaming = StreamingRecord {
            bound_bytes_growth_target: 0,
            estimators,
            out_of_scope: HashMap::new(),
        };
        r
    }

    /// A flat (bounded) streaming sub-record for the synthetic factor/latency
    /// fixtures, so their non-streaming checks are not accidentally tripped.
    fn flat_streaming() -> StreamingRecord {
        let mut estimators = HashMap::new();
        estimators.insert(
            "Flat".to_owned(),
            EstimatorRecord {
                state_bytes: 24,
                growth_bytes: 0,
                scope: "in_scope".to_owned(),
            },
        );
        StreamingRecord {
            bound_bytes_growth_target: 0,
            estimators,
            out_of_scope: HashMap::new(),
        }
    }
}
