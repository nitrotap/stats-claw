//! Phase 3 performance gate (AC-7 stories 7.1, 7.2, 7.3).
//!
//! Reads the committed results under `benches/results/` and fails if any recorded
//! factor or latency misses its documented target, or if streaming state is not
//! bounded. The gate is offline and hermetic — it checks recorded measurements,
//! never re-runs a benchmark — so it behaves like the golden-fixture suites.
//!
//! Coverage is **per family** (AC-7 / roadmap §W3): the gate discovers every
//! `benches/results/*.json` record and enforces the unified [`gate::TARGET_FACTOR`]
//! (2.0×) against each, so a new family record automatically joins the gate.
//!
//! The gate-checker logic lives in [`gate`] and is fed both the real recorded
//! results and synthetic rows, so the failure paths (below-target factor,
//! over-target latency, growing streaming state) are genuinely exercised.

#[path = "perf/gate.rs"]
mod gate;

use std::path::PathBuf;

use gate::PerfRecord;

/// Absolute path to the committed `benches/results/` directory.
fn results_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/benches/results"))
}

/// Loads every committed `benches/results/*.json` family record, sorted by family.
///
/// # Errors
///
/// Returns `Err` if the results directory cannot be read, a record cannot be read
/// or parsed, or no `*.json` records are present at all.
fn load_all_families() -> Result<Vec<PerfRecord>, String> {
    let dir = results_dir();
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("read dir {}: {e}", dir.display()))?;
    let mut records = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| format!("dir entry: {e}"))?.path();
        if path.extension().is_some_and(|e| e == "json") {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            let record: PerfRecord = serde_json::from_str(&text)
                .map_err(|e| format!("parse {}: {e}", path.display()))?;
            records.push(record);
        }
    }
    if records.is_empty() {
        return Err(format!(
            "no results/*.json records found under {}",
            dir.display()
        ));
    }
    records.sort_by(|a, b| a.family.cmp(&b.family));
    Ok(records)
}

#[test]
fn every_family_factor_meets_target() -> Result<(), String> {
    let records = load_all_families()?;
    for record in &records {
        gate::check_factor(record)?;
    }
    Ok(())
}

#[test]
fn every_family_publishes_complete_results() -> Result<(), String> {
    let records = load_all_families()?;
    for record in &records {
        gate::check_published_completeness(record)?;
    }
    Ok(())
}

#[test]
fn all_numeric_families_are_covered() -> Result<(), String> {
    let records = load_all_families()?;
    let present: std::collections::BTreeSet<&str> =
        records.iter().map(|r| r.family.as_str()).collect();
    let expected = [
        "algorithms",
        "distributions",
        "optimizers",
        "resampling",
        "stat_tests",
    ];
    let missing: Vec<&str> = expected
        .iter()
        .filter(|f| !present.contains(*f))
        .copied()
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "per-family perf coverage incomplete: missing records for {missing:?} (have {present:?})"
        ))
    }
}

#[test]
fn synthetic_below_target_factor_fails_the_gate() {
    let record = PerfRecord::synthetic_below_target();
    let result = gate::check_factor(&record);
    assert!(
        result.is_err(),
        "gate must fail on a below-target factor, got {result:?}"
    );
    let msg = result.err().unwrap_or_default();
    assert!(
        msg.contains("synthetic_dataset") && msg.contains("shortfall"),
        "failure report must name the dataset and shortfall, got: {msg}"
    );
}

#[test]
fn factor_just_below_two_x_fails_the_gate() {
    // A record measured at 1.6x — comfortably above the retired 1.5x bar — must
    // now FAIL the unified 2.0x target. This pins the resolved D2 decision.
    let record = PerfRecord::synthetic_with_factor(1.6);
    let result = gate::check_factor(&record);
    assert!(
        result.is_err(),
        "a 1.6x factor must fail the 2.0x gate, got {result:?}"
    );
    let msg = result.err().unwrap_or_default();
    assert!(
        msg.contains("2.000") && msg.contains("1.600"),
        "failure report must name the target and measured factor, got: {msg}"
    );
}

#[test]
fn record_declaring_a_weaker_target_fails_the_gate() {
    // Even a fast family fails if it publishes a self-declared target below the
    // unified bar — the published target must be consistent across families.
    let mut record = PerfRecord::synthetic_with_factor(5.0);
    record.gate.target_factor = 1.5;
    let result = gate::check_factor(&record);
    assert!(
        result.is_err(),
        "a record declaring a 1.5x target must fail, got {result:?}"
    );
    let msg = result.err().unwrap_or_default();
    assert!(
        msg.contains("unified target"),
        "failure report must call out the inconsistent declared target, got: {msg}"
    );
}

#[test]
fn synthetic_over_target_latency_fails_the_gate() {
    let record = PerfRecord::synthetic_over_latency();
    let result = gate::check_latency(&record);
    assert!(
        result.is_err(),
        "gate must fail on over-target latency, got {result:?}"
    );
    let msg = result.err().unwrap_or_default();
    assert!(
        msg.contains("measured") && msg.contains("target"),
        "failure report must name measured latency and target, got: {msg}"
    );
}

#[test]
fn every_family_latency_meets_sub_millisecond_target() -> Result<(), String> {
    let records = load_all_families()?;
    for record in &records {
        gate::check_latency(record)?;
    }
    Ok(())
}

#[test]
fn every_family_streaming_state_is_bounded() -> Result<(), String> {
    let records = load_all_families()?;
    for record in &records {
        gate::check_streaming_bounded(record)?;
    }
    Ok(())
}

#[test]
fn synthetic_growing_streaming_state_fails_the_gate() {
    let record = PerfRecord::synthetic_growing_streaming();
    let result = gate::check_streaming_bounded(&record);
    assert!(
        result.is_err(),
        "gate must fail on growing streaming state, got {result:?}"
    );
    let msg = result.err().unwrap_or_default();
    assert!(
        msg.contains("growth"),
        "failure report must describe the growth trend, got: {msg}"
    );
}

#[test]
fn every_family_streaming_coverage_is_auditable() -> Result<(), String> {
    let records = load_all_families()?;
    for record in &records {
        gate::check_streaming_coverage(record)?;
    }
    Ok(())
}
