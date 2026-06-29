//! Equivalence suite for the association-analysis family.
//!
//! Loads a committed golden fixture and asserts the stats-claw association miner
//! reproduces the reference quantities it pins: the per-itemset **support** from
//! `mlxtend.frequent_patterns.apriori` and the per-rule **support / confidence /
//! lift / leverage / conviction** from
//! `mlxtend.frequent_patterns.association_rules` (metric `"confidence"`). Python
//! never runs here — the fixture is the offline source of truth.
//!
//! Both miners consume the same one-hot boolean transaction matrix and compute the
//! identical integer-count support ratios, so every metric is an exact rational and
//! agrees to machine precision. Itemsets are matched by their item-index set and
//! rules by their `(antecedent, consequent)` pair, so the comparison is independent
//! of either side's ordering (both are canonically sorted regardless). mlxtend's
//! `+inf` conviction for a perfectly-confident rule is stored as JSON `null`; the
//! Rust side must produce an infinite conviction there.

use crate::common;
use crate::common::HarnessError;
use stats_claw::algorithms::association::{
    AssociationError, AssociationRule, FrequentItemset, RuleMetric, apriori, association_rules,
};

/// Tolerance for the equivalence comparisons.
///
/// Supports, confidences, lifts, leverages, and convictions are exact rational
/// ratios of integer transaction counts on both sides (mlxtend and stats-claw use the
/// same `count / n` arithmetic), so the achieved max-abs difference is at the
/// `~1e-15` floating-point floor; the gate is set at the family-standard `1e-12`.
const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-12;

/// Maps a borrowed [`AssociationError`] into the harness error type so tests use
/// `?` on a mining result.
fn mine_err(e: &AssociationError) -> HarnessError {
    HarnessError::Parse(format!("association mining failed: {e}"))
}

/// Reads the fixture's one-hot transaction matrix (`0`/`1` ints) as `Vec<Vec<bool>>`.
fn load_transactions(fx: &serde_json::Value) -> Result<Vec<Vec<bool>>, HarnessError> {
    let rows = fx
        .get("transactions")
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape("transactions"))?;
    rows.iter()
        .map(|row| {
            row.as_array()
                .ok_or(HarnessError::Shape("transactions"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n != 0)
                        .ok_or(HarnessError::Shape("transactions"))
                })
                .collect()
        })
        .collect()
}

/// Reads an array of unsigned-integer item indices into a `Vec<usize>`.
fn index_list(value: Option<&serde_json::Value>) -> Result<Vec<usize>, HarnessError> {
    let arr = value
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape("items"))?;
    arr.iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(HarnessError::Shape("items"))
        })
        .collect()
}

/// Reads the fixture's min-support / min-threshold settings and the rule metric.
fn load_settings(fx: &serde_json::Value) -> Result<(f64, f64), HarnessError> {
    let min_support = common::scalar(fx, "min_support")?;
    let min_threshold = common::scalar(fx, "min_threshold")?;
    // The fixture is generated with metric="confidence"; the Rust test pins the same.
    let metric = fx
        .get("rule_metric")
        .and_then(serde_json::Value::as_str)
        .ok_or(HarnessError::Shape("rule_metric"))?;
    assert_eq!(
        metric, "confidence",
        "fixture must use the confidence metric"
    );
    Ok((min_support, min_threshold))
}

/// Finds the stats-claw frequent itemset whose item set equals `items`.
fn find_itemset<'a>(
    frequent: &'a [FrequentItemset],
    items: &[usize],
) -> Option<&'a FrequentItemset> {
    frequent.iter().find(|s| s.items() == items)
}

/// Finds the stats-claw rule whose antecedent and consequent both match.
fn find_rule<'a>(
    rules: &'a [AssociationRule],
    antecedent: &[usize],
    consequent: &[usize],
) -> Option<&'a AssociationRule> {
    rules
        .iter()
        .find(|r| r.antecedent() == antecedent && r.consequent() == consequent)
}

/// The stats-claw frequent itemsets reproduce mlxtend's apriori output: the same set
/// of itemsets, each at the same support.
#[test]
fn frequent_itemsets_agree_with_mlxtend() -> Result<(), HarnessError> {
    let fx = common::load("association")?;
    let (min_support, _) = load_settings(&fx)?;
    let transactions = load_transactions(&fx)?;
    let frequent = apriori(&transactions, min_support).map_err(|e| mine_err(&e))?;

    let expected = fx
        .get("itemsets")
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape("itemsets"))?;

    // Same count of frequent itemsets on both sides.
    assert_eq!(
        frequent.len(),
        expected.len(),
        "itemset count mismatch: rust={}, mlxtend={}",
        frequent.len(),
        expected.len()
    );
    // Every mlxtend itemset is present in the stats-claw output at the same support.
    for entry in expected {
        let items = index_list(entry.get("items"))?;
        let want = common::field(entry, "support")?;
        let got = find_itemset(&frequent, &items)
            .ok_or(HarnessError::Shape("itemsets"))?
            .support();
        common::assert_close(got, want, ATOL, RTOL);
    }
    Ok(())
}

/// The stats-claw association rules reproduce mlxtend's output: the same set of
/// `(antecedent ⇒ consequent)` rules, each with matching metrics.
#[test]
fn association_rules_agree_with_mlxtend() -> Result<(), HarnessError> {
    let fx = common::load("association")?;
    let (min_support, min_threshold) = load_settings(&fx)?;
    let transactions = load_transactions(&fx)?;
    let frequent = apriori(&transactions, min_support).map_err(|e| mine_err(&e))?;
    let rules = association_rules(&frequent, RuleMetric::Confidence, min_threshold)
        .map_err(|e| mine_err(&e))?;

    let expected = fx
        .get("rules")
        .and_then(serde_json::Value::as_array)
        .ok_or(HarnessError::Shape("rules"))?;

    // Same count of rules on both sides (no extra or missing partitions).
    assert_eq!(
        rules.len(),
        expected.len(),
        "rule count mismatch: rust={}, mlxtend={}",
        rules.len(),
        expected.len()
    );
    for entry in expected {
        let antecedent = index_list(entry.get("antecedent"))?;
        let consequent = index_list(entry.get("consequent"))?;
        let rule =
            find_rule(&rules, &antecedent, &consequent).ok_or(HarnessError::Shape("rules"))?;

        common::assert_close(rule.support(), common::field(entry, "support")?, ATOL, RTOL);
        common::assert_close(
            rule.confidence(),
            common::field(entry, "confidence")?,
            ATOL,
            RTOL,
        );
        common::assert_close(rule.lift(), common::field(entry, "lift")?, ATOL, RTOL);
        common::assert_close(
            rule.leverage(),
            common::field(entry, "leverage")?,
            ATOL,
            RTOL,
        );
        assert_conviction(rule, entry);
    }
    Ok(())
}

/// Asserts the rule's conviction matches the fixture, treating a JSON `null` as
/// mlxtend's `+inf` (a perfectly-confident rule).
fn assert_conviction(rule: &AssociationRule, entry: &serde_json::Value) {
    match entry.get("conviction").and_then(serde_json::Value::as_f64) {
        Some(want) => common::assert_close(rule.conviction(), want, ATOL, RTOL),
        None => assert!(
            rule.conviction().is_infinite(),
            "expected infinite conviction (mlxtend null), got {}",
            rule.conviction()
        ),
    }
}
