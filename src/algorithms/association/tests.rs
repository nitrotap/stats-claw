//! Unit tests for the association-rule base block, added one behaviour at a time.
//!
//! These exercise Apriori frequent-itemset generation and the rule metrics against
//! hand-computed values and every typed error path; the cross-library equivalence
//! (vs `mlxtend`) lives in `tests/equiv/association.rs`.

use super::*;

/// Apriori finds the frequent single items at their exact supports.
#[test]
fn apriori_finds_frequent_singletons() -> Result<(), AssociationError> {
    // Item 0 appears in 3/4, item 1 in 3/4, item 2 in 1/4.
    let t = vec![
        vec![true, true, false],
        vec![true, true, true],
        vec![true, true, false],
        vec![false, false, false],
    ];
    let frequent = apriori(&t, 0.5)?;
    // Only items 0 and 1 clear support 0.5; item 2 (0.25) is dropped.
    let singles: Vec<&FrequentItemset> = frequent.iter().filter(|s| s.items().len() == 1).collect();
    assert_eq!(singles.len(), 2, "two frequent singletons, got {singles:?}");
    let s0 = singles
        .iter()
        .find(|s| s.items() == [0])
        .map_or(f64::NAN, |s| s.support());
    assert!((s0 - 0.75).abs() < 1e-12, "item 0 support was {s0}");
    Ok(())
}

/// Apriori grows frequent pairs (and triples) from frequent singletons.
#[test]
fn apriori_finds_frequent_pairs_and_triples() -> Result<(), AssociationError> {
    // All three items co-occur in 2/4 transactions; items 0,1 in 3/4.
    let t = vec![
        vec![true, true, true],
        vec![true, true, true],
        vec![true, true, false],
        vec![false, false, false],
    ];
    let frequent = apriori(&t, 0.5)?;
    // Pair {0,1}: present in rows 0,1,2 -> 3/4.
    let pair01 = frequent
        .iter()
        .find(|s| s.items() == [0, 1])
        .map_or(f64::NAN, FrequentItemset::support);
    assert!(
        (pair01 - 0.75).abs() < 1e-12,
        "pair {{0,1}} support was {pair01}"
    );
    // Triple {0,1,2}: present in rows 0,1 -> 2/4 = 0.5, exactly at the threshold.
    let triple = frequent
        .iter()
        .find(|s| s.items() == [0, 1, 2])
        .map_or(f64::NAN, FrequentItemset::support);
    assert!((triple - 0.5).abs() < 1e-12, "triple support was {triple}");
    Ok(())
}

/// Rule metrics match the hand-computed support / confidence / lift / leverage /
/// conviction for a known partition.
#[test]
fn rule_metrics_match_hand_values() -> Result<(), String> {
    // 4 transactions: {0,1},{0,1},{0},{1}.
    //   support(0)=3/4, support(1)=3/4, support({0,1})=2/4=0.5.
    let t = vec![
        vec![true, true],
        vec![true, true],
        vec![true, false],
        vec![false, true],
    ];
    let frequent = apriori(&t, 0.25).map_err(|e| e.to_string())?;
    let rules =
        association_rules(&frequent, RuleMetric::Confidence, 0.0).map_err(|e| e.to_string())?;
    let r = rules
        .iter()
        .find(|r| r.antecedent() == [0] && r.consequent() == [1])
        .ok_or_else(|| "rule {0}=>{1} present".to_owned())?;
    // confidence = 0.5 / 0.75 = 2/3.
    assert!(
        (r.confidence() - 2.0 / 3.0).abs() < 1e-12,
        "conf was {}",
        r.confidence()
    );
    // lift = confidence / support(1) = (2/3) / 0.75 = 8/9.
    assert!(
        (r.lift() - 8.0 / 9.0).abs() < 1e-12,
        "lift was {}",
        r.lift()
    );
    // leverage = 0.5 - 0.75*0.75 = -0.0625.
    assert!(
        (r.leverage() - (-0.0625)).abs() < 1e-12,
        "leverage was {}",
        r.leverage()
    );
    // conviction = (1 - 0.75) / (1 - 2/3) = 0.25 / (1/3) = 0.75.
    assert!(
        (r.conviction() - 0.75).abs() < 1e-12,
        "conviction was {}",
        r.conviction()
    );
    // support of the rule = joint support = 0.5.
    assert!(
        (r.support() - 0.5).abs() < 1e-12,
        "support was {}",
        r.support()
    );
    Ok(())
}

/// A perfectly-confident rule (confidence == 1) has conviction `+∞`.
#[test]
fn perfect_confidence_gives_infinite_conviction() -> Result<(), String> {
    // Item 0 always co-occurs with item 1: {0,1},{0,1},{1}. support(0)=2/3,
    // support({0,1})=2/3, so confidence({0}=>{1}) = 1.
    let t = vec![vec![true, true], vec![true, true], vec![false, true]];
    let frequent = apriori(&t, 0.25).map_err(|e| e.to_string())?;
    let rules =
        association_rules(&frequent, RuleMetric::Confidence, 0.0).map_err(|e| e.to_string())?;
    let r = rules
        .iter()
        .find(|r| r.antecedent() == [0] && r.consequent() == [1])
        .ok_or_else(|| "rule {0}=>{1} present".to_owned())?;
    assert!(
        (r.confidence() - 1.0).abs() < 1e-12,
        "conf was {}",
        r.confidence()
    );
    assert!(
        r.conviction().is_infinite(),
        "conviction was {}",
        r.conviction()
    );
    Ok(())
}

/// The metric threshold filters out rules below `min_threshold` for that metric.
#[test]
fn metric_threshold_filters_rules() -> Result<(), AssociationError> {
    // {0,1},{0,1},{0},{1}: confidence({0}=>{1}) = 2/3 ≈ 0.667.
    let t = vec![
        vec![true, true],
        vec![true, true],
        vec![true, false],
        vec![false, true],
    ];
    let frequent = apriori(&t, 0.25)?;
    // A 0.9 confidence floor drops the 2/3-confidence rule entirely.
    let strict = association_rules(&frequent, RuleMetric::Confidence, 0.9)?;
    assert!(
        !strict
            .iter()
            .any(|r| r.antecedent() == [0] && r.consequent() == [1]),
        "0.9 floor must drop the 2/3-confidence rule, got {strict:?}"
    );
    // A 0.5 floor keeps it.
    let loose = association_rules(&frequent, RuleMetric::Confidence, 0.5)?;
    assert!(
        loose
            .iter()
            .any(|r| r.antecedent() == [0] && r.consequent() == [1]),
        "0.5 floor must keep the 2/3-confidence rule"
    );
    Ok(())
}

/// A frequent triple yields every non-empty antecedent/consequent partition
/// (2^3 − 2 = 6 rules), all with multi-item sides represented.
#[test]
fn triple_generates_all_partitions() -> Result<(), AssociationError> {
    // All three items co-occur in every transaction -> {0,1,2} frequent at 1.0.
    let t = vec![vec![true, true, true], vec![true, true, true]];
    let frequent = apriori(&t, 0.5)?;
    let rules = association_rules(&frequent, RuleMetric::Support, 0.0)?;
    // Partitions of {0,1,2}: 3 singleton-antecedent + 3 pair-antecedent = 6.
    let from_triple: Vec<&AssociationRule> = rules
        .iter()
        .filter(|r| r.antecedent().len() + r.consequent().len() == 3)
        .collect();
    assert_eq!(
        from_triple.len(),
        6,
        "6 partitions of a triple, got {}",
        from_triple.len()
    );
    // One of them has a two-item antecedent.
    assert!(
        from_triple.iter().any(|r| r.antecedent().len() == 2),
        "a pair-antecedent rule must be present"
    );
    Ok(())
}

/// Mining is deterministic: the same input yields byte-identical itemset and rule
/// orderings across repeated calls.
#[test]
fn mining_is_deterministic() -> Result<(), AssociationError> {
    let t = vec![
        vec![true, true, false],
        vec![true, false, true],
        vec![true, true, true],
        vec![false, true, true],
    ];
    let a = apriori(&t, 0.25)?;
    let b = apriori(&t, 0.25)?;
    assert_eq!(a, b, "frequent itemsets must be deterministic");
    let ra = association_rules(&a, RuleMetric::Lift, 0.0)?;
    let rb = association_rules(&b, RuleMetric::Lift, 0.0)?;
    assert_eq!(ra, rb, "rules must be deterministic");
    Ok(())
}

/// An empty transaction matrix is rejected with [`AssociationError::EmptyInput`].
#[test]
fn empty_matrix_is_rejected() {
    let t: Vec<Vec<bool>> = Vec::new();
    assert_eq!(apriori(&t, 0.5), Err(AssociationError::EmptyInput));
}

/// A matrix with rows but no item columns is rejected with `NoItems`.
#[test]
fn no_items_is_rejected() {
    let t = vec![Vec::new(), Vec::new()];
    assert_eq!(apriori(&t, 0.5), Err(AssociationError::NoItems));
}

/// Rows of differing length are rejected with `RaggedRows`.
#[test]
fn ragged_rows_are_rejected() {
    let t = vec![vec![true, false], vec![true]];
    assert_eq!(apriori(&t, 0.5), Err(AssociationError::RaggedRows));
}

/// A min-support outside `(0, 1]` is rejected with `InvalidSupport`.
#[test]
fn invalid_support_is_rejected() {
    let t = vec![vec![true], vec![false]];
    assert_eq!(apriori(&t, 0.0), Err(AssociationError::InvalidSupport));
    assert_eq!(apriori(&t, 1.5), Err(AssociationError::InvalidSupport));
    assert_eq!(apriori(&t, f64::NAN), Err(AssociationError::InvalidSupport));
}

/// A non-finite metric threshold is rejected with `InvalidThreshold`.
#[test]
fn invalid_threshold_is_rejected() -> Result<(), AssociationError> {
    let t = vec![vec![true, true], vec![true, true]];
    let frequent = apriori(&t, 0.5)?;
    assert_eq!(
        association_rules(&frequent, RuleMetric::Confidence, f64::INFINITY),
        Err(AssociationError::InvalidThreshold)
    );
    Ok(())
}
