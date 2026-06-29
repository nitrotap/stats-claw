//! Association-rule generation from frequent itemsets: partition enumeration,
//! metric scoring, the metric-threshold filter, and the deterministic ordering.
//!
//! Split out of `mod.rs` to keep that file inside the project's 500-line `style.rs`
//! cap. The public entry point [`association_rules`] reproduces
//! `mlxtend.frequent_patterns.association_rules(frequent, metric, min_threshold)`:
//! every frequent itemset of size `≥ 2` is split into all non-empty antecedent /
//! consequent partitions, each scored with `support` / `confidence` / `lift` /
//! `leverage` / `conviction`, and emitted only when the chosen metric clears the
//! threshold. As a child module of `association`, this file can read the private
//! fields of the [`FrequentItemset`] and [`AssociationRule`] types declared in the
//! parent.

use std::collections::BTreeMap;

use super::{AssociationError, AssociationRule, FrequentItemset, RuleMetric};

/// Generates association rules from frequent itemsets, filtered by a metric
/// threshold.
///
/// Every frequent itemset of size `≥ 2` is split into all `2^m − 2` non-empty
/// antecedent / consequent partitions; each partition's `support`, `confidence`,
/// `lift`, `leverage`, and `conviction` are computed from the frequent-itemset
/// supports, and the rule is emitted only when the chosen `metric` is
/// `≥ min_threshold`. This mirrors
/// `mlxtend.frequent_patterns.association_rules(frequent, metric, min_threshold)`.
/// Rules are returned sorted by `(antecedent, consequent)` for determinism.
///
/// # Arguments
///
/// * `frequent` — frequent itemsets with supports, as returned by
///   [`apriori`](super::apriori); the single-item supports it contains are required
///   to score `lift` / `conviction`.
/// * `metric` — which metric the `min_threshold` filter applies to.
/// * `min_threshold` — the minimum value of `metric` a rule must reach; must be
///   finite.
///
/// # Returns
///
/// Every qualifying association rule with its metrics, deterministically ordered.
///
/// # Errors
///
/// Returns [`AssociationError::InvalidThreshold`] if `min_threshold` is not finite.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::association::{apriori, association_rules, RuleMetric};
///
/// let t = vec![
///     vec![true, true],
///     vec![true, true],
///     vec![true, false],
///     vec![false, true],
/// ];
/// let frequent = apriori(&t, 0.25)?;
/// let rules = association_rules(&frequent, RuleMetric::Confidence, 0.5)?;
/// // {0} => {1} has confidence 2/3 ≥ 0.5, so it is emitted.
/// assert!(rules.iter().any(|r| r.antecedent() == [0] && r.consequent() == [1]));
/// # Ok::<(), stats_claw::algorithms::association::AssociationError>(())
/// ```
pub fn association_rules(
    frequent: &[FrequentItemset],
    metric: RuleMetric,
    min_threshold: f64,
) -> Result<Vec<AssociationRule>, AssociationError> {
    if !min_threshold.is_finite() {
        return Err(AssociationError::InvalidThreshold);
    }
    // Index supports by itemset for O(log n) lookup of antecedent / consequent.
    let support_index: BTreeMap<Vec<usize>, f64> = frequent
        .iter()
        .map(|s| (s.items.clone(), s.support))
        .collect();

    let mut rules: Vec<AssociationRule> = Vec::new();
    for set in frequent {
        if set.items.len() < 2 {
            continue;
        }
        for antecedent in non_empty_proper_subsets(&set.items) {
            let consequent: Vec<usize> = set
                .items
                .iter()
                .filter(|i| !antecedent.contains(i))
                .copied()
                .collect();
            // Both sides are looked up from the frequent index; an itemset that
            // is frequent has all its subsets frequent, so the lookups succeed.
            let (Some(&sup_a), Some(&sup_c)) = (
                support_index.get(&antecedent),
                support_index.get(&consequent),
            ) else {
                continue;
            };
            let rule = score_rule(antecedent, consequent, set.support, sup_a, sup_c);
            if metric_value(&rule, metric) >= min_threshold {
                rules.push(rule);
            }
        }
    }
    sort_rules(&mut rules);
    Ok(rules)
}

/// Scores a single antecedent ⇒ consequent partition into an [`AssociationRule`].
///
/// Applies the standard metric definitions from the three supports: the joint
/// support of the full itemset, the antecedent support, and the consequent support.
///
/// # Arguments
///
/// * `antecedent` — the ascending antecedent items.
/// * `consequent` — the ascending consequent items.
/// * `sup_union` — support of `antecedent ∪ consequent` (the parent itemset).
/// * `sup_a` — support of the antecedent alone.
/// * `sup_c` — support of the consequent alone.
///
/// # Returns
///
/// The fully-scored rule.
fn score_rule(
    antecedent: Vec<usize>,
    consequent: Vec<usize>,
    sup_union: f64,
    sup_a: f64,
    sup_c: f64,
) -> AssociationRule {
    let confidence = sup_union / sup_a;
    let lift = confidence / sup_c;
    // `mul_add` computes `sup_union − sup_a·sup_c` as one fused op (lint-clean and
    // at least as accurate); the supports are exact small rationals here, so the
    // result is bit-identical to mlxtend's plain `support − A·C`.
    let leverage = sup_a.mul_add(-sup_c, sup_union);
    // mlxtend reports conviction as +inf when confidence == 1 (denominator 0).
    let conviction = if (1.0 - confidence).abs() < f64::EPSILON {
        f64::INFINITY
    } else {
        (1.0 - sup_c) / (1.0 - confidence)
    };
    AssociationRule {
        antecedent,
        consequent,
        support: sup_union,
        confidence,
        lift,
        leverage,
        conviction,
    }
}

/// Returns the value of the selected `metric` for a scored rule.
///
/// The dispatch behind the [`association_rules`] threshold filter.
///
/// # Arguments
///
/// * `rule` — the scored rule.
/// * `metric` — which metric to read.
///
/// # Returns
///
/// The rule's value for that metric.
const fn metric_value(rule: &AssociationRule, metric: RuleMetric) -> f64 {
    match metric {
        RuleMetric::Confidence => rule.confidence,
        RuleMetric::Lift => rule.lift,
        RuleMetric::Support => rule.support,
        RuleMetric::Leverage => rule.leverage,
        RuleMetric::Conviction => rule.conviction,
    }
}

/// Enumerates the non-empty proper subsets of an ascending itemset.
///
/// Used to form every antecedent of an itemset's rules; each returned subset keeps
/// the items in ascending order, and the full set itself is excluded so the
/// consequent is always non-empty.
///
/// # Arguments
///
/// * `items` — the ascending itemset to split (length `m`, with `m ≤` machine word
///   bits, which always holds for realistic baskets).
///
/// # Returns
///
/// Each non-empty proper subset once, in ascending item order within each subset.
fn non_empty_proper_subsets(items: &[usize]) -> Vec<Vec<usize>> {
    let m = items.len();
    let full = 1u64 << m;
    let mut subsets: Vec<Vec<usize>> = Vec::new();
    // Masks 1..(2^m - 1) skip the empty set (0) and the full set (2^m - 1).
    for mask in 1..full - 1 {
        let subset: Vec<usize> = items
            .iter()
            .enumerate()
            .filter_map(|(bit, &item)| (mask >> bit & 1 == 1).then_some(item))
            .collect();
        subsets.push(subset);
    }
    subsets
}

/// Sorts rules by ascending antecedent, then ascending consequent.
///
/// Gives a deterministic, reproducible ordering so the equivalence suite can sort
/// both sides to the same canonical order before comparing.
///
/// # Arguments
///
/// * `rules` — the rules to order in place.
fn sort_rules(rules: &mut [AssociationRule]) {
    rules.sort_by(|a, b| {
        a.antecedent
            .cmp(&b.antecedent)
            .then(a.consequent.cmp(&b.consequent))
    });
}
