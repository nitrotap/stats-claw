//! Market-basket association-rule mining: Apriori frequent-itemset generation and
//! the support / confidence / lift (plus leverage / conviction) rule metrics.
//!
//! Transactions are given as a one-hot boolean matrix `transactions`, one inner
//! `Vec<bool>` per **transaction** (row), with `transactions[t][i] == true` meaning
//! item `i` is present in transaction `t`; every row must share the same item
//! count (the number of columns). This is exactly the boolean-DataFrame shape
//! `mlxtend.frequent_patterns.apriori` consumes, so the numerics reproduce its
//! conventions the equivalence suite checks:
//!
//! * **Apriori** enumerates every itemset whose **support** — the fraction of
//!   transactions containing all its items — is `≥ min_support`, growing
//!   `k`-itemsets from frequent `(k − 1)`-itemsets via the downward-closure
//!   pruning that names the algorithm, matching `apriori(df, min_support)`;
//! * **association rules** split each frequent itemset of size `≥ 2` into every
//!   non-empty antecedent / consequent partition and report `support`,
//!   `confidence`, `lift`, `leverage`, and `conviction`, matching
//!   `association_rules(frequent, metric, min_threshold)`.
//!
//! Supports, confidences, and lifts are exact rational ratios of integer
//! transaction counts, so the results agree with mlxtend to machine precision.
//! Output is deterministic: frequent itemsets sort by ascending size then by
//! ascending item indices, and rules sort by `(antecedent, consequent)` lexically.
//!
//! This base block backs the `AssociationAnalysis` computational method.

mod rules;
mod support;

pub use rules::association_rules;

use std::collections::BTreeSet;
use support::count_to_f64;

/// Errors that prevent association mining over a transaction matrix.
///
/// Returned (never panicked) so callers stay clear of the crate's `unwrap`/`panic`
/// lint gate and can surface a clean diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssociationError {
    /// The transaction matrix had no rows, so no support can be measured.
    EmptyInput,
    /// The matrix had rows but no item columns, so there is nothing to mine.
    NoItems,
    /// The rows had differing item counts (a ragged matrix), so columns are
    /// undefined.
    RaggedRows,
    /// The minimum support was not in the open-closed range `(0, 1]`, so the
    /// frequent-set threshold would be ill-defined.
    InvalidSupport,
    /// The minimum metric threshold was not finite, so the rule filter would be
    /// ill-defined.
    InvalidThreshold,
}

impl std::fmt::Display for AssociationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "transaction matrix has no rows"),
            Self::NoItems => write!(f, "transaction matrix has no item columns"),
            Self::RaggedRows => write!(f, "transaction matrix rows have differing lengths"),
            Self::InvalidSupport => write!(f, "minimum support must lie in (0, 1]"),
            Self::InvalidThreshold => write!(f, "minimum metric threshold must be finite"),
        }
    }
}

impl std::error::Error for AssociationError {}

/// A frequent itemset and its support over the transaction matrix.
///
/// `items` is the set of item (column) indices in ascending order; `support` is the
/// fraction of transactions that contain every item in the set, in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequentItemset {
    /// The item (column) indices forming the set, in ascending order.
    items: Vec<usize>,
    /// The support: fraction of transactions containing all of `items`, in `[0, 1]`.
    support: f64,
}

impl FrequentItemset {
    /// The item indices forming this set, in ascending order.
    #[must_use]
    pub fn items(&self) -> &[usize] {
        &self.items
    }

    /// The support of this set (fraction of transactions containing every item).
    #[must_use]
    pub const fn support(&self) -> f64 {
        self.support
    }
}

/// An association rule `antecedent ⇒ consequent` with its standard metrics.
///
/// Both sides are disjoint, non-empty sets of item indices in ascending order. The
/// metrics follow the standard market-basket definitions (and mlxtend's):
/// `support` is the joint support of `antecedent ∪ consequent`; `confidence` is
/// `support(A ∪ C) / support(A)`; `lift` is `confidence / support(C)`; `leverage`
/// is `support(A ∪ C) − support(A)·support(C)`; `conviction` is
/// `(1 − support(C)) / (1 − confidence)` (`+∞` for a perfectly-confident rule).
#[derive(Debug, Clone, PartialEq)]
pub struct AssociationRule {
    /// The antecedent ("if") item indices, ascending and disjoint from `consequent`.
    antecedent: Vec<usize>,
    /// The consequent ("then") item indices, ascending and disjoint from `antecedent`.
    consequent: Vec<usize>,
    /// Joint support of `antecedent ∪ consequent`, in `[0, 1]`.
    support: f64,
    /// `support(A ∪ C) / support(A)`, in `[0, 1]`.
    confidence: f64,
    /// `confidence / support(C)`; `> 1` indicates positive association.
    lift: f64,
    /// `support(A ∪ C) − support(A)·support(C)`, in `[-0.25, 0.25]`.
    leverage: f64,
    /// `(1 − support(C)) / (1 − confidence)`; `+∞` when `confidence == 1`.
    conviction: f64,
}

impl AssociationRule {
    /// The antecedent item indices, in ascending order.
    #[must_use]
    pub fn antecedent(&self) -> &[usize] {
        &self.antecedent
    }

    /// The consequent item indices, in ascending order.
    #[must_use]
    pub fn consequent(&self) -> &[usize] {
        &self.consequent
    }

    /// The joint support of the rule's item union, in `[0, 1]`.
    #[must_use]
    pub const fn support(&self) -> f64 {
        self.support
    }

    /// The confidence `support(A ∪ C) / support(A)`, in `[0, 1]`.
    #[must_use]
    pub const fn confidence(&self) -> f64 {
        self.confidence
    }

    /// The lift `confidence / support(C)`.
    #[must_use]
    pub const fn lift(&self) -> f64 {
        self.lift
    }

    /// The leverage `support(A ∪ C) − support(A)·support(C)`.
    #[must_use]
    pub const fn leverage(&self) -> f64 {
        self.leverage
    }

    /// The conviction `(1 − support(C)) / (1 − confidence)` (`+∞` if `confidence == 1`).
    #[must_use]
    pub const fn conviction(&self) -> f64 {
        self.conviction
    }
}

/// The metric `association_rules` filters generated rules by.
///
/// Mirrors mlxtend's `metric` argument: a rule is emitted only when the chosen
/// metric is `≥ min_threshold`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleMetric {
    /// Filter on the rule's confidence.
    Confidence,
    /// Filter on the rule's lift.
    Lift,
    /// Filter on the rule's joint support.
    Support,
    /// Filter on the rule's leverage.
    Leverage,
    /// Filter on the rule's conviction.
    Conviction,
}

/// Validates a transaction matrix is non-empty, rectangular, returning the
/// `(n_transactions, n_items)` shape.
///
/// Shared front-door check so both the miner and the rule generator reject the same
/// degenerate inputs before any counting runs.
///
/// # Arguments
///
/// * `transactions` — the one-hot boolean matrix, one inner slice per transaction.
///
/// # Returns
///
/// `(n_transactions, n_items)` on success.
///
/// # Errors
///
/// Returns [`AssociationError::EmptyInput`], [`AssociationError::NoItems`], or
/// [`AssociationError::RaggedRows`].
fn validate_matrix(transactions: &[Vec<bool>]) -> Result<(usize, usize), AssociationError> {
    let n_transactions = transactions.len();
    if n_transactions == 0 {
        return Err(AssociationError::EmptyInput);
    }
    let n_items = transactions.first().map_or(0, Vec::len);
    if n_items == 0 {
        return Err(AssociationError::NoItems);
    }
    if transactions.iter().any(|row| row.len() != n_items) {
        return Err(AssociationError::RaggedRows);
    }
    Ok((n_transactions, n_items))
}

/// Counts how many transactions contain every item in `items`.
///
/// A transaction contains the itemset when each listed column is `true`; `items`
/// must hold valid column indices (the miner only ever builds them from `0..n_items`).
///
/// # Arguments
///
/// * `transactions` — the validated boolean matrix.
/// * `items` — the item (column) indices to test for joint presence.
///
/// # Returns
///
/// The number of transactions containing all of `items`.
fn cover_count(transactions: &[Vec<bool>], items: &[usize]) -> usize {
    transactions
        .iter()
        .filter(|row| items.iter().all(|&i| row.get(i).copied().unwrap_or(false)))
        .count()
}

/// Enumerates every frequent itemset (support `≥ min_support`) via Apriori.
///
/// Starts from the frequent single items, then repeatedly joins frequent
/// `(k − 1)`-itemsets that share their first `k − 2` items into candidate
/// `k`-itemsets, pruning any candidate with an infrequent `(k − 1)`-subset
/// (the downward-closure property), and keeps those candidates whose support
/// clears the threshold. This reproduces `mlxtend.frequent_patterns.apriori`'s
/// output set. Itemsets are returned sorted by ascending size, then by ascending
/// item indices, for deterministic downstream rule generation.
///
/// # Arguments
///
/// * `transactions` — the one-hot boolean matrix, one inner `Vec<bool>` per
///   transaction; must be non-empty and rectangular.
/// * `min_support` — the minimum support an itemset must reach to be frequent; must
///   lie in `(0, 1]`.
///
/// # Returns
///
/// Every frequent itemset with its support, deterministically ordered.
///
/// # Errors
///
/// Returns [`AssociationError::EmptyInput`], [`AssociationError::NoItems`],
/// [`AssociationError::RaggedRows`], or [`AssociationError::InvalidSupport`].
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::association::apriori;
///
/// // Items 0 and 1 co-occur in 3 of 4 transactions (support 0.75).
/// let t = vec![
///     vec![true, true, false],
///     vec![true, true, true],
///     vec![true, true, false],
///     vec![false, false, true],
/// ];
/// let frequent = apriori(&t, 0.5)?;
/// // The pair {0, 1} is frequent at support 0.75.
/// let pair = frequent.iter().find(|s| s.items() == [0, 1]).expect("pair present");
/// assert!((pair.support() - 0.75).abs() < 1e-12, "support was {}", pair.support());
/// # Ok::<(), stats_claw::algorithms::association::AssociationError>(())
/// ```
pub fn apriori(
    transactions: &[Vec<bool>],
    min_support: f64,
) -> Result<Vec<FrequentItemset>, AssociationError> {
    let (n_transactions, n_items) = validate_matrix(transactions)?;
    if !(min_support > 0.0 && min_support <= 1.0) {
        return Err(AssociationError::InvalidSupport);
    }
    let n = count_to_f64(n_transactions);
    // The count threshold: support >= min_support  <=>  count >= min_support * n.
    // Comparing counts (integers) avoids float drift in the frequency test.
    let support_of = |count: usize| count_to_f64(count) / n;

    let mut all: Vec<FrequentItemset> = Vec::new();

    // Level 1: frequent single items, in ascending index order.
    let mut current: Vec<Vec<usize>> = Vec::new();
    for item in 0..n_items {
        let count = cover_count(transactions, &[item]);
        let support = support_of(count);
        if support >= min_support {
            current.push(vec![item]);
            all.push(FrequentItemset {
                items: vec![item],
                support,
            });
        }
    }

    // Levels 2..: join + prune until no candidates survive.
    while !current.is_empty() {
        let candidates = join_and_prune(&current);
        let mut next: Vec<Vec<usize>> = Vec::new();
        for cand in candidates {
            let count = cover_count(transactions, &cand);
            let support = support_of(count);
            if support >= min_support {
                next.push(cand.clone());
                all.push(FrequentItemset {
                    items: cand,
                    support,
                });
            }
        }
        current = next;
    }

    sort_itemsets(&mut all);
    Ok(all)
}

/// Joins frequent `(k − 1)`-itemsets sharing their first `k − 2` items into sorted
/// `k`-candidates, dropping any whose `(k − 1)`-subsets are not all frequent.
///
/// The classic Apriori-gen step: two frequent sets `[a₀..a_{k-2}, x]` and
/// `[a₀..a_{k-2}, y]` with `x < y` join to `[a₀..a_{k-2}, x, y]`; the candidate
/// survives only when every one of its `k − 1`-element subsets is itself frequent
/// (downward closure), which prunes most candidates before any support count.
///
/// # Arguments
///
/// * `frequent` — the frequent `(k − 1)`-itemsets, each ascending; assumed sorted
///   lexicographically by [`apriori`]'s level ordering.
///
/// # Returns
///
/// The surviving `k`-candidate itemsets, each ascending, de-duplicated.
fn join_and_prune(frequent: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let frequent_set: BTreeSet<&Vec<usize>> = frequent.iter().collect();
    let mut candidates: BTreeSet<Vec<usize>> = BTreeSet::new();
    for (ai, a) in frequent.iter().enumerate() {
        for b in frequent.iter().skip(ai + 1) {
            // Join only when the prefixes (all but the last item) match and the last
            // items differ in ascending order.
            let k = a.len();
            if a.get(..k - 1) != b.get(..k - 1) {
                continue;
            }
            let (Some(&last_a), Some(&last_b)) = (a.last(), b.last()) else {
                continue;
            };
            if last_a >= last_b {
                continue;
            }
            let mut candidate = a.clone();
            candidate.push(last_b);
            if all_subsets_frequent(&candidate, &frequent_set) {
                candidates.insert(candidate);
            }
        }
    }
    candidates.into_iter().collect()
}

/// Returns whether every `(k − 1)`-element subset of `candidate` is frequent.
///
/// The downward-closure prune: a `k`-itemset can only be frequent if all of its
/// immediate subsets are, so any candidate with a missing subset is discarded
/// before its support is counted.
///
/// # Arguments
///
/// * `candidate` — the ascending `k`-itemset to test.
/// * `frequent` — the set of frequent `(k − 1)`-itemsets, by reference.
///
/// # Returns
///
/// `true` iff dropping each single item in turn leaves a frequent subset.
fn all_subsets_frequent(candidate: &[usize], frequent: &BTreeSet<&Vec<usize>>) -> bool {
    (0..candidate.len()).all(|drop| {
        let subset: Vec<usize> = candidate
            .iter()
            .enumerate()
            .filter_map(|(idx, &item)| (idx != drop).then_some(item))
            .collect();
        frequent.contains(&subset)
    })
}

/// Sorts frequent itemsets by ascending size, then by ascending item indices.
///
/// Gives a deterministic, reproducible ordering so the equivalence suite can sort
/// both sides to the same canonical order before comparing.
///
/// # Arguments
///
/// * `itemsets` — the itemsets to order in place.
fn sort_itemsets(itemsets: &mut [FrequentItemset]) {
    itemsets.sort_by(|a, b| {
        a.items
            .len()
            .cmp(&b.items.len())
            .then(a.items.cmp(&b.items))
    });
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
