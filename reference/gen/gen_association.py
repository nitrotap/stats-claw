"""Golden fixtures for the association-analysis family.

Run via ``make fixtures`` or ``python3 -m gen.gen_association`` from
``stats-claw/reference``.

The fixture records a fixed one-hot boolean transaction matrix and the
identifiable, reference-pinned quantities the Rust equivalence suite asserts:

  * the **frequent itemsets** with their support from
    ``mlxtend.frequent_patterns.apriori(df, min_support)``, and
  * the **association rules** with support / confidence / lift / leverage /
    conviction from
    ``mlxtend.frequent_patterns.association_rules(frequent, metric="confidence",
    min_threshold=...)``.

mlxtend consumes a pandas DataFrame of booleans (one column per item, one row per
transaction) and reports support as ``count / n_transactions``; the Rust miner
takes the same boolean matrix and computes the identical integer-count ratios, so
supports, confidences, and lifts agree to machine precision (~1e-12). Itemsets and
rules are emitted as sorted lists of integer item (column) indices so the Rust
suite can sort both sides to a canonical order before comparing.

Settings (documented so the Rust gate matches mlxtend's semantics exactly):
  * ``MIN_SUPPORT`` — the apriori frequent-itemset support floor.
  * ``RULE_METRIC`` / ``MIN_THRESHOLD`` — the rule-generation filter; mlxtend emits
    a rule only when the chosen metric is ``>= min_threshold``.

Reference:
  * ``mlxtend.frequent_patterns.apriori`` for the frequent itemsets + supports.
  * ``mlxtend.frequent_patterns.association_rules`` (metric="confidence") for the
    rules + their metrics.
"""
import mlxtend
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from ._common import write_fixture

VER = mlxtend.__version__
PD_VER = pd.__version__
#: Apriori frequent-itemset support floor.
MIN_SUPPORT = 0.3
#: Rule-generation metric and its minimum threshold.
RULE_METRIC = "confidence"
MIN_THRESHOLD = 0.5


def _build_transactions():
    """Build a fixed 8-transaction, 5-item one-hot boolean basket matrix.

    The items co-occur with a deliberate structure (items 0 and 1 travel together,
    item 2 trails them, items 3/4 are sparser) so apriori finds frequent singles,
    pairs, and at least one triple, and the rules span a range of confidences and
    lifts. Returned as a list-of-rows of 0/1 ints for the fixture and as the boolean
    DataFrame mlxtend consumes.
    """
    rows = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ]
    matrix = np.array(rows, dtype=bool)
    # mlxtend keys itemsets by DataFrame column position; integer columns keep the
    # mapping identity so the fixture's indices match the Rust item indices.
    df = pd.DataFrame(matrix, columns=list(range(matrix.shape[1])))
    return rows, df


def _itemset_to_sorted_list(itemset):
    """Convert an mlxtend frozenset of column indices to a sorted int list."""
    return sorted(int(i) for i in itemset)


def gen_association():
    """Write the association golden fixture from mlxtend references."""
    rows, df = _build_transactions()

    frequent = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
    itemsets = [
        {
            "items": _itemset_to_sorted_list(row.itemsets),
            "support": float(row.support),
        }
        for row in frequent.itertuples()
    ]
    # Canonical order: by size, then by item indices (matches the Rust ordering).
    itemsets.sort(key=lambda d: (len(d["items"]), d["items"]))

    rules_df = association_rules(
        frequent, metric=RULE_METRIC, min_threshold=MIN_THRESHOLD
    )
    rules = [
        {
            "antecedent": _itemset_to_sorted_list(row.antecedents),
            "consequent": _itemset_to_sorted_list(row.consequents),
            "support": float(row.support),
            "confidence": float(row.confidence),
            "lift": float(row.lift),
            "leverage": float(row.leverage),
            "conviction": _finite_or_none(row.conviction),
        }
        for row in rules_df.itertuples()
    ]
    # Canonical order: by antecedent, then consequent (matches the Rust ordering).
    rules.sort(key=lambda d: (d["antecedent"], d["consequent"]))

    write_fixture(
        "association",
        {
            # The one-hot transaction matrix as a list-of-rows of 0/1 ints.
            "transactions": [[int(v) for v in row] for row in rows],
            "min_support": MIN_SUPPORT,
            "rule_metric": RULE_METRIC,
            "min_threshold": MIN_THRESHOLD,
            "itemsets": itemsets,
            "rules": rules,
        },
        library=f"mlxtend.frequent_patterns.apriori + association_rules (pandas {PD_VER})",
        version=VER,
        seed=None,
    )


def _finite_or_none(value):
    """Return a finite conviction as float, or ``None`` for mlxtend's ``inf``.

    mlxtend reports conviction as ``+inf`` for a perfectly-confident rule (the Rust
    side does too); JSON has no infinity literal, so the fixture stores ``null`` and
    the Rust loader treats a missing/`null` conviction as the infinite case.
    """
    return None if not np.isfinite(value) else float(value)


def main():
    """Regenerate every association golden fixture."""
    gen_association()


if __name__ == "__main__":
    main()
