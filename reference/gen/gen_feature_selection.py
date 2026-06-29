"""Golden fixtures for the feature-selection family.

Run via ``make fixtures`` or ``python3 -m gen.gen_feature_selection`` from
``stats-claw/reference``.

The fixture records a fixed labelled feature matrix and the identifiable,
reference-pinned quantities the Rust equivalence suite asserts:

  * the per-feature **ANOVA F-scores and p-values** from
    ``sklearn.feature_selection.f_classif`` (each feature's one-way ANOVA across
    the class groups), and
  * the per-feature **population variances** from
    ``sklearn.feature_selection.VarianceThreshold`` (``ddof=0``).

The Rust ANOVA F path reuses the framework's one-way ANOVA, which is the same
computation ``f_classif`` performs, so the F-scores agree to ~1e-9; the p-values
route through the framework F distribution (regularized incomplete beta), whose
deep-tail precision is the asymptotic 1e-6 band documented for every F p-value in
the build-plan tolerance table. The population variances use the same ``ddof=0``
divisor as ``VarianceThreshold`` and agree exactly.

Reference:
  * ``sklearn.feature_selection.f_classif`` for the per-feature F-scores/p-values.
  * ``sklearn.feature_selection.VarianceThreshold`` (``.variances_``, ``ddof=0``)
    for the per-feature population variances.
"""
import numpy as np
import sklearn
from sklearn.feature_selection import VarianceThreshold, f_classif

from ._common import write_fixture

VER = sklearn.__version__
NP_VER = np.__version__
SEED = 7
#: Number of features in the fixed design matrix.
N_FEATURES = 5


def _build_dataset():
    """Build a fixed three-class, five-feature labelled matrix.

    Each class is a Gaussian blob with a class-dependent mean on the informative
    features, so the ANOVA F-scores span a useful range (some features separate
    the classes strongly, others barely). The RandomState seed pins the matrix.
    """
    rng = np.random.RandomState(SEED)
    blocks = [
        rng.normal(loc=0.0, scale=1.0, size=(10, N_FEATURES)),
        rng.normal(loc=1.2, scale=1.0, size=(12, N_FEATURES)),
        rng.normal(loc=-0.5, scale=1.3, size=(8, N_FEATURES)),
    ]
    x = np.vstack(blocks)
    y = np.array([0] * 10 + [1] * 12 + [2] * 8)
    return x, y


def gen_feature_selection():
    """Write the feature-selection golden fixture from sklearn references."""
    x, y = _build_dataset()
    # sklearn.feature_selection.f_classif: per-feature one-way ANOVA F + p-value.
    f_scores, p_values = f_classif(x, y)
    # VarianceThreshold.variances_: per-feature population variance (ddof=0).
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(x)
    write_fixture(
        "feature_selection",
        {
            # The labelled design matrix as a list-of-rows (one row per sample).
            "x": [[float(v) for v in row] for row in x],
            "labels": [int(c) for c in y],
            # f_classif outputs, per feature in column order.
            "f_scores": [float(v) for v in f_scores],
            "p_values": [float(v) for v in p_values],
            # VarianceThreshold population variances, per feature in column order.
            "variances": [float(v) for v in vt.variances_],
        },
        library=f"sklearn.feature_selection.f_classif + VarianceThreshold (numpy {NP_VER})",
        version=VER,
        seed=SEED,
    )


def main():
    """Regenerate every feature-selection golden fixture."""
    gen_feature_selection()


if __name__ == "__main__":
    main()
