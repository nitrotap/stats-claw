"""Entry point for regenerating all golden fixtures: ``make fixtures``.

Each family registers its ``gen_*`` module here. Family generators (which import
scipy/numpy/etc.) are added by their respective tracks; this scaffold deliberately
imports nothing heavy so the harness can land before the Python env is set up.
"""


def main():
    """Run every registered family generator (regenerates all golden fixtures)."""
    from . import (
        gen_algorithms,
        gen_association,
        gen_cardinality,
        gen_density,
        gen_distributions,
        gen_feature_selection,
        gen_optimizers,
        gen_outlier,
        gen_regression,
        gen_resampling,
        gen_tests,
    )

    gen_distributions.main()
    gen_optimizers.main()
    gen_resampling.main()
    gen_algorithms.main()
    gen_regression.main()
    gen_density.main()
    gen_outlier.main()
    gen_feature_selection.main()
    gen_association.main()
    gen_cardinality.main()
    gen_tests.main()


if __name__ == "__main__":
    main()
