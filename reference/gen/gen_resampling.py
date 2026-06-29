"""Golden fixtures for the resampling family.

Produces a seed-fixed median-bootstrap percentile confidence interval from
``scipy.stats.bootstrap`` (the trusted reference for AC-5 Story 5.2 / QA-DIST-084).
The Rust suite recomputes a percentile CI of the bootstrap median over its own
seeded resamples and asserts the bounds fall within Monte-Carlo relative error of
the stored 2.5 / 97.5 reference.

Run directly with the pinned Python stack::

    cd stats-claw/reference && python3 -m gen.gen_resampling

``cargo test`` never invokes this; it reads the committed JSON offline.
"""
import numpy as np
import scipy
from scipy import stats

from ._common import write_fixture

#: scipy version recorded in the fixture provenance, for drift detection.
VER = scipy.__version__

#: Number of bootstrap resamples; large so the percentile bounds are stable to
#: well within the 1e-2 relative Monte-Carlo tolerance the Rust test asserts.
N_RESAMPLES = 20_000

#: Total tail mass for the interval (95% confidence => 2.5 / 97.5 percentiles).
ALPHA = 0.05

#: Seed fixed so the reference is reproducible on regeneration.
SEED = 12345


def gen_median_ci():
    """Write the seeded median-bootstrap percentile-CI fixture ``resamp_median_ci``."""
    rng = np.random.default_rng(SEED)
    # A moderately skewed, fixed dataset; sampled once from the seeded generator
    # then frozen into the fixture so Rust and scipy share identical observations.
    data = rng.gamma(shape=2.0, scale=2.0, size=200)

    res = stats.bootstrap(
        (data,),
        np.median,
        n_resamples=N_RESAMPLES,
        confidence_level=1.0 - ALPHA,
        method="percentile",
        random_state=np.random.default_rng(SEED),
    )

    write_fixture(
        "resamp_median_ci",
        {
            "data": data.tolist(),
            "alpha": ALPHA,
            "n_resamples": N_RESAMPLES,
            "statistic": "median",
            "ci_low": float(res.confidence_interval.low),
            "ci_high": float(res.confidence_interval.high),
        },
        library="scipy.stats.bootstrap",
        version=VER,
        seed=SEED,
    )


def main():
    """Generate every resampling fixture."""
    gen_median_ci()


if __name__ == "__main__":
    main()
