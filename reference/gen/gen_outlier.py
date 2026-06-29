"""Golden fixtures for the outlier / anomaly-detection family.

Run via ``make fixtures`` or ``python3 -m gen.gen_outlier`` from
``stats-claw/reference``.

The fixture records a fixed sample and the two identifiable, reference-pinned
quantities the Rust equivalence suite asserts:

  * the **z-scores** from ``scipy.stats.zscore`` (population std, ``ddof=0``), and
  * the **IQR / Tukey fences** ``Q1 - k*IQR .. Q3 + k*IQR`` whose quartiles come
    from ``numpy.percentile`` with its default ``method='linear'`` interpolation.

Both conventions are reproduced exactly by the Rust detectors, so the agreement
is at machine precision (the z-scores at ~1e-12, the fences exact).

Reference:
  * ``scipy.stats.zscore`` (default ``ddof=0``) for the per-point z-scores.
  * ``numpy.percentile(..., method='linear')`` (the numpy default) for Q1/Q3 and
    hence the Tukey fences.
"""
import numpy as np
import scipy
from scipy.stats import zscore

from ._common import write_fixture

VER = scipy.__version__
NP_VER = np.__version__
SEED = 17
#: A fixed sample: a tight bulk plus one extreme high value and one low value, so
#: both fences and the z-scores have something to flag.
DATA = [10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0, 60.0, -25.0]
#: The Tukey fence multiplier the suite pins (classic 1.5*IQR rule).
K = 1.5


def gen_outlier():
    """Write the outlier golden fixture from scipy zscore + numpy percentile."""
    data = np.array(DATA, dtype=float)
    # scipy.stats.zscore: population std (ddof=0), per-point signed z-scores.
    z = zscore(data)
    # numpy 'linear' (default) quartiles -> Tukey fences.
    q1, q3 = np.percentile(data, [25.0, 75.0], method="linear")
    iqr = q3 - q1
    lower = q1 - K * iqr
    upper = q3 + K * iqr
    write_fixture(
        "outlier_detect",
        {
            "data": DATA,
            "k": K,
            # scipy.stats.zscore (ddof=0) per-point scores.
            "zscores": [float(v) for v in z],
            # numpy 'linear' quartiles and the resulting Tukey fences.
            "q1": float(q1),
            "q3": float(q3),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
        },
        library=f"scipy.stats.zscore + numpy.percentile (numpy {NP_VER})",
        version=VER,
        seed=SEED,
    )


def main():
    """Regenerate every outlier-detection golden fixture."""
    gen_outlier()


if __name__ == "__main__":
    main()
