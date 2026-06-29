"""Golden fixtures for the density-estimation family (Gaussian KDE).

Run via ``make fixtures`` or ``python3 -m gen.gen_density`` from
``stats-claw/reference``.

The fixture records a fixed sample, scipy's Scott covariance factor and resulting
bandwidth, and the density evaluated on a query grid, so the Rust equivalence
suite can assert agreement on every reported quantity.

Reference:
  * ``scipy.stats.gaussian_kde`` with the default ``bw_method='scott'``. In one
    dimension Scott's factor is ``n**(-1/5)`` and the kernel variance is
    ``factor**2 * var(data, ddof=1)`` — exactly what the Rust ``gaussian_kde``
    reproduces, so the agreement is at machine precision.
"""
import numpy as np
import scipy
from scipy.stats import gaussian_kde

from ._common import write_fixture

VER = scipy.__version__
SEED = 11
#: A small fixed sample (two loose clusters) the suite estimates a density over.
DATA = [1.0, 1.5, 2.0, 2.5, 3.0, 6.5, 7.0, 7.5, 8.0, 9.0]
#: Query points spanning both clusters and the sparse gap between them.
GRID = [-1.0, 0.0, 1.5, 2.5, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0]


def gen_kde():
    """Write the Gaussian-KDE golden fixture from scipy's ``gaussian_kde``."""
    data = np.array(DATA, dtype=float)
    kde = gaussian_kde(data, bw_method="scott")
    grid = np.array(GRID, dtype=float)
    write_fixture(
        "density_kde",
        {
            "data": DATA,
            "grid": GRID,
            # scipy's covariance factor (Scott) and the kernel variance/bandwidth.
            "factor": float(kde.factor),
            "variance": float(kde.covariance.ravel()[0]),
            "bandwidth": float(np.sqrt(kde.covariance.ravel()[0])),
            "density": [float(v) for v in kde.evaluate(grid)],
        },
        library="scipy.stats.gaussian_kde",
        version=VER,
        seed=SEED,
    )


def main():
    """Regenerate every density-estimation golden fixture."""
    gen_kde()


if __name__ == "__main__":
    main()
