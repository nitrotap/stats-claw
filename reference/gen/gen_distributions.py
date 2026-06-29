"""Golden fixtures for the distribution family.

Run as a module from the ``reference/`` directory so the package-relative import
of :mod:`gen._common` resolves::

    python3 -m gen.gen_distributions

Python runs here ONLY at generation time; ``cargo test`` reads the committed JSON
under ``reference/golden/`` offline and never invokes Python. Each ``gen_*``
function writes one fixture (params, an x-grid spanning the body and both tails,
pdf/cdf, a probability grid, ppf, mean, variance) stamped with its scipy version.
"""
import numpy as np
import scipy
from scipy import stats

from ._common import write_fixture

#: Actual scipy version producing these fixtures, recorded for drift detection.
VER = scipy.__version__

#: Probability grid for ppf / round-trip: spans near-0, central, and near-1.
PS = [1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-6]


def _moment(value):
    """Map an undefined (NaN/inf) moment to JSON ``null`` so the fixture stays
    strict JSON (``serde_json`` rejects a bare ``NaN`` token); finite moments
    pass through as floats."""
    v = float(value)
    return None if not np.isfinite(v) else v


def _continuous(name, dist, params, xs, *, library):
    """Write a continuous-distribution fixture from a frozen scipy ``dist``."""
    write_fixture(
        name,
        {
            "params": params,
            "x": list(xs),
            "pdf": dist.pdf(xs).tolist(),
            "cdf": dist.cdf(xs).tolist(),
            "p": PS,
            "ppf": dist.ppf(PS).tolist(),
            "mean": _moment(dist.mean()),
            "variance": _moment(dist.var()),
        },
        library=library,
        version=VER,
    )


def _discrete(name, dist, params, ks, *, library):
    """Write a discrete-distribution fixture (pmf over integer support ``ks``)."""
    ks = list(ks)
    write_fixture(
        name,
        {
            "params": params,
            "k": ks,
            "pmf": [float(dist.pmf(k)) for k in ks],
            "cdf": [float(dist.cdf(k)) for k in ks],
            "p": PS,
            "ppf": [float(dist.ppf(p)) for p in PS],
            "mean": _moment(dist.mean()),
            "variance": _moment(dist.var()),
        },
        library=library,
        version=VER,
    )


def gen_normal():
    mu, sigma = 1.5, 2.0
    d = stats.norm(loc=mu, scale=sigma)
    xs = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 41)
    _continuous(
        "dist_normal",
        d,
        {"mean": mu, "standard_deviation": sigma},
        xs,
        library="scipy.stats.norm",
    )


def gen_laplace():
    loc, scale = -0.5, 1.5
    d = stats.laplace(loc=loc, scale=scale)
    xs = np.linspace(loc - 8 * scale, loc + 8 * scale, 41)
    _continuous(
        "dist_laplace",
        d,
        {"location": loc, "scale": scale},
        xs,
        library="scipy.stats.laplace",
    )


def gen_cauchy():
    loc, scale = 0.0, 1.0
    d = stats.cauchy(loc=loc, scale=scale)
    # Cauchy has heavy tails; keep the grid moderate so cdf/pdf stay well-resolved.
    xs = np.linspace(loc - 10 * scale, loc + 10 * scale, 41)
    write_fixture(
        "dist_cauchy",
        {
            "params": {"location": loc, "scale": scale},
            "x": list(xs),
            "pdf": d.pdf(xs).tolist(),
            "cdf": d.cdf(xs).tolist(),
            "p": PS,
            "ppf": d.ppf(PS).tolist(),
            "mean": _moment(d.mean()),  # null — undefined
            "variance": _moment(d.var()),  # null — undefined
        },
        library="scipy.stats.cauchy",
        version=VER,
    )


def gen_uniform():
    a, b = -2.0, 3.0
    d = stats.uniform(loc=a, scale=b - a)
    xs = np.linspace(a - 1.0, b + 1.0, 41)
    _continuous(
        "dist_uniform",
        d,
        {"lower_bound": a, "upper_bound": b},
        xs,
        library="scipy.stats.uniform",
    )


def gen_exponential():
    rate = 0.75
    d = stats.expon(scale=1.0 / rate)
    xs = np.linspace(0.0, 12.0 / rate, 41)
    _continuous(
        "dist_exponential",
        d,
        {"rate_parameter": rate},
        xs,
        library="scipy.stats.expon",
    )


def gen_weibull():
    shape, scale = 1.8, 2.5
    d = stats.weibull_min(c=shape, scale=scale)
    xs = np.linspace(0.0, 5.0 * scale, 41)
    _continuous(
        "dist_weibull",
        d,
        {"shape_parameter": shape, "scale_parameter": scale},
        xs,
        library="scipy.stats.weibull_min",
    )


def gen_lognormal():
    mu, sigma = 0.3, 0.6
    d = stats.lognorm(s=sigma, scale=np.exp(mu))
    xs = np.linspace(1e-3, 10.0, 41)
    _continuous(
        "dist_lognormal",
        d,
        {"mean_log_value": mu, "std_log_value": sigma},
        xs,
        library="scipy.stats.lognorm",
    )


def gen_gamma():
    shape, scale = 2.5, 1.5
    d = stats.gamma(shape, scale=scale)
    xs = np.linspace(0.0, shape * scale + 12.0 * (shape ** 0.5) * scale, 41)
    _continuous(
        "dist_gamma",
        d,
        {"shape_parameter": shape, "scale_parameter": scale},
        xs,
        library="scipy.stats.gamma",
    )


def gen_beta():
    a, b = 2.0, 5.0
    d = stats.beta(a, b)
    xs = np.linspace(0.0, 1.0, 41)
    _continuous(
        "dist_beta",
        d,
        {"alpha_parameter": a, "beta_parameter": b},
        xs,
        library="scipy.stats.beta",
    )


def gen_chi_squared():
    df = 5
    d = stats.chi2(df)
    xs = np.linspace(0.0, 30.0, 41)
    _continuous(
        "dist_chi_squared",
        d,
        {"degrees_of_freedom": df},
        xs,
        library="scipy.stats.chi2",
    )


def gen_students_t():
    df = 7
    d = stats.t(df)
    xs = np.linspace(-8.0, 8.0, 41)
    _continuous(
        "dist_students_t",
        d,
        {"degrees_of_freedom": df},
        xs,
        library="scipy.stats.t",
    )


def gen_f():
    d1, d2 = 6, 12
    d = stats.f(d1, d2)
    xs = np.linspace(1e-3, 8.0, 41)
    _continuous(
        "dist_f",
        d,
        {"numerator_df": d1, "denominator_df": d2},
        xs,
        library="scipy.stats.f",
    )


def gen_binomial():
    n, p = 20, 0.35
    d = stats.binom(n, p)
    ks = list(range(0, n + 1))
    _discrete(
        "dist_binomial",
        d,
        {"number_of_trials": n, "success_probability": p},
        ks,
        library="scipy.stats.binom",
    )


def gen_poisson():
    lam = 4.0
    d = stats.poisson(lam)
    ks = list(range(0, 25))
    _discrete(
        "dist_poisson",
        d,
        {"rate_parameter": lam},
        ks,
        library="scipy.stats.poisson",
    )


def main():
    """Regenerate every distribution fixture."""
    gen_normal()
    gen_laplace()
    gen_cauchy()
    gen_uniform()
    gen_exponential()
    gen_weibull()
    gen_lognormal()
    gen_gamma()
    gen_beta()
    gen_chi_squared()
    gen_students_t()
    gen_f()
    gen_binomial()
    gen_poisson()


if __name__ == "__main__":
    main()
