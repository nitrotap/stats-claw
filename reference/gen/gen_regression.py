"""Golden fixtures for the linear-regression family (OLS + ridge).

Run via ``make fixtures`` or ``python3 -m gen.gen_regression`` from
``stats-claw/reference``.

A single fixed-seed multivariate dataset is shared by both estimators. Each
fixture records the design matrix, the target, the fitted intercept and
coefficients, the in-sample R², and the predictions on a small held-out grid, so
the Rust equivalence suite can assert agreement on every reported quantity.

References:
  * ``sklearn.linear_model.LinearRegression`` — ordinary least squares.
  * ``sklearn.linear_model.Ridge`` (default dense ``cholesky`` solver) — the L2
    penalised normal equations the Rust ``ridge`` reproduces exactly:
    minimise ``‖y − Xβ − β₀‖² + α‖β‖²`` with an unpenalised intercept.
"""
import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge

from ._common import write_fixture

VER = sklearn.__version__
SEED = 7
N_SAMPLES = 60
N_FEATURES = 3
NOISE = 8.0
#: Ridge penalty used for the ridge fixture (matches the Rust test argument).
ALPHA = 2.5
#: A few held-out rows the suite scores predictions against.
GRID = [
    [0.5, -1.0, 2.0],
    [-2.0, 0.3, 1.5],
    [1.0, 1.0, -1.0],
]


def _dataset():
    """Return the shared (X, y) regression dataset as numpy arrays."""
    x, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=NOISE,
        random_state=SEED,
    )
    return x, y


def gen_ols():
    """Write the ordinary-least-squares golden fixture."""
    x, y = _dataset()
    model = LinearRegression(fit_intercept=True).fit(x, y)
    write_fixture(
        "regression_ols",
        {
            "data": x.tolist(),
            "target": y.tolist(),
            "intercept": float(model.intercept_),
            "coefficients": [float(c) for c in model.coef_],
            "r_squared": float(model.score(x, y)),
            "grid": GRID,
            "grid_predictions": [float(p) for p in model.predict(np.array(GRID))],
        },
        library="sklearn.linear_model.LinearRegression",
        version=VER,
        seed=SEED,
    )


def gen_ridge():
    """Write the ridge-regression golden fixture (alpha = ALPHA)."""
    x, y = _dataset()
    model = Ridge(alpha=ALPHA, fit_intercept=True).fit(x, y)
    write_fixture(
        "regression_ridge",
        {
            "data": x.tolist(),
            "target": y.tolist(),
            "alpha": ALPHA,
            "intercept": float(model.intercept_),
            "coefficients": [float(c) for c in model.coef_],
            "r_squared": float(model.score(x, y)),
            "grid": GRID,
            "grid_predictions": [float(p) for p in model.predict(np.array(GRID))],
        },
        library="sklearn.linear_model.Ridge",
        version=VER,
        seed=SEED,
    )


def main():
    """Regenerate every linear-regression golden fixture."""
    gen_ols()
    gen_ridge()


if __name__ == "__main__":
    main()
