"""Golden fixtures for the optimizer family (Track B / AC-3).

Run via ``make fixtures`` or ``python3 -m gen.gen_optimizers`` from
``stats-claw/reference``. Python (scipy) runs ONLY here at generation time;
``cargo test`` reads the committed JSON offline.

Two kinds of fixture are written:

* ``opt_problems`` — the analytic ground truth (known minimizers / minima) for
  the shared quadratic and Rosenbrock objectives, plus the shared starting
  points. Every optimizer converges against these.
* ``opt_scipy`` — the ``scipy.optimize`` results for the documented method
  mapping (CG, Newton-CG, L-BFGS-B, dual_annealing, differential_evolution), so
  the stats-claw optimizers with a faithful counterpart can be cross-checked.

The stats-claw <-> scipy mapping is recorded here and mirrored in the Rust
``optimizers/mod.rs`` doc-comment (AC-3 story 3.3 auditability).
"""
import numpy as np
import scipy
from scipy import optimize

from ._common import write_fixture

VER = scipy.__version__

#: Shared quadratic: f(x) = (x0-3)^2 + (x1+2)^2 ; min 0 at [3, -2].
QUAD_CENTER = [3.0, -2.0]
QUAD_X0 = [0.0, 0.0]

#: Shared Rosenbrock: min 0 at [1, 1].
ROSEN_X0 = [-1.2, 1.0]


def _quadratic(x):
    return (x[0] - QUAD_CENTER[0]) ** 2 + (x[1] - QUAD_CENTER[1]) ** 2


def _quadratic_grad(x):
    return np.array([2.0 * (x[0] - QUAD_CENTER[0]), 2.0 * (x[1] - QUAD_CENTER[1])])


def _quadratic_hess(_x):
    return np.array([[2.0, 0.0], [0.0, 2.0]])


def _rosen(x):
    return optimize.rosen(x)


def gen_problems():
    """Analytic ground truth for the shared objectives."""
    write_fixture(
        "opt_problems",
        {
            "quadratic": {
                "center": QUAD_CENTER,
                "x0": QUAD_X0,
                "minimizer": QUAD_CENTER,
                "min_value": 0.0,
            },
            "rosenbrock": {
                "x0": ROSEN_X0,
                "minimizer": [1.0, 1.0],
                "min_value": 0.0,
            },
        },
        library="analytic",
        version="n/a",
    )


def gen_scipy():
    """scipy.optimize results for the documented method mapping."""
    cg_quad = optimize.minimize(
        _quadratic, QUAD_X0, jac=_quadratic_grad, method="CG"
    )
    cg_rosen = optimize.minimize(
        _rosen, ROSEN_X0, jac=optimize.rosen_der, method="CG"
    )
    newton_quad = optimize.minimize(
        _quadratic, QUAD_X0, jac=_quadratic_grad, hess=_quadratic_hess,
        method="Newton-CG",
    )
    newton_rosen = optimize.minimize(
        _rosen, ROSEN_X0, jac=optimize.rosen_der, hess=optimize.rosen_hess,
        method="Newton-CG",
    )
    lbfgs_quad = optimize.minimize(
        _quadratic, QUAD_X0, jac=_quadratic_grad, method="L-BFGS-B"
    )
    lbfgs_rosen = optimize.minimize(
        _rosen, ROSEN_X0, jac=optimize.rosen_der, method="L-BFGS-B"
    )
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    da = optimize.dual_annealing(_quadratic, bounds, seed=1)
    de = optimize.differential_evolution(_quadratic, bounds, seed=1)

    def pack(res):
        return {"x": res.x.tolist(), "fx": float(res.fun)}

    write_fixture(
        "opt_scipy",
        {
            "cg_quadratic": pack(cg_quad),
            "cg_rosenbrock": pack(cg_rosen),
            "newton_quadratic": pack(newton_quad),
            "newton_rosenbrock": pack(newton_rosen),
            "lbfgs_quadratic": pack(lbfgs_quad),
            "lbfgs_rosenbrock": pack(lbfgs_rosen),
            "dual_annealing_quadratic": pack(da),
            "differential_evolution_quadratic": pack(de),
        },
        library="scipy.optimize",
        version=VER,
        seed=1,
    )


def main():
    """Regenerate every optimizer fixture."""
    gen_problems()
    gen_scipy()


if __name__ == "__main__":
    main()
