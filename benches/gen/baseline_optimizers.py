#!/usr/bin/env python3
"""Time the Python baseline for the optimizer-family benchmark workload.

Runs the *same* workload the Rust criterion bench runs — a multi-start
conjugate-gradient sweep solving Rosenbrock from N_STARTS deterministic starting
points to convergence — using `scipy.optimize.minimize(method="CG")`, the same
Fletcher–Reeves algorithm the Rust `conjugate_gradient` implements. This is the
realistic global-optimization usage pattern (multi-start to escape the non-convex
valley) and a loop-bound hot path: scipy pays per-iteration Python-callback and
dispatch overhead on every objective/gradient evaluation.

Discards a warm-up run and reports the median throughput (solves/sec). Prints
JSON so the recorded results file can embed it. No Rust here.
"""

import json
import platform
import sys
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize

N_STARTS = 2_000
MAX_ITER = 200
TOL = 1e-6
REPEATS = 5  # one warm-up + four timed


def rosenbrock(x):
    """Rosenbrock value f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2."""
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    """Analytic Rosenbrock gradient (matches the Rust objective's grad)."""
    d0 = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    d1 = 200.0 * (x[1] - x[0] ** 2)
    return np.array([d0, d1])


def starts(n):
    """The same N deterministic starting points the Rust bench uses."""
    out = []
    for i in range(n):
        frac = i / n
        out.append(np.array([4.0 * frac - 2.0, 4.0 * frac - 1.0]))
    return out


def multistart():
    """Solve Rosenbrock from every start with CG; return the summed minima."""
    acc = 0.0
    for x0 in POINTS:
        res = minimize(
            rosenbrock,
            x0,
            method="CG",
            jac=rosenbrock_grad,
            options={"maxiter": MAX_ITER, "gtol": TOL},
        )
        acc += res.fun
    return acc


POINTS = starts(N_STARTS)


def throughput(fn, n):
    """Return median items/sec over REPEATS runs, discarding the warm-up."""
    times = []
    for _ in range(REPEATS):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    timed = sorted(times)[1:]
    return n / median(timed)


def main():
    results = {
        "cg_multistart_rosenbrock": throughput(multistart, N_STARTS),
    }
    out = {
        "n_starts": N_STARTS,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "baseline_throughput_solves_per_sec": results,
        "env": {
            "python": sys.version.split()[0],
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
