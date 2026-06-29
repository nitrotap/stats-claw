#!/usr/bin/env python3
"""Time the Python baseline for the distribution-family benchmark workload.

The gated metric is the **genuine batch elementwise op** — `norm.pdf(xs)` /
`norm.cdf(xs)` / `norm.rvs(size=N)` over an N-element array — compared against the
matching stats-claw batch op. This is the honest apples-to-apples comparison:
vectorized scipy.stats.norm vs the stats-claw batch loop. (The previously-gated
"per-draw vs stdlib random.gauss" metric was a favorable swap and is no longer the
gate; it is still timed below and DISCLOSED FOR CONTEXT only.)

On dense batch ops scipy dispatches to vectorized SIMD C ufuncs over the whole
array, so a scalar Rust loop is at an architectural disadvantage; on this gate
machine (x86_64 under Rosetta 2) hand-written SIMD is infeasible (x86 intrinsics
translate to NEON). See benches/results/distributions.json `measured_ceiling`.

Discards a warm-up run and reports the median throughput (elements/sec) per
operation. Prints JSON so the recorded results file can embed it. No Rust here.
"""

import json
import platform
import random
import sys
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy.stats import norm

N = 100_000
# Per-draw sampling is ~1000x slower per element than vectorized ops, so it runs a
# smaller workload to keep the script quick; throughput is per-element regardless.
N_PER_DRAW = 50_000
REPEATS = 7  # one warm-up + six timed


def throughput(fn, n=N):
    """Return median elements/sec over REPEATS runs, discarding the warm-up."""
    times = []
    for _ in range(REPEATS):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    timed = sorted(times)[1:]  # drop the slowest cold-start sample
    return n / median(timed)


def per_draw_gauss():
    """Draw N_PER_DRAW standard-normal variates one at a time (online pattern)."""
    rng = random.Random(0xC0FFEE)
    acc = 0.0
    for _ in range(N_PER_DRAW):
        acc += rng.gauss(0.0, 1.0)
    return acc


def main():
    xs = np.linspace(-5.0, 5.0, N)
    rng = np.random.default_rng(0xC0FFEE)
    results = {
        "pdf": throughput(lambda: norm.pdf(xs)),
        "cdf": throughput(lambda: norm.cdf(xs)),
        "sample": throughput(lambda: norm.rvs(size=N, random_state=rng)),
        "sample_per_draw": throughput(per_draw_gauss, n=N_PER_DRAW),
    }
    out = {
        "n": N,
        "n_per_draw": N_PER_DRAW,
        "baseline_throughput_elems_per_sec": results,
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
