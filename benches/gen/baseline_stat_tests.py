#!/usr/bin/env python3
"""Time the Python baseline for the statistical-test family benchmark workload.

Runs the *same* workload the Rust criterion bench runs — N_TESTS independent
two-sample t-tests, each over SAMPLE_SIZE observations per group — using
`scipy.stats.ttest_ind` called once per pair, the idiomatic way a Python user
runs many independent tests. This is the realistic feature/hypothesis-screening
usage pattern and a loop-bound hot path: scipy pays per-call dispatch and
result-object construction overhead that numpy cannot fuse across independent
tests.

Discards a warm-up run and reports the median throughput (tests/sec). Prints JSON
so the recorded results file can embed it. No Rust here.
"""

import json
import platform
import sys
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy.stats import ttest_ind

N_TESTS = 5_000
SAMPLE_SIZE = 64
REPEATS = 5  # one warm-up + four timed


def build_pairs(n, m):
    """Deterministic n sample-pairs, each m observations, with a mean shift."""
    rng = np.random.default_rng(0x5EED1234)
    pairs = []
    for _ in range(n):
        a = rng.standard_normal(m)
        b = rng.standard_normal(m) + 0.3
        pairs.append((a, b))
    return pairs


PAIRS = build_pairs(N_TESTS, SAMPLE_SIZE)


def batch():
    """Run an independent two-sample t-test per pair; return the summed stat."""
    acc = 0.0
    for a, b in PAIRS:
        res = ttest_ind(a, b, equal_var=True)
        acc += res.statistic
    return acc


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
        "t_test_ind_batch": throughput(batch, N_TESTS),
    }
    out = {
        "n_tests": N_TESTS,
        "sample_size": SAMPLE_SIZE,
        "baseline_throughput_tests_per_sec": results,
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
