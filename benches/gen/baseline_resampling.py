#!/usr/bin/env python3
"""Time the Python baseline for the resampling-family benchmark workload.

Runs the *same* workload the Rust criterion bench runs — a bootstrap of the
median: draw B with-replacement resamples of a length-SAMPLE_SIZE sample and
recompute the median on each. The baseline is the idiomatic numpy bootstrap loop
a competent Python user writes: a Python `for` over B resamples, each doing a
vectorized `rng.integers` resample and `np.median`. This is the realistic
bootstrap-confidence-interval usage pattern; the per-resample median (a sort)
cannot collapse into one vectorized call, so the loop pays per-resample
interpreter overhead.

Discards a warm-up run and reports the median throughput (resamples/sec). Prints
JSON so the recorded results file can embed it. No Rust here.
"""

import json
import platform
import sys
from statistics import median as stat_median
from time import perf_counter

import numpy as np
import scipy

B = 5_000
SAMPLE_SIZE = 200
REPEATS = 5  # one warm-up + four timed


def build_sample(n):
    """A deterministic sample to bootstrap."""
    rng = np.random.default_rng(0xB0075747)
    return rng.standard_normal(n)


DATA = build_sample(SAMPLE_SIZE)


def bootstrap_median():
    """Bootstrap of the median over B resamples (idiomatic numpy loop)."""
    rng = np.random.default_rng(0xC0FFEE)
    n = DATA.shape[0]
    stats = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        stats[i] = np.median(DATA[idx])
    return stats


def throughput(fn, n):
    """Return median resamples/sec over REPEATS runs, discarding the warm-up."""
    times = []
    for _ in range(REPEATS):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    timed = sorted(times)[1:]
    return n / stat_median(timed)


def main():
    results = {
        "bootstrap_median": throughput(bootstrap_median, B),
    }
    out = {
        "b": B,
        "sample_size": SAMPLE_SIZE,
        "baseline_throughput_resamples_per_sec": results,
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
