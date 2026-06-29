#!/usr/bin/env python3
"""Time the Python baseline for the algorithms-family benchmark workload.

Runs the *same* workload the Rust criterion bench runs — PELT change-point
detection (L2 cost) on a length-N piecewise-constant signal — using
`ruptures.Pelt(model="l2")`, the same reference the equivalence suite checks
against. PELT is an inherently sequential dynamic program that ruptures
implements in pure Python; neither library vectorizes it, so the interpreted
loop pays full per-step overhead.

Discards a warm-up run and reports the median throughput (samples/sec, i.e.
signal length processed per second). Prints JSON so the recorded results file can
embed it. No Rust here.

Requires ruptures: `pip3 install --break-system-packages --user ruptures`.
"""

import json
import platform
import sys
from statistics import median
from time import perf_counter

import numpy as np
import ruptures

N = 2_000
PENALTY = 10.0
MIN_SIZE = 5
REPEATS = 5  # one warm-up + four timed


def build_signal(n):
    """The same deterministic piecewise-constant signal the Rust bench uses."""
    xs = np.empty(n)
    for i in range(n):
        seg = (i * 5) // n
        level = float(seg) * 3.0
        t = float(i % 7)
        xs[i] = level + t * 0.1 - 0.3
    return xs.reshape(-1, 1)


SIGNAL = build_signal(N)


def pelt():
    """Detect change-points with ruptures PELT (L2)."""
    algo = ruptures.Pelt(model="l2", min_size=MIN_SIZE, jump=1).fit(SIGNAL)
    return algo.predict(pen=PENALTY)


def throughput(fn, n):
    """Return median samples/sec over REPEATS runs, discarding the warm-up."""
    times = []
    for _ in range(REPEATS):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    timed = sorted(times)[1:]
    return n / median(timed)


def main():
    results = {
        "pelt_l2_changepoint": throughput(pelt, N),
    }
    out = {
        "n": N,
        "penalty": PENALTY,
        "min_size": MIN_SIZE,
        "baseline_throughput_samples_per_sec": results,
        "env": {
            "python": sys.version.split()[0],
            "ruptures": ruptures.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
