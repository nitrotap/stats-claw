"""Golden fixtures for the cardinality-estimation (HyperLogLog) family.

Run via ``make fixtures`` or ``python3 -m gen.gen_cardinality`` from
``stats-claw/reference``.

IMPORTANT — this is an EXACT-GROUND-TRUTH accuracy fixture, NOT a library
bit-match. HyperLogLog has no canonical reference implementation in scipy /
numpy / scikit-learn to diff against bit-for-bit (the way the regression family
diffs against ``sklearn.LinearRegression``). Instead the fixture records the
**exact** distinct count of each test stream, computed with Python's built-in
``set`` (the unarguable ground truth), and the Rust suite asserts that the
HyperLogLog estimate lands inside HyperLogLog's *theoretical* relative standard
error band (``1.04 / sqrt(m)`` with ``m = 2**precision``). The provenance stamp
therefore names the ground truth as ``python set()`` with no external library
version — there is no library equivalence being claimed.

Reproducibility across languages: storing the raw 1k / 10k / 100k-element streams
as JSON would bloat the fixture, so each stream is defined by a small,
fully-documented 64-bit linear congruential generator (LCG) that BOTH this script
and the Rust test evaluate identically:

    state_0 = seed
    state_{i+1} = (state_i * 6364136223846793005 + 1442695040888963407) mod 2**64
    element_i  = state_{i+1} mod modulus      (i = 0 .. length-1)

The ``mod modulus`` step forces collisions so the distinct count is genuinely
less than ``length`` (otherwise ``0..n`` would make the test trivial). Python
computes ``len(set(elements))`` as the exact count; the Rust test regenerates the
identical LCG stream, feeds HyperLogLog, and checks the estimate against this
exact count. The LCG constants are the well-known Knuth/PCG multiplier and
increment (Numerical Recipes ``ranqd`` / PCG), chosen only because they are fixed,
portable, and produce a long full-period stream.

Reference:
  * Ground truth: Python ``set`` exact distinct count (NO external library).
  * Accuracy bound: HyperLogLog theory, relative standard error ``1.04/sqrt(m)``
    (Flajolet, Fusy, Gandouet & Meunier 2007).
"""
import math

from ._common import write_fixture

#: LCG multiplier (Knuth MMIX / PCG); fixed, portable, full-period over 2**64.
LCG_MULT = 6364136223846793005
#: LCG increment (Knuth MMIX / PCG); any odd constant gives a full period.
LCG_INC = 1442695040888963407
#: Modulus of the 64-bit LCG state.
LCG_MOD = 1 << 64
#: HyperLogLog precision the Rust suite uses for these fixtures (m = 2**14 = 16384,
#: relative standard error ~= 1.04/128 ~= 0.81%).
PRECISION = 14
#: Seed for every stream's LCG (one fixed seed keeps the fixture deterministic).
SEED = 0x1234_5678_9ABC_DEF0
#: The element-space modulus that forces duplicate elements into each stream so the
#: distinct count is non-trivially below the stream length. Sized so the 100k-element
#: stream draws from a 60k-value space: the longer streams genuinely repeat elements,
#: making ``len(set(...))`` a real exact distinct count rather than the trivial
#: ``length``.
ELEMENT_MODULUS = 60_000
#: Stream lengths to generate; the exact distinct count of each is the ground truth.
STREAM_LENGTHS = [1_000, 10_000, 100_000]


def _lcg_stream(length):
    """Yield ``length`` elements of the documented 64-bit LCG, modulo the element space.

    Mirrors the Rust test's generator exactly so both sides see the same multiset.
    """
    state = SEED
    out = []
    for _ in range(length):
        state = (state * LCG_MULT + LCG_INC) % LCG_MOD
        out.append(state % ELEMENT_MODULUS)
    return out


def _standard_error():
    """Return HyperLogLog's theoretical relative standard error for ``PRECISION``."""
    m = 1 << PRECISION
    return 1.04 / math.sqrt(m)


def gen_cardinality():
    """Write the cardinality golden fixture (exact distinct counts as ground truth)."""
    std_err = _standard_error()
    cases = []
    for length in STREAM_LENGTHS:
        elements = _lcg_stream(length)
        exact = len(set(elements))  # ground truth: exact distinct count.
        cases.append(
            {
                "length": length,
                "exact_distinct": exact,
            }
        )

    write_fixture(
        "cardinality",
        {
            # The LCG parameters the Rust test regenerates the identical streams from.
            "lcg_multiplier": LCG_MULT,
            "lcg_increment": LCG_INC,
            "seed": SEED,
            "element_modulus": ELEMENT_MODULUS,
            "precision": PRECISION,
            # HyperLogLog's theoretical relative standard error 1.04/sqrt(m).
            "standard_error": std_err,
            # Per-stream exact distinct counts (the ground truth to bound against).
            "cases": cases,
            # The accuracy gate the Rust suite applies: |est-exact|/exact <= bound.
            "error_bound_multiple": 3.0,
        },
        library="python set() exact distinct count (no external library)",
        version="stdlib",
        seed=SEED,
    )


def main():
    """Regenerate the cardinality golden fixture."""
    gen_cardinality()


if __name__ == "__main__":
    main()
