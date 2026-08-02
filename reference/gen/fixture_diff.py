"""Numeric drift check for the committed golden fixtures.

``make fixture-drift`` regenerates every fixture from the pinned Python stack
into a scratch directory and runs this module over the two trees. It answers one
question: **has a reference library changed its answer?** — which is not the same
question as "are the two files byte-identical".

Why not ``git diff --exit-code``
--------------------------------

The fixtures are IEEE-754 doubles serialised at full precision. Most of them are
produced by BLAS/LAPACK-backed paths (``PCA``, ``FastICA``, ``FactorAnalysis``,
``LocallyLinearEmbedding``, ``LinearRegression``, ``Ridge``) whose reductions are
blocked and threaded: the summation order, and therefore the last bits, depend on
the CPU instruction set, the thread count, and which BLAS the wheel was built
against. The ``scipy.special``-backed distribution and test fixtures drift for a
different reason — those kernels are sub-ULP accurate but not correctly rounded,
so a different libm rounds the final digit differently.

A byte-equality gate therefore asserts that floating point is bit-identical
across machines, which it is not. This is the same mistake this repository has
already corrected once, for the same reason, in
``src/resampling/monte_carlo/tests.rs``: a bit-exact pin is only ever true on the
machine that recorded it, and belongs only where bit-exactness is a guarantee the
producer can actually keep. Measured drift between the recording machine and a
GitHub ``ubuntu-24.04`` x86-64 runner, and between that runner and an aarch64
macOS/Accelerate build, is reported in ``docs/fixture-drift.md``.

What this gate does assert
--------------------------

Everything that is *not* a float is compared exactly: the set of fixture files,
every object key, every array length, every integer, boolean, and string. That
includes each fixture's ``_provenance`` block, so a version bump in
``requirements.txt`` that changes an answer is reported by name.

Floats are compared inside the scipy band ``|a - b| <= atol + rtol * |committed|``
with :data:`DEFAULT_RTOL` / :data:`DEFAULT_ATOL`. Those defaults are chosen from
two independent numbers, not from what makes the job pass:

* the *floor* is the measured cross-platform noise on these fixtures, and
* the *ceiling* is the tightest tolerance the Rust equivalence suite asserts
  against any fixture value — ``atol = rtol = 1e-12`` on distribution moments
  (``tests/dist/mod.rs``). The gate must sit below that ceiling, or a drift that
  the gate waves through could consume a Rust test's whole error budget and the
  fixture would become the weaker constraint.

A handful of quantities are not reproducible at that band on *any* machine, for
reasons intrinsic to the quantity rather than to floating point. Each one has an
entry in :data:`POLICIES` carrying the reason and the Rust assertion it has to
stay tighter than. Nothing else is exempt: a quantity is only listed here if its
own definition, not its rounding, is platform-dependent.
"""
import argparse
import fnmatch
import json
import math
import pathlib
import re
import sys

#: Relative half-width of the accepted band, ``rtol`` in
#: ``|a - b| <= atol + rtol * |committed|``.
#:
#: ``1e-13`` is an order of magnitude below the tightest tolerance the Rust suite
#: asserts against a fixture value (``1e-12`` on distribution ``mean`` /
#: ``variance``, ``tests/dist/mod.rs``), and two to seven orders below every other
#: consumed tolerance (``1e-12`` association metrics, ``1e-9`` regression /
#: density / outlier / ``ppf``, ``1e-8`` test statistics, ``1e-6`` decomposition).
#: It is also two orders above the measured cross-platform noise on the
#: quantities the suite consumes.
DEFAULT_RTOL = 1e-13

#: Absolute floor of the accepted band, ``atol``.
#:
#: A pure relative test is the wrong instrument for a value that is a residual
#: rather than a measurement: ``opt_scipy``'s L-BFGS Rosenbrock ``fx`` is the
#: objective at a minimiser, ``2.8e-12``, where the answer is "indistinguishable
#: from zero" and the last bits carry no information. ``1e-13`` is likewise an
#: order below the tightest absolute tolerance the Rust suite asserts against a
#: fixture value (``1e-12`` on distribution moments, which for Student's t
#: ``mean = 0`` is a pure absolute bound).
DEFAULT_ATOL = 1e-13


class Policy:
    """A comparison rule for one fixture field, with the reason it exists."""

    def __init__(self, kind, why, *, rtol=None, atol=None):
        """Build a policy.

        Args:
            kind: ``"band"`` (widened numeric band) or ``"permutation"``
                (integer labels equal up to a relabelling bijection).
            why: prose justification, printed when the policy is exercised.
            rtol: relative half-width, for ``kind="band"``.
            atol: absolute floor, for ``kind="band"``.
        """
        self.kind = kind
        self.why = why
        self.rtol = rtol
        self.atol = atol


#: Fields whose *definition* — not merely whose rounding — is platform-dependent.
#:
#: Keyed by ``"<fixture>:<pointer>"``, where the pointer has array indices
#: collapsed to ``[]`` and may use ``fnmatch`` globs. Every entry states what the
#: quantity is, the drift measured on it, and the Rust assertion it must stay
#: tighter than; none may be widened past that assertion, because beyond it the
#: fixture stops constraining anything the suite does not already constrain.
#:
#: Measurements quoted below are from the two comparisons in
#: ``docs/fixture-drift.md``: the committed fixtures against a regeneration on a
#: GitHub ``ubuntu-24.04`` x86-64 runner (OpenBLAS), and against one on aarch64
#: macOS (Accelerate), both on the pinned stack.
POLICIES = {
    "algo_spectral_blobs:/labels[]": Policy(
        "permutation",
        "sklearn's SpectralClustering takes cluster identity from the sign and "
        "ordering of Laplacian eigenvectors, which are defined only up to a "
        "permutation and are not fixed across LAPACK implementations: the "
        "partition is the answer, the integer names are not. Measured: unchanged "
        "on the x86-64 runner, wholly relabelled (1 <-> 2) on aarch64. The Rust "
        "consumer (tests/equiv/algorithms.rs, spectral_agrees_with_sklearn_by_ari) "
        "scores it by adjusted Rand index >= 0.99, which is invariant under "
        "exactly this relabelling, so demanding the names match asserts strictly "
        "more than the suite does — and something that is not true. This policy "
        "accepts a bijection, not a rewrite: moving one point to another cluster "
        "still fails, which is a stronger check than ARI >= 0.99, since that "
        "would tolerate a few misplaced points and this tolerates none.",
    ),
    "algo_lle:/trustworthiness": Policy(
        "band",
        "A rank statistic over 5-nearest-neighbour sets, so a step function of "
        "the embedding with quantum 1 / (n k (2n - 3k - 1)) = 1/15600 here; and "
        "the LLE embedding is the null space of a near-singular matrix, whose "
        "basis genuinely differs between LAPACK builds. This is not rounding. "
        "Measured: +134 quanta (0.9206 -> 0.9292, +0.92% relative) on the x86-64 "
        "runner and +83 quanta (-> 0.9259, +0.58%) on aarch64. 2.5e-2 is 2.7x the "
        "largest of those and still 2.2x tighter than the 0.05 one-sided slack "
        "the Rust consumer allows (tests/equiv/algorithms_decomp.rs, "
        "lle_embedding_trustworthiness_matches_sklearn), so the gate fires well "
        "before drift could threaten that assertion. Note honestly what this "
        "buys: platform noise here already spends 17% of the suite's slack, so "
        "the gate can catch a collapse in embedding quality — tens of percent — "
        "but cannot resolve a one-percent upstream regression. Pinning a floor "
        "instead of a value, as algo_umap already does with "
        "trustworthiness_target, would be the structural fix.",
        rtol=2.5e-2,
        atol=0.0,
    ),
    "algo_tsne:/trustworthiness": Policy(
        "band",
        "The same rank-statistic step function as algo_lle, over an embedding "
        "produced by a gradient descent whose per-iteration reductions are "
        "threaded. Measured: unchanged on the x86-64 runner, -0.17% on aarch64. "
        "Banded with algo_lle at 2.5e-2 for the same reasons, against the same "
        "0.05 Rust slack (tests/equiv/algorithms_decomp.rs, "
        "tsne_embedding_trustworthiness_matches_sklearn).",
        rtol=2.5e-2,
        atol=0.0,
    ),
    "opt_scipy:/*/x[]": Policy(
        "band",
        "Where an iterative optimiser stops is not a mathematically determined "
        "point: it is wherever the convergence test tripped, and near a minimum "
        "the objective is flat, so an O(eps) change in the gradient displaces the "
        "iterate by O(eps / lambda_min). On Rosenbrock's ill-conditioned valley "
        "that amplification is about five orders of magnitude. Measured: 2.9e-11 "
        "relative on newton_rosenbrock/x[1] on the x86-64 runner (1.4e-14 on "
        "aarch64). 1e-8 is 344x that and still four orders tighter than the "
        "loosest and 1e4 tighter than the tightest tolerance the Rust consumer "
        "asserts on these coordinates (tests/equiv/optimizers.rs, "
        "assert_agrees_scipy: rtol 1e-4 with atol 1e-5 to 1e-2 by problem).",
        rtol=1e-8,
        atol=1e-8,
    ),
    "opt_scipy:/*/fx": Policy(
        "band",
        "The objective *at* a minimiser, spanning 1.97e-30 to 3.03e-10 across "
        "these problems. The answer is 'indistinguishable from zero'; the digits "
        "are the noise floor of the solver, not a measurement, so a relative test "
        "is the wrong instrument. Measured: lbfgs_quadratic/fx moved a whole "
        "decade on the x86-64 runner, 1.97e-30 -> 1.97e-31 (0.9 relative, 1.8e-30 "
        "absolute) — a full-precision relative gate would fail on that forever "
        "while learning nothing. No Rust test reads fx at all "
        "(tests/equiv/optimizers.rs compares only x), so 1e-8 absolute asserts "
        "the only thing worth asserting: the minimiser is still a minimiser. A "
        "genuine failure to converge leaves fx at O(1e-3) or worse and is caught "
        "with five orders to spare.",
        rtol=0.0,
        atol=1e-8,
    ),
}

_INDEX = re.compile(r"\[\d+\]")


def _shape(pointer):
    """Collapse array indices in a JSON pointer: ``/a[3][0]`` -> ``/a[][]``."""
    return _INDEX.sub("[]", pointer)


def _match_policy(key):
    """Find the policy governing ``key``, preferring an exact entry to a glob.

    Args:
        key: ``"<fixture>:<shaped pointer>"``.

    Returns:
        A ``(matched key, policy)`` pair, or ``(None, None)``.

    Raises:
        KeyError: if two glob policies both match, which would make the applied
            tolerance depend on dictionary order.
    """
    if key in POLICIES:
        return key, POLICIES[key]
    hits = [k for k in POLICIES if fnmatch.fnmatchcase(key, k)]
    if not hits:
        return None, None
    if len(hits) > 1:
        raise KeyError(f"{key} matches several policies: {sorted(hits)}")
    return hits[0], POLICIES[hits[0]]


def _is_bijection(left, right):
    """Report whether two equal-length label sequences differ only by relabelling.

    Args:
        left: the committed label sequence.
        right: the regenerated label sequence.

    Returns:
        ``True`` when a single consistent relabelling maps ``left`` onto
        ``right`` and back, i.e. the two induce the same partition.
    """
    if len(left) != len(right):
        return False
    forward, backward = {}, {}
    for a, b in zip(left, right):
        if forward.setdefault(a, b) != b or backward.setdefault(b, a) != a:
            return False
    return True


class Comparison:
    """Walks two fixture trees and accumulates violations and near-misses."""

    def __init__(self, rtol, atol):
        """Build a comparison at the given band.

        Args:
            rtol: relative half-width applied to fields without a policy.
            atol: absolute floor applied to fields without a policy.
        """
        self.rtol = rtol
        self.atol = atol
        #: ``(fixture, pointer, committed, regenerated, detail)`` past tolerance.
        self.violations = []
        #: ``(headroom, fixture, pointer, committed, regenerated)`` inside it.
        self.accepted = []
        #: Policy keys actually exercised, for the report.
        self.policies_used = set()
        #: Hand-written fixtures no generator produces; see :data:`HAND_WRITTEN`.
        self.skipped = []

    def _policy(self, fixture, pointer):
        """Return the policy governing ``pointer`` in ``fixture``, or ``None``."""
        matched, policy = _match_policy(f"{fixture}:{_shape(pointer)}")
        if policy is not None:
            self.policies_used.add(matched)
        return policy

    def _fail(self, fixture, pointer, a, b, detail):
        """Record a violation."""
        self.violations.append((fixture, pointer, a, b, detail))

    def compare_float(self, fixture, pointer, a, b):
        """Compare two floats inside the band that governs this field."""
        if math.isnan(a) and math.isnan(b):
            return
        policy = self._policy(fixture, pointer)
        rtol = self.rtol if policy is None else policy.rtol
        atol = self.atol if policy is None else policy.atol
        if a == b:
            self.accepted.append((0.0, fixture, pointer, a, b))
            return
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            self._fail(fixture, pointer, a, b, "non-finite mismatch")
            return
        tol = atol + rtol * abs(a)
        deviation = abs(a - b)
        if deviation > tol:
            self._fail(
                fixture, pointer, a, b,
                f"|delta| {deviation:.6e} > atol {atol:.1e} + rtol {rtol:.1e} "
                f"* |committed| = {tol:.6e}",
            )
        else:
            self.accepted.append(
                (deviation / tol if tol else 0.0, fixture, pointer, a, b)
            )

    def walk(self, fixture, pointer, a, b):
        """Recursively compare two decoded JSON values."""
        if isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                self._fail(fixture, pointer, a, b, "boolean mismatch")
            return
        if isinstance(a, dict):
            if not isinstance(b, dict):
                self._fail(fixture, pointer, type(a).__name__, type(b).__name__,
                           "type mismatch")
                return
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    self._fail(fixture, f"{pointer}/{key}", a.get(key), b.get(key),
                               "key present on only one side")
                else:
                    self.walk(fixture, f"{pointer}/{key}", a[key], b[key])
            return
        if isinstance(a, list):
            if not isinstance(b, list):
                self._fail(fixture, pointer, type(a).__name__, type(b).__name__,
                           "type mismatch")
                return
            if len(a) != len(b):
                self._fail(fixture, pointer, len(a), len(b), "array length changed")
                return
            policy = self._policy(fixture, f"{pointer}[]")
            if policy is not None and policy.kind == "permutation":
                if not _is_bijection(a, b):
                    self._fail(fixture, pointer, a, b,
                               "partition changed (not a relabelling)")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                self.walk(fixture, f"{pointer}[{i}]", x, y)
            return
        if isinstance(a, float) or isinstance(b, float):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                self._fail(fixture, pointer, a, b, "type mismatch")
                return
            self.compare_float(fixture, pointer, float(a), float(b))
            return
        if a != b:
            self._fail(fixture, pointer, a, b, "value changed")


#: ``_provenance.library`` marking a fixture that no generator produces.
#:
#: ``harness_smoke.json`` is authored by hand to smoke-test the Rust loader
#: itself, so it is legitimately absent from a regeneration. Note that the byte
#: gate this replaced could not have told the difference: ``git diff`` says
#: nothing about a tracked file that was simply never rewritten, so a generator
#: silently dropping a fixture would have gone unnoticed indefinitely.
HAND_WRITTEN = "hand-written"


def compare_trees(committed, regenerated, rtol, atol):
    """Compare two fixture directories.

    Args:
        committed: directory holding the tracked ``golden/*.json``.
        regenerated: directory holding a fresh regeneration.
        rtol: relative half-width for fields without a policy.
        atol: absolute floor for fields without a policy.

    Returns:
        The completed :class:`Comparison`.
    """
    comparison = Comparison(rtol, atol)
    left = {p.name for p in committed.glob("*.json")}
    right = {p.name for p in regenerated.glob("*.json")}
    for name in sorted(left - right):
        doc = json.loads((committed / name).read_text())
        if doc.get("_provenance", {}).get("library") == HAND_WRITTEN:
            comparison.skipped.append(name)
            continue
        comparison.violations.append(
            (name, "", name, None,
             "a generator stopped emitting this fixture (and it is not marked "
             f"{HAND_WRITTEN!r} in its _provenance)"),
        )
    for name in sorted(right - left):
        comparison.violations.append(
            (name, "", None, name,
             "a generator emitted a fixture that is not committed"),
        )
    for name in sorted(left & right):
        fixture = name[: -len(".json")]
        comparison.walk(
            fixture,
            "",
            json.loads((committed / name).read_text()),
            json.loads((regenerated / name).read_text()),
        )
    return comparison


def report(comparison, top):
    """Print the drift report.

    Always prints the widest accepted deviations, so a run that passes still
    shows how much headroom is left before the gate trips.

    Args:
        comparison: the completed :class:`Comparison`.
        top: how many of the widest accepted deviations to list.
    """
    print(f"band: |delta| <= {comparison.atol:.1e} + {comparison.rtol:.1e} "
          f"* |committed|")
    accepted = sorted(comparison.accepted, reverse=True)
    moved = [row for row in accepted if row[0] > 0.0]
    print(f"floats compared: {len(comparison.accepted)}; "
          f"moved but within band: {len(moved)}; "
          f"violations: {len(comparison.violations)}")
    if comparison.skipped:
        print(f"hand-written, not regenerated: {', '.join(comparison.skipped)}")
    if moved:
        print("\nwidest accepted deviations (fraction of the allowed band):")
        for headroom, fixture, pointer, a, b in moved[:top]:
            rel = abs(a - b) / abs(a) if a else float("nan")
            print(f"  {headroom:7.1%} of band  rel {rel:9.2e}  "
                  f"{fixture}{pointer}\n"
                  f"      committed   {a!r}\n"
                  f"      regenerated {b!r}")
    if comparison.policies_used:
        print("\npolicies exercised:")
        for key in sorted(comparison.policies_used):
            print(f"  {key}\n      {POLICIES[key].why}")
    if comparison.violations:
        print(f"\nDRIFT: {len(comparison.violations)} value(s) outside tolerance")
        for fixture, pointer, a, b, detail in comparison.violations:
            print(f"  {fixture}{pointer}: {detail}\n"
                  f"      committed   {a!r}\n"
                  f"      regenerated {b!r}")


def main(argv=None):
    """Run the drift check. Returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("committed", type=pathlib.Path,
                        help="directory of tracked golden fixtures")
    parser.add_argument("regenerated", type=pathlib.Path,
                        help="directory of freshly regenerated fixtures")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                        help=f"relative half-width (default {DEFAULT_RTOL:g})")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL,
                        help=f"absolute floor (default {DEFAULT_ATOL:g})")
    parser.add_argument("--top", type=int, default=10,
                        help="how many widest accepted deviations to list")
    parser.add_argument("--report-only", action="store_true",
                        help="print the report but always exit 0 (diagnostics)")
    args = parser.parse_args(argv)

    comparison = compare_trees(
        args.committed, args.regenerated, args.rtol, args.atol
    )
    report(comparison, args.top)
    if args.report_only:
        return 0
    if comparison.violations:
        print("\nA reference library changed an answer, or a fixture was edited "
              "by hand. Investigate before regenerating: see the module docstring "
              "in reference/gen/fixture_diff.py.")
        return 1
    print("\nno drift: every fixture matches the reference stack within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
