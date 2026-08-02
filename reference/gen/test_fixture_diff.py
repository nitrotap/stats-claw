"""Fault-injection tests for the golden-fixture drift gate.

The gate in :mod:`gen.fixture_diff` exists to fail when a reference library
changes an answer. A gate that has been loosened until it accepts everything
looks exactly like a gate that is passing, so these tests inject each kind of
change the gate is supposed to catch and assert that it does.

Pure stdlib and pure synthetic data: no numpy, no scipy, no committed fixture is
read. It runs in the ``gates`` job on every push and PR, and again inside the
drift job itself, so the gate is proven to still bite on the same run that
reports no drift.

Run directly: ``python3 -m gen.test_fixture_diff`` from ``reference/``.
"""
import copy
import json
import pathlib
import tempfile
import unittest

from . import fixture_diff

#: A miniature fixture tree exercising each shape the real fixtures use.
BASELINE = {
    "dist_demo": {
        "mean": 1.5,
        "variance": 4.0,
        "pdf": [0.3989422804014327, 0.24197072451914337],
        "df": 3,
        "_provenance": {"library": "scipy.stats.norm", "version": "1.17.1",
                        "seed": None},
    },
    "algo_spectral_blobs": {
        "labels": [0, 0, 1, 1, 2, 2, 0, 1],
        "gamma": 1.0,
        "_provenance": {"library": "sklearn.cluster.SpectralClustering",
                        "version": "1.9.0", "seed": 42},
    },
    "algo_lle": {
        "trustworthiness": 0.9205769230769231,
        "_provenance": {"library": "sklearn.manifold.LocallyLinearEmbedding",
                        "version": "1.9.0", "seed": 42},
    },
    "opt_scipy": {
        # `fx` is the objective at a minimiser: indistinguishable from zero, so
        # its digits carry no information. `x` is where the solver stopped, which
        # is reproducible only to its own convergence noise. `fun_calls` sits
        # beside both and must NOT inherit their widened bands.
        "lbfgs_rosenbrock": {
            "fx": 2.807650087972804e-12,
            "x": [1.0, 1.0],
            "fun_calls": 40.0,
        },
        "_provenance": {"library": "scipy.optimize", "version": "1.17.1",
                        "seed": None},
    },
}


def _write(directory, tree):
    """Serialise ``tree`` into ``directory`` the way ``write_fixture`` does."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, doc in tree.items():
        (directory / f"{name}.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n"
        )


class DriftGateTest(unittest.TestCase):
    """Each test mutates one value and asserts the gate's verdict."""

    def run_gate(self, mutate=None):
        """Compare the baseline against a mutated copy of itself.

        Args:
            mutate: callable receiving a deep copy of :data:`BASELINE` to
                modify in place, or ``None`` for an unmodified comparison.

        Returns:
            The completed :class:`~gen.fixture_diff.Comparison`.
        """
        regenerated = copy.deepcopy(BASELINE)
        if mutate is not None:
            mutate(regenerated)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "committed", BASELINE)
            _write(root / "regenerated", regenerated)
            return fixture_diff.compare_trees(
                root / "committed", root / "regenerated",
                fixture_diff.DEFAULT_RTOL, fixture_diff.DEFAULT_ATOL,
            )

    def assert_clean(self, mutate=None, msg=""):
        """Assert the gate accepts the mutation."""
        violations = self.run_gate(mutate).violations
        self.assertEqual(violations, [], msg or "expected no violations")

    def assert_caught(self, mutate, msg=""):
        """Assert the gate rejects the mutation."""
        violations = self.run_gate(mutate).violations
        self.assertTrue(violations, msg or "the gate failed to catch this")

    # -- the gate must not be noise-triggered -------------------------------

    def test_identical_trees_are_clean(self):
        """An unmodified regeneration reports nothing."""
        self.assert_clean()

    def test_last_bit_rounding_is_accepted(self):
        """A few-ULP move — the thing byte-equality wrongly failed on — passes."""
        def mutate(tree):
            tree["dist_demo"]["variance"] = 4.0 * (1.0 + 1e-15)
            tree["dist_demo"]["pdf"][0] = 0.39894228040143276
        self.assert_clean(mutate, "few-ULP rounding must not fail the gate")

    def test_near_zero_residual_is_governed_by_the_absolute_floor(self):
        """An optimiser residual may move a whole decade and still pass.

        ``fx`` is the objective *at* a minimiser. This is the real movement
        measured on the x86-64 runner for ``lbfgs_quadratic``: 1.97e-30 to
        1.97e-31, a relative change of 0.9 and an absolute change of 1.8e-30. A
        full-precision relative gate would fail on it forever while learning
        nothing, because both numbers say the same thing — zero.
        """
        def mutate(tree):
            tree["opt_scipy"]["lbfgs_rosenbrock"]["fx"] = 1.9721522630525295e-31
        self.assert_clean(mutate)

    def test_optimizer_iterate_noise_is_accepted(self):
        """An iterate displaced by the solver's own stopping noise passes.

        The real movement measured on the x86-64 runner for
        ``newton_rosenbrock/x[1]``: 2.9e-11 relative, five orders of magnitude
        above machine epsilon because Rosenbrock's valley is flat enough to
        amplify a last-bit change in the gradient by that much.
        """
        def mutate(tree):
            tree["opt_scipy"]["lbfgs_rosenbrock"]["x"][1] = 1.0 * (1 + 2.9e-11)
        self.assert_clean(mutate)

    def test_spectral_relabelling_is_accepted(self):
        """Renaming clusters is not a change of partition."""
        def mutate(tree):
            rename = {0: 2, 1: 0, 2: 1}
            tree["algo_spectral_blobs"]["labels"] = [
                rename[x] for x in tree["algo_spectral_blobs"]["labels"]
            ]
        self.assert_clean(mutate, "a pure relabelling must be accepted")

    def test_trustworthiness_moves_within_its_own_step_size(self):
        """A sub-1% move in a rank statistic passes; see the policy's rationale."""
        def mutate(tree):
            tree["algo_lle"]["trustworthiness"] *= 1.0 - 5e-3
        self.assert_clean(mutate)

    # -- the gate must still bite -------------------------------------------

    def test_catches_drift_at_the_rust_suite_tolerance(self):
        """A move of 1e-12 — the Rust suite's tightest budget — is caught.

        This is the load-bearing test. The whole point of the band is to sit
        below the tolerance the Rust tests assert against these same values, so
        the fixtures never become the weaker constraint.
        """
        def mutate(tree):
            tree["dist_demo"]["variance"] = 4.0 * (1.0 + 1e-12)
        self.assert_caught(mutate)

    def test_catches_drift_just_past_the_band(self):
        """Three times the band is caught, so the edge is where it is claimed."""
        def mutate(tree):
            tree["dist_demo"]["variance"] = 4.0 * (1.0 + 3e-13)
        self.assert_caught(mutate)

    def test_catches_drift_in_an_array_element(self):
        """Array elements are compared, not just scalars."""
        def mutate(tree):
            tree["dist_demo"]["pdf"][1] *= 1.0 + 1e-12
        self.assert_caught(mutate)

    def test_catches_a_provenance_version_bump(self):
        """A library version change is reported by name, not smoothed over."""
        def mutate(tree):
            tree["dist_demo"]["_provenance"]["version"] = "1.18.0"
        self.assert_caught(mutate)

    def test_catches_an_integer_change(self):
        """Degrees of freedom and other integers are compared exactly."""
        def mutate(tree):
            tree["dist_demo"]["df"] += 1
        self.assert_caught(mutate)

    def test_catches_a_repartition_disguised_as_a_relabelling(self):
        """Moving one point to another cluster is a real change, not a rename."""
        def mutate(tree):
            labels = tree["algo_spectral_blobs"]["labels"]
            labels[0] = 1
        self.assert_caught(mutate, "a changed partition must fail")

    def test_catches_a_trustworthiness_collapse(self):
        """A quality metric falling past its policy band fails.

        Five percent is well short of what a real upstream regression in LLE
        would look like, and it is already caught — but note the policy's own
        docstring about what this band cannot resolve.
        """
        def mutate(tree):
            tree["algo_lle"]["trustworthiness"] *= 1.0 - 5e-2
        self.assert_caught(mutate)

    def test_catches_an_optimizer_that_stops_converging(self):
        """An objective that stops being zero fails, which is the point of fx.

        No Rust test reads ``fx``, so this gate is the only thing asserting that
        scipy still drives these problems to a minimum at all.
        """
        def mutate(tree):
            tree["opt_scipy"]["lbfgs_rosenbrock"]["fx"] = 1e-3
        self.assert_caught(mutate)

    def test_catches_an_optimizer_landing_somewhere_else(self):
        """An iterate moving past the solver's stopping noise fails.

        1e-7 relative is four orders above the measured 2.9e-11 platform noise
        and still three orders inside the Rust suite's loosest check on these
        coordinates, so the gate sits between the two as designed.
        """
        def mutate(tree):
            tree["opt_scipy"]["lbfgs_rosenbrock"]["x"][0] = 1.0 + 1e-7
        self.assert_caught(mutate)

    def test_catches_a_changed_array_length(self):
        """A grid that changes size is a changed fixture, not drift."""
        def mutate(tree):
            tree["dist_demo"]["pdf"].append(0.05)
        self.assert_caught(mutate)

    def test_catches_a_new_or_removed_key(self):
        """A field appearing or vanishing fails rather than being ignored."""
        self.assert_caught(lambda tree: tree["dist_demo"].pop("mean"))
        self.assert_caught(lambda tree: tree["dist_demo"].update(extra=1.0))

    def test_catches_a_missing_fixture(self):
        """A fixture the generators stop emitting fails.

        ``git diff --exit-code`` could not see this at all: a tracked file that
        is simply never rewritten produces no diff.
        """
        self.assert_caught(lambda tree: tree.pop("algo_lle"))

    def test_catches_an_unexpected_new_fixture(self):
        """A fixture emitted but not committed fails too."""
        def mutate(tree):
            tree["algo_brand_new"] = {"value": 1.0, "_provenance": {
                "library": "sklearn", "version": "1.9.0", "seed": None}}
        self.assert_caught(mutate)

    def test_hand_written_fixtures_are_not_expected_from_a_regeneration(self):
        """A fixture marked hand-written may be absent without failing.

        ``harness_smoke.json`` is authored by hand to smoke-test the Rust
        loader; no generator emits it. The exemption is keyed on the committed
        ``_provenance.library``, so it cannot be claimed by a fixture that
        really is generated.
        """
        comparison = self.run_gate(lambda tree: tree.pop("dist_demo"))
        self.assertTrue(comparison.violations, "a generated fixture must fail")

        hand_written = copy.deepcopy(BASELINE)
        hand_written["smoke"] = {"value": 1.5, "_provenance": {
            "library": fixture_diff.HAND_WRITTEN, "version": "n/a", "seed": None}}
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "committed", hand_written)
            _write(root / "regenerated", BASELINE)
            result = fixture_diff.compare_trees(
                root / "committed", root / "regenerated",
                fixture_diff.DEFAULT_RTOL, fixture_diff.DEFAULT_ATOL,
            )
        self.assertEqual(result.violations, [])
        self.assertEqual(result.skipped, ["smoke.json"])

    def test_catches_a_non_finite_result(self):
        """A value going NaN is a failure, not a match."""
        def mutate(tree):
            tree["dist_demo"]["variance"] = float("nan")
        self.assert_caught(mutate)

    # -- the band is where the docstring says it is -------------------------

    def test_band_is_tighter_than_every_rust_tolerance_it_guards(self):
        """The gate must stay below the tolerances the Rust suite asserts.

        ``1e-12`` absolute and relative on distribution moments
        (``tests/dist/mod.rs``) is the tightest assertion made against any
        fixture value; the band has to be strictly under it, or drift the gate
        accepts could consume a Rust test's entire error budget.
        """
        tightest_rust_tolerance = 1e-12
        self.assertLess(fixture_diff.DEFAULT_RTOL, tightest_rust_tolerance)
        self.assertLess(fixture_diff.DEFAULT_ATOL, tightest_rust_tolerance)

    def test_every_widened_policy_stays_under_its_rust_assertion(self):
        """Each policy band must be tighter than the assertion that consumes it.

        Widening a policy past the Rust tolerance it guards would make the
        fixture the weaker constraint, which is the failure mode this whole
        module exists to avoid. The ceilings:

        * trustworthiness — Rust checks ``trust >= reference - 0.05`` on values
          near ``0.92``, a slack of ``5.4e-2`` relative; and
        * ``opt_scipy`` iterates — Rust checks ``rtol 1e-4`` with ``atol`` from
          ``1e-5`` to ``1e-2`` by problem, so ``1e-4`` is the tightest ceiling.
        """
        rust_trustworthiness_slack = 0.05 / 0.92
        for key in ("algo_lle:/trustworthiness", "algo_tsne:/trustworthiness"):
            policy = fixture_diff.POLICIES[key]
            self.assertLess(policy.rtol, rust_trustworthiness_slack, key)
        iterates = fixture_diff.POLICIES["opt_scipy:/*/x[]"]
        self.assertLess(iterates.rtol, 1e-4, "opt_scipy iterate rtol")
        self.assertLess(iterates.atol, 1e-5, "opt_scipy iterate atol")

    def test_glob_policies_do_not_overlap(self):
        """Two globs matching one field would make the band order-dependent."""
        for key in fixture_diff.POLICIES:
            fixture, pointer = key.split(":", 1)
            matched, _ = fixture_diff._match_policy(f"{fixture}:{pointer}")
            self.assertEqual(matched, key)
        # A concrete pointer under a glob resolves to exactly that glob.
        matched, _ = fixture_diff._match_policy("opt_scipy:/newton_rosenbrock/x[]")
        self.assertEqual(matched, "opt_scipy:/*/x[]")

    def test_unpoliced_fields_use_the_default_band(self):
        """A policy must never leak onto a neighbouring field.

        ``opt_scipy`` carries globbed policies for ``x`` and ``fx``; ``fun_calls``
        sits beside them in the same object and has to fall through to the strict
        default band.
        """
        def mutate(tree):
            tree["opt_scipy"]["lbfgs_rosenbrock"]["fun_calls"] *= 1.0 + 1e-12
        self.assert_caught(mutate, "the fx/x policies must not leak sideways")

    def test_every_policy_documents_itself(self):
        """No exemption may be added without a written reason."""
        for key, policy in fixture_diff.POLICIES.items():
            self.assertIn(policy.kind, ("band", "permutation"), key)
            self.assertGreater(len(policy.why), 200,
                               f"{key} needs a real justification, not a label")


if __name__ == "__main__":
    unittest.main()
