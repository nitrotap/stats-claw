"""Shared helpers for golden-fixture generators.

Python runs here ONLY at generation time; ``cargo test`` reads the committed JSON
offline. Every fixture carries a ``_provenance`` block recording which reference
library and version produced it, so a future drift check can detect when a pinned
dependency changes its output.

Set ``FIXTURE_GOLDEN_DIR`` to redirect the emitted fixtures somewhere other than
the committed tree. The drift check (``make fixture-drift``) uses it to
regenerate into a scratch directory and compare numerically, leaving the working
tree untouched; see :mod:`gen.fixture_diff`.
"""
import json
import os
import pathlib

#: Directory the Rust loader reads from: ``reference/golden`` (sibling of ``gen``).
DEFAULT_GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "golden"

#: Where this run writes. Defaults to :data:`DEFAULT_GOLDEN`; overridden by
#: ``FIXTURE_GOLDEN_DIR`` so a drift check can regenerate out-of-tree.
GOLDEN = pathlib.Path(os.environ.get("FIXTURE_GOLDEN_DIR") or DEFAULT_GOLDEN)


def write_fixture(name, payload, *, library, version, seed=None):
    """Write ``payload`` to ``golden/<name>.json`` with a provenance stamp.

    Args:
        name: fixture basename (no extension); becomes ``<name>.json``.
        payload: JSON-serialisable dict of reference values.
        library: reference library that produced the values (e.g. ``scipy.stats``).
        version: that library's version string, for drift detection.
        seed: optional RNG seed recorded for sampling fixtures.
    """
    GOLDEN.mkdir(parents=True, exist_ok=True)
    doc = dict(payload)
    doc["_provenance"] = {"library": library, "version": version, "seed": seed}
    path = GOLDEN / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
