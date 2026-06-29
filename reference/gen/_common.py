"""Shared helpers for golden-fixture generators.

Python runs here ONLY at generation time; ``cargo test`` reads the committed JSON
offline. Every fixture carries a ``_provenance`` block recording which reference
library and version produced it, so a future drift check can detect when a pinned
dependency changes its output.
"""
import json
import pathlib

#: Directory the Rust loader reads from: ``reference/golden`` (sibling of ``gen``).
GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "golden"


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
