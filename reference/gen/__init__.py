"""Golden-fixture generators for the stats-claw equivalence harness.

Python runs here ONLY at generation time (`make fixtures`); ``cargo test`` reads
the committed JSON under ``reference/golden/`` offline and never invokes Python.
"""
