#!/usr/bin/env bash
# Runs INSIDE the native linux/arm64 container (see measure.sh / Dockerfile).
#
# 1. Builds stats-claw in release and runs the distributions criterion bench (the
#    `normal_batch` group: pdf / cdf / sample), which exercises the NEON SIMD
#    `pdf_batch` path on real ARM Linux.
# 2. Runs the vectorized scipy baseline (benches/gen/baseline.py).
# 3. Computes the genuine batch throughput factors (rust / scipy) per op and
#    prints a JSON summary to stdout.
#
# No fabrication: every number comes from a measured run in this invocation.
set -euo pipefail

REPO=/work
CRATE="$REPO/crates/stats-claw"
N=100000

echo "== environment ==" >&2
uname -a >&2
rustc --version >&2
python3 -c "import platform,numpy,scipy;print('python',platform.python_version(),'numpy',numpy.__version__,'scipy',scipy.__version__)" >&2

cd "$REPO"

echo "== building + running criterion distributions bench (NEON) ==" >&2
# A short measurement is plenty for a throughput ratio; keep it bounded.
cargo bench --bench distributions -- --warm-up-time 1 --measurement-time 5 normal_batch >&2

# Criterion writes nanoseconds-per-iteration; each iteration processes N elements.
est() {
    python3 - "$1" "$N" <<'PY'
import json, sys
path, n = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    mean_ns = json.load(f)["mean"]["point_estimate"]
print(n / (mean_ns * 1e-9))  # elements per second
PY
}

CRIT="$REPO/target/criterion/normal_batch"
RUST_PDF=$(est "$CRIT/pdf/new/estimates.json")
RUST_CDF=$(est "$CRIT/cdf/new/estimates.json")
RUST_SAMPLE=$(est "$CRIT/sample/new/estimates.json")

echo "== running vectorized scipy baseline ==" >&2
BASELINE_JSON=$(python3 "$CRATE/benches/gen/baseline.py")

# Combine: emit the per-op rust/scipy throughputs and factors as JSON on stdout.
python3 - "$RUST_PDF" "$RUST_CDF" "$RUST_SAMPLE" "$N" <<PY
import json, sys
rust = {"pdf": float(sys.argv[1]), "cdf": float(sys.argv[2]), "sample": float(sys.argv[3])}
n = int(sys.argv[4])
baseline = json.loads('''$BASELINE_JSON''')
b = baseline["baseline_throughput_elems_per_sec"]
out = {"n": n, "rust_throughput_elems_per_sec": rust,
       "baseline_throughput_elems_per_sec": b,
       "factors": {k: rust[k] / b[k] for k in rust},
       "env": baseline["env"]}
print(json.dumps(out, indent=2))
PY
