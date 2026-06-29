#!/usr/bin/env bash
# Host-side driver for the authoritative Linux/NEON distributions measurement.
#
# Builds the native linux/arm64 image and runs run.sh inside it with the repo
# bind-mounted, so the criterion bench (NEON SIMD pdf path) and the vectorized
# scipy baseline are both measured on real ARM Linux. Prints run.sh's JSON
# summary (rust/scipy throughputs + factors) to stdout; everything else to stderr.
#
# Usage (from the repo root):
#   stats-claw/benches/simd-linux/measure.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = four levels up from this script (benches/simd-linux -> crates/stats-claw -> crates -> root).
REPO="$(cd "$HERE/../../../.." && pwd)"
IMAGE=stats-claw-simd-bench

echo "== building native linux/arm64 image ==" >&2
docker build --platform linux/arm64 -t "$IMAGE" -f "$HERE/Dockerfile" "$HERE" >&2

echo "== running measurement in native linux/arm64 container ==" >&2
# Mount the repo read-write at /work (cargo needs to write target/). A named
# volume isolates the container's target dir from the host's macOS build.
docker run --rm --platform linux/arm64 \
    -v "$REPO":/work \
    -v stats-claw-simd-target:/work/target \
    "$IMAGE" \
    /work/stats-claw/benches/simd-linux/run.sh
