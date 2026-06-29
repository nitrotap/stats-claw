# Authoritative Linux measurement environment for the distributions SIMD throughput gate.
#
# The deployment target is Linux; the dev box is Apple Silicon under macOS. This
# image runs as a NATIVE `linux/arm64` container (Docker `--platform linux/arm64`
# on an Apple-Silicon host runs ARM Linux natively — no Rosetta), so the NEON
# `core::arch` intrinsics in `distributions/simd.rs` execute on real ARM Linux and
# the measured pdf/cdf factors are honest. Build with:
#
#   docker build --platform linux/arm64 -t stats-claw-simd-bench -f Dockerfile .
#
# then run `run.sh` inside it with the repo mounted (see measure.sh).
FROM --platform=linux/arm64 rust:1-bookworm

# scipy/numpy for the vectorized baseline; build tools for any native wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-numpy python3-scipy ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
# Source is bind-mounted at run time (measure.sh), so nothing is COPY'd here; the
# image is just the toolchain + scipy. The bench is built fresh against the mount.
ENTRYPOINT ["/bin/bash"]
