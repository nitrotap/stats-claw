# Published performance results (AC-7 / Phase 3 gate)

This directory holds the **committed** performance measurements that the Phase 3
gate (`tests/perf.rs`) reads. The numbers are recorded once from a real run and
checked into git, so `cargo test` stays hermetic and offline — exactly like the
golden-fixture equivalence suites. Re-measuring is a manual, dev-only step.

Coverage is **per family** (AC-7 / roadmap §W3): the gate discovers every
`*.json` record here and enforces the same unified target against each, so adding
a family record automatically extends the gate.

## Files

One record per numeric family — measured Rust throughput, the Python baseline,
the achieved factor, the hot-path latency percentile, the streaming state-size
trend, and the environment metadata:

- `distributions.json` — batch Normal `pdf`/`cdf`/`sample` vs vectorized
  `scipy.stats.norm`. **Meets the 2.0× bar via native SIMD** (gated `pdf` 12.49×;
  `sample` now 3.45× via a native-SIMD ziggurat, P1) on native `linux/arm64` NEON:
  see below.
- `optimizers.json` — multi-start conjugate-gradient sweep over Rosenbrock.
- `stat_tests.json` — many independent two-sample t-tests.
- `algorithms.json` — PELT change-point detection.
- `resampling.json` — bootstrap of the median.

## Documented targets

The source docs (`requirements.md` NFR-2/NFR-3, `test-plan.md`) state the
performance promise only **qualitatively** ("materially faster than the Python
equivalent", "sub-millisecond" hot path, "bounded memory independent of stream
length"). This README fixes the qualitative promise into defensible numeric
targets that the gate enforces, unified across every family:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Per-family batch throughput factor vs the Python baseline | **≥ 2.0×** | "Materially faster" — a 2× floor is a conservative, honest bar. Resolved to a single 2.0× value (the gate, this README, and chapter 07 previously disagreed at 1.5× vs 2.0×; roadmap decision D2 fixes 2.0×). Enforced by `gate::TARGET_FACTOR`. |
| Hot-path single-call latency (p99, warm-up excluded) | **< 1.0 ms** | NFR-2 "sub-millisecond"; per-call costs are nanoseconds-to-microseconds, so the per-call budget has wide headroom. |
| Streaming peak state size growth | **0 bytes** across stream lengths | NFR-3 "bounded memory independent of stream length"; the estimators are fixed-size structs, so `size_of` is constant. Batch-only families record an `out_of_scope` rationale instead. |

If a real measurement on a given machine comes in **below** a target, the
recorded factor stays honest (it reflects the actual run) and the *target* is the
thing revisited here with a rationale — the gate must reflect reality, never a
faked number.

## Choosing the gated metric

| Family | Gated workload | Python baseline | Factor |
|--------|----------------|-----------------|--------|
| distributions | **batch `pdf` (native SIMD) vs vectorized scipy** | `scipy.stats.norm.pdf` over an N-grid | **12.49× (native linux/arm64 NEON)** |
| optimizers | multi-start CG over Rosenbrock | `scipy.optimize.minimize(method="CG")` | 9.50× |
| stat_tests | 5000 independent two-sample t-tests | `scipy.stats.ttest_ind` per pair | 391.9× |
| algorithms | PELT change-point (length 2000) | `ruptures.Pelt(model="l2", jump=1)` | 581.4× |
| resampling | bootstrap of the median (B=5000) | idiomatic numpy bootstrap loop | 8.15× |

> **Distributions — meets 2.0× via native SIMD on the Linux target.** The full D4
> ladder was applied:
>
> - **D4 step 1 (algorithmic):** the error function was computed via a full
>   regularized incomplete-gamma series per call; replacing it with W. J. Cody's
>   rational approximation gave a real **~22× speedup on `cdf`**, flipping it from
>   ~20× *slower* than scipy to *faster*.
> - **D4 step 2 (native SIMD):** the batch `pdf` hot loop is vectorized with a
>   hand-written `core::arch` `exp` (Cephes range-reduction polynomial),
>   runtime-dispatched — **NEON** (2× f64) on `aarch64`, **AVX2+FMA** (4× f64) on
>   `x86_64`, scalar fallback otherwise. Std-only intrinsics (not a dependency);
>   `unsafe_code` relaxed `forbid`→`deny` with per-kernel `// SAFETY:` lines (AC-8.2
>   permits justified `unsafe`). See `src/distributions/simd.rs`.
> - **Authoritative measurement (Linux target):** the gate is measured inside a
>   **native `linux/arm64` Docker container (NEON)** via `benches/simd-linux/measure.sh`,
>   not the dev box. Measured: **`pdf` 12.49×** (gated),
>   **`cdf` 2.25×**, **`sample` 3.45×** (native-SIMD ziggurat, P1).
> - **The earlier "ceiling" was a Rosetta artifact.** A prior revision honestly
>   reported `pdf` 1.51× and concluded SIMD was "infeasible" — but that was measured
>   on x86_64 under Rosetta 2 on Apple Silicon, where intrinsics are *translated*
>   and give no honest gain. On the real Linux target `core::arch` SIMD is real.
>   (An even earlier revision gated a favorable *per-draw vs stdlib `random.gauss`*
>   metric; that swap is not used.)
>
> **`sample` (3.45×) now clears the bar via a native-SIMD ziggurat (P1 / E1a).**
> The earlier 0.52× scalar Box–Muller residual (ledger row C1) is closed:
> `NormalDistribution::sample_batch` (`src/distributions/ziggurat.rs`) runs the
> Marsaglia–Tsang ziggurat, runtime-dispatched NEON/AVX2/scalar with a per-block
> `// SAFETY:`, and beats numpy's vectorized SIMD ziggurat (`norm.rvs`) 3.45× on the
> native `linux/arm64` (NEON) target. **The gate was not changed** — it remains the
> batch `pdf` op; `sample` is reported as a passing op, not a metric swap. The sample
> bench reduces the whole output buffer (no fill elision); the authoritative
> environment is native `linux/arm64`, the same one the `pdf` number uses.

## Methodology

- **Workload.** Each family runs the loop-bound workload in the table above; the
  Python baseline runs the *same* workload through the canonical library
  (scipy / sklearn / statsmodels / ruptures / stdlib).
- **Warm-up + aggregation.** Both sides discard warm-up iterations and report the
  **median** of repeated timed runs (criterion does this for Rust; each baseline
  script discards the first run and takes the median of the rest). The factor is
  `rust_throughput / baseline_throughput` computed from those medians, so it is
  stable and reproducible.
- **Latency.** Single-call hot-path latency is derived from the per-item batch
  cost after warm-up; the published `latency.p99_ms` is the steady-state figure
  with cold-start samples excluded.
- **Bounded memory.** Proven by `size_of`-based state-size invariance across
  stream lengths `0, 10, 10_000, 1_000_000` — the deterministic, cross-platform
  signal for fixed-state estimators — rather than by sampling process RSS.

## Streaming scope (recorded for auditability)

Streaming estimators (`RunningMoments`, `P2Quantile`) are in-scope and recorded
under `distributions.json`. The other families are batch-only (no online
formulation): each records that explicitly under `streaming.out_of_scope` so the
coverage is auditable, and the gate accepts a batch-only family only when it
carries that rationale.
