# Golden-fixture drift: what actually moves, and by how much

The scheduled `golden-fixture drift` job regenerates every fixture in
`reference/golden` from the pinned Python stack and checks it
against the committed tree. This note records the measurements the job's
tolerances are derived from, so the numbers in
`reference/gen/fixture_diff.py` can be checked rather than
trusted.

## Why the gate is numeric

The job used to end in:

```yaml
- run: git diff --exit-code -- reference/golden
```

which asserts that IEEE-754 doubles are bit-identical across machines. They are
not, for two independent reasons:

* the BLAS/LAPACK-backed paths (`PCA`, `FastICA`, `FactorAnalysis`,
  `LocallyLinearEmbedding`, `LinearRegression`, `Ridge`) use blocked, threaded
  reductions, so the summation order — and therefore the last bits — depends on
  the CPU instruction set, the thread count, and which BLAS the wheel was built
  against; and
* the `scipy.special` kernels behind the distribution and test fixtures are
  sub-ULP accurate but not correctly rounded, so a different libm rounds the
  final digit differently.

This is the same mistake, for the same reason, that
`src/resampling/monte_carlo/tests.rs` already corrected for
`standard_normal_mean_matches_theory`: a bit-exact pin is true only on the
machine that recorded it, and belongs only where bit-exactness is a guarantee the
producer can keep. The fix follows that precedent — tolerances derived from each
quantity's own error, exactness kept where exactness is real (every integer,
string, key, array length, and `_provenance` stamp is still compared exactly).

The committed JSON is **not** rounded or rewritten. The Rust suite keeps
asserting against the exact recorded bits; only the drift job's comparison
changed.

## The two measurements

Both compare the committed fixtures against a full regeneration on the pinned
stack (`numpy 2.4.4`, `scipy 1.17.1`, `scikit-learn 1.9.0`, `statsmodels 0.14.6`,
`ruptures 1.1.9`, `mlxtend 0.24.0`, `pandas 3.0.2`).

| | **A — GitHub runner** | **B — local** |
|---|---|---|
| Platform | `ubuntu-24.04`, x86-64 | macOS, aarch64 |
| BLAS | OpenBLAS (manylinux wheels) | Accelerate |
| Python | 3.12.13 | 3.12 |
| Fixtures differing | 14 of 63 | 36 of 63 |
| Lines differing | 514 | 1737 |

Measurement A is the diff printed by this job's last byte-gate failure
([run 30683858643](https://github.com/nitrotap/stats-claw/actions/runs/30683858643),
2026-08-01) and confirmed by a passing numeric run of the same job in the
upstream `yee-claw` repository, whose `reference/golden` tree is byte-identical
to this one for all 14 affected files. This gate's own report on
[run 30728434836](https://github.com/nitrotap/stats-claw/actions/runs/30728434836)
gives the same numbers: 514 floats moved, of which 504 sit under the default band
using at most 3.4% of it, and 10 fall under a documented policy. Measurement B is
a local regeneration.

Regenerating **twice on the same machine** produces byte-identical output, so all
of this is cross-machine variation, not run-to-run nondeterminism.

### Pinning the thread count does not fix it

The obvious cheaper fix is to force the BLAS to one thread, since thread count
changes reduction order. It was tried and it does not work. Regenerating with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`
moves exactly **one** value against the same machine's default-threaded run —
`algo_kmeans_blobs/inertia`, by 2 ULP — and leaves the comparison against the
committed tree at all 36 files and 1737 lines. The drift is dominated by *which*
BLAS and libm are linked, not by how many threads they use, and that cannot be
pinned by an environment variable. Short of vendoring a fixed BLAS build, a
numeric comparison is the only thing that makes this gate meaningful.

Note also that neither measured environment reproduces the committed fixtures
exactly, so the machine that recorded them is a third one. `_provenance` records
library versions but not platform, BLAS, or thread count; recording those would
make future drift attributable to a cause rather than merely visible.

## Is any of it a real change?

Mostly no — and in three places, yes. The claim that all 514 lines are 1–7 ULP
last-digit noise does not survive measurement.

**Last-digit noise (the large majority).** Every distribution, statistical-test,
regression, PCA, factor-analysis and feature-selection value moves by at most
`9.7e-14` relative on the runner and `1.4e-14` on aarch64 — a few hundred ULP at
worst, most of them 1–5 ULP. Worst offenders on the runner:

| Quantity | Committed | Regenerated | Relative |
|---|---|---|---|
| `algo_ica/sources[52][0]` | `0.012038356995753808` | `0.012038356995754977` | `9.7e-14` |
| `test_anova_oneway/p_value` | `1.4416576847147908e-07` | `1.4416576847148085e-07` | `1.2e-14` |
| `regression_ols/intercept` | `0.5497738476936007` | `0.5497738476935954` | `9.7e-15` |
| `algo_pca/components[0][1]` | `0.04391298869027316` | `0.043912988690272825` | `7.6e-15` |
| `algo_factor_analysis/reconstruction_error` | `0.5597721343937888` | `0.5597721343937881` | `1.2e-15` |

**Not last-digit noise.** Three quantities move far past rounding. None is an
upstream regression — the pinned versions are identical on both sides — but each
is a genuine platform-dependent difference in a reference answer, not a rounding
artefact, and each needed a stated policy rather than a wider global tolerance.

1. **`algo_lle/trustworthiness`: `0.9205769230769231` → `0.9291666666666667`**,
   `+9.2e-3` relative. Trustworthiness is a rank statistic over 5-nearest-
   neighbour sets with quantum `1 / (n k (2n - 3k - 1)) = 1/15600` here, so this
   is a move of **134 quanta** — a substantially different LLE embedding, not a
   rounded one. LLE takes its embedding from the null space of a near-singular
   matrix, whose basis genuinely differs between LAPACK builds. On aarch64 the
   same value moves `+83` quanta. Worth knowing about; not a defect.
2. **`opt_scipy/newton_rosenbrock/x`: `2.9e-11` relative** (258 729 ULP). Where
   an iterative optimiser stops is not a mathematically determined point — it is
   wherever the convergence test tripped. Near a minimum the objective is flat,
   so an `O(eps)` change in the gradient displaces the iterate by
   `O(eps / lambda_min)`; on Rosenbrock's valley that is about five orders of
   amplification.
3. **`opt_scipy/lbfgs_quadratic/fx`: `1.97e-30` → `1.97e-31`**, a whole decade,
   `0.9` relative. Both numbers mean "zero": this is the objective *at* a
   minimiser, and its digits are the solver's noise floor rather than a
   measurement. The absolute change is `1.8e-30`.

**Also found, unrelated to floating point.** `algo_spectral_blobs/labels` is
unchanged on the runner but wholly relabelled (`1 <-> 2`) on aarch64.
`SpectralClustering` takes cluster identity from the sign and ordering of
Laplacian eigenvectors, which are defined only up to a permutation. The partition
is the answer; the integer names are not.

## The tolerances

Default band, applied to every float without a policy:

```
|committed - regenerated| <= 1e-13 + 1e-13 * |committed|
```

Chosen between two independent bounds:

* **Floor** — the widest measured deviation on any unpoliced quantity uses
  **3.4 %** of this band on the runner and **4.1 %** on aarch64, so there is
  roughly 25x headroom against observed platform noise.
* **Ceiling** — the tightest tolerance the Rust suite asserts against any fixture
  value is `atol = rtol = 1e-12` on distribution `mean`/`variance`
  (`tests/dist/mod.rs`), which for Student's t (`mean = 0`) is a pure `1e-12`
  absolute bound. The gate sits an order of magnitude below it, so drift the gate
  accepts can never consume more than a tenth of a Rust test's budget. Every
  other consumed tolerance is `1e-12` (association metrics) to `1e-6`
  (decomposition), where the margin is 10 to 10 000 000x.

Four quantities carry a stated policy. Each is widened only to what its own
definition requires, and each stays tighter than the Rust assertion that consumes
it; the full rationale for each lives beside it in `POLICIES` in
`reference/gen/fixture_diff.py`.

| Field | Policy | Measured drift | Rust assertion it guards | Margin |
|---|---|---|---|---|
| `algo_{lle,tsne}:/trustworthiness` | `rtol 2.5e-2` | `+9.2e-3` (A), `+5.8e-3` / `-1.7e-3` (B) | `trust >= reference - 0.05`, i.e. `5.4e-2` relative | 2.7x above noise, 2.2x under the assertion |
| `opt_scipy:/*/x[]` | `rtol 1e-8`, `atol 1e-8` | `2.9e-11` (A), `1.4e-14` (B) | `rtol 1e-4`, `atol 1e-5`…`1e-2` | 344x above noise, 10 000x under |
| `opt_scipy:/*/fx` | `atol 1e-8`, `rtol 0` | `1.8e-30` absolute (A) | none — no Rust test reads `fx` | catches non-convergence (`O(1e-3)`) with 5 orders to spare |
| `algo_spectral_blobs:/labels[]` | equal up to a relabelling | full `1 <-> 2` swap (B) | ARI `>= 0.99` (permutation-invariant) | gate is *stricter*: ARI tolerates a few misplaced points, this tolerates none |

### An honest caveat on trustworthiness

Platform noise on `algo_lle/trustworthiness` (0.86 percentage points) already
spends **17 %** of the 0.05 slack the Rust suite allows. So this gate can catch a
collapse in sklearn's embedding quality — tens of percent — but cannot resolve a
one-percent upstream regression. The structural fix is to pin a *floor* rather
than a value, which `algo_umap` already does with `trustworthiness_target`;
changing `algo_lle` and `algo_tsne` to match would need a matching change to
their Rust assertions and is left as follow-up.

## What the gate still catches

* **Any** change to a non-float: every integer, boolean, string, object key,
  array length, and the `_provenance` `library` / `version` / `seed` of every
  fixture. A version bump that changes an answer is reported by name.
* Any float moving more than `1e-13` relative (about 900 ULP at unit magnitude,
  falling to ~700 ULP where the relative term dominates), or `1e-13` absolute
  for values below that scale — for all 8 046
  floats outside the four policies.
* A generator that stops emitting a fixture, or emits one that is not committed.
  The byte gate structurally could not see this: `git diff` says nothing about a
  tracked file that was never rewritten.
* A change to the *partition* found by spectral clustering, as opposed to a
  renaming of its clusters.
* An optimiser that stops converging, or lands more than `1e-8` from where it
  used to.
* Trustworthiness collapsing by more than 2.5 %.

The gate is itself fault-injection tested by
`reference/gen/test_fixture_diff.py` (26 cases, stdlib only), which runs in the
`gates` job on every push and PR **and** inside the drift job itself — so the run
that reports "no drift" has just demonstrated, on the same machine, that it would
have failed on a real change.

## Note on `requirements.txt`

The pins are exact and complete. The consequence of exact pins is worth stating plainly: **this job does not
discover new upstream releases.** Upstream can only move when someone edits
`requirements.txt`. What the job verifies is that the *pinned* stack still
produces the committed answers on a machine that is not the one that recorded
them — that is, it is a portability check on the fixtures plus a tripwire on
deliberate version bumps. Making it an early-warning system for upstream change
instead would mean floating the pins (or adding a second, unpinned job), and
would trade reproducibility of `make fixtures` for that warning.

## History

The job failed every scheduled run from 2026-06-30 onward and had never been
green, always on the same 514-line, 14-file byte diff analysed above.

The same job in the upstream `yee-claw` repository failed 33 of 33 runs for a
different reason — its `requirements.txt` omitted `mlxtend`, so every run died at
`ModuleNotFoundError` before reaching the comparison at all. This repository's
`requirements.txt` already listed `mlxtend`, which is why it got as far as the
diff and exposed the byte-equality defect. The numeric gate landed upstream first
([yee-claw#5](https://github.com/nitrotap/yee-claw/pull/5)); this change ports it
here. The `reference/` tree is `preserve`d by the carve, so it does not arrive
automatically.
