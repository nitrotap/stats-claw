# Formal verification

`stats-claw` carries **68 [Kani](https://model-checking.github.io/kani/) proof
harnesses** alongside its test suite. This file states exactly what they prove,
what they do not, and how to reproduce them.

## What Kani is, and how this differs from testing

Kani is a bit-precise model checker for Rust. It compiles a function together
with an assertion into a logical formula and hands it to CBMC, an SMT-backed
solver, which searches for *any* assignment of the inputs that violates the
assertion. If none exists, the property holds for every input in the modelled
domain; if one does, Kani reports it as a concrete counterexample.

The practical difference from a unit test:

| | Test | Proof harness |
|---|---|---|
| Inputs | the ones you wrote down | every value of the declared type |
| A pass means | those inputs behaved | no input in the domain misbehaves |
| A failure gives you | a red assertion | a concrete counterexample |

A test that samples 10,000 RNG states tells you those 10,000 states were fine.
`rng_next_f64_in_unit_interval` tells you that *all 2^64* generator states yield a
value in `[0, 1)`. That is the class of statement worth making about a numerics
crate whose failure mode is a panic on the hot path.

One caveat to hold onto throughout: CBMC is a *bounded* model checker. Scalar
inputs (`u64` states, hash words, `f64` parameters) are genuinely symbolic over
their whole type. Properties parameterised by a *collection size* are discharged
at small sizes — the size bound is stated for each one below, and it is not
sound to generalise past it by inspection.

The harnesses live in `#[cfg(kani)] mod verification` blocks co-located with the
code they verify. `cfg(kani)` is registered in `Cargo.toml`'s `[lints.rust]`, so
these blocks are invisible to `cargo build`, `cargo test`, and `cargo clippy`,
compile to nothing in a normal build, and add no dependency — `stats-claw` still
has zero runtime dependencies. Modules near the repository's 500-line style cap
keep their harness bodies in a sibling file — three pulled in with `include!`
(`ziggurat_verification.rs`, `feature_selection/verification.rs`,
`two_way_anova_verification.rs`) and one as a submodule
(`resampling/monte_carlo/verification.rs`).

## Running the proofs

```sh
cargo install --locked kani-verifier
cargo kani setup            # one-time: fetches CBMC and Kani's pinned nightly

cargo kani -Z stubbing      # the full suite
```

`-Z stubbing` is required. The `ln_choose` proof is *modular*: it stubs
`ln_gamma`, whose transcendental body CBMC cannot model, and proves the
surrounding control flow panic-free independently of it.

To work on one property at a time:

```sh
cargo kani -Z stubbing --harness resampling_uniform_index_in_bounds
```

Run harnesses one per invocation when targeting subsets — several `--harness`
flags solve in parallel and compete for memory.

In CI the suite runs as the `kani` job in `.github/workflows/ci.yml`: nightly, on
manual dispatch, and on every `v*` release tag, where the `publish` job waits on
it. It is deliberately not on every push — model checking takes tens of minutes
where the other gates take seconds.

### Targets the suite is discharged on

A proof is a claim about this crate's code, not about a machine. *Discharging*
one is not platform-neutral: a harness links the host's `std`, and CBMC
model-checks whatever that `std` does, so a harness can terminate on one target
and not on another without anything about the crate changing. Which targets the
suite is actually discharged on therefore belongs in this file, and is recorded
here rather than assumed.

| Target | Result | `cargo kani -Z stubbing` |
|---|---|---|
| `x86_64-unknown-linux-gnu` (GitHub Actions `ubuntu-latest`) | 68 verified, 0 failures | 9m03s |
| `aarch64-apple-darwin` | 68 verified, 0 failures | 6m28s |

`x86_64-unknown-linux-gnu` is the CI target, so every scheduled run, manual
dispatch, and `v*` tag re-establishes that row. Both targets discharge the same
68 harnesses with the same results: there is no platform-conditional harness, no
target-specific stub, and no bound that is loosened on one and not the other.
The timings are whole-command wall clock, compilation included.

## What is proven

68 harnesses across 43 modules. Grouped by the layer they cover:

### Core primitives — 14

| Module | Property proved |
|---|---|
| `rng` (4) | `next_u64` is panic- and overflow-free from **every** generator state; `u64_to_f64` widens faithfully (finite, non-negative, bounded) across the whole `u64` range; `next_f64()` lands in `[0, 1)` for every state |
| `streaming::moments` (2) | the Welford update never panics or overflows and `variance()` is non-negative (never `NaN`) for finite inputs; after a single update the variance is exactly `0.0` |
| `streaming::p2` (1) | over a symbolic bootstrap-plus-update sequence the P² marker `positions` stay strictly ascending — which formally discharges the module's scoped `allow(indexing_slicing)` |
| `distributions::ziggurat` (4) | `unpack` yields in-range layer/j/sign fields for every `u64` word; every derived index into the 128-entry `ZIG_K`/`ZIG_W`/`ZIG_F` tables is in bounds; the scalar fast path stays finite; the wedge fixup is panic-free |
| `distributions::symmetric::normal` (1) | `pdf` is panic-, overflow- and UB-free over the valid parameter box |
| `special::gamma` (2) | `usize_to_f64` is panic-/overflow-free and non-negative for every `usize`; `ln_choose`'s control flow is panic-free for fully symbolic `n, k` (the modular stubbed proof) |

### Algorithms — 17

Covering the shared `algorithms` primitives and nine of the ten algorithm
submodules — `clustering` is the exception, and is covered by tests instead
because its relabelling step iterates a `HashMap` (see the limits below).

The properties: the widening and distance primitives
(`count_to_f64`, `euclidean_sq`); HyperLogLog cardinality — hash finalizer total,
leading-zero rank in range, and register index `< 2^p` for every hash word at
every precision `p`; outlier `floor_to_usize` exactly floors, and its validation
rejects degenerate input; feature-selection matrix validation, `top_k_mask`
selecting exactly `min(k, n)` features, and population variance non-negative;
the regression 2×2 solver **total** (either `Ok` with two roots or `Singular` —
never a panic); decomposition covariance total; association and classification
shape validation; classification `argmax` in bounds; and the PELT L2 segment-cost
kernel total.

### Resampling — 15

The strongest results in the crate, because this is where an off-by-one becomes
an index panic in a caller's hot loop. The RNG state is fully symbolic (all 2^64)
in each; collection sizes are bounded as noted:

- `uniform_index` returns a value in `0..n` for **every** generator state, at
  every size `n ∈ 1..=5`.
- A generated permutation is a **bijection** of `0..N` — no element lost, none
  duplicated — at `N = 4`.
- k-fold test sets **partition** `0..N`: every observation appears in exactly one
  fold. At `N = 4`, `k = 2`.
- Bootstrap indices are in bounds (`N = 3`, `B = 2`).
- Jackknife and leave-one-out folds partition correctly, with fold *i* omitting
  exactly observation *i* (`n = 3`).
- Sizes too small to be meaningful are rejected with `InsufficientData` /
  `InvalidInput` rather than proceeding. The leave-one-out and jackknife guards
  take a symbolic `n` under `assume(n < 2)`; the stratified `k < 2` guard
  enumerates `k` over `{0, 1}` instead, which is the complete set that
  `k: usize` with `k < 2` denotes, so it covers the same domain — see the note on
  `kani::assume` under *What is not proven* for why it is written that way.
- The Phipson–Smyth corrected Monte-Carlo p-value stays in `(0, 1]` for symbolic
  finite null draws.

The small size bounds are a solver constraint, not a claim about typical inputs:
the index arithmetic under proof does not branch on magnitude, so a defect would
almost certainly show at these sizes — but "almost certainly" is the honest
strength of the size-parameterised results, whereas the scalar-domain results
(`rng`, ziggurat, cardinality) are exhaustive.

### Optimizers, error, and hypothesis tests — 22

The optimizer step and norm primitives are panic- and overflow-free on symbolic
finite state; `error::Error`'s `Display` is total over its unit variants; and 17
input-validation proofs cover five of the six `tests_stat` module directories —
parametric, categorical, correlation, nonparametric, goodness-of-fit; the
combinatorial `exact` module has no harness. Each proves that for
symbolic malformed input the validation path returns `Err` at its leading
structural guard with zero loop iterations executed — so a caller cannot reach
the numeric body through a shape the function does not accept.

## What is *not* proven

These limits are real and are documented in-code at each site. Read them before
treating the list above as broader than it is.

- **Transcendentals are over-approximated.** Kani does not model `exp`, `ln`, or
  `erf` precisely, so value-range properties *through* them (`erf ∈ [-1, 1]`,
  `pdf ≥ 0`) fail spuriously. Numerical accuracy is owned by the equivalence
  suite instead: golden fixtures generated from `scipy` / `scikit-learn`, checked
  to documented tolerances. **The proofs are about totality and memory safety,
  not about numerical correctness.**
- **`Vec`-growing functions do not converge.** CBMC models capacity-doubling
  reallocation, and does not terminate on functions that grow a `Vec` by `push`
  — the PELT dynamic-programming body, `gradient_descent` with a
  `Vec`-returning objective, `cross_validate`'s delegation. Those stay with the
  unit and integration suites.
- **Rank helpers and hash-map paths** (`mid_ranks`, `tie_correction`, the
  clustering relabel) loop off `Vec`-derived lengths or `HashMap` iteration and
  are likewise unit-test covered.
- **A `kani::assume` does not stop CBMC from executing the excluded path.**
  Symbolic execution walks both sides of a branch it cannot fold at symex time;
  the assumption prunes the impossible side later, at the solver. A harness must
  therefore keep its *unreachable* branches tractable too, not only the reachable
  ones — and what counts as tractable depends on the host `std`. Concretely: a
  branch that constructs a `HashMap` seeds a `RandomState` from OS entropy, which
  on Linux is `std::sys::random::linux::getrandom`, a retry loop whose trip count
  depends on a foreign call CBMC cannot model and so unwinds without bound. This
  is why `resampling_stratified_rejects_small_k` enumerates `k` over `{0, 1}` as
  a const parameter rather than drawing it with `kani::any()` under
  `assume(k < 2)`: `{0, 1}` is the complete domain of `k: usize` with `k < 2`, so
  the guard folds at symex time and the `k >= 2` branch of
  `stratified_kfold_indices` — the one holding the `HashMap` — is never executed.
  **The property proved is the same either way**: the same rejection, over the
  same set of `k`, with the generator state fully symbolic across all 2^64
  values. The sibling `resampling_loo_rejects_small_n` and
  `resampling_jackknife_rejects_small_n` keep the symbolic-plus-`assume` form,
  which is sound for them because their guarded branches reach no `HashMap`.
- **Unbounded rejection loops** (the ziggurat tail) are verified per iteration,
  not as a whole loop.
- **Collection sizes are bounded**, as listed per property above. CBMC is a
  bounded model checker; only scalar input domains are covered exhaustively.
- Coverage is per-function and per-property. A module appearing in the table
  above does not mean the module is verified end to end.

## Miri — the undefined-behaviour backstop

Kani cannot model the `core::arch` SIMD intrinsics, which is exactly where the
crate's `unsafe` lives. Miri covers that gap dynamically: it interprets real test
executions and traps undefined behaviour — out-of-bounds access, invalid values,
aliasing violations, misaligned reads.

```sh
cargo +nightly miri test --lib -- distributions:: streaming:: \
  --skip batch_pdf_matches \
  --skip scalar_ziggurat_fits_standard_normal
```

Those two module trees contain **all** of the crate's `unsafe` (the SIMD batch
paths, the ziggurat sampler, the streaming estimators). Last run for this
release: **35 passed, 0 failed, no undefined behaviour.**

The two skips are tool limitations, not defects:

- `batch_pdf_matches` — two SIMD pdf tests use a NEON intrinsic Miri cannot
  interpret. That path stays covered by `tests/gates.rs` and the equivalence
  suite.
- `scalar_ziggurat_fits_standard_normal` — draws and sorts 100k samples, which
  takes roughly a quarter of an hour under the interpreter and is OOM-killed on a
  loaded machine. Drop the skip and run it on a quiet machine for the full 36.

Miri is not wired into CI; it is a local pre-release check.

## Honest summary

All of the crate's `unsafe` lives in three files (`distributions/simd.rs`,
`distributions/ziggurat.rs`, `streaming/p2.rs`), each block carrying a `// SAFETY:`
justification that `tests/gates.rs` enforces, and all three are interpreted under
Miri — bar the one NEON test Miri cannot execute at all.
68 harnesses prove panic-freedom, index-bounds, and structural invariants —
exhaustively over scalar domains, at bounded sizes over collections — for the
primitives most likely to fault at runtime. What
they do **not** establish is numerical accuracy — that claim rests on the
equivalence suite and its golden fixtures, and is made separately in the README.
