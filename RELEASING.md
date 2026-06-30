# Releasing stats-claw

Maintainer runbook for cutting a release to [crates.io](https://crates.io). This file is
intentionally **not** part of the published crate (it is excluded from the `include`
whitelist in `Cargo.toml`), so it stays in the repository only.

## 1. Pre-flight — all must be green

```sh
cargo test
cargo clippy --all-targets   # manifest is authoritative: deny groups fail; pedantic/nursery only warn
cargo fmt --check
```

## 2. Verify the package without uploading

```sh
cargo publish --dry-run   # packages + compiles from the tarball; "aborting upload due to dry run" = success
cargo package --list      # exactly what ships: src/**, README.md, CHANGELOG.md, LICENSE-MIT, LICENSE-APACHE
```

The `--dry-run` "ignoring test `…`/benchmark `…`" warnings are **expected**: `tests/` and
`benches/` are intentionally excluded from the published tarball.

## 3. Bump version & changelog

- Bump `version` in `Cargo.toml`.
- Add the release entry to `CHANGELOG.md`.
- Commit.

## 4. Publish

```sh
cargo login               # one-time: paste a token from https://crates.io/settings/tokens
cargo publish
git tag v<version> && git push origin v<version>
```

## Notes

- Publishing is **permanent** — a version cannot be deleted, only withdrawn with
  `cargo yank` (reverse with `cargo yank --undo`). Ship fixes as a new version.
- The source repository can stay private; `cargo publish` uploads only the packaged crate
  (`src/**` + the docs/license files), never the repo's `tests/`, `benches/`, or `reference/`.
- The crate name on crates.io is first-come; confirm `stats-claw` is available (or claimed by
  you) before the first publish.
