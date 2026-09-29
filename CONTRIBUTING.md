# Contributing to lateweave

Thanks for taking the time. Bug reports, fixes, and cookbook recipes are all
welcome; for anything that changes a contract (`CandidateGenerator`,
`Reranker`, `MultiVectorSource`, the [store format](STORE_FORMAT.md)), please
open an issue first so the design can be discussed before the code.
[ARCHITECTURE.md](ARCHITECTURE.md) explains the ownership rules those contracts
rest on.

## Development setup

You need a stable Rust toolchain (the minimum supported version is the
`rust-version` in `Cargo.toml`) and Python 3.11 or later.
[uv](https://docs.astral.sh/uv/) is the easiest way to get the Python side:

```bash
git clone https://github.com/pau-mensa/lateweave
cd lateweave

# Rust: the library, its tests, and the example.
cargo test

# Python: build the extension into a virtualenv and run the suite. The cookbook
# tests need the recipe's own dependencies, declared in its PEP 723 header.
uv venv
uv pip install -e . pytest -r cookbook/bm25_stored_maxsim.py
.venv/bin/python -m pytest
```

After changing Rust code, reinstall (`uv pip install -e .`) so the Python tests
see the new extension.

## Before opening a pull request

CI runs the same checks on Linux and macOS, on every supported Python:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test
.venv/bin/python -m pytest
```

Record user-visible changes under `## [Unreleased]` in
[CHANGELOG.md](CHANGELOG.md).

## Releasing

The crate and the Python package share one version, `[workspace.package]
version` in `Cargo.toml`; maturin reads the Python version from it.

1. In a pull request, bump that version and move the `[Unreleased]` entries of
   `CHANGELOG.md` under a new `## [X.Y.Z] - YYYY-MM-DD` heading, with its
   compare link at the foot of the file.
2. Once it is merged, run the **Release** workflow on `main` from the Actions
   tab. Tick *dry run* first to build and test every wheel without publishing.

The workflow refuses a version that is already tagged or has no changelog
section. It runs CI, builds the wheels (manylinux x86_64/aarch64 with a
vendored OpenBLAS, macOS arm64/x86_64) and the sdist, installs and tests each
wheel on every supported Python, then publishes to PyPI and crates.io, and
finally tags `vX.Y.Z` and creates the GitHub release from the changelog
section.
