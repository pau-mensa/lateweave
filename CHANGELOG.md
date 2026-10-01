# Changelog

All notable changes to lateweave are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
crate and the Python package share one version.

## [Unreleased]

## [0.2.0] - 2026-10-01

### Added
- `Subset::excluding` / `Subset.excluding`: a corpus may name the documents a
  search must not return, as well as the ones it may (`Restriction`).
- `SearchPipeline` stages may borrow per-query state (`SearchPipeline<'a>`).
- `lateweave.Subset` in Python.

### Changed
- `Subset::with` is now `Subset::including`; `Subset::ids` is replaced by
  `Subset::restriction`.
- Python: `SearchPipeline.search(subset=…)` takes a `lateweave.Subset`, and
  gatherers receive one, in place of a mapping of corpora to frozensets.

## [0.1.0] - 2026-09-29

First release.

- `SearchPipeline`: a candidate generator, an optional reranker, and a
  deterministic top-k, with compatibility checks, subset enforcement,
  freshness (`as_of`), provenance, and timings.
- `Query` and `Feature`: raw text plus named features stamped with the
  `Representation` that produced them, materialized at most once.
- `MaxSimReranker` over one `MultiVectorSource` per corpus, backed by a packed
  CPU MaxSim kernel.
- `VectorStore` and `VectorStoreWriter` for the
  [documented store format](https://github.com/pau-mensa/lateweave/blob/main/STORE_FORMAT.md), in `float32` and `int8`.
- Python bindings over the Rust crate, for Python 3.11 to 3.14.

[Unreleased]: https://github.com/pau-mensa/lateweave/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pau-mensa/lateweave/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pau-mensa/lateweave/releases/tag/v0.1.0
