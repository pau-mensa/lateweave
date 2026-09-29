# Changelog

All notable changes to lateweave are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
crate and the Python package share one version.

## [Unreleased]

## [0.1.0]

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

[Unreleased]: https://github.com/pau-mensa/lateweave/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pau-mensa/lateweave/releases/tag/v0.1.0
