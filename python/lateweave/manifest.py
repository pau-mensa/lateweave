"""Identity contracts checked before any search runs.

Two identities are kept apart. A :class:`CorpusManifest` says *which documents*
a stage indexes: every stage in a pipeline must agree on it. A
:class:`Representation` says *which encoder* produced a vector feature: a stage
that consumes such a feature must agree on it with the query that supplies it.

Validation and compatibility are decided by the Rust library, the same checks a
Rust pipeline runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ._native import (
    IncompatibleIndexError,
    IncompatibleQueryError,
    _assert_corpora_compatible,
    _assert_representations_compatible,
    _check_corpus_manifest,
    _check_representation,
    document_ids_digest,
)

__all__ = [
    "CorpusManifest",
    "IncompatibleIndexError",
    "IncompatibleQueryError",
    "Representation",
    "document_ids_digest",
]


@dataclass(frozen=True)
class CorpusManifest:
    """Identity of one indexed document set at one mutation generation."""

    corpus_id: str
    corpus_version: str
    document_count: int
    document_ids_sha256: str
    generation: int = 0

    def __post_init__(self) -> None:
        _check_corpus_manifest(self)

    @classmethod
    def read(cls, path: str | Path) -> "CorpusManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def assert_compatible(self, other: "CorpusManifest") -> None:
        _assert_corpora_compatible(self, other)


@dataclass(frozen=True)
class Representation:
    """Identity of the encoder that produced a vector feature."""

    encoder: str
    encoder_revision: str
    dimension: int
    normalized: bool
    similarity: str = "dot"
    query_template: str = ""
    document_template: str = ""

    def __post_init__(self) -> None:
        _check_representation(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Representation":
        return cls(**value)

    def assert_compatible(self, other: "Representation") -> None:
        _assert_representations_compatible(self, other)
