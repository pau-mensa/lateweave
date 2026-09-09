"""Identity contracts checked before any search runs.

Two identities are kept apart. A :class:`CorpusManifest` says *which documents*
a stage indexes: every stage in a pipeline must agree on it. A
:class:`Representation` says *which encoder* produced a vector feature: a stage
that consumes such a feature must agree on it with the query that supplies it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class IncompatibleIndexError(ValueError):
    """Two stages do not index the same documents."""


class IncompatibleQueryError(ValueError):
    """A query lacks a feature a stage needs, or supplies it from another encoder."""


def document_ids_digest(document_ids: Sequence[str]) -> str:
    """SHA-256 over external IDs in internal-ID order; identifies an ID binding."""
    digest = hashlib.sha256()
    for document_id in document_ids:
        encoded = document_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _mismatches(left: Any, right: Any) -> list[str]:
    return [
        f"{field.name}: {getattr(left, field.name)!r} != {getattr(right, field.name)!r}"
        for field in fields(left)
        if getattr(left, field.name) != getattr(right, field.name)
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
        for name in ("corpus_id", "corpus_version", "document_ids_sha256"):
            if not getattr(self, name):
                raise ValueError(f"corpus manifest {name} must not be empty")
        if self.document_count < 0:
            raise ValueError("corpus manifest document_count must not be negative")
        if self.generation < 0:
            raise ValueError("corpus manifest generation must not be negative")

    @classmethod
    def read(cls, path: str | Path) -> "CorpusManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def assert_compatible(self, other: "CorpusManifest") -> None:
        mismatches = _mismatches(self, other)
        if mismatches:
            raise IncompatibleIndexError(
                "stages index different corpora (" + "; ".join(mismatches) + ")"
            )


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
        for name in ("encoder", "encoder_revision", "similarity"):
            if not getattr(self, name):
                raise ValueError(f"representation {name} must not be empty")
        if self.dimension <= 0:
            raise ValueError("representation dimension must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Representation":
        return cls(**value)

    def assert_compatible(self, other: "Representation") -> None:
        mismatches = _mismatches(self, other)
        if mismatches:
            raise IncompatibleQueryError(
                "representations differ (" + "; ".join(mismatches) + ")"
            )
