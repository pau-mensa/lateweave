from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


class IncompatibleIndexError(ValueError):
    pass


def document_ids_digest(document_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for document_id in document_ids:
        encoded = document_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class IndexManifest:
    corpus_id: str
    corpus_version: str
    document_count: int
    document_ids_sha256: str
    encoder: str
    encoder_revision: str
    tokenizer: str
    dimension: int
    dtype: str
    normalized: bool
    similarity: str = "dot"
    query_template: str = ""
    document_template: str = ""
    truncation: str = ""
    token_filtering: str = ""
    representation: str = "unknown"
    representation_version: int = 1
    score_semantics: str = "unknown"
    generation: int = 0
    build_parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "corpus_id",
            "corpus_version",
            "document_ids_sha256",
            "encoder",
            "encoder_revision",
            "tokenizer",
            "dtype",
            "similarity",
            "representation",
            "score_semantics",
        ):
            if not getattr(self, name):
                raise ValueError(f"manifest {name} must not be empty")
        if self.document_count < 0:
            raise ValueError("manifest document_count must not be negative")
        if self.dimension <= 0:
            raise ValueError("manifest dimension must be positive")
        if self.representation_version <= 0:
            raise ValueError("manifest representation_version must be positive")
        if self.generation < 0:
            raise ValueError("manifest generation must not be negative")

    @classmethod
    def read(cls, path: str | Path) -> "IndexManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**value)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def assert_compatible(self, other: "IndexManifest") -> None:
        fields = (
            "corpus_id",
            "corpus_version",
            "document_count",
            "document_ids_sha256",
            "encoder",
            "encoder_revision",
            "tokenizer",
            "dimension",
            "normalized",
            "similarity",
            "query_template",
            "document_template",
            "truncation",
            "token_filtering",
            "generation",
        )
        mismatches = [
            f"{name}: {getattr(self, name)!r} != {getattr(other, name)!r}"
            for name in fields
            if getattr(self, name) != getattr(other, name)
        ]
        if mismatches:
            raise IncompatibleIndexError(
                "generator and scorer indexes are incompatible ("
                + "; ".join(mismatches)
                + ")"
            )
