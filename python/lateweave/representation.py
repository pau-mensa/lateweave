"""Identity of the encoder behind a vector feature, checked before any stage
runs: a stage that consumes a feature must agree on it with the query that
supplies it.

Validation and compatibility are decided by the Rust library, the same checks a
Rust pipeline runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ._native import (
    IncompatibleIndexError,
    IncompatibleQueryError,
    _assert_representations_compatible,
    _check_representation,
)

__all__ = ["IncompatibleIndexError", "IncompatibleQueryError", "Representation"]


@dataclass(frozen=True)
class Representation:
    """Identity of the encoder that produced a vector feature."""

    encoder: str
    encoder_revision: str
    dimension: int
    normalized: bool
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
