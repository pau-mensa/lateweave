from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from .manifest import IncompatibleIndexError, IndexManifest, document_ids_digest


_DOCUMENTS_TABLE = "documents"
_MANIFEST_TABLE = "_lateweave_metadata_manifest"
_STORE_FORMAT = "lateweave-duckdb-metadata"
_STORE_VERSION = 1
_RESERVED_FIELDS = frozenset({"document_id", "external_id"})


def _require_duckdb() -> Any:
    try:
        return importlib.import_module("duckdb")
    except ModuleNotFoundError as error:
        raise ImportError(
            "DuckDBMetadataStore requires the optional metadata dependency; "
            "install it with `pip install 'lateweave[metadata]'`"
        ) from error


@dataclass(frozen=True)
class MetadataRecord:
    """One generator ID binding and its typed, filterable metadata fields."""

    document_id: int
    external_id: str
    fields: Mapping[str, Any] = field(default_factory=dict)


class DuckDBMetadataStore:
    """Immutable, generation-bound metadata filtering backed by DuckDB.

    The store does not define mutation semantics. Integrations create a new
    store from the final generator ID bindings whenever an index generation is
    replaced. DuckDB is imported only when a store is created or opened, so it
    remains an optional package dependency.
    """

    def __init__(self, path: str | Path, manifest: IndexManifest) -> None:
        duckdb = _require_duckdb()
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"metadata store not found: {self.path}")
        self.manifest = manifest
        self._connection = duckdb.connect(str(self.path), read_only=True)
        try:
            self._validate(manifest)
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(
        cls, path: str | Path, manifest: IndexManifest
    ) -> "DuckDBMetadataStore":
        """Open a store read-only and validate it against a generator manifest."""
        return cls(path, manifest)

    @classmethod
    def create(
        cls,
        path: str | Path,
        records: Sequence[MetadataRecord],
        manifest: IndexManifest,
    ) -> "DuckDBMetadataStore":
        """Create a static store, refusing bindings that disagree with the manifest."""
        duckdb = _require_duckdb()
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"metadata store already exists: {path}")
        prepared = cls._prepare_records(records, manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            connection = duckdb.connect(str(temporary))
            try:
                connection.execute("BEGIN TRANSACTION")
                if prepared:
                    connection.execute(
                        f"CREATE TABLE {_DOCUMENTS_TABLE} AS "
                        "SELECT row.* FROM UNNEST(?) AS source(row)",
                        [prepared],
                    )
                    connection.execute(
                        f"ALTER TABLE {_DOCUMENTS_TABLE} "
                        "ALTER COLUMN document_id SET NOT NULL"
                    )
                    connection.execute(
                        f"ALTER TABLE {_DOCUMENTS_TABLE} "
                        "ALTER COLUMN external_id SET NOT NULL"
                    )
                else:
                    connection.execute(
                        f"CREATE TABLE {_DOCUMENTS_TABLE} "
                        "(document_id BIGINT NOT NULL, external_id VARCHAR NOT NULL)"
                    )
                connection.execute(
                    f"CREATE UNIQUE INDEX document_id_unique "
                    f"ON {_DOCUMENTS_TABLE}(document_id)"
                )
                connection.execute(
                    f"CREATE UNIQUE INDEX external_id_unique "
                    f"ON {_DOCUMENTS_TABLE}(external_id)"
                )
                connection.execute(
                    f"CREATE TABLE {_MANIFEST_TABLE} ("
                    "store_format VARCHAR NOT NULL, "
                    "store_version INTEGER NOT NULL, "
                    "corpus_id VARCHAR NOT NULL, "
                    "corpus_version VARCHAR NOT NULL, "
                    "document_count BIGINT NOT NULL, "
                    "document_ids_sha256 VARCHAR NOT NULL, "
                    "generation BIGINT NOT NULL, "
                    "generator_representation VARCHAR NOT NULL, "
                    "generator_representation_version INTEGER NOT NULL)"
                )
                connection.execute(
                    f"INSERT INTO {_MANIFEST_TABLE} "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        _STORE_FORMAT,
                        _STORE_VERSION,
                        manifest.corpus_id,
                        manifest.corpus_version,
                        manifest.document_count,
                        manifest.document_ids_sha256,
                        manifest.generation,
                        manifest.representation,
                        manifest.representation_version,
                    ],
                )
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except BaseException:
                    pass
                raise
            finally:
                connection.close()
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return cls(path, manifest)

    @staticmethod
    def _prepare_records(
        records: Sequence[MetadataRecord], manifest: IndexManifest
    ) -> list[dict[str, Any]]:
        if len(records) != manifest.document_count:
            raise IncompatibleIndexError(
                "metadata record count does not match the index manifest "
                f"({len(records)} != {manifest.document_count})"
            )
        prepared: list[tuple[int, str, dict[str, Any]]] = []
        document_ids: set[int] = set()
        external_ids: set[str] = set()
        field_spellings: dict[str, str] = {}
        for position, record in enumerate(records):
            if not isinstance(record, MetadataRecord):
                raise TypeError(f"metadata record {position} is not a MetadataRecord")
            if isinstance(record.document_id, bool) or not isinstance(
                record.document_id, int
            ):
                raise TypeError("metadata document IDs must be integers")
            if not 0 <= record.document_id <= np.iinfo(np.int64).max:
                raise ValueError(
                    "metadata document IDs must be non-negative int64 values"
                )
            if record.document_id in document_ids:
                raise ValueError(f"duplicate metadata document ID {record.document_id}")
            if not isinstance(record.external_id, str):
                raise TypeError("metadata external IDs must be strings")
            if record.external_id in external_ids:
                raise ValueError(
                    f"duplicate metadata external ID {record.external_id!r}"
                )
            if not isinstance(record.fields, Mapping):
                raise TypeError("metadata fields must be a mapping")
            fields: dict[str, Any] = {}
            for name, value in record.fields.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("metadata field names must be non-empty strings")
                folded = name.casefold()
                if folded in _RESERVED_FIELDS:
                    raise ValueError(f"metadata field name {name!r} is reserved")
                previous = field_spellings.setdefault(folded, name)
                if previous != name:
                    raise ValueError(
                        f"metadata field names {previous!r} and {name!r} differ only by case"
                    )
                fields[name] = value
            document_ids.add(record.document_id)
            external_ids.add(record.external_id)
            prepared.append((record.document_id, record.external_id, fields))

        prepared.sort(key=lambda item: item[0])
        digest = document_ids_digest([item[1] for item in prepared])
        if digest != manifest.document_ids_sha256:
            raise IncompatibleIndexError(
                "metadata external IDs do not match the index manifest digest"
            )
        return [
            {"document_id": document_id, "external_id": external_id, **fields}
            for document_id, external_id, fields in prepared
        ]

    def _validate(self, manifest: IndexManifest) -> None:
        try:
            rows = self._connection.execute(
                f"SELECT store_format, store_version, corpus_id, corpus_version, "
                "document_count, document_ids_sha256, generation, "
                "generator_representation, generator_representation_version "
                f"FROM {_MANIFEST_TABLE}"
            ).fetchall()
        except Exception as error:
            raise ValueError(
                f"{self.path} is not a supported lateweave metadata store"
            ) from error
        if len(rows) != 1:
            raise ValueError(f"{self.path} has invalid metadata-store identity")
        row = rows[0]
        if row[0] != _STORE_FORMAT or int(row[1]) != _STORE_VERSION:
            raise ValueError(f"{self.path} uses an unsupported metadata-store format")
        actual = {
            "corpus_id": str(row[2]),
            "corpus_version": str(row[3]),
            "document_count": int(row[4]),
            "document_ids_sha256": str(row[5]),
            "generation": int(row[6]),
            "representation": str(row[7]),
            "representation_version": int(row[8]),
        }
        expected = {
            "corpus_id": manifest.corpus_id,
            "corpus_version": manifest.corpus_version,
            "document_count": manifest.document_count,
            "document_ids_sha256": manifest.document_ids_sha256,
            "generation": manifest.generation,
            "representation": manifest.representation,
            "representation_version": manifest.representation_version,
        }
        mismatches = [
            f"{name}: {actual[name]!r} != {expected[name]!r}"
            for name in expected
            if actual[name] != expected[name]
        ]
        if mismatches:
            raise IncompatibleIndexError(
                "metadata store is incompatible (" + "; ".join(mismatches) + ")"
            )
        try:
            count = int(
                self._connection.execute(
                    f"SELECT count(*) FROM {_DOCUMENTS_TABLE}"
                ).fetchone()[0]
            )
        except Exception as error:
            raise ValueError(
                f"{self.path} has no valid metadata documents table"
            ) from error
        if count != manifest.document_count:
            raise ValueError(
                "metadata documents do not match the stored document count "
                f"({count} != {manifest.document_count})"
            )

    def select(self, expression: Any | None = None) -> np.ndarray:
        """Return matching generator document IDs in ascending order."""
        if self._connection is None:
            raise RuntimeError("metadata store is closed")
        duckdb = _require_duckdb()
        if expression is not None and not isinstance(expression, duckdb.Expression):
            raise TypeError("metadata filter must be a DuckDB Expression")
        relation = self._connection.table(_DOCUMENTS_TABLE)
        if expression is not None:
            relation = relation.filter(expression)
        values = relation.project("document_id").order("document_id").fetchnumpy()
        return np.asarray(values["document_id"], dtype=np.int64)

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "DuckDBMetadataStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
