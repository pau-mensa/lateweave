from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from lateweave import (  # noqa: E402
    DuckDBMetadataStore,
    IncompatibleIndexError,
    IndexManifest,
    MetadataRecord,
    document_ids_digest,
)


def manifest(
    external_ids: list[str], *, generation: int = 0
) -> IndexManifest:
    return IndexManifest(
        corpus_id="laws",
        corpus_version="2026-09-02",
        document_count=len(external_ids),
        document_ids_sha256=document_ids_digest(external_ids),
        encoder="encoder",
        encoder_revision="main",
        tokenizer="tokenizer",
        dimension=8,
        dtype="float32",
        normalized=True,
        representation="bm25s-sparse-index",
        score_semantics="bm25s-lucene",
        generation=generation,
    )


def records() -> list[MetadataRecord]:
    return [
        MetadataRecord(
            12,
            "law-c",
            {
                "country": "ES",
                "year": 2020,
                "tags": ["water"],
                "published": date(2020, 5, 2),
            },
        ),
        MetadataRecord(
            4,
            "law-a",
            {
                "country": "ES",
                "year": 2024,
                "tags": ["tax", "procedure"],
                "published": date(2024, 1, 3),
                "source": {"court": "TS"},
            },
        ),
        MetadataRecord(
            9,
            "law-b",
            {
                "country": "FR",
                "year": 2025,
                "tags": ["tax"],
                "published": date(2025, 7, 1),
            },
        ),
    ]


def test_static_store_filters_typed_metadata_and_reopens(tmp_path) -> None:
    index_manifest = manifest(["law-a", "law-b", "law-c"], generation=3)
    path = tmp_path / "metadata.duckdb"
    store = DuckDBMetadataStore.create(path, records(), index_manifest)

    country = duckdb.ColumnExpression("country")
    year = duckdb.ColumnExpression("year")
    expression = (country == duckdb.ConstantExpression("ES")) & (
        year >= duckdb.ConstantExpression(2024)
    )
    selected = store.select(expression)

    assert selected.dtype == np.int64
    assert selected.tolist() == [4]
    assert store.select().tolist() == [4, 9, 12]
    store.close()

    with DuckDBMetadataStore.open(path, index_manifest) as reopened:
        has_tax = duckdb.FunctionExpression(
            "list_contains",
            duckdb.ColumnExpression("tags"),
            duckdb.ConstantExpression("tax"),
        )
        assert reopened.select(has_tax).tolist() == [4, 9]
    with pytest.raises(RuntimeError, match="closed"):
        reopened.select()


def test_store_is_bound_to_the_generator_manifest_generation(tmp_path) -> None:
    index_manifest = manifest(["law-a", "law-b", "law-c"], generation=2)
    path = tmp_path / "metadata.duckdb"
    store = DuckDBMetadataStore.create(path, records(), index_manifest)
    store.close()

    with pytest.raises(IncompatibleIndexError, match="generation"):
        DuckDBMetadataStore(path, replace(index_manifest, generation=3))
    with pytest.raises(IncompatibleIndexError, match="representation"):
        DuckDBMetadataStore(
            path, replace(index_manifest, representation="another-generator")
        )


def test_rebuild_uses_the_new_generators_authoritative_ids(tmp_path) -> None:
    generation_one = manifest(["law-a", "law-c"], generation=1)
    rebuilt = DuckDBMetadataStore.create(
        tmp_path / "generation-1.duckdb",
        [
            MetadataRecord(0, "law-a", {"country": "ES"}),
            MetadataRecord(1, "law-c", {"country": "ES"}),
        ],
        generation_one,
    )

    assert rebuilt.select().tolist() == [0, 1]
    rebuilt.close()


def test_create_validates_id_bindings_against_the_manifest(tmp_path) -> None:
    index_manifest = manifest(["law-a", "law-b", "law-c"])

    with pytest.raises(IncompatibleIndexError, match="digest"):
        DuckDBMetadataStore.create(
            tmp_path / "wrong-order.duckdb",
            [
                replace(record, external_id=f"changed-{record.external_id}")
                for record in records()
            ],
            index_manifest,
        )
    with pytest.raises(ValueError, match="duplicate metadata document ID"):
        DuckDBMetadataStore.create(
            tmp_path / "duplicate.duckdb",
            [*records()[:2], replace(records()[2], document_id=4)],
            index_manifest,
        )
    with pytest.raises(ValueError, match="reserved"):
        DuckDBMetadataStore.create(
            tmp_path / "reserved.duckdb",
            [
                replace(record, fields={**record.fields, "Document_ID": 7})
                for record in records()
            ],
            index_manifest,
        )


def test_store_rejects_non_expression_filters(tmp_path) -> None:
    index_manifest = manifest(["law-a", "law-b", "law-c"])
    store = DuckDBMetadataStore.create(
        tmp_path / "metadata.duckdb", records(), index_manifest
    )
    with pytest.raises(TypeError, match="DuckDB Expression"):
        store.select("country = 'ES'")
    store.close()


def test_empty_generation_has_a_valid_static_store(tmp_path) -> None:
    index_manifest = manifest([])
    store = DuckDBMetadataStore.create(
        tmp_path / "metadata.duckdb", [], index_manifest
    )
    assert store.select().shape == (0,)
    assert store.select().dtype == np.int64
    store.close()


def test_open_detects_documents_changed_outside_the_static_api(tmp_path) -> None:
    index_manifest = manifest(["law-a", "law-b", "law-c"])
    path = tmp_path / "metadata.duckdb"
    store = DuckDBMetadataStore.create(path, records(), index_manifest)
    store.close()
    connection = duckdb.connect(str(path))
    connection.execute("DELETE FROM documents WHERE document_id = 9")
    connection.close()

    with pytest.raises(ValueError, match="stored document count"):
        DuckDBMetadataStore(path, index_manifest)
