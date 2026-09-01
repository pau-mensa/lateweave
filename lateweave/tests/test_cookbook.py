from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from lateweave import IndexManifest


SCRIPT = Path(__file__).parents[1] / "cookbook" / "bm25_stored_maxsim.py"
SPEC = importlib.util.spec_from_file_location("lateweave_cookbook_bm25_stored", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cookbook = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cookbook
SPEC.loader.exec_module(cookbook)


def manifests() -> tuple[IndexManifest, IndexManifest]:
    common = {
        "corpus_id": "laws",
        "corpus_version": "1",
        "document_count": 3,
        "document_ids_sha256": "abc",
        "encoder": "encoder",
        "encoder_revision": "1",
        "tokenizer": "tokenizer",
        "dimension": 8,
        "dtype": "float32",
        "normalized": True,
    }
    return (
        IndexManifest(
            **common,
            representation="bm25x-inverted-index",
            score_semantics="bm25x-lucene",
        ),
        IndexManifest(
            **common,
            representation="lateweave-int8-rowwise-v1",
            score_semantics="int8-reconstructed-approximate-full-maxsim",
        ),
    )


def test_cookbook_dependencies_remain_out_of_package_metadata() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert "bm25x" not in pyproject
    assert "fast-plaid" not in pyproject
    assert '"torch' not in pyproject
    assert "bm25x==" in script
    assert "fast-plaid" not in script.lower()


def test_mutations_advance_both_manifest_identities_together() -> None:
    generator, scorer = manifests()
    documents = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]

    mutated_generator = cookbook.mutated_manifest(
        generator, documents, operation="delete", affected=1
    )
    mutated_scorer = cookbook.mutated_manifest(
        scorer, documents, operation="delete", affected=1
    )

    mutated_generator.assert_compatible(mutated_scorer)
    assert mutated_generator.generation == mutated_scorer.generation == 1
    assert mutated_generator.build_parameters["cookbook_mutations"] == [
        {"generation": 1, "operation": "delete", "affected_documents": 1}
    ]


def test_cookbook_parser_exposes_all_stores_and_mutations() -> None:
    parser = cookbook.parser()
    help_text = parser.format_help()
    build = parser.parse_args(
        [
            "build",
            "--index",
            "index",
            "--documents",
            "documents.jsonl",
            "--embeddings",
            "embeddings.npy",
            "--document-lengths",
            "lengths.npy",
            "--storage",
            "turboquant4",
            "--corpus-id",
            "corpus",
            "--corpus-version",
            "1",
            "--encoder",
            "encoder",
            "--encoder-revision",
            "1",
            "--tokenizer",
            "tokenizer",
        ]
    )

    assert "update" in help_text
    assert "delete" in help_text
    assert build.storage == "turboquant4"
    jzip = parser.parse_args(
        [
            "build",
            "--index",
            "index",
            "--documents",
            "documents.jsonl",
            "--embeddings",
            "embeddings.npy",
            "--document-lengths",
            "lengths.npy",
            "--storage",
            "jzip",
            "--compression-level",
            "3",
            "--corpus-id",
            "corpus",
            "--corpus-version",
            "1",
            "--encoder",
            "encoder",
            "--encoder-revision",
            "1",
            "--tokenizer",
            "tokenizer",
        ]
    )
    assert jzip.storage == "jzip"
    assert jzip.compression_level == 3
