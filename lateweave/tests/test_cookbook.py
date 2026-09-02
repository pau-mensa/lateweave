from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

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
            representation="bm25s-sparse-index",
            score_semantics="bm25s-lucene",
            build_parameters=cookbook.Analyzer().build_parameters(),
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

    assert "bm25s" not in pyproject
    assert "fast-plaid" not in pyproject
    assert '"torch' not in pyproject
    assert "bm25s==" in script
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


# -- the lexical stage -----------------------------------------------------


def test_analyzer_folds_accents_and_case() -> None:
    assert cookbook.Analyzer().tokens("Artículo 14: LOS españoles") == [
        "articulo",
        "14",
        "los",
        "espanoles",
    ]


def test_analyzer_default_does_not_stem() -> None:
    assert cookbook.Analyzer().tokens("concesiones") == ["concesiones"]


def test_analyzer_stems_when_a_language_is_named() -> None:
    assert cookbook.Analyzer(stemmer="spanish").tokens("concesiones") == ["concesion"]


def test_analyzer_rejects_an_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unknown Snowball algorithm"):
        cookbook.Analyzer(stemmer="klingon")


def test_analyzer_round_trips_through_the_generator_manifest() -> None:
    generator, _ = manifests()
    stemmed = replace(
        generator, build_parameters=cookbook.Analyzer(stemmer="spanish").build_parameters()
    )
    assert cookbook.Analyzer.from_manifest(generator) == cookbook.Analyzer()
    assert cookbook.Analyzer.from_manifest(stemmed) == cookbook.Analyzer(
        stemmer="spanish"
    )


def test_a_foreign_tokenizer_is_refused_rather_than_approximated() -> None:
    # Querying through a chain the postings were not built with has to raise,
    # not quietly return fewer documents.
    generator, _ = manifests()
    foreign = replace(generator, build_parameters={"tokenizer": "unicode"})
    with pytest.raises(ValueError, match="cannot reproduce"):
        cookbook.Analyzer.from_manifest(foreign)


# -- end to end, through the command line ----------------------------------

DOCUMENTS = (
    {"id": "law-1", "text": "concesión de obra pública y autorización previa"},
    {"id": "law-2", "text": "plazo máximo de inscripción de treinta días"},
    {"id": "law-3", "text": "chair water index unrelated"},
    {"id": "law-4", "text": "autorización de una concesión administrativa"},
)
DIMENSION = 8


def write_corpus(directory: Path, documents, seed: int = 0) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    documents_path = directory / "documents.jsonl"
    with documents_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    rng = np.random.default_rng(seed)
    lengths = np.full(len(documents), 3, dtype=np.int64)
    vectors = rng.normal(size=(int(lengths.sum()), DIMENSION)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    embeddings_path = directory / "embeddings.npy"
    lengths_path = directory / "lengths.npy"
    np.save(embeddings_path, vectors)
    np.save(lengths_path, lengths)
    return {
        "documents": documents_path,
        "embeddings": embeddings_path,
        "lengths": lengths_path,
    }


def run_cookbook(*argv: str) -> None:
    args = cookbook.parser().parse_args(list(argv))
    args.function(args)


def build_corpus(tmp_path: Path, documents=DOCUMENTS, stemmer: str | None = None):
    inputs = write_corpus(tmp_path / "input", documents)
    index = tmp_path / "index"
    argv = [
        "build",
        "--index", str(index),
        "--documents", str(inputs["documents"]),
        "--embeddings", str(inputs["embeddings"]),
        "--document-lengths", str(inputs["lengths"]),
        "--corpus-id", "laws",
        "--corpus-version", "1",
        "--encoder", "encoder",
        "--encoder-revision", "1",
        "--tokenizer", "tokenizer",
    ]
    if stemmer is not None:
        argv += ["--stemmer", stemmer]
    run_cookbook(*argv)
    return index


def gathered_ids(index: Path, query: str, limit: int = 10) -> list[int]:
    """Internal IDs a query reaches, sorted; gather itself ranks by score."""
    manifest = IndexManifest.read(index / cookbook.GENERATOR_MANIFEST)
    generator = cookbook.LexicalCandidateGenerator.open(
        index / cookbook.LEXICAL_DIRECTORY, manifest
    )
    query_object = cookbook.Query(query, embeddings=np.zeros((1, DIMENSION), np.float32))
    return sorted(
        candidate.document_id for candidate in generator.gather(query_object, limit)
    )


def test_build_writes_a_lexical_index_that_reopens(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    assert (index / cookbook.LEXICAL_DIRECTORY).is_dir()
    assert gathered_ids(index, "concesión") == [0, 3]


def test_gather_returns_only_matching_documents(tmp_path: Path) -> None:
    # A short row is returned short; nothing is padded to the requested limit.
    index = build_corpus(tmp_path)
    assert gathered_ids(index, "zzzzzz") == []
    assert gathered_ids(index, "chair") == [2]


def test_gather_labels_candidates_with_the_backend(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    manifest = IndexManifest.read(index / cookbook.GENERATOR_MANIFEST)
    generator = cookbook.LexicalCandidateGenerator.open(
        index / cookbook.LEXICAL_DIRECTORY, manifest
    )
    query = cookbook.Query("concesión", embeddings=np.zeros((1, DIMENSION), np.float32))
    candidates = generator.gather(query, 10)
    assert {candidate.provenance for candidate in candidates} == {"bm25s"}
    # gather_rank is dense and score-ordered, with no gaps left by filtering.
    assert [candidate.gather_rank for candidate in candidates] == list(
        range(len(candidates))
    )
    scores = [candidate.gather_score for candidate in candidates]
    assert scores == sorted(scores, reverse=True)


def test_a_stemmed_index_matches_inflected_queries(tmp_path: Path) -> None:
    plain = build_corpus(tmp_path / "plain")
    stemmed = build_corpus(tmp_path / "stemmed", stemmer="spanish")
    assert gathered_ids(plain, "concesiones") == []
    assert gathered_ids(stemmed, "concesiones") == [0, 3]


def test_update_rebuilds_the_lexical_index_over_the_whole_corpus(
    tmp_path: Path,
) -> None:
    index = build_corpus(tmp_path)
    additions = ({"id": "law-5", "text": "nueva concesión de aguas"},)
    inputs = write_corpus(tmp_path / "added", additions, seed=1)
    run_cookbook(
        "update",
        "--index", str(index),
        "--documents", str(inputs["documents"]),
        "--embeddings", str(inputs["embeddings"]),
        "--document-lengths", str(inputs["lengths"]),
    )
    # The appended document is reachable, and so are the originals.
    assert gathered_ids(index, "concesión") == [0, 3, 4]
    assert gathered_ids(index, "aguas") == [4]


def test_update_carries_the_analyzer_forward(tmp_path: Path) -> None:
    index = build_corpus(tmp_path, stemmer="spanish")
    additions = ({"id": "law-5", "text": "nuevas concesiones de aguas"},)
    inputs = write_corpus(tmp_path / "added", additions, seed=1)
    run_cookbook(
        "update",
        "--index", str(index),
        "--documents", str(inputs["documents"]),
        "--embeddings", str(inputs["embeddings"]),
        "--document-lengths", str(inputs["lengths"]),
    )
    manifest = IndexManifest.read(index / cookbook.GENERATOR_MANIFEST)
    assert cookbook.Analyzer.from_manifest(manifest) == cookbook.Analyzer(
        stemmer="spanish"
    )
    assert gathered_ids(index, "concesion") == [0, 3, 4]


def test_delete_compacts_internal_ids_in_document_order(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    run_cookbook("delete", "--index", str(index), "--document-id", "law-2")
    documents = cookbook.load_documents(index / cookbook.DOCUMENTS_FILE)
    assert [row["id"] for row in documents] == ["law-1", "law-3", "law-4"]
    # law-4 was internal 3 and is internal 2 now; the lexical index agrees.
    assert gathered_ids(index, "concesión") == [0, 2]
    assert gathered_ids(index, "inscripción") == []


def test_generator_refuses_an_index_whose_size_disagrees(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    manifest = IndexManifest.read(index / cookbook.GENERATOR_MANIFEST)
    with pytest.raises(RuntimeError, match="documents but the manifest"):
        cookbook.LexicalCandidateGenerator.open(
            index / cookbook.LEXICAL_DIRECTORY, replace(manifest, document_count=99)
        )


def test_search_gathers_then_scores_and_reports_external_ids(
    tmp_path: Path, capsys
) -> None:
    index = build_corpus(tmp_path)
    query_path = tmp_path / "query.npy"
    rng = np.random.default_rng(7)
    query_vectors = rng.normal(size=(2, DIMENSION)).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    np.save(query_path, query_vectors)
    capsys.readouterr()  # discard the build's own output

    run_cookbook(
        "search",
        "--index", str(index),
        "--query", "concesión",
        "--query-embeddings", str(query_path),
        "--gather-limit", "10",
        "--limit", "5",
    )
    output = json.loads(capsys.readouterr().out)

    # Only the two documents mentioning the term are gathered, so only they can
    # be scored: the lexical stage sets the ceiling.
    assert {row["external_id"] for row in output["results"]} == {"law-1", "law-4"}
    # SearchPipeline ranks results from 1; gather_rank above is 0-based.
    assert [row["rank"] for row in output["results"]] == [1, 2]
