from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from lateweave import CorpusManifest, IncompatibleIndexError, Segment


SCRIPT = Path(__file__).parents[1] / "cookbook" / "bm25_stored_maxsim.py"
SPEC = importlib.util.spec_from_file_location("lateweave_cookbook_bm25_stored", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cookbook = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cookbook
SPEC.loader.exec_module(cookbook)

DIMENSION = 8
DOCUMENTS = (
    {"id": "law-1", "text": "concesión de obra pública y autorización previa"},
    {"id": "law-2", "text": "plazo máximo de inscripción de treinta días"},
    {"id": "law-3", "text": "chair water index unrelated"},
    {"id": "law-4", "text": "autorización de una concesión administrativa"},
)


def test_cookbook_dependencies_remain_out_of_package_metadata() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    assert "bm25s" not in pyproject
    assert "bm25s==" in script


# -- the lexical stage -----------------------------------------------------


def test_analyzer_folds_accents_and_case() -> None:
    assert cookbook.Analyzer().tokens("Artículo 14: LOS españoles") == [
        "articulo", "14", "los", "espanoles",
    ]


def test_analyzer_stems_only_when_a_language_is_named() -> None:
    assert cookbook.Analyzer().tokens("concesiones") == ["concesiones"]
    assert cookbook.Analyzer(stemmer="spanish").tokens("concesiones") == ["concesion"]
    with pytest.raises(ValueError, match="unknown Snowball algorithm"):
        cookbook.Analyzer(stemmer="klingon")


def test_analyzer_round_trips_and_refuses_a_foreign_tokenizer(tmp_path: Path) -> None:
    cookbook.Analyzer(stemmer="spanish").write(tmp_path / "analyzer.json")
    assert cookbook.Analyzer.read(tmp_path / "analyzer.json") == cookbook.Analyzer(stemmer="spanish")
    (tmp_path / "foreign.json").write_text(json.dumps({"tokenizer": "unicode"}))
    with pytest.raises(ValueError, match="cannot reproduce"):
        cookbook.Analyzer.read(tmp_path / "foreign.json")


# -- end to end, through the command line ----------------------------------


def unit_vectors(rows: int, seed: int) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(rows, DIMENSION)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def write_corpus(directory: Path, documents, seed: int = 0) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "documents": directory / "documents.jsonl",
        "embeddings": directory / "embeddings.npy",
        "lengths": directory / "lengths.npy",
    }
    with paths["documents"].open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    lengths = np.full(len(documents), 3, dtype=np.int64)
    np.save(paths["embeddings"], unit_vectors(int(lengths.sum()), seed))
    np.save(paths["lengths"], lengths)
    return paths


def run_cookbook(*argv: str) -> None:
    args = cookbook.parser().parse_args(list(argv))
    args.function(args)


def build_corpus(tmp_path: Path, documents=DOCUMENTS, stemmer: str | None = None, storage="float32"):
    inputs = write_corpus(tmp_path / "input", documents)
    index = tmp_path / "index"
    argv = [
        "build",
        "--index", str(index),
        "--documents", str(inputs["documents"]),
        "--embeddings", str(inputs["embeddings"]),
        "--document-lengths", str(inputs["lengths"]),
        "--storage", storage,
        "--corpus-id", "laws",
        "--corpus-version", "1",
        "--encoder", "encoder",
        "--encoder-revision", "1",
    ]
    if stemmer is not None:
        argv += ["--stemmer", stemmer]
    run_cookbook(*argv)
    return index


def open_gatherer(index: Path) -> "cookbook.LexicalCandidateGenerator":
    return cookbook.LexicalCandidateGenerator.open(
        index / cookbook.LEXICAL_DIRECTORY,
        cookbook.read_segment(index),
        cookbook.Analyzer.read(index / cookbook.ANALYZER_FILE),
    )


def gathered_ids(index: Path, query: str, limit: int = 10, subset=None) -> list[int]:
    """Internal IDs a query reaches, sorted; gather itself ranks by score."""
    gatherer = open_gatherer(index)
    if subset is not None:
        subset = {gatherer.segment.corpus_id: np.asarray(subset, dtype=np.int64)}
    candidates = gatherer.gather(cookbook.Query(query), limit, subset=subset)
    return sorted(candidate.document_id for candidate in candidates)


def search(index: Path, capsys, *argv: str) -> dict:
    capsys.readouterr()
    run_cookbook("search", "--index", str(index), "--gather-limit", "10", "--limit", "5", *argv)
    return json.loads(capsys.readouterr().out)


def test_build_writes_a_lexical_index_that_gathers_only_matching_documents(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    assert (index / cookbook.LEXICAL_DIRECTORY).is_dir()
    assert gathered_ids(index, "concesión") == [0, 3]
    assert gathered_ids(index, "zzzzzz") == []
    assert gathered_ids(index, "chair") == [2]


def test_gather_labels_candidates_with_dense_score_ordered_ranks(tmp_path: Path) -> None:
    candidates = open_gatherer(build_corpus(tmp_path)).gather(cookbook.Query("concesión"), 10)
    assert {candidate.provenance for candidate in candidates} == {"bm25s"}
    assert {candidate.external_id for candidate in candidates} == {"law-1", "law-4"}
    assert [candidate.gather_rank for candidate in candidates] == list(range(len(candidates)))
    scores = [candidate.gather_score for candidate in candidates]
    assert scores == sorted(scores, reverse=True)


def test_gather_honours_a_subset(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    assert gathered_ids(index, "concesión", subset=[3]) == [3]
    assert gathered_ids(index, "concesión", subset=[1]) == []


def test_a_stemmed_index_matches_inflected_queries(tmp_path: Path) -> None:
    plain = build_corpus(tmp_path / "plain")
    stemmed = build_corpus(tmp_path / "stemmed", stemmer="spanish")
    assert gathered_ids(plain, "concesiones") == []
    assert gathered_ids(stemmed, "concesiones") == [0, 3]


def test_update_appends_and_carries_the_analyzer_forward(tmp_path: Path) -> None:
    index = build_corpus(tmp_path, stemmer="spanish")
    inputs = write_corpus(tmp_path / "added", ({"id": "law-5", "text": "nuevas concesiones de aguas"},), seed=1)
    run_cookbook(
        "update",
        "--index", str(index),
        "--documents", str(inputs["documents"]),
        "--embeddings", str(inputs["embeddings"]),
        "--document-lengths", str(inputs["lengths"]),
    )
    manifest = CorpusManifest.read(index / cookbook.CORPUS_MANIFEST)
    assert (manifest.document_count, manifest.generation) == (5, 1)
    assert cookbook.open_vector_store(index / cookbook.VECTOR_DIRECTORY).segment == cookbook.read_segment(index)
    assert gathered_ids(index, "concesion") == [0, 3, 4]
    assert gathered_ids(index, "aguas") == [4]


def test_delete_compacts_internal_ids_in_document_order(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    run_cookbook("delete", "--index", str(index), "--document-id", "law-2")
    documents = cookbook.load_documents(index / cookbook.DOCUMENTS_FILE)
    assert [row["id"] for row in documents] == ["law-1", "law-3", "law-4"]
    # law-4 was internal 3 and is internal 2 now; the lexical index agrees.
    assert gathered_ids(index, "concesión") == [0, 2]
    assert gathered_ids(index, "inscripción") == []
    assert CorpusManifest.read(index / cookbook.CORPUS_MANIFEST).generation == 1
    store = cookbook.open_vector_store(index / cookbook.VECTOR_DIRECTORY)
    assert store.segment.document_ids == ["law-1", "law-3", "law-4"]
    assert store.segment == cookbook.read_segment(index)


def test_gatherer_refuses_an_index_whose_size_disagrees(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    with pytest.raises(RuntimeError, match="documents but the segment"):
        cookbook.LexicalCandidateGenerator.open(
            index / cookbook.LEXICAL_DIRECTORY,
            Segment("laws", "1", [str(item) for item in range(99)]),
            cookbook.Analyzer(),
        )


def test_search_refuses_a_store_at_another_generation(tmp_path: Path, capsys) -> None:
    index = build_corpus(tmp_path)
    query_path = tmp_path / "query.npy"
    np.save(query_path, unit_vectors(2, seed=7))
    # Mutate the store alone, as a crash between the two stages' updates would.
    cookbook.open_vector_store(index / cookbook.VECTOR_DIRECTORY).delete(["law-2"])
    with pytest.raises(IncompatibleIndexError, match="generation"):
        search(index, capsys, "--query", "concesión", "--query-embeddings", str(query_path))


@pytest.mark.parametrize("storage", ["float32", "int8"])
def test_search_reranks_when_embeddings_are_given(tmp_path: Path, capsys, storage: str) -> None:
    index = build_corpus(tmp_path, storage=storage)
    query_path = tmp_path / "query.npy"
    np.save(query_path, unit_vectors(2, seed=7))

    output = search(index, capsys, "--query", "concesión", "--query-embeddings", str(query_path))

    # Only the two documents mentioning the term are gathered: BM25 sets the ceiling.
    assert {row["external_id"] for row in output["results"]} == {"law-1", "law-4"}
    assert [row["rank"] for row in output["results"]] == [1, 2]
    assert output["diagnostics"]["reranker"] == "MaxSimReranker"
    assert output["diagnostics"]["score_semantics"] == cookbook.STORES[storage].score_semantics


def test_search_without_embeddings_is_gather_only(tmp_path: Path, capsys) -> None:
    output = search(build_corpus(tmp_path), capsys, "--query", "concesión")
    assert {row["external_id"] for row in output["results"]} == {"law-1", "law-4"}
    assert output["diagnostics"]["reranker"] is None
    assert output["diagnostics"]["score_semantics"] == "bm25s-lucene"


def test_search_subset_restricts_by_external_id(tmp_path: Path, capsys) -> None:
    output = search(build_corpus(tmp_path), capsys, "--query", "concesión", "--subset-id", "law-4")
    assert [row["external_id"] for row in output["results"]] == ["law-4"]


def test_search_refuses_embeddings_of_the_wrong_dimension(tmp_path: Path, capsys) -> None:
    index = build_corpus(tmp_path)
    query_path = tmp_path / "query.npy"
    np.save(query_path, np.ones((2, DIMENSION + 1), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        search(index, capsys, "--query", "concesión", "--query-embeddings", str(query_path))
