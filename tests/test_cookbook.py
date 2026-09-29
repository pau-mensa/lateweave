from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest



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


def build_corpus(tmp_path: Path, documents=DOCUMENTS, stemmer: str | None = None, storage="float32"):  # type: ignore[no-untyped-def]
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
        "--encoder", "encoder",
        "--encoder-revision", "1",
    ]
    if stemmer is not None:
        argv += ["--stemmer", stemmer]
    run_cookbook(*argv)
    return index


def upsert(index: Path, directory: Path, documents, seed: int = 1) -> None:  # type: ignore[no-untyped-def]
    inputs = write_corpus(directory, documents, seed=seed)
    run_cookbook(
        "upsert",
        "--index", str(index),
        "--documents", str(inputs["documents"]),
        "--embeddings", str(inputs["embeddings"]),
        "--document-lengths", str(inputs["lengths"]),
    )


def open_gatherer(index: Path) -> "cookbook.LexicalCandidateGenerator":
    return cookbook.LexicalCandidateGenerator(
        index / cookbook.LEXICAL_DIRECTORY, cookbook.Analyzer.read(index / cookbook.ANALYZER_FILE)
    )


def gathered_ids(index: Path, query: str, limit: int = 10, subset=None) -> list[str]:  # type: ignore[no-untyped-def]
    """Document IDs a query reaches, sorted; gather itself ranks by score."""
    if subset is not None:
        subset = {"laws": frozenset(subset)}
    gathered = open_gatherer(index).gather(cookbook.Query(query), limit, subset=subset)
    return sorted(candidate.document_id for candidate in gathered.candidates)


def search(index: Path, capsys, *argv: str) -> dict:  # type: ignore[no-untyped-def]
    capsys.readouterr()
    run_cookbook("search", "--index", str(index), "--gather-limit", "10", "--limit", "5", *argv)
    return json.loads(capsys.readouterr().out)


def query_embeddings(tmp_path: Path, dimension: int = DIMENSION) -> str:
    path = tmp_path / "query.npy"
    vectors = unit_vectors(2, seed=7) if dimension == DIMENSION else np.ones((2, dimension), dtype=np.float32)
    np.save(path, vectors)
    return str(path)


def result_ids(output: dict) -> set[str]:
    return {row["document_id"] for row in output["results"]}


def test_build_writes_a_lexical_index_that_gathers_only_matching_documents(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    assert gathered_ids(index, "concesión") == ["law-1", "law-4"]
    assert gathered_ids(index, "zzzzzz") == []
    assert gathered_ids(index, "chair") == ["law-3"]


def test_gather_labels_candidates_with_dense_score_ordered_ranks(tmp_path: Path) -> None:
    gathered = open_gatherer(build_corpus(tmp_path)).gather(cookbook.Query("concesión"), 10)
    candidates = gathered.candidates
    assert {(candidate.corpus, candidate.provenance) for candidate in candidates} == {("laws", "bm25s")}
    assert [candidate.gather_rank for candidate in candidates] == list(range(len(candidates)))
    scores = [candidate.gather_score for candidate in candidates]
    assert scores == sorted(scores, reverse=True)
    assert gathered.as_of.tzinfo is not None


def test_gather_honours_a_subset_by_document_id(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    assert gathered_ids(index, "concesión", subset=["law-4", "unknown"]) == ["law-4"]
    assert gathered_ids(index, "concesión", subset=["law-2"]) == []


def test_a_stemmed_index_matches_inflected_queries(tmp_path: Path) -> None:
    plain = build_corpus(tmp_path / "plain")
    stemmed = build_corpus(tmp_path / "stemmed", stemmer="spanish")
    assert gathered_ids(plain, "concesiones") == []
    assert gathered_ids(stemmed, "concesiones") == ["law-1", "law-4"]


def test_upsert_adds_and_replaces_and_carries_the_analyzer_forward(tmp_path: Path) -> None:
    index = build_corpus(tmp_path, stemmer="spanish")
    upsert(
        index,
        tmp_path / "added",
        (
            {"id": "law-5", "text": "nuevas concesiones de aguas"},
            {"id": "law-3", "text": "aguas subterráneas"},
        ),
    )
    assert gathered_ids(index, "concesion") == ["law-1", "law-4", "law-5"]
    assert gathered_ids(index, "aguas") == ["law-3", "law-5"]
    assert gathered_ids(index, "chair") == []
    view = cookbook.VectorStore(index / cookbook.VECTOR_DIRECTORY).view()
    assert view.document_ids == ["law-1", "law-2", "law-3", "law-4", "law-5"]


def test_delete_removes_documents_from_both_indexes(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    run_cookbook("delete", "--index", str(index), "--document-id", "law-2", "--document-id", "missing")
    assert [row["id"] for row in cookbook.load_documents(index / cookbook.DOCUMENTS_FILE)] == [
        "law-1", "law-3", "law-4",
    ]
    assert gathered_ids(index, "inscripción") == []
    assert gathered_ids(index, "concesión") == ["law-1", "law-4"]
    view = cookbook.VectorStore(index / cookbook.VECTOR_DIRECTORY).view()
    assert view.document_ids == ["law-1", "law-3", "law-4"]


def test_a_running_server_serves_what_the_indexer_commits_next(tmp_path: Path) -> None:
    index = build_corpus(tmp_path)
    gatherer = open_gatherer(index)
    store = cookbook.VectorStore(index / cookbook.VECTOR_DIRECTORY)
    pipeline = cookbook.SearchPipeline(gatherer, cookbook.MaxSimReranker([store]))
    query = cookbook.Query(
        "concesión", multi_vector=cookbook.Feature(store.representation, unit_vectors(2, seed=7))
    )

    def ids() -> set[str]:
        return {row.document_id for row in pipeline.search(query, gather_limit=10, limit=5).documents}

    assert ids() == {"law-1", "law-4"}
    upsert(index, tmp_path / "added", ({"id": "law-5", "text": "concesión de aguas"},))
    assert ids() == {"law-1", "law-4", "law-5"}
    run_cookbook("delete", "--index", str(index), "--document-id", "law-1")
    assert ids() == {"law-4", "law-5"}


def test_compaction_changes_no_result(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    index = build_corpus(tmp_path)
    upsert(index, tmp_path / "added", ({"id": "law-4", "text": "concesión renovada"},))
    run_cookbook("delete", "--index", str(index), "--document-id", "law-2")
    embeddings = query_embeddings(tmp_path)
    before = search(index, capsys, "--query", "concesión", "--query-embeddings", embeddings)
    run_cookbook("compact", "--index", str(index))
    after = search(index, capsys, "--query", "concesión", "--query-embeddings", embeddings)
    assert after["results"] == before["results"]


@pytest.mark.parametrize("storage", ["float32", "int8"])
def test_search_reranks_when_embeddings_are_given(tmp_path: Path, capsys, storage: str) -> None:  # type: ignore[no-untyped-def]
    index = build_corpus(tmp_path, storage=storage)
    output = search(index, capsys, "--query", "concesión", "--query-embeddings", query_embeddings(tmp_path))

    # Only the two documents mentioning the term are gathered: BM25 sets the ceiling.
    assert result_ids(output) == {"law-1", "law-4"}
    assert [row["rank"] for row in output["results"]] == [1, 2]
    assert output["diagnostics"]["reranker"] == "MaxSimReranker"
    assert output["diagnostics"]["score_semantics"] == cookbook.VectorStore(
        index / cookbook.VECTOR_DIRECTORY
    ).score_semantics


def test_search_without_embeddings_is_gather_only(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output = search(build_corpus(tmp_path), capsys, "--query", "concesión")
    assert result_ids(output) == {"law-1", "law-4"}
    assert output["diagnostics"]["reranker"] is None
    assert output["diagnostics"]["score_semantics"] == "bm25s-lucene"


def test_search_subset_restricts_by_document_id(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output = search(build_corpus(tmp_path), capsys, "--query", "concesión", "--subset-id", "law-4")
    assert [row["document_id"] for row in output["results"]] == ["law-4"]


def test_search_reports_the_older_commit_of_the_two_indexes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    index = build_corpus(tmp_path)
    output = search(index, capsys, "--query", "concesión", "--query-embeddings", query_embeddings(tmp_path))
    store = cookbook.VectorStore(index / cookbook.VECTOR_DIRECTORY).view().as_of
    lexical = open_gatherer(index).current().committed_at
    assert cookbook.datetime.fromisoformat(output["as_of"]) == min(store, lexical)


def test_search_refuses_embeddings_of_the_wrong_dimension(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    index = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="dimension"):
        search(index, capsys, "--query", "concesión", "--query-embeddings", query_embeddings(tmp_path, DIMENSION + 1))
