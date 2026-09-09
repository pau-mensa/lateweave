# Cookbook: BM25 gather, optional stored MaxSim rerank

This recipe maintains one retrieval index, a bm25s lexical index, plus a
lateweave vector store used to rerank BM25 candidates when the query supplies
token embeddings. It is the reference case for the substrate: a gatherer that
consumes only text, feeding a reranker that consumes a multi-vector feature.

bm25s is declared in the script's PEP 723 metadata rather than in lateweave's
package dependencies:

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py --help
```

## Layout

```text
index/
  corpus-manifest.json   CorpusManifest: which documents, which generation
  analyzer.json          lexical analysis chain, read back on every command
  documents.jsonl        external ID and text, in internal-ID order
  bm25/                  bm25s index
  vectors/               lateweave store; carries its Representation
```

The corpus manifest is the one identity both stages share. The store's
representation is what a query's embeddings are checked against.

## The lexical stage

Documents and queries are analyzed the same way: casefold, strip combining
marks, take `\w+` runs. `--stemmer` adds a Snowball algorithm (`spanish`,
`english`, ...) and defaults to none. The chain is persisted in `analyzer.json`
so an index cannot be queried through a different chain than its postings were
built with, which would return fewer documents with no error.

## Store choices

| `--storage` | Representation | Score semantics |
|---|---|---|
| `float32` | exact token vectors, `4D` bytes per token | `float32-exact-full-maxsim` |
| `int8` | row-wise symmetric INT8, `D + 4` bytes per token | `int8-reconstructed-approximate-full-maxsim` |

## Inputs

Documents are JSON Lines in stable internal order:

```json
{"id": "law-1", "text": "document text"}
{"id": "law-2", "text": "another document"}
```

Embeddings use two NumPy files: `embeddings.npy`, a finite unit-norm float32
`[total_tokens, dimension]` matrix packed by document, and
`document-lengths.npy`, a positive int64 `[documents]` vector summing to
`total_tokens`. Queries use a float32 `[query_tokens, dimension]` `.npy` file
from the same encoder.

## Build

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py build \
  --index var/laws \
  --documents var/documents.jsonl \
  --embeddings var/embeddings.npy \
  --document-lengths var/document-lengths.npy \
  --storage float32 \
  --corpus-id laws \
  --corpus-version 2026-09-01 \
  --encoder lightonai/LateOn-Code \
  --encoder-revision main \
  --threads 8
```

## Append and delete

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py update \
  --index var/laws \
  --documents var/new-documents.jsonl \
  --embeddings var/new-embeddings.npy \
  --document-lengths var/new-document-lengths.npy

uv run --with-editable . cookbook/bm25_stored_maxsim.py delete \
  --index var/laws \
  --document-id law-17 \
  --document-id law-42
```

Existing external IDs are rejected on append. After a delete, remaining
documents are renumbered 0..n-1 in document order. The lexical index has no
incremental path, so both commands rebuild it from `documents.jsonl`; the store
appends or compacts in place. Each mutation advances the corpus generation.

Build, append, and delete publish copy-on-write directory replacements while a
file lock excludes readers, so a failure cannot expose a BM25 generation paired
with a different store generation.

## Search

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py search \
  --index var/laws \
  --query "prescripción de una deuda tributaria" \
  --query-embeddings var/query.npy \
  --gather-limit 500 \
  --limit 100
```

Without `--query-embeddings` the search is gather-only and BM25 scores rank.
With them, only BM25 candidates are fetched from the store and scored by the
CPU MaxSim kernel. `--subset-id EXTERNAL_ID` (repeatable) restricts the search
to those documents; the gatherer honours it through bm25s's weight mask.
