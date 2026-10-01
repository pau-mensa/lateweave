# Cookbook: BM25 gather, optional stored MaxSim rerank

This recipe keeps one retrieval index, a bm25s lexical index plus a lateweave
vector store used to rerank BM25 candidates when the query supplies token
embeddings. It is the reference case for the substrate: a gatherer that
consumes only text, feeding a reranker that consumes a multi-vector feature,
over indexes an indexer keeps moving while a server searches them.

bm25s is declared in the script's PEP 723 metadata rather than in lateweave's
package dependencies:

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py --help
```

## Layout

```text
index/
  analyzer.json          lexical analysis chain, read back on every command
  documents.jsonl        the indexer's record of every document: ID and text
  bm25/
    current.json         corpus, live generation, and when it was committed
    <generation>/        bm25s index plus ids.json, its row order
  vectors/               lateweave vector store; carries its Representation
```

The two indexes share document IDs and nothing else. Each is committed on its
own: bm25 by renaming `current.json`, the store by renaming its
`manifest.json`. Neither knows the other's row order, and the server never
needs them to agree: it ranks only the documents both hold and reports the
older of the two commits as the answer's `as_of`.

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

Documents are JSON Lines:

```json
{"id": "law-1", "text": "document text"}
{"id": "law-2", "text": "another document"}
```

Embeddings use two NumPy files: `embeddings.npy`, a finite unit-norm float32
`[total_tokens, dimension]` matrix packed by document, and
`document-lengths.npy`, a positive int64 `[documents]` vector summing to
`total_tokens`. Queries use a float32 `[query_tokens, dimension]` `.npy` file
from the same encoder.

## The indexer

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py build \
  --index var/laws \
  --documents var/documents.jsonl \
  --embeddings var/embeddings.npy \
  --document-lengths var/document-lengths.npy \
  --storage float32 \
  --corpus-id laws \
  --encoder lightonai/LateOn-Code \
  --encoder-revision main \
  --threads 8

uv run --with-editable . cookbook/bm25_stored_maxsim.py upsert \
  --index var/laws \
  --documents var/new-documents.jsonl \
  --embeddings var/new-embeddings.npy \
  --document-lengths var/new-document-lengths.npy

uv run --with-editable . cookbook/bm25_stored_maxsim.py delete \
  --index var/laws \
  --document-id law-17 \
  --document-id law-42

uv run --with-editable . cookbook/bm25_stored_maxsim.py compact --index var/laws
```

`upsert` replaces any document with the same ID; `delete` ignores IDs it does
not hold. The store appends one segment or writes one tombstone file per
mutation and never rewrites existing vectors; `compact` merges them whenever
convenient and changes no result. bm25s has no incremental path, so every
mutation builds the next lexical generation from `documents.jsonl` and renames
`current.json` to publish it. A file lock serializes indexers; searches never
take it.

## The server

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
CPU MaxSim kernel; a candidate the store does not hold yet, or no longer, is
dropped and counted in `diagnostics.dropped`. `--subset-id ID` (repeatable)
restricts the search to those documents (`Subset().including(...)`); the
gatherer honours include and exclude restrictions alike through bm25s's weight
mask. The output's `as_of` is the older of the two indexes'
commits. A long-running server built the same way, as `LexicalCandidateGenerator` plus `VectorStore`, serves each commit on
its next search.
