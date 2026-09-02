# Cookbook: BM25 + stored vectors + MaxSim

This recipe maintains one retrieval index: a bm25s lexical index. Its only other
persistent artifact is an optional lateweave vector store used to rerank BM25
candidates. There are no vector postings or second candidate index.

bm25s is declared in the script's PEP 723 metadata rather than in lateweave's
package dependencies:

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py --help
```

## The lexical stage

Documents and queries are analyzed the same way: casefold, strip combining
marks, take `\w+` runs. `--stemmer` adds a Snowball algorithm (`spanish`,
`portuguese`, `english`, ...) and defaults to none, which is language-agnostic.

The chain is recorded in the generator manifest's `build_parameters` and read
back on every open and every mutation, so an index cannot be queried through a
different chain than its postings were built with. Analysis silently diverging
from the postings is the one failure in a lexical stage that produces no error
and no warning — it just returns fewer documents.

## Store choices

| `--storage` | Representation | Intended use |
|---|---|---|
| `int8` | row-wise symmetric INT8 | fast, moderate-loss default |
| `turboquant4` | four-bit TurboQuant-MSE | maximum built-in reduction |
| `jzip` | spherical angles plus document-framed zstd | near-lossless fidelity |

INT8 uses `D + 4` bytes per token. TurboQuant uses `D / 2` bytes per token.
Jzip is variable-length; compression depends on dimension and the number of
token vectors in each document. Its reconstruction is below float32-scale
error in the published evaluation, but it is not bit-exact lossless.

## Inputs

Documents are JSON Lines in stable internal order:

```json
{"id": "law-1", "text": "document text"}
{"id": "law-2", "text": "another document"}
```

Embeddings use two NumPy files:

- `embeddings.npy`: finite, normalized float32
  `[total_tokens, dimension]`, packed by document;
- `document-lengths.npy`: positive int64 `[documents]` summing to
  `total_tokens`.

Queries use a normalized float32 `[query_tokens, dimension]` `.npy` file.

## Build

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py build \
  --index var/laws \
  --documents var/documents.jsonl \
  --embeddings var/embeddings.npy \
  --document-lengths var/document-lengths.npy \
  --storage jzip \
  --compression-level 1 \
  --corpus-id laws \
  --corpus-version 2026-09-01 \
  --encoder lightonai/mLateOn \
  --encoder-revision main \
  --tokenizer lightonai/mLateOn \
  --threads 8
```

`--compression-level` applies only to jzip and accepts zstd levels 1–22.
Level 1 is recommended: the spherical transform already supplies most of the
entropy reduction, while higher levels cost substantially more build time.

## Append

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py update \
  --index var/laws \
  --documents var/new-documents.jsonl \
  --embeddings var/new-embeddings.npy \
  --document-lengths var/new-document-lengths.npy \
  --threads 8
```

Existing external IDs are rejected. Fixed-record stores create replacement
arrays; jzip appends newly compressed document frames without touching existing
frames. The lexical index has no incremental append, so it is rebuilt over the
whole corpus from `documents.jsonl` — a few seconds for tens of thousands of
documents, and O(corpus) rather than O(delta).

## Delete

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py delete \
  --index var/laws \
  --document-id law-17 \
  --document-id law-42
```

Remaining documents are renumbered 0..n-1 in document order. The lexical index
is rebuilt over the survivors, so it lands on that numbering by construction
rather than by two implementations agreeing. INT8 and TurboQuant copy live token
records. Jzip copies live compressed frames without decoding or recompressing
them.

## Search

```bash
uv run --with-editable . cookbook/bm25_stored_maxsim.py search \
  --index var/laws \
  --query "prescripción de una deuda tributaria" \
  --query-embeddings var/query.npy \
  --gather-limit 500 \
  --limit 100 \
  --max-batch-tokens 131072 \
  --max-documents-per-batch 256 \
  --threads 8
```

Only BM25 candidates transition out of the store. Reconstructed token vectors
are packed into bounded batches and passed to the fused CPU MaxSim kernel.

Build, append, and delete publish copy-on-write directory replacements while a
file lock excludes readers. A failure cannot expose a BM25 generation paired
with a different vector-store generation.
