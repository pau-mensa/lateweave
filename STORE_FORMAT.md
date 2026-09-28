# The lateweave vector store format

A vector store holds the token vectors of one corpus's documents, keyed by
document ID. lateweave only reads stores. Any process that writes the files
below is a writer; `VectorStoreWriter` is one, and nothing depends on using it.

This document describes format `lateweave-vectors-1`.

## Files

```text
vectors/
  manifest.json                     the only file that ever changes
  segment-<S>.ids.json              JSON array of document IDs, in row order
  segment-<S>.offsets.npy           uint64 [rows + 1]
  segment-<S>.vectors.npy           float32 [tokens, D]        encoding "float32"
  segment-<S>.codes.npy             int8    [tokens, D]        encoding "int8"
  segment-<S>.scales.npy            float32 [tokens]           encoding "int8"
  segment-<S>.tombstones-<C>.npy    uint64  [deleted rows]
```

Arrays are NumPy `.npy` files, little-endian and C-ordered, as `numpy.save`
writes them. `<S>` is a segment ID and `<C>` a commit number, both decimal.

**A segment** is written once and never modified. Row `r` is document
`ids[r]`, whose tokens are rows `offsets[r]..offsets[r + 1]` of the token
arrays. `offsets[0]` is 0, offsets strictly increase (every document has at
least one token), and the last offset is the segment's token count. IDs are
unique within a segment.

- `float32`: `vectors` holds the token vectors as given.
- `int8`: token `t` is `codes[t] * scales[t]`, where `scales[t]` is the
  token's largest absolute value divided by 127 (at least `1e-12`) and
  `codes[t]` the rounded
  quotient, clamped to `[-127, 127]`. Readers renormalize decoded tokens when
  the representation is `normalized`.

**A tombstone file** lists, strictly ascending, the rows of one segment that
are deleted. It is written once; a later delete in the same segment writes a
new file under a new commit number, listing every deleted row.

## `manifest.json`

```json
{
  "format": "lateweave-vectors-1",
  "encoding": "float32",
  "corpus": "laws",
  "representation": {
    "encoder": "lightonai/LateOn-Code",
    "encoder_revision": "main",
    "dimension": 128,
    "normalized": true,
    "query_template": "",
    "document_template": ""
  },
  "commit": 12,
  "committed_at": 1790000000.25,
  "segments": [
    {"id": 3, "documents": 1000, "tokens": 51234, "tombstones": 11},
    {"id": 5, "documents": 20, "tokens": 900, "tombstones": null}
  ]
}
```

| Field | Meaning |
|---|---|
| `encoding` | `"float32"` or `"int8"`; fixed for the life of the store |
| `corpus` | the corpus ID candidates name; fixed for the life of the store |
| `representation` | the encoder of the vectors; fixed for the life of the store |
| `commit` | increases with every commit |
| `committed_at` | seconds since the Unix epoch at which the writer committed: every write made before it is in this manifest |
| `segments` | the live segments, oldest first, with strictly ascending IDs |
| `segments[].documents`, `segments[].tokens` | the segment's row and token counts |
| `segments[].tombstones` | the commit number of the segment's tombstone file, or `null` |

## Which document is present

For a document ID, find the **last** segment in `segments` whose IDs contain it.
The document is present with that row's vectors unless the row is listed in
that segment's tombstones. Older rows of the same ID never matter. So:

- appending a document, new or already present, is writing a segment that
  holds it;
- deleting a document is tombstoning its row in the last segment that holds it;
- compacting is writing one segment of the present documents and dropping the
  rest from `segments`. It changes no read.

## Committing

A writer makes every new file durable, then replaces `manifest.json` by
atomic rename; until the rename, readers see nothing new. After it, files the
new manifest no longer names may be removed. A reader already mapping them
keeps its view, and one that finds a named file missing rereads the manifest.

Segment IDs only grow and are never reused, and neither are commit numbers,
so a file name always means the same content. A file named by no manifest is
garbage, such as one a writer left staged when it stopped before committing;
the next writer may remove or overwrite it.

A writer should commit periodically even when it has nothing to write:
`committed_at` is how readers know the store is current, and a search with a
maximum lag refuses a store that has not committed within it.

One writer writes a store at a time.
