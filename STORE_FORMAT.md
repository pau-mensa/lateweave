# The lateweave vector store format

A vector store holds the token vectors of one corpus's documents, keyed by
document ID. lateweave only reads stores. Any process that writes the files
below is a writer; `VectorStoreWriter` is one, and nothing depends on using it.

This document describes format `lateweave-vectors-2`.

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
  "format": "lateweave-vectors-2",
  "store_id": "5f0c8e2a9b1d4c7e8a3f6b2d1e9c4a70",
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
  "next_segment_id": 8,
  "committed_at": 1790000000.25,
  "segments": [
    {"id": 3, "documents": 1000, "tokens": 51234, "tombstones": 11},
    {"id": 5, "documents": 20, "tokens": 900, "tombstones": null}
  ]
}
```

| Field | Meaning |
|---|---|
| `store_id` | a random string chosen when the store is created; fixed for the life of the store |
| `encoding` | `"float32"` or `"int8"`; fixed for the life of the store |
| `corpus` | the corpus ID candidates name; fixed for the life of the store |
| `representation` | the encoder of the vectors; fixed for the life of the store |
| `commit` | increases with every commit; never wraps |
| `next_segment_id` | required allocation watermark; never decreases, and exceeds every live segment ID |
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

A writer stages files privately, finishes and syncs their contents, and installs
all new segment and tombstone files at their final paths without replacing
existing files. It syncs the store directory before publishing a manifest that
references those files. Filesystem support for hard links and directory syncing
is required by `VectorStoreWriter`.

The writer writes and syncs a new manifest, then replaces `manifest.json` by
atomic rename. This rename is the visibility point: until then readers see
nothing new. The writer syncs the store directory again before reporting a
durable commit or removing files that the new manifest no longer names. A
reader already mapping removed files keeps its view, and one that finds a
named file missing rereads the manifest.

`next_segment_id` preserves allocation history independently of the live
segments. Writers allocate ascending IDs starting at this watermark and publish
the updated watermark with the segment list, including for empty stores.
Compaction changes the live set without resetting allocation. Segment IDs and
commit numbers are unsigned 64-bit integers; exhaustion is an error.

Once a segment ID or commit number appears in a published manifest, it is never
reused for different content. Every published filename identifies immutable
bytes, even after the file is absent from the current manifest. Unreferenced
files may be unlinked, but must never be truncated or overwritten in place:
older readers may still map them. IDs used only by unpublished staging attempts
may be allocated again after recovery.

These guarantees hold within one `store_id`. A store deleted and created again
at the same path starts over at segment 0 and commit 0 under a new `store_id`,
so a reader whose manifest changes `store_id` discards everything it loaded.

Opening a writer establishes durability of the current manifest by syncing the
store directory before reclaiming unreferenced files and private staging
artifacts. Recovery uses the manifest's allocation watermark, never the live
segment list or the remaining filenames. If reclamation fails, an existing
filename blocks publication at that path rather than being replaced.

A failure before manifest replacement leaves the previous commit authoritative.
A failure syncing the directory after replacement means the new manifest is
visible but its durability is uncertain. The writer retains its referenced
files, adopts that manifest in memory, and reports an error. Retrying commit or
reopening the writer establishes durability before reclamation proceeds.

The allocation watermark is required; readers and writers reject manifests
without it. This version does not infer allocation history or migrate earlier
formats.

A writer should commit periodically even when it has nothing to write:
`committed_at` is how readers know the store is current.

One writer writes a store at a time.
