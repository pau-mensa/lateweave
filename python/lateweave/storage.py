"""Multi-vector document stores owned by lateweave.

A store is one :class:`MultiVectorSource`: it holds every token vector of every
document for one :class:`Representation` and hands back packed float32 rows for
a candidate set. The MaxSim reranker consumes any source; an engine that already
holds document vectors can implement the protocol itself and never write a
store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import gc
import json
from pathlib import Path
import shutil
from typing import ClassVar, Mapping, Protocol, Sequence, runtime_checkable
import uuid

import numpy as np

from ._native import _storage_int8_decode, _storage_int8_encode
from .manifest import Representation


METADATA_FILE = "storage.json"
OFFSETS_FILE = "document-offsets.npy"


@runtime_checkable
class MultiVectorSource(Protocol):
    """Token vectors of documents, fetched by internal ID.

    ``fetch`` returns a contiguous float32 ``[tokens, dimension]`` matrix holding
    the requested documents in the requested order, plus their int64 lengths.
    ``score_semantics`` qualifies what MaxSim over those vectors means, since a
    source may reconstruct from a lossy code.
    """

    representation: Representation
    score_semantics: str
    document_count: int

    def document_lengths(self, document_ids: Sequence[int]) -> dict[int, int]: ...

    def fetch(
        self, document_ids: Sequence[int], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _validate_embeddings(
    embeddings: np.ndarray, lengths: np.ndarray, representation: Representation
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings)
    lengths = np.asarray(lengths, dtype=np.int64)
    if embeddings.dtype != np.float32 or embeddings.ndim != 2:
        raise ValueError("embeddings must be a float32 [tokens, dimension] array")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must contain at least one token")
    if embeddings.shape[1] != representation.dimension:
        raise ValueError("embedding dimension differs from the representation")
    if lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths <= 0):
        raise ValueError("document lengths must be a non-empty positive vector")
    if int(lengths.sum()) != len(embeddings):
        raise ValueError("document lengths do not match packed embedding rows")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain a non-finite value")
    if representation.normalized and not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-3, atol=1e-4
    ):
        raise ValueError("representation declares unit vectors, but rows are not normalized")
    return embeddings, lengths


def _offsets(lengths: np.ndarray) -> np.ndarray:
    output = np.empty(len(lengths) + 1, dtype=np.uint64)
    output[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=output[1:])
    return output


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.npy"
    np.save(temporary, values)
    temporary.replace(path)


class FixedRecordVectorStore(ABC):
    """Memory-mapped fixed-width token records, one file per array."""

    format: ClassVar[str]
    score_semantics: ClassVar[str]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._load()

    # -- layout ---------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _arrays(cls, dimension: int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
        """Array name to (dtype, per-token shape)."""

    @classmethod
    @abstractmethod
    def _encode(cls, embeddings: np.ndarray, *, threads: int | None) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def _decode(self, arrays: Mapping[str, np.ndarray], *, threads: int | None) -> np.ndarray: ...

    # -- lifecycle ------------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        embeddings: np.ndarray,
        document_lengths: np.ndarray,
        representation: Representation,
        *,
        chunk_tokens: int = 131_072,
        threads: int | None = None,
    ) -> "FixedRecordVectorStore":
        path = Path(path)
        embeddings, document_lengths = _validate_embeddings(
            embeddings, document_lengths, representation
        )
        if path.exists():
            raise FileExistsError(f"vector store already exists: {path}")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        path.mkdir(parents=True)
        try:
            arrays = cls._open_new(path, len(embeddings), representation.dimension, "")
            cls._encode_into(embeddings, arrays, 0, chunk_tokens, threads)
            _flush(arrays)
            np.save(path / OFFSETS_FILE, _offsets(document_lengths))
            cls._write_metadata(path, representation, len(document_lengths), len(embeddings))
        except BaseException:
            shutil.rmtree(path)
            raise
        return cls(path)

    def _load(self) -> None:
        metadata = json.loads((self.path / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("format") != self.format:
            raise ValueError(f"{self.path} is not a {self.format} store")
        self.representation = Representation.from_dict(metadata["representation"])
        self.dimension = self.representation.dimension
        self.document_count = int(metadata["document_count"])
        self.token_count = int(metadata["token_count"])
        self.offsets = np.load(self.path / OFFSETS_FILE, mmap_mode="r")
        if (
            self.offsets.dtype != np.uint64
            or self.offsets.shape != (self.document_count + 1,)
            or int(self.offsets[0]) != 0
            or int(self.offsets[-1]) != self.token_count
            or np.any(self.offsets[1:] <= self.offsets[:-1])
        ):
            raise ValueError("vector-store document offsets are invalid")
        self._store = {}
        for name, (dtype, tail) in self._arrays(self.dimension).items():
            array = np.load(self.path / f"{name}.npy", mmap_mode="r")
            if array.dtype != dtype or array.shape != (self.token_count, *tail):
                raise ValueError(f"vector-store array {name!r} has the wrong shape or dtype")
            self._store[name] = array

    def _close(self) -> None:
        del self._store, self.offsets
        gc.collect()

    @classmethod
    def _write_metadata(
        cls, path: Path, representation: Representation, documents: int, tokens: int
    ) -> None:
        metadata = {
            "format": cls.format,
            "representation": representation.to_dict(),
            "document_count": documents,
            "token_count": tokens,
        }
        temporary = path / f".{METADATA_FILE}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary.replace(path / METADATA_FILE)

    @classmethod
    def _open_new(
        cls, path: Path, tokens: int, dimension: int, suffix: str
    ) -> dict[str, np.memmap]:
        return {
            name: np.lib.format.open_memmap(
                path / f"{name}{suffix}.npy", mode="w+", dtype=dtype, shape=(tokens, *tail)
            )
            for name, (dtype, tail) in cls._arrays(dimension).items()
        }

    def _publish(self, suffix: str) -> None:
        for name in self._arrays(self.dimension):
            (self.path / f"{name}{suffix}.npy").replace(self.path / f"{name}.npy")

    @classmethod
    def _encode_into(
        cls,
        embeddings: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        offset: int,
        chunk_tokens: int,
        threads: int | None,
    ) -> None:
        for first in range(0, len(embeddings), chunk_tokens):
            last = min(len(embeddings), first + chunk_tokens)
            chunk = np.ascontiguousarray(embeddings[first:last], dtype=np.float32)
            for name, values in cls._encode(chunk, threads=threads).items():
                arrays[name][offset + first : offset + last] = values

    # -- MultiVectorSource ----------------------------------------------------

    def _validate_document_ids(self, document_ids: Sequence[int]) -> None:
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document IDs must be unique")
        if any(not 0 <= item < self.document_count for item in document_ids):
            raise ValueError("document ID is outside the vector store")

    def document_lengths(self, document_ids: Sequence[int]) -> dict[int, int]:
        self._validate_document_ids(document_ids)
        return {
            item: int(self.offsets[item + 1] - self.offsets[item]) for item in document_ids
        }

    def fetch(
        self, document_ids: Sequence[int], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        lengths = self.document_lengths(document_ids)
        gathered = {
            name: np.empty((sum(lengths.values()), *array.shape[1:]), dtype=array.dtype)
            for name, array in self._store.items()
        }
        position = 0
        for document_id, length in lengths.items():
            first = int(self.offsets[document_id])
            for name, array in self._store.items():
                gathered[name][position : position + length] = array[first : first + length]
            position += length
        return (
            self._decode(gathered, threads=threads),
            np.fromiter(lengths.values(), dtype=np.int64, count=len(lengths)),
        )

    # -- mutation -------------------------------------------------------------

    def append(
        self,
        embeddings: np.ndarray,
        document_lengths: np.ndarray,
        *,
        chunk_tokens: int = 131_072,
        copy_chunk_tokens: int = 1_000_000,
        threads: int | None = None,
    ) -> None:
        embeddings, document_lengths = _validate_embeddings(
            embeddings, document_lengths, self.representation
        )
        if chunk_tokens <= 0 or copy_chunk_tokens <= 0:
            raise ValueError("chunk sizes must be positive")
        old_tokens = self.token_count
        new_tokens = old_tokens + len(embeddings)
        suffix = f".{uuid.uuid4().hex}"
        arrays = self._open_new(self.path, new_tokens, self.dimension, suffix)
        for first in range(0, old_tokens, copy_chunk_tokens):
            last = min(old_tokens, first + copy_chunk_tokens)
            for name, source in self._store.items():
                arrays[name][first:last] = source[first:last]
        self._encode_into(embeddings, arrays, old_tokens, chunk_tokens, threads)
        _flush(arrays)
        offsets = np.concatenate([self.offsets, _offsets(document_lengths)[1:] + old_tokens])
        documents = self.document_count + len(document_lengths)
        self._close()
        self._publish(suffix)
        _save_array_atomic(self.path / OFFSETS_FILE, offsets)
        self._write_metadata(self.path, self.representation, documents, new_tokens)
        self._load()

    def delete(
        self, document_ids: Sequence[int], *, copy_chunk_tokens: int = 1_000_000
    ) -> None:
        self._validate_document_ids(document_ids)
        if not document_ids:
            return
        if len(document_ids) == self.document_count:
            raise ValueError("delete cannot remove every vector-store document")
        if copy_chunk_tokens <= 0:
            raise ValueError("copy_chunk_tokens must be positive")
        deleted = set(document_ids)
        retained = [item for item in range(self.document_count) if item not in deleted]
        lengths = np.asarray(
            [int(self.offsets[item + 1] - self.offsets[item]) for item in retained],
            dtype=np.int64,
        )
        tokens = int(lengths.sum())
        suffix = f".{uuid.uuid4().hex}"
        arrays = self._open_new(self.path, tokens, self.dimension, suffix)
        output = 0
        for document_id, length in zip(retained, lengths.tolist(), strict=True):
            source = int(self.offsets[document_id])
            for relative in range(0, length, copy_chunk_tokens):
                count = min(copy_chunk_tokens, length - relative)
                for name, array in self._store.items():
                    arrays[name][output + relative : output + relative + count] = array[
                        source + relative : source + relative + count
                    ]
            output += length
        _flush(arrays)
        self._close()
        self._publish(suffix)
        _save_array_atomic(self.path / OFFSETS_FILE, _offsets(lengths))
        self._write_metadata(self.path, self.representation, len(retained), tokens)
        self._load()


def _flush(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()
    gc.collect()


class Float32VectorStore(FixedRecordVectorStore):
    """Exact float32 token vectors."""

    format = "lateweave-float32-v1"
    score_semantics = "float32-exact-full-maxsim"

    @classmethod
    def _arrays(cls, dimension: int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
        return {"vectors": (np.dtype(np.float32), (dimension,))}

    @classmethod
    def _encode(cls, embeddings: np.ndarray, *, threads: int | None) -> dict[str, np.ndarray]:
        return {"vectors": embeddings}

    def _decode(self, arrays: Mapping[str, np.ndarray], *, threads: int | None) -> np.ndarray:
        return arrays["vectors"]


class Int8VectorStore(FixedRecordVectorStore):
    """Symmetric INT8 per token with one float32 row scale; lossy."""

    format = "lateweave-int8-rowwise-v1"
    score_semantics = "int8-reconstructed-approximate-full-maxsim"

    @classmethod
    def _arrays(cls, dimension: int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
        return {
            "codes": (np.dtype(np.int8), (dimension,)),
            "scales": (np.dtype(np.float32), ()),
        }

    @classmethod
    def _encode(cls, embeddings: np.ndarray, *, threads: int | None) -> dict[str, np.ndarray]:
        codes, scales = _storage_int8_encode(embeddings, threads=threads)
        return {"codes": codes, "scales": scales}

    def _decode(self, arrays: Mapping[str, np.ndarray], *, threads: int | None) -> np.ndarray:
        return _storage_int8_decode(
            arrays["codes"],
            arrays["scales"],
            normalize=self.representation.normalized,
            threads=threads,
        )


def open_vector_store(path: str | Path) -> FixedRecordVectorStore:
    path = Path(path)
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    implementations = {
        Float32VectorStore.format: Float32VectorStore,
        Int8VectorStore.format: Int8VectorStore,
    }
    try:
        return implementations[str(metadata["format"])](path)
    except KeyError as error:
        raise ValueError(f"unsupported vector-store format {metadata.get('format')!r}") from error
