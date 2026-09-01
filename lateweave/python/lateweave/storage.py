"""Optional vector stores for generators that do not own scorer representations.

Storage is not part of the CandidateGenerator/CandidateScorer contract. Each
store owns its encoding, files, transition into scoring space, and mutations.
Native engines can ignore this module entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import gc
import json
import os
from pathlib import Path
import shutil
from typing import ClassVar, Mapping, Sequence
import uuid

import numpy as np

from ._native import (
    _storage_int8_decode,
    _storage_int8_encode,
    _storage_jzip_decode_documents,
    _storage_jzip_encode_documents,
    _storage_turboquant4_decode_rotated,
    _storage_turboquant4_encode,
    _storage_turboquant_rotate,
)


METADATA_FILE = "storage.json"
OFFSETS_FILE = "document-offsets.npy"


def _validate_embeddings(
    embeddings: np.ndarray, lengths: np.ndarray, dimension: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings)
    lengths = np.asarray(lengths, dtype=np.int64)
    if embeddings.dtype != np.float32 or embeddings.ndim != 2:
        raise ValueError("embeddings must be a float32 [tokens, dimension] array")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have non-zero axes")
    if dimension is not None and embeddings.shape[1] != dimension:
        raise ValueError("embedding dimension differs from the vector store")
    if lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths <= 0):
        raise ValueError("document lengths must be a non-empty positive vector")
    if int(lengths.sum()) != len(embeddings):
        raise ValueError("document lengths do not match packed embedding rows")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain a non-finite value")
    if not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-3, atol=1e-4
    ):
        raise ValueError("vector stores currently require normalized embeddings")
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


def _document_batches(lengths: np.ndarray, maximum_tokens: int):
    first = 0
    tokens = 0
    for position, raw_length in enumerate(lengths):
        length = int(raw_length)
        if position > first and tokens + length > maximum_tokens:
            yield first, position
            first = position
            tokens = 0
        tokens += length
    yield first, len(lengths)


class VectorStore(ABC):
    """Behavioral base for optional lateweave-owned vector stores.

    This class is deliberately absent from the stable search protocols. It is
    an implementation substrate for :class:`StoredMaxSimScorer`; backends may
    use fixed records, compressed frames, a database, or another private
    physical layout.
    """

    format: ClassVar[str]
    score_semantics: ClassVar[str]
    scoring_space: ClassVar[str]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._load()

    @classmethod
    @abstractmethod
    def create(
        cls,
        path: str | Path,
        embeddings: np.ndarray,
        document_lengths: np.ndarray,
        *,
        chunk_tokens: int = 131_072,
        threads: int | None = None,
    ) -> "VectorStore": ...

    @classmethod
    def _write_metadata(cls, path: Path, metadata: dict[str, object]) -> None:
        temporary = path / f".{METADATA_FILE}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path / METADATA_FILE)

    @classmethod
    def _metadata(
        cls, dimension: int, document_count: int, token_count: int
    ) -> dict[str, object]:
        return {
            "format": cls.format,
            "version": 1,
            "dimension": int(dimension),
            "document_count": int(document_count),
            "token_count": int(token_count),
            "normalized": True,
            "scoring_space": cls.scoring_space,
        }

    @classmethod
    def _validate_dimension(cls, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")

    def _load(self) -> None:
        metadata = json.loads((self.path / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("format") != self.format or metadata.get("version") != 1:
            raise ValueError(f"{self.path} is not a supported {self.format} store")
        self.dimension = int(metadata["dimension"])
        self.document_count = int(metadata["document_count"])
        self.token_count = int(metadata["token_count"])
        self.normalized = bool(metadata["normalized"])
        if metadata.get("scoring_space") != self.scoring_space:
            raise ValueError("vector-store scoring space is incompatible")
        self._validate_dimension(self.dimension)
        self.offsets = np.load(self.path / OFFSETS_FILE, mmap_mode="r")
        if self.offsets.dtype != np.uint64 or self.offsets.shape != (
            self.document_count + 1,
        ):
            raise ValueError("vector-store document offsets are invalid")
        if (
            int(self.offsets[0]) != 0
            or int(self.offsets[-1]) != self.token_count
            or np.any(self.offsets[1:] <= self.offsets[:-1])
        ):
            raise ValueError("vector-store document offsets are invalid")
        self._load_backend(metadata)

    @abstractmethod
    def _load_backend(self, metadata: Mapping[str, object]) -> None: ...

    def document_lengths(self, document_ids: Sequence[int]) -> dict[int, int]:
        self._validate_document_ids(document_ids)
        return {
            document_id: int(
                self.offsets[document_id + 1] - self.offsets[document_id]
            )
            for document_id in document_ids
        }

    def _validate_document_ids(self, document_ids: Sequence[int]) -> None:
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document IDs must be unique")
        if any(not 0 <= item < self.document_count for item in document_ids):
            raise ValueError("document ID is outside the vector store")

    @abstractmethod
    def prepare_query(
        self, embeddings: np.ndarray, *, threads: int | None = None
    ) -> np.ndarray: ...

    @abstractmethod
    def transition(
        self, document_ids: Sequence[int], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    @abstractmethod
    def encoded_bytes_per_token(self) -> float: ...

    def estimated_workspace_bytes(self, token_count: int, query_tokens: int) -> int:
        return int(
            token_count
            * (
                self.encoded_bytes_per_token
                + self.dimension * np.dtype(np.float32).itemsize
                + query_tokens * np.dtype(np.float32).itemsize
            )
        )

    @abstractmethod
    def append(
        self,
        embeddings: np.ndarray,
        document_lengths: np.ndarray,
        *,
        chunk_tokens: int = 131_072,
        copy_chunk_tokens: int = 1_000_000,
        threads: int | None = None,
    ) -> None: ...

    @abstractmethod
    def delete(
        self,
        document_ids: Sequence[int],
        *,
        copy_chunk_tokens: int = 1_000_000,
    ) -> None: ...

    def _close(self) -> None:
        self._close_backend()
        del self.offsets
        gc.collect()

    @abstractmethod
    def _close_backend(self) -> None: ...


class FixedRecordVectorStore(VectorStore):
    """Shared physical implementation for fixed-width per-token records."""

    @classmethod
    def create(
        cls,
        path: str | Path,
        embeddings: np.ndarray,
        document_lengths: np.ndarray,
        *,
        chunk_tokens: int = 131_072,
        threads: int | None = None,
    ) -> "FixedRecordVectorStore":
        path = Path(path)
        embeddings, document_lengths = _validate_embeddings(
            embeddings, document_lengths
        )
        if path.exists():
            raise FileExistsError(f"vector store already exists: {path}")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        cls._validate_dimension(embeddings.shape[1])
        path.mkdir(parents=True)
        try:
            cls._write_metadata(
                path,
                cls._metadata(
                    embeddings.shape[1], len(document_lengths), len(embeddings)
                ),
            )
            np.save(path / OFFSETS_FILE, _offsets(document_lengths))
            arrays = cls._create_arrays(path, len(embeddings), embeddings.shape[1])
            cls._encode_into(
                embeddings,
                arrays,
                output_offset=0,
                chunk_tokens=chunk_tokens,
                threads=threads,
            )
            for array in arrays.values():
                array.flush()
            del arrays
            gc.collect()
        except BaseException:
            shutil.rmtree(path)
            raise
        return cls(path)

    @classmethod
    @abstractmethod
    def _create_arrays(
        cls, path: Path, token_count: int, dimension: int, *, suffix: str = ""
    ) -> dict[str, np.memmap]: ...

    @classmethod
    @abstractmethod
    def _encode_chunk(
        cls, embeddings: np.ndarray, *, threads: int | None
    ) -> dict[str, np.ndarray]: ...

    @classmethod
    def _encode_into(
        cls,
        embeddings: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        *,
        output_offset: int,
        chunk_tokens: int,
        threads: int | None,
    ) -> None:
        for first in range(0, len(embeddings), chunk_tokens):
            last = min(len(embeddings), first + chunk_tokens)
            chunk = np.ascontiguousarray(embeddings[first:last], dtype=np.float32)
            encoded = cls._encode_chunk(chunk, threads=threads)
            destination = slice(output_offset + first, output_offset + last)
            for name, values in encoded.items():
                arrays[name][destination] = values

    def _load_backend(self, metadata: Mapping[str, object]) -> None:
        del metadata
        self._arrays = self._open_arrays()

    @abstractmethod
    def _open_arrays(self) -> dict[str, np.ndarray]: ...

    def _gather_arrays(self, document_ids: Sequence[int]) -> dict[str, np.ndarray]:
        self._validate_document_ids(document_ids)
        token_count = sum(
            int(self.offsets[item + 1] - self.offsets[item]) for item in document_ids
        )
        output = {
            name: np.empty((token_count, *array.shape[1:]), dtype=array.dtype)
            for name, array in self._arrays.items()
        }
        position = 0
        for document_id in document_ids:
            first = int(self.offsets[document_id])
            last = int(self.offsets[document_id + 1])
            length = last - first
            for name, array in self._arrays.items():
                output[name][position : position + length] = array[first:last]
            position += length
        return output

    @abstractmethod
    def _decode(
        self, arrays: Mapping[str, np.ndarray], *, threads: int | None
    ) -> np.ndarray: ...

    def transition(
        self, document_ids: Sequence[int], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray(
            list(self.document_lengths(document_ids).values()), dtype=np.int64
        )
        arrays = self._gather_arrays(document_ids)
        return self._decode(arrays, threads=threads), lengths

    @property
    def encoded_bytes_per_token(self) -> float:
        return float(sum(array[0].nbytes for array in self._arrays.values()))

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
            embeddings, document_lengths, self.dimension
        )
        if chunk_tokens <= 0 or copy_chunk_tokens <= 0:
            raise ValueError("chunk sizes must be positive")
        old_tokens = self.token_count
        new_tokens = old_tokens + len(embeddings)
        suffix = f".{uuid.uuid4().hex}.npy"
        arrays = self._create_arrays(
            self.path, new_tokens, self.dimension, suffix=suffix
        )
        for first in range(0, old_tokens, copy_chunk_tokens):
            last = min(old_tokens, first + copy_chunk_tokens)
            for name, source in self._arrays.items():
                arrays[name][first:last] = source[first:last]
        self._encode_into(
            embeddings,
            arrays,
            output_offset=old_tokens,
            chunk_tokens=chunk_tokens,
            threads=threads,
        )
        new_offsets = np.empty(
            self.document_count + len(document_lengths) + 1, dtype=np.uint64
        )
        new_offsets[: self.document_count + 1] = self.offsets
        np.cumsum(
            document_lengths,
            dtype=np.uint64,
            out=new_offsets[self.document_count + 1 :],
        )
        new_offsets[self.document_count + 1 :] += old_tokens
        for array in arrays.values():
            array.flush()
        del arrays
        self._close()
        self._publish_arrays(suffix)
        _save_array_atomic(self.path / OFFSETS_FILE, new_offsets)
        self._write_metadata(
            self.path,
            self._metadata(
                self.dimension,
                self.document_count + len(document_lengths),
                new_tokens,
            ),
        )
        self._load()

    def delete(
        self,
        document_ids: Sequence[int],
        *,
        copy_chunk_tokens: int = 1_000_000,
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
        lengths = [
            int(self.offsets[item + 1] - self.offsets[item]) for item in retained
        ]
        token_count = sum(lengths)
        suffix = f".{uuid.uuid4().hex}.npy"
        arrays = self._create_arrays(
            self.path, token_count, self.dimension, suffix=suffix
        )
        output = 0
        for document_id, length in zip(retained, lengths, strict=True):
            source = int(self.offsets[document_id])
            for relative in range(0, length, copy_chunk_tokens):
                count = min(copy_chunk_tokens, length - relative)
                for name, source_array in self._arrays.items():
                    arrays[name][output + relative : output + relative + count] = (
                        source_array[source + relative : source + relative + count]
                    )
            output += length
        for array in arrays.values():
            array.flush()
        del arrays
        new_offsets = _offsets(np.asarray(lengths, dtype=np.int64))
        self._close()
        self._publish_arrays(suffix)
        _save_array_atomic(self.path / OFFSETS_FILE, new_offsets)
        self._write_metadata(
            self.path,
            self._metadata(self.dimension, len(retained), token_count),
        )
        self._load()

    def _close_backend(self) -> None:
        del self._arrays

    @abstractmethod
    def _publish_arrays(self, suffix: str) -> None: ...


class Int8VectorStore(FixedRecordVectorStore):
    """Per-token symmetric INT8 storage with one float32 row scale."""

    format = "lateweave-int8-rowwise-v1"
    score_semantics = "int8-reconstructed-approximate-full-maxsim"
    scoring_space = "encoder"
    CODES_FILE = "codes.npy"
    SCALES_FILE = "scales.npy"

    @classmethod
    def _create_arrays(
        cls, path: Path, token_count: int, dimension: int, *, suffix: str = ""
    ) -> dict[str, np.memmap]:
        return {
            "codes": np.lib.format.open_memmap(
                path / f"codes{suffix or '.npy'}",
                mode="w+",
                dtype=np.int8,
                shape=(token_count, dimension),
            ),
            "scales": np.lib.format.open_memmap(
                path / f"scales{suffix or '.npy'}",
                mode="w+",
                dtype=np.float32,
                shape=(token_count,),
            ),
        }

    @classmethod
    def _encode_chunk(
        cls, embeddings: np.ndarray, *, threads: int | None
    ) -> dict[str, np.ndarray]:
        codes, scales = _storage_int8_encode(embeddings, threads=threads)
        return {"codes": codes, "scales": scales}

    def _open_arrays(self) -> dict[str, np.ndarray]:
        codes = np.load(self.path / self.CODES_FILE, mmap_mode="r")
        scales = np.load(self.path / self.SCALES_FILE, mmap_mode="r")
        if codes.dtype != np.int8 or codes.shape != (self.token_count, self.dimension):
            raise ValueError("INT8 code storage has the wrong shape or dtype")
        if scales.dtype != np.float32 or scales.shape != (self.token_count,):
            raise ValueError("INT8 scale storage has the wrong shape or dtype")
        return {"codes": codes, "scales": scales}

    def prepare_query(
        self, embeddings: np.ndarray, *, threads: int | None = None
    ) -> np.ndarray:
        del threads
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def _decode(
        self, arrays: Mapping[str, np.ndarray], *, threads: int | None
    ) -> np.ndarray:
        return _storage_int8_decode(
            arrays["codes"], arrays["scales"], normalize=True, threads=threads
        )

    def _publish_arrays(self, suffix: str) -> None:
        (self.path / f"codes{suffix}").replace(self.path / self.CODES_FILE)
        (self.path / f"scales{suffix}").replace(self.path / self.SCALES_FILE)


class TurboQuantVectorStore(FixedRecordVectorStore):
    """Training-free TurboQuant-MSE 4-bit storage in rotated scoring space.

    Reconstruction is approximate. The store rotates full-precision queries
    into the same orthogonal space and reconstructs normalized Lloyd-Max values
    there, avoiding an inverse transform on every candidate token.
    """

    format = "lateweave-turboquant4-mse-v1"
    score_semantics = "turboquant4-rotated-approximate-full-maxsim"
    scoring_space = "turboquant4-rotation-v1"
    CODES_FILE = "codes.npy"

    @classmethod
    def _validate_dimension(cls, dimension: int) -> None:
        super()._validate_dimension(dimension)
        if dimension % 2:
            raise ValueError("TurboQuantVectorStore requires an even dimension")

    @classmethod
    def _create_arrays(
        cls, path: Path, token_count: int, dimension: int, *, suffix: str = ""
    ) -> dict[str, np.memmap]:
        return {
            "codes": np.lib.format.open_memmap(
                path / f"codes{suffix or '.npy'}",
                mode="w+",
                dtype=np.uint8,
                shape=(token_count, dimension // 2),
            )
        }

    @classmethod
    def _encode_chunk(
        cls, embeddings: np.ndarray, *, threads: int | None
    ) -> dict[str, np.ndarray]:
        return {"codes": _storage_turboquant4_encode(embeddings, threads=threads)}

    def _open_arrays(self) -> dict[str, np.ndarray]:
        codes = np.load(self.path / self.CODES_FILE, mmap_mode="r")
        if codes.dtype != np.uint8 or codes.shape != (
            self.token_count,
            self.dimension // 2,
        ):
            raise ValueError("TurboQuant code storage has the wrong shape or dtype")
        return {"codes": codes}

    def prepare_query(
        self, embeddings: np.ndarray, *, threads: int | None = None
    ) -> np.ndarray:
        return _storage_turboquant_rotate(
            np.ascontiguousarray(embeddings, dtype=np.float32), threads=threads
        )

    def _decode(
        self, arrays: Mapping[str, np.ndarray], *, threads: int | None
    ) -> np.ndarray:
        return _storage_turboquant4_decode_rotated(
            arrays["codes"], self.dimension, normalize=True, threads=threads
        )

    def _publish_arrays(self, suffix: str) -> None:
        (self.path / f"codes{suffix}").replace(self.path / self.CODES_FILE)


class JzipVectorStore(VectorStore):
    """Near-lossless spherical-coordinate zstd frames, one per document.

    The representation is intentionally document-framed rather than compatible
    with the upstream jzip CLI. This gives candidate-level random access and
    makes update/delete policy part of the store instead of a public codec.
    """

    format = "lateweave-jzip-document-zstd-v1"
    score_semantics = "jzip-reconstructed-near-lossless-full-maxsim"
    scoring_space = "encoder"
    DIRECTORY_FILE = "frame-directory.npy"
    DATA_FILE = "frames.bin"

    @classmethod
    def _validate_dimension(cls, dimension: int) -> None:
        super()._validate_dimension(dimension)
        if dimension < 2:
            raise ValueError("JzipVectorStore requires dimension >= 2")

    @classmethod
    def _jzip_metadata(
        cls,
        dimension: int,
        document_count: int,
        token_count: int,
        compression_level: int,
    ) -> dict[str, object]:
        metadata = cls._metadata(dimension, document_count, token_count)
        metadata["compression_level"] = compression_level
        metadata["frame_unit"] = "document"
        return metadata

    @classmethod
    def _encode_to_file(
        cls,
        handle,
        embeddings: np.ndarray,
        document_lengths: np.ndarray,
        *,
        first_byte: int,
        chunk_tokens: int,
        compression_level: int,
        threads: int | None,
    ) -> np.ndarray:
        token_offsets = _offsets(document_lengths)
        directory = np.empty((len(document_lengths), 3), dtype=np.uint64)
        output_byte = first_byte
        for first_document, last_document in _document_batches(
            document_lengths, chunk_tokens
        ):
            first_token = int(token_offsets[first_document])
            last_token = int(token_offsets[last_document])
            chunk = np.ascontiguousarray(
                embeddings[first_token:last_token], dtype=np.float32
            )
            lengths = np.ascontiguousarray(
                document_lengths[first_document:last_document], dtype=np.int64
            )
            payload, frame_lengths = _storage_jzip_encode_documents(
                chunk,
                lengths,
                compression_level=compression_level,
                threads=threads,
            )
            handle.write(payload)
            for relative, raw_frame_length in enumerate(frame_lengths):
                document = first_document + relative
                frame_length = int(raw_frame_length)
                directory[document] = (
                    output_byte,
                    frame_length,
                    int(document_lengths[document]),
                )
                output_byte += frame_length
        return directory

    @classmethod
    def create(
        cls,
        path: str | Path,
        embeddings: np.ndarray,
        document_lengths: np.ndarray,
        *,
        chunk_tokens: int = 131_072,
        compression_level: int = 1,
        threads: int | None = None,
    ) -> "JzipVectorStore":
        path = Path(path)
        embeddings, document_lengths = _validate_embeddings(
            embeddings, document_lengths
        )
        if path.exists():
            raise FileExistsError(f"vector store already exists: {path}")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if not 1 <= compression_level <= 22:
            raise ValueError("jzip compression_level must be between 1 and 22")
        cls._validate_dimension(embeddings.shape[1])
        path.mkdir(parents=True)
        try:
            with (path / cls.DATA_FILE).open("xb") as handle:
                directory = cls._encode_to_file(
                    handle,
                    embeddings,
                    document_lengths,
                    first_byte=0,
                    chunk_tokens=chunk_tokens,
                    compression_level=compression_level,
                    threads=threads,
                )
                handle.flush()
                os.fsync(handle.fileno())
            np.save(path / cls.DIRECTORY_FILE, directory)
            np.save(path / OFFSETS_FILE, _offsets(document_lengths))
            cls._write_metadata(
                path,
                cls._jzip_metadata(
                    embeddings.shape[1],
                    len(document_lengths),
                    len(embeddings),
                    compression_level,
                ),
            )
        except BaseException:
            shutil.rmtree(path)
            raise
        return cls(path)

    def _load_backend(self, metadata: Mapping[str, object]) -> None:
        self.compression_level = int(metadata.get("compression_level", 0))
        if not 1 <= self.compression_level <= 22:
            raise ValueError("jzip compression level is invalid")
        if metadata.get("frame_unit") != "document":
            raise ValueError("jzip frame unit is incompatible")
        self.directory = np.load(self.path / self.DIRECTORY_FILE, mmap_mode="r")
        if self.directory.dtype != np.uint64 or self.directory.shape != (
            self.document_count,
            3,
        ):
            raise ValueError("jzip frame directory has the wrong shape or dtype")
        data_size = (self.path / self.DATA_FILE).stat().st_size
        expected_lengths = self.offsets[1:] - self.offsets[:-1]
        if np.any(self.directory[:, 1] < 16) or not np.array_equal(
            self.directory[:, 2], expected_lengths
        ):
            raise ValueError("jzip frame directory is inconsistent")
        previous_end = 0
        for raw_offset, raw_length, _ in self.directory:
            offset = int(raw_offset)
            end = offset + int(raw_length)
            if offset < previous_end or end > data_size:
                raise ValueError("jzip frame directory points outside frame data")
            previous_end = end
        self._frame_data = np.memmap(
            self.path / self.DATA_FILE, mode="r", dtype=np.uint8, shape=(data_size,)
        )

    def prepare_query(
        self, embeddings: np.ndarray, *, threads: int | None = None
    ) -> np.ndarray:
        del threads
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def transition(
        self, document_ids: Sequence[int], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray(
            list(self.document_lengths(document_ids).values()), dtype=np.int64
        )
        frame_lengths = np.asarray(
            [self.directory[document_id, 1] for document_id in document_ids],
            dtype=np.uint64,
        )
        payload = np.empty(int(frame_lengths.sum()), dtype=np.uint8)
        output = 0
        for document_id, raw_length in zip(document_ids, frame_lengths, strict=True):
            offset = int(self.directory[document_id, 0])
            length = int(raw_length)
            payload[output : output + length] = self._frame_data[
                offset : offset + length
            ]
            output += length
        decoded = _storage_jzip_decode_documents(
            payload, frame_lengths, lengths, self.dimension, threads=threads
        )
        return decoded, lengths

    @property
    def encoded_bytes_per_token(self) -> float:
        return float(self._frame_data.nbytes) / self.token_count

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
            embeddings, document_lengths, self.dimension
        )
        if chunk_tokens <= 0 or copy_chunk_tokens <= 0:
            raise ValueError("chunk sizes must be positive")
        old_directory = np.array(self.directory, copy=True)
        old_offsets = np.array(self.offsets, copy=True)
        old_data_size = self._frame_data.nbytes
        old_document_count = self.document_count
        old_token_count = self.token_count
        self._close()
        try:
            with (self.path / self.DATA_FILE).open("r+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() != old_data_size:
                    raise RuntimeError("jzip frame data changed during append")
                try:
                    addition = self._encode_to_file(
                        handle,
                        embeddings,
                        document_lengths,
                        first_byte=old_data_size,
                        chunk_tokens=chunk_tokens,
                        compression_level=self.compression_level,
                        threads=threads,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException:
                    handle.truncate(old_data_size)
                    handle.flush()
                    os.fsync(handle.fileno())
                    raise
            directory = np.concatenate((old_directory, addition), axis=0)
            new_offsets = np.empty(
                old_document_count + len(document_lengths) + 1, dtype=np.uint64
            )
            new_offsets[: old_document_count + 1] = old_offsets
            np.cumsum(
                document_lengths,
                dtype=np.uint64,
                out=new_offsets[old_document_count + 1 :],
            )
            new_offsets[old_document_count + 1 :] += old_token_count
            _save_array_atomic(self.path / self.DIRECTORY_FILE, directory)
            _save_array_atomic(self.path / OFFSETS_FILE, new_offsets)
            self._write_metadata(
                self.path,
                self._jzip_metadata(
                    self.dimension,
                    old_document_count + len(document_lengths),
                    old_token_count + len(embeddings),
                    self.compression_level,
                ),
            )
        finally:
            self._load()

    def delete(
        self,
        document_ids: Sequence[int],
        *,
        copy_chunk_tokens: int = 1_000_000,
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
            [self.directory[item, 2] for item in retained], dtype=np.int64
        )
        directory = np.empty((len(retained), 3), dtype=np.uint64)
        temporary = self.path / f".{self.DATA_FILE}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                output = 0
                for new_document, document_id in enumerate(retained):
                    offset = int(self.directory[document_id, 0])
                    frame_length = int(self.directory[document_id, 1])
                    for relative in range(0, frame_length, copy_chunk_tokens):
                        count = min(copy_chunk_tokens, frame_length - relative)
                        handle.write(
                            self._frame_data[
                                offset + relative : offset + relative + count
                            ]
                        )
                    directory[new_document] = (
                        output,
                        frame_length,
                        int(lengths[new_document]),
                    )
                    output += frame_length
                handle.flush()
                os.fsync(handle.fileno())
            token_count = int(lengths.sum())
            self._close()
            temporary.replace(self.path / self.DATA_FILE)
            _save_array_atomic(self.path / self.DIRECTORY_FILE, directory)
            _save_array_atomic(self.path / OFFSETS_FILE, _offsets(lengths))
            self._write_metadata(
                self.path,
                self._jzip_metadata(
                    self.dimension,
                    len(retained),
                    token_count,
                    self.compression_level,
                ),
            )
            self._load()
        finally:
            if temporary.exists():
                temporary.unlink()

    def _close_backend(self) -> None:
        del self._frame_data, self.directory


def open_vector_store(path: str | Path) -> VectorStore:
    path = Path(path)
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    implementations = {
        Int8VectorStore.format: Int8VectorStore,
        TurboQuantVectorStore.format: TurboQuantVectorStore,
        JzipVectorStore.format: JzipVectorStore,
    }
    try:
        implementation = implementations[str(metadata["format"])]
    except KeyError as error:
        raise ValueError(
            f"unsupported vector-store format {metadata.get('format')!r}"
        ) from error
    return implementation(path)
