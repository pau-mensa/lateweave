//! Multi-vector document stores owned by lateweave.
//!
//! A store holds every token vector of every document for one
//! [`Representation`] and hands back packed float32 rows for a candidate set.
//! It is a [`MultiVectorSource`] for gatherers, such as BM25, that keep no
//! document vectors.
//!
//! Layout: one memory-mapped `.npy` per record array with a fixed width per
//! token, `document-offsets.npy`, and `storage.json` carrying the format,
//! representation, and counts. A mutation writes a replacement set of files
//! and publishes each by rename.

mod int8;
mod npy;

use std::collections::HashSet;
use std::fs;
use std::io;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError, RwLock, RwLockReadGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::manifest::Representation;
use crate::ranking::all_finite;
use crate::source::{MultiVectorSource, PackedDocuments};
use npy::{Dtype, Element, NpyArray, NpyWriter};

pub const METADATA_FILE: &str = "storage.json";
pub const OFFSETS_FILE: &str = "document-offsets.npy";

/// Bounds the temporary INT8 buffers while encoding.
const ENCODE_CHUNK_TOKENS: usize = 131_072;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreFormat {
    /// Exact float32 token vectors, `4D` bytes per token.
    Float32,
    /// Symmetric INT8 per token with one float32 row scale, `D + 4` bytes per
    /// token; lossy.
    Int8,
}

struct ArraySpec {
    name: &'static str,
    dtype: Dtype,
    /// Values per token; `None` stores one scalar per token as a 1-D array.
    columns: Option<usize>,
}

impl ArraySpec {
    fn values_per_token(&self) -> usize {
        self.columns.unwrap_or(1)
    }

    fn bytes_per_token(&self) -> usize {
        self.values_per_token() * self.dtype.size()
    }

    fn shape(&self, tokens: usize) -> Vec<usize> {
        std::iter::once(tokens).chain(self.columns).collect()
    }

    fn path(&self, directory: &Path, suffix: &str) -> PathBuf {
        directory.join(format!("{}{suffix}.npy", self.name))
    }
}

impl StoreFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::Float32 => "lateweave-float32-v1",
            Self::Int8 => "lateweave-int8-rowwise-v1",
        }
    }

    pub fn score_semantics(self) -> &'static str {
        match self {
            Self::Float32 => "float32-exact-full-maxsim",
            Self::Int8 => "int8-reconstructed-approximate-full-maxsim",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Float32, Self::Int8]
            .into_iter()
            .find(|format| format.name() == name)
    }

    fn arrays(self, dimension: usize) -> Vec<ArraySpec> {
        match self {
            Self::Float32 => vec![ArraySpec {
                name: "vectors",
                dtype: Dtype::F32,
                columns: Some(dimension),
            }],
            Self::Int8 => vec![
                ArraySpec {
                    name: "codes",
                    dtype: Dtype::I8,
                    columns: Some(dimension),
                },
                ArraySpec {
                    name: "scales",
                    dtype: Dtype::F32,
                    columns: None,
                },
            ],
        }
    }

    /// Encodes `embeddings` into this format's arrays, in `arrays()` order.
    fn encode(
        self,
        embeddings: &[f32],
        dimension: usize,
        threads: Option<usize>,
        writers: &mut [NpyWriter],
    ) -> Result<()> {
        for chunk in embeddings.chunks(ENCODE_CHUNK_TOKENS * dimension) {
            match self {
                Self::Float32 => writers[0].write(chunk)?,
                Self::Int8 => {
                    let (codes, scales) =
                        int8::int8_encode(chunk, chunk.len() / dimension, dimension, threads)?;
                    writers[0].write(&codes)?;
                    writers[1].write(&scales)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    document_count: u64,
    format: String,
    representation: Representation,
    token_count: u64,
}

struct State {
    document_count: u64,
    token_count: u64,
    offsets: NpyArray,
    arrays: Vec<NpyArray>,
}

impl State {
    fn offsets(&self) -> &[u64] {
        self.offsets
            .values()
            .expect("offsets are validated as u64 when the store opens")
    }

    fn document_length(&self, document_id: u64) -> usize {
        let offsets = self.offsets();
        (offsets[document_id as usize + 1] - offsets[document_id as usize]) as usize
    }

    fn validate_document_ids(&self, document_ids: &[u64]) -> Result<()> {
        let mut seen = HashSet::with_capacity(document_ids.len());
        for &document_id in document_ids {
            if document_id >= self.document_count {
                return Err(Error::invalid(format!(
                    "document ID {document_id} is outside the vector store of {} documents",
                    self.document_count
                )));
            }
            if !seen.insert(document_id) {
                return Err(Error::invalid(format!(
                    "document ID {document_id} is requested more than once"
                )));
            }
        }
        Ok(())
    }

    /// Bytes of documents `first..last` in array `index`.
    fn record_bytes(&self, index: usize, spec: &ArraySpec, first: u64, last: u64) -> &[u8] {
        let offsets = self.offsets();
        let width = spec.bytes_per_token();
        &self.arrays[index].bytes()
            [offsets[first as usize] as usize * width..offsets[last as usize] as usize * width]
    }
}

/// A memory-mapped multi-vector store.
///
/// Reads and mutations may run concurrently from several threads. Mutations
/// run one at a time and stage their files while reads continue; reads wait
/// only while the staged files are renamed into place. A mutation moves
/// [`MultiVectorSource::generation`], so a reranker built before it refuses
/// to score afterwards instead of reading IDs that now name other documents.
pub struct VectorStore {
    path: PathBuf,
    format: StoreFormat,
    representation: Representation,
    /// Serializes mutations, so staging needs only a shared lock on `state`.
    mutation: Mutex<()>,
    generation: AtomicU64,
    /// `None` once a failed publish could not reload the previous files.
    state: RwLock<Option<State>>,
}

struct ReadState<'a>(RwLockReadGuard<'a, Option<State>>);

impl Deref for ReadState<'_> {
    type Target = State;

    fn deref(&self) -> &State {
        self.0
            .as_ref()
            .expect("a read guard is only handed out over a loaded state")
    }
}

impl VectorStore {
    /// Writes a new store at `path`, which must not exist.
    ///
    /// `embeddings` is a row-major float32 `[sum(lengths), dimension]` matrix
    /// packed by document, and must agree with `representation`.
    pub fn create(
        path: impl AsRef<Path>,
        format: StoreFormat,
        embeddings: &[f32],
        dimension: usize,
        lengths: &[usize],
        representation: Representation,
        threads: Option<usize>,
    ) -> Result<Self> {
        let path = path.as_ref();
        validate_embeddings(embeddings, dimension, lengths, &representation)?;
        if path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("vector store already exists: {}", path.display()),
            )
            .into());
        }
        fs::create_dir_all(path)?;
        let written = (|| {
            let tokens = embeddings.len() / dimension;
            let specs = format.arrays(dimension);
            let mut writers = specs
                .iter()
                .map(|spec| NpyWriter::create(spec.path(path, ""), spec.dtype, &spec.shape(tokens)))
                .collect::<Result<Vec<_>>>()?;
            format.encode(embeddings, dimension, threads, &mut writers)?;
            writers.into_iter().try_for_each(NpyWriter::finish)?;
            write_offsets(&path.join(OFFSETS_FILE), 0, lengths.iter().copied(), &[])?;
            write_metadata(path, "", format, &representation, lengths.len(), tokens)?;
            Self::open(path)
        })();
        if written.is_err() {
            let _ = fs::remove_dir_all(path);
        }
        written
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata: Metadata = serde_json::from_slice(&fs::read(path.join(METADATA_FILE))?)
            .map_err(|error| {
                Error::storage(format!(
                    "invalid {METADATA_FILE} in {}: {error}",
                    path.display()
                ))
            })?;
        let format = StoreFormat::from_name(&metadata.format).ok_or_else(|| {
            Error::storage(format!(
                "unsupported vector-store format {:?}",
                metadata.format
            ))
        })?;
        let state = load_state(&path, format, &metadata)?;
        Ok(Self {
            path,
            format,
            representation: metadata.representation,
            mutation: Mutex::new(()),
            generation: AtomicU64::new(0),
            state: RwLock::new(Some(state)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn format(&self) -> StoreFormat {
        self.format
    }

    pub fn representation(&self) -> &Representation {
        &self.representation
    }

    pub fn dimension(&self) -> usize {
        self.representation.dimension()
    }

    /// Zero once a failed mutation has left the store unreadable.
    pub fn document_count(&self) -> u64 {
        self.read().map_or(0, |state| state.document_count)
    }

    /// Zero once a failed mutation has left the store unreadable.
    pub fn token_count(&self) -> u64 {
        self.read().map_or(0, |state| state.token_count)
    }

    /// Counts published mutations since the store was opened.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Token counts of `document_ids`, which must be unique, in order.
    pub fn document_lengths(&self, document_ids: &[u64]) -> Result<Vec<usize>> {
        let state = self.read()?;
        state.validate_document_ids(document_ids)?;
        Ok(document_ids
            .iter()
            .map(|&document_id| state.document_length(document_id))
            .collect())
    }

    /// Decoded vectors of `document_ids`, which must be unique, in order.
    pub fn fetch(&self, document_ids: &[u64], threads: Option<usize>) -> Result<PackedDocuments> {
        let state = self.read()?;
        state.validate_document_ids(document_ids)?;
        let dimension = self.dimension();
        let lengths = document_ids
            .iter()
            .map(|&document_id| state.document_length(document_id))
            .collect::<Vec<_>>();
        let tokens = lengths.iter().sum::<usize>();
        let offsets = state.offsets();
        let vectors = match self.format {
            StoreFormat::Float32 => {
                gather(&state.arrays[0], offsets, document_ids, dimension, tokens)?
            }
            StoreFormat::Int8 => {
                let codes =
                    gather::<i8>(&state.arrays[0], offsets, document_ids, dimension, tokens)?;
                let scales = gather::<f32>(&state.arrays[1], offsets, document_ids, 1, tokens)?;
                int8::int8_decode(
                    &codes,
                    &scales,
                    tokens,
                    dimension,
                    self.representation.normalized(),
                    threads,
                )?
            }
        };
        PackedDocuments::new(vectors, lengths, dimension)
    }

    /// Appends documents after the existing ones; their internal IDs continue
    /// from `document_count()`.
    pub fn append(
        &self,
        embeddings: &[f32],
        dimension: usize,
        lengths: &[usize],
        threads: Option<usize>,
    ) -> Result<()> {
        validate_embeddings(embeddings, dimension, lengths, &self.representation)?;
        let _mutation = self.mutation.lock().unwrap_or_else(PoisonError::into_inner);
        let suffix = unique_suffix();
        let specs = self.format.arrays(dimension);
        let staged = {
            let state = self.read()?;
            let old_tokens = state.token_count as usize;
            let tokens = old_tokens + embeddings.len() / dimension;
            (|| {
                let mut writers = specs
                    .iter()
                    .map(|spec| {
                        NpyWriter::create(
                            spec.path(&self.path, &suffix),
                            spec.dtype,
                            &spec.shape(tokens),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                for (writer, array) in writers.iter_mut().zip(&state.arrays) {
                    writer.write_bytes(array.bytes())?;
                }
                self.format
                    .encode(embeddings, dimension, threads, &mut writers)?;
                writers.into_iter().try_for_each(NpyWriter::finish)?;
                write_offsets(
                    &self.path.join(format!("{OFFSETS_FILE}{suffix}")),
                    old_tokens as u64,
                    lengths.iter().copied(),
                    state.offsets(),
                )?;
                write_metadata(
                    &self.path,
                    &suffix,
                    self.format,
                    &self.representation,
                    state.document_count as usize + lengths.len(),
                    tokens,
                )
            })()
        };
        self.publish(&specs, &suffix, staged)
    }

    /// Removes documents and compacts the survivors' internal IDs to
    /// `0..n-1`, preserving their order.
    pub fn delete(&self, document_ids: &[u64]) -> Result<()> {
        let _mutation = self.mutation.lock().unwrap_or_else(PoisonError::into_inner);
        let suffix = unique_suffix();
        let specs = self.format.arrays(self.dimension());
        let staged = {
            let state = self.read()?;
            state.validate_document_ids(document_ids)?;
            if document_ids.is_empty() {
                return Ok(());
            }
            if document_ids.len() as u64 == state.document_count {
                return Err(Error::invalid(
                    "delete cannot remove every vector-store document",
                ));
            }
            let mut deleted = document_ids.to_vec();
            deleted.sort_unstable();
            let mut retained_runs = Vec::with_capacity(deleted.len() + 1);
            let mut first = 0;
            for &document_id in deleted.iter().chain(std::iter::once(&state.document_count)) {
                if first < document_id {
                    retained_runs.push((first, document_id));
                }
                first = document_id + 1;
            }
            let retained_lengths = retained_runs
                .iter()
                .flat_map(|&(first, last)| first..last)
                .map(|document_id| state.document_length(document_id))
                .collect::<Vec<_>>();
            let tokens = retained_lengths.iter().sum::<usize>();
            (|| {
                for (index, spec) in specs.iter().enumerate() {
                    let mut writer = NpyWriter::create(
                        spec.path(&self.path, &suffix),
                        spec.dtype,
                        &spec.shape(tokens),
                    )?;
                    for &(first, last) in &retained_runs {
                        writer.write_bytes(state.record_bytes(index, spec, first, last))?;
                    }
                    writer.finish()?;
                }
                write_offsets(
                    &self.path.join(format!("{OFFSETS_FILE}{suffix}")),
                    0,
                    retained_lengths.iter().copied(),
                    &[],
                )?;
                write_metadata(
                    &self.path,
                    &suffix,
                    self.format,
                    &self.representation,
                    retained_lengths.len(),
                    tokens,
                )
            })()
        };
        self.publish(&specs, &suffix, staged)
    }

    /// Renames staged files over the live ones and reloads, or removes them
    /// if staging failed.
    ///
    /// The live files are unmapped first, since a mapped file cannot be
    /// replaced on every platform, and moved aside rather than overwritten:
    /// if any rename or the reload fails, they are moved back and reloaded, so
    /// the store is left as it was.
    fn publish(&self, specs: &[ArraySpec], suffix: &str, staged: Result<()>) -> Result<()> {
        let files = specs
            .iter()
            .map(|spec| (spec.path(&self.path, suffix), spec.path(&self.path, "")))
            .chain([
                (
                    self.path.join(format!("{OFFSETS_FILE}{suffix}")),
                    self.path.join(OFFSETS_FILE),
                ),
                (
                    self.path.join(format!("{METADATA_FILE}{suffix}")),
                    self.path.join(METADATA_FILE),
                ),
            ])
            .collect::<Vec<_>>();
        let remove_staged = || {
            for (staged, _) in &files {
                let _ = fs::remove_file(staged);
            }
        };
        if let Err(error) = staged {
            remove_staged();
            return Err(error);
        }
        let backup_suffix = unique_suffix();
        let backup = |live: &Path| {
            let mut name = live.as_os_str().to_owned();
            name.push(&backup_suffix);
            PathBuf::from(name)
        };

        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        self.generation.fetch_add(1, Ordering::SeqCst);
        *state = None;
        let published = files
            .iter()
            .try_for_each(|(_, live)| fs::rename(live, backup(live)))
            .and_then(|()| {
                files
                    .iter()
                    .try_for_each(|(staged, live)| fs::rename(staged, live))
            })
            .map_err(Error::from)
            .and_then(|()| self.load());
        match published {
            Ok(published) => {
                *state = Some(published);
                for (_, live) in &files {
                    let _ = fs::remove_file(backup(live));
                }
                Ok(())
            }
            Err(error) => {
                for (_, live) in &files {
                    let backup = backup(live);
                    if backup.exists() {
                        let _ = fs::rename(backup, live);
                    }
                }
                remove_staged();
                *state = self.load().ok();
                Err(error)
            }
        }
    }

    fn load(&self) -> Result<State> {
        let metadata: Metadata = serde_json::from_slice(&fs::read(self.path.join(METADATA_FILE))?)
            .map_err(|error| Error::storage(error.to_string()))?;
        load_state(&self.path, self.format, &metadata)
    }

    fn read(&self) -> Result<ReadState<'_>> {
        let state = self.state.read().unwrap_or_else(PoisonError::into_inner);
        if state.is_none() {
            return Err(Error::storage(format!(
                "vector store {} could not be reloaded after a failed mutation; reopen it",
                self.path.display()
            )));
        }
        Ok(ReadState(state))
    }
}

impl MultiVectorSource for VectorStore {
    fn representation(&self) -> &Representation {
        &self.representation
    }

    fn score_semantics(&self) -> &str {
        self.format.score_semantics()
    }

    fn document_count(&self) -> u64 {
        VectorStore::document_count(self)
    }

    fn generation(&self) -> u64 {
        VectorStore::generation(self)
    }

    fn document_lengths(&self, document_ids: &[u64]) -> Result<Vec<usize>> {
        VectorStore::document_lengths(self, document_ids)
    }

    fn fetch(&self, document_ids: &[u64], threads: Option<usize>) -> Result<PackedDocuments> {
        VectorStore::fetch(self, document_ids, threads)
    }
}

/// The records of `document_ids`, in order, from an array of `width` values per
/// token.
fn gather<T: Element>(
    array: &NpyArray,
    offsets: &[u64],
    document_ids: &[u64],
    width: usize,
    tokens: usize,
) -> Result<Vec<T>> {
    let values = array.values::<T>()?;
    let mut output = Vec::with_capacity(tokens * width);
    for &document_id in document_ids {
        let first = offsets[document_id as usize] as usize;
        let last = offsets[document_id as usize + 1] as usize;
        output.extend_from_slice(&values[first * width..last * width]);
    }
    Ok(output)
}

fn load_state(path: &Path, format: StoreFormat, metadata: &Metadata) -> Result<State> {
    let offsets = NpyArray::open(&path.join(OFFSETS_FILE))?;
    let document_count = metadata.document_count;
    let token_count = metadata.token_count;
    let valid_offsets = offsets.dtype() == Dtype::U64
        && offsets.shape() == [document_count as usize + 1]
        && offsets.values::<u64>().is_ok_and(|values| {
            values[0] == 0
                && values[values.len() - 1] == token_count
                && values.windows(2).all(|pair| pair[0] < pair[1])
        });
    if !valid_offsets {
        return Err(Error::storage("vector-store document offsets are invalid"));
    }
    let arrays = format
        .arrays(metadata.representation.dimension())
        .iter()
        .map(|spec| {
            let array = NpyArray::open(&spec.path(path, ""))?;
            if array.dtype() != spec.dtype || array.shape() != spec.shape(token_count as usize) {
                return Err(Error::storage(format!(
                    "vector-store array '{}' has the wrong shape or dtype",
                    spec.name
                )));
            }
            Ok(array)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(State {
        document_count,
        token_count,
        offsets,
        arrays,
    })
}

fn validate_embeddings(
    embeddings: &[f32],
    dimension: usize,
    lengths: &[usize],
    representation: &Representation,
) -> Result<()> {
    if dimension != representation.dimension() {
        return Err(Error::invalid(format!(
            "embedding dimension {dimension} differs from the representation dimension {}",
            representation.dimension()
        )));
    }
    if embeddings.is_empty() || embeddings.len() % dimension != 0 {
        return Err(Error::invalid(
            "embeddings must be a non-empty [tokens, dimension] matrix",
        ));
    }
    if lengths.is_empty() || lengths.contains(&0) {
        return Err(Error::invalid(
            "document lengths must be a non-empty positive vector",
        ));
    }
    if lengths.iter().sum::<usize>() != embeddings.len() / dimension {
        return Err(Error::invalid(
            "document lengths do not match packed embedding rows",
        ));
    }
    if !all_finite(embeddings) {
        return Err(Error::invalid("embeddings contain a non-finite value"));
    }
    // numpy.allclose(norms, 1, rtol=1e-3, atol=1e-4), as the format has always
    // accepted.
    if representation.normalized()
        && !embeddings.par_chunks(dimension).all(|row| {
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            (norm - 1.0).abs() <= 1.1e-3
        })
    {
        return Err(Error::invalid(
            "representation declares unit vectors, but rows are not normalized",
        ));
    }
    Ok(())
}

/// Writes `existing` followed by the running sum of `lengths` from `base`.
fn write_offsets(
    path: &Path,
    base: u64,
    lengths: impl ExactSizeIterator<Item = usize>,
    existing: &[u64],
) -> Result<()> {
    let prefix = if existing.is_empty() {
        &[0u64][..]
    } else {
        existing
    };
    let mut writer = NpyWriter::create(
        path.to_path_buf(),
        Dtype::U64,
        &[prefix.len() + lengths.len()],
    )?;
    writer.write(prefix)?;
    let offsets = lengths
        .scan(base, |total, length| {
            *total += length as u64;
            Some(*total)
        })
        .collect::<Vec<_>>();
    writer.write(&offsets)?;
    writer.finish()
}

fn write_metadata(
    path: &Path,
    suffix: &str,
    format: StoreFormat,
    representation: &Representation,
    documents: usize,
    tokens: usize,
) -> Result<()> {
    let metadata = Metadata {
        document_count: documents as u64,
        format: format.name().to_string(),
        representation: representation.clone(),
        token_count: tokens as u64,
    };
    let mut encoded = serde_json::to_string_pretty(&metadata)
        .map_err(|error| Error::storage(error.to_string()))?;
    encoded.push('\n');
    fs::write(path.join(format!("{METADATA_FILE}{suffix}")), encoded)?;
    Ok(())
}

fn unique_suffix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    format!(
        ".{}-{nanos}-{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    const DIMENSION: usize = 4;

    fn representation() -> Representation {
        Representation::new("encoder", "1", DIMENSION, true).unwrap()
    }

    fn unit(axis: usize) -> [f32; DIMENSION] {
        let mut row = [0.0; DIMENSION];
        row[axis] = 1.0;
        row
    }

    fn rows(axes: &[usize]) -> Vec<f32> {
        axes.iter().flat_map(|&axis| unit(axis)).collect()
    }

    #[test]
    fn stores_reopen_and_fetch_in_requested_order() {
        for format in [StoreFormat::Float32, StoreFormat::Int8] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("vectors");
            VectorStore::create(
                &path,
                format,
                &rows(&[0, 1, 2, 3]),
                DIMENSION,
                &[2, 1, 1],
                representation(),
                Some(1),
            )
            .unwrap();
            let store = VectorStore::open(&path).unwrap();
            assert_eq!(store.format(), format);
            assert_eq!(store.document_lengths(&[2, 0]).unwrap(), vec![1, 2]);
            let packed = store.fetch(&[2, 0], Some(1)).unwrap();
            assert_eq!(packed.lengths(), &[1, 2]);
            assert_eq!(packed.vectors(), rows(&[3, 0, 1]).as_slice());
            assert!(store.fetch(&[0, 0], None).is_err());
            assert!(store.fetch(&[3], None).is_err());
        }
    }

    #[test]
    fn append_and_delete_compact_internal_ids() {
        for format in [StoreFormat::Float32, StoreFormat::Int8] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("vectors");
            let store = VectorStore::create(
                &path,
                format,
                &rows(&[0, 1, 2, 3]),
                DIMENSION,
                &[1, 2, 1],
                representation(),
                None,
            )
            .unwrap();
            store.append(&rows(&[0]), DIMENSION, &[1], None).unwrap();
            assert_eq!((store.document_count(), store.token_count()), (4, 5));

            store.delete(&[1]).unwrap();
            assert_eq!((store.document_count(), store.token_count()), (3, 3));
            assert_eq!(
                store.fetch(&[0, 1, 2], None).unwrap().vectors(),
                rows(&[0, 3, 0]).as_slice()
            );
            assert!(store.delete(&[0, 1, 2]).is_err());

            let reopened = VectorStore::open(&path).unwrap();
            assert_eq!((reopened.document_count(), reopened.token_count()), (3, 3));
            assert_eq!(temporary_files(&path), 0);
        }
    }

    fn temporary_files(path: &Path) -> usize {
        fs::read_dir(path)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".tmp")
            })
            .count()
    }

    #[test]
    fn a_failed_publish_restores_the_previous_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let store = VectorStore::create(
            &path,
            StoreFormat::Int8,
            &rows(&[0, 1, 2]),
            DIMENSION,
            &[1, 2],
            representation(),
            None,
        )
        .unwrap();
        let before = store.fetch(&[0, 1], None).unwrap();
        // Nothing was staged under this suffix, so the first staged rename
        // fails after every live file has been moved aside.
        let specs = store.format.arrays(DIMENSION);
        assert!(store.publish(&specs, &unique_suffix(), Ok(())).is_err());
        assert_eq!(store.fetch(&[0, 1], None).unwrap(), before);
        assert_eq!(store.generation(), 1);
        assert_eq!(temporary_files(&path), 0);
        assert_eq!(VectorStore::open(&path).unwrap().document_count(), 2);
    }

    #[test]
    fn a_reranker_refuses_a_store_mutated_after_it_was_built() {
        use crate::manifest::CorpusManifest;
        use crate::maxsim::{MaxSimReranker, DEFAULT_FEATURE};
        use crate::query::{Feature, Query, TokenMatrix};
        use crate::stage::{Candidate, Reranker, ResourceBudget};

        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(
            VectorStore::create(
                directory.path().join("vectors"),
                StoreFormat::Float32,
                &rows(&[0, 1, 2, 3]),
                DIMENSION,
                &[1, 1, 1, 1],
                representation(),
                None,
            )
            .unwrap(),
        );
        let reranker = MaxSimReranker::new(
            store.clone(),
            CorpusManifest::new("corpus", "1", 4, "abc").unwrap(),
            DEFAULT_FEATURE,
        )
        .unwrap();
        let query = Query::new("query").with_feature(
            DEFAULT_FEATURE,
            Feature::new(
                representation(),
                TokenMatrix::new(rows(&[1]), DIMENSION).unwrap(),
            ),
        );
        let candidates = [Candidate {
            document_id: 1,
            gather_score: 0.0,
            gather_rank: 0,
            provenance: "test".to_string(),
        }];
        let budget = ResourceBudget::default();
        assert_eq!(
            reranker.rerank(&query, &candidates, &budget).unwrap()[0].value,
            1.0
        );

        // Same document count afterwards, but ID 1 now holds another vector.
        store.delete(&[0]).unwrap();
        store.append(&rows(&[0]), DIMENSION, &[1], None).unwrap();
        let error = reranker.rerank(&query, &candidates, &budget).unwrap_err();
        assert!(matches!(error, Error::IncompatibleIndex(_)));
    }

    #[test]
    fn vectors_must_agree_with_the_representation() {
        let directory = tempfile::tempdir().unwrap();
        let wrong_dimension = Representation::new("encoder", "1", 2, true).unwrap();
        let error = VectorStore::create(
            directory.path().join("a"),
            StoreFormat::Float32,
            &rows(&[0]),
            DIMENSION,
            &[1],
            wrong_dimension,
            None,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("dimension"));
        let doubled = rows(&[0])
            .iter()
            .map(|value| value * 2.0)
            .collect::<Vec<_>>();
        let error = VectorStore::create(
            directory.path().join("b"),
            StoreFormat::Float32,
            &doubled,
            DIMENSION,
            &[1],
            representation(),
            None,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("normalized"));
        assert!(!directory.path().join("b").exists());
    }
}
