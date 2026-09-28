//! Multi-vector document stores owned by lateweave.
//!
//! A store holds every token vector of every document of one segment for one
//! [`Representation`], and hands out immutable [`StoreSnapshot`]s that are
//! the [`MultiVectorSource`] for gatherers, such as BM25, that keep no
//! document vectors.
//!
//! Layout: each generation is its own set of files: one memory-mapped `.npy`
//! per record array with a fixed width per token, `document-offsets-G.npy`,
//! and `document-ids-G.json` with the segment's external IDs. `storage.json`
//! names the live generation and carries the format, representation, and
//! corpus manifest. A mutation writes the next generation's files and
//! publishes them by replacing `storage.json`; files are never modified after
//! they are written, so a snapshot keeps reading its generation after later
//! mutations.

mod int8;
mod npy;

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::manifest::{CorpusManifest, Representation};
use crate::ranking::all_finite;
use crate::segment::Segment;
use crate::source::{MultiVectorSource, PackedDocuments};
use npy::{Dtype, Element, NpyArray, NpyWriter};

pub const METADATA_FILE: &str = "storage.json";
const OFFSETS: &str = "document-offsets";
const DOCUMENT_IDS: &str = "document-ids";

/// Bounds the temporary INT8 buffers while encoding.
const ENCODE_CHUNK_TOKENS: usize = 131_072;

/// How often `open` rereads `storage.json` when a writer in another process
/// publishes and removes the generation it was about to map.
const OPEN_ATTEMPTS: usize = 8;

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

    fn path(&self, directory: &Path, generation: u64) -> PathBuf {
        generation_file(directory, self.name, generation, "npy")
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
    format: String,
    representation: Representation,
    corpus: CorpusManifest,
    token_count: u64,
}

impl Metadata {
    fn new(
        format: StoreFormat,
        representation: &Representation,
        segment: &Segment,
        token_count: u64,
    ) -> Self {
        Self {
            format: format.name().to_string(),
            representation: representation.clone(),
            corpus: segment.manifest().clone(),
            token_count,
        }
    }
}

fn generation_file(directory: &Path, stem: &str, generation: u64, extension: &str) -> PathBuf {
    directory.join(format!("{stem}-{generation}.{extension}"))
}

/// The files of one generation, in no particular order.
fn generation_files(
    directory: &Path,
    format: StoreFormat,
    dimension: usize,
    generation: u64,
) -> Vec<PathBuf> {
    format
        .arrays(dimension)
        .iter()
        .map(|spec| spec.path(directory, generation))
        .chain([
            generation_file(directory, OFFSETS, generation, "npy"),
            generation_file(directory, DOCUMENT_IDS, generation, "json"),
        ])
        .collect()
}

/// One immutable generation of a [`VectorStore`]: its segment and memory
/// maps. It stays readable after the store publishes later generations.
pub struct StoreSnapshot {
    segment: Segment,
    format: StoreFormat,
    representation: Representation,
    token_count: u64,
    offsets: NpyArray,
    arrays: Vec<NpyArray>,
}

impl StoreSnapshot {
    pub fn segment(&self) -> &Segment {
        &self.segment
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

    pub fn document_count(&self) -> u64 {
        self.segment.document_count()
    }

    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Token counts of `document_ids`, which must be unique, in order.
    pub fn document_lengths(&self, document_ids: &[u64]) -> Result<Vec<usize>> {
        self.validate_document_ids(document_ids)?;
        Ok(document_ids
            .iter()
            .map(|&document_id| self.document_length(document_id))
            .collect())
    }

    /// Decoded vectors of `document_ids`, which must be unique, in order.
    pub fn fetch(&self, document_ids: &[u64], threads: Option<usize>) -> Result<PackedDocuments> {
        self.validate_document_ids(document_ids)?;
        let dimension = self.dimension();
        let lengths = document_ids
            .iter()
            .map(|&document_id| self.document_length(document_id))
            .collect::<Vec<_>>();
        let tokens = lengths.iter().sum::<usize>();
        let offsets = self.offsets();
        let vectors = match self.format {
            StoreFormat::Float32 => {
                gather(&self.arrays[0], offsets, document_ids, dimension, tokens)?
            }
            StoreFormat::Int8 => {
                let codes =
                    gather::<i8>(&self.arrays[0], offsets, document_ids, dimension, tokens)?;
                let scales = gather::<f32>(&self.arrays[1], offsets, document_ids, 1, tokens)?;
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

    fn offsets(&self) -> &[u64] {
        self.offsets
            .values()
            .expect("offsets are validated as u64 when the snapshot loads")
    }

    fn document_length(&self, document_id: u64) -> usize {
        let offsets = self.offsets();
        (offsets[document_id as usize + 1] - offsets[document_id as usize]) as usize
    }

    fn validate_document_ids(&self, document_ids: &[u64]) -> Result<()> {
        let mut seen = HashSet::with_capacity(document_ids.len());
        for &document_id in document_ids {
            if !self.segment.contains(document_id) {
                return Err(Error::invalid(format!(
                    "document ID {document_id} is outside the vector store of {} documents",
                    self.document_count()
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

impl std::fmt::Debug for StoreSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StoreSnapshot")
            .field("segment", &self.segment)
            .field("format", &self.format)
            .field("token_count", &self.token_count)
            .finish()
    }
}

impl MultiVectorSource for StoreSnapshot {
    fn segment(&self) -> &Segment {
        &self.segment
    }

    fn representation(&self) -> &Representation {
        &self.representation
    }

    fn score_semantics(&self) -> &str {
        self.format.score_semantics()
    }

    fn document_lengths(&self, document_ids: &[u64]) -> Result<Vec<usize>> {
        StoreSnapshot::document_lengths(self, document_ids)
    }

    fn fetch(&self, document_ids: &[u64], threads: Option<usize>) -> Result<PackedDocuments> {
        StoreSnapshot::fetch(self, document_ids, threads)
    }
}

/// A memory-mapped multi-vector store for one segment.
///
/// Reads go through [`snapshot`](VectorStore::snapshot)s. Mutations run one at
/// a time, take external IDs, and return the snapshot they publish, whose
/// segment is exactly [`Segment::appended`] or [`Segment::deleted`] of the one
/// before. Snapshots taken earlier are unaffected. One process writes a store;
/// any number read it.
pub struct VectorStore {
    path: PathBuf,
    format: StoreFormat,
    representation: Representation,
    /// Serializes mutations.
    mutation: Mutex<()>,
    current: RwLock<Arc<StoreSnapshot>>,
}

impl VectorStore {
    /// Writes a new store at `path`, which must not exist, holding `segment`.
    ///
    /// `embeddings` is a row-major float32 `[sum(lengths), dimension]` matrix
    /// packed by document in the segment's internal-ID order, and must agree
    /// with `representation`.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        path: impl AsRef<Path>,
        format: StoreFormat,
        segment: &Segment,
        embeddings: &[f32],
        dimension: usize,
        lengths: &[usize],
        representation: Representation,
        threads: Option<usize>,
    ) -> Result<Self> {
        let path = path.as_ref();
        validate_embeddings(embeddings, dimension, lengths, &representation)?;
        require_lengths_for(segment, lengths.len())?;
        if path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("vector store already exists: {}", path.display()),
            )
            .into());
        }
        fs::create_dir_all(path)?;
        let written = (|| {
            let generation = segment.generation();
            let tokens = embeddings.len() / dimension;
            let mut writers = format
                .arrays(dimension)
                .iter()
                .map(|spec| {
                    NpyWriter::create(spec.path(path, generation), spec.dtype, &spec.shape(tokens))
                })
                .collect::<Result<Vec<_>>>()?;
            format.encode(embeddings, dimension, threads, &mut writers)?;
            writers.into_iter().try_for_each(NpyWriter::finish)?;
            write_offsets(
                &generation_file(path, OFFSETS, generation, "npy"),
                0,
                lengths.iter().copied(),
                &[],
            )?;
            write_document_ids(path, segment)?;
            write_metadata(
                path,
                &Metadata::new(format, &representation, segment, tokens as u64),
            )?;
            Self::open(path)
        })();
        if written.is_err() {
            let _ = fs::remove_dir_all(path);
        }
        written
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut attempts = 0;
        let (metadata, snapshot) = loop {
            let metadata = read_metadata(&path)?;
            match load_snapshot(&path, &metadata) {
                Ok(snapshot) => break (metadata, snapshot),
                Err(Error::Io(error))
                    if error.kind() == io::ErrorKind::NotFound
                        && attempts + 1 < OPEN_ATTEMPTS
                        && read_metadata(&path)?.corpus != metadata.corpus =>
                {
                    attempts += 1;
                }
                Err(error) => return Err(error),
            }
        };
        Ok(Self {
            path,
            format: snapshot.format,
            representation: metadata.representation,
            mutation: Mutex::new(()),
            current: RwLock::new(Arc::new(snapshot)),
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

    /// The live generation.
    pub fn snapshot(&self) -> Arc<StoreSnapshot> {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The live generation's segment.
    pub fn segment(&self) -> Segment {
        self.snapshot().segment.clone()
    }

    /// Appends documents after the existing ones; their internal IDs continue
    /// from the current document count, in the order of `document_ids`.
    pub fn append<I>(
        &self,
        document_ids: I,
        embeddings: &[f32],
        dimension: usize,
        lengths: &[usize],
        threads: Option<usize>,
    ) -> Result<Arc<StoreSnapshot>>
    where
        I: IntoIterator,
        I::Item: Into<Arc<str>>,
    {
        validate_embeddings(embeddings, dimension, lengths, &self.representation)?;
        let _mutation = self.mutation.lock().unwrap_or_else(PoisonError::into_inner);
        let current = self.snapshot();
        let segment = current.segment.appended(document_ids)?;
        require_lengths_for(&segment, current.document_count() as usize + lengths.len())?;
        let generation = segment.generation();
        let tokens = current.token_count as usize + embeddings.len() / dimension;
        self.publish(&segment, tokens as u64, || {
            let mut writers = self
                .format
                .arrays(dimension)
                .iter()
                .map(|spec| {
                    NpyWriter::create(
                        spec.path(&self.path, generation),
                        spec.dtype,
                        &spec.shape(tokens),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            for (writer, array) in writers.iter_mut().zip(&current.arrays) {
                writer.write_bytes(array.bytes())?;
            }
            self.format
                .encode(embeddings, dimension, threads, &mut writers)?;
            writers.into_iter().try_for_each(NpyWriter::finish)?;
            write_offsets(
                &generation_file(&self.path, OFFSETS, generation, "npy"),
                current.token_count,
                lengths.iter().copied(),
                current.offsets(),
            )
        })
    }

    /// Removes documents by external ID and compacts the survivors' internal
    /// IDs to `0..n-1`, preserving their order.
    pub fn delete<I>(&self, document_ids: I) -> Result<Arc<StoreSnapshot>>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let document_ids = document_ids
            .into_iter()
            .map(|document_id| document_id.as_ref().to_string())
            .collect::<Vec<_>>();
        let _mutation = self.mutation.lock().unwrap_or_else(PoisonError::into_inner);
        let current = self.snapshot();
        let deleted = current.segment.deletion(&document_ids)?;
        if deleted.is_empty() {
            return Ok(current);
        }
        if deleted.len() as u64 == current.document_count() {
            return Err(Error::invalid(
                "delete cannot remove every vector-store document",
            ));
        }
        let segment = current.segment.deleted(&document_ids)?;
        let mut deleted = deleted.into_iter().collect::<Vec<_>>();
        deleted.sort_unstable();
        let mut retained_runs = Vec::with_capacity(deleted.len() + 1);
        let mut first = 0;
        for &document_id in deleted
            .iter()
            .chain(std::iter::once(&current.document_count()))
        {
            if first < document_id {
                retained_runs.push((first, document_id));
            }
            first = document_id + 1;
        }
        let retained_lengths = retained_runs
            .iter()
            .flat_map(|&(first, last)| first..last)
            .map(|document_id| current.document_length(document_id))
            .collect::<Vec<_>>();
        let tokens = retained_lengths.iter().sum::<usize>();
        let generation = segment.generation();
        self.publish(&segment, tokens as u64, || {
            for (index, spec) in self.format.arrays(self.dimension()).iter().enumerate() {
                let mut writer = NpyWriter::create(
                    spec.path(&self.path, generation),
                    spec.dtype,
                    &spec.shape(tokens),
                )?;
                for &(first, last) in &retained_runs {
                    writer.write_bytes(current.record_bytes(index, spec, first, last))?;
                }
                writer.finish()?;
            }
            write_offsets(
                &generation_file(&self.path, OFFSETS, generation, "npy"),
                0,
                retained_lengths.iter().copied(),
                &[],
            )
        })
    }

    /// Writes the next generation's files with `stage`, loads them, and makes
    /// them live by replacing `storage.json`; until that rename nothing a
    /// reader can see has changed, and on failure the staged files are
    /// removed.
    fn publish(
        &self,
        segment: &Segment,
        token_count: u64,
        stage: impl FnOnce() -> Result<()>,
    ) -> Result<Arc<StoreSnapshot>> {
        let staged = generation_files(
            &self.path,
            self.format,
            self.dimension(),
            segment.generation(),
        );
        let published = (|| {
            stage()?;
            write_document_ids(&self.path, segment)?;
            let metadata = Metadata::new(self.format, &self.representation, segment, token_count);
            let snapshot = Arc::new(load_snapshot(&self.path, &metadata)?);
            write_metadata(&self.path, &metadata)?;
            Ok(snapshot)
        })();
        match published {
            Ok(snapshot) => {
                *self.current.write().unwrap_or_else(PoisonError::into_inner) = snapshot.clone();
                remove_generations_before(&self.path, segment.generation());
                Ok(snapshot)
            }
            Err(error) => {
                for file in staged {
                    let _ = fs::remove_file(file);
                }
                Err(error)
            }
        }
    }
}

fn require_lengths_for(segment: &Segment, documents: usize) -> Result<()> {
    if segment.document_count() != documents as u64 {
        return Err(Error::invalid(format!(
            "{documents} documents have vectors, but the segment holds {}",
            segment.document_count()
        )));
    }
    Ok(())
}

/// Best effort: a file another snapshot still maps may not be removable on
/// every platform, and the next publish tries again.
fn remove_generations_before(directory: &Path, generation: u64) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let older = name
            .rsplit_once('.')
            .and_then(|(stem, _)| stem.rsplit_once('-'))
            .and_then(|(_, tag)| tag.parse::<u64>().ok())
            .is_some_and(|tag| tag < generation);
        if older {
            let _ = fs::remove_file(entry.path());
        }
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

fn read_metadata(path: &Path) -> Result<Metadata> {
    serde_json::from_slice(&fs::read(path.join(METADATA_FILE))?).map_err(|error| {
        Error::storage(format!(
            "invalid {METADATA_FILE} in {}: {error}",
            path.display()
        ))
    })
}

fn load_snapshot(path: &Path, metadata: &Metadata) -> Result<StoreSnapshot> {
    let format = StoreFormat::from_name(&metadata.format).ok_or_else(|| {
        Error::storage(format!(
            "unsupported vector-store format {:?}",
            metadata.format
        ))
    })?;
    let generation = metadata.corpus.generation();
    let document_ids: Vec<String> = serde_json::from_reader(BufReader::new(File::open(
        generation_file(path, DOCUMENT_IDS, generation, "json"),
    )?))
    .map_err(|error| Error::storage(format!("invalid vector-store document IDs: {error}")))?;
    let segment = Segment::from_manifest(&metadata.corpus, document_ids)?;
    let offsets = NpyArray::open(&generation_file(path, OFFSETS, generation, "npy"))?;
    let document_count = segment.document_count();
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
            let array = NpyArray::open(&spec.path(path, generation))?;
            if array.dtype() != spec.dtype || array.shape() != spec.shape(token_count as usize) {
                return Err(Error::storage(format!(
                    "vector-store array '{}' has the wrong shape or dtype",
                    spec.name
                )));
            }
            Ok(array)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StoreSnapshot {
        segment,
        format,
        representation: metadata.representation.clone(),
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
    // numpy.allclose(norms, 1, rtol=1e-3, atol=1e-4).
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

fn write_document_ids(path: &Path, segment: &Segment) -> Result<()> {
    let mut file = BufWriter::new(File::create(generation_file(
        path,
        DOCUMENT_IDS,
        segment.generation(),
        "json",
    ))?);
    serde_json::to_writer(&mut file, &segment.document_ids().collect::<Vec<_>>())
        .map_err(|error| Error::storage(error.to_string()))?;
    file.into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;
    Ok(())
}

/// Replaces `storage.json` by rename, which is what publishes a generation.
fn write_metadata(path: &Path, metadata: &Metadata) -> Result<()> {
    let mut encoded = serde_json::to_string_pretty(metadata)
        .map_err(|error| Error::storage(error.to_string()))?;
    encoded.push('\n');
    let staged = path.join(format!(
        "{METADATA_FILE}.{}.tmp",
        metadata.corpus.generation()
    ));
    let written = (|| {
        let mut file = File::create(&staged)?;
        file.write_all(encoded.as_bytes())?;
        file.sync_all()?;
        fs::rename(&staged, path.join(METADATA_FILE))
    })();
    if written.is_err() {
        let _ = fs::remove_file(&staged);
    }
    Ok(written?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxsim::{MaxSimReranker, DEFAULT_FEATURE};
    use crate::query::{Feature, Query, TokenMatrix};
    use crate::stage::{Candidate, Reranker, ResourceBudget};

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

    fn segment(ids: &[&str]) -> Segment {
        Segment::new("docs", "1", 0, ids.iter().copied()).unwrap()
    }

    fn create(
        path: &Path,
        format: StoreFormat,
        ids: &[&str],
        axes: &[usize],
        lengths: &[usize],
    ) -> VectorStore {
        VectorStore::create(
            path,
            format,
            &segment(ids),
            &rows(axes),
            DIMENSION,
            lengths,
            representation(),
            Some(1),
        )
        .unwrap()
    }

    fn files(path: &Path) -> Vec<String> {
        let mut names = fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    #[test]
    fn stores_reopen_and_fetch_in_requested_order() {
        for format in [StoreFormat::Float32, StoreFormat::Int8] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("vectors");
            create(&path, format, &["a", "b", "c"], &[0, 1, 2, 3], &[2, 1, 1]);
            let store = VectorStore::open(&path).unwrap();
            assert_eq!(store.format(), format);
            assert_eq!(store.segment(), segment(&["a", "b", "c"]));
            let snapshot = store.snapshot();
            assert_eq!(snapshot.document_lengths(&[2, 0]).unwrap(), vec![1, 2]);
            let packed = snapshot.fetch(&[2, 0], Some(1)).unwrap();
            assert_eq!(packed.lengths(), &[1, 2]);
            assert_eq!(packed.vectors(), rows(&[3, 0, 1]).as_slice());
            assert!(snapshot.fetch(&[0, 0], None).is_err());
            assert!(snapshot.fetch(&[3], None).is_err());
        }
    }

    #[test]
    fn mutations_follow_the_segment_and_compact_internal_ids() {
        for format in [StoreFormat::Float32, StoreFormat::Int8] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("vectors");
            let store = create(&path, format, &["a", "b", "c"], &[0, 1, 2, 3], &[1, 2, 1]);
            let initial = store.segment();

            let appended = store
                .append(["d"], &rows(&[0]), DIMENSION, &[1], None)
                .unwrap();
            assert_eq!(*appended.segment(), initial.appended(["d"]).unwrap());
            assert_eq!((appended.document_count(), appended.token_count()), (4, 5));

            let deleted = store.delete(["b"]).unwrap();
            assert_eq!(
                *deleted.segment(),
                appended.segment().deleted(["b"]).unwrap()
            );
            assert_eq!(deleted.segment().generation(), 2);
            assert_eq!((deleted.document_count(), deleted.token_count()), (3, 3));
            assert_eq!(
                deleted.fetch(&[0, 1, 2], None).unwrap().vectors(),
                rows(&[0, 3, 0]).as_slice()
            );
            assert!(store.delete(["a", "c", "d"]).is_err());
            assert!(store.delete(["zzz"]).is_err());
            assert!(Arc::ptr_eq(
                &store.delete(Vec::<String>::new()).unwrap(),
                &store.snapshot()
            ));

            let reopened = VectorStore::open(&path).unwrap();
            assert_eq!(reopened.segment(), *deleted.segment());
            assert_eq!(
                files(&path),
                [
                    format!("{}-2.npy", format.arrays(DIMENSION)[0].name),
                    "document-ids-2.json".to_string(),
                    "document-offsets-2.npy".to_string(),
                ]
                .into_iter()
                .chain((format == StoreFormat::Int8).then(|| "scales-2.npy".to_string()))
                .chain(["storage.json".to_string()])
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn a_snapshot_keeps_reading_its_generation_after_mutations() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let store = create(&path, StoreFormat::Float32, &["a", "b"], &[0, 1], &[1, 1]);
        let before = store.snapshot();

        store.delete(["a"]).unwrap();
        store
            .append(["c"], &rows(&[2]), DIMENSION, &[1], None)
            .unwrap();

        // Same count as before, but internal ID 1 now names "c".
        let after = store.snapshot();
        assert_eq!(
            after.fetch(&[1], None).unwrap().vectors(),
            rows(&[2]).as_slice()
        );
        assert_eq!(
            before.fetch(&[1], None).unwrap().vectors(),
            rows(&[1]).as_slice()
        );
        assert_eq!(before.segment().external(1), Some("b"));
        assert_eq!(after.segment().external(1), Some("c"));
    }

    #[test]
    fn a_rejected_mutation_changes_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let store = create(&path, StoreFormat::Float32, &["a", "b"], &[0, 1], &[1, 1]);
        let before = files(&path);
        let snapshot = store.snapshot();
        assert!(store
            .append(["a"], &rows(&[2]), DIMENSION, &[1], None)
            .is_err());
        assert!(store
            .append(["c", "d"], &rows(&[2]), DIMENSION, &[1], None)
            .is_err());
        assert!(Arc::ptr_eq(&store.snapshot(), &snapshot));
        assert_eq!(files(&path), before);
    }

    #[test]
    fn a_failed_publish_leaves_the_live_generation_and_no_staged_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let store = create(&path, StoreFormat::Float32, &["a", "b"], &[0, 1], &[1, 1]);
        let before = files(&path);
        let snapshot = store.snapshot();
        let next = snapshot.segment().appended(["c"]).unwrap();
        // The stage writes one array and then fails.
        let error = store
            .publish(&next, 3, || {
                let spec = &store.format.arrays(DIMENSION)[0];
                NpyWriter::create(
                    spec.path(&path, next.generation()),
                    spec.dtype,
                    &spec.shape(1),
                )?
                .finish()
            })
            .unwrap_err();
        assert!(error.to_string().contains("less data"));
        assert!(Arc::ptr_eq(&store.snapshot(), &snapshot));
        assert_eq!(files(&path), before);
        assert_eq!(
            VectorStore::open(&path).unwrap().segment(),
            *snapshot.segment()
        );
    }

    #[test]
    fn a_reranker_over_a_snapshot_is_unaffected_by_later_mutations() {
        let directory = tempfile::tempdir().unwrap();
        let store = create(
            &directory.path().join("vectors"),
            StoreFormat::Float32,
            &["a", "b", "c", "d"],
            &[0, 1, 2, 3],
            &[1, 1, 1, 1],
        );
        let snapshot = store.snapshot();
        let reranker = MaxSimReranker::new(
            [snapshot.clone() as Arc<dyn MultiVectorSource>],
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
            segment: snapshot.segment().clone(),
            document_id: 1,
            gather_score: 0.0,
            gather_rank: 0,
            provenance: "test".to_string(),
        }];
        let budget = ResourceBudget::default();
        assert_eq!(
            reranker.rerank(&query, &candidates, &budget).unwrap(),
            [1.0]
        );

        // Same document count afterwards, but ID 1 now names another document.
        store.delete(["a"]).unwrap();
        let after = store
            .append(["e"], &rows(&[0]), DIMENSION, &[1], None)
            .unwrap();
        assert_eq!(
            reranker.rerank(&query, &candidates, &budget).unwrap(),
            [1.0]
        );

        let current = [Candidate {
            segment: after.segment().clone(),
            ..candidates[0].clone()
        }];
        let error = reranker.rerank(&query, &current, &budget).unwrap_err();
        assert!(matches!(error, Error::IncompatibleIndex(_)));
    }

    #[test]
    fn vectors_must_agree_with_the_representation_and_the_segment() {
        let directory = tempfile::tempdir().unwrap();
        let wrong_dimension = Representation::new("encoder", "1", 2, true).unwrap();
        let error = VectorStore::create(
            directory.path().join("a"),
            StoreFormat::Float32,
            &segment(&["a"]),
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
            &segment(&["a"]),
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
        let error = VectorStore::create(
            directory.path().join("c"),
            StoreFormat::Float32,
            &segment(&["a", "b"]),
            &rows(&[0]),
            DIMENSION,
            &[1],
            representation(),
            None,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("segment holds 2"));
    }
}
