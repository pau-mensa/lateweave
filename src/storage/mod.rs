//! The lateweave vector store: memory-mapped multi-vector documents of one
//! corpus, for gatherers, such as BM25, that keep no document vectors.
//!
//! The on-disk format is the contract, documented in `STORE_FORMAT.md`, so
//! any process can write a store. [`VectorStore`] only reads it, following
//! whatever the writer last committed; [`VectorStoreWriter`] is one writer.
//!
//! A store is a list of immutable segments, each a set of `.npy` arrays plus
//! its document IDs, and per segment an immutable tombstone file naming its
//! deleted rows. `manifest.json`, replaced by rename, is the only file that
//! changes: it names the live segments and tombstones and when they were
//! committed. A document's newest row, in the latest segment that holds its
//! ID, decides whether it is present.

mod int8;
mod npy;
mod writer;

use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::representation::Representation;
use crate::source::{MultiVectorSource, PackedDocuments, VectorView};
use npy::{Dtype, Element, NpyArray};

pub use writer::VectorStoreWriter;

pub const MANIFEST_FILE: &str = "manifest.json";
/// The `format` every manifest this version reads declares.
pub const STORE_FORMAT: &str = "lateweave-vectors-1";

/// How often a read retries when the writer commits and removes a file the
/// manifest it read still named.
const OPEN_ATTEMPTS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
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
    fn bytes_per_token(&self) -> usize {
        self.columns.unwrap_or(1) * self.dtype.size()
    }

    fn shape(&self, tokens: usize) -> Vec<usize> {
        std::iter::once(tokens).chain(self.columns).collect()
    }
}

impl Encoding {
    pub fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Int8 => "int8",
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
            .find(|encoding| encoding.name() == name)
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Manifest {
    format: String,
    encoding: String,
    corpus: String,
    representation: Representation,
    commit: u64,
    /// Seconds since the Unix epoch.
    committed_at: f64,
    segments: Vec<SegmentEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SegmentEntry {
    id: u64,
    documents: u64,
    tokens: u64,
    /// The commit whose tombstone file lists this segment's deleted rows.
    tombstones: Option<u64>,
}

impl Manifest {
    fn read(path: &Path) -> Result<Self> {
        let manifest: Self =
            serde_json::from_slice(&fs::read(path.join(MANIFEST_FILE))?).map_err(|error| {
                Error::storage(format!(
                    "invalid {MANIFEST_FILE} in {}: {error}",
                    path.display()
                ))
            })?;
        if manifest.format != STORE_FORMAT {
            return Err(Error::storage(format!(
                "unsupported vector-store format {:?}",
                manifest.format
            )));
        }
        manifest.encoding()?;
        manifest.committed_at()?;
        if manifest.corpus.is_empty() {
            return Err(Error::storage("a vector store's corpus must not be empty"));
        }
        if manifest
            .segments
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err(Error::storage("vector-store segment IDs must ascend"));
        }
        if manifest.segments.iter().any(|segment| {
            segment
                .tombstones
                .is_some_and(|commit| commit > manifest.commit)
        }) {
            return Err(Error::storage(
                "a vector-store tombstone file is newer than its manifest",
            ));
        }
        Ok(manifest)
    }

    fn encoding(&self) -> Result<Encoding> {
        Encoding::from_name(&self.encoding).ok_or_else(|| {
            Error::storage(format!(
                "unsupported vector-store encoding {:?}",
                self.encoding
            ))
        })
    }

    fn committed_at(&self) -> Result<SystemTime> {
        Duration::try_from_secs_f64(self.committed_at)
            .map(|since| SystemTime::UNIX_EPOCH + since)
            .map_err(|_| Error::storage("vector-store committed_at must be a non-negative time"))
    }

    /// Every file this manifest names.
    fn files(&self, directory: &Path) -> Result<BTreeSet<PathBuf>> {
        let dimension = self.representation.dimension();
        let arrays = self.encoding()?.arrays(dimension);
        let mut files = BTreeSet::from([directory.join(MANIFEST_FILE)]);
        for segment in &self.segments {
            files.extend(segment_files(directory, segment.id, &arrays));
            if let Some(commit) = segment.tombstones {
                files.insert(tombstones_file(directory, segment.id, commit));
            }
        }
        Ok(files)
    }
}

fn segment_file(directory: &Path, segment: u64, name: &str) -> PathBuf {
    directory.join(format!("segment-{segment}.{name}"))
}

fn tombstones_file(directory: &Path, segment: u64, commit: u64) -> PathBuf {
    segment_file(directory, segment, &format!("tombstones-{commit}.npy"))
}

fn segment_files(directory: &Path, segment: u64, arrays: &[ArraySpec]) -> Vec<PathBuf> {
    arrays
        .iter()
        .map(|spec| segment_file(directory, segment, &format!("{}.npy", spec.name)))
        .chain([
            segment_file(directory, segment, "offsets.npy"),
            segment_file(directory, segment, "ids.json"),
        ])
        .collect()
}

/// One immutable segment: its document IDs, in row order, and memory maps.
struct SegmentData {
    ids: Vec<Arc<str>>,
    rows: HashMap<Arc<str>, usize>,
    offsets: NpyArray,
    arrays: Vec<NpyArray>,
}

impl SegmentData {
    fn load(
        directory: &Path,
        entry: SegmentEntry,
        encoding: Encoding,
        dimension: usize,
    ) -> Result<Self> {
        let path = segment_file(directory, entry.id, "ids.json");
        let ids: Vec<String> = serde_json::from_reader(BufReader::new(File::open(&path)?))
            .map_err(|error| Error::storage(format!("{}: {error}", path.display())))?;
        let ids = ids.into_iter().map(Arc::<str>::from).collect::<Vec<_>>();
        let mut rows = HashMap::with_capacity(ids.len());
        for (row, id) in ids.iter().enumerate() {
            if rows.insert(id.clone(), row).is_some() {
                return Err(Error::storage(format!(
                    "document ID {id:?} appears more than once in segment {}",
                    entry.id
                )));
            }
        }
        if ids.len() as u64 != entry.documents {
            return Err(Error::storage(format!(
                "segment {} holds {} documents, but the manifest says {}",
                entry.id,
                ids.len(),
                entry.documents
            )));
        }
        let offsets = NpyArray::open(&segment_file(directory, entry.id, "offsets.npy"))?;
        let valid_offsets = offsets.dtype() == Dtype::U64
            && offsets.shape() == [ids.len() + 1]
            && offsets.values::<u64>().is_ok_and(|values| {
                values[0] == 0
                    && values[values.len() - 1] == entry.tokens
                    && values.windows(2).all(|pair| pair[0] < pair[1])
            });
        if !valid_offsets {
            return Err(Error::storage(format!(
                "the offsets of segment {} are invalid",
                entry.id
            )));
        }
        let arrays = encoding
            .arrays(dimension)
            .iter()
            .map(|spec| {
                let array = NpyArray::open(&segment_file(
                    directory,
                    entry.id,
                    &format!("{}.npy", spec.name),
                ))?;
                if array.dtype() != spec.dtype || array.shape() != spec.shape(entry.tokens as usize)
                {
                    return Err(Error::storage(format!(
                        "array '{}' of segment {} has the wrong shape or dtype",
                        spec.name, entry.id
                    )));
                }
                Ok(array)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ids,
            rows,
            offsets,
            arrays,
        })
    }

    fn offsets(&self) -> &[u64] {
        self.offsets
            .values()
            .expect("offsets are validated as u64 when the segment loads")
    }

    fn tokens(&self, row: usize) -> (usize, usize) {
        let offsets = self.offsets();
        (offsets[row] as usize, offsets[row + 1] as usize)
    }

    fn values<T: Element>(&self, array: usize, row: usize, width: usize) -> &[T] {
        let (first, last) = self.tokens(row);
        &self.arrays[array]
            .values::<T>()
            .expect("arrays are validated when the segment loads")[first * width..last * width]
    }

    fn record_bytes(&self, array: usize, spec: &ArraySpec, row: usize) -> &[u8] {
        let (first, last) = self.tokens(row);
        let width = spec.bytes_per_token();
        &self.arrays[array].bytes()[first * width..last * width]
    }
}

/// A segment as one commit sees it: its data and deleted rows, ascending.
#[derive(Clone)]
struct LiveSegment {
    entry: SegmentEntry,
    data: Arc<SegmentData>,
    tombstones: Arc<Vec<u64>>,
}

impl LiveSegment {
    fn deleted(&self, row: usize) -> bool {
        self.tombstones.binary_search(&(row as u64)).is_ok()
    }
}

fn load_tombstones(directory: &Path, entry: &SegmentEntry) -> Result<Vec<u64>> {
    let Some(commit) = entry.tombstones else {
        return Ok(Vec::new());
    };
    let array = NpyArray::open(&tombstones_file(directory, entry.id, commit))?;
    let rows = if array.shape().len() == 1 {
        array.values::<u64>().ok()
    } else {
        None
    };
    match rows {
        Some(rows)
            if rows.windows(2).all(|pair| pair[0] < pair[1])
                && rows.last().map_or(true, |&last| last < entry.documents) =>
        {
            Ok(rows.to_vec())
        }
        _ => Err(Error::storage(format!(
            "the tombstones of segment {} must be ascending rows inside it",
            entry.id
        ))),
    }
}

/// Segments `manifest` names, reusing what `previous` already loaded.
fn load_segments(
    directory: &Path,
    manifest: &Manifest,
    previous: &[LiveSegment],
) -> Result<Vec<LiveSegment>> {
    let encoding = manifest.encoding()?;
    let dimension = manifest.representation.dimension();
    let loaded = previous
        .iter()
        .map(|segment| (segment.entry.id, segment))
        .collect::<HashMap<_, _>>();
    manifest
        .segments
        .iter()
        .map(|&entry| {
            let known = loaded.get(&entry.id).filter(|segment| {
                (segment.entry.documents, segment.entry.tokens) == (entry.documents, entry.tokens)
            });
            let data = match known {
                Some(segment) => segment.data.clone(),
                None => Arc::new(SegmentData::load(directory, entry, encoding, dimension)?),
            };
            let tombstones = match known {
                Some(segment) if segment.entry.tombstones == entry.tombstones => {
                    segment.tombstones.clone()
                }
                _ => Arc::new(load_tombstones(directory, &entry)?),
            };
            Ok(LiveSegment {
                entry,
                data,
                tombstones,
            })
        })
        .collect()
}

/// The newest row of `id` among `segments`, when it is not deleted.
fn locate(segments: &[LiveSegment], id: &str) -> Option<(usize, usize)> {
    segments
        .iter()
        .enumerate()
        .rev()
        .find_map(|(position, segment)| segment.data.rows.get(id).map(|&row| (position, row)))
        .filter(|&(position, row)| !segments[position].deleted(row))
}

/// One commit of a [`VectorStore`]. It stays readable after later commits.
pub struct StoreView {
    manifest: Manifest,
    encoding: Encoding,
    as_of: SystemTime,
    segments: Vec<LiveSegment>,
}

impl StoreView {
    fn load(directory: &Path, manifest: Manifest, previous: Option<&StoreView>) -> Result<Self> {
        let segments = load_segments(
            directory,
            &manifest,
            previous.map_or(&[], |view| view.segments.as_slice()),
        )?;
        Ok(Self {
            encoding: manifest.encoding()?,
            as_of: manifest.committed_at()?,
            manifest,
            segments,
        })
    }

    /// When the writer committed this view.
    pub fn as_of(&self) -> SystemTime {
        self.as_of
    }

    pub fn commit(&self) -> u64 {
        self.manifest.commit
    }

    pub fn corpus(&self) -> &str {
        &self.manifest.corpus
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    pub fn representation(&self) -> &Representation {
        &self.manifest.representation
    }

    pub fn contains(&self, document_id: &str) -> bool {
        locate(&self.segments, document_id).is_some()
    }

    /// The IDs of every document present, sorted.
    pub fn document_ids(&self) -> Vec<&str> {
        let mut ids = self
            .segments
            .iter()
            .flat_map(|segment| segment.data.ids.iter())
            .map(AsRef::as_ref)
            .filter(|id| self.contains(id))
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Token counts of `document_ids`, in order; `None` for one not present.
    pub fn document_lengths(&self, document_ids: &[&str]) -> Vec<Option<usize>> {
        document_ids
            .iter()
            .map(|id| {
                locate(&self.segments, id).map(|(segment, row)| {
                    let (first, last) = self.segments[segment].data.tokens(row);
                    last - first
                })
            })
            .collect()
    }

    /// Decoded vectors of `document_ids`, which must be present, in order.
    pub fn fetch(&self, document_ids: &[&str], threads: Option<usize>) -> Result<PackedDocuments> {
        let dimension = self.manifest.representation.dimension();
        let located = document_ids
            .iter()
            .map(|id| {
                let (segment, row) = locate(&self.segments, id).ok_or_else(|| {
                    Error::invalid(format!(
                        "document {id:?} is not in the vector store of corpus {:?}",
                        self.manifest.corpus
                    ))
                })?;
                Ok((&self.segments[segment].data, row))
            })
            .collect::<Result<Vec<_>>>()?;
        let lengths = located
            .iter()
            .map(|(segment, row)| {
                let (first, last) = segment.tokens(*row);
                last - first
            })
            .collect::<Vec<_>>();
        let tokens = lengths.iter().sum::<usize>();
        let vectors = match self.encoding {
            Encoding::Float32 => gather::<f32>(&located, 0, dimension, tokens),
            Encoding::Int8 => int8::int8_decode(
                &gather::<i8>(&located, 0, dimension, tokens),
                &gather::<f32>(&located, 1, 1, tokens),
                tokens,
                dimension,
                self.manifest.representation.normalized(),
                threads,
            )?,
        };
        PackedDocuments::new(vectors, lengths, dimension)
    }
}

impl std::fmt::Debug for StoreView {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StoreView")
            .field("corpus", &self.manifest.corpus)
            .field("commit", &self.manifest.commit)
            .field("segments", &self.segments.len())
            .finish()
    }
}

impl VectorView for StoreView {
    fn as_of(&self) -> SystemTime {
        self.as_of
    }

    fn document_lengths(&self, document_ids: &[&str]) -> Result<Vec<Option<usize>>> {
        Ok(StoreView::document_lengths(self, document_ids))
    }

    fn fetch(&self, document_ids: &[&str], threads: Option<usize>) -> Result<PackedDocuments> {
        StoreView::fetch(self, document_ids, threads)
    }
}

/// The records of `located`, in order, from array `array` of `width` values
/// per token.
fn gather<T: Element>(
    located: &[(&Arc<SegmentData>, usize)],
    array: usize,
    width: usize,
    tokens: usize,
) -> Vec<T> {
    let mut output = Vec::with_capacity(tokens * width);
    for (segment, row) in located {
        output.extend_from_slice(segment.values::<T>(array, *row, width));
    }
    output
}

/// Reads a vector store, following what its writer commits.
///
/// [`view`](VectorStore::view) rereads `manifest.json` and loads only the
/// segments and tombstones it has not seen, so a commit by any process is
/// served on the next view. A view already taken keeps reading its commit.
/// As a [`MultiVectorSource`] the store hands each rerank its latest view.
pub struct VectorStore {
    path: PathBuf,
    corpus: String,
    encoding: Encoding,
    representation: Representation,
    current: RwLock<Arc<StoreView>>,
}

impl VectorStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let view = load_committed(&path, None)?;
        Ok(Self {
            corpus: view.manifest.corpus.clone(),
            encoding: view.encoding,
            representation: view.manifest.representation.clone(),
            current: RwLock::new(Arc::new(view)),
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn corpus(&self) -> &str {
        &self.corpus
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    pub fn representation(&self) -> &Representation {
        &self.representation
    }

    pub fn dimension(&self) -> usize {
        self.representation.dimension()
    }

    /// The last commit. Costs one read of `manifest.json`, and loading only
    /// when a writer has committed since.
    pub fn view(&self) -> Result<Arc<StoreView>> {
        let cached = self
            .current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if Manifest::read(&self.path)? == cached.manifest {
            return Ok(cached);
        }
        let view = Arc::new(load_committed(&self.path, Some(&cached))?);
        if view.manifest.corpus != self.corpus
            || view.encoding != self.encoding
            || view.manifest.representation != self.representation
        {
            return Err(Error::storage(format!(
                "vector store {} was replaced by one of another corpus, encoding, or representation",
                self.path.display()
            )));
        }
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = view.clone();
        Ok(view)
    }
}

impl MultiVectorSource for VectorStore {
    fn corpus(&self) -> &str {
        &self.corpus
    }

    fn representation(&self) -> &Representation {
        &self.representation
    }

    fn score_semantics(&self) -> &str {
        self.encoding.score_semantics()
    }

    fn view(&self) -> Result<Arc<dyn VectorView>> {
        Ok(VectorStore::view(self)?)
    }
}

/// The committed view, rereading `manifest.json` when the writer commits
/// and removes a file of the manifest just read.
fn load_committed(path: &Path, previous: Option<&StoreView>) -> Result<StoreView> {
    let mut attempts = 0;
    loop {
        let manifest = Manifest::read(path)?;
        match StoreView::load(path, manifest.clone(), previous) {
            Ok(view) => return Ok(view),
            Err(Error::Io(error))
                if error.kind() == io::ErrorKind::NotFound
                    && attempts + 1 < OPEN_ATTEMPTS
                    && Manifest::read(path)? != manifest =>
            {
                attempts += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests;
