//! A writer of the vector store format. Nothing that reads a store depends
//! on it: any process that writes the same files is an equal writer.

use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use super::npy::{Dtype, NpyWriter};
use super::{
    int8, locate, segment_file, segment_files, tombstones_file, Encoding, LiveSegment, Manifest,
    SegmentData, SegmentEntry, MANIFEST_FILE, STORE_FORMAT,
};
use crate::error::{Error, Result};
use crate::ranking::all_finite;
use crate::representation::Representation;

/// Bounds the temporary INT8 buffers while encoding.
const ENCODE_CHUNK_TOKENS: usize = 131_072;

/// Stages appends, deletes, and compactions of one store, and publishes them
/// together on [`commit`](VectorStoreWriter::commit).
///
/// Nothing a reader sees changes before a commit. Appending a document that
/// is already present replaces it; deleting one that is not present does
/// nothing. A commit with nothing staged still advances the time readers
/// report as fresh, so an idle writer can keep a store within a reader's
/// allowed lag. One writer writes a store at a time.
pub struct VectorStoreWriter {
    path: PathBuf,
    encoding: Encoding,
    committed: Manifest,
    segments: Vec<LiveSegment>,
    /// Segments whose tombstones changed since the last commit.
    retombstoned: BTreeSet<u64>,
    next_segment: u64,
}

impl VectorStoreWriter {
    /// Writes an empty store at `path`, which must not exist.
    pub fn create(
        path: impl AsRef<Path>,
        encoding: Encoding,
        corpus: impl Into<String>,
        representation: Representation,
    ) -> Result<Self> {
        let path = path.as_ref();
        let corpus = corpus.into();
        if corpus.is_empty() {
            return Err(Error::invalid("a vector store's corpus must not be empty"));
        }
        if path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("vector store already exists: {}", path.display()),
            )
            .into());
        }
        fs::create_dir_all(path)?;
        let manifest = Manifest {
            format: STORE_FORMAT.to_string(),
            encoding: encoding.name().to_string(),
            corpus,
            representation,
            commit: 0,
            committed_at: now(),
            segments: Vec::new(),
        };
        if let Err(error) = write_manifest(path, &manifest) {
            let _ = fs::remove_dir_all(path);
            return Err(error);
        }
        Self::open(path)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let committed = Manifest::read(&path)?;
        let segments = super::load_segments(&path, &committed, &[])?;
        Ok(Self {
            encoding: committed.encoding()?,
            next_segment: committed
                .segments
                .last()
                .map_or(0, |segment| segment.id + 1),
            path,
            committed,
            segments,
            retombstoned: BTreeSet::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn corpus(&self) -> &str {
        &self.committed.corpus
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    pub fn representation(&self) -> &Representation {
        &self.committed.representation
    }

    /// Stages documents as a new segment. `embeddings` is a row-major float32
    /// `[sum(lengths), dimension]` matrix packed by document in the order of
    /// `document_ids`.
    pub fn append<I>(
        &mut self,
        document_ids: I,
        embeddings: &[f32],
        dimension: usize,
        lengths: &[usize],
        threads: Option<usize>,
    ) -> Result<()>
    where
        I: IntoIterator,
        I::Item: Into<Arc<str>>,
    {
        let ids = document_ids
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Arc<str>>>();
        if ids.is_empty() && lengths.is_empty() && embeddings.is_empty() {
            return Ok(());
        }
        validate_embeddings(embeddings, dimension, lengths, self.representation())?;
        if ids.len() != lengths.len() {
            return Err(Error::invalid(format!(
                "{} document IDs were given for {} documents",
                ids.len(),
                lengths.len()
            )));
        }
        let mut seen = HashSet::with_capacity(ids.len());
        if let Some(repeated) = ids.iter().find(|&id| !seen.insert(id)) {
            return Err(Error::invalid(format!(
                "document ID {repeated:?} appears more than once"
            )));
        }
        let entry = SegmentEntry {
            id: self.next_segment,
            documents: ids.len() as u64,
            tokens: (embeddings.len() / dimension) as u64,
            tombstones: None,
        };
        let data = self.stage(entry, |path, arrays| {
            encode(self.encoding, embeddings, dimension, threads, arrays)?;
            write_segment_index(path, entry.id, &ids, lengths.iter().copied())
        })?;
        self.next_segment += 1;
        self.segments.push(LiveSegment {
            entry,
            data: Arc::new(data),
            tombstones: Arc::new(Vec::new()),
        });
        Ok(())
    }

    /// Stages the deletion of every present document among `document_ids`,
    /// returning how many there were.
    pub fn delete<I>(&mut self, document_ids: I) -> usize
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut deleted = 0;
        for id in document_ids {
            let Some((position, row)) = locate(&self.segments, id.as_ref()) else {
                continue;
            };
            let segment = &mut self.segments[position];
            let tombstones = Arc::make_mut(&mut segment.tombstones);
            let at = tombstones
                .binary_search(&(row as u64))
                .expect_err("a located row is not deleted");
            tombstones.insert(at, row as u64);
            self.retombstoned.insert(segment.entry.id);
            deleted += 1;
        }
        deleted
    }

    /// Stages the store's present documents as one segment, dropping deleted
    /// and replaced rows.
    pub fn compact(&mut self) -> Result<()> {
        let reclaimable = self.segments.len() > 1
            || self
                .segments
                .iter()
                .any(|segment| !segment.tombstones.is_empty());
        if !reclaimable {
            return Ok(());
        }
        let kept = self
            .segments
            .iter()
            .enumerate()
            .flat_map(|(position, segment)| {
                segment
                    .data
                    .ids
                    .iter()
                    .enumerate()
                    .map(move |(row, id)| (position, row, id))
            })
            .filter(|&(position, row, id)| locate(&self.segments, id) == Some((position, row)))
            .map(|(position, row, id)| (&self.segments[position].data, row, id.clone()))
            .collect::<Vec<_>>();
        self.retombstoned.clear();
        if kept.is_empty() {
            self.segments.clear();
            return Ok(());
        }
        let lengths = kept
            .iter()
            .map(|(data, row, _)| {
                let (first, last) = data.tokens(*row);
                last - first
            })
            .collect::<Vec<_>>();
        let entry = SegmentEntry {
            id: self.next_segment,
            documents: kept.len() as u64,
            tokens: lengths.iter().sum::<usize>() as u64,
            tombstones: None,
        };
        let specs = self.encoding.arrays(self.representation().dimension());
        let data = self.stage(entry, |path, arrays| {
            for (index, (writer, spec)) in arrays.iter_mut().zip(&specs).enumerate() {
                for (data, row, _) in &kept {
                    writer.write_bytes(data.record_bytes(index, spec, *row))?;
                }
            }
            let ids = kept.iter().map(|(_, _, id)| id.clone()).collect::<Vec<_>>();
            write_segment_index(path, entry.id, &ids, lengths.iter().copied())
        })?;
        self.next_segment += 1;
        self.segments = vec![LiveSegment {
            entry,
            data: Arc::new(data),
            tombstones: Arc::new(Vec::new()),
        }];
        Ok(())
    }

    /// Publishes everything staged, and returns the commit time readers
    /// report for it. Files no longer named are then removed.
    pub fn commit(&mut self) -> Result<SystemTime> {
        let commit = self.committed.commit + 1;
        let mut written = Vec::new();
        let published = (|| {
            for segment in &mut self.segments {
                if !self.retombstoned.contains(&segment.entry.id) {
                    continue;
                }
                let file = tombstones_file(&self.path, segment.entry.id, commit);
                let mut writer =
                    NpyWriter::create(file.clone(), Dtype::U64, &[segment.tombstones.len()])?;
                written.push(file);
                writer.write(&segment.tombstones)?;
                writer.finish()?;
                segment.entry.tombstones = Some(commit);
            }
            let manifest = Manifest {
                commit,
                committed_at: now(),
                segments: self.segments.iter().map(|segment| segment.entry).collect(),
                ..self.committed.clone()
            };
            write_manifest(&self.path, &manifest)?;
            Ok(manifest)
        })();
        let manifest = match published {
            Ok(manifest) => manifest,
            Err(error) => {
                for file in written {
                    let _ = fs::remove_file(file);
                }
                for segment in &mut self.segments {
                    if let Some(entry) = self
                        .committed
                        .segments
                        .iter()
                        .find(|entry| entry.id == segment.entry.id)
                    {
                        segment.entry.tombstones = entry.tombstones;
                    }
                }
                return Err(error);
            }
        };
        self.retombstoned.clear();
        self.committed = manifest;
        remove_unreferenced(&self.path, &self.committed);
        self.committed.committed_at()
    }

    /// Writes one segment's arrays with `write`, which also writes its index
    /// files, and loads it; on failure no file of the segment remains.
    fn stage(
        &self,
        entry: SegmentEntry,
        write: impl FnOnce(&Path, &mut [NpyWriter]) -> Result<()>,
    ) -> Result<SegmentData> {
        let dimension = self.representation().dimension();
        let specs = self.encoding.arrays(dimension);
        let staged = (|| {
            let mut arrays = specs
                .iter()
                .map(|spec| {
                    NpyWriter::create(
                        segment_file(&self.path, entry.id, &format!("{}.npy", spec.name)),
                        spec.dtype,
                        &spec.shape(entry.tokens as usize),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            write(&self.path, &mut arrays)?;
            arrays.into_iter().try_for_each(NpyWriter::finish)?;
            SegmentData::load(&self.path, entry, self.encoding, dimension)
        })();
        if staged.is_err() {
            for file in segment_files(&self.path, entry.id, &specs) {
                let _ = fs::remove_file(file);
            }
        }
        staged
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64())
}

fn encode(
    encoding: Encoding,
    embeddings: &[f32],
    dimension: usize,
    threads: Option<usize>,
    arrays: &mut [NpyWriter],
) -> Result<()> {
    for chunk in embeddings.chunks(ENCODE_CHUNK_TOKENS * dimension) {
        match encoding {
            Encoding::Float32 => arrays[0].write(chunk)?,
            Encoding::Int8 => {
                let (codes, scales) =
                    int8::int8_encode(chunk, chunk.len() / dimension, dimension, threads)?;
                arrays[0].write(&codes)?;
                arrays[1].write(&scales)?;
            }
        }
    }
    Ok(())
}

/// Writes a segment's offsets, the running sum of `lengths`, and its IDs.
fn write_segment_index(
    path: &Path,
    segment: u64,
    ids: &[Arc<str>],
    lengths: impl ExactSizeIterator<Item = usize>,
) -> Result<()> {
    let mut offsets = NpyWriter::create(
        segment_file(path, segment, "offsets.npy"),
        Dtype::U64,
        &[lengths.len() + 1],
    )?;
    let values = std::iter::once(0)
        .chain(lengths.scan(0u64, |total, length| {
            *total += length as u64;
            Some(*total)
        }))
        .collect::<Vec<_>>();
    offsets.write(&values)?;
    offsets.finish()?;

    let mut file = BufWriter::new(File::create(segment_file(path, segment, "ids.json"))?);
    let ids = ids.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
    serde_json::to_writer(&mut file, &ids).map_err(|error| Error::storage(error.to_string()))?;
    file.into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;
    Ok(())
}

/// Replaces `manifest.json` by rename, which is what commits.
fn write_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    let mut encoded = serde_json::to_string_pretty(manifest)
        .map_err(|error| Error::storage(error.to_string()))?;
    encoded.push('\n');
    let staged = path.join(format!("{MANIFEST_FILE}.{}.tmp", manifest.commit));
    let written = (|| {
        let mut file = File::create(&staged)?;
        file.write_all(encoded.as_bytes())?;
        file.sync_all()?;
        fs::rename(&staged, path.join(MANIFEST_FILE))
    })();
    if written.is_err() {
        let _ = fs::remove_file(&staged);
    }
    Ok(written?)
}

/// Best effort: a file a reader still maps may not be removable on every
/// platform, and the next commit tries again.
fn remove_unreferenced(path: &Path, manifest: &Manifest) {
    let (Ok(referenced), Ok(entries)) = (manifest.files(path), fs::read_dir(path)) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let ours = name.starts_with("segment-") || name.starts_with(&format!("{MANIFEST_FILE}."));
        if ours && !referenced.contains(&entry.path()) {
            let _ = fs::remove_file(entry.path());
        }
    }
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
