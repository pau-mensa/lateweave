use std::collections::BTreeSet;
use std::time::Duration;

use super::npy::NpyWriter;
use super::*;
use crate::maxsim::{MaxSimReranker, DEFAULT_FEATURE};
use crate::query::{Feature, Query, TokenMatrix};
use crate::stage::{Candidate, DocumentKey, Reranker, ResourceBudget};

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

fn writer(path: &Path, encoding: Encoding) -> VectorStoreWriter {
    VectorStoreWriter::create(path, encoding, "docs", representation()).unwrap()
}

/// Appends one single-token document per `(id, axis)`.
fn append(writer: &mut VectorStoreWriter, documents: &[(&str, usize)]) {
    let axes = documents.iter().map(|&(_, axis)| axis).collect::<Vec<_>>();
    writer
        .append(
            documents.iter().map(|&(id, _)| id),
            &rows(&axes),
            DIMENSION,
            &vec![1; documents.len()],
            None,
        )
        .unwrap();
}

fn files(path: &Path) -> BTreeSet<String> {
    fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

fn vectors(view: &StoreView, ids: &[&str]) -> Vec<f32> {
    view.fetch(ids, Some(1)).unwrap().vectors().to_vec()
}

#[test]
fn a_store_serves_what_its_writer_commits_in_requested_order() {
    for encoding in [Encoding::Float32, Encoding::Int8] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let mut writer = writer(&path, encoding);
        writer
            .append(
                ["a", "b", "c"],
                &rows(&[0, 1, 2, 3]),
                DIMENSION,
                &[2, 1, 1],
                None,
            )
            .unwrap();
        let store = VectorStore::open(&path).unwrap();
        assert!(store.view().unwrap().document_ids().is_empty());

        let committed = writer.commit().unwrap();
        let view = store.view().unwrap();
        assert_eq!(view.as_of(), committed);
        assert_eq!((view.corpus(), view.encoding()), ("docs", encoding));
        assert_eq!(view.document_ids(), ["a", "b", "c"]);
        assert_eq!(
            view.document_lengths(&["c", "zzz", "a"]),
            [Some(1), None, Some(2)]
        );
        let packed = view.fetch(&["c", "a"], Some(1)).unwrap();
        assert_eq!(packed.lengths(), [1, 2]);
        assert_eq!(packed.vectors(), rows(&[3, 0, 1]));
        assert!(view.fetch(&["zzz"], None).is_err());
    }
}

#[test]
fn an_append_replaces_and_a_delete_hides_the_newest_row() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0), ("b", 1)]);
    writer.commit().unwrap();
    append(&mut writer, &[("b", 2), ("c", 3)]);
    assert_eq!(writer.delete(["a", "missing"]), 1);
    writer.commit().unwrap();

    let view = VectorStore::open(&path).unwrap().view().unwrap();
    assert_eq!(view.document_ids(), ["b", "c"]);
    assert_eq!(vectors(&view, &["b"]), rows(&[2]));

    assert_eq!(writer.delete(["b"]), 1);
    writer.commit().unwrap();
    let view = VectorStore::open(&path).unwrap().view().unwrap();
    assert_eq!(view.document_ids(), ["c"]);
    assert!(!view.contains("b"));

    append(&mut writer, &[("a", 1)]);
    writer.commit().unwrap();
    let view = VectorStore::open(&path).unwrap().view().unwrap();
    assert_eq!(view.document_ids(), ["a", "c"]);
    assert_eq!(vectors(&view, &["a"]), rows(&[1]));
}

#[test]
fn a_view_keeps_reading_its_commit() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0), ("b", 1)]);
    writer.commit().unwrap();
    let store = VectorStore::open(&path).unwrap();
    let before = store.view().unwrap();
    assert!(Arc::ptr_eq(&before, &store.view().unwrap()));

    writer.delete(["a"]);
    append(&mut writer, &[("b", 3)]);
    writer.compact().unwrap();
    writer.commit().unwrap();

    let after = store.view().unwrap();
    assert_eq!(after.document_ids(), ["b"]);
    assert_eq!(vectors(&after, &["b"]), rows(&[3]));
    assert_eq!(before.document_ids(), ["a", "b"]);
    assert_eq!(vectors(&before, &["a", "b"]), rows(&[0, 1]));
}

#[test]
fn compaction_keeps_only_present_documents_and_their_files() {
    for encoding in [Encoding::Float32, Encoding::Int8] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let mut writer = writer(&path, encoding);
        append(&mut writer, &[("a", 0), ("b", 1), ("c", 2)]);
        writer.commit().unwrap();
        append(&mut writer, &[("c", 3)]);
        writer.delete(["a"]);
        writer.commit().unwrap();
        let before = VectorStore::open(&path).unwrap().view().unwrap();

        writer.compact().unwrap();
        writer.commit().unwrap();
        let after = VectorStore::open(&path).unwrap().view().unwrap();
        assert_eq!(after.document_ids(), before.document_ids());
        assert_eq!(vectors(&after, &["b", "c"]), vectors(&before, &["b", "c"]));
        let arrays = encoding
            .arrays(DIMENSION)
            .iter()
            .map(|spec| format!("segment-2.{}.npy", spec.name))
            .collect::<Vec<_>>();
        let expected = arrays
            .into_iter()
            .chain(["segment-2.ids.json".into(), "segment-2.offsets.npy".into()])
            .chain([MANIFEST_FILE.to_string()])
            .collect::<BTreeSet<_>>();
        assert_eq!(files(&path), expected);
    }
}

#[test]
fn compacting_away_every_document_leaves_an_empty_store() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0)]);
    writer.commit().unwrap();
    writer.delete(["a"]);
    writer.compact().unwrap();
    writer.commit().unwrap();
    assert!(VectorStore::open(&path)
        .unwrap()
        .view()
        .unwrap()
        .document_ids()
        .is_empty());
    assert_eq!(files(&path), BTreeSet::from([MANIFEST_FILE.to_string()]));
}

#[test]
fn segment_identity_survives_an_empty_commit_and_writer_restart() {
    for encoding in [Encoding::Float32, Encoding::Int8] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let mut writer = writer(&path, encoding);
        append(&mut writer, &[("a", 0)]);
        writer.commit().unwrap();
        let store = VectorStore::open(&path).unwrap();
        let before = store.view().unwrap();
        let watermark = Manifest::read(&path).unwrap().next_segment_id;

        writer.delete(["a"]);
        writer.compact().unwrap();
        writer.commit().unwrap();
        assert_eq!(Manifest::read(&path).unwrap().next_segment_id, watermark);
        drop(writer);

        let mut writer = VectorStoreWriter::open(&path).unwrap();
        append(&mut writer, &[("b", 1)]);
        writer.commit().unwrap();
        let after = store.view().unwrap();
        assert_eq!(after.document_ids(), ["b"]);
        assert_eq!(vectors(&after, &["b"]), rows(&[1]));
        assert_eq!(before.document_ids(), ["a"]);
        assert_eq!(vectors(&before, &["a"]), rows(&[0]));
        assert_eq!(after.manifest.segments[0].id, watermark);
    }
}

#[test]
fn an_existing_segment_file_is_preserved_when_publication_collides() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    let destination = segment_file(&path, 0, "vectors.npy");
    let mut array = NpyWriter::create(destination.clone(), Dtype::F32, &[1, DIMENSION]).unwrap();
    array.write(&rows(&[0])).unwrap();
    array.finish().unwrap();
    let mapped = NpyArray::open(&destination).unwrap();
    let original = fs::read(&destination).unwrap();
    assert!(NpyWriter::create(destination.clone(), Dtype::F32, &[1, DIMENSION]).is_err());

    assert!(writer
        .append(["b"], &rows(&[1]), DIMENSION, &[1], None)
        .is_err());
    assert_eq!(fs::read(&destination).unwrap(), original);
    assert_eq!(mapped.values::<f32>().unwrap(), rows(&[0]));
    assert_eq!(
        files(&path),
        BTreeSet::from([
            MANIFEST_FILE.to_string(),
            "segment-0.vectors.npy".to_string()
        ])
    );
    assert!(Manifest::read(&path).unwrap().segments.is_empty());
}

#[test]
fn a_failed_compaction_preserves_pending_deletions() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0), ("b", 1)]);
    writer.commit().unwrap();
    writer.delete(["a"]);
    let collision = segment_file(&path, 1, "ids.json");
    fs::write(&collision, b"existing").unwrap();
    assert!(writer.compact().is_err());
    assert_eq!(fs::read(&collision).unwrap(), b"existing");
    writer.commit().unwrap();
    let view = VectorStore::open(&path).unwrap().view().unwrap();
    assert_eq!(view.document_ids(), ["b"]);
    assert_eq!(vectors(&view, &["b"]), rows(&[1]));
}

#[test]
fn a_tombstone_collision_preserves_existing_files_and_can_be_retried() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0), ("b", 1)]);
    writer.commit().unwrap();
    writer.delete(["a"]);
    let collision = tombstones_file(&path, 0, 2);
    fs::write(&collision, b"existing").unwrap();
    let manifest = fs::read(path.join(MANIFEST_FILE)).unwrap();
    assert!(writer.commit().is_err());
    assert_eq!(fs::read(&collision).unwrap(), b"existing");
    assert_eq!(fs::read(path.join(MANIFEST_FILE)).unwrap(), manifest);
    fs::remove_file(collision).unwrap();
    writer.commit().unwrap();
    let view = VectorStore::open(&path).unwrap().view().unwrap();
    assert_eq!(view.document_ids(), ["b"]);
}

#[test]
fn a_known_segment_cannot_change_dimensions() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0)]);
    writer.commit().unwrap();
    let store = VectorStore::open(&path).unwrap();
    let mut manifest = Manifest::read(&path).unwrap();
    manifest.segments[0].tokens += 1;
    fs::write(
        path.join(MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(matches!(store.view(), Err(Error::Storage(_))));
}

#[test]
fn recovery_reclaims_unpublished_files_before_reusing_their_ids() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0)]);
    writer.commit().unwrap();
    append(&mut writer, &[("unpublished", 1)]);
    drop(writer);
    fs::create_dir(path.join(".staging-interrupted")).unwrap();
    fs::write(path.join(".staging-interrupted/partial"), b"partial").unwrap();
    fs::write(path.join("manifest.json.interrupted"), b"partial").unwrap();

    let mut writer = VectorStoreWriter::open(&path).unwrap();
    assert!(!path.join(".staging-interrupted").exists());
    assert!(!path.join("manifest.json.interrupted").exists());
    assert!(!segment_file(&path, 1, "vectors.npy").exists());
    append(&mut writer, &[("b", 2)]);
    writer.commit().unwrap();
    let view = VectorStore::open(&path).unwrap().view().unwrap();
    assert_eq!(view.document_ids(), ["a", "b"]);
    assert_eq!(vectors(&view, &["a", "b"]), rows(&[0, 2]));
    assert_eq!(view.manifest.segments[1].id, 1);
}

#[test]
fn manifest_publication_failures_preserve_the_authoritative_files() {
    for after_rename in [false, true] {
        for reopen in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("vectors");
            let mut writer = writer(&path, Encoding::Float32);
            append(&mut writer, &[("a", 0), ("b", 1)]);
            writer.commit().unwrap();
            let store = VectorStore::open(&path).unwrap();
            let before = store.view().unwrap();
            writer.delete(["a"]);
            let result = writer.commit_with(|path, manifest, published| {
                if after_rename {
                    let mut file = File::create(path.join("manifest.json.test")).unwrap();
                    serde_json::to_writer(&mut file, manifest).unwrap();
                    file.sync_all().unwrap();
                    fs::rename(path.join("manifest.json.test"), path.join(MANIFEST_FILE)).unwrap();
                    *published = true;
                }
                Err(Error::storage("injected publication failure"))
            });
            assert!(result.is_err());
            let manifest = Manifest::read(&path).unwrap();
            assert_eq!(manifest.commit, if after_rename { 2 } else { 1 });
            assert_eq!(tombstones_file(&path, 0, 2).exists(), after_rename);
            for file in manifest.files(&path).unwrap() {
                assert!(file.exists());
            }
            let visible = store.view().unwrap();
            assert_eq!(
                visible.document_ids(),
                if after_rename {
                    vec!["b"]
                } else {
                    vec!["a", "b"]
                }
            );
            if reopen {
                drop(writer);
                writer = VectorStoreWriter::open(&path).unwrap();
                writer.delete(["a"]);
            }
            writer.commit().unwrap();
            let after = store.view().unwrap();
            assert_eq!(after.document_ids(), ["b"]);
            assert_eq!(vectors(&before, &["a", "b"]), rows(&[0, 1]));
        }
    }
}

#[test]
fn allocation_metadata_is_required_and_checked() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0)]);
    writer.commit().unwrap();
    drop(writer);
    let original: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join(MANIFEST_FILE)).unwrap()).unwrap();
    for watermark in [None, Some(serde_json::json!(0))] {
        let mut manifest = original.clone();
        manifest.as_object_mut().unwrap().remove("next_segment_id");
        if let Some(value) = watermark {
            manifest["next_segment_id"] = value;
        }
        fs::write(
            path.join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(VectorStore::open(&path).is_err());
        assert!(VectorStoreWriter::open(&path).is_err());
    }
}

#[test]
fn identifiers_do_not_wrap_on_exhaustion() {
    for field in ["next_segment_id", "commit"] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        drop(writer(&path, Encoding::Float32));
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(MANIFEST_FILE)).unwrap()).unwrap();
        manifest[field] = serde_json::json!(u64::MAX);
        fs::write(
            path.join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let mut writer = VectorStoreWriter::open(&path).unwrap();
        let before = fs::read(path.join(MANIFEST_FILE)).unwrap();
        if field == "commit" {
            assert!(writer.commit().is_err());
        } else {
            assert!(writer
                .append(["a"], &rows(&[0]), DIMENSION, &[1], None)
                .is_err());
        }
        assert_eq!(fs::read(path.join(MANIFEST_FILE)).unwrap(), before);
        assert_eq!(files(&path), BTreeSet::from([MANIFEST_FILE.to_string()]));
    }
}

#[test]
fn nothing_staged_is_visible_before_a_commit_and_an_empty_commit_advances_as_of() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0)]);
    let first = writer.commit().unwrap();
    append(&mut writer, &[("b", 1)]);
    writer.delete(["a"]);
    let store = VectorStore::open(&path).unwrap();
    assert_eq!(store.view().unwrap().document_ids(), ["a"]);

    let mut reopened = VectorStoreWriter::open(&path).unwrap();
    std::thread::sleep(Duration::from_millis(5));
    let heartbeat = reopened.commit().unwrap();
    assert!(heartbeat > first);
    let view = store.view().unwrap();
    assert_eq!((view.as_of(), view.document_ids()), (heartbeat, vec!["a"]));
    assert!(!files(&path)
        .iter()
        .any(|name| name.starts_with("segment-1.")));
}

#[test]
fn a_rejected_append_stages_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    let before = files(&path);
    let error = writer
        .append(["a", "a"], &rows(&[0, 1]), DIMENSION, &[1, 1], None)
        .unwrap_err();
    assert!(error.to_string().contains("more than once"));
    assert!(writer
        .append(["a"], &rows(&[0, 1]), DIMENSION, &[1, 1], None)
        .is_err());
    let doubled = rows(&[0])
        .iter()
        .map(|value| value * 2.0)
        .collect::<Vec<_>>();
    let error = writer
        .append(["a"], &doubled, DIMENSION, &[1], None)
        .unwrap_err();
    assert!(error.to_string().contains("normalized"));
    assert_eq!(files(&path), before);
}

#[test]
fn a_store_written_by_another_process_is_read_from_its_files_alone() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path();
    let mut array = NpyWriter::create(
        path.join("segment-7.vectors.npy"),
        Dtype::F32,
        &[3, DIMENSION],
    )
    .unwrap();
    array.write(&rows(&[0, 1, 2])).unwrap();
    array.finish().unwrap();
    let mut offsets =
        NpyWriter::create(path.join("segment-7.offsets.npy"), Dtype::U64, &[3]).unwrap();
    offsets.write(&[0u64, 1, 3]).unwrap();
    offsets.finish().unwrap();
    fs::write(path.join("segment-7.ids.json"), r#"["x", "y"]"#).unwrap();
    let mut tombstones =
        NpyWriter::create(path.join("segment-7.tombstones-4.npy"), Dtype::U64, &[1]).unwrap();
    tombstones.write(&[0u64]).unwrap();
    tombstones.finish().unwrap();
    fs::write(
        path.join(MANIFEST_FILE),
        r#"{
          "format": "lateweave-vectors-2",
          "store_id": "5f0c8e2a9b1d4c7e8a3f6b2d1e9c4a70",
          "encoding": "float32",
          "corpus": "docs",
          "representation": {"encoder": "encoder", "encoder_revision": "1", "dimension": 4, "normalized": true},
          "commit": 4,
          "next_segment_id": 8,
          "committed_at": 1700000000.5,
          "segments": [{"id": 7, "documents": 2, "tokens": 3, "tombstones": 4}]
        }"#,
    )
    .unwrap();

    let view = VectorStore::open(path).unwrap().view().unwrap();
    assert_eq!(
        view.as_of(),
        SystemTime::UNIX_EPOCH + Duration::from_secs_f64(1_700_000_000.5)
    );
    assert_eq!(view.document_ids(), ["y"]);
    assert_eq!(vectors(&view, &["y"]), rows(&[1, 2]));
}

#[test]
fn a_reader_reloads_a_store_recreated_at_the_same_path() {
    for recreated in [&[("b", 1)][..], &[("b", 1), ("c", 2)][..]] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors");
        let mut first = writer(&path, Encoding::Float32);
        append(&mut first, &[("a", 0)]);
        first.commit().unwrap();
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.view().unwrap().document_ids(), ["a"]);
        drop(first);

        fs::remove_dir_all(&path).unwrap();
        let mut second = writer(&path, Encoding::Float32);
        append(&mut second, recreated);
        second.commit().unwrap();
        let view = store.view().unwrap();
        let ids = recreated.iter().map(|&(id, _)| id).collect::<Vec<_>>();
        let values = recreated.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        assert_eq!(view.document_ids(), ids);
        assert_eq!(vectors(&view, &ids), rows(&values));
    }
}

#[test]
fn a_manifest_without_a_store_id_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    drop(writer(&path, Encoding::Float32));
    let mut manifest = Manifest::read(&path).unwrap();
    manifest.store_id.clear();
    fs::write(
        path.join(MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(VectorStore::open(&path).is_err());
    assert!(VectorStoreWriter::open(&path).is_err());
}

#[test]
fn a_manifest_this_version_cannot_read_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    writer(&path, Encoding::Float32);
    let manifest = fs::read_to_string(path.join(MANIFEST_FILE)).unwrap();
    for (from, to) in [
        ("lateweave-vectors-2", "lateweave-vectors-9"),
        ("\"float32\"", "\"float16\""),
    ] {
        fs::write(path.join(MANIFEST_FILE), manifest.replace(from, to)).unwrap();
        assert!(matches!(VectorStore::open(&path), Err(Error::Storage(_))));
    }
}

#[test]
fn a_store_is_a_source_whose_views_feed_a_reranker() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    let mut writer = writer(&path, Encoding::Float32);
    append(&mut writer, &[("a", 0), ("b", 1)]);
    writer.commit().unwrap();
    let store = Arc::new(VectorStore::open(&path).unwrap());
    let reranker = MaxSimReranker::new(
        [store.clone() as Arc<dyn MultiVectorSource>],
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
    let candidates = ["a", "b"].map(|id| Candidate {
        key: DocumentKey::new("docs", id),
        gather_score: 0.0,
        gather_rank: 0,
        provenance: "test".to_string(),
    });
    let budget = ResourceBudget::default();
    assert_eq!(
        reranker
            .rerank(&query, &candidates, &budget)
            .unwrap()
            .scores,
        [Some(0.0), Some(1.0)]
    );

    writer.delete(["b"]);
    let committed = writer.commit().unwrap();
    let scored = reranker.rerank(&query, &candidates, &budget).unwrap();
    assert_eq!(scored.scores, [Some(0.0), None]);
    assert_eq!(scored.as_of, committed);
}

#[test]
fn a_store_replaced_by_another_corpus_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors");
    writer(&path, Encoding::Float32);
    let store = VectorStore::open(&path).unwrap();
    fs::remove_dir_all(&path).unwrap();
    VectorStoreWriter::create(&path, Encoding::Float32, "other", representation()).unwrap();
    assert!(matches!(store.view(), Err(Error::Storage(_))));
}
