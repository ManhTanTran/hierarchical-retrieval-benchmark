from __future__ import annotations

import numpy as np
from test_scalable_store import make_store

from dapr_hhr import Query
from dapr_hhr.scalable import MemmapDenseIndex, run_scalable_phase1_benchmark


class FakeEncoder:
    def get_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([float("apple" in lowered), float("banana" in lowered)])
        matrix = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)


class LegacyEncoder(FakeEncoder):
    get_embedding_dimension = None

    def get_sentence_embedding_dimension(self):
        return 2


def test_memmap_dense_index_builds_in_batches_and_searches_candidates(tmp_path):
    store = make_store(tmp_path)
    index = MemmapDenseIndex(
        store,
        "passage",
        tmp_path / "dense",
        encoder=FakeEncoder(),
        batch_size=2,
        search_chunk_size=2,
    )

    index.build()
    assert index.embedding_path.is_file()
    assert index.metadata_path.is_file()
    assert [row.item_id for row in index.search("apple", 2)] == ["p1", "p2"]
    assert [row.item_id for row in index.search("apple", 2, candidate_ids=["p2", "p3"])] == [
        "p2",
        "p3",
    ]


def test_memmap_dense_index_reuses_completed_index(tmp_path):
    store = make_store(tmp_path)
    encoder = FakeEncoder()
    first = MemmapDenseIndex(store, "document", tmp_path / "dense", encoder=encoder)
    first.build()
    original_mtime = first.embedding_path.stat().st_mtime_ns

    second = MemmapDenseIndex(store, "document", tmp_path / "dense", encoder=encoder)
    second.build()

    assert second.embedding_path.stat().st_mtime_ns == original_mtime
    assert [row.item_id for row in second.search("banana", 1)] == ["d2"]


def test_memmap_dense_index_supports_legacy_dimension_api(tmp_path):
    store = make_store(tmp_path)
    index = MemmapDenseIndex(
        store,
        "document",
        tmp_path / "dense",
        encoder=LegacyEncoder(),
    ).build()

    assert index.embedding_path.is_file()


def test_memmap_dense_index_searches_many_without_loading_matrix(tmp_path):
    store = make_store(tmp_path)
    index = MemmapDenseIndex(
        store,
        "document",
        tmp_path / "dense",
        encoder=FakeEncoder(),
        search_chunk_size=1,
    ).build()

    results = index.search_many([Query("apple", "apple"), Query("banana", "banana")], 1)

    assert results["apple"][0].item_id == "d1"
    assert results["banana"][0].item_id == "d2"


def test_scalable_workflow_runs_full_matrix_and_reuses_indexes(tmp_path):
    store = make_store(tmp_path)

    report = run_scalable_phase1_benchmark(
        store,
        run_mode="full",
        index_dir=tmp_path / "indexes",
        output_dir=tmp_path / "outputs",
        encoder=FakeEncoder(),
        document_top_k=2,
        passage_top_k=3,
        verbose=False,
    )

    assert len(report.leaderboard) == 9
    assert set(report.leaderboard["experiment"]) == {
        f"{document}__{passage}"
        for document in ("sparse", "dense", "combined")
        for passage in ("sparse", "dense", "combined")
    }
    assert set((tmp_path / "indexes").glob("*_embeddings.npy")) == {
        tmp_path / "indexes" / "document_embeddings.npy",
        tmp_path / "indexes" / "passage_embeddings.npy",
    }
    assert report.archive_path.is_file()
