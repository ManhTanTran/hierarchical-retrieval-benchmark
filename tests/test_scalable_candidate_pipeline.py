from __future__ import annotations

import json

import numpy as np
import pytest

from dapr_hhr import Document, Passage, Qrel, Query
from dapr_hhr.scalable import (
    CandidateSparseIndex,
    DenseEncoderRuntime,
    DiskDatasetStore,
    MemmapDenseIndex,
    run_scalable_phase1_benchmark,
)


class TrackingEncoder:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.corpus_texts: list[str] = []
        self.calls = 0
        self.fail_after = fail_after

    def get_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        del kwargs
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated interruption")
        self.corpus_texts.extend(text for text in texts if text.startswith("passage: "))
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([float("apple" in lowered), float("banana" in lowered)])
        matrix = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)


class PoolEncoder(TrackingEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.started_with = None
        self.stopped_with = None
        self.pool = {"pool": "shared"}

    def start_multi_process_pool(self, target_devices):
        self.started_with = list(target_devices)
        return self.pool

    def stop_multi_process_pool(self, pool):
        self.stopped_with = pool


def make_two_query_store(tmp_path):
    store = DiskDatasetStore.create(tmp_path / "dataset.sqlite", name="candidate-test")
    store.add_documents(
        [
            Document("d1", "Apple", "apple orchard"),
            Document("d2", "Banana", "banana grove"),
            Document("d3", "Cherry", "cherry field"),
        ]
    )
    store.add_passages(
        [
            Passage("p1", "d1", "Apple", "red apple", 0),
            Passage("p2", "d1", "Apple", "green apple", 1),
            Passage("p3", "d2", "Banana", "yellow banana", 0),
            Passage("p4", "d3", "Cherry", "red cherry", 0),
        ]
    )
    store.add_queries([Query("q1", "apple"), Query("q2", "banana")])
    store.add_qrels([Qrel("q1", "p3", 1.0), Qrel("q2", "p3", 1.0)])
    store.finalize()
    return store


def test_pipeline_embeds_only_union_of_retrieved_passages_and_prevents_leakage(tmp_path):
    store = make_two_query_store(tmp_path)
    encoder = TrackingEncoder()

    report = run_scalable_phase1_benchmark(
        store,
        run_mode="full",
        experiment_names=["dense__dense"],
        index_dir=tmp_path / "indexes",
        output_dir=tmp_path / "outputs",
        encoder=encoder,
        document_top_k=1,
        passage_top_k=3,
        verbose=False,
    )

    result = report.experiment_results["dense__dense"]
    assert [row.item_id for row in result["runs"]["q1"].passage_results] == ["p1", "p2"]
    assert [row.item_id for row in result["runs"]["q2"].passage_results] == ["p3"]
    candidate_metadata = list((tmp_path / "indexes").glob("candidate_passage_*.json"))
    assert candidate_metadata
    metadata = json.loads(candidate_metadata[0].read_text(encoding="utf-8"))
    assert metadata["count"] == 3
    assert not (tmp_path / "indexes" / "passage_embeddings.npy").exists()
    assert encoder.corpus_texts[:3] == [
        "passage: Apple\napple orchard",
        "passage: Banana\nbanana grove",
        "passage: Cherry\ncherry field",
    ]
    assert set(encoder.corpus_texts[3:]) == {
        "passage: Apple\nred apple",
        "passage: Apple\ngreen apple",
        "passage: Banana\nyellow banana",
    }


def test_resumable_index_continues_after_last_checkpoint(tmp_path):
    store = make_two_query_store(tmp_path)
    interrupted = TrackingEncoder(fail_after=1)
    index_dir = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        MemmapDenseIndex(
            store,
            "document",
            index_dir,
            encoder=interrupted,
            embedding_write_chunk_size=1,
            checkpoint_rows=1,
        ).build()

    progress = json.loads(
        (index_dir / "document_embeddings.progress.json").read_text(encoding="utf-8")
    )
    assert progress["completed_count"] == 1

    resumed = TrackingEncoder()
    MemmapDenseIndex(
        store,
        "document",
        index_dir,
        encoder=resumed,
        embedding_write_chunk_size=1,
        checkpoint_rows=1,
    ).build()
    assert resumed.calls == 2
    assert all("apple orchard" not in text for text in resumed.corpus_texts)
    assert not (index_dir / "document_embeddings.progress.json").exists()


def test_multi_gpu_runtime_reuses_and_stops_one_pool():
    encoder = PoolEncoder()
    runtime = DenseEncoderRuntime(
        encoder=encoder,
        enable_multi_gpu=True,
        preferred_devices=("cuda:0", "cuda:1"),
        available_devices=("cuda:0", "cuda:1"),
        batch_size=2,
        multi_process_chunk_size=4,
    )

    with runtime:
        runtime.encode(["passage: apple"])
        runtime.encode(["passage: banana"])

    assert encoder.started_with == ["cuda:0", "cuda:1"]
    assert encoder.stopped_with is encoder.pool
    assert runtime.pool_start_count == 1


def test_multi_gpu_pool_stops_when_encoding_fails():
    encoder = PoolEncoder()
    runtime = DenseEncoderRuntime(
        encoder=encoder,
        enable_multi_gpu=True,
        preferred_devices=("cuda:0", "cuda:1"),
        available_devices=("cuda:0", "cuda:1"),
    )

    with pytest.raises(RuntimeError, match="stop test"):
        with runtime:
            raise RuntimeError("stop test")

    assert encoder.stopped_with is encoder.pool


def test_incompatible_partial_checkpoint_is_rejected(tmp_path):
    store = make_two_query_store(tmp_path)
    index_dir = tmp_path / "incompatible"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        MemmapDenseIndex(
            store,
            "document",
            index_dir,
            encoder=TrackingEncoder(fail_after=1),
            embedding_write_chunk_size=1,
            checkpoint_rows=1,
        ).build()

    with pytest.raises(ValueError, match="Incompatible partial"):
        MemmapDenseIndex(
            store,
            "document",
            index_dir,
            encoder=TrackingEncoder(),
            model_name="different-model",
            embedding_write_chunk_size=1,
            checkpoint_rows=1,
        ).build()


def test_each_document_method_uses_its_own_passage_candidates(tmp_path):
    store = DiskDatasetStore.create(tmp_path / "method-dataset.sqlite", name="method-test")
    store.add_documents(
        [Document("d1", "Alpha", "apple"), Document("d2", "Beta", "fruit banana")]
    )
    store.add_passages(
        [Passage("p1", "d1", "Alpha", "apple", 0), Passage("p2", "d2", "Beta", "banana", 0)]
    )
    store.add_queries([Query("q", "fruit")])
    store.add_qrels([Qrel("q", "p2", 1.0)])
    store.finalize()

    report = run_scalable_phase1_benchmark(
        store,
        run_mode="full",
        experiment_names=["sparse__dense", "dense__dense", "combined__dense"],
        index_dir=tmp_path / "method-indexes",
        output_dir=tmp_path / "method-outputs",
        encoder=TrackingEncoder(),
        document_top_k=1,
        passage_top_k=1,
        verbose=False,
    )

    sparse_run = report.experiment_results["sparse__dense"]["runs"]["q"]
    dense_run = report.experiment_results["dense__dense"]["runs"]["q"]
    assert [row.item_id for row in sparse_run.passage_results] == ["p2"]
    assert [row.item_id for row in dense_run.passage_results] == ["p1"]


def test_natural_questions_and_nq_hard_can_share_document_embeddings(tmp_path):
    provenance = {
        "source": "UKPLab/dapr",
        "revision": "fixed",
        "corpus_key": "natural_questions",
    }

    def create_store(path, name):
        store = DiskDatasetStore.create(path, name=name, provenance=provenance)
        store.add_documents([Document("d", "Shared", "apple")])
        store.add_passages([Passage("p", "d", "Shared", "apple", 0)])
        store.add_queries([Query("q", "apple")])
        store.add_qrels([Qrel("q", "p", 1.0)])
        store.finalize()
        return store

    natural_questions = create_store(tmp_path / "nq.sqlite", "natural_questions")
    nq_hard = create_store(tmp_path / "nq-hard.sqlite", "nq_hard")
    checkpoint_dir = tmp_path / "shared-checkpoint"
    MemmapDenseIndex(
        natural_questions,
        "document",
        checkpoint_dir,
        encoder=TrackingEncoder(),
    ).build()
    reuse_encoder = TrackingEncoder()
    MemmapDenseIndex(
        nq_hard,
        "document",
        checkpoint_dir,
        encoder=reuse_encoder,
    ).build()

    assert reuse_encoder.calls == 0


def test_candidate_sparse_ranks_locally_without_a_global_fts_database(tmp_path):
    store = make_two_query_store(tmp_path)
    index = CandidateSparseIndex(
        store,
        tmp_path / "candidate-index",
        ["p1", "p2", "p3"],
        cache_name="candidate",
    ).build()

    results = index.search("apple", 2, ["p1", "p2"])

    assert {result.item_id for result in results} == {"p1", "p2"}
    assert not index.database_path.exists()


def test_full_matrix_reuses_each_base_passage_retrieval(monkeypatch, tmp_path):
    store = make_two_query_store(tmp_path)
    sparse_calls = 0
    dense_calls = 0
    original_sparse = CandidateSparseIndex.search
    original_dense = MemmapDenseIndex.search_vector

    def count_sparse(self, *args, **kwargs):
        nonlocal sparse_calls
        sparse_calls += 1
        return original_sparse(self, *args, **kwargs)

    def count_dense(self, *args, **kwargs):
        nonlocal dense_calls
        dense_calls += 1
        return original_dense(self, *args, **kwargs)

    monkeypatch.setattr(CandidateSparseIndex, "search", count_sparse)
    monkeypatch.setattr(MemmapDenseIndex, "search_vector", count_dense)
    run_scalable_phase1_benchmark(
        store,
        run_mode="full",
        index_dir=tmp_path / "reuse-indexes",
        output_dir=tmp_path / "reuse-outputs",
        encoder=TrackingEncoder(),
        document_top_k=1,
        passage_top_k=2,
        verbose=False,
    )

    # Two queries x three document methods. Combined passage retrieval must fuse
    # these cached base results instead of searching the same candidates again.
    assert sparse_calls == 6
    assert dense_calls == 6
