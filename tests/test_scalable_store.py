from __future__ import annotations

from dapr_hhr import Document, Passage, Qrel, Query
from dapr_hhr.scalable import DiskDatasetStore


def make_store(tmp_path):
    store = DiskDatasetStore.create(tmp_path / "dataset.sqlite", name="tiny")
    store.add_documents(
        [
            Document("d1", "Alpha", "apple orchard"),
            Document("d2", "Beta", "banana grove"),
        ]
    )
    store.add_passages(
        [
            Passage("p1", "d1", "Alpha", "red apple", 0),
            Passage("p2", "d1", "Alpha", "green apple", 1),
            Passage("p3", "d2", "Beta", "yellow banana", 0),
        ]
    )
    store.add_queries([Query("q1", "apple")])
    store.add_qrels([Qrel("q1", "p1", 1.0)])
    store.finalize()
    return store


def test_disk_store_validates_summarizes_and_streams(tmp_path):
    store = make_store(tmp_path)

    store.validate()
    assert store.summary() == {
        "name": "tiny",
        "documents": 2,
        "passages": 3,
        "queries": 1,
        "qrels": 1,
        "graded_qrels": 0,
        "queries_with_relevant_passages": 1,
        "provenance": {},
    }
    assert list(store.iter_items("document")) == [
        (0, "d1", "Alpha\napple orchard"),
        (1, "d2", "Beta\nbanana grove"),
    ]
    assert store.passage_ids_for_documents(["d1"]) == ["p1", "p2"]
    assert store.passage_to_document(["p1", "p3"]) == {"p1": "d1", "p3": "d2"}


def test_disk_store_fts_search_supports_candidates(tmp_path):
    store = make_store(tmp_path)

    assert [row.item_id for row in store.sparse_search("document", "apple", 2)] == ["d1"]
    assert [row.item_id for row in store.sparse_search("passage", "apple", 2)] == [
        "p1",
        "p2",
    ]
    assert [
        row.item_id for row in store.sparse_search("passage", "apple", 2, candidate_ids=["p3"])
    ] == []
