import pytest

from dapr_hhr.metrics import compute_ndcg_at_k, compute_recall_at_k


def test_metrics_are_one_for_perfect_ranking():
    relevance = {"p1": 2.0, "p2": 1.0}
    assert compute_ndcg_at_k(["p1", "p2"], relevance, 2) == pytest.approx(1.0)
    assert compute_recall_at_k(["p1", "p2"], relevance, 2) == pytest.approx(1.0)


def test_recall_handles_no_relevant_items():
    assert compute_recall_at_k(["p1"], {}, 10) == 0.0
