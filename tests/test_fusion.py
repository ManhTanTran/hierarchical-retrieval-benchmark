from dapr_hhr.fusion import hhr_interleave, reciprocal_rank_fusion
from dapr_hhr.retrieval.base import SearchResult


def ranking(*ids: str):
    return [SearchResult(item_id, 1.0 / rank, rank) for rank, item_id in enumerate(ids, 1)]


def test_interleave_deduplicates_and_round_robins():
    result = hhr_interleave(ranking("a", "b"), ranking("b", "c"), k=3)
    assert [item.item_id for item in result] == ["a", "b", "c"]


def test_rrf_rewards_items_found_by_both_retrievers():
    result = reciprocal_rank_fusion(ranking("a", "b"), ranking("b", "c"), k=3, rrf_k=1)
    assert result[0].item_id == "b"
