"""Rank-level fusion functions kept independent for ablation tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .retrieval.base import SearchResult


def deduplicate_results(
    results: Iterable[SearchResult],
    k: int | None = None,
) -> list[SearchResult]:
    best: dict[str, float] = {}
    order: list[str] = []
    for result in results:
        if result.item_id not in best:
            order.append(result.item_id)
            best[result.item_id] = result.score
        else:
            best[result.item_id] = max(best[result.item_id], result.score)
    if k is not None:
        order = order[:k]
    return [SearchResult(item_id, best[item_id], rank) for rank, item_id in enumerate(order, 1)]


def hhr_interleave(*rankings: list[SearchResult], k: int) -> list[SearchResult]:
    """Round-robin interleaving compatible with the simple HHR hybrid baseline."""
    merged: list[SearchResult] = []
    max_length = max((len(ranking) for ranking in rankings), default=0)
    for index in range(max_length):
        for ranking in rankings:
            if index < len(ranking):
                merged.append(ranking[index])
    return deduplicate_results(merged, k=k)


def reciprocal_rank_fusion(
    *rankings: list[SearchResult],
    k: int,
    rrf_k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for result in ranking:
            scores[result.item_id] += 1.0 / (rrf_k + result.rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [SearchResult(item_id, score, rank) for rank, (item_id, score) in enumerate(ordered, 1)]
