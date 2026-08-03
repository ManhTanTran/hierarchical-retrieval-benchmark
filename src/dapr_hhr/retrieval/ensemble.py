"""A fused sparse+dense retriever used at either HHR level."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..fusion import hhr_interleave, reciprocal_rank_fusion
from .base import BaseRetriever, SearchResult


class CombinedRetriever(BaseRetriever):
    def __init__(
        self,
        sparse: BaseRetriever,
        dense: BaseRetriever,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self.fusion = fusion
        self.rrf_k = rrf_k

    def fit(self, items: Mapping[str, str]) -> CombinedRetriever:
        self.sparse.fit(items)
        self.dense.fit(items)
        return self

    def search(
        self,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        sparse_results = self.sparse.search(query, k, candidate_ids)
        dense_results = self.dense.search(query, k, candidate_ids)
        if self.fusion == "interleave":
            return hhr_interleave(sparse_results, dense_results, k=k)
        if self.fusion != "rrf":
            raise ValueError(f"Unknown fusion method: {self.fusion}")
        return reciprocal_rank_fusion(sparse_results, dense_results, k=k, rrf_k=self.rrf_k)
