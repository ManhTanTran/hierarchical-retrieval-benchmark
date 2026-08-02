"""Small retrieval interface shared across sparse, dense, and fused methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float
    rank: int


class BaseRetriever(ABC):
    @abstractmethod
    def fit(self, items: Mapping[str, str]) -> BaseRetriever:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError

