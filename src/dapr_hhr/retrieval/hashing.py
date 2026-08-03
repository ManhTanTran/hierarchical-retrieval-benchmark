"""Deterministic dependency-free dense-like retriever for plumbing smoke tests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping

import numpy as np

from .base import BaseRetriever, SearchResult

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


class HashingRetriever(BaseRetriever):
    """Signed feature hashing; never a substitute for a benchmark encoder."""

    def __init__(self, dimensions: int = 512, **_: object) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.item_ids: list[str] = []
        self.id_to_index: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        matrix = np.zeros((len(values), self.dimensions), dtype=np.float32)
        for row, text in enumerate(values):
            for token in TOKEN_PATTERN.findall(str(text).lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                matrix[row, value % self.dimensions] += 1.0 if (value >> 8) % 2 == 0 else -1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0.0, 1.0, norms)

    def fit(self, items: Mapping[str, str]) -> HashingRetriever:
        if not items:
            raise ValueError("Cannot fit hashing retriever on an empty collection.")
        self.item_ids = list(items)
        self.id_to_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        self.embeddings = self._encode(items[item_id] for item_id in self.item_ids)
        return self

    def search(
        self,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before search().")
        if candidate_ids is None:
            indices = np.arange(len(self.item_ids))
        else:
            indices = np.asarray(
                [
                    self.id_to_index[item_id]
                    for item_id in candidate_ids
                    if item_id in self.id_to_index
                ],
                dtype=int,
            )
        if not len(indices) or k <= 0:
            return []
        query_embedding = self._encode([query])[0]
        scores = self.embeddings[indices] @ query_embedding
        order = sorted(
            range(len(indices)),
            key=lambda position: (
                -float(scores[position]),
                self.item_ids[int(indices[position])],
            ),
        )[:k]
        return [
            SearchResult(
                self.item_ids[int(indices[position])],
                float(scores[position]),
                rank,
            )
            for rank, position in enumerate(order, start=1)
        ]
