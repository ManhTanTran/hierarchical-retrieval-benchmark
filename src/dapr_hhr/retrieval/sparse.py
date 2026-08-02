"""A compact, dependency-light BM25 implementation backed by SciPy sparse matrices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .base import BaseRetriever, SearchResult


class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
        self.item_ids: list[str] = []
        self.id_to_index: dict[str, int] = {}
        self.matrix: sparse.csr_matrix | None = None

    def fit(self, items: Mapping[str, str]) -> BM25Retriever:
        if not items:
            raise ValueError("Cannot fit BM25 on an empty collection.")
        self.item_ids = list(items)
        self.id_to_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        counts = self.vectorizer.fit_transform(items[item_id] for item_id in self.item_ids).tocsr()
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        avg_len = float(doc_len.mean()) or 1.0
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(self.item_ids) - df + 0.5) / (df + 0.5))

        row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        denominator = counts.data + self.k1 * (
            1 - self.b + self.b * doc_len[row_ids] / avg_len
        )
        weights = counts.data * (self.k1 + 1) / denominator
        weights *= idf[counts.indices]
        self.matrix = sparse.csr_matrix(
            (weights.astype(np.float32), counts.indices, counts.indptr),
            shape=counts.shape,
        )
        return self

    def search(
        self,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("Call fit() before search().")
        query_counts = self.vectorizer.transform([query])
        scores = (self.matrix @ query_counts.T).toarray().ravel()
        if candidate_ids is None:
            indices = np.arange(len(self.item_ids))
        else:
            indices = np.array(
                [
                    self.id_to_index[item_id]
                    for item_id in candidate_ids
                    if item_id in self.id_to_index
                ],
                dtype=int,
            )
        if not len(indices) or k <= 0:
            return []
        k = min(k, len(indices))
        candidate_scores = scores[indices]
        local = np.argpartition(-candidate_scores, k - 1)[:k]
        ranked_indices = indices[local[np.argsort(-candidate_scores[local], kind="stable")]]
        return [
            SearchResult(self.item_ids[index], float(scores[index]), rank)
            for rank, index in enumerate(ranked_indices, start=1)
        ]
