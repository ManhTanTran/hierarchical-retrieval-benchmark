"""Sentence-transformer exact retrieval suitable for smoke and medium-size Kaggle runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .base import BaseRetriever, SearchResult


class SentenceTransformerRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        batch_size: int = 64,
        query_prefix: str = "query: ",
        corpus_prefix: str = "passage: ",
        cache_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.corpus_prefix = corpus_prefix
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = device
        self.model = None
        self.item_ids: list[str] = []
        self.id_to_index: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None

    def _get_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError("Install `sentence-transformers` for dense retrieval.") from exc
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def fit(self, items: Mapping[str, str]) -> SentenceTransformerRetriever:
        if not items:
            raise ValueError("Cannot fit dense retriever on an empty collection.")
        self.item_ids = list(items)
        self.id_to_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        if self.cache_path and self.cache_path.exists():
            cached = np.load(self.cache_path, allow_pickle=False)
            if cached.shape[0] == len(self.item_ids):
                self.embeddings = cached
                return self
        texts = [self.corpus_prefix + items[item_id] for item_id in self.item_ids]
        self.embeddings = np.asarray(
            self._get_model().encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, self.embeddings, allow_pickle=False)
        return self

    def search(
        self,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before search().")
        query_embedding = np.asarray(
            self._get_model().encode(
                [self.query_prefix + query],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0],
            dtype=np.float32,
        )
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
        scores = self.embeddings[indices] @ query_embedding
        k = min(k, len(indices))
        local = np.argpartition(-scores, k - 1)[:k]
        order = local[np.argsort(-scores[local], kind="stable")]
        return [
            SearchResult(self.item_ids[indices[position]], float(scores[position]), rank)
            for rank, position in enumerate(order, start=1)
        ]
