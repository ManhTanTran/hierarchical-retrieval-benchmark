"""Two-stage document-to-passage hierarchical retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

from .data import DatasetBundle, Query
from .retrieval.base import BaseRetriever, SearchResult
from .text import build_document_texts, build_passage_text


@dataclass
class QueryRun:
    query_id: str
    document_results: list[SearchResult]
    passage_results: list[SearchResult]
    latency_ms: float


class HHRPipeline:
    def __init__(
        self,
        document_retriever: BaseRetriever,
        passage_retriever: BaseRetriever,
        document_top_k: int = 20,
        passage_top_k: int = 100,
        document_text_strategy: str = "title_body",
        passage_text_strategy: str = "title_text",
    ) -> None:
        self.document_retriever = document_retriever
        self.passage_retriever = passage_retriever
        self.document_top_k = document_top_k
        self.passage_top_k = passage_top_k
        self.document_text_strategy = document_text_strategy
        self.passage_text_strategy = passage_text_strategy
        self.passages_by_doc: dict[str, list[str]] = {}

    def fit(self, bundle: DatasetBundle) -> HHRPipeline:
        document_texts = build_document_texts(
            bundle.documents,
            bundle.passages,
            strategy=self.document_text_strategy,
        )
        passage_texts = {
            passage.passage_id: build_passage_text(passage, self.passage_text_strategy)
            for passage in bundle.passages
        }
        grouped: dict[str, list[str]] = defaultdict(list)
        for passage in bundle.passages:
            grouped[passage.doc_id].append(passage.passage_id)
        self.passages_by_doc = dict(grouped)
        self.document_retriever.fit(document_texts)
        self.passage_retriever.fit(passage_texts)
        return self

    def search(self, query: Query) -> QueryRun:
        start = perf_counter()
        document_results = self.document_retriever.search(query.text, self.document_top_k)
        candidate_passages = [
            passage_id
            for result in document_results
            for passage_id in self.passages_by_doc.get(result.item_id, [])
        ]
        passage_results = self.passage_retriever.search(
            query.text,
            self.passage_top_k,
            candidate_ids=candidate_passages,
        )
        return QueryRun(
            query_id=query.query_id,
            document_results=document_results,
            passage_results=passage_results,
            latency_ms=(perf_counter() - start) * 1000,
        )

    def run(self, queries: Iterable[Query]) -> dict[str, QueryRun]:
        return {query.query_id: self.search(query) for query in queries}


def run_hhr_pipeline(
    bundle: DatasetBundle,
    document_retriever: BaseRetriever,
    passage_retriever: BaseRetriever,
    document_top_k: int = 20,
    passage_top_k: int = 100,
) -> dict[str, QueryRun]:
    pipeline = HHRPipeline(
        document_retriever,
        passage_retriever,
        document_top_k=document_top_k,
        passage_top_k=passage_top_k,
    ).fit(bundle)
    return pipeline.run(bundle.queries)
