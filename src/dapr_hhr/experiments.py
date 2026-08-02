"""Experiment registry and reproducible HHR runner."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import BenchmarkConfig
from .data import DatasetBundle
from .metrics import evaluate_runs
from .pipeline import HHRPipeline
from .retrieval import BM25Retriever, CombinedRetriever, SentenceTransformerRetriever
from .retrieval.base import BaseRetriever

METHODS = ("sparse", "dense", "combined")


@dataclass(frozen=True)
class HHRExperiment:
    name: str
    document_method: str
    passage_method: str


def build_experiment_registry() -> dict[str, HHRExperiment]:
    return {
        f"{document_method}__{passage_method}": HHRExperiment(
            name=f"{document_method}__{passage_method}",
            document_method=document_method,
            passage_method=passage_method,
        )
        for document_method in METHODS
        for passage_method in METHODS
    }


def build_retriever(
    method: str,
    config: BenchmarkConfig,
    cache_path: str | Path | None = None,
    dense_factory: Callable[..., BaseRetriever] = SentenceTransformerRetriever,
) -> BaseRetriever:
    if method == "sparse":
        return BM25Retriever()
    dense = dense_factory(
        model_name=config.retrieval.dense_model,
        batch_size=config.retrieval.dense_batch_size,
        query_prefix=config.retrieval.dense_query_prefix,
        corpus_prefix=config.retrieval.dense_corpus_prefix,
        cache_path=cache_path,
    )
    if method == "dense":
        return dense
    if method == "combined":
        return CombinedRetriever(
            BM25Retriever(),
            dense,
            fusion=config.retrieval.fusion,
            rrf_k=config.retrieval.rrf_k,
        )
    raise ValueError(f"Unknown retrieval method: {method}")


def run_hhr_experiment(
    experiment: HHRExperiment,
    bundle: DatasetBundle,
    config: BenchmarkConfig,
    cache_dir: str | Path | None = None,
    dense_factory: Callable[..., BaseRetriever] = SentenceTransformerRetriever,
) -> dict:
    cache_dir = Path(cache_dir) if cache_dir else None
    fingerprint_hash = hashlib.sha1(config.retrieval.dense_model.encode("utf-8"))
    for document in bundle.documents:
        fingerprint_hash.update(b"\0d\0")
        fingerprint_hash.update(document.doc_id.encode("utf-8"))
    for passage in bundle.passages:
        fingerprint_hash.update(b"\0p\0")
        fingerprint_hash.update(passage.passage_id.encode("utf-8"))
    fingerprint = fingerprint_hash.hexdigest()[:12]
    document_cache = (
        cache_dir / f"{bundle.name}_{fingerprint}_documents.npy" if cache_dir else None
    )
    passage_cache = (
        cache_dir / f"{bundle.name}_{fingerprint}_passages.npy" if cache_dir else None
    )
    document_retriever = build_retriever(
        experiment.document_method,
        config,
        cache_path=document_cache,
        dense_factory=dense_factory,
    )
    passage_retriever = build_retriever(
        experiment.passage_method,
        config,
        cache_path=passage_cache,
        dense_factory=dense_factory,
    )
    pipeline = HHRPipeline(
        document_retriever,
        passage_retriever,
        document_top_k=config.retrieval.document_top_k,
        passage_top_k=config.retrieval.passage_top_k,
    ).fit(bundle)
    runs = pipeline.run(bundle.queries)
    metrics, per_query = evaluate_runs(
        runs,
        bundle,
        ndcg_k=config.evaluation.ndcg_k,
        recall_k=config.evaluation.recall_k,
        document_k=config.retrieval.document_top_k,
    )
    return {
        "experiment": asdict(experiment),
        "dataset": bundle.name,
        "metrics": metrics,
        "per_query": per_query,
        "runs": runs,
    }
