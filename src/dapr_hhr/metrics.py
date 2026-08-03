"""Pure-Python DAPR retrieval metrics and grouped diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from .data import DatasetBundle
from .pipeline import QueryRun


def compute_ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int = 10) -> float:
    dcg = sum(
        (2 ** relevance.get(item_id, 0.0) - 1) / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def compute_recall_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int = 100) -> float:
    relevant_ids = {item_id for item_id, score in relevance.items() if score > 0}
    if not relevant_ids:
        return 0.0
    return len(relevant_ids.intersection(ranked_ids[:k])) / len(relevant_ids)


def _qrels_by_query(bundle: DatasetBundle) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for qrel in bundle.qrels:
        output[qrel.query_id][qrel.passage_id] = qrel.score
    return dict(output)


def evaluate_query(
    run: QueryRun,
    passage_relevance: dict[str, float],
    passage_to_doc: dict[str, str],
    ndcg_k: int = 10,
    recall_k: int = 100,
    document_k: int | None = None,
) -> dict[str, float]:
    passage_ids = [result.item_id for result in run.passage_results]
    document_ids = [result.item_id for result in run.document_results]
    document_relevance: dict[str, float] = {}
    for passage_id, score in passage_relevance.items():
        if passage_id in passage_to_doc:
            doc_id = passage_to_doc[passage_id]
            document_relevance[doc_id] = max(score, document_relevance.get(doc_id, 0.0))
    document_k = document_k or recall_k
    return {
        f"passage_ndcg@{ndcg_k}": compute_ndcg_at_k(passage_ids, passage_relevance, ndcg_k),
        f"passage_recall@{recall_k}": compute_recall_at_k(passage_ids, passage_relevance, recall_k),
        f"document_ndcg@{ndcg_k}": compute_ndcg_at_k(document_ids, document_relevance, ndcg_k),
        f"document_recall@{document_k}": compute_recall_at_k(
            document_ids, document_relevance, document_k
        ),
        "latency_ms": run.latency_ms,
    }


def evaluate_runs(
    runs: dict[str, QueryRun],
    bundle: DatasetBundle,
    ndcg_k: int = 10,
    recall_k: int = 100,
    document_k: int | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    qrels = _qrels_by_query(bundle)
    passage_to_doc = {passage.passage_id: passage.doc_id for passage in bundle.passages}
    per_query = {
        query_id: evaluate_query(
            run,
            qrels.get(query_id, {}),
            passage_to_doc,
            ndcg_k,
            recall_k,
            document_k,
        )
        for query_id, run in runs.items()
    }
    metric_names = next(iter(per_query.values())).keys() if per_query else []
    aggregate = {metric: mean(row[metric] for row in per_query.values()) for metric in metric_names}
    aggregate["evaluated_queries"] = float(len(per_query))
    return aggregate, per_query


def evaluate_by_group(
    per_query: dict[str, dict[str, float]],
    query_metadata: dict[str, dict[str, Any]],
    category_key: str = "categories",
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query_id, metrics in per_query.items():
        categories: Iterable[str] = query_metadata.get(query_id, {}).get(category_key, [])
        for category in categories:
            grouped[str(category)].append(metrics)
    output = {}
    for category, rows in grouped.items():
        output[category] = {metric: mean(row[metric] for row in rows) for metric in rows[0]}
        output[category]["queries"] = float(len(rows))
    return output
