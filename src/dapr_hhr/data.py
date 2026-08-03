"""Dataset-agnostic retrieval contracts and structural validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    title: str
    text: str
    paragraph_no: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Qrel:
    query_id: str
    passage_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    documents: list[Document]
    passages: list[Passage]
    queries: list[Query]
    qrels: list[Qrel]
    name: str = "unknown"
    query_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_dataset(self)


def documents_from_passages(passages: Iterable[Passage]) -> list[Document]:
    grouped: dict[str, list[Passage]] = {}
    for passage in passages:
        grouped.setdefault(passage.doc_id, []).append(passage)
    documents = []
    for doc_id, items in grouped.items():
        ordered = sorted(
            items,
            key=lambda item: item.paragraph_no if item.paragraph_no is not None else 10**9,
        )
        body = "\n".join(item.text for item in ordered)
        documents.append(Document(doc_id, ordered[0].title, body))
    return documents


def validate_dataset(bundle: DatasetBundle) -> None:
    errors: list[str] = []
    if not bundle.queries:
        errors.append("dataset contains no queries")
    if not bundle.passages:
        errors.append("dataset contains no passages")
    if not bundle.documents:
        errors.append("dataset contains no documents")

    query_ids = [str(item.query_id) for item in bundle.queries]
    passage_ids = [str(item.passage_id) for item in bundle.passages]
    document_ids = [str(item.doc_id) for item in bundle.documents]
    if len(set(query_ids)) != len(query_ids):
        errors.append("duplicate query IDs found")
    if len(set(passage_ids)) != len(passage_ids):
        errors.append("duplicate passage IDs found")
    if len(set(document_ids)) != len(document_ids):
        errors.append("duplicate document IDs found")
    if any(not value.strip() for value in query_ids):
        errors.append("query IDs must not be empty")
    if any(not value.strip() for value in passage_ids):
        errors.append("passage IDs must not be empty")
    if any(not value.strip() for value in document_ids):
        errors.append("document IDs must not be empty")

    query_id_set = set(query_ids)
    passage_id_set = set(passage_ids)
    document_id_set = set(document_ids)
    if missing := {str(item.doc_id) for item in bundle.passages} - document_id_set:
        errors.append(f"passages reference missing documents: {sorted(missing)[:3]}")
    if missing := {str(item.query_id) for item in bundle.qrels} - query_id_set:
        errors.append(f"qrels reference missing queries: {sorted(missing)[:3]}")
    if missing := {str(item.passage_id) for item in bundle.qrels} - passage_id_set:
        errors.append(f"qrels reference missing passages: {sorted(missing)[:3]}")
    if any(item.paragraph_no is not None and item.paragraph_no < 0 for item in bundle.passages):
        errors.append("paragraph numbers must be non-negative")
    if any(float(item.score) < 0 for item in bundle.qrels):
        errors.append("qrel scores must be non-negative")
    if unknown := set(bundle.query_metadata) - query_id_set:
        errors.append(f"query metadata references missing queries: {sorted(unknown)[:3]}")
    if errors:
        raise ValueError("Invalid dataset bundle:\n- " + "\n- ".join(errors))


def summarize_dataset(bundle: DatasetBundle) -> dict[str, Any]:
    relevant = [qrel for qrel in bundle.qrels if qrel.score > 0]
    return {
        "name": bundle.name,
        "documents": len(bundle.documents),
        "passages": len(bundle.passages),
        "queries": len(bundle.queries),
        "qrels": len(bundle.qrels),
        "graded_qrels": sum(qrel.score > 1 for qrel in bundle.qrels),
        "queries_with_relevant_passages": len({qrel.query_id for qrel in relevant}),
        "provenance": dict(bundle.provenance),
    }
