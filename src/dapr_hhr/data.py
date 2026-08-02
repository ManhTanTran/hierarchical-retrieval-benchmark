"""Dataset contracts and a Hugging Face adapter for the official DAPR release."""

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

    def validate(self) -> None:
        validate_dataset(self)


def _pick(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


class DAPRDatasetAdapter:
    """Load and normalize any of the five official DAPR datasets."""

    SUPPORTED = {
        "MSMARCO",
        "NaturalQuestions",
        "MIRACL",
        "Genomics",
        "ConditionalQA",
    }

    def __init__(self, hf_repo: str = "UKPLab/dapr") -> None:
        self.hf_repo = hf_repo

    def _load(self, config_name: str, split: str):
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("Install `datasets>=2.15` to load DAPR.") from exc
        return load_dataset(self.hf_repo, config_name, split=split)

    def load_queries(self, dataset_name: str, split: str = "test") -> list[Query]:
        rows = self._load(f"{dataset_name}-queries", split)
        return [
            Query(
                query_id=str(_pick(row, "_id", "query_id", "id")),
                text=str(_pick(row, "text", "query")),
                metadata={
                    k: v
                    for k, v in row.items()
                    if k not in {"_id", "query_id", "id", "text", "query"}
                },
            )
            for row in rows
        ]

    def load_qrels(self, dataset_name: str, split: str = "test") -> list[Qrel]:
        rows = self._load(f"{dataset_name}-qrels", split)
        return [
            Qrel(
                query_id=str(_pick(row, "query_id", "qid")),
                passage_id=str(_pick(row, "corpus_id", "passage_id", "pid")),
                score=float(_pick(row, "score", "relevance", default=1.0)),
                metadata={
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "query_id",
                        "qid",
                        "corpus_id",
                        "passage_id",
                        "pid",
                        "score",
                        "relevance",
                    }
                },
            )
            for row in rows
        ]

    def load_passages(self, dataset_name: str) -> list[Passage]:
        rows = self._load(f"{dataset_name}-corpus", "test")
        return [self._passage_from_row(row) for row in rows]

    def load_documents(self, dataset_name: str) -> list[Document]:
        try:
            rows = self._load(f"{dataset_name}-docs", "test")
            return [self._document_from_row(row) for row in rows]
        except Exception:
            return []

    @staticmethod
    def _passage_from_row(row: dict[str, Any]) -> Passage:
        excluded = {"_id", "passage_id", "id", "doc_id", "title", "text", "paragraph_no"}
        paragraph_no = _pick(row, "paragraph_no", default=None)
        return Passage(
            passage_id=str(_pick(row, "_id", "passage_id", "id")),
            doc_id=str(_pick(row, "doc_id", "document_id")),
            title=str(_pick(row, "title")),
            text=str(_pick(row, "text", "passage")),
            paragraph_no=int(paragraph_no) if paragraph_no is not None else None,
            metadata={k: v for k, v in row.items() if k not in excluded},
        )

    @staticmethod
    def _document_from_row(row: dict[str, Any]) -> Document:
        passages = _pick(row, "passages", default=[])
        text = _pick(row, "text", "document", default="")
        if not text and passages:
            text = "\n".join(str(item) for item in passages)
        return Document(
            doc_id=str(_pick(row, "doc_id", "_id", "id")),
            title=str(_pick(row, "title")),
            text=str(text),
        )

    def load_bundle(
        self,
        dataset_name: str,
        split: str = "test",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
        preserve_gold: bool = True,
    ) -> DatasetBundle:
        if dataset_name not in self.SUPPORTED:
            raise ValueError(f"Unsupported DAPR dataset: {dataset_name}")

        queries = self.load_queries(dataset_name, split)
        if query_limit is not None:
            queries = queries[:query_limit]
        query_ids = {query.query_id for query in queries}

        qrels = [
            qrel
            for qrel in self.load_qrels(dataset_name, split)
            if qrel.query_id in query_ids
        ]
        passages = self.load_passages(dataset_name)
        if corpus_limit is not None and len(passages) > corpus_limit:
            selected = passages[:corpus_limit]
            if preserve_gold:
                gold_ids = {qrel.passage_id for qrel in qrels if qrel.score > 0}
                selected_ids = {passage.passage_id for passage in selected}
                selected.extend(
                    passage
                    for passage in passages[corpus_limit:]
                    if passage.passage_id in gold_ids and passage.passage_id not in selected_ids
                )
            passages = selected

        passage_ids = {passage.passage_id for passage in passages}
        qrels = [qrel for qrel in qrels if qrel.passage_id in passage_ids]

        if corpus_limit is not None:
            # Avoid materializing a second, potentially multi-million-document config
            # during smoke runs. Passage rows contain enough context for this baseline.
            documents = documents_from_passages(passages)
        else:
            doc_ids = {passage.doc_id for passage in passages}
            documents = [doc for doc in self.load_documents(dataset_name) if doc.doc_id in doc_ids]
            if not documents:
                documents = documents_from_passages(passages)

        bundle = DatasetBundle(documents, passages, queries, qrels, dataset_name)
        bundle.validate()
        return bundle

    def load_query_metadata(self, dataset_name: str) -> dict[str, dict[str, Any]]:
        """Return NQ-hard diagnostic categories, or an empty mapping for other datasets."""
        if dataset_name != "NaturalQuestions":
            return {}
        rows = self._load("nq-hard", "test")
        output: dict[str, dict[str, Any]] = {}
        for row in rows:
            query_id = str(_pick(row, "query_id", "qid"))
            output.setdefault(query_id, {})["categories"] = row.get("categories", [])
        return output


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
    if not bundle.queries:
        raise ValueError("Dataset contains no queries.")
    if not bundle.passages:
        raise ValueError("Dataset contains no passages.")
    if not bundle.documents:
        raise ValueError("Dataset contains no documents.")
    if len({item.query_id for item in bundle.queries}) != len(bundle.queries):
        raise ValueError("Duplicate query IDs found.")
    if len({item.passage_id for item in bundle.passages}) != len(bundle.passages):
        raise ValueError("Duplicate passage IDs found.")
    passage_ids = {item.passage_id for item in bundle.passages}
    doc_ids = {item.doc_id for item in bundle.documents}
    if missing := {item.doc_id for item in bundle.passages} - doc_ids:
        raise ValueError(f"Passages reference missing documents: {sorted(missing)[:3]}")
    if missing := {item.passage_id for item in bundle.qrels} - passage_ids:
        raise ValueError(f"Qrels reference missing passages: {sorted(missing)[:3]}")


def summarize_dataset(bundle: DatasetBundle) -> dict[str, Any]:
    relevant = [qrel for qrel in bundle.qrels if qrel.score > 0]
    return {
        "name": bundle.name,
        "documents": len(bundle.documents),
        "passages": len(bundle.passages),
        "queries": len(bundle.queries),
        "qrels": len(bundle.qrels),
        "queries_with_relevant_passages": len({qrel.query_id for qrel in relevant}),
    }


def get_dataset_adapter(name: str = "dapr", **kwargs: Any) -> DAPRDatasetAdapter:
    if name.lower() != "dapr":
        raise ValueError(f"Unknown dataset adapter: {name}")
    return DAPRDatasetAdapter(**kwargs)
