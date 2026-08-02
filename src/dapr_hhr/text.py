"""Reusable document and passage text construction strategies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .data import Document, Passage


def build_document_text(
    document: Document,
    passages: Iterable[Passage] = (),
    strategy: str = "title_body",
    max_passages: int | None = None,
) -> str:
    """Build document retrieval text without changing dataset records."""
    doc_passages = [item for item in passages if item.doc_id == document.doc_id]
    doc_passages.sort(
        key=lambda item: item.paragraph_no if item.paragraph_no is not None else 10**9
    )
    if max_passages is not None:
        doc_passages = doc_passages[:max_passages]

    if strategy == "title":
        return document.title.strip()
    if strategy == "title_passages" and doc_passages:
        body = "\n".join(item.text for item in doc_passages)
        return f"{document.title}\n{body}".strip()
    if strategy == "body":
        return document.text.strip()
    if strategy != "title_body":
        raise ValueError(f"Unknown document text strategy: {strategy}")
    return f"{document.title}\n{document.text}".strip()


def build_document_texts(
    documents: Iterable[Document],
    passages: Iterable[Passage],
    strategy: str = "title_body",
    max_passages: int | None = None,
) -> dict[str, str]:
    passage_map: dict[str, list[Passage]] = defaultdict(list)
    for passage in passages:
        passage_map[passage.doc_id].append(passage)
    return {
        document.doc_id: build_document_text(
            document,
            passage_map.get(document.doc_id, ()),
            strategy=strategy,
            max_passages=max_passages,
        )
        for document in documents
    }


def build_passage_text(passage: Passage, strategy: str = "title_text") -> str:
    if strategy == "text":
        return passage.text.strip()
    if strategy != "title_text":
        raise ValueError(f"Unknown passage text strategy: {strategy}")
    return f"{passage.title}\n{passage.text}".strip()
