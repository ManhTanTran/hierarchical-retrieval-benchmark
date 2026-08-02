"""Tiny synthetic collection for fast end-to-end verification without downloads."""

from __future__ import annotations

from .data import DatasetBundle, Document, Passage, Qrel, Query


def make_synthetic_bundle() -> DatasetBundle:
    passages = [
        Passage(
            "p1",
            "d1",
            "Half Moon Putney",
            "Artists at the venue included the Rolling Stones.",
            0,
        ),
        Passage("p2", "d1", "Half Moon Putney", "The pub is a music venue in London.", 1),
        Passage("p3", "d2", "Apollo Theatre", "The theatre hosts plays and musicals.", 0),
        Passage("p4", "d3", "Genome", "DNA contains genetic instructions.", 0),
    ]
    documents = [
        Document(
            "d1",
            "Half Moon Putney",
            "A London pub and music venue. Artists included the Rolling Stones.",
        ),
        Document("d2", "Apollo Theatre", "A theatre for plays and musicals."),
        Document("d3", "Genome", "DNA and genetic instructions."),
    ]
    queries = [
        Query("q1", "who played at the Half Moon Putney"),
        Query("q2", "genetic instructions DNA"),
    ]
    qrels = [Qrel("q1", "p1", 1.0), Qrel("q2", "p4", 1.0)]
    return DatasetBundle(documents, passages, queries, qrels, name="synthetic")
