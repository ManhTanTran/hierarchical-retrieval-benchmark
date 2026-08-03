"""Disk-backed dataset and exact dense retrieval primitives for large corpora."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from .artifacts import save_json, save_run_artifacts
from .config import BenchmarkConfig, validate_config
from .data import Document, Passage, Qrel, Query
from .experiments import build_experiment_registry
from .fusion import hhr_interleave, reciprocal_rank_fusion
from .metrics import evaluate_by_group, evaluate_query
from .pipeline import QueryRun
from .retrieval.base import SearchResult

Level = Literal["document", "passage"]
_SQLITE_BATCH_SIZE = 10_000
_SQLITE_PARAMETER_CHUNK = 800


def _chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class DiskDatasetStore:
    """Immutable-after-finalize SQLite store with FTS5 sparse indexes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        name: str,
        provenance: dict[str, Any] | None = None,
    ) -> DiskDatasetStore:
        resolved = Path(path)
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite dataset store: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=FILE;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE documents (
                    rowid INTEGER PRIMARY KEY,
                    item_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL
                );
                CREATE TABLE passages (
                    rowid INTEGER PRIMARY KEY,
                    item_id TEXT NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    paragraph_no INTEGER,
                    metadata_json TEXT NOT NULL,
                    search_text TEXT NOT NULL
                );
                CREATE INDEX passages_doc_id_idx ON passages(doc_id);
                CREATE TABLE queries (
                    query_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE qrels (
                    query_id TEXT NOT NULL,
                    passage_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (query_id, passage_id)
                );
                CREATE VIRTUAL TABLE documents_fts USING fts5(
                    item_id UNINDEXED,
                    search_text,
                    content='documents',
                    content_rowid='rowid',
                    tokenize='unicode61'
                );
                CREATE VIRTUAL TABLE passages_fts USING fts5(
                    item_id UNINDEXED,
                    search_text,
                    content='passages',
                    content_rowid='rowid',
                    tokenize='unicode61'
                );
                """
            )
            values = {
                "dataset_id": uuid.uuid4().hex,
                "name": name,
                "provenance": json.dumps(provenance or {}, sort_keys=True),
                "finalized": "0",
            }
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", values.items())
            connection.commit()
        finally:
            connection.close()
        return cls(resolved)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _metadata(self) -> dict[str, str]:
        with self._connect() as connection:
            return {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }

    @property
    def dataset_id(self) -> str:
        return self._metadata()["dataset_id"]

    @property
    def name(self) -> str:
        return self._metadata()["name"]

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(json.loads(self._metadata()["provenance"]))

    def _require_mutable(self) -> None:
        if self._metadata().get("finalized") == "1":
            raise RuntimeError("Dataset store is finalized and cannot be modified.")

    def add_documents(
        self, documents: Iterable[Document], batch_size: int = _SQLITE_BATCH_SIZE
    ) -> None:
        self._require_mutable()
        batch: list[tuple[str, str, str, str]] = []
        with self._connect() as connection:
            for document in documents:
                search_text = f"{document.title}\n{document.text}".strip()
                batch.append((document.doc_id, document.title, document.text, search_text))
                if len(batch) >= batch_size:
                    connection.executemany(
                        """INSERT INTO documents(item_id, title, text, search_text)
                        VALUES (?, ?, ?, ?)""",
                        batch,
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO documents(item_id, title, text, search_text) VALUES (?, ?, ?, ?)",
                    batch,
                )

    def add_passages(
        self, passages: Iterable[Passage], batch_size: int = _SQLITE_BATCH_SIZE
    ) -> None:
        self._require_mutable()
        batch: list[tuple[Any, ...]] = []
        with self._connect() as connection:
            for passage in passages:
                search_text = f"{passage.title}\n{passage.text}".strip()
                batch.append(
                    (
                        passage.passage_id,
                        passage.doc_id,
                        passage.title,
                        passage.text,
                        passage.paragraph_no,
                        json.dumps(passage.metadata, sort_keys=True),
                        search_text,
                    )
                )
                if len(batch) >= batch_size:
                    connection.executemany(
                        """INSERT INTO passages(
                            item_id, doc_id, title, text, paragraph_no, metadata_json, search_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        batch,
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    """INSERT INTO passages(
                        item_id, doc_id, title, text, paragraph_no, metadata_json, search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )

    def add_queries(self, queries: Iterable[Query]) -> None:
        self._require_mutable()
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO queries(query_id, text, metadata_json) VALUES (?, ?, ?)",
                (
                    (query.query_id, query.text, json.dumps(query.metadata, sort_keys=True))
                    for query in queries
                ),
            )

    def add_qrels(self, qrels: Iterable[Qrel]) -> None:
        self._require_mutable()
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO qrels(query_id, passage_id, score, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_id, passage_id) DO UPDATE SET
                    score = max(score, excluded.score), metadata_json = excluded.metadata_json""",
                (
                    (
                        qrel.query_id,
                        qrel.passage_id,
                        float(qrel.score),
                        json.dumps(qrel.metadata, sort_keys=True),
                    )
                    for qrel in qrels
                ),
            )

    def finalize(self) -> None:
        self._require_mutable()
        with self._connect() as connection:
            connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
            connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
            connection.execute("UPDATE metadata SET value = '1' WHERE key = 'finalized'")
            connection.execute("ANALYZE")
        self.validate()

    def item_count(self, level: Level) -> int:
        table = self._table(level)
        with self._connect() as connection:
            return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    @staticmethod
    def _table(level: Level) -> str:
        if level == "document":
            return "documents"
        if level == "passage":
            return "passages"
        raise ValueError(f"Unknown retrieval level: {level}")

    def iter_items(self, level: Level) -> Iterator[tuple[int, str, str]]:
        table = self._table(level)
        connection = self._connect()
        try:
            cursor = connection.execute(
                f"SELECT rowid, item_id, search_text FROM {table} ORDER BY rowid"
            )
            for row in cursor:
                yield int(row["rowid"]) - 1, str(row["item_id"]), str(row["search_text"])
        finally:
            connection.close()

    def queries(self) -> list[Query]:
        with self._connect() as connection:
            return [
                Query(str(row["query_id"]), str(row["text"]), json.loads(row["metadata_json"]))
                for row in connection.execute("SELECT * FROM queries ORDER BY query_id")
            ]

    def qrels(self) -> list[Qrel]:
        with self._connect() as connection:
            return [
                Qrel(
                    str(row["query_id"]),
                    str(row["passage_id"]),
                    float(row["score"]),
                    json.loads(row["metadata_json"]),
                )
                for row in connection.execute("SELECT * FROM qrels ORDER BY query_id, passage_id")
            ]

    def passage_ids_for_documents(self, document_ids: Iterable[str]) -> list[str]:
        values = list(dict.fromkeys(map(str, document_ids)))
        output: list[tuple[int, str]] = []
        with self._connect() as connection:
            for chunk in _chunks(values, _SQLITE_PARAMETER_CHUNK):
                placeholders = ",".join("?" for _ in chunk)
                output.extend(
                    (int(row["rowid"]), str(row["item_id"]))
                    for row in connection.execute(
                        f"SELECT rowid, item_id FROM passages WHERE doc_id IN ({placeholders})",
                        tuple(chunk),
                    )
                )
        return [item_id for _, item_id in sorted(output)]

    def passage_to_document(self, passage_ids: Iterable[str]) -> dict[str, str]:
        values = list(dict.fromkeys(map(str, passage_ids)))
        output: dict[str, str] = {}
        with self._connect() as connection:
            for chunk in _chunks(values, _SQLITE_PARAMETER_CHUNK):
                placeholders = ",".join("?" for _ in chunk)
                for row in connection.execute(
                    f"SELECT item_id, doc_id FROM passages WHERE item_id IN ({placeholders})",
                    tuple(chunk),
                ):
                    output[str(row["item_id"])] = str(row["doc_id"])
        return output

    def row_indices_for_ids(self, level: Level, item_ids: Iterable[str]) -> dict[str, int]:
        table = self._table(level)
        values = list(dict.fromkeys(map(str, item_ids)))
        output: dict[str, int] = {}
        with self._connect() as connection:
            for chunk in _chunks(values, _SQLITE_PARAMETER_CHUNK):
                placeholders = ",".join("?" for _ in chunk)
                for row in connection.execute(
                    f"SELECT rowid, item_id FROM {table} WHERE item_id IN ({placeholders})",
                    tuple(chunk),
                ):
                    output[str(row["item_id"])] = int(row["rowid"]) - 1
        return output

    def ids_for_row_indices(self, level: Level, indices: Iterable[int]) -> dict[int, str]:
        table = self._table(level)
        values = list(dict.fromkeys(int(index) + 1 for index in indices))
        output: dict[int, str] = {}
        with self._connect() as connection:
            for chunk in _chunks(values, _SQLITE_PARAMETER_CHUNK):
                placeholders = ",".join("?" for _ in chunk)
                for row in connection.execute(
                    f"SELECT rowid, item_id FROM {table} WHERE rowid IN ({placeholders})",
                    tuple(chunk),
                ):
                    output[int(row["rowid"]) - 1] = str(row["item_id"])
        return output

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
        return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)

    def sparse_search(
        self,
        level: Level,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []
        table = self._table(level)
        fts_table = f"{table}_fts"
        match_query = self._fts_query(query)
        if not match_query:
            return []
        candidate_rows: list[int] | None = None
        if candidate_ids is not None:
            mapping = self.row_indices_for_ids(level, candidate_ids)
            candidate_rows = [index + 1 for index in mapping.values()]
            if not candidate_rows:
                return []
        row_chunks: list[Sequence[int] | None] = (
            list(_chunks(candidate_rows, _SQLITE_PARAMETER_CHUNK))
            if candidate_rows is not None
            else [None]
        )
        candidates: dict[str, float] = {}
        with self._connect() as connection:
            for chunk in row_chunks:
                where = f"{fts_table} MATCH ?"
                parameters: list[Any] = [match_query]
                if chunk is not None:
                    placeholders = ",".join("?" for _ in chunk)
                    where += f" AND {fts_table}.rowid IN ({placeholders})"
                    parameters.extend(chunk)
                parameters.append(k)
                sql = (
                    f"SELECT {fts_table}.item_id AS item_id, -bm25({fts_table}) AS score "
                    f"FROM {fts_table} WHERE {where} "
                    f"ORDER BY bm25({fts_table}), {fts_table}.rowid LIMIT ?"
                )
                for row in connection.execute(sql, tuple(parameters)):
                    item_id = str(row["item_id"])
                    candidates[item_id] = max(float(row["score"]), candidates.get(item_id, -np.inf))
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            SearchResult(item_id, score, rank) for rank, (item_id, score) in enumerate(ranked, 1)
        ]

    def validate(self) -> None:
        errors: list[str] = []
        metadata = self._metadata()
        if metadata.get("finalized") != "1":
            errors.append("dataset store is not finalized")
        with self._connect() as connection:
            for table in ("documents", "passages", "queries"):
                if connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0:
                    errors.append(f"dataset contains no {table}")
            missing_docs = connection.execute(
                """SELECT count(*) FROM passages p
                LEFT JOIN documents d ON d.item_id = p.doc_id WHERE d.item_id IS NULL"""
            ).fetchone()[0]
            if missing_docs:
                errors.append(f"{missing_docs} passages reference missing documents")
            missing_queries = connection.execute(
                """SELECT count(*) FROM qrels r
                LEFT JOIN queries q ON q.query_id = r.query_id WHERE q.query_id IS NULL"""
            ).fetchone()[0]
            if missing_queries:
                errors.append(f"{missing_queries} qrels reference missing queries")
            missing_passages = connection.execute(
                """SELECT count(*) FROM qrels r
                LEFT JOIN passages p ON p.item_id = r.passage_id WHERE p.item_id IS NULL"""
            ).fetchone()[0]
            if missing_passages:
                errors.append(f"{missing_passages} qrels reference missing passages")
        if errors:
            raise ValueError("Invalid disk dataset:\n- " + "\n- ".join(errors))

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = {
                table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in ("documents", "passages", "queries", "qrels")
            }
            graded = int(
                connection.execute("SELECT count(*) FROM qrels WHERE score > 1").fetchone()[0]
            )
            relevant_queries = int(
                connection.execute(
                    "SELECT count(DISTINCT query_id) FROM qrels WHERE score > 0"
                ).fetchone()[0]
            )
        return {
            "name": self.name,
            **counts,
            "graded_qrels": graded,
            "queries_with_relevant_passages": relevant_queries,
            "provenance": self.provenance,
        }


class MemmapDenseIndex:
    """Batched encoder with exact cosine search over a memory-mapped matrix."""

    def __init__(
        self,
        store: DiskDatasetStore,
        level: Level,
        index_dir: str | Path,
        *,
        model_name: str = "intfloat/e5-small-v2",
        model_revision: str | None = None,
        encoder: Any | None = None,
        batch_size: int = 64,
        search_chunk_size: int = 50_000,
        query_prefix: str = "query: ",
        corpus_prefix: str = "passage: ",
    ) -> None:
        self.store = store
        self.level = level
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.model_revision = model_revision
        self.encoder = encoder
        self.batch_size = batch_size
        self.search_chunk_size = search_chunk_size
        self.query_prefix = query_prefix
        self.corpus_prefix = corpus_prefix
        self.embedding_path = self.index_dir / f"{level}_embeddings.npy"
        self.metadata_path = self.index_dir / f"{level}_embeddings.json"
        self._embeddings: np.ndarray | None = None

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name, revision=self.model_revision)
        return self.encoder

    @staticmethod
    def _embedding_dimension(encoder: Any) -> int:
        current = getattr(encoder, "get_embedding_dimension", None)
        if callable(current):
            return int(current())
        legacy = getattr(encoder, "get_sentence_embedding_dimension", None)
        if callable(legacy):
            return int(legacy())
        raise TypeError(
            "Dense encoder must provide get_embedding_dimension() or "
            "get_sentence_embedding_dimension()."
        )

    def _expected_metadata(self, dimensions: int) -> dict[str, Any]:
        return {
            "dataset_id": self.store.dataset_id,
            "level": self.level,
            "count": self.store.item_count(self.level),
            "dimensions": dimensions,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "query_prefix": self.query_prefix,
            "corpus_prefix": self.corpus_prefix,
            "dtype": "float32",
        }

    def build(self) -> MemmapDenseIndex:
        encoder = self._get_encoder()
        dimensions = self._embedding_dimension(encoder)
        expected = self._expected_metadata(dimensions)
        if self.metadata_path.is_file() and self.embedding_path.is_file():
            current = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if current == expected:
                matrix = np.load(self.embedding_path, mmap_mode="r", allow_pickle=False)
                if matrix.shape == (expected["count"], dimensions):
                    self._embeddings = matrix
                    return self

        self.index_dir.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            self.embedding_path,
            mode="w+",
            dtype=np.float32,
            shape=(expected["count"], dimensions),
        )
        batch_rows: list[int] = []
        batch_texts: list[str] = []

        def write_batch() -> None:
            if not batch_rows:
                return
            encoded = np.asarray(
                encoder.encode(
                    [self.corpus_prefix + text for text in batch_texts],
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
            matrix[np.asarray(batch_rows, dtype=np.int64)] = encoded
            batch_rows.clear()
            batch_texts.clear()

        for row_index, _, text in self.store.iter_items(self.level):
            batch_rows.append(row_index)
            batch_texts.append(text)
            if len(batch_rows) >= self.batch_size:
                write_batch()
        write_batch()
        matrix.flush()
        temporary_metadata = self.metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary_metadata.replace(self.metadata_path)
        self._embeddings = np.load(self.embedding_path, mmap_mode="r", allow_pickle=False)
        return self

    def _matrix(self) -> np.ndarray:
        if self._embeddings is None:
            raise RuntimeError("Call build() before search().")
        return self._embeddings

    def _encode_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self._get_encoder().encode(
                [self.query_prefix + query],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0],
            dtype=np.float32,
        )

    @staticmethod
    def _top_k(indices: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not len(indices):
            return indices, scores
        selected_k = min(k, len(indices))
        selected = np.argpartition(-scores, selected_k - 1)[:selected_k]
        order = selected[np.argsort(-scores[selected], kind="stable")]
        return indices[order], scores[order]

    def search(
        self,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []
        matrix = self._matrix()
        query_vector = self._encode_query(query)
        best_indices = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        if candidate_ids is None:
            sources = (
                np.arange(start, min(start + self.search_chunk_size, len(matrix)), dtype=np.int64)
                for start in range(0, len(matrix), self.search_chunk_size)
            )
        else:
            mapping = self.store.row_indices_for_ids(self.level, candidate_ids)
            sources = [np.asarray(list(mapping.values()), dtype=np.int64)]
        for indices in sources:
            if not len(indices):
                continue
            scores = np.asarray(matrix[indices] @ query_vector, dtype=np.float32)
            local_indices, local_scores = self._top_k(indices, scores, k)
            best_indices, best_scores = self._top_k(
                np.concatenate((best_indices, local_indices)),
                np.concatenate((best_scores, local_scores)),
                k,
            )
        id_map = self.store.ids_for_row_indices(self.level, best_indices.tolist())
        return [
            SearchResult(id_map[int(index)], float(score), rank)
            for rank, (index, score) in enumerate(zip(best_indices, best_scores, strict=True), 1)
        ]

    def search_many(
        self,
        queries: Sequence[Query],
        k: int,
        *,
        query_batch_size: int = 64,
    ) -> dict[str, list[SearchResult]]:
        """Exact global search while bounding score-matrix memory."""
        if not queries or k <= 0:
            return {query.query_id: [] for query in queries}
        matrix = self._matrix()
        query_vectors = np.asarray(
            self._get_encoder().encode(
                [self.query_prefix + query.text for query in queries],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        selected_k = min(k, len(matrix))
        best_indices = np.full((len(queries), selected_k), -1, dtype=np.int64)
        best_scores = np.full((len(queries), selected_k), -np.inf, dtype=np.float32)
        for corpus_start in range(0, len(matrix), self.search_chunk_size):
            corpus_stop = min(corpus_start + self.search_chunk_size, len(matrix))
            corpus = np.asarray(matrix[corpus_start:corpus_stop], dtype=np.float32)
            corpus_indices = np.arange(corpus_start, corpus_stop, dtype=np.int64)
            for query_start in range(0, len(queries), query_batch_size):
                query_stop = min(query_start + query_batch_size, len(queries))
                scores = corpus @ query_vectors[query_start:query_stop].T
                for local_query, query_index in enumerate(range(query_start, query_stop)):
                    local_indices, local_scores = self._top_k(
                        corpus_indices, scores[:, local_query], selected_k
                    )
                    merged_indices, merged_scores = self._top_k(
                        np.concatenate((best_indices[query_index], local_indices)),
                        np.concatenate((best_scores[query_index], local_scores)),
                        selected_k,
                    )
                    best_indices[query_index] = merged_indices
                    best_scores[query_index] = merged_scores
        all_indices = best_indices[best_indices >= 0].tolist()
        id_map = self.store.ids_for_row_indices(self.level, all_indices)
        return {
            query.query_id: [
                SearchResult(id_map[int(index)], float(score), rank)
                for rank, (index, score) in enumerate(
                    zip(best_indices[position], best_scores[position], strict=True), 1
                )
                if int(index) >= 0
            ]
            for position, query in enumerate(queries)
        }


class StoreSparseIndex:
    """Thin retrieval adapter over a store's persistent FTS5 index."""

    def __init__(self, store: DiskDatasetStore, level: Level) -> None:
        self.store = store
        self.level = level

    def search(
        self,
        query: str,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        return self.store.sparse_search(self.level, query, k, candidate_ids)


@dataclass
class ScalablePhase1Report:
    run_id: str
    config: BenchmarkConfig
    dataset_summary: dict[str, Any]
    leaderboard: pd.DataFrame
    artifact_root: Path
    archive_path: Path
    grouped_metrics: dict[str, dict[str, float]]
    experiment_results: dict[str, dict[str, Any]]

    @property
    def best_experiment(self) -> str:
        if self.leaderboard.empty:
            raise RuntimeError("No completed experiment is available.")
        return str(self.leaderboard.iloc[0]["experiment"])


def _fuse(
    sparse: list[SearchResult],
    dense: list[SearchResult],
    *,
    method: str,
    k: int,
    rrf_k: int,
) -> list[SearchResult]:
    if method == "interleave":
        return hhr_interleave(sparse, dense, k=k)
    if method == "rrf":
        return reciprocal_rank_fusion(sparse, dense, k=k, rrf_k=rrf_k)
    raise ValueError(f"Unknown fusion method: {method}")


def _resolved_scalable_config(
    config: BenchmarkConfig | None,
    *,
    run_mode: str,
    fusion: str,
    dense_model: str | None,
    dense_model_revision: str | None,
    document_top_k: int | None,
    passage_top_k: int | None,
) -> BenchmarkConfig:
    base = config or BenchmarkConfig()
    overrides: dict[str, Any] = {"fusion": fusion, "dense_backend": "sentence_transformers"}
    for key, value in (
        ("dense_model", dense_model),
        ("dense_model_revision", dense_model_revision),
        ("document_top_k", document_top_k),
        ("passage_top_k", passage_top_k),
    ):
        if value is not None:
            overrides[key] = value
    resolved = replace(
        base,
        retrieval=replace(base.retrieval, **overrides),
        run=replace(base.run, mode=run_mode),
    )
    validate_config(resolved)
    return resolved


def run_scalable_phase1_benchmark(
    store: DiskDatasetStore,
    run_mode: str = "baseline",
    *,
    fusion: str = "rrf",
    dense_model: str | None = None,
    dense_model_revision: str | None = None,
    document_top_k: int | None = None,
    passage_top_k: int | None = None,
    experiment_names: Sequence[str] | None = None,
    index_dir: str | Path,
    output_dir: str | Path,
    config: BenchmarkConfig | None = None,
    query_metadata: dict[str, dict[str, Any]] | None = None,
    encoder: Any | None = None,
    verbose: bool = True,
) -> ScalablePhase1Report:
    """Run all selected HHR combinations using persistent, reusable indexes."""
    store.validate()
    resolved = _resolved_scalable_config(
        config,
        run_mode=run_mode,
        fusion=fusion,
        dense_model=dense_model,
        dense_model_revision=dense_model_revision,
        document_top_k=document_top_k,
        passage_top_k=passage_top_k,
    )
    registry = build_experiment_registry()
    defaults = {
        "smoke": resolved.run.smoke_experiments,
        "baseline": resolved.run.baseline_experiments,
        "full": tuple(registry),
    }
    selected_names = list(experiment_names or defaults[run_mode])
    unknown = sorted(set(selected_names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown experiment names: {unknown}")
    selected = [registry[name] for name in selected_names]
    queries = store.queries()
    qrels = store.qrels()
    summary = store.summary()
    if verbose:
        print(f"Disk dataset summary: {summary}")

    index_path = Path(index_dir)
    output_path = Path(output_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    need_dense = any(
        method in {"dense", "combined"}
        for experiment in selected
        for method in (experiment.document_method, experiment.passage_method)
    )
    document_dense = passage_dense = None
    if need_dense:
        shared = {
            "model_name": resolved.retrieval.dense_model,
            "model_revision": resolved.retrieval.dense_model_revision,
            "encoder": encoder,
            "batch_size": resolved.retrieval.dense_batch_size,
            "query_prefix": resolved.retrieval.dense_query_prefix,
            "corpus_prefix": resolved.retrieval.dense_corpus_prefix,
        }
        document_dense = MemmapDenseIndex(store, "document", index_path, **shared).build()
        passage_dense = MemmapDenseIndex(store, "passage", index_path, **shared).build()
    document_sparse = StoreSparseIndex(store, "document")
    passage_sparse = StoreSparseIndex(store, "passage")

    required_document_methods = {experiment.document_method for experiment in selected}
    document_results: dict[str, dict[str, list[SearchResult]]] = {}
    document_latency: dict[str, float] = {}
    if "sparse" in required_document_methods or "combined" in required_document_methods:
        start = perf_counter()
        document_results["sparse"] = {
            query.query_id: document_sparse.search(query.text, resolved.retrieval.document_top_k)
            for query in queries
        }
        document_latency["sparse"] = (perf_counter() - start) * 1000 / max(1, len(queries))
    if "dense" in required_document_methods or "combined" in required_document_methods:
        assert document_dense is not None
        start = perf_counter()
        document_results["dense"] = document_dense.search_many(
            queries, resolved.retrieval.document_top_k
        )
        document_latency["dense"] = (perf_counter() - start) * 1000 / max(1, len(queries))
    if "combined" in required_document_methods:
        document_results["combined"] = {
            query.query_id: _fuse(
                document_results["sparse"][query.query_id],
                document_results["dense"][query.query_id],
                method=resolved.retrieval.fusion,
                k=resolved.retrieval.document_top_k,
                rrf_k=resolved.retrieval.rrf_k,
            )
            for query in queries
        }
        document_latency["combined"] = document_latency["sparse"] + document_latency["dense"]

    candidates = {
        method: {
            query.query_id: store.passage_ids_for_documents(
                result.item_id for result in document_results[method][query.query_id]
            )
            for query in queries
        }
        for method in required_document_methods
    }
    qrels_by_query: dict[str, dict[str, float]] = {}
    for qrel in qrels:
        qrels_by_query.setdefault(qrel.query_id, {})[qrel.passage_id] = qrel.score
    passage_to_doc = store.passage_to_document(qrel.passage_id for qrel in qrels)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_root = output_path / run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    save_json(artifact_root / "config.json", asdict(resolved))
    save_json(artifact_root / "dataset_summary.json", summary)
    rows: list[dict[str, Any]] = []
    experiment_results: dict[str, dict[str, Any]] = {}
    for experiment in selected:
        if verbose:
            print(f"Running scalable {experiment.name} ...")
        runs: dict[str, QueryRun] = {}
        for query in queries:
            candidate_ids = candidates[experiment.document_method][query.query_id]
            start = perf_counter()
            if experiment.passage_method == "sparse":
                passage_results = passage_sparse.search(
                    query.text, resolved.retrieval.passage_top_k, candidate_ids
                )
            elif experiment.passage_method == "dense":
                assert passage_dense is not None
                passage_results = passage_dense.search(
                    query.text, resolved.retrieval.passage_top_k, candidate_ids
                )
            else:
                assert passage_dense is not None
                sparse_results = passage_sparse.search(
                    query.text, resolved.retrieval.passage_top_k, candidate_ids
                )
                dense_results = passage_dense.search(
                    query.text, resolved.retrieval.passage_top_k, candidate_ids
                )
                passage_results = _fuse(
                    sparse_results,
                    dense_results,
                    method=resolved.retrieval.fusion,
                    k=resolved.retrieval.passage_top_k,
                    rrf_k=resolved.retrieval.rrf_k,
                )
            latency = document_latency[experiment.document_method] + (perf_counter() - start) * 1000
            runs[query.query_id] = QueryRun(
                query.query_id,
                document_results[experiment.document_method][query.query_id],
                passage_results,
                latency,
            )
        per_query = {
            query_id: evaluate_query(
                run,
                qrels_by_query.get(query_id, {}),
                passage_to_doc,
                resolved.evaluation.ndcg_k,
                resolved.evaluation.recall_k,
                resolved.retrieval.document_top_k,
            )
            for query_id, run in runs.items()
        }
        metric_names = next(iter(per_query.values())).keys() if per_query else []
        metrics = {
            metric: mean(row[metric] for row in per_query.values()) for metric in metric_names
        }
        metrics["evaluated_queries"] = float(len(per_query))
        result = {
            "experiment": asdict(experiment),
            "dataset": store.name,
            "metrics": metrics,
            "per_query": per_query,
            "runs": runs,
        }
        artifact_dir = save_run_artifacts(artifact_root, experiment.name, result)
        rows.append({"experiment": experiment.name, **metrics, "artifact_dir": str(artifact_dir)})
        experiment_results[experiment.name] = {
            "experiment": result["experiment"],
            "metrics": metrics,
            "per_query": per_query,
            "artifact_dir": artifact_dir,
        }

    score_column = f"passage_ndcg@{resolved.evaluation.ndcg_k}"
    leaderboard = (
        pd.DataFrame(rows).sort_values(score_column, ascending=False).reset_index(drop=True)
    )
    leaderboard.to_csv(artifact_root / "leaderboard.csv", index=False)
    grouped_metrics: dict[str, dict[str, float]] = {}
    if query_metadata:
        best_name = str(leaderboard.iloc[0]["experiment"])
        grouped_metrics = evaluate_by_group(
            experiment_results[best_name]["per_query"], query_metadata
        )
        save_json(artifact_root / "grouped_metrics.json", grouped_metrics)
    import shutil

    archive_path = Path(
        shutil.make_archive(str(output_path / f"{run_id}_artifacts"), "zip", root_dir=artifact_root)
    )
    return ScalablePhase1Report(
        run_id,
        resolved,
        summary,
        leaderboard,
        artifact_root,
        archive_path,
        grouped_metrics,
        experiment_results,
    )
