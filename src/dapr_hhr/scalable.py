"""Disk-backed dataset and exact dense retrieval primitives for large corpora."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import log
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
_CACHE_FORMAT_VERSION = 2


def _chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _sparse_tokens(text: str) -> list[str]:
    """Approximate SQLite unicode61 tokenization without loading a global posting list."""
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[^\W_]+", without_marks, flags=re.UNICODE)


def _candidate_bm25(
    items: Sequence[tuple[str, str]], query: str, k: int
) -> list[SearchResult]:
    """Rank a query's small candidate set with BM25 in bounded memory."""
    query_terms = tuple(dict.fromkeys(_sparse_tokens(query)))
    if not query_terms or not items or k <= 0:
        return []
    term_set = set(query_terms)
    tokenized: list[tuple[str, Counter[str], int]] = []
    document_frequency: Counter[str] = Counter()
    for item_id, text in items:
        tokens = _sparse_tokens(text)
        frequencies = Counter(token for token in tokens if token in term_set)
        document_frequency.update(frequencies.keys())
        tokenized.append((item_id, frequencies, len(tokens)))
    count = len(tokenized)
    average_length = sum(length for _, _, length in tokenized) / max(count, 1)
    k1 = 1.2
    b = 0.75
    scored: list[tuple[str, float]] = []
    for item_id, frequencies, length in tokenized:
        score = 0.0
        length_factor = k1 * (1.0 - b + b * length / max(average_length, 1e-9))
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            frequency_docs = document_frequency[term]
            inverse_frequency = log(
                1.0 + (count - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            score += inverse_frequency * frequency * (k1 + 1.0) / (
                frequency + length_factor
            )
        if score > 0.0:
            scored.append((item_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        SearchResult(item_id, score, rank)
        for rank, (item_id, score) in enumerate(scored[:k], 1)
    ]


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
                "document_fts_ready": "0",
                "passage_fts_ready": "0",
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

    def finalize(self, *, build_passage_fts: bool = False) -> None:
        self._require_mutable()
        with self._connect() as connection:
            connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
            connection.execute(
                "UPDATE metadata SET value = '1' WHERE key = 'document_fts_ready'"
            )
            if build_passage_fts:
                connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
                connection.execute(
                    "UPDATE metadata SET value = '1' WHERE key = 'passage_fts_ready'"
                )
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

    def items_for_ids(self, level: Level, item_ids: Iterable[str]) -> list[tuple[str, str]]:
        """Return selected item IDs and search text in stable corpus order."""
        table = self._table(level)
        values = list(dict.fromkeys(map(str, item_ids)))
        rows: list[tuple[int, str, str]] = []
        with self._connect() as connection:
            for chunk in _chunks(values, _SQLITE_PARAMETER_CHUNK):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    (int(row["rowid"]), str(row["item_id"]), str(row["search_text"]))
                    for row in connection.execute(
                        f"SELECT rowid, item_id, search_text FROM {table} "
                        f"WHERE item_id IN ({placeholders})",
                        tuple(chunk),
                    )
                )
        rows.sort(key=lambda row: row[0])
        return [(item_id, text) for _, item_id, text in rows]

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
        ready_key = f"{level}_fts_ready"
        metadata = self._metadata()
        ready = metadata.get(ready_key)
        if ready is None and metadata.get("finalized") == "1":
            ready = "1"  # Legacy stores built both FTS indexes during finalize().
        if ready != "1":
            raise RuntimeError(
                f"The global {level} FTS5 index is not available. "
                "Use a candidate sparse index for passage retrieval."
            )
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


class DenseEncoderRuntime:
    """One reusable Sentence Transformers runtime and optional persistent GPU pool."""

    def __init__(
        self,
        *,
        model_name: str = "intfloat/e5-small-v2",
        model_revision: str | None = None,
        encoder: Any | None = None,
        enable_multi_gpu: bool = True,
        preferred_devices: Sequence[str] = ("cuda:0", "cuda:1"),
        available_devices: Sequence[str] | None = None,
        batch_size: int = 64,
        multi_process_chunk_size: int = 5_000,
        verbose: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.encoder = encoder
        self.enable_multi_gpu = enable_multi_gpu
        self.preferred_devices = tuple(preferred_devices)
        self._available_devices = (
            tuple(available_devices) if available_devices is not None else None
        )
        self.batch_size = batch_size
        self.multi_process_chunk_size = multi_process_chunk_size
        self.verbose = verbose
        self.pool: Any | None = None
        self.pool_start_count = 0
        self.devices: tuple[str, ...] = ()
        self._entered = False

    @staticmethod
    def detect_available_devices() -> tuple[str, ...]:
        try:
            import torch
        except ImportError:
            return ("cpu",)
        if torch.cuda.is_available():
            return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))
        return ("cpu",)

    @staticmethod
    def embedding_dimension(encoder: Any) -> int:
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

    def __enter__(self) -> DenseEncoderRuntime:
        if self._entered:
            return self
        available = self._available_devices or self.detect_available_devices()
        selected = tuple(device for device in self.preferred_devices if device in available)
        self.devices = selected or (available[0],)
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"revision": self.model_revision}
            if len(self.devices) == 1:
                kwargs["device"] = self.devices[0]
            self.encoder = SentenceTransformer(self.model_name, **kwargs)
        if (
            self.enable_multi_gpu
            and len(self.devices) > 1
            and callable(getattr(self.encoder, "start_multi_process_pool", None))
        ):
            self.pool = self.encoder.start_multi_process_pool(target_devices=list(self.devices))
            self.pool_start_count += 1
        self._entered = True
        if self.verbose:
            print(f"Device configuration: {list(self.devices)}")
            print(f"Multi-GPU encoding: {'enabled' if self.pool is not None else 'disabled'}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        if self.pool is not None:
            self.encoder.stop_multi_process_pool(self.pool)
            self.pool = None
        self._entered = False

    @property
    def dimension(self) -> int:
        if not self._entered:
            self.__enter__()
        return self.embedding_dimension(self.encoder)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._entered:
            self.__enter__()
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        if self.pool is not None:
            kwargs["pool"] = self.pool
            kwargs["chunk_size"] = self.multi_process_chunk_size
        return np.asarray(self.encoder.encode(list(texts), **kwargs), dtype=np.float32)


class MemmapDenseIndex:
    """Resumable exact cosine index over a full corpus or a selected candidate set."""

    def __init__(
        self,
        store: DiskDatasetStore,
        level: Level,
        index_dir: str | Path,
        *,
        model_name: str = "intfloat/e5-small-v2",
        model_revision: str | None = None,
        encoder: Any | None = None,
        runtime: DenseEncoderRuntime | None = None,
        batch_size: int = 64,
        search_chunk_size: int = 50_000,
        query_prefix: str = "query: ",
        corpus_prefix: str = "passage: ",
        selected_ids: Sequence[str] | None = None,
        cache_name: str | None = None,
        cache_context: dict[str, Any] | None = None,
        embedding_write_chunk_size: int = 50_000,
        checkpoint_rows: int = 50_000,
        enable_resume: bool = True,
        dtype: str = "float32",
        progress_interval: int = 50_000,
        verbose: bool = False,
    ) -> None:
        self.store = store
        self.level = level
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.model_revision = model_revision
        self.encoder = encoder
        self.runtime = runtime
        self.batch_size = batch_size
        self.search_chunk_size = search_chunk_size
        self.query_prefix = query_prefix
        self.corpus_prefix = corpus_prefix
        self.selected_ids = (
            tuple(dict.fromkeys(map(str, selected_ids))) if selected_ids is not None else None
        )
        self.cache_name = cache_name or level
        self.cache_context = dict(cache_context or {})
        self.embedding_write_chunk_size = embedding_write_chunk_size
        self.checkpoint_rows = checkpoint_rows
        self.enable_resume = enable_resume
        self.dtype = dtype
        self.progress_interval = progress_interval
        self.verbose = verbose
        self.embedding_path = self.index_dir / f"{self.cache_name}_embeddings.npy"
        self.metadata_path = self.index_dir / f"{self.cache_name}_embeddings.json"
        self.progress_path = self.index_dir / f"{self.cache_name}_embeddings.progress.json"
        self.ids_path = self.index_dir / f"{self.cache_name}_item_ids.txt"
        self._embeddings: np.ndarray | None = None
        self._item_ids: tuple[str, ...] | None = None
        self._id_to_index: dict[str, int] | None = None
        self.build_status = "not_built"
        self.search_backend = "numpy_cpu"

    def _get_runtime(self) -> DenseEncoderRuntime:
        if self.runtime is None:
            self.runtime = DenseEncoderRuntime(
                model_name=self.model_name,
                model_revision=self.model_revision,
                encoder=self.encoder,
                enable_multi_gpu=False,
                batch_size=self.batch_size,
                verbose=False,
            )
        self.runtime.__enter__()
        return self.runtime

    @staticmethod
    def _embedding_dimension(encoder: Any) -> int:
        return DenseEncoderRuntime.embedding_dimension(encoder)

    def _items(self) -> list[tuple[str, str]] | None:
        if self.selected_ids is None:
            return None
        items = self.store.items_for_ids(self.level, self.selected_ids)
        if len(items) != len(self.selected_ids):
            found = {item_id for item_id, _ in items}
            missing = sorted(set(self.selected_ids) - found)
            raise ValueError(f"Candidate index references missing {self.level} IDs: {missing[:5]}")
        return items

    def _expected_metadata(
        self, dimensions: int, count: int, item_ids_fingerprint: str
    ) -> dict[str, Any]:
        provenance = self.store.provenance
        corpus_identity = provenance.get("corpus_key") or self.store.dataset_id
        return {
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "corpus_identity": corpus_identity,
            "dataset_provenance": provenance,
            "level": self.level,
            "count": count,
            "item_ids_fingerprint": item_ids_fingerprint,
            "dimensions": dimensions,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "query_prefix": self.query_prefix,
            "corpus_prefix": self.corpus_prefix,
            "text_strategy": "title_body" if self.level == "document" else "title_text",
            "normalize_embeddings": True,
            "dtype": self.dtype,
            "cache_context": self.cache_context,
        }

    @staticmethod
    def _metadata_matches(current: dict[str, Any], expected: dict[str, Any]) -> bool:
        return all(current.get(key) == value for key, value in expected.items())

    def _load_item_ids(self) -> tuple[str, ...]:
        return tuple(self.ids_path.read_text(encoding="utf-8").splitlines())

    def build(self) -> MemmapDenseIndex:
        runtime = self._get_runtime()
        dimensions = runtime.dimension
        selected_items = self._items()
        if selected_items is None:
            count = self.store.item_count(self.level)
            item_ids_fingerprint = _fingerprint(
                item_id for _, item_id, _ in self.store.iter_items(self.level)
            )
            item_ids: tuple[str, ...] | None = None
        else:
            item_ids = tuple(item_id for item_id, _ in selected_items)
            count = len(selected_items)
            item_ids_fingerprint = _fingerprint(item_ids)
        expected = self._expected_metadata(dimensions, count, item_ids_fingerprint)

        complete_files_exist = (
            self.metadata_path.is_file()
            and self.embedding_path.is_file()
            and self.ids_path.is_file()
        )
        if complete_files_exist:
            current = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self._metadata_matches(current, expected) and current.get("state") == "complete":
                matrix = np.load(self.embedding_path, mmap_mode="r", allow_pickle=False)
                if matrix.shape == (count, dimensions):
                    self._embeddings = matrix
                    if self.selected_ids is not None:
                        self._item_ids = self._load_item_ids()
                        self._id_to_index = {
                            item_id: index for index, item_id in enumerate(self._item_ids)
                        }
                    if self.verbose:
                        print(f"Reusing complete {self.cache_name} index ({count:,} rows).")
                    self.build_status = "reused"
                    return self

        completed = 0
        if self.enable_resume and self.progress_path.is_file():
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if not self._metadata_matches(progress, expected):
                raise ValueError(
                    f"Incompatible partial embedding checkpoint: {self.progress_path}"
                )
            completed = int(progress.get("completed_count", 0))
            if not self.embedding_path.is_file() or not self.ids_path.is_file():
                raise ValueError("Embedding checkpoint is missing its matrix or ordered ID file.")
            if self.verbose:
                print(f"Resuming {self.cache_name} at row {completed:,} / {count:,}.")
            self.build_status = "resumed"
            matrix = np.lib.format.open_memmap(
                self.embedding_path, mode="r+", dtype=self.dtype, shape=(count, dimensions)
            )
        else:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            matrix = np.lib.format.open_memmap(
                self.embedding_path, mode="w+", dtype=self.dtype, shape=(count, dimensions)
            )
            temporary_ids = self.ids_path.with_suffix(self.ids_path.suffix + ".tmp")
            with temporary_ids.open("w", encoding="utf-8") as handle:
                if item_ids is None:
                    for _, item_id, _ in self.store.iter_items(self.level):
                        handle.write(item_id + "\n")
                else:
                    for item_id in item_ids:
                        handle.write(item_id + "\n")
            temporary_ids.replace(self.ids_path)
            _atomic_write_json(
                self.progress_path,
                {**expected, "state": "partial", "completed_count": 0},
            )
            if self.verbose:
                print(f"Creating {self.cache_name} index ({count:,} rows).")
            self.build_status = "created"

        chunk_size = min(self.embedding_write_chunk_size, self.checkpoint_rows)
        started = perf_counter()
        initial_completed = completed
        last_progress = completed
        if selected_items is None:
            source = (
                (item_id, text)
                for row_index, item_id, text in self.store.iter_items(self.level)
                if row_index >= completed
            )
        else:
            source = iter(selected_items[completed:])
        pending: list[tuple[str, str]] = []
        for item in source:
            pending.append(item)
            if len(pending) < chunk_size:
                continue
            completed = self._write_chunk(matrix, completed, pending, runtime)
            _atomic_write_json(
                self.progress_path,
                {**expected, "state": "partial", "completed_count": completed},
            )
            if self.verbose and completed - last_progress >= self.progress_interval:
                self._print_progress(
                    completed, count, started, runtime.devices, initial_completed
                )
                last_progress = completed
            pending.clear()
        if pending:
            completed = self._write_chunk(matrix, completed, pending, runtime)
            _atomic_write_json(
                self.progress_path,
                {**expected, "state": "partial", "completed_count": completed},
            )
        matrix.flush()
        if completed != count:
            raise RuntimeError(f"Encoded {completed:,} rows but expected {count:,}.")
        _atomic_write_json(
            self.metadata_path,
            {**expected, "state": "complete", "completed_count": completed},
        )
        self.progress_path.unlink(missing_ok=True)
        self._embeddings = np.load(self.embedding_path, mmap_mode="r", allow_pickle=False)
        if item_ids is not None:
            self._item_ids = item_ids
            self._id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
        if self.verbose:
            self._print_progress(completed, count, started, runtime.devices, initial_completed)
        return self

    def _write_chunk(
        self,
        matrix: np.ndarray,
        completed: int,
        pending: Sequence[tuple[str, str]],
        runtime: DenseEncoderRuntime,
    ) -> int:
        texts = [self.corpus_prefix + text for _, text in pending]
        encoded = runtime.encode(texts).astype(self.dtype, copy=False)
        stop = completed + len(pending)
        if encoded.shape != (len(pending), matrix.shape[1]):
            raise ValueError(
                f"Encoder returned {encoded.shape}; expected {(len(pending), matrix.shape[1])}."
            )
        matrix[completed:stop] = encoded
        matrix.flush()
        return stop

    def _print_progress(
        self,
        completed: int,
        count: int,
        started: float,
        devices: Sequence[str],
        initial_completed: int,
    ) -> None:
        elapsed = max(perf_counter() - started, 1e-9)
        processed = max(completed - initial_completed, 1)
        speed = processed / elapsed
        eta = (count - completed) / max(speed, 1e-9)
        print(
            f"{self.cache_name}: {completed:,} / {count:,} | "
            f"{speed:,.1f} rows/s | elapsed {_format_seconds(elapsed)} | "
            f"ETA {_format_seconds(eta)} | devices {list(devices)}"
        )

    def _matrix(self) -> np.ndarray:
        if self._embeddings is None:
            raise RuntimeError("Call build() before search().")
        return self._embeddings

    def _encode_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self._get_runtime().encode([self.query_prefix + query])[0], dtype=np.float32
        )

    def encode_queries(self, queries: Sequence[Query]) -> dict[str, np.ndarray]:
        vectors = np.asarray(
            self._get_runtime().encode(
                [self.query_prefix + query.text for query in queries]
            ),
            dtype=np.float32,
        )
        return {
            query.query_id: vectors[position]
            for position, query in enumerate(queries)
        }

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
        return self.search_vector(self._encode_query(query), k, candidate_ids)

    def search_vector(
        self,
        query_vector: np.ndarray,
        k: int,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        """Search with a pre-encoded query to avoid one GPU IPC call per query."""
        if k <= 0:
            return []
        matrix = self._matrix()
        query_vector = np.asarray(query_vector, dtype=np.float32)
        best_indices = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        if candidate_ids is None:
            sources = (
                np.arange(start, min(start + self.search_chunk_size, len(matrix)), dtype=np.int64)
                for start in range(0, len(matrix), self.search_chunk_size)
            )
        else:
            if self._id_to_index is None:
                selected_mapping = self.store.row_indices_for_ids(self.level, candidate_ids)
            else:
                selected_mapping = self._id_to_index
            sources = [
                np.asarray(
                    [
                        selected_mapping[item_id]
                        for item_id in candidate_ids
                        if item_id in selected_mapping
                    ],
                    dtype=np.int64,
                )
            ]
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
        if self._item_ids is None:
            id_map = self.store.ids_for_row_indices(self.level, best_indices.tolist())
        else:
            id_map = {int(index): self._item_ids[int(index)] for index in best_indices}
        return [
            SearchResult(id_map[int(index)], float(score), rank)
            for rank, (index, score) in enumerate(zip(best_indices, best_scores, strict=True), 1)
        ]

    def search_many(
        self,
        queries: Sequence[Query],
        k: int,
        *,
        query_batch_size: int = 256,
    ) -> dict[str, list[SearchResult]]:
        """Exact global search while bounding score-matrix memory."""
        if not queries or k <= 0:
            return {query.query_id: [] for query in queries}
        matrix = self._matrix()
        query_vectors = np.asarray(
            self._get_runtime().encode(
                [self.query_prefix + query.text for query in queries],
            ),
            dtype=np.float32,
        )
        selected_k = min(k, len(matrix))
        accelerated = self._search_many_torch(
            matrix, query_vectors, selected_k, query_batch_size
        )
        if accelerated is not None:
            best_indices, best_scores = accelerated
            return self._format_many_results(queries, best_indices, best_scores)
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
        self.search_backend = "numpy_cpu"
        return self._format_many_results(queries, best_indices, best_scores)

    def _search_many_torch(
        self,
        matrix: np.ndarray,
        query_vectors: np.ndarray,
        selected_k: int,
        query_batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Use one CUDA device for exact batched search when PyTorch CUDA is available."""
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        runtime_devices = self._get_runtime().devices
        device_name = next(
            (device for device in runtime_devices if device.startswith("cuda")), "cuda:0"
        )
        device = torch.device(device_name)
        if self.verbose:
            print(f"Dense global search: exact batched matrix search on {device_name}.")
        query_tensor = torch.as_tensor(query_vectors, dtype=torch.float32, device=device)
        best_scores = torch.full(
            (len(query_vectors), selected_k),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        best_indices = torch.full(
            (len(query_vectors), selected_k),
            -1,
            dtype=torch.int64,
            device=device,
        )
        with torch.inference_mode():
            for corpus_start in range(0, len(matrix), self.search_chunk_size):
                corpus_stop = min(corpus_start + self.search_chunk_size, len(matrix))
                corpus = torch.as_tensor(
                    np.asarray(matrix[corpus_start:corpus_stop], dtype=np.float32),
                    device=device,
                )
                local_k = min(selected_k, len(corpus))
                for query_start in range(0, len(query_vectors), query_batch_size):
                    query_stop = min(query_start + query_batch_size, len(query_vectors))
                    scores = corpus @ query_tensor[query_start:query_stop].T
                    local_scores, local_indices = torch.topk(
                        scores, local_k, dim=0, largest=True, sorted=True
                    )
                    local_scores = local_scores.T
                    local_indices = local_indices.T + corpus_start
                    merged_scores = torch.cat(
                        (best_scores[query_start:query_stop], local_scores), dim=1
                    )
                    merged_indices = torch.cat(
                        (best_indices[query_start:query_stop], local_indices), dim=1
                    )
                    top_scores, top_positions = torch.topk(
                        merged_scores, selected_k, dim=1, largest=True, sorted=True
                    )
                    best_scores[query_start:query_stop] = top_scores
                    best_indices[query_start:query_stop] = torch.gather(
                        merged_indices, 1, top_positions
                    )
        self.search_backend = f"torch_{device_name}"
        return best_indices.cpu().numpy(), best_scores.cpu().numpy()

    def _format_many_results(
        self,
        queries: Sequence[Query],
        best_indices: np.ndarray,
        best_scores: np.ndarray,
    ) -> dict[str, list[SearchResult]]:
        all_indices = best_indices[best_indices >= 0].tolist()
        if self._item_ids is None:
            id_map = self.store.ids_for_row_indices(self.level, all_indices)
        else:
            id_map = {int(index): self._item_ids[int(index)] for index in all_indices}
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


class CandidateSparseIndex:
    """Bounded BM25 reranker over each query's candidate passages."""

    def __init__(
        self,
        store: DiskDatasetStore,
        index_dir: str | Path,
        candidate_ids: Sequence[str],
        *,
        cache_name: str,
        cache_context: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.index_dir = Path(index_dir)
        self.candidate_ids = tuple(dict.fromkeys(map(str, candidate_ids)))
        self.cache_name = cache_name
        self.cache_context = dict(cache_context or {})
        self.database_path = self.index_dir / f"{cache_name}_sparse.sqlite"
        self.metadata_path = self.index_dir / f"{cache_name}_sparse.json"
        self.build_status = "not_built"

    def _expected_metadata(self) -> dict[str, Any]:
        return {
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "dataset_id": self.store.dataset_id,
            "level": "passage",
            "count": len(self.candidate_ids),
            "item_ids_fingerprint": _fingerprint(self.candidate_ids),
            "text_strategy": "title_text",
            "backend": "query_candidate_bm25",
            "ranking_scope": "per_query_candidate_set",
            "cache_context": self.cache_context,
        }

    def build(self) -> CandidateSparseIndex:
        expected = self._expected_metadata()
        if not self.database_path.exists() and self.metadata_path.is_file():
            current = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if current == expected:
                self.build_status = "reused"
                return self
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Remove the obsolete union-wide FTS database. It made common terms scan
        # postings across millions of passages before applying a tiny candidate filter.
        self.database_path.unlink(missing_ok=True)
        _atomic_write_json(self.metadata_path, expected)
        self.build_status = "created"
        return self

    def search(
        self, query: str, k: int, candidate_ids: Iterable[str] | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            return []
        selected = self.candidate_ids if candidate_ids is None else tuple(candidate_ids)
        return _candidate_bm25(self.store.items_for_ids("passage", selected), query, k)


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
    document_checkpoint_dir: str | Path | None = None,
    candidate_passage_cache_dir: str | Path | None = None,
    config: BenchmarkConfig | None = None,
    query_metadata: dict[str, dict[str, Any]] | None = None,
    encoder: Any | None = None,
    verbose: bool = True,
) -> ScalablePhase1Report:
    """Run document-first HHR with global documents and candidate-only passages."""
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
    document_index_path = Path(document_checkpoint_dir or index_path)
    candidate_index_path = Path(candidate_passage_cache_dir or index_path)
    output_path = Path(output_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    document_index_path.mkdir(parents=True, exist_ok=True)
    candidate_index_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    required_document_methods = {experiment.document_method for experiment in selected}
    required_passage_methods = {experiment.passage_method for experiment in selected}
    need_document_dense = bool(required_document_methods & {"dense", "combined"})
    need_passage_dense = bool(required_passage_methods & {"dense", "combined"})
    need_runtime = need_document_dense or need_passage_dense
    runtime = (
        DenseEncoderRuntime(
            model_name=resolved.retrieval.dense_model,
            model_revision=resolved.retrieval.dense_model_revision,
            encoder=encoder,
            enable_multi_gpu=resolved.retrieval.enable_multi_gpu,
            preferred_devices=resolved.retrieval.preferred_devices,
            batch_size=resolved.retrieval.dense_batch_size,
            multi_process_chunk_size=resolved.retrieval.multi_process_chunk_size,
            verbose=verbose,
        )
        if need_runtime
        else None
    )
    manager = runtime if runtime is not None else nullcontext(None)
    with manager as active_runtime:
        return _execute_scalable_benchmark(
            store=store,
            resolved=resolved,
            selected=selected,
            queries=queries,
            qrels=qrels,
            summary=summary,
            document_index_path=document_index_path,
            candidate_index_path=candidate_index_path,
            output_path=output_path,
            query_metadata=query_metadata,
            runtime=active_runtime,
            required_document_methods=required_document_methods,
            required_passage_methods=required_passage_methods,
            need_document_dense=need_document_dense,
            need_passage_dense=need_passage_dense,
            verbose=verbose,
        )


def _execute_scalable_benchmark(
    *,
    store: DiskDatasetStore,
    resolved: BenchmarkConfig,
    selected: Sequence[Any],
    queries: Sequence[Query],
    qrels: Sequence[Qrel],
    summary: dict[str, Any],
    document_index_path: Path,
    candidate_index_path: Path,
    output_path: Path,
    query_metadata: dict[str, dict[str, Any]] | None,
    runtime: DenseEncoderRuntime | None,
    required_document_methods: set[str],
    required_passage_methods: set[str],
    need_document_dense: bool,
    need_passage_dense: bool,
    verbose: bool,
) -> ScalablePhase1Report:
    """Execute one run while keeping the shared encoder pool alive."""
    retrieval = resolved.retrieval
    shared_dense: dict[str, Any] = {
        "model_name": retrieval.dense_model,
        "model_revision": retrieval.dense_model_revision,
        "runtime": runtime,
        "batch_size": retrieval.dense_batch_size,
        "search_chunk_size": retrieval.dense_search_chunk_size,
        "query_prefix": retrieval.dense_query_prefix,
        "corpus_prefix": retrieval.dense_corpus_prefix,
        "embedding_write_chunk_size": retrieval.embedding_write_chunk_size,
        "checkpoint_rows": retrieval.embedding_checkpoint_rows,
        "enable_resume": retrieval.enable_resume,
        "dtype": retrieval.embedding_dtype,
        "progress_interval": retrieval.progress_interval,
        "verbose": verbose,
    }
    document_dense = None
    document_preflight: dict[str, float] = {}
    if need_document_dense:
        if retrieval.preflight_sample_size:
            document_preflight = _run_preflight(
                store,
                "document",
                runtime,
                retrieval.preflight_sample_size,
                retrieval.preflight_timeout_threshold_hours,
                retrieval.strict_preflight,
                retrieval.dense_corpus_prefix,
                retrieval.embedding_dtype,
                verbose,
            )
        document_dense = MemmapDenseIndex(
            store, "document", document_index_path, cache_name="document", **shared_dense
        ).build()
    document_sparse = StoreSparseIndex(store, "document")

    document_results: dict[str, dict[str, list[SearchResult]]] = {}
    document_latency: dict[str, float] = {}
    if "sparse" in required_document_methods or "combined" in required_document_methods:
        start = perf_counter()
        document_results["sparse"] = {
            query.query_id: document_sparse.search(query.text, resolved.retrieval.document_top_k)
            for query in queries
        }
        document_latency["sparse"] = (perf_counter() - start) * 1000 / max(1, len(queries))
        if verbose:
            print(
                "Sparse document retrieval completed in "
                f"{_format_seconds(perf_counter() - start)}."
            )
    if "dense" in required_document_methods or "combined" in required_document_methods:
        assert document_dense is not None
        start = perf_counter()
        document_results["dense"] = document_dense.search_many(
            queries, resolved.retrieval.document_top_k
        )
        document_latency["dense"] = (perf_counter() - start) * 1000 / max(1, len(queries))
        if verbose:
            print(
                f"Dense document retrieval ({document_dense.search_backend}) completed in "
                f"{_format_seconds(perf_counter() - start)}."
            )
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

    candidates: dict[str, dict[str, list[str]]] = {
        method: {
            query.query_id: store.passage_ids_for_documents(
                result.item_id for result in document_results[method][query.query_id]
            )
            for query in queries
        }
        for method in required_document_methods
    }
    candidate_union = sorted(
        {
            passage_id
            for method_candidates in candidates.values()
            for query_candidates in method_candidates.values()
            for passage_id in query_candidates
        }
    )
    query_fingerprint = _fingerprint(query.query_id for query in queries)
    candidate_context = {
        "query_ids_fingerprint": query_fingerprint,
        "document_methods": sorted(required_document_methods),
        "document_top_k": retrieval.document_top_k,
        "fusion": retrieval.fusion,
        "rrf_k": retrieval.rrf_k,
        "passage_text_strategy": "title_text",
    }
    candidate_hash = _fingerprint(candidate_union)[:16]
    candidate_name = f"candidate_passage_{candidate_hash}"
    candidate_summary = {
        "total_passages": summary["passages"],
        "unique_candidate_documents": len(
            {
                result.item_id
                for method_results in document_results.values()
                for query_results in method_results.values()
                for result in query_results
            }
        ),
        "unique_candidate_passages": len(candidate_union),
        "candidate_reduction_percent": (
            100.0 * (1.0 - len(candidate_union) / max(1, int(summary["passages"])))
        ),
        "candidate_passages_by_document_method": {
            method: len(
                {
                    passage_id
                    for query_candidates in method_candidates.values()
                    for passage_id in query_candidates
                }
            )
            for method, method_candidates in candidates.items()
        },
        "selected_devices": list(runtime.devices) if runtime is not None else [],
        "document_dense_search_backend": (
            document_dense.search_backend if document_dense is not None else "not_needed"
        ),
        "passage_sparse_scope": "per_query_candidate_set",
    }
    summary = {**summary, **candidate_summary}
    if verbose:
        print(f"Candidate passage summary: {candidate_summary}")

    passage_sparse = None
    if required_passage_methods & {"sparse", "combined"}:
        passage_sparse = CandidateSparseIndex(
            store,
            candidate_index_path,
            candidate_union,
            cache_name=candidate_name,
            cache_context=candidate_context,
        ).build()
    passage_dense = None
    passage_preflight: dict[str, float] = {}
    if need_passage_dense:
        if retrieval.preflight_sample_size:
            passage_preflight = _run_preflight(
                store,
                "passage",
                runtime,
                retrieval.preflight_sample_size,
                retrieval.preflight_timeout_threshold_hours,
                retrieval.strict_preflight,
                retrieval.dense_corpus_prefix,
                retrieval.embedding_dtype,
                verbose,
                selected_ids=candidate_union,
            )
        passage_dense = MemmapDenseIndex(
            store,
            "passage",
            candidate_index_path,
            selected_ids=candidate_union,
            cache_name=candidate_name,
            cache_context=candidate_context,
            **shared_dense,
        ).build()

    summary["embedding_estimates"] = {
        "document": document_preflight,
        "candidate_passage": passage_preflight,
    }
    summary["index_status"] = {
        "document_dense": (
            document_dense.build_status if document_dense is not None else "not_needed"
        ),
        "candidate_passage_sparse": (
            passage_sparse.build_status if passage_sparse is not None else "not_needed"
        ),
        "candidate_passage_dense": (
            passage_dense.build_status if passage_dense is not None else "not_needed"
        ),
    }

    qrels_by_query: dict[str, dict[str, float]] = {}
    for qrel in qrels:
        qrels_by_query.setdefault(qrel.query_id, {})[qrel.passage_id] = qrel.score
    passage_to_doc = store.passage_to_document(qrel.passage_id for qrel in qrels)

    base_passage_methods: set[str] = set()
    if required_passage_methods & {"sparse", "combined"}:
        base_passage_methods.add("sparse")
    if required_passage_methods & {"dense", "combined"}:
        base_passage_methods.add("dense")
    dense_query_vectors: dict[str, np.ndarray] = {}
    dense_query_encoding_ms = 0.0
    if "dense" in base_passage_methods:
        assert passage_dense is not None
        started = perf_counter()
        dense_query_vectors = passage_dense.encode_queries(queries)
        dense_query_encoding_ms = (perf_counter() - started) * 1000 / max(1, len(queries))

    passage_result_cache: dict[
        tuple[str, str], dict[str, list[SearchResult]]
    ] = {}
    passage_latency_cache: dict[tuple[str, str], dict[str, float]] = {}
    for document_method in sorted(required_document_methods):
        for passage_method in sorted(base_passage_methods):
            cache_key = (document_method, passage_method)
            method_results: dict[str, list[SearchResult]] = {}
            method_latencies: dict[str, float] = {}
            started = perf_counter()
            for position, query in enumerate(queries, 1):
                candidate_ids = candidates[document_method][query.query_id]
                query_started = perf_counter()
                if passage_method == "sparse":
                    assert passage_sparse is not None
                    result = passage_sparse.search(
                        query.text, retrieval.passage_top_k, candidate_ids
                    )
                    encoding_latency = 0.0
                else:
                    assert passage_dense is not None
                    result = passage_dense.search_vector(
                        dense_query_vectors[query.query_id],
                        retrieval.passage_top_k,
                        candidate_ids,
                    )
                    encoding_latency = dense_query_encoding_ms
                method_results[query.query_id] = result
                method_latencies[query.query_id] = (
                    (perf_counter() - query_started) * 1000 + encoding_latency
                )
                if verbose and position % 250 == 0:
                    print(
                        f"Passage {document_method}__{passage_method}: "
                        f"{position:,} / {len(queries):,} queries"
                    )
            passage_result_cache[cache_key] = method_results
            passage_latency_cache[cache_key] = method_latencies
            if verbose:
                print(
                    f"Passage {document_method}__{passage_method} completed in "
                    f"{_format_seconds(perf_counter() - started)}."
                )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_root = output_path / run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    save_json(artifact_root / "config.json", asdict(resolved))
    save_json(artifact_root / "dataset_summary.json", summary)
    save_json(artifact_root / "candidate_passage_audit.json", candidate_summary)
    rows: list[dict[str, Any]] = []
    experiment_results: dict[str, dict[str, Any]] = {}
    for experiment in selected:
        if verbose:
            print(f"Running scalable {experiment.name} ...")
        runs: dict[str, QueryRun] = {}
        for query in queries:
            if experiment.passage_method == "sparse":
                cache_key = (experiment.document_method, "sparse")
                passage_results = passage_result_cache[cache_key][query.query_id]
                passage_latency = passage_latency_cache[cache_key][query.query_id]
            elif experiment.passage_method == "dense":
                cache_key = (experiment.document_method, "dense")
                passage_results = passage_result_cache[cache_key][query.query_id]
                passage_latency = passage_latency_cache[cache_key][query.query_id]
            else:
                sparse_key = (experiment.document_method, "sparse")
                dense_key = (experiment.document_method, "dense")
                sparse_results = passage_result_cache[sparse_key][query.query_id]
                dense_results = passage_result_cache[dense_key][query.query_id]
                passage_results = _fuse(
                    sparse_results,
                    dense_results,
                    method=resolved.retrieval.fusion,
                    k=resolved.retrieval.passage_top_k,
                    rrf_k=resolved.retrieval.rrf_k,
                )
                passage_latency = (
                    passage_latency_cache[sparse_key][query.query_id]
                    + passage_latency_cache[dense_key][query.query_id]
                )
            latency = document_latency[experiment.document_method] + passage_latency
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
            "runs": runs,
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


def _run_preflight(
    store: DiskDatasetStore,
    level: Level,
    runtime: DenseEncoderRuntime | None,
    sample_size: int,
    threshold_hours: float,
    strict: bool,
    corpus_prefix: str,
    dtype: str,
    verbose: bool,
    *,
    selected_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Measure a small encoding sample and estimate total embedding work."""
    if runtime is None or sample_size <= 0:
        return {}
    if selected_ids is None:
        total = store.item_count(level)
        texts: list[str] = []
        for _, _, text in store.iter_items(level):
            texts.append(corpus_prefix + text)
            if len(texts) >= sample_size:
                break
    else:
        items = store.items_for_ids(level, selected_ids[:sample_size])
        total = len(selected_ids)
        texts = [corpus_prefix + text for _, text in items]
    if not texts:
        return {"rows_per_second": 0.0, "estimated_hours": 0.0, "estimated_gib": 0.0}
    started = perf_counter()
    runtime.encode(texts)
    elapsed = max(perf_counter() - started, 1e-9)
    rows_per_second = len(texts) / elapsed
    estimated_seconds = total / rows_per_second
    bytes_per_value = np.dtype(dtype).itemsize
    estimated_bytes = total * runtime.dimension * bytes_per_value
    estimate = {
        "rows_per_second": rows_per_second,
        "estimated_hours": estimated_seconds / 3600,
        "estimated_gib": estimated_bytes / (1024**3),
    }
    if verbose:
        print(
            f"Preflight {level}: {rows_per_second:,.1f} rows/s; "
            f"estimated {_format_seconds(estimated_seconds)}; "
            f"{estimate['estimated_gib']:.2f} GiB"
        )
    if strict and estimate["estimated_hours"] >= threshold_hours:
        raise RuntimeError(
            f"Preflight estimates {estimate['estimated_hours']:.2f} hours for {level}, "
            f"which exceeds the {threshold_hours:.2f}-hour safety threshold."
        )
    return estimate
