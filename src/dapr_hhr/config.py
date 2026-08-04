"""Typed, dataset-agnostic benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RetrievalConfig:
    document_top_k: int = 20
    passage_top_k: int = 100
    fusion: str = "rrf"
    rrf_k: int = 60
    dense_backend: str = "sentence_transformers"
    dense_model: str = "intfloat/e5-small-v2"
    dense_model_revision: str = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
    dense_batch_size: int = 64
    multi_process_chunk_size: int = 5_000
    embedding_write_chunk_size: int = 50_000
    embedding_checkpoint_rows: int = 50_000
    dense_search_chunk_size: int = 50_000
    preferred_devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    enable_multi_gpu: bool = True
    enable_resume: bool = True
    embedding_dtype: str = "float32"
    preflight_sample_size: int = 0
    preflight_timeout_threshold_hours: float = 10.0
    strict_preflight: bool = False
    progress_interval: int = 50_000
    dense_query_prefix: str = "query: "
    dense_corpus_prefix: str = "passage: "
    hashing_features: int = 512


@dataclass(frozen=True)
class EvaluationConfig:
    ndcg_k: int = 10
    recall_k: int = 100


@dataclass(frozen=True)
class RunConfig:
    mode: str = "smoke"
    smoke_experiments: tuple[str, ...] = (
        "sparse__sparse",
        "sparse__dense",
        "dense__dense",
        "combined__combined",
    )
    baseline_experiments: tuple[str, ...] = (
        "sparse__dense",
        "dense__dense",
        "combined__dense",
        "combined__combined",
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    project_name: str = "dapr-hhr-phase1"
    seed: int = 42
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _section(cls: type, payload: dict[str, Any], name: str):
    values = dict(payload.get(name, {}))
    if cls is RunConfig:
        for key in ("smoke_experiments", "baseline_experiments"):
            if key in values:
                values[key] = tuple(values[key])
    if cls is RetrievalConfig and "preferred_devices" in values:
        values["preferred_devices"] = tuple(values["preferred_devices"])
    return cls(**values)


def validate_config(config: BenchmarkConfig) -> None:
    errors: list[str] = []
    retrieval = config.retrieval
    if retrieval.document_top_k <= 0 or retrieval.passage_top_k <= 0:
        errors.append("retrieval depths must be positive")
    if retrieval.fusion not in {"rrf", "interleave"}:
        errors.append("fusion must be 'rrf' or 'interleave'")
    if retrieval.rrf_k <= 0:
        errors.append("rrf_k must be positive")
    if retrieval.dense_backend not in {"hashing", "sentence_transformers"}:
        errors.append("dense_backend must be hashing or sentence_transformers")
    if retrieval.dense_batch_size <= 0 or retrieval.hashing_features <= 0:
        errors.append("dense batch size and hashing features must be positive")
    if any(
        value <= 0
        for value in (
            retrieval.multi_process_chunk_size,
            retrieval.embedding_write_chunk_size,
            retrieval.embedding_checkpoint_rows,
            retrieval.dense_search_chunk_size,
            retrieval.progress_interval,
        )
    ):
        errors.append("dense chunk, checkpoint, search, and progress sizes must be positive")
    if retrieval.preflight_sample_size < 0:
        errors.append("preflight sample size cannot be negative")
    if retrieval.preflight_timeout_threshold_hours <= 0:
        errors.append("preflight timeout threshold must be positive")
    if retrieval.embedding_dtype not in {"float32", "float16"}:
        errors.append("embedding_dtype must be float32 or float16")
    if config.evaluation.ndcg_k <= 0 or config.evaluation.recall_k <= 0:
        errors.append("evaluation cutoffs must be positive")
    if config.run.mode not in {"smoke", "baseline", "full"}:
        errors.append("run mode must be smoke, baseline, or full")
    if errors:
        raise ValueError("Invalid benchmark configuration:\n- " + "\n- ".join(errors))


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load reusable retrieval/evaluation settings from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = BenchmarkConfig(
        project_name=payload.get("project_name", "dapr-hhr-phase1"),
        seed=int(payload.get("seed", 42)),
        retrieval=_section(RetrievalConfig, payload, "retrieval"),
        evaluation=_section(EvaluationConfig, payload, "evaluation"),
        run=_section(RunConfig, payload, "run"),
    )
    validate_config(config)
    return config
