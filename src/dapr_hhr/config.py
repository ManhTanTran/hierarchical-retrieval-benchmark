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
