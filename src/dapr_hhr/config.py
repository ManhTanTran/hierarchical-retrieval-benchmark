"""Typed configuration shared by local scripts and Kaggle notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    hf_repo: str = "UKPLab/dapr"
    dataset_name: str = "ConditionalQA"
    split: str = "test"
    query_limit: int | None = 25
    corpus_limit: int | None = 5000
    preserve_gold: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    document_top_k: int = 20
    passage_top_k: int = 100
    fusion: str = "rrf"
    rrf_k: int = 60
    dense_model: str = "intfloat/e5-small-v2"
    dense_batch_size: int = 64
    dense_query_prefix: str = "query: "
    dense_corpus_prefix: str = "passage: "


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


@dataclass(frozen=True)
class BenchmarkConfig:
    project_name: str = "dapr-hhr-phase1"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _section(cls: type, payload: dict[str, Any], name: str):
    values = dict(payload.get(name, {}))
    if cls is RunConfig and "smoke_experiments" in values:
        values["smoke_experiments"] = tuple(values["smoke_experiments"])
    return cls(**values)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a YAML configuration into immutable dataclasses."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return BenchmarkConfig(
        project_name=payload.get("project_name", "dapr-hhr-phase1"),
        seed=int(payload.get("seed", 42)),
        data=_section(DataConfig, payload, "data"),
        retrieval=_section(RetrievalConfig, payload, "retrieval"),
        evaluation=_section(EvaluationConfig, payload, "evaluation"),
        run=_section(RunConfig, payload, "run"),
    )

