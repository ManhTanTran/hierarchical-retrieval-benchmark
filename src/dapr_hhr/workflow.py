"""Dataset-agnostic high-level Phase 1 benchmark API."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import save_json, save_run_artifacts
from .config import BenchmarkConfig, validate_config
from .data import DatasetBundle, summarize_dataset
from .experiments import build_experiment_registry, run_hhr_experiment
from .metrics import evaluate_by_group
from .retrieval.base import BaseRetriever


@dataclass
class Phase1Report:
    """Compact handle returned to notebooks after a benchmark run."""

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


def _default_work_dirs() -> tuple[Path, Path]:
    on_kaggle = os.name != "nt" and Path("/kaggle/working").is_dir()
    base = Path("/kaggle/working") if on_kaggle else Path.cwd()
    return base / "dapr_hhr_cache", base / "dapr_hhr_outputs"


def _build_config(
    base: BenchmarkConfig,
    run_mode: str,
    fusion: str,
    dense_backend: str | None,
    dense_model: str | None,
    dense_model_revision: str | None,
    document_top_k: int | None,
    passage_top_k: int | None,
) -> BenchmarkConfig:
    retrieval_overrides: dict[str, Any] = {"fusion": fusion}
    for key, value in (
        ("dense_backend", dense_backend),
        ("dense_model", dense_model),
        ("dense_model_revision", dense_model_revision),
        ("document_top_k", document_top_k),
        ("passage_top_k", passage_top_k),
    ):
        if value is not None:
            retrieval_overrides[key] = value
    config = replace(
        base,
        retrieval=replace(base.retrieval, **retrieval_overrides),
        run=replace(base.run, mode=run_mode),
    )
    validate_config(config)
    return config


def run_phase1_benchmark(
    bundle: DatasetBundle,
    run_mode: str = "smoke",
    *,
    fusion: str = "rrf",
    dense_backend: str | None = None,
    dense_model: str | None = None,
    dense_model_revision: str | None = None,
    document_top_k: int | None = None,
    passage_top_k: int | None = None,
    experiment_names: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config: BenchmarkConfig | None = None,
    query_metadata: dict[str, dict[str, Any]] | None = None,
    dense_factory: Callable[..., BaseRetriever] | None = None,
    verbose: bool = True,
) -> Phase1Report:
    """Run Phase 1 on a validated, already-normalized dataset bundle.

    Dataset acquisition, source-schema normalization, and sampling stay upstream
    (for DAPR, in the Kaggle notebook). This function is reusable for any dataset
    that can produce :class:`DatasetBundle`.
    """
    bundle.validate()
    resolved_config = _build_config(
        config or BenchmarkConfig(),
        run_mode,
        fusion,
        dense_backend,
        dense_model,
        dense_model_revision,
        document_top_k,
        passage_top_k,
    )

    default_cache, default_output = _default_work_dirs()
    cache_path = Path(cache_dir) if cache_dir is not None else default_cache
    output_path = Path(output_dir) if output_dir is not None else default_output
    cache_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_summary = summarize_dataset(bundle)
    if verbose:
        print(f"Dataset summary: {dataset_summary}")
    if dataset_summary["queries_with_relevant_passages"] == 0:
        raise ValueError("The selected dataset contains no positive qrels.")

    registry = build_experiment_registry()
    defaults = {
        "smoke": resolved_config.run.smoke_experiments,
        "baseline": resolved_config.run.baseline_experiments,
        "full": tuple(registry),
    }
    selected_names = list(experiment_names or defaults[run_mode])
    unknown = sorted(set(selected_names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown experiment names: {unknown}")
    if not selected_names:
        raise ValueError("At least one experiment must be selected.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_root = output_path / run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    save_json(artifact_root / "config.json", asdict(resolved_config))
    save_json(artifact_root / "dataset_summary.json", dataset_summary)

    rows: list[dict[str, Any]] = []
    experiment_results: dict[str, dict[str, Any]] = {}
    for name in selected_names:
        if verbose:
            print(f"Running {name} ...")
        result = run_hhr_experiment(
            registry[name],
            bundle,
            resolved_config,
            cache_dir=cache_path,
            dense_factory=dense_factory,
        )
        artifact_dir = save_run_artifacts(artifact_root, name, result)
        rows.append(
            {
                "experiment": name,
                **result["metrics"],
                "artifact_dir": str(artifact_dir),
            }
        )
        experiment_results[name] = {
            "experiment": result["experiment"],
            "metrics": result["metrics"],
            "per_query": result["per_query"],
            "artifact_dir": artifact_dir,
        }

    score_column = f"passage_ndcg@{resolved_config.evaluation.ndcg_k}"
    leaderboard = (
        pd.DataFrame(rows).sort_values(score_column, ascending=False).reset_index(drop=True)
    )
    leaderboard.to_csv(artifact_root / "leaderboard.csv", index=False)

    metadata = query_metadata if query_metadata is not None else bundle.query_metadata
    grouped_metrics: dict[str, dict[str, float]] = {}
    if metadata:
        best_name = str(leaderboard.iloc[0]["experiment"])
        grouped_metrics = evaluate_by_group(experiment_results[best_name]["per_query"], metadata)
        save_json(artifact_root / "grouped_metrics.json", grouped_metrics)

    archive_path = Path(
        shutil.make_archive(
            str(output_path / f"{run_id}_artifacts"),
            "zip",
            root_dir=artifact_root,
        )
    )
    if verbose:
        print(f"Completed {len(selected_names)} experiment(s). Artifacts: {archive_path}")
    return Phase1Report(
        run_id=run_id,
        config=resolved_config,
        dataset_summary=dataset_summary,
        leaderboard=leaderboard,
        artifact_root=artifact_root,
        archive_path=archive_path,
        grouped_metrics=grouped_metrics,
        experiment_results=experiment_results,
    )
