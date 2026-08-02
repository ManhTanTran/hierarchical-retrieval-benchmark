"""High-level Phase 1 API used by the thin Kaggle notebook."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import save_json, save_run_artifacts
from .config import BenchmarkConfig
from .data import DAPRDatasetAdapter, DatasetBundle, summarize_dataset
from .experiments import build_experiment_registry, run_hhr_experiment
from .metrics import evaluate_by_group
from .retrieval import SentenceTransformerRetriever
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
    base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    return base / "dapr_hhr_cache", base / "dapr_hhr_outputs"


def _build_config(
    base: BenchmarkConfig,
    dataset_name: str,
    run_mode: str,
    query_limit: int | None,
    corpus_limit: int | None,
    preserve_gold: bool,
    fusion: str,
    dense_model: str | None,
    document_top_k: int | None,
    passage_top_k: int | None,
) -> BenchmarkConfig:
    retrieval_overrides: dict[str, Any] = {"fusion": fusion}
    if dense_model is not None:
        retrieval_overrides["dense_model"] = dense_model
    if document_top_k is not None:
        retrieval_overrides["document_top_k"] = document_top_k
    if passage_top_k is not None:
        retrieval_overrides["passage_top_k"] = passage_top_k
    return replace(
        base,
        data=replace(
            base.data,
            dataset_name=dataset_name,
            query_limit=query_limit,
            corpus_limit=corpus_limit,
            preserve_gold=preserve_gold,
        ),
        retrieval=replace(base.retrieval, **retrieval_overrides),
        run=replace(base.run, mode=run_mode),
    )


def run_phase1_benchmark(
    dataset_name: str = "ConditionalQA",
    run_mode: str = "smoke",
    *,
    query_limit: int | None = None,
    corpus_limit: int | None = None,
    preserve_gold: bool = True,
    fusion: str = "rrf",
    dense_model: str | None = None,
    document_top_k: int | None = None,
    passage_top_k: int | None = None,
    experiment_names: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config: BenchmarkConfig | None = None,
    bundle: DatasetBundle | None = None,
    adapter: DAPRDatasetAdapter | None = None,
    dense_factory: Callable[..., BaseRetriever] = SentenceTransformerRetriever,
    verbose: bool = True,
) -> Phase1Report:
    """Run the complete DAPR Phase 1 workflow without notebook-owned logic.

    Smoke mode defaults to 25 queries and 5,000 ordinary passages while preserving
    every gold passage for those queries. Full mode defaults to no sampling. Tests may
    inject a prepared ``bundle`` to avoid network and model downloads.
    """
    if run_mode not in {"smoke", "full"}:
        raise ValueError("run_mode must be 'smoke' or 'full'.")
    if query_limit is None and run_mode == "smoke":
        query_limit = 25
    if corpus_limit is None and run_mode == "smoke":
        corpus_limit = 5_000

    resolved_config = _build_config(
        config or BenchmarkConfig(),
        dataset_name,
        run_mode,
        query_limit,
        corpus_limit,
        preserve_gold,
        fusion,
        dense_model,
        document_top_k,
        passage_top_k,
    )

    default_cache, default_output = _default_work_dirs()
    cache_path = Path(cache_dir) if cache_dir is not None else default_cache
    output_path = Path(output_dir) if output_dir is not None else default_output
    cache_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    data_adapter = adapter or DAPRDatasetAdapter(resolved_config.data.hf_repo)
    if bundle is None:
        if verbose:
            print(f"Loading {dataset_name} ({run_mode}) from {resolved_config.data.hf_repo} ...")
        bundle = data_adapter.load_bundle(
            dataset_name=resolved_config.data.dataset_name,
            split=resolved_config.data.split,
            query_limit=resolved_config.data.query_limit,
            corpus_limit=resolved_config.data.corpus_limit,
            preserve_gold=resolved_config.data.preserve_gold,
        )
    dataset_summary = summarize_dataset(bundle)
    if verbose:
        print(f"Dataset summary: {dataset_summary}")
    if dataset_summary["queries_with_relevant_passages"] == 0:
        raise ValueError("The selected sample contains no positive qrels.")

    registry = build_experiment_registry()
    selected_names = list(
        experiment_names
        or (
            resolved_config.run.smoke_experiments
            if run_mode == "smoke"
            else tuple(registry)
        )
    )
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

    grouped_metrics: dict[str, dict[str, float]] = {}
    if dataset_name == "NaturalQuestions":
        query_metadata = data_adapter.load_query_metadata(dataset_name)
        best_name = str(leaderboard.iloc[0]["experiment"])
        grouped_metrics = evaluate_by_group(
            experiment_results[best_name]["per_query"],
            query_metadata,
        )
        save_json(artifact_root / "nq_hard_grouped_metrics.json", grouped_metrics)

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
