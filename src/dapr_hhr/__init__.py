"""Reusable components for the DAPR hierarchical-retrieval benchmark."""

from .config import BenchmarkConfig, RetrievalConfig, load_config, validate_config
from .data import (
    DatasetBundle,
    Document,
    Passage,
    Qrel,
    Query,
    summarize_dataset,
    validate_dataset,
)
from .experiments import HHRExperiment, build_experiment_registry, run_hhr_experiment
from .scalable import (
    CandidateSparseIndex,
    DenseEncoderRuntime,
    DiskDatasetStore,
    MemmapDenseIndex,
    ScalablePhase1Report,
    run_scalable_phase1_benchmark,
)
from .workflow import Phase1Report, run_phase1_benchmark

__all__ = [
    "BenchmarkConfig",
    "CandidateSparseIndex",
    "DatasetBundle",
    "DenseEncoderRuntime",
    "DiskDatasetStore",
    "Document",
    "HHRExperiment",
    "MemmapDenseIndex",
    "Passage",
    "Phase1Report",
    "Qrel",
    "Query",
    "RetrievalConfig",
    "ScalablePhase1Report",
    "build_experiment_registry",
    "load_config",
    "run_hhr_experiment",
    "run_phase1_benchmark",
    "run_scalable_phase1_benchmark",
    "summarize_dataset",
    "validate_config",
    "validate_dataset",
]
