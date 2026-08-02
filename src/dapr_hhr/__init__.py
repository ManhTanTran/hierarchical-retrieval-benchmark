"""Reusable components for the DAPR hierarchical-retrieval benchmark."""

from .config import BenchmarkConfig, load_config
from .data import DAPRDatasetAdapter, DatasetBundle, Document, Passage, Qrel, Query
from .experiments import HHRExperiment, build_experiment_registry, run_hhr_experiment

__all__ = [
    "BenchmarkConfig",
    "DAPRDatasetAdapter",
    "DatasetBundle",
    "Document",
    "HHRExperiment",
    "Passage",
    "Qrel",
    "Query",
    "build_experiment_registry",
    "load_config",
    "run_hhr_experiment",
]

