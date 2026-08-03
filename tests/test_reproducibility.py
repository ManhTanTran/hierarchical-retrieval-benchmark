from dataclasses import replace

from dapr_hhr.config import BenchmarkConfig
from dapr_hhr.experiments import build_experiment_registry, run_hhr_experiment
from dapr_hhr.retrieval.sparse import BM25Retriever
from dapr_hhr.smoke import make_synthetic_bundle


class RecordingRetriever(BM25Retriever):
    calls: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        super().__init__()


def test_model_revision_is_forwarded_and_text_changes_cache_key(tmp_path):
    RecordingRetriever.calls.clear()
    config = BenchmarkConfig()
    experiment = build_experiment_registry()["dense__dense"]
    original = make_synthetic_bundle()
    run_hhr_experiment(
        experiment,
        original,
        config,
        cache_dir=tmp_path,
        dense_factory=RecordingRetriever,
    )
    first_paths = [call["cache_path"] for call in RecordingRetriever.calls]
    assert all(
        call["model_revision"] == config.retrieval.dense_model_revision
        for call in RecordingRetriever.calls
    )

    RecordingRetriever.calls.clear()
    changed = make_synthetic_bundle()
    changed.documents[0] = replace(changed.documents[0], text="changed body")
    run_hhr_experiment(
        experiment,
        changed,
        config,
        cache_dir=tmp_path,
        dense_factory=RecordingRetriever,
    )
    second_paths = [call["cache_path"] for call in RecordingRetriever.calls]
    assert first_paths != second_paths
