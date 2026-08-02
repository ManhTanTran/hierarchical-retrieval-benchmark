from zipfile import ZipFile

from dapr_hhr.smoke import make_synthetic_bundle
from dapr_hhr.workflow import run_phase1_benchmark


def test_high_level_workflow_runs_without_notebook_logic(tmp_path):
    report = run_phase1_benchmark(
        dataset_name="ConditionalQA",
        run_mode="smoke",
        experiment_names=["sparse__sparse"],
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        bundle=make_synthetic_bundle(),
        document_top_k=2,
        passage_top_k=3,
        verbose=False,
    )

    assert report.best_experiment == "sparse__sparse"
    assert report.dataset_summary["queries"] == 2
    assert report.artifact_root.exists()
    assert report.archive_path.exists()
    with ZipFile(report.archive_path) as archive:
        assert "leaderboard.csv" in archive.namelist()
        assert "config.json" in archive.namelist()
