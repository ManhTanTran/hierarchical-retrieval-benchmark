from dapr_hhr.experiments import build_experiment_registry


def test_registry_contains_all_nine_hhr_combinations():
    registry = build_experiment_registry()
    assert len(registry) == 9
    assert "sparse__sparse" in registry
    assert "combined__combined" in registry
