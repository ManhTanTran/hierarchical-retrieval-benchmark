from dapr_hhr.metrics import evaluate_runs
from dapr_hhr.pipeline import HHRPipeline
from dapr_hhr.retrieval.sparse import BM25Retriever
from dapr_hhr.smoke import make_synthetic_bundle


def test_sparse_hierarchy_runs_end_to_end():
    bundle = make_synthetic_bundle()
    pipeline = HHRPipeline(
        BM25Retriever(),
        BM25Retriever(),
        document_top_k=2,
        passage_top_k=3,
    ).fit(bundle)
    runs = pipeline.run(bundle.queries)
    metrics, per_query = evaluate_runs(runs, bundle, ndcg_k=2, recall_k=3)
    assert set(runs) == {"q1", "q2"}
    assert metrics["passage_recall@3"] == 1.0
    assert set(per_query) == {"q1", "q2"}

