"""Run a no-download sparse HHR smoke test and print measured metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dapr_hhr.metrics import evaluate_runs  # noqa: E402
from dapr_hhr.pipeline import HHRPipeline  # noqa: E402
from dapr_hhr.retrieval import BM25Retriever  # noqa: E402
from dapr_hhr.smoke import make_synthetic_bundle  # noqa: E402


def main() -> None:
    bundle = make_synthetic_bundle()
    pipeline = HHRPipeline(
        BM25Retriever(),
        BM25Retriever(),
        document_top_k=2,
        passage_top_k=3,
    ).fit(bundle)
    metrics, _ = evaluate_runs(pipeline.run(bundle.queries), bundle, ndcg_k=2, recall_k=3)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

