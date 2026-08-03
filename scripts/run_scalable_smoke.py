"""Run the full scalable matrix with a deterministic two-dimensional encoder."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dapr_hhr import (  # noqa: E402
    DiskDatasetStore,
    Document,
    Passage,
    Qrel,
    Query,
    run_scalable_phase1_benchmark,
)


class KeywordEncoder:
    def get_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, **_kwargs):
        rows = [[float("apple" in text.lower()), float("banana" in text.lower())] for text in texts]
        matrix = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dapr-scalable-smoke-") as directory:
        root = Path(directory)
        store = DiskDatasetStore.create(root / "dataset.sqlite", name="scalable-smoke")
        store.add_documents(
            [Document("d1", "Apple", "apple orchard"), Document("d2", "Banana", "banana grove")]
        )
        store.add_passages(
            [
                Passage("p1", "d1", "Apple", "red apple", 0),
                Passage("p2", "d2", "Banana", "yellow banana", 0),
            ]
        )
        store.add_queries([Query("q1", "apple")])
        store.add_qrels([Qrel("q1", "p1", 1.0)])
        store.finalize()
        report = run_scalable_phase1_benchmark(
            store,
            run_mode="full",
            index_dir=root / "indexes",
            output_dir=root / "outputs",
            encoder=KeywordEncoder(),
            document_top_k=2,
            passage_top_k=2,
            verbose=False,
        )
        if len(report.leaderboard) != 9:
            raise RuntimeError("Scalable smoke did not complete all nine experiments.")
        print("Scalable full-matrix smoke passed.")


if __name__ == "__main__":
    main()
