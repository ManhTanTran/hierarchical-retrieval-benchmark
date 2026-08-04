# Hierarchical Retrieval Benchmark

Reusable Phase 1 infrastructure for comparing the nine document-retriever plus
passage-retriever combinations:

| Document stage | Passage stage |
|---|---|
| sparse / dense / combined | sparse / dense / combined |

`combined` supports reciprocal-rank fusion and HHR-style interleaving.

## Responsibility boundary

The Kaggle notebook is maintained locally and uploaded directly to Kaggle. It
is intentionally not committed to this repository. That notebook owns
DAPR-specific work:

- download from the pinned `UKPLab/dapr` Hugging Face revision;
- normalize official docs, corpus, query, qrel, and NQ-hard schemas;
- select datasets and deterministically sample queries;
- enforce the MS MARCO zero-shot protocol;
- map NQ-hard labels to CR, MT, MHR, and AC.

Reusable code under `src/dapr_hhr` owns retrieval, fusion, metrics, the
experiment matrix, caching, and artifacts. Small datasets can use an in-memory
`DatasetBundle`; large datasets can stream normalized records into a
`DiskDatasetStore`. Neither path imports DAPR-specific code.

## Layout

```text
hierarchical-retrieval-benchmark/
├── configs/phase1.yaml
├── scripts/run_synthetic_smoke.py
├── src/dapr_hhr/
│   ├── data.py
│   ├── scalable.py
│   ├── config.py
│   ├── retrieval/
│   ├── fusion.py
│   ├── pipeline.py
│   ├── metrics.py
│   ├── experiments.py
│   ├── workflow.py
│   └── artifacts.py
└── tests/
```

## Kaggle

1. Upload the locally maintained notebook directly to Kaggle.
2. Enable Internet. A GPU is only needed for real sentence-transformer runs.
3. Run all cells unchanged in `smoke` mode first. It uses a deterministic local
   bundle and hashing backend, so its scores are not DAPR results.
4. In the central cell, change `run_mode` to `baseline`, initially override
   `datasets` with one dataset, and optionally set `query_sample_size`.
5. Download the ZIP paths printed in the final cell from Kaggle Output.

The local notebook's install cell uses this repository only for reusable Python
code. Keep `REPO_REF="main"` while developing; use a tested commit SHA for a
reported experiment.

Real DAPR modes use a document-first scalable path. Hugging Face rows stream into
SQLite, global sparse and dense document indexes are built once, and document
retrieval runs before any passage index is created. The backend then builds one
candidate-only passage FTS5 index and one candidate-only dense memmap for the
union of passages under the retrieved documents. Every query is still restricted
to its own document-derived passage candidates.

Sentence Transformers can use one persistent process pool across `cuda:0` and
`cuda:1`. Document embeddings are checkpointed atomically and resume from the
last completed chunk. Put disposable stores and candidate caches under
`/kaggle/tmp`; put document checkpoints that must be exported under
`/kaggle/working`. Factory Reset deletes `/kaggle/tmp`.

## Reusable API

```python
from dapr_hhr import DatasetBundle, run_phase1_benchmark

bundle = DatasetBundle(...)
report = run_phase1_benchmark(
    bundle,
    run_mode="baseline",
    dense_backend="sentence_transformers",
)
print(report.leaderboard)
```

For a large normalized corpus:

```python
from dapr_hhr import DiskDatasetStore, run_scalable_phase1_benchmark

store = DiskDatasetStore.create("dataset.sqlite", name="my_dataset")
store.add_documents(document_iterator)
store.add_passages(passage_iterator)
store.add_queries(query_iterator)
store.add_qrels(qrel_iterator)
store.finalize()

report = run_scalable_phase1_benchmark(
    store,
    run_mode="full",
    index_dir="indexes",
    document_checkpoint_dir="checkpoints",
    candidate_passage_cache_dir="candidate-cache",
    output_dir="outputs",
)
```

## Local verification

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python scripts/run_synthetic_smoke.py
python scripts/run_scalable_smoke.py
```

## Protocol and scale boundary

- Tune only on MS MARCO train/dev, then freeze choices.
- Treat NQ-hard as diagnostic-only.
- Preserve graded Genomics relevance for nDCG.
- Report both passage and document metrics.
- Query sampling does not reduce global document ingestion or dense document
  indexing, but it now reduces the candidate passage union and passage indexing.
- The scalable dense backend uses exact cosine search over a memory-mapped
  global document matrix and candidate-only passage matrix. FTS5 provides global
  document ranking and candidate-only passage ranking. Results produced by the
  old global-passage or in-memory path are not directly comparable, so rerun
  every dataset used in one comparison with the same backend.
- MIRACL can still require multiple Kaggle sessions because dense retrieval must
  embed all 5.76 million documents. Checkpoint/resume avoids restarting from row
  zero, but checkpoints must be exported before a Factory Reset.

No notebooks, benchmark scores, datasets, embeddings, model weights, or
credentials are committed.

## Sources

- [DAPR dataset](https://huggingface.co/datasets/UKPLab/dapr)
- [DAPR paper](https://aclanthology.org/2024.acl-long.236/)
- [HHR paper](https://aclanthology.org/2023.findings-acl.679/)
