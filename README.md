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

Real DAPR modes use the scalable path: Hugging Face rows stream into SQLite,
FTS5 supplies the persistent sparse index, and sentence-transformer embeddings
are generated in batches into memory-mapped `.npy` files. Indexes are built once
per retrieval level and reused by the selected experiment matrix. Temporary
stores and indexes belong under `/kaggle/tmp`; only final artifacts belong under
`/kaggle/working`.

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
- Query sampling reduces evaluation work but does not reduce corpus ingestion or
  index construction.
- The scalable dense backend uses exact cosine search over a memory-mapped
  matrix; FTS5 provides the disk-backed sparse ranking. Results produced by the
  old in-memory BM25 path are not directly comparable, so rerun every dataset
  used in one comparison with the same backend.
- Very large MIRACL and Genomics builds can still exceed Kaggle's temporary-disk
  or session-time limits. The backend prevents full-corpus RAM materialization;
  it does not make storage or encoding cost disappear.

No notebooks, benchmark scores, datasets, embeddings, model weights, or
credentials are committed.

## Sources

- [DAPR dataset](https://huggingface.co/datasets/UKPLab/dapr)
- [DAPR paper](https://aclanthology.org/2024.acl-long.236/)
- [HHR paper](https://aclanthology.org/2023.findings-acl.679/)
