# Hierarchical Retrieval Benchmark on DAPR

Phase 1 benchmark for deciding which hierarchical retrieval combination works best for a
document-aware passage retrieval setting. The implementation evaluates the nine HHR combinations:

| Document stage | Passage stage |
|---|---|
| sparse / dense / combined | sparse / dense / combined |

`combined` supports both reciprocal-rank fusion (RRF, the default) and the simple interleaving
baseline used in HHR-style experiments.

## What is ready

- Official `UKPLab/dapr` Hugging Face adapter for MS MARCO, Natural Questions, MIRACL, Genomics,
  and ConditionalQA.
- Two-stage query → documents → passages pipeline.
- Sparse BM25 and dense E5 retrieval, plus sparse+dense fusion at either stage.
- Passage nDCG@10 / Recall@100 and document nDCG / Recall.
- Per-query results, latency, NQ-hard category analysis, run metadata, and exportable artifacts.
- A Kaggle notebook whose cells only orchestrate reusable code from `src/dapr_hhr`.
- A single high-level `run_phase1_benchmark(...)` API; Kaggle owns no benchmark logic.
- Unit tests and a synthetic end-to-end smoke check.

No benchmark scores are committed. Run the notebook to produce them from the selected data and
model version.

## Project layout

```text
hierarchical-retrieval-benchmark/
├── configs/phase1.yaml
├── notebooks/01_phase1_hhr_on_dapr.ipynb
├── scripts/run_synthetic_smoke.py
├── src/dapr_hhr/
│   ├── data.py                 # DAPR adapter + stable data contracts
│   ├── text.py                 # reusable document/passage text strategies
│   ├── retrieval/              # BM25, sentence-transformer, combined
│   ├── fusion.py               # HHR interleave and RRF
│   ├── pipeline.py             # document → candidate passages
│   ├── metrics.py              # document/passage metrics and groups
│   ├── experiments.py          # nine-run registry and runner
│   ├── workflow.py             # complete Phase 1 public API for notebooks
│   └── artifacts.py            # metrics, rankings, environment metadata
└── tests/
```

## Run on Kaggle

1. Open `notebooks/01_phase1_hhr_on_dapr.ipynb` as a Kaggle notebook.
2. In **Notebook options**, enable Internet and select a GPU accelerator for dense runs.
3. Keep `RUN_MODE = "smoke"` for the first run. It uses 25 ConditionalQA queries, up to 5,000
   regular passages, and adds all gold passages for those queries.
4. Run all cells. Results are written under `/kaggle/working/dapr_hhr_outputs` and embeddings under
   `/kaggle/working/dapr_hhr_cache`.
5. Download `dapr_hhr_outputs.zip` from the Kaggle Output panel.

The bootstrap cell installs the package directly from the selected GitHub branch or commit. The
notebook then calls only this public function:

```python
from dapr_hhr import run_phase1_benchmark

report = run_phase1_benchmark(
    dataset_name="ConditionalQA",
    run_mode="smoke",
    fusion="rrf",
)
report.leaderboard
```

If a reusable function has a bug, fix it under `src/dapr_hhr`, add a regression test, and push the
commit. Restart the Kaggle session and rerun the installation cell; the notebook itself does not
need to be copied or edited. Keep `REPO_REF="main"` while developing, then use a commit SHA for final
reproducible runs.

DAPR is read through the official Hugging Face dataset API. Internet is required for the first run.
To run with Internet disabled later, save the Hugging Face/model cache as a private Kaggle Dataset
and point the cache environment variables to that input.

## Local verification

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_synthetic_smoke.py
```

Install the dense extra for real dense experiments:

```powershell
python -m pip install -e ".[dense,dev]"
```

## Experimental protocol

- Tune method choices and hyperparameters on MS MARCO train/dev.
- Freeze them before zero-shot evaluation on the other DAPR domains.
- Treat NQ-hard as a diagnostic subset; use its CR, MT, MHR, and AC categories for error analysis,
  not as another tuning set.
- Report both passage and document metrics. Passage-only scores cannot reveal whether failure
  occurred at document routing or passage ranking.
- Compare `interleave` with RRF as a declared fusion ablation rather than silently changing HHR.
- Record model ID, dataset split, limits, Git commit, latency, and retrieval depth with each run.

## Scale boundary

The included dense backend performs exact in-memory search, which is appropriate for smoke runs and
smaller sampled experiments. Full MIRACL and Genomics contain tens of millions of passages and need
a sharded/indexed backend such as FAISS or Pyserini plus persisted Kaggle datasets. The interfaces in
`BaseRetriever` let that backend be added without changing the notebook, metrics, or dataset adapter.

## Phase 2 boundary

HiREC-style cross-encoder reranking and evidence curation are deliberately not mixed into Phase 1.
After the HHR candidate generator is selected, add a reranker that consumes its top passages and
measure the incremental gain with the same artifact and metric functions.

## Sources

- [DAPR dataset](https://huggingface.co/datasets/UKPLab/dapr)
- [DAPR paper](https://aclanthology.org/2024.acl-long.236/)
- [HHR paper](https://aclanthology.org/2023.findings-acl.679/)
