# Repository guidance

- Keep every Jupyter/Kaggle notebook local; never commit `*.ipynb` files.
- The local notebook owns dataset-specific DAPR download, official-schema
  normalization, sampling, and diagnostic label mapping.
- Put reusable dataset contracts, retrieval, fusion, evaluation, experiment, and
  artifact logic under `src/dapr_hhr`.
- Keep the reusable workflow dependent only on a validated `DatasetBundle`.
- Add or update tests whenever reusable behavior changes.
- Never commit DAPR data, model weights, embeddings, credentials, or large outputs.
- Do not report benchmark numbers unless they were produced by a recorded run.
- Preserve the DAPR zero-shot protocol: tune on MS MARCO and freeze choices before
  evaluating other domains.
