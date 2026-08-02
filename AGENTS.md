# Repository guidance

- Keep notebooks as orchestration and explanation layers only.
- Put reusable loading, retrieval, fusion, evaluation, and artifact logic under `src/dapr_hhr`.
- Add or update tests whenever reusable behavior changes.
- Never commit DAPR data, model weights, embeddings, credentials, or large run outputs.
- Do not report benchmark numbers unless they were produced by a recorded run.
- Preserve the DAPR zero-shot protocol: tune choices on MS MARCO and freeze them before evaluating other domains.

