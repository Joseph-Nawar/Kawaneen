# Testing policy

Public CI runs hermetic tests from the tracked repository only. It never uploads
or downloads private corpora, frozen evaluation releases, model caches, or raw
source material.

Tests marked `private_artifact` remain available for local validation of
intentionally untracked integration assets. Public CI excludes them with:

```bash
pytest -m "not private_artifact" \
  --cov=kawaneen --cov-branch --cov-report=term-missing --cov-fail-under=85
```

Run the private integration subset locally with:

```bash
KAWANEEN_PRIVATE_EXTERNAL_ROOT=/path/to/private/reviews \
pytest -m private_artifact --no-cov
```

The public coverage denominator omits only named offline corpus/evaluation,
persisted-artifact, and model/provider experiment modules whose behavior
intrinsically requires those untracked assets. This includes Phase 11 private
model-run, checkpoint, readiness, orchestration, prompt/provider, and offline
evaluation tooling; Phase 11 contracts, span validation, candidate generation,
normalization, and deterministic/hybrid assembly remain in the coverage scope.
Public acquisition, parsing, retrieval primitives, validation, and normal
application logic remain in the coverage scope. The CI threshold remains 85%.
