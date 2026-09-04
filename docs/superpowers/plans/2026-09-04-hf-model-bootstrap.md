# HF Model Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one idempotent Compose init job that repairs the persistent Hugging Face cache at the exact frozen Phase-8 revisions before the API starts.

**Architecture:** A short-lived `docker/hf-model-init.py` process loads `FrozenServingConfiguration` from the tracked Phase-8 manifests, calls `huggingface_hub.snapshot_download` once for the dense BGE-M3 and once for the frozen reranker, and checks each returned snapshot for ordinary metadata. Compose mounts the same `huggingface_cache` volume into the init job and makes `kawaneen-api` depend on successful completion; the API preload remains the runtime compatibility check.

**Tech Stack:** Python 3.12, `huggingface_hub`, Docker Compose, pytest, uv, Ruff, Pyright.

**Spec:** User-provided Phase 17 continuation request (`fe4317a9-8f2d-4a34-b007-8bbb34ba11bc/pasted-text.txt`).

## Global Constraints

- Do not redesign Phase 17 or change retrieval, model, generation, precision, or Qwen behavior.
- Derive both Hugging Face identities through `load_frozen_serving_configuration(...)`; do not duplicate revisions in Compose.
- Use exact immutable revisions only; never resolve `main`/latest and never force-download on every startup.
- Repair normal Hugging Face cache state without deleting the cache or touching unrelated models.
- Print only model IDs, revisions, and cache status; never print credentials, private artifacts, or source text.
- Do not access HOLDOUT, publish Hugging Face, or merge PR #13.

---

### Task 1: Define bootstrap behavior with focused offline tests

**Files:**
- Create: `tests/test_hf_model_init.py`
- Read: `src/kawaneen/api/composition.py`

**Interfaces:**
- The tests will specify `bootstrap_models(data_directory: Path, cache_directory: Path, snapshot_downloader: Callable[..., str | Path]) -> tuple[ModelSnapshot, ModelSnapshot]` in `docker/hf-model-init.py`.
- The downloader receives `repo_id`, `revision`, `cache_dir`, and `local_files_only=False`.

- [ ] **Step 1: Write failing tests**

Cover these behaviors with temporary tracked-style Phase-8 manifests and a fake downloader:

```python
def test_bootstrap_passes_frozen_dense_and_reranker_identities(tmp_path, monkeypatch):
    calls = []
    make_frozen_data(tmp_path / "data")
    def download(**kwargs):
        calls.append(kwargs)
        snapshot = tmp_path / "cache" / kwargs["repo_id"].replace("/", "--")
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot)
    result = load_hf_bootstrap_module().bootstrap_models(tmp_path / "data", tmp_path / "cache", download)
    assert [(call["repo_id"], call["revision"]) for call in calls] == [
        ("BAAI/bge-m3", "b" * 40),
        ("BAAI/bge-reranker-v2-m3", "r" * 40),
    ]
    assert all(snapshot.path.is_dir() for snapshot in result)

def test_bootstrap_is_idempotent_and_does_not_force_download(tmp_path):
    calls = []
    make_frozen_data(tmp_path / "data")
    def download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / "cache" / str(len(calls))
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        return path
    module = load_hf_bootstrap_module()
    module.bootstrap_models(tmp_path / "data", tmp_path / "cache", download)
    module.bootstrap_models(tmp_path / "data", tmp_path / "cache", download)
    assert all(call["local_files_only"] is False for call in calls)
    assert all("force_download" not in call for call in calls)

def test_bootstrap_propagates_download_failure(tmp_path):
    make_frozen_data(tmp_path / "data")
    def download(**kwargs):
        raise RuntimeError("offline frozen revision")
    with pytest.raises(RuntimeError, match="offline frozen revision"):
        load_hf_bootstrap_module().bootstrap_models(tmp_path / "data", tmp_path / "cache", download)

def test_bootstrap_rejects_snapshot_without_metadata(tmp_path):
    make_frozen_data(tmp_path / "data")
    def download(**kwargs):
        path = tmp_path / "cache" / "empty"
        path.mkdir(parents=True)
        return path
    with pytest.raises(RuntimeError, match="metadata"):
        load_hf_bootstrap_module().bootstrap_models(tmp_path / "data", tmp_path / "cache", download)
```

The helper must load the script with `importlib.util` because `docker` is not a Python package. The fixture must contain only frozen configuration manifests and no private artifacts or network access.

- [ ] **Step 2: Run the focused tests and verify the expected RED failure**

Run: `uv run pytest tests/test_hf_model_init.py -q`

Expected: collection or test failure because `docker/hf-model-init.py` and `bootstrap_models` do not exist yet.

---

### Task 2: Implement the minimal frozen-model bootstrap

**Files:**
- Create: `docker/hf-model-init.py`
- Modify: `tests/test_hf_model_init.py`

**Interfaces:**
- `ModelSnapshot` is a frozen dataclass containing `model_id: str`, `revision: str`, and `path: Path`.
- `bootstrap_models(...)` returns the two validated snapshots in dense-then-reranker order.
- `main()` reads `KAWANEEN_DATA_DIRECTORY` and `HF_HOME`, imports `snapshot_download`, and exits nonzero on any error.

- [ ] **Step 1: Implement the smallest passing code**

Implement these exact rules:

```python
configuration = load_frozen_serving_configuration(data_directory)
models = (
    (configuration.dense_model_id, configuration.dense_model_revision),
    (configuration.reranker.model_id, configuration.reranker.model_revision),
)
for model_id, revision in models:
    path = Path(snapshot_downloader(repo_id=model_id, revision=revision, cache_dir=cache_directory, local_files_only=False))
    if not path.is_dir() or not any((path / name).is_file() for name in ("config.json", "config.yaml", "tokenizer.json", "tokenizer_config.json")):
        raise RuntimeError(f"frozen snapshot metadata is missing for {model_id} {revision}")
```

Use no model loading, no `HfApi`, no revision lookup, no cache deletion, and no `force_download`. Log only safe identity and `snapshot_ready`/failure status. Let downloader exceptions propagate so Compose marks the job failed.

- [ ] **Step 2: Run the focused tests and verify GREEN**

Run: `uv run pytest tests/test_hf_model_init.py -q`

Expected: all focused bootstrap tests pass without network access.

---

### Task 3: Wire the init job into Compose

**Files:**
- Modify: `compose.yaml`

**Interfaces:**
- `hf-model-init` uses the `docker/full/Dockerfile` image, `python /app/docker/hf-model-init.py`, `HF_HOME=/opt/huggingface`, and the external artifact/data paths already used by the full image.
- `kawaneen-api` adds `hf-model-init: condition: service_completed_successfully` while retaining qdrant-init, ollama-init, and healthy mlflow dependencies.

- [ ] **Step 1: Add the service and dependency**

Mount `${KAWANEEN_HOST_ARTIFACTS_DIR:-./artifacts}:/app/artifacts:ro` only where needed for the existing configuration boundary, mount `huggingface_cache:/opt/huggingface` on `hf-model-init`, and do not add dependencies from qdrant-init or ollama-init to the HF job.

- [ ] **Step 2: Validate the rendered Compose graph**

Run: `docker compose config`

Expected: valid YAML showing `hf-model-init`, the shared `huggingface_cache` volume, and the API completion dependency.

---

### Task 4: Verify the repair and full local acceptance

**Files:**
- Modify: `docs/deployment/full-local.md` only if the complete frozen stack succeeds, as a factual deployment record.

**Interfaces:**
- Runtime evidence must contain only safe statuses, IDs, revisions, request IDs, trace IDs, answerability, citation counts, and memory measurements.

- [ ] **Step 1: Recheck Docker and artifact preconditions**

Run `docker context show` and `docker info --format '{{.MemTotal}}'`. Validate the existing artifact root read-only, without copying or printing source text.

- [ ] **Step 2: Run the repair against the preserved named HF volume**

Run the canonical Compose command with `/Volumes/JOSEPH/projects/Kawaneen/artifacts`, wait for `hf-model-init` to exit 0, and inspect only exact snapshot metadata. Do not remove the volume first.

- [ ] **Step 3: Run full acceptance after API readiness**

Verify all service/init health conditions, frozen Ollama identity, `/v1/health`, `/v1/models`, real search, real answer through Stage-D and citation verification, deterministic extraction, MLflow hierarchy, UI health, and two safe `docker stats --no-stream` snapshots. Stop and report the exact failure if any container OOMs or the stack remains unhealthy.

- [ ] **Step 4: Run final checks and integrate only if applicable**

If the full stack succeeds, run the specified Ruff, Pyright, Make, Compose, and diff checks, update the deployment record and PR body, push the branch, and wait for exact-head CI. If acceptance fails, do not claim success, do not optimize frozen behavior, and do not push unrelated changes.

