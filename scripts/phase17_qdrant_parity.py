from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from kawaneen.api.composition import load_frozen_serving_configuration
from kawaneen.core.config import Settings
from kawaneen.retrieval.qdrant_bootstrap import collection_name_for, load_qdrant_seed
from kawaneen.retrieval.qdrant_index import QdrantExactIndex
from kawaneen.retrieval.qdrant_parity import compare_dense_indexes
from kawaneen.retrieval.vector_index import NumpyExactIndex


def main() -> None:
    """Run parity with an explicit precomputed DEV query-vector file.

    The query vectors are intentionally supplied out-of-band because the
    tracked repository contains no private query text. The caller must point
    this command at a DEV-only .npy matrix and matching stable ID JSON.
    """
    vector_path = os.environ.get("KAWANEEN_PHASE17_DEV_QUERY_VECTORS")
    ids_path = os.environ.get("KAWANEEN_PHASE17_DEV_QUERY_IDS")
    if not vector_path or not ids_path:
        raise SystemExit(
            "set KAWANEEN_PHASE17_DEV_QUERY_VECTORS and "
            "KAWANEEN_PHASE17_DEV_QUERY_IDS to DEV-only local inputs"
        )
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise SystemExit("install the full deployment dependencies to run Qdrant parity") from error
    settings = Settings()
    configuration = load_frozen_serving_configuration(settings.data_directory)
    private = settings.artifacts_directory / "private" / "phase7_retrieval"
    seed = load_qdrant_seed(
        private,
        expected_corpus_hash=configuration.corpus_hash,
        expected_model_revision=configuration.dense_model_revision,
    )
    numpy_index = NumpyExactIndex.build(seed.vectors, seed.chunk_ids)
    qdrant_index = QdrantExactIndex.build(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=collection_name_for(configuration.corpus_hash),
        vectors=seed.vectors,
        chunk_ids=seed.chunk_ids,
    )
    queries = np.asarray(np.load(Path(vector_path), allow_pickle=False), dtype=np.float32)
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    if not isinstance(ids, list) or len(ids) != len(queries):
        raise SystemExit("DEV query vector and ID counts do not match")
    result = compare_dense_indexes(
        numpy_index=numpy_index,
        qdrant_index=qdrant_index,
        queries=tuple(zip((str(item) for item in ids), queries, strict=True)),
    )
    result.update(
        {
            "corpus_hash": configuration.corpus_hash,
            "model_id": configuration.dense_model_id,
            "model_revision": configuration.dense_model_revision,
        }
    )
    output = Path("data/evaluation/phase17_qdrant_parity.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
