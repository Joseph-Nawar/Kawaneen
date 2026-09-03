from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kawaneen.api.composition import load_frozen_serving_configuration
from kawaneen.core.config import Settings
from kawaneen.phase15.inputs import Phase15InputRoots, load_dev_query_records
from kawaneen.retrieval.dense_models import BGEM3Adapter
from kawaneen.retrieval.qdrant_bootstrap import (
    collection_name_for,
    load_qdrant_seed,
    seed_qdrant_collection,
)
from kawaneen.retrieval.qdrant_index import QdrantExactIndex
from kawaneen.retrieval.qdrant_parity import compare_dense_indexes, select_dev_query_records
from kawaneen.retrieval.vector_index import NumpyExactIndex


def main() -> None:
    """Run exact parity over a stable sample of the existing Phase 8 DEV queries."""
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
    client = QdrantClient(url=settings.qdrant_url)
    seed_qdrant_collection(client, seed)
    qdrant_index = QdrantExactIndex.build(
        client=client,
        collection_name=collection_name_for(configuration.corpus_hash),
        vectors=seed.vectors,
        chunk_ids=seed.chunk_ids,
    )
    roots = Phase15InputRoots(
        historical_private_root=settings.artifacts_directory / "private",
        output_root=Path("."),
    )
    selected = select_dev_query_records(load_dev_query_records(roots), sample_count=20)
    adapter = BGEM3Adapter(revision=configuration.dense_model_revision)
    vectors = adapter.encode_queries(
        tuple(str(record["query_text"]) for record in selected), batch_size=1
    )
    result = compare_dense_indexes(
        numpy_index=numpy_index,
        qdrant_index=qdrant_index,
        queries=tuple(
            (str(record["query_id"]), vector)
            for record, vector in zip(selected, vectors, strict=True)
        ),
        sample_count=20,
        top_k=50,
    )
    result.update(
        {
            "corpus_hash": configuration.corpus_hash,
            "bge_model_id": adapter.model_id,
            "bge_model_revision": adapter.revision,
            "qdrant_exact": True,
            "selection_rule": "sha256(query_id) ascending over split=dev, first 20",
            "query_id_hash": hashlib.sha256(
                "\n".join(str(record["query_id"]) for record in selected).encode("utf-8")
            ).hexdigest(),
        }
    )
    output = Path("data/evaluation/phase17_qdrant_parity.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
