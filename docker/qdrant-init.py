from __future__ import annotations

import os
from pathlib import Path

from qdrant_client import QdrantClient

from kawaneen.api.composition import load_frozen_serving_configuration
from kawaneen.retrieval.qdrant_bootstrap import load_qdrant_seed, seed_qdrant_collection


def main() -> None:
    data = Path(os.environ.get("KAWANEEN_DATA_DIRECTORY", "/app/data"))
    artifacts = Path(os.environ.get("KAWANEEN_ARTIFACTS_DIRECTORY", "/app/artifacts"))
    config = load_frozen_serving_configuration(data)
    seed = load_qdrant_seed(
        artifacts / "private" / "phase7_retrieval",
        expected_corpus_hash=config.corpus_hash,
        expected_model_revision=config.dense_model_revision,
        expected_dimension=1024,
    )
    collection = seed_qdrant_collection(
        QdrantClient(url=os.environ.get("KAWANEEN_QDRANT_URL", "http://qdrant:6333")),
        seed,
    )
    print(f"Qdrant collection ready: {collection}")


if __name__ == "__main__":
    main()
