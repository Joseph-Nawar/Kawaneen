# Data manifests

This directory contains version-controlled metadata only. `source_registry.csv` is a governance registry, not a dataset and not an ingestion input.

The registry records evidence and decisions for candidate legal sources. It intentionally contains no court text, downloaded records, credentials, or source payloads. A record may be used operationally only after its decision and required manual actions are resolved under [data governance](../../docs/data-governance.md).

The CSV contains technical/provenance fields (`publisher`, `original_publisher`, `task`, `language`, `size`, `size_unit`, `file_format`, `content_unit`, and `citation`), quality/privacy/access fields, and separate paper, code, dataset, original-source, automated-access, public-display, model-training, and public-demo rights. `yes` requires explicit permission evidence; `conditional` requires conditions.
