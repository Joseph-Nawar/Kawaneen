# ADR 0005: Gated acquisition and immutable raw data

## Status

Accepted for Phase 2.

## Decision

Acquisition is authorized by the Phase 1 source registry and a version-controlled source specification. Only explicitly permitted source-purpose pairs can acquire or import local files. Raw bytes are namespaced by source and version, copied through partial files, atomically installed, and never modified in place. Integrity, duplicate, and privacy inspection produce derived metadata and masked private review bundles without rewriting raw data.

The implementation uses the official Hugging Face client for the pinned ALARB revision. ArabiCCR uses a safe local-file import fallback until a stable documented official API route is verified. No generic bypass option exists.

## Consequences

The workflow is fail-closed and reproducible, but manual legal and privacy review remains necessary. Raw data stays outside Git; sanitized hashes, schemas, counts, and review statuses can be version controlled.
