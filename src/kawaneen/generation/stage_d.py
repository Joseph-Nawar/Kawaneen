"""Phase-10 Stage-D experiment identity and fingerprinting."""

from __future__ import annotations

from kawaneen.generation.answerability import ANSWERABILITY_POLICY_VERSION
from kawaneen.generation.artifacts import artifact_fingerprint
from kawaneen.generation.contracts import STAGE_D_GENERATION_SETTINGS
from kawaneen.generation.quote_registry import QuoteRegistry
from kawaneen.grounding.contracts import ContextPack

STAGE_D_GENERATOR_NAME = "qwen-ollama-stage-d"
STAGE_D_TIMEOUT_SECONDS = 60.0
STAGE_D_QUOTE_REGISTRY_POLICY_VERSION = "phase10-stage-d-quote-registry-v1"


def stage_d_fingerprint(
    *,
    query_id: str,
    context_pack: ContextPack,
    registry: QuoteRegistry,
    model_revision: str,
    ollama_digest: str,
    tokenizer_identity: str,
    tokenizer_revision: str | None,
    prompt_hash: str,
    schema_hash: str,
    policy_hash: str,
    answerability_policy_hash: str,
) -> str:
    return artifact_fingerprint(
        {
            "experiment_id": STAGE_D_GENERATOR_NAME,
            "query_id": query_id,
            "phase8_selection_sha256": context_pack.phase8_selection_sha256,
            "ordered_input_chunk_ids": list(context_pack.input_chunk_ids),
            "context_pack_hash": artifact_fingerprint(context_pack.model_dump(mode="json")),
            "quote_registry_policy_version": registry.policy_version,
            "quote_registry_hash": registry.fingerprint,
            "phase9_assembly_policy": context_pack.assembly_policy_version,
            "answerability_policy_version": ANSWERABILITY_POLICY_VERSION,
            "answerability_policy_hash": answerability_policy_hash,
            "model_revision": model_revision,
            "ollama_digest": ollama_digest,
            "tokenizer_identity": tokenizer_identity,
            "tokenizer_revision": tokenizer_revision,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "generation_policy_hash": policy_hash,
            "timeout_seconds": STAGE_D_TIMEOUT_SECONDS,
            "settings": STAGE_D_GENERATION_SETTINGS.model_dump(mode="json"),
        }
    )
