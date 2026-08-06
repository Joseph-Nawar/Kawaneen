# Project Charter

## Purpose

Kawaneen will become a jurisdiction-aware Arabic legal and regulatory intelligence system. Its long-term purpose is to help people organize, understand, and trace legal and regulatory material with explicit jurisdiction and source context.

## Phase 0 boundary

This phase establishes a safe, installable Python foundation only: package identity, validated local settings, structured logging, a diagnostic CLI, quality automation, documentation, and data safeguards. No legal corpus, legal advice, document parsing, NLP, search, retrieval, RAG, API, UI, model, or vector database is included.

## Principles

- Treat jurisdiction, provenance, and language support as first-class future design concerns.
- Keep local development reproducible and free of paid services, secrets, and network-dependent tests.
- Prefer explicit configuration and observable behavior over hidden magic.
- Document planned capabilities as planned until they are implemented and verified.
