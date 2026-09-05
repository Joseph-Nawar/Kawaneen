# Kawaneen documentation

This index is the short route through a deliberately detailed repository.
Historical phase reports remain intact for provenance; they are not the
recommended landing page.

## Start here

- [Architecture](architecture/phase17-deployment.mmd) — full-local and public-demo profiles.
- [Evaluation](reports/phase-15-evaluation-and-experiment-report.md) — measured DEV evidence, negative results, and error analysis.
- [Model card](model-card.md) — system/component identities and frozen contracts.
- [Dataset card](dataset-card.md) — private, frozen-evaluation, and synthetic-demo data governance.
- [Safety and limitations](safety-and-limitations.md) — canonical public safety boundary.
- [Full local deployment](deployment/full-local.md) — private-artifact local profile.
- [Public demo deployment](deployment/public-demo.md) — synthetic, reduced, unpublished profile.
- [Demo script](demo/three-minute-script.md) and [shot list](demo/shot-list.md) — manual recording plan.
- [Portfolio completion](portfolio-completion.md) — roadmap-to-artifact closeout map.

## Deep technical record

- **Data pipeline:** [acquisition](data-acquisition.md), [governance](data-governance.md), [canonical corpus](canonical-corpus.md), [parsing/OCR](parsing-and-ocr.md).
- **Retrieval pipeline:** [architecture](architecture.md), [normalization](reports/phase-04-arabic-normalization-report.md), [chunking](reports/phase-05-legal-structure-and-chunking-report.md), and [Phase 16 reproducibility](reports/phase-16-observability-and-reproducibility.md).
- **Serving:** [API guide](api.md), [API examples](deployment/api-examples.md), [testing guide](testing.md), and [development](development.md).
- **Extraction and grounding:** [Phase 11 extraction](phase11-extraction.md) and [Phase 9/10 reports](reports/phase11-final-selection.md).
- **Phase reports and ADRs:** [historical reports](reports/) and [architecture decisions](adr/).

## Public-data rule

Tracked `data/manifests/` and `data/evaluation/` files contain metadata,
hashes, and aggregate results. Raw legal text, private corpora, model caches,
per-example evidence, and local MLflow state remain outside version control.
