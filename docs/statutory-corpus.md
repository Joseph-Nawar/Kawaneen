# Statutory corpus qualification

## Authority and provenance

The Saudi Ministry of Justice Legal Portal is the authoritative verification source for titles, article text, status, dates, amendments, and official URLs. Kawaneen must use the portal only for permitted manual verification and user-provided official URLs; Phase 2B does not scrape or bulk-download it.

The Bureau of Experts legislation portal and the Saudi National Platform rules catalogue are additional official manual reconciliation authorities. They may help verify titles, status, amendments, and official identifiers, but they do not authorize bulk scraping or automated acquisition.

Saudi Copyright Law Article 4 is relevant evidence for the underlying official-text layer: Saudi laws, judicial judgments, administrative decisions, international agreements, official documents, and official translations are excluded from Saudi copyright protection, subject to circulation provisions. This does not override circulation restrictions, portal terms, privacy obligations, automated-access rules, or public-display restrictions. The uploader’s derived-dataset licence remains separate from the official-text layer.

`WafaaFraih/saudi-legal-moj` is a derived local-research seed. Its dataset card declares CC BY 4.0, 3,185 Arabic article records, and 71 Saudi laws and regulations, and states the Saudi Ministry of Justice as the source. The uploader’s dataset licence is separate from rights in MOJ source text. The seed is therefore suitable for controlled private quality inspection and local parsing, not public display, training, redistribution, or legal authority.

## v1 scope and reconciliation

The proposed v1 statutory scope is 10–20 commercial/civil laws, beginning with the 12 named candidates in `data/manifests/reconciliation/core-commercial-civil-v1.csv`. Corpus quality and domain coverage matter more than the raw count of 71 laws. The current list is litigation-heavy and should be expanded or reprioritized to include core commercial instruments such as companies/business-formation and digital-commerce legislation when authoritative sources are identified. Each candidate requires a manually recorded official URL from an approved authority, status, publication date, latest amendment, article-count comparison, reviewed samples, and discrepancy decision. Blank fields mean not reviewed; they are not numeric zeroes.

The dataset currently has 3,185 rows and 71 law names, with 1,993 duplicate law/article keys. It is suitable as a raw parsing seed, but rows are not authoritative article units and the seed is unsuitable as a v1 gold corpus before reconstruction and reconciliation. Its statutory-quality summary records exact duplicate rows, duplicate law/article keys, missing fields, length anomalies, character checks, and availability of dates/status/amendment/official URL fields without storing article text.

An official MOJ machine-readable article-level export would replace this derived seed for any approved corpus. The requested export should include stable identifiers, title, article text, law type, issuing authority, publication/effective dates, status, amendments, source URL, and explicit terms for automated access, model processing, quotation, redistribution, and public demonstration.

## Use layers

- Real ALARB and ArabiCCR material remains private local research/evaluation material under their source-specific decisions.
- The MOJ-derived seed is private local parsing material pending reconciliation; its legal clearance remains false.
- Human-reviewed real-data evaluation is a separate gold layer.
- Semi-synthetic training pairs require a later training decision and provenance labels.
- Fictional statutes and synthetic safety/abstention cases are the only planned public-demo layer.

Synthetic content must never be presented as real law or replace final human-reviewed evaluation on real data.
