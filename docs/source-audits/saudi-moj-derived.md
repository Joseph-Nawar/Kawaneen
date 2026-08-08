# Saudi Legal MOJ Dataset — source audit

## Canonical identity

- Dataset: `WafaaFraih/saudi-legal-moj`
- Canonical dataset card: <https://huggingface.co/datasets/WafaaFraih/saudi-legal-moj>
- Pinned revision: `8b55ef5a666ad773c81086051813582fd14eb466`
- Verification date: 2026-08-07

## Pages and evidence inspected

The Hugging Face dataset card was inspected. It states 3,185 Arabic article records, 71 Saudi laws and regulations, Parquet format, CC BY 4.0, and Saudi Ministry of Justice as the stated source. The repository tree identifies `README.md` and `data/train-00000-of-00001.parquet`.

## What the evidence proves

The card supports the uploader’s declared dataset licence, the claimed dataset size and format, and the uploader’s description of provenance. The pinned revision makes the selected files reproducible.

## What it does not prove

CC BY 4.0 on the derived dataset does not prove that the uploader had authority to relicense or redistribute the underlying Ministry of Justice text. It does not establish public display, model training, quotation, automated processing, or replacement of the dataset with an official MOJ export. Those rights remain unresolved.

Saudi Copyright Law Article 4 is a separate underlying-official-text evidence item: it excludes Saudi laws, judicial judgments, administrative decisions, international agreements, official documents, and official translations from copyright protection, subject to circulation provisions. It does not grant blanket permission to scrape the MOJ portal. Circulation restrictions, portal terms, privacy, automated access, public display, and the uploader’s compilation licence remain separate review layers.

## Technical and privacy assessment

The seed is an article-level Parquet dataset with required fields for article text, article label, law name, and law type. Quality metrics are recorded in the sanitized statutory summary; raw article text is never copied into committed output. Automated privacy screening remains a signal only and does not establish legal clearance. The 1,993 duplicate law/article keys mean it is a raw parsing seed, not an authoritative or gold article corpus before Phase 3 reconstruction and reconciliation.

## Decision

`local_research_only`. Private local parsing and quality inspection are permitted after acquisition. Training, publication, public display, public demonstration, and redistribution remain denied. An official MOJ article-level export and source-rights clarification are required before the seed can support an approved public or training corpus.
