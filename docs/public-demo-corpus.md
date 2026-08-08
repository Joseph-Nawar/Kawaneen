# Public-demo corpus gate

Phase 2 does not approve a public-demo corpus. A dataset licence alone does not resolve privacy, original court-record rights, attribution, or display terms.

ALARB, ArabiCCR, and the Saudi MOJ-derived seed are acquired only under source-specific private gates. ALARB and ArabiCCR are eligible for controlled local parsing/evaluation; the MOJ-derived seed is eligible only as a raw parsing seed and is not evaluation-approved. None is eligible for training or public display under the current policy.

The next-stage public-demo candidate must be a separately licensed, text-bearing collection whose dataset rights, original-source rights, privacy mitigation, automated-access terms, attribution, quoting, and display permissions are documented for the intended jurisdiction and use. Aggregate government statistics are not legal-text corpora and cannot substitute for that corpus. Phase 2 therefore leaves public-demo corpus selection blocked pending a dedicated licence review or a separately commissioned/licensed demo corpus.

## Hybrid corpus strategy

The planned corpus is explicitly hybrid, with provenance labels attached to every document or pair:

1. **Real local research:** ALARB and ArabiCCR remain private, source-versioned research material. They may be inspected locally only under their source-specific gates.
2. **Human-reviewed gold evaluation:** a small, separately permissioned set of real legal documents and questions will be reviewed by qualified humans, with document-level train/test separation and a recorded adjudication protocol.
3. **Semi-synthetic training pairs:** synthetic queries and passages derived from approved, non-public local material may support retriever/reranker development only after a separate training authorization. They must be labelled synthetic-derived and must not be treated as source documents.
4. **Fictional public-demo statutes:** the public demo will use explicitly fictional statutes and fictional fact patterns, clearly labelled as demonstrations rather than law.
5. **Synthetic safety and abstention tests:** generated adversarial cases will test refusal, uncertainty, jurisdiction mismatch, and privacy-safe behaviour; they are not legal authorities.

Synthetic material cannot replace final human-reviewed evaluation on real data. Synthetic data may not be represented as real law, an official legal source, or evidence that a model is legally accurate. Train/test document separation and provenance labels are mandatory across all five groups.
