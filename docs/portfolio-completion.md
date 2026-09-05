# Portfolio completion checklist

This maps the original roadmap completion criteria to exact repository
evidence. `COMPLETE` means the artifact exists and is tracked or reproducibly
documented. The video remains a deliberate manual checkpoint until a real MP4
is supplied.

| Criterion | Status | Evidence |
| --- | --- | --- |
| Reproducible data acquisition | COMPLETE | `docs/data-acquisition.md`, `src/kawaneen/acquisition/`, `data/manifests/acquisition_lock.json` |
| Documented licenses | COMPLETE | `docs/data-governance.md`, `docs/data_sources.md`, source-audit records |
| Tested parsing | COMPLETE | `src/kawaneen/parsing/`, `tests/test_parsing_*.py`, `docs/parsing-and-ocr.md` |
| Arabic normalization experiments | COMPLETE | `docs/reports/phase-04-arabic-normalization-report.md`, `data/evaluation/phase4_*.json` |
| Structure-aware chunking | COMPLETE | `docs/reports/phase-05-legal-structure-and-chunking-report.md`, `data/evaluation/phase5_chunking_metrics.json` |
| Frozen retrieval evaluation set | COMPLETE | `data/manifests/evaluation/phase6_ai_reviewed_v1_manifest.json`, `data/evaluation/README.md` |
| BM25 and dense baselines | COMPLETE | `data/evaluation/phase7_baseline_comparison.json`, `docs/reports/phase-15-evaluation-and-experiment-report.md` |
| Hybrid retrieval | COMPLETE | `data/manifests/retrieval/phase8_final_manifest.json`, `src/kawaneen/retrieval/hybrid/` |
| Reranking | COMPLETE | `data/manifests/retrieval/phase8_model_lock.json`, `tests/test_retrieval_hybrid_reranker.py` |
| Citation verification | COMPLETE | `docs/reports/phase-15-evaluation-and-experiment-report.md`, `src/kawaneen/grounding/verification.py` |
| Abstention evaluation | COMPLETE | `data/evaluation/phase15_abstention_sensitivity.json`, Phase 10 policy and tests |
| Structured extraction | COMPLETE | `docs/phase11-extraction.md`, `data/manifests/extraction/phase11_selected_configuration_v1.json` |
| FastAPI | COMPLETE | `docs/api.md`, `src/kawaneen/api/`, `tests/test_api_*.py` |
| Streamlit | COMPLETE | `src/kawaneen/ui/`, `docs/reports/phase-13-ui-report.md`, screenshots under `docs/assets/ui/` |
| Docker Compose | COMPLETE | `compose.yaml`, `docs/deployment/full-local.md`, `make test-e2e` |
| CI | COMPLETE | `.github/workflows/ci.yml` |
| Unit and integration tests | COMPLETE | `tests/`, `docs/testing.md`, `make check` |
| MLflow experiment records | COMPLETE | `docs/reports/phase-16-observability-and-reproducibility.md`, `docs/mlflow-evidence.md`, `data/evaluation/portfolio_mlflow_evidence.json` |
| Error analysis | COMPLETE | `data/evaluation/phase15_error_analysis.json`, Phase 15 report |
| Model card | COMPLETE | `docs/model-card.md` |
| Dataset card | COMPLETE | `docs/dataset-card.md` |
| Limitations and safety documentation | COMPLETE | `docs/safety-and-limitations.md` |
| Demo video | MANUAL VIDEO PENDING | `docs/demo/three-minute-script.md`, `docs/demo/shot-list.md`; target `docs/demo/kawaneen-demo.mp4` does not yet exist |
| Professional README | COMPLETE | `README.md` |

The project must not be called fully portfolio-complete until the real video
exists, is inspected, and is linked from the README.

## Optional future action

Publish the already-qualified synthetic demo to Hugging Face if the user's
account supports the desired no-cost configuration. No Space is published by
this closeout.
