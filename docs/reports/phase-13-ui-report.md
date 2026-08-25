# Phase 13 UI Report

## Scope

Phase 13 adds the recruiter-facing Streamlit interface over the Phase 12 `/v1` HTTP API. Demo fixtures are synthetic and the interface never imports model-serving, retrieval, generation, extraction, or corpus runtime services.

## Architecture and files

- `src/kawaneen/ui/app.py`: Streamlit page registration with `st.Page` and top navigation.
- `src/kawaneen/ui/client.py`, `demo.py`: typed HTTP boundary and synthetic demo client.
- `src/kawaneen/ui/config.py`, `state.py`: explicit live/demo/degraded mode and session state.
- `src/kawaneen/ui/formatting.py`, `uploads.py`, `exports.py`: safe markup, RTL, quote location, upload limits, segmentation, and downloads.
- `src/kawaneen/ui/evaluation.py`: allowlisted tracked-source snapshot and session latency aggregation.
- `src/kawaneen/ui/components.py`, `styles.py`, `pages/`: shared visual system and four screens.
- `data/manifests/ui/phase13_evaluation_snapshot.json`: sanitized metric snapshot with source SHA-256 values.

## Verification record

| Gate | Result |
|---|---|
| UI unit tests | 39 passed |
| Hermetic page/component render coverage | 5 passed |
| Streamlit AppTest | 4 page/interactions passed |
| Ruff/Pyright on UI | Passed |
| Private integration smoke | Targeted test skipped as expected without `KAWANEEN_PRIVATE_PHASE12_API_URL` |
| `make check` | Passed: 824 passed, 1 skipped, 85.01% branch-aware coverage |
| Browser viewport QA | Blocked: no browser backend is available in this environment |
| Final screenshots | Blocked by browser availability; none fabricated |
| Push/PR | Pending browser gate and final verification |

## Safety confirmations

- Demo fixtures contain synthetic Arabic and English text only.
- No HOLDOUT content was accessed, no evaluation was rerun, no model was tuned, and no frozen Phase 7–12 result was altered.
- Evaluation snapshot sources are restricted to tracked Phase 8/10/11 evidence and record SHA-256 values.
- Hybrid extraction is labelled `PHASE11_HYBRID_EXPERIMENTAL_LIMITED`.
- Auto mode does not silently downgrade to demo mode.
