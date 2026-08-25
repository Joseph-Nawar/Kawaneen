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
| UI helper/page tests | Included in public run: see final handoff run |
| Hermetic page/component render coverage | Passed |
| Streamlit AppTest | Passed, including demo visual-state safety tests |
| Ruff/Pyright on UI | Passed |
| Private integration smoke | Targeted test skipped as expected without `KAWANEEN_PRIVATE_PHASE12_API_URL` |
| `make check` | Passed: 832 passed, 1 skipped, 85.02% branch-aware coverage |
| Fresh PR CI | `pull_request` run on final head passed: Python 3.11 and 3.12 quality jobs |
| Browser tooling | Native `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` 151.0.7922.174; `agent-browser`/Playwright/Selenium unavailable |
| Browser viewport QA | Passed with Chrome headless at 1440×900, 1280×800, and 390×844; Arabic Search, English Search, grounded Ask, abstention Ask, Extract, and Evaluation checked |
| Final screenshots | Four committed 1440×900 synthetic screenshots; primary capture used Chrome `--headless=new --window-size=1440,900` with CDP `Page.captureScreenshot` after Streamlit hydration |
| Push/PR | Branch rebased onto `main`, force-pushed, and [PR #9](https://github.com/Joseph-Nawar/Kawaneen/pull/9) retargeted to `main`, not merged |

## Product follow-up

- Search filters only the returned result set by document ID/title and preserves API ranks.
- Ask inspection shows exact canonical-unit context, safe verified-quote highlighting, RTL direction, metadata, and conditional real source links.
- Extract corpus mode requests paginated document pages and shows visible bounds; findings expose summary counts, rule structure, exact deadline spans, regulated entities, exceptions, source provenance, and segment IDs.
- Evaluation shows current capability readiness, frozen Phase 8 architecture/configuration, common tracked retrieval comparisons/deltas, Phase 10 cards, Phase 11 summary/error taxonomy, separate Search/Answer/Extract latency, and collapsed technical hashes.
- Visual QA uses a demo-only `KAWANEEN_UI_VISUAL_QA` switch to seed populated Search, Ask, and Extract states from the existing `DemoClient` fixtures; it is ignored outside demo mode and ordinary demo behavior is unchanged when absent.
- The final visual pass compacted the extraction input and added a non-metric portfolio summary strip to keep the recruiter-facing states legible in the primary viewport.

## History and release gate

- Requested pre-rewrite head: `03cbb23fc9889d4a06f574a2b179541c7aef2161`.
- Final rebased head: recorded in the final handoff after the post-QA commit.
- `origin/main`: `f33a0448f4de8128962995d5bc3be538300c6162`.
- `git merge-base HEAD origin/main` equals `origin/main`.
- `origin/main`, old Phase 12, and the frozen Phase 12 tree share tree `38cb494ea859a443efe3bff0c6225486564b12b9`.
- Screenshots: `docs/assets/ui/search.png`, `docs/assets/ui/ask.png`, `docs/assets/ui/extract.png`, and `docs/assets/ui/evaluation.png`; all are synthetic portfolio demo data.
- PR remains unmerged; rendered QA and the four screenshot gate are complete, but merge remains explicitly prohibited.
- PR #9 is currently `CLEAN`/`MERGEABLE` against `main`; merge is still explicitly prohibited by the release gate.

## Safety confirmations

- Demo fixtures contain synthetic Arabic and English text only.
- No HOLDOUT content was accessed, no evaluation was rerun, no model was tuned, and no frozen Phase 7–12 result was altered.
- Evaluation snapshot sources are restricted to tracked Phase 8/10/11 evidence and record SHA-256 values.
- Hybrid extraction is labelled `PHASE11_HYBRID_EXPERIMENTAL_LIMITED`.
- Auto mode does not silently downgrade to demo mode.
