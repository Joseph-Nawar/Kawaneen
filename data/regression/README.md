# Phase 14 public regression cases

This directory contains only fictional, public synthetic behavior locks for the
Phase 14 test harness. The cases protect observable retrieval/answer/abstention
behavior; they are not model-quality scores and do not use Phase 8/10/11
HOLDOUT material.

To intentionally change a baseline: run `make test-regression`, review the
behavioral differences, update the case or lock explicitly, and document the
reason in the Phase 14 report. No test or command rewrites these files
automatically.
