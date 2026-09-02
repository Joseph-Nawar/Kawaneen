from __future__ import annotations

import json
import platform
import resource
import time
from pathlib import Path
from statistics import median
from typing import Any, cast

FIXED_QUERIES = (
    "ما هي مدة الإرجاع؟",
    "ما هي مدة إشعار العمل؟",
    "متى تصدر الفاتورة؟",
    "ما مدة إشعار العقد؟",
    "متى يقدم الطلب؟",
    "ما مدة الضمان؟",
    "متى يتم التسليم؟",
    "كيف يقدم الاعتراض؟",
    "ما مهلة الرد؟",
    "متى يبدأ الموعد؟",
    "هل يجوز تمديد مدة الإرجاع؟",
    "كيف يصحح خطأ الفاتورة؟",
    "ما أثر الإشعار؟",
    "متى ينتهي الضمان؟",
    "كيف يثبت التسليم؟",
    "ما شروط الاعتراض؟",
    "ما مدة الإصلاح؟",
    "هل يقبل الطلب المتأخر؟",
    "متى يصدر القرار؟",
    "ما كلفة الإرجاع؟",
)


def empty_report() -> dict[str, object]:
    return {
        "schema": "phase17-demo-qualification-v1",
        "provenance": "PHASE17_DEV",
        "qualification_scope": "local_constrained_not_huggingface_host",
        "publication_status": "NOT_PUBLISHED_USER_APPROVAL_REQUIRED",
        "architecture": platform.machine(),
        "operating_system": platform.platform(),
        "cpu_limit": "2 CPUs requested when containerized",
        "memory_limit": "12 GB requested when containerized",
        "container_image_digest": None,
        "model": {
            "id": "intfloat/multilingual-e5-small",
            "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        },
        "reranker": {
            "enabled": True,
            "decision": "DEMO_RERANKER_ENABLED_TOP4",
            "model_id": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "depth": 4,
        },
        "timings_ms": {
            "search_p50": None,
            "search_p95": None,
            "answer_p50": None,
            "answer_p95": None,
        },
        "memory_mb": {"idle_rss": None, "peak_rss": None},
        "image_size_bytes": None,
        "fixed_query_count": 20,
        "fixed_query_errors": 0,
        "qualification_decision": "HF_SPACE_RESOURCE_NOT_QUALIFIED",
        "qualification_note": (
            "Measurements are local and are not Hugging Face host performance claims."
        ),
    }


def run_qualification(
    output: Path = Path("data/evaluation/phase17_demo_qualification.json"),
) -> dict[str, object]:
    from fastapi.testclient import TestClient

    from kawaneen.demo.runtime import create_demo_app

    report = empty_report()
    search_times: list[float] = []
    answer_times: list[float] = []
    errors = 0
    client: Any
    with TestClient(create_demo_app(request_rate_limit=len(FIXED_QUERIES) * 2)) as client:
        for query in FIXED_QUERIES:
            start = time.perf_counter()
            search = client.post("/v1/search", json={"query": query, "limit": 5})
            search_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            answer = client.post("/v1/answer", json={"query": query})
            answer_times.append((time.perf_counter() - start) * 1000)
            errors += int(search.status_code != 200 or answer.status_code != 200)

    def p95(values: list[float]) -> float | None:
        return round(sorted(values)[max(0, int(len(values) * 0.95) - 1)], 3) if values else None

    timings = cast(dict[str, object], report["timings_ms"])
    timings.update(
        {
            "search_p50": round(median(search_times), 3),
            "search_p95": p95(search_times),
            "answer_p50": round(median(answer_times), 3),
            "answer_p95": p95(answer_times),
        }
    )
    report["fixed_query_errors"] = errors
    report["memory_mb"] = {
        "idle_rss": None,
        "peak_rss": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 3),
    }
    report["qualification_decision"] = (
        "HF_SPACE_RESOURCE_NOT_QUALIFIED" if errors else "HF_SPACE_RESOURCE_QUALIFIED"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_qualification(), ensure_ascii=False, indent=2))
