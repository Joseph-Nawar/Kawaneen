from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import cast

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

QUALIFICATION_LIMITS = {
    "peak_rss_mb": 10_000,
    "image_size_bytes": 8_000_000_000,
    "startup_time_ms": 120_000,
    "search_p95_ms": 8_000,
    "answer_p95_ms": 10_000,
}


def qualifies_resource_run(report: dict[str, object]) -> bool:
    digest = report.get("container_image_digest")
    memory_value = report.get("memory_mb")
    timings_value = report.get("timings_ms")
    if not isinstance(digest, str) or not digest:
        return False
    if not isinstance(report.get("image_size_bytes"), int):
        return False
    if not isinstance(report.get("startup_time_ms"), (int, float)):
        return False
    if not isinstance(memory_value, dict) or not isinstance(timings_value, dict):
        return False
    memory = cast(dict[str, object], memory_value)
    timings = cast(dict[str, object], timings_value)
    required = (
        memory.get("idle_rss"),
        memory.get("peak_rss"),
        timings.get("search_p50"),
        timings.get("search_p95"),
        timings.get("answer_p50"),
        timings.get("answer_p95"),
    )
    if any(not isinstance(value, (int, float)) for value in required):
        return False
    image_size = report.get("image_size_bytes")
    startup = report.get("startup_time_ms")
    peak_rss = memory.get("peak_rss")
    search_p95 = timings.get("search_p95")
    answer_p95 = timings.get("answer_p95")
    if not isinstance(image_size, (int, float)):
        return False
    if not isinstance(startup, (int, float)):
        return False
    if not isinstance(peak_rss, (int, float)):
        return False
    if not isinstance(search_p95, (int, float)):
        return False
    if not isinstance(answer_p95, (int, float)):
        return False
    return (
        image_size < QUALIFICATION_LIMITS["image_size_bytes"]
        and startup < QUALIFICATION_LIMITS["startup_time_ms"]
        and peak_rss < QUALIFICATION_LIMITS["peak_rss_mb"]
        and search_p95 < QUALIFICATION_LIMITS["search_p95_ms"]
        and answer_p95 < QUALIFICATION_LIMITS["answer_p95_ms"]
        and report.get("fixed_query_errors") == 0
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
        "container_platform": "linux/arm64",
        "startup_time_ms": None,
        "container_image_id": None,
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


def _write_report(output: Path, report: dict[str, object]) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _image_metadata(image: str) -> tuple[str, int, str]:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}|{{.Size}}|{{.Architecture}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    image_id, size, architecture = result.stdout.strip().split("|", 2)
    return image_id, int(size), architecture


def _container_memory_mb(container_id: str, filename: str) -> float | None:
    result = subprocess.run(
        ["docker", "exec", container_id, "sh", "-c", f"cat /sys/fs/cgroup/{filename} 2>/dev/null"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    try:
        return round(int(result.stdout.strip()) / (1024 * 1024), 3)
    except (TypeError, ValueError):
        return None


def _container_requests(container_id: str) -> tuple[list[float], list[float], int]:
    code = "\n".join(
        (
            "import json",
            "import sys",
            "import time",
            "import urllib.request",
            "queries = json.loads(sys.argv[1])",
            "search = []",
            "answer = []",
            "errors = 0",
            "for query in queries:",
            "    start = time.perf_counter()",
            "    try:",
            "        request = urllib.request.Request(",
            "            'http://127.0.0.1:8000/v1/search',",
            "            data=json.dumps({'query': query, 'limit': 5}, "
            "ensure_ascii=False).encode(),",
            "            headers={'content-type': 'application/json'},",
            "        )",
            "        with urllib.request.urlopen(request, timeout=30) as response:",
            "            errors += response.status != 200",
            "    except Exception:",
            "        errors += 1",
            "    search.append((time.perf_counter() - start) * 1000)",
            "    start = time.perf_counter()",
            "    try:",
            "        request = urllib.request.Request(",
            "            'http://127.0.0.1:8000/v1/answer',",
            "            data=json.dumps({'query': query}, ensure_ascii=False).encode(),",
            "            headers={'content-type': 'application/json'},",
            "        )",
            "        with urllib.request.urlopen(request, timeout=30) as response:",
            "            errors += response.status != 200",
            "    except Exception:",
            "        errors += 1",
            "    answer.append((time.perf_counter() - start) * 1000)",
            "print(json.dumps({'search': search, 'answer': answer, 'errors': errors}))",
        )
    )
    result = subprocess.run(
        ["docker", "exec", container_id, "python", "-c", code, json.dumps(FIXED_QUERIES)],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    payload = json.loads(result.stdout)
    return (
        [float(value) for value in payload["search"]],
        [float(value) for value in payload["answer"]],
        int(payload["errors"]),
    )


def run_constrained_container(
    image: str,
    *,
    output: Path = Path("data/evaluation/phase17_demo_qualification.json"),
) -> dict[str, object]:
    """Measure the final Space image under the approved native container limits."""

    report = empty_report()
    report["container_image_digest"] = None
    container_id: str | None = None
    try:
        image_id, image_size, image_architecture = _image_metadata(image)
        report["container_image_id"] = image_id
        report["container_image_digest"] = image_id
        report["image_size_bytes"] = image_size
        report["architecture"] = image_architecture
        start = time.perf_counter()
        launched = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--cpus=2",
                "--memory=12g",
                "--network=none",
                "--platform=linux/arm64",
                image,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        container_id = launched.stdout.strip()
        deadline = start + 150
        while time.perf_counter() < deadline:
            health = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_id,
                    "python",
                    "-c",
                    "import urllib.request; "
                    "urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3)",
                ],
                check=False,
                capture_output=True,
                timeout=10,
            )
            if health.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Space container did not become healthy within 150 seconds")
        report["startup_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
        report["memory_mb"] = {
            "idle_rss": _container_memory_mb(container_id, "memory.current"),
            "peak_rss": None,
        }
        search_times, answer_times, errors = _container_requests(container_id)

        def p95(values: list[float]) -> float:
            return round(sorted(values)[max(0, int(len(values) * 0.95) - 1)], 3)

        timings = cast(dict[str, object], report["timings_ms"])
        timings.update(
            {
                "search_p50": round(median(search_times), 3),
                "search_p95": p95(search_times),
                "answer_p50": round(median(answer_times), 3),
                "answer_p95": p95(answer_times),
            }
        )
        memory = cast(dict[str, object], report["memory_mb"])
        memory["peak_rss"] = _container_memory_mb(container_id, "memory.peak")
        report["fixed_query_errors"] = errors
        report["qualification_decision"] = (
            "HF_SPACE_RESOURCE_QUALIFIED"
            if qualifies_resource_run(report)
            else "HF_SPACE_RESOURCE_NOT_QUALIFIED"
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as error:
        report["qualification_note"] = (
            "Constrained container qualification failed before all measurements were collected: "
            f"{type(error).__name__}. "
            "Measurements are local and are not Hugging Face host performance claims."
        )
        report["qualification_decision"] = "HF_SPACE_RESOURCE_NOT_QUALIFIED"
    finally:
        if container_id:
            subprocess.run(["docker", "rm", "-f", container_id], check=False, capture_output=True)
    return _write_report(output, report)


def run_qualification(
    output: Path = Path("data/evaluation/phase17_demo_qualification.json"),
) -> dict[str, object]:
    image = os.environ.get("KAWANEEN_SPACE_IMAGE", "kawaneen-phase17-space:local")
    return run_constrained_container(image, output=output)


if __name__ == "__main__":
    print(json.dumps(run_qualification(), ensure_ascii=False, indent=2))
