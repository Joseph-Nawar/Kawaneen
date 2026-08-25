FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

COPY docker/e2e/requirements.txt /app/docker/e2e/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/docker/e2e/requirements.txt

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY tests/fixtures/phase14 /app/tests/fixtures/phase14
COPY tests/phase14_support.py /app/tests/phase14_support.py
COPY tests/e2e /app/tests/e2e
RUN python -m pip install --no-cache-dir --no-deps .
