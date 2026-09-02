FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    HF_HOME=/opt/huggingface

WORKDIR /app

COPY docker/full/requirements.txt /app/docker/full/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/docker/full/requirements.txt

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY data /app/data
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN python -m pip install --no-cache-dir "transformers==4.57.6" "sentence-transformers==5.7.0"
RUN python -m pip install --no-cache-dir ".[retrieval,ui]"

COPY docker /app/docker
