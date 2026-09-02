FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    KAWANEEN_UI_MODE=live \
    KAWANEEN_UI_PUBLIC_DEMO=true \
    KAWANEEN_API_URL=http://127.0.0.1:8000 \
    HF_HOME=/data/huggingface

WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY uv.lock /app/uv.lock
COPY src /app/src
COPY data/demo /app/data/demo
COPY deploy/hf-space/entrypoint.sh /app/entrypoint.sh
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN python -m pip install --no-cache-dir "transformers==4.57.6" "sentence-transformers==5.7.0"
RUN python -m pip install --no-cache-dir ".[retrieval,ui]"

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
