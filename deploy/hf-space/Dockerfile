FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    KAWANEEN_UI_MODE=live \
    KAWANEEN_UI_PUBLIC_DEMO=true \
    KAWANEEN_API_URL=http://127.0.0.1:8000 \
    HF_HOME=/opt/huggingface

WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY uv.lock /app/uv.lock
COPY src /app/src
COPY data/demo /app/data/demo
COPY deploy/hf-space/entrypoint.sh /app/entrypoint.sh
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN python -m pip install --no-cache-dir "transformers==4.57.6" "sentence-transformers==5.7.0"
RUN python -m pip install --no-cache-dir ".[retrieval,ui]"
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3', device='cpu'); CrossEncoder('BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', device='cpu', max_length=512)"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
