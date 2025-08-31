# ... je bestaande basis
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Zet alle caches naar een eigen pad in het image
ENV XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
RUN mkdir -p $XDG_CACHE_HOME $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $SENTENCE_TRANSFORMERS_HOME

# Kopieer code
COPY . /app

# Pre-download (prewarm) embeddingmodel tijdens de build
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/multilingual-e5-base")  # of jouw model
PY

# Start je app
CMD ["sh", "-c", "gunicorn -k gthread -w 1 -b 0.0.0.0:${PORT:-7860} app:app"]

