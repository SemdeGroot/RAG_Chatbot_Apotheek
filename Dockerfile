FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# faiss-cpu wheel werkt zo; libgomp1 helpt soms bij BLAS
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# kopieer app + (kleine) vector DB mee in het image
COPY . /app

# jouw app kijkt hiernaar
ENV VECTORDB_DIR=/app/data/vectordb

# Spaces levert $PORT; bind daar op
CMD ["gunicorn", "-k", "gthread", "-w", "1", "-b", "0.0.0.0:${PORT}", "app:app"]
