FROM python:3.11-slim

WORKDIR /app

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev libgomp1 libstdc++6 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Installazione Optimum e Torch CPU
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download del modello di reranking multilingua
COPY ./model_cache /app/model_cache/

COPY src/. .

ENV PYTHONUNBUFFERED=1
ENV FASTEMBED_CACHE_PATH=/app/model_cache

EXPOSE 5000

CMD ["gunicorn", "--worker-class", "gevent", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]