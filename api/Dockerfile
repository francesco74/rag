FROM python:3.11-slim

WORKDIR /app

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev libgomp1 libstdc++6 curl vim procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Modelli locali (reranking ONNX mmarco, embedding sentence-transformers): solo con
# --build-arg MODELLI_LOCALI=1 (deploy/immagini.sh lo passa con MODELLI_LOCALI=1). Senza,
# l'immagine è più leggera e reranking ed embedding vanno su un server (RERANK_PROVIDER=vllm,
# EMBEDDING_PROVIDER=vllm o un fornitore esterno).
ARG MODELLI_LOCALI=0

COPY api/requirements.txt api/requirements-locali.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$MODELLI_LOCALI" = "1" ]; then pip install --no-cache-dir -r requirements-locali.txt; fi

# Modello di reranking, preparato con api/build_fast_model.py (non è nel repository):
# la cartella api/model_cache deve esistere, anche vuota, quando non si copia.
RUN --mount=type=bind,source=api/model_cache,target=/tmp/model_cache \
    mkdir -p /app/model_cache && \
    if [ "$MODELLI_LOCALI" = "1" ]; then \
        cp -r /tmp/model_cache/. /app/model_cache/ && \
        { ls -d /app/model_cache/*/ >/dev/null 2>&1 || { echo "api/model_cache è vuota: cd api && python build_fast_model.py"; exit 1; }; }; \
    fi

COPY api/src/. .
COPY common/ ./common/
COPY utils/ ./utils/

ENV PYTHONUNBUFFERED=1
ENV FASTEMBED_CACHE_PATH=/app/model_cache
# Letto all'avvio: con un'immagine senza modelli locali, RERANK_PROVIDER=onnx o
# EMBEDDING_PROVIDER=local danno un errore che dice come ricostruirla.
ENV MODELLI_LOCALI=$MODELLI_LOCALI

EXPOSE 5000

CMD ["gunicorn", "--worker-class", "gevent", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]
