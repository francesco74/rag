FROM python:3.11-slim

WORKDIR /app

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev libgomp1 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Installazione Optimum e Torch CPU
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir "optimum[onnxruntime]" transformers \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Scaricamento del modello già quantizzato (INT8) dal nuovo repository
# NOTA: Usiamo 'onnx-community/bge-reranker-v2-m3-ONNX'
RUN python -c "\
from optimum.onnxruntime import ORTModelForSequenceClassification; \
from transformers import AutoTokenizer; \
model_id = 'onnx-community/bge-reranker-v2-m3-ONNX'; \
# Specifichiamo subfolder='onnx' per trovare il file correttamente \
model = ORTModelForSequenceClassification.from_pretrained( \
    model_id, \
    export=False, \
    subfolder='onnx', \
    file_name='model_quantized.onnx' \
); \
tokenizer = AutoTokenizer.from_pretrained(model_id); \
model.save_pretrained('./model_onnx'); \
tokenizer.save_pretrained('./model_onnx')"

COPY src/. .

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
EXPOSE 5000

CMD ["gunicorn", "--worker-class", "gevent", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]