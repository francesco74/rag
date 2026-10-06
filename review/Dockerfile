FROM python:3.11-slim

WORKDIR /app

COPY review/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY review/src/. .
COPY common/ ./common/

# common/config.py legge queste variabili con os.environ[...] e fallirebbe
# all'import se assenti; il servizio di revisione non le usa.
ENV DOCWS_RICERCA_ENDPOINT=unused \
    DOCWS_ATTI_ENDPOINT=unused \
    DOCWS_CODICE_AMMINISTRAZIONE=unused \
    DOCWS_CODICE_AOO=unused \
    RUOLO_DOCWS=unused \
    PYTHONUNBUFFERED=1

EXPOSE 5001

# Build dalla radice del repository:  docker build -f review/Dockerfile .
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:5001", "--timeout", "120", "review_app:app"]
