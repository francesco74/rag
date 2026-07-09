import logging
import uuid
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
from celery.result import AsyncResult
import json

from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool

from common.config import settings

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("api_gateway")

app = Flask(__name__)

CORS(app, origins=settings.allowed_origins)  # Enable CORS for frontend access



 
redis_conn_string = f"redis://{settings.redis_host}:{settings.redis_port}/0"
celery_client = Celery(
    'rag_queue', 
    broker=redis_conn_string, 
    backend=redis_conn_string
)

try:
    db_pool = init_db_pool()
except Exception as e:
    log.critical(f"Failed to initialize API DB Pool: {e}")
    # Consider whether the app should crash here if the DB is critical


UNPROTECTED_ROUTES = {"/health"}

# ==============================================================================
# MIDDLEWARE
# ==============================================================================
@app.before_request
def start_timer_and_add_id():
    request.request_id = str(uuid.uuid4())
    request.start_time = time.time()
    log.debug(f"[{request.request_id}] START {request.method} {request.path}")

    if request.method == "OPTIONS":
        return

    # --- Authentication ---
    # Skip auth for health checks and when no key is configured (dev mode).
    if settings.api_secret_key and request.path not in UNPROTECTED_ROUTES:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != settings.api_secret_key:
            log.warning(f"[{request.request_id}] Unauthorized request to {request.path}")
            return jsonify({"error": "Unauthorized"}), 401

@app.after_request
def log_response(response):
    duration = time.time() - request.start_time
    log.debug(
        f"[{request.request_id}] END status={response.status_code} "
        f"time={duration:.3f}s"
    )
    return response

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.route("/health", methods=["GET"])
def health_check():
    try:
        # Quick check if DB pool is alive
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "DB Pool not initialized"}), 503
        
        conn.ping(reconnect=True)
        conn.close()
        return jsonify({"status": "ok", "database": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503
    
@app.route("/config", methods=["POST"])
def get_config():
    """Invia al frontend i subtopic relativi al topic richiesto."""
    data = request.json # <--- Leggi il body JSON
    if not data:
        return jsonify({"error": "Bad Request", "message": "Invalid JSON"}), 400
        
    topic_id = data.get("topic_id") # <--- Recupera topic_id dal JSON
    if not topic_id:
        return jsonify({"error": "Bad Request", "message": "topic_id is required"}), 400

    conn = None
    cursor = None

    try:
        conn = get_db_connection()  # <-- Modificato qui
        if not conn:
            return jsonify({"error": "Service Unavailable"}), 503
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT sub_topic_id, description FROM sub_topics WHERE topic_id = %s", (topic_id,))
        rows = cursor.fetchall()

        # Conteggio documenti e anno più vecchio per sub_topic, per mostrare
        # nel frontend qualcosa tipo "Determine presenti nel sistema: 1834
        # a partire dal 2015".
        # Usiamo il campo "anno" del JSON di metadata (es. "2025") invece di
        # "data": quest'ultimo è salvato in formato italiano "DD/MM/YYYY"
        # (es. "26/11/2025"), non ISO, quindi né un MIN() lessicografico né
        # un'estrazione di sottostringa darebbero un anno corretto. "anno"
        # è già la stringa numerica dell'anno, quindi basta un CAST.
        cursor.execute(
            """
            SELECT sub_topic_id,
                   COUNT(*) AS doc_count,
                   MIN(
                       CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.anno')), '') AS UNSIGNED)
                   ) AS since_year
            FROM parent_documents
            WHERE topic_id = %s
            GROUP BY sub_topic_id
            """,
            (topic_id,)
        )
        stats_rows = cursor.fetchall()
        stats_by_subtopic = {
            row['sub_topic_id']: {
                "doc_count": row['doc_count'],
                "since_year": row['since_year']
            }
            for row in stats_rows
        }

        sub_topics_data = [
            {
                "id": row['sub_topic_id'],
                "desc": row['description'] or row['sub_topic_id'],
                "doc_count": stats_by_subtopic.get(row['sub_topic_id'], {}).get("doc_count", 0),
                "since_year": stats_by_subtopic.get(row['sub_topic_id'], {}).get("since_year"),
            }
            for row in rows
        ]

        return jsonify({
            "allow_subtopic_selection": settings.allow_subtopic_selection,
            "allow_date_filter": settings.allow_date_filter,
            "sub_topics": sub_topics_data 
        }), 200
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route("/chat", methods=["POST"])
def chat_handler():
    try:
        data = request.json
        if not data: return jsonify({"error": "Bad Request", "message": "Invalid JSON"}), 400

        query = data.get("query")
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return jsonify({"error": "Bad Request", "message": "Valid 'query' string is required"}), 400
        
        if len(query) > 2000:
            return jsonify({"error": "Payload Too Large", "message": "Query exceeds maximum length"}), 413
        

        history = data.get("history", [])
        if not isinstance(history, list):
            return jsonify({"error": "Bad Request", "message": "'history' must be a list"}), 400
        history = [
            h for h in history[:settings.max_history_items]
            if isinstance(h, dict) and "role" in h and "text" in h
        ]

        topic_id = data.get("topic_id")
        if not topic_id:
             return jsonify({"error": "Bad Request", "message": "topic_id is required"}), 400

        selected_sub_topics = data.get("sub_topics", []) 
        metadata_filters = data.get("filters", {})

        # --- Filtro opzionale sul range temporale ---
        # Il campo "data" su Qdrant è una stringa "YYYY-MM-DD" (vedi
        # estrattore.py: clean_iso_date() rimuove sempre la componente oraria),
        # quindi confrontiamo direttamente stringhe: il confronto lessicografico
        # su ISO 8601 coincide con l'ordine cronologico.
        date_from = data.get("date_from") if settings.allow_date_filter else None
        date_to = data.get("date_to") if settings.allow_date_filter else None

        include_undated = data.get("include_undated", True)
        if not isinstance(include_undated, bool):
            return jsonify({
                "error": "Bad Request",
                "message": "include_undated deve essere un booleano"
            }), 400

        if date_from or date_to:
            log.info(f"Received data range: {date_from}...{date_to}  for topic {topic_id}")
            date_range = {}
            try:
                if date_from:
                    datetime.strptime(date_from, "%Y-%m-%d")  # solo validazione formato
                    date_range["gte"] = date_from

                if date_to:
                    datetime.strptime(date_to, "%Y-%m-%d")  # solo validazione formato
                    date_range["lte"] = date_to

                metadata_filters = {**metadata_filters, "data": date_range}
            except ValueError:
                return jsonify({
                    "error": "Bad Request",
                    "message": "date_from/date_to devono essere in formato YYYY-MM-DD"
                }), 400

        log.info(f"Received query: '{query[:50]}...'. Offloading to Worker.")

        task = celery_client.send_task(
            'rag_queue', 
            args=[query, history, topic_id, selected_sub_topics, metadata_filters, include_undated] 
        )

        return jsonify({
            "task_id": task.id,
            "status": "processing",
            "message": "Query received."
        }), 202

    except Exception as e:
        log.error(f"[{request.request_id}] Failed to dispatch task: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": "Failed to queue task."}), 500

@app.route("/status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """
    Polling Endpoint.
    Frontend calls this every 1-2 seconds to check if the answer is ready.
    """
    try:
        task_result = AsyncResult(task_id, app=celery_client)

        if task_result.state == 'PENDING':
            return jsonify({"task_id": task_id, "status": "processing"}), 202
        
        elif task_result.state == 'SUCCESS':
            result_data = task_result.result
            # Handle logical errors returned by worker
            if isinstance(result_data, dict) and "error" in result_data and result_data.get("status") != "success":
                 return jsonify(result_data), 200
            
            return jsonify({
                "task_id": task_id,
                "status": "completed",
                "data": result_data
            }), 200
            
        elif task_result.state == 'FAILURE':
            return jsonify({
                "task_id": task_id,
                "status": "failed",
                "error": str(task_result.info)
            }), 500
        
        else:
            return jsonify({"task_id": task_id, "status": "failed", "substatus": task_result.state}), 202

    except Exception as e:
        log.error(f"Error checking status for {task_id}: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/feedback", methods=["POST"])
def feedback_handler():
    if not db_pool:
        log.error("Feedback rejected: DB pool not initialized.")
        return jsonify({"error": "Service Unavailable", "message": "Database not available."}), 503
    
    conn = get_db_connection()  # <-- Ottieni subito la connessione
    if not conn:
        log.error("Feedback rejected: DB pool not initialized or exhausted.")
        return jsonify({"error": "Service Unavailable", "message": "Database not available."}), 503
    
    cursor = None

    try:
        data = request.json
        query = data.get("query")
        answer = data.get("answer")
        topic_id = data.get("topic_id")
        rating = data.get("rating")

        history = data.get("history", [])
        comment = data.get("comment", "")
        
        if not all([query, answer, rating is not None]):
            return jsonify({"error": "Missing fields"}), 400
        
        history_json = json.dumps(history)
        cursor = conn.cursor()

        sql = "INSERT INTO chat_feedback (user_query, ai_response, topic_id, rating, chat_history, comment) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (query, answer, topic_id, rating, history_json, comment))
        conn.commit()
        return jsonify({"status": "success"}), 200

    except Exception as e:
        log.error(f"[{request.request_id}] Feedback DB error: {e}", exc_info=True)
        
        if conn:
            conn.rollback()

        return jsonify({"error": "Internal database error"}), 500
    
    finally:
        # FIX: Guaranteed connection closure
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    # In production, Gunicorn starts the app, so this is just for local debug
    app.run(host="0.0.0.0", port=5000, debug=True)