import os
import logging
import uuid
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
from celery.result import AsyncResult
from mysql.connector import pooling
import json

from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("api_gateway")

app = Flask(__name__)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
CORS(app, origins=ALLOWED_ORIGINS)  # Enable CORS for frontend access

API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
if not API_SECRET_KEY:
    log.warning("API_SECRET_KEY is not set — setting default value. This is not secure for production!")
    API_SECRET_KEY = "default_secret_key"
 
MAX_HISTORY_ITEMS = int(os.environ.get("MAX_HISTORY_ITEMS", 20))
 

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_client = Celery(
    'rag_queue', 
    broker=REDIS_URL, 
    backend=REDIS_URL
)

try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="api_pool",
        pool_size=5,
        pool_reset_session=True,
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASS"),
        database=os.environ.get("DB_NAME", "rag_system")
    )
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

    # --- Authentication ---
    # Skip auth for health checks and when no key is configured (dev mode).
    if API_SECRET_KEY and request.path not in UNPROTECTED_ROUTES:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != API_SECRET_KEY:
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
        conn = db_pool.get_connection()
        conn.ping(reconnect=True)
        conn.close()
        return jsonify({"status": "ok", "database": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503
    
@app.route("/chat", methods=["POST"])
def chat_handler():
    try:
        data = request.json
        if not data: return jsonify({"error": "Bad Request", "message": "Invalid JSON"}), 400

        query = data.get("query")
        history = data.get("history", [])

        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return jsonify({"error": "Bad Request", "message": "Valid 'query' string is required"}), 400
        
        if len(query) > 2000:
            return jsonify({"error": "Payload Too Large", "message": "Query exceeds maximum length"}), 413
        
        if not isinstance(history, list):
            return jsonify({"error": "Bad Request", "message": "'history' must be a list"}), 400
        history = [
            h for h in history[:MAX_HISTORY_ITEMS]
            if isinstance(h, dict) and "role" in h and "text" in h
        ]

        log.info(f"Received query: '{query[:50]}...'. Offloading to Worker.")

        # --- FIX 1: Send Task by Name ---
        # We use send_task() instead of importing the function.
        # This prevents loading the AI models in the API container.
        task = celery_client.send_task(
            'rag_queue', 
            args=[query, history]
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

# ==============================================================================
# FEEDBACK ENDPOINT (Direct DB Access)
# ==============================================================================
# Note: Feedback is light (SQL Insert), so we can keep it synchronous here 
# or move it to a worker if you expect massive scale.




@app.route("/feedback", methods=["POST"])
def feedback_handler():
    # ... (Your existing feedback logic remains the same) ...
    # Since it's a simple INSERT, it doesn't strictly need Celery yet.

    if not db_pool:
        log.error("Feedback rejected: DB pool not initialized.")
        return jsonify({"error": "Service Unavailable", "message": "Database not available."}), 503
    
    conn = None
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
        
        conn = db_pool.get_connection()
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