import os
import logging
import uuid
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
from celery.result import AsyncResult
import mysql.connector

from dotenv import load_dotenv

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
CORS(app)  # Enable CORS for frontend access

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_client = Celery(
    'rag_queue', 
    broker=REDIS_URL, 
    backend=REDIS_URL
)

load_dotenv()

# ==============================================================================
# MIDDLEWARE
# ==============================================================================
@app.before_request
def start_timer_and_add_id():
    request.request_id = str(uuid.uuid4())
    request.start_time = time.time()
    log.debug(f"[{request.request_id}] START {request.method} {request.path}")

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
    """Simple health check for Docker/Load Balancer."""
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat_handler():
    try:
        data = request.json
        if not data: return jsonify({"error": "Bad Request", "message": "Invalid JSON"}), 400

        query = data.get("query")
        history = data.get("history", [])

        if not query:
            return jsonify({"error": "Bad Request", "message": "Missing 'query' field"}), 400

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
        log.error(f"Failed to dispatch task: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

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


# Re-define minimal DB config for app.py (Worker has its own)
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME", "rag_system")
}

@app.route("/feedback", methods=["POST"])
def feedback_handler():
    # ... (Your existing feedback logic remains the same) ...
    # Since it's a simple INSERT, it doesn't strictly need Celery yet.
    try:
        data = request.json
        query = data.get("query")
        answer = data.get("answer")
        topic_id = data.get("topic_id")
        rating = data.get("rating")
        
        if not all([query, answer, rating is not None]):
            return jsonify({"error": "Missing fields"}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        sql = "INSERT INTO chat_feedback (user_query, ai_response, topic_id, rating) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (query, answer, topic_id, rating))
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        log.error(f"Feedback error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # In production, Gunicorn starts the app, so this is just for local debug
    app.run(host="0.0.0.0", port=5000, debug=True)