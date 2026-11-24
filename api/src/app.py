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
from common.utility import quiet_http_loggers
from review_routes import auth_bp, bp as review_bp
from files_routes import bp as files_bp
from external_routes import bp as external_bp
import external_sessions  # noqa: F401  (rotte /external/v1/sessions sullo stesso blueprint)
from admin_routes import bp as admin_bp
from upload_routes import bp as upload_bp
from external_ocr_routes import bp as external_ocr_bp
from site_routes import bp as site_bp, request_hostname, site_for_host

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("api_gateway")
quiet_http_loggers()

app = Flask(__name__)

# Indirizzo reale del client dietro ingress e nginx (X-Forwarded-For), per i
# log e per il limite dei tentativi di login per IP. Solo con proxy fidati
# dichiarati: senza, X-Forwarded-For lo potrebbe scrivere il client stesso.
if settings.trusted_proxy_hops > 0:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=settings.trusted_proxy_hops)

CORS(app, origins=settings.allowed_origins)  # Enable CORS for frontend access

# Revisione documentale (frontend web/revisione): rotte sotto /review con
# autenticazione propria a utenti e password, vedi review_routes.py.
# L'amministrazione (admin_routes.py) è sotto /review/admin: ne eredita controlli
# ed errori; va agganciata prima di registrare il blueprint principale.
review_bp.register_blueprint(admin_bp)
review_bp.register_blueprint(upload_bp)
review_bp.register_blueprint(external_ocr_bp)
app.register_blueprint(review_bp)
# Login/logout generici (/auth/...), usati dalla chat per i documenti riservati.
app.register_blueprint(auth_bp)
# Download dei documenti originali per chat e revisione: archivi pubblici
# liberi, riservati solo per utenti autenticati, vedi files_routes.py.
app.register_blueprint(files_bp)
# API per i sistemi esterni: chiavi proprie, vedi external_routes.py.
app.register_blueprint(external_bp)
# Riconoscimento del sito della chat dal dominio, vedi site_routes.py.
app.register_blueprint(site_bp)



 
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


UNPROTECTED_ROUTES = {"/health", "/site", "/site/icon"}

# Rotte che la chat chiama senza credenziali: le autorizza il sito (dominio) da
# cui arriva la richiesta, limitato al suo archivio. La vecchia chiave resta
# valida per chi chiama l'API da un programma; ai frontend non serve più.
# "/config" copre anche "/config/stats" (confronto per prefisso).
SITE_ROUTES = ("/config", "/chat", "/status/", "/feedback")

# Le rotte di revisione non usano API_SECRET_KEY: ogni revisore si autentica
# con le proprie credenziali e il blueprint verifica il suo token di sessione.
# Lo stesso per i documenti: i link sono aperti dal browser, che non può
# inviare la API key; l'accesso lo decide files_routes.py per archivio.
SELF_AUTHENTICATED_PREFIXES = ("/review/", "/files/", "/auth/", "/external/")

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
    if (settings.api_secret_key and request.path not in UNPROTECTED_ROUTES
            and not request.path.startswith(SELF_AUTHENTICATED_PREFIXES)):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != settings.api_secret_key:
            site = site_for_host(request_hostname()) if request.path.startswith(SITE_ROUTES) else None
            if not site:
                log.warning(f"[{request.request_id}] Unauthorized request to {request.path}")
                return jsonify({"error": "Unauthorized"}), 401
            body = request.get_json(silent=True)
            topic_id = body.get("topic_id") if isinstance(body, dict) else None
            if topic_id and topic_id != site["topic_id"]:
                log.warning(f"[{request.request_id}] Il sito '{site['hostname']}' ha chiesto l'archivio '{topic_id}'")
                return jsonify({"error": "Forbidden"}), 403

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
    
def _topic_dalla_richiesta():
    """topic_id dal body JSON, oppure (None, risposta di errore)."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "Bad Request", "message": "Invalid JSON"}), 400)
    topic_id = data.get("topic_id")
    if not topic_id:
        return None, (jsonify({"error": "Bad Request", "message": "topic_id is required"}), 400)
    return topic_id, None


@app.route("/config", methods=["POST"])
def get_config():
    """Invia al frontend i subtopic relativi al topic richiesto. Le statistiche della
    home (documenti per serie) sono a parte, in /config/stats: qui servono solo le
    serie, perché il filtro si possa usare subito."""
    topic_id, errore = _topic_dalla_richiesta()
    if errore:
        return errore

    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Service Unavailable"}), 503
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT sub_topic_id, description, description_long FROM sub_topics WHERE topic_id = %s", (topic_id,))
        sub_topics_data = [
            {"id": row['sub_topic_id'], "desc": row['description'], "desc_long": row['description_long']}
            for row in cursor.fetchall()
        ]

        # Si configura sul sito dell'archivio; senza sito (accesso con chiave) le serie si vedono.
        cursor.execute("SELECT allow_subtopic_selection FROM chat_sites WHERE topic_id = %s", (topic_id,))
        site = cursor.fetchone()

        return jsonify({
            "allow_subtopic_selection": bool(site["allow_subtopic_selection"]) if site else True,
            "sub_topics": sub_topics_data
        }), 200
    finally:
        if conn and conn.is_connected():
            conn.close()


_redis_stats = None


def _cache_stats():
    """Client Redis per la cache delle statistiche (lo stesso Redis di Celery)."""
    global _redis_stats
    if _redis_stats is None:
        import redis
        _redis_stats = redis.Redis.from_url(redis_conn_string, socket_timeout=0.5, socket_connect_timeout=0.5)
    return _redis_stats


def _calcola_stats(topic_id):
    """Documenti e anno più vecchio per sub_topic, per la home della chat ("Determine
    presenti nel sistema: 1834 a partire dal 2015").
    In parent_documents c'è una riga per ogni parte (parent) di un documento: si conta
    solo la prima (parent_index = 0), una per documento, che porta anche i metadati.
    Si usa il campo "anno" del JSON (es. "2025") e non "data", che è in formato
    italiano "DD/MM/YYYY": né un MIN() lessicografico né una sottostringa darebbero
    l'anno giusto."""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT sub_topic_id,
                   COUNT(*) AS doc_count,
                   MIN(
                       CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.anno')), '') AS UNSIGNED)
                   ) AS since_year
            FROM parent_documents
            WHERE topic_id = %s AND parent_index = 0
            GROUP BY sub_topic_id
            """,
            (topic_id,)
        )
        return {row['sub_topic_id']: {"doc_count": row['doc_count'], "since_year": row['since_year']}
                for row in cursor.fetchall()}
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route("/config/stats", methods=["POST"])
def get_config_stats():
    """Statistiche della home per serie: {"stats": {sub_topic_id: {doc_count, since_year}}}.
    Cambiano solo con i caricamenti: si tengono in Redis per STATS_CACHE_SECONDS. Se
    Redis non risponde si calcolano comunque."""
    topic_id, errore = _topic_dalla_richiesta()
    if errore:
        return errore
    chiave = f"rag:config_stats:{topic_id}"
    if settings.stats_cache_seconds > 0:
        try:
            salvato = _cache_stats().get(chiave)
            if salvato:
                return jsonify({"stats": json.loads(salvato)}), 200
        except Exception as e:
            log.warning(f"Cache delle statistiche non disponibile: {e}")
    stats = _calcola_stats(topic_id)
    if stats is None:
        return jsonify({"error": "Service Unavailable"}), 503
    if settings.stats_cache_seconds > 0:
        try:
            _cache_stats().set(chiave, json.dumps(stats), ex=settings.stats_cache_seconds)
        except Exception as e:
            log.warning(f"Cache delle statistiche non disponibile: {e}")
    return jsonify({"stats": stats}), 200


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
        date_from = data.get("date_from")
        date_to = data.get("date_to")

        include_undated = data.get("include_undated", True)
        if not isinstance(include_undated, bool):
            return jsonify({
                "error": "Bad Request",
                "message": "include_undated deve essere un booleano"
            }), 400

        # Ricerca approfondita: più passaggi per sotto-domanda (RERANK_MAX_SIZE
        # invece di RERANK_SIZE).
        deep_search = data.get("deep_search", False)
        if not isinstance(deep_search, bool):
            return jsonify({
                "error": "Bad Request",
                "message": "deep_search deve essere un booleano"
            }), 400
        if deep_search:
            log.info(f"[{request.request_id}] Ricerca approfondita richiesta per il topic {topic_id}")

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
            args=[query, history, topic_id, selected_sub_topics, metadata_filters, include_undated, deep_search] 
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