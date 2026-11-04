"""
API per i sistemi esterni, con chiavi proprie (tabelle external_api_key*, gestite
con utils/manage_api_keys.py) inviate come `Authorization: Bearer <chiave>`.

    GET  /external/v1/topics                          topic abilitati per la chiave
    GET  /external/v1/topics/<topic>/documents        documenti del topic, a pagine, con i metadati
    GET  /external/v1/topics/<topic>/documents/<id>   testo del documento presente nel database
    POST /external/v1/convert                         converte un documento in Markdown o testo
    GET  /external/v1/convert/<job_id>                stato e risultato della conversione (una volta sola)

I topic limitano la consultazione; per la conversione basta una chiave valida,
perché non legge nulla dell'archivio. La conversione è asincrona (una scansione
può richiedere minuti): la svolge il converter, sulla cartella condivisa.
"""

import base64
import binascii
import hashlib
import json
import logging
import pathlib
import re
import shutil
import time
import uuid

from flask import Blueprint, g, jsonify, request

from common.config import settings
from common.db_logger import MySQLLogHandler
from common.utility import CONVERSION_QUEUE, CONVERSION_QUEUE_ARGS, markdown_to_text
from review_routes import ApiError, Db, RequestLogContext, _handle_api_error, _handle_unexpected, _parse_json, publish_to_queue

log = logging.getLogger("external_api")
log.setLevel(getattr(logging, settings.log_level, logging.INFO))
log.addFilter(RequestLogContext())
db_handler = MySQLLogHandler()
db_handler.setLevel(logging.WARNING)
log.addHandler(db_handler)

bp = Blueprint("external", __name__, url_prefix="/external/v1")
bp.register_error_handler(ApiError, _handle_api_error)
bp.register_error_handler(Exception, _handle_unexpected)

CONVERSIONS_FOLDER = pathlib.Path(settings.data_folder) / "conversions"
CONVERT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".docx", ".odt", ".xlsx", ".ods"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
# Intestazioni di sezione del singolo frammento: non descrivono il documento.
HEADER_KEY_RE = re.compile(r"^Header \d+$")


@bp.before_request
def _authenticate():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise ApiError(401, "Credenziale mancante")
    key_hash = hashlib.sha256(header[7:].strip().encode()).hexdigest()
    with Db() as db:
        key = db.one(
            "SELECT id, name FROM external_api_keys WHERE key_hash = %s AND active = TRUE "
            "AND (expires_at IS NULL OR expires_at > NOW())",
            (key_hash,),
        )
        if not key:
            log.warning(f"Credenziale esterna non valida da {request.remote_addr} su {request.path}")
            raise ApiError(401, "Credenziale non valida")
        g.api_key = key
        g.api_topics = {r["topic_id"] for r in db.all(
            "SELECT topic_id FROM external_api_key_topics WHERE key_id = %s", (key["id"],))}
        # Al massimo una scrittura al minuto per chiave.
        db.execute("UPDATE external_api_keys SET last_used_at = NOW() WHERE id = %s AND "
                   "(last_used_at IS NULL OR last_used_at < NOW() - INTERVAL 60 SECOND)", (key["id"],))
        db.commit()
    log.info(f"[{key['name']}] {request.method} {request.path}")


def _check_topic(topic_id):
    if topic_id not in g.api_topics:
        log.warning(f"[{g.api_key['name']}] topic '{topic_id}' non abilitato")
        raise ApiError(403, "Topic non abilitato per questa credenziale")


def _useful_metadata(raw):
    """Metadati che descrivono il documento: via campi di sistema, tecnici, intestazioni e valori vuoti."""
    return {
        key: value for key, value in _parse_json(raw).items()
        if key not in settings.protected_keys
        and key not in settings.context_metadata_exclude
        and not key.startswith("_")
        and not HEADER_KEY_RE.match(key)
        and value not in (None, "", [])
    }


def _document_id(sub_topic_id, source):
    raw = json.dumps([sub_topic_id, source], ensure_ascii=False).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _parse_document_id(document_id):
    try:
        sub_topic_id, source = json.loads(base64.urlsafe_b64decode(document_id + "=" * (-len(document_id) % 4)))
        if isinstance(sub_topic_id, str) and isinstance(source, str):
            return sub_topic_id, source
    except (ValueError, TypeError, binascii.Error):
        pass
    raise ApiError(404, "Documento non trovato")


@bp.route("/topics", methods=["GET"])
def list_topics():
    if not g.api_topics:
        return jsonify({"topics": []})
    placeholders = ", ".join(["%s"] * len(g.api_topics))
    params = tuple(sorted(g.api_topics))
    with Db() as db:
        topics = db.all(f"SELECT topic_id, description FROM topics WHERE topic_id IN ({placeholders}) "
                        "ORDER BY topic_id", params)
        subs = db.all(f"SELECT topic_id, sub_topic_id, description FROM sub_topics WHERE topic_id IN ({placeholders}) "
                      "ORDER BY sub_topic_id", params)
    return jsonify({"topics": [
        {"id": t["topic_id"], "description": t["description"],
         "sub_topics": [{"id": s["sub_topic_id"], "description": s["description"]}
                        for s in subs if s["topic_id"] == t["topic_id"]]}
        for t in topics
    ]})


@bp.route("/topics/<topic_id>/documents", methods=["GET"])
def list_documents(topic_id):
    _check_topic(topic_id)
    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(200, max(1, int(request.args.get("page_size", 50))))
    except ValueError:
        raise ApiError(400, "page e page_size devono essere numeri")
    where, params = "topic_id = %s AND parent_index = 0", [topic_id]
    sub_topic_id = request.args.get("sub_topic_id", "").strip()
    if sub_topic_id:
        where += " AND sub_topic_id = %s"
        params.append(sub_topic_id)
    with Db() as db:
        total = db.one(f"SELECT COUNT(*) AS n FROM parent_documents WHERE {where}", tuple(params))["n"]
        rows = db.all(
            f"SELECT sub_topic_id, source, file_name, metadata FROM parent_documents WHERE {where} "
            "ORDER BY sub_topic_id, source LIMIT %s OFFSET %s",
            tuple(params) + (page_size, (page - 1) * page_size),
        )
    return jsonify({
        "total": total, "page": page, "page_size": page_size,
        "items": [{
            "id": _document_id(r["sub_topic_id"], r["source"]),
            "sub_topic_id": r["sub_topic_id"],
            "file_name": r["file_name"],
            "metadata": _useful_metadata(r["metadata"]),
        } for r in rows],
    })


@bp.route("/topics/<topic_id>/documents/<document_id>", methods=["GET"])
def get_document(topic_id, document_id):
    _check_topic(topic_id)
    sub_topic_id, source = _parse_document_id(document_id)
    with Db() as db:
        parents = db.all(
            "SELECT file_name, parent_index, content, metadata FROM parent_documents "
            "WHERE topic_id = %s AND sub_topic_id = %s AND source = %s ORDER BY parent_index",
            (topic_id, sub_topic_id, source),
        )
    if not parents:
        raise ApiError(404, "Documento non trovato")
    return jsonify({
        "id": document_id,
        "topic_id": topic_id,
        "sub_topic_id": sub_topic_id,
        "file_name": parents[0]["file_name"],
        "metadata": _useful_metadata(parents[0]["metadata"]),
        "content": "\n\n".join(p["content"] for p in parents),
    })


def _remove_expired_jobs():
    if not CONVERSIONS_FOLDER.is_dir():
        return
    limit = time.time() - settings.convert_retention_hours * 3600
    for job_dir in CONVERSIONS_FOLDER.iterdir():
        try:
            if job_dir.is_dir() and job_dir.stat().st_mtime < limit:
                shutil.rmtree(job_dir, ignore_errors=True)
        except OSError:
            continue


@bp.route("/convert", methods=["POST"])
def create_conversion():
    """
    multipart/form-data: file; format=markdown|text (default markdown).
    Usa solo il modello OCR configurato: se una pagina non riesce, il lavoro fallisce.
    """
    max_bytes = settings.convert_max_upload_mb * 1024 * 1024
    if request.content_length and request.content_length > max_bytes:
        raise ApiError(413, f"Documento troppo grande (massimo {settings.convert_max_upload_mb} MB)")
    upload = request.files.get("file")
    if not upload or not upload.filename:
        raise ApiError(400, "Manca il documento (campo 'file')")
    file_name = pathlib.PurePosixPath(upload.filename.replace("\\", "/")).name
    extension = pathlib.PurePosixPath(file_name).suffix.lower()
    if extension not in CONVERT_EXTENSIONS:
        raise ApiError(415, f"Formato non gestito. Ammessi: {', '.join(sorted(CONVERT_EXTENSIONS))}")
    output_format = request.form.get("format", "markdown").strip().lower()
    if output_format not in ("markdown", "text"):
        raise ApiError(400, "format deve essere markdown o text")

    _remove_expired_jobs()
    job_id = uuid.uuid4().hex
    job_dir = CONVERSIONS_FOLDER / job_id
    job_dir.mkdir(parents=True)
    try:
        input_file = f"input{extension}"
        upload.save(job_dir / input_file)
        if (job_dir / input_file).stat().st_size > max_bytes:
            raise ApiError(413, f"Documento troppo grande (massimo {settings.convert_max_upload_mb} MB)")
        (job_dir / "request.json").write_text(json.dumps({
            "key_id": g.api_key["id"], "file_name": file_name, "input_file": input_file,
            "format": output_format,
        }), encoding="utf-8")
        (job_dir / "status.json").write_text(json.dumps({"status": "queued"}), encoding="utf-8")
        try:
            publish_to_queue(CONVERSION_QUEUE, {"job_id": job_id}, CONVERSION_QUEUE_ARGS)
        except Exception as e:
            log.error(f"Pubblicazione del lavoro di conversione fallita: {e}", exc_info=True)
            raise ApiError(503, "Servizio di conversione non raggiungibile. Riprova.")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    log.info(f"[{g.api_key['name']}] conversione {job_id}: {file_name}, formato {output_format}")
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@bp.route("/convert/<job_id>", methods=["GET"])
def get_conversion(job_id):
    job_dir = CONVERSIONS_FOLDER / job_id
    # Id inesistente o lavoro di un'altra chiave: stessa risposta.
    if not JOB_ID_RE.match(job_id) or not (job_dir / "request.json").is_file():
        raise ApiError(404, "Lavoro di conversione non trovato")
    job = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    if job["key_id"] != g.api_key["id"]:
        raise ApiError(404, "Lavoro di conversione non trovato")
    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    response = {"job_id": job_id, "file_name": job["file_name"], "format": job["format"], **status}
    if status["status"] == "completed":
        markdown = (job_dir / "result.md").read_text(encoding="utf-8")
        response["content"] = markdown_to_text(markdown) if job["format"] == "text" else markdown
    if status["status"] in ("completed", "failed"):
        # Consegnato l'esito, sul server non resta nulla del documento.
        shutil.rmtree(job_dir, ignore_errors=True)
        log.info(f"[{g.api_key['name']}] conversione {job_id}: esito consegnato, lavoro cancellato")
    return jsonify(response)
