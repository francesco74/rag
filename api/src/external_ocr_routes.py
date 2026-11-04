"""
Trascrizione di un documento con l'OCR esterno, su richiesta del revisore.

Il documento va a questi servizi solo da qui, per scelta di chi lo sta correggendo:
il modello esterno (OCR_EXTERNAL_FALLBACK) e, se fallisce, Google Cloud Vision
(OCR_VISION_FALLBACK). Il testo ottenuto è una proposta: entra nell'editor e diventa
definitivo solo col salvataggio.

Il lavoro è legato al documento, non alla pagina: chi esce e rientra lo ritrova, e
finché è in corso o non è stato usato un altro clic non ne avvia un secondo.

    POST   /review/document/external-ocr   avvia la trascrizione (chiave del documento)
    GET    /review/document/external-ocr   stato e, a fine lavoro, testo
    DELETE /review/document/external-ocr   toglie il risultato: usato o scartato
"""

import hashlib
import json
import logging
import pathlib
import shutil
import time

from flask import Blueprint, g, jsonify, request

from common.review_permissions import PERM_TEXT
from common.utility import CONVERSION_QUEUE, CONVERSION_QUEUE_ARGS
from external_routes import CONVERSIONS_FOLDER, _remove_expired_jobs
from review_routes import (ApiError, Db, _check_topic, _doc_key_from_request, _load_document, _require_lock,
                           external_ocr_available, original_file_path, publish_to_queue, require_permission)

log = logging.getLogger("EXTERNAL_OCR")

bp = Blueprint("external_ocr", __name__, url_prefix="/document/external-ocr")

EXTERNAL_OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
# Un lavoro fermo da così tanto è perso (converter riavviato, messaggio scartato): si può rifare.
STALE_SECONDS = 2 * 3600


def _job_dir(source, topic_id, sub_topic_id):
    """Un solo lavoro per documento: l'id deriva dalla chiave, con la forma che il converter accetta."""
    key = json.dumps([source, topic_id, sub_topic_id])
    return CONVERSIONS_FOLDER / hashlib.sha256(key.encode()).hexdigest()[:32]


def _job_state(job_dir):
    """Stato del lavoro del documento: "none" se non ce n'è."""
    if not (job_dir / "request.json").exists():
        return {"status": "none"}
    job = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    status_path = job_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "queued"}
    state = {"status": status["status"], "error": status.get("error"), "requested_by": job.get("requested_by")}
    if status["status"] == "completed":
        state["content"] = (job_dir / "result.md").read_text(encoding="utf-8")
    elif status["status"] != "failed" and time.time() - job_dir.stat().st_mtime > STALE_SECONDS:
        state.update(status="failed", error="La trascrizione si è interrotta: puoi riprovare.")
    return state


@bp.route("", methods=["POST"])
@require_permission(PERM_TEXT)
def start_external_ocr():
    if not external_ocr_available():
        raise ApiError(409, "L'OCR esterno non è configurato")
    source, topic_id, sub_topic_id = _doc_key_from_request(request.get_json(silent=True) or {})
    _check_topic(g.user, topic_id)
    with Db() as db:
        _require_lock(db, source, topic_id, sub_topic_id)
        doc, _, _ = _load_document(db, source, topic_id, sub_topic_id)
    original = original_file_path(topic_id, sub_topic_id, doc["file_name"])
    if not original:
        raise ApiError(409, "Il file originale non è in archivio")
    if original.suffix.lower() not in EXTERNAL_OCR_EXTENSIONS:
        raise ApiError(422, "L'OCR esterno trascrive solo PDF e immagini")

    _remove_expired_jobs()
    job_dir = _job_dir(source, topic_id, sub_topic_id)
    state = _job_state(job_dir)
    if state["status"] == "failed":
        shutil.rmtree(job_dir, ignore_errors=True)
    elif state["status"] != "none":
        return jsonify(state), 200
    job_id = job_dir.name
    try:
        job_dir.mkdir(parents=True)
    except FileExistsError:
        # Due clic insieme: vale il primo.
        return jsonify({"status": "queued"}), 200
    try:
        input_file = f"input{original.suffix.lower()}"
        shutil.copyfile(original, job_dir / input_file)
        (job_dir / "request.json").write_text(json.dumps({
            "input_file": input_file, "file_name": original.name, "external_ocr": True,
            "topic_id": topic_id, "requested_by": g.user["username"],
        }), encoding="utf-8")
        publish_to_queue(CONVERSION_QUEUE, {"job_id": job_id}, CONVERSION_QUEUE_ARGS)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        log.error(f"Trascrizione esterna non avviata per {source}: {e}")
        raise ApiError(503, "Trascrizione non avviata: riprova tra poco")
    log.info(f"{g.user['username']}: {source} inviato all'OCR esterno (lavoro {job_id})")
    return jsonify({"status": "queued", "requested_by": g.user["username"]}), 202


@bp.route("", methods=["GET"])
@require_permission(PERM_TEXT)
def external_ocr_state():
    source, topic_id, sub_topic_id = _doc_key_from_request()
    return jsonify(_job_state(_job_dir(source, topic_id, sub_topic_id)))


@bp.route("", methods=["DELETE"])
@require_permission(PERM_TEXT)
def remove_external_ocr():
    source, topic_id, sub_topic_id = _doc_key_from_request()
    with Db() as db:
        _require_lock(db, source, topic_id, sub_topic_id)
    job_dir = _job_dir(source, topic_id, sub_topic_id)
    if _job_state(job_dir)["status"] in ("queued", "running"):
        raise ApiError(409, "La trascrizione è ancora in corso")
    shutil.rmtree(job_dir)
    log.info(f"{g.user['username']}: tolta la trascrizione esterna di {source}")
    return jsonify({"status": "none"})
