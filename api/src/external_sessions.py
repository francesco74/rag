"""
Sessioni di documenti per i sistemi esterni (stessa autenticazione di external_routes.py).

    POST   /external/v1/sessions                              crea: {owner} -> {session_id, expires_at}
    GET    /external/v1/sessions/<sid>                        stato, documenti e limiti (rinnova la scadenza)
    DELETE /external/v1/sessions/<sid>                        chiude e cancella tutto, subito
    POST   /external/v1/sessions/<sid>/documents              carica un file (multipart, campo file) -> 202 {doc_id}
    GET    /external/v1/sessions/<sid>/documents/<did>        stato; a documento pronto: tokens, sections
    DELETE /external/v1/sessions/<sid>/documents/<did>        toglie il documento
    POST   /external/v1/sessions/<sid>/search                 {query, doc_ids?, top_k?} -> sezioni pertinenti
    GET    /external/v1/sessions/<sid>/documents/<did>/sections?from=&limit=   sezioni in ordine

Una sessione appartiene alla chiave che l'ha creata: con un'altra chiave, o dopo la
scadenza, la risposta è 404 come per una sessione inesistente. Il rag indicizza e
cerca, non genera: la risposta la scrive il sistema esterno. Percorso dei documenti
e scadenza: common/document_sessions.py; ricerca: session_search.py.
"""

import json
import pathlib
import shutil
import uuid

from flask import g, jsonify, request

import external_routes as ex
import review_routes
import session_search
from common import document_sessions as ds
from common.config import settings
from common.utility import CONVERSION_QUEUE, CONVERSION_QUEUE_ARGS
from external_routes import CONVERT_EXTENSIONS, bp, log
from review_routes import ApiError, Db, _parse_json

MAX_TOP_K = 20
MAX_SECTIONS_PAGE = 50
MAX_QUERY_CHARS = 2000


def _qdrant():
    return review_routes.qdrant()


def _limits():
    return {"max_documents": settings.session_max_documents, "max_tokens": settings.session_max_tokens,
            "ttl_minutes": settings.session_ttl_minutes, "max_hours": settings.session_max_hours}


def _as_datetime(value):
    if isinstance(value, str):   # SQLite nei test
        from datetime import datetime
        return datetime.fromisoformat(value)
    return value


def _open_session(db, session_id):
    """Sessione della chiave, non scaduta, con la scadenza rinnovata. 404 altrimenti."""
    if not ds.ID_RE.match(session_id):
        raise ApiError(404, "Sessione non trovata")
    row = db.one("SELECT id, key_id, owner, created_at, expires_at FROM sessions WHERE id = %s", (session_id,))
    now = ds.utcnow()
    if not row or row["key_id"] != g.api_key["id"] or _as_datetime(row["expires_at"]) <= now:
        raise ApiError(404, "Sessione non trovata")
    row["created_at"] = _as_datetime(row["created_at"])
    row["expires_at"] = ds.next_expiry(row["created_at"], now)
    db.execute("UPDATE sessions SET expires_at = %s WHERE id = %s", (row["expires_at"], session_id))
    db.commit()
    return row


def _document_view(d):
    view = {"doc_id": d["id"], "file_name": d["file_name"], "status": d["status"]}
    if d.get("error"):
        view["error"] = d["error"]
    if d.get("tokens") is not None:
        view["tokens"] = d["tokens"]
    if d.get("sections") is not None:
        view["sections"] = d["sections"]
    return view


def _documents(db, session_id):
    return db.all("SELECT id, file_name, status, error, tokens, sections FROM session_documents "
                  "WHERE session_id = %s ORDER BY created_at, id", (session_id,))


def _document(db, session_id, doc_id):
    if not ds.ID_RE.match(doc_id):
        raise ApiError(404, "Documento non trovato")
    d = db.one("SELECT id, file_name, status, error, tokens, sections FROM session_documents "
               "WHERE id = %s AND session_id = %s", (doc_id, session_id))
    if not d:
        raise ApiError(404, "Documento non trovato")
    return d


@bp.route("/sessions", methods=["POST"])
def create_session():
    body = request.get_json(silent=True) or {}
    owner = body.get("owner", "")
    if not isinstance(owner, str) or len(owner) > 128:
        raise ApiError(400, "owner deve essere un testo di al massimo 128 caratteri")
    session_id = uuid.uuid4().hex
    now = ds.utcnow()
    expires_at = ds.next_expiry(now, now)
    with Db() as db:
        db.execute("INSERT INTO sessions (id, key_id, owner, created_at, expires_at) VALUES (%s, %s, %s, %s, %s)",
                   (session_id, g.api_key["id"], owner, now, expires_at))
        db.commit()
    log.info(f"[{g.api_key['name']}] sessione {session_id} creata")
    return jsonify({"session_id": session_id, "owner": owner, "expires_at": ds.iso(expires_at),
                    "limits": _limits()}), 201


@bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    with Db() as db:
        s = _open_session(db, session_id)
        docs = _documents(db, session_id)
    return jsonify({
        "session_id": session_id, "owner": s["owner"],
        "created_at": ds.iso(s["created_at"]), "expires_at": ds.iso(s["expires_at"]),
        "documents": [_document_view(d) for d in docs],
        "tokens": sum(d["tokens"] or 0 for d in docs if d["status"] != ds.FAILED),
        "limits": _limits(),
    })


@bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    with Db() as db:
        _open_session(db, session_id)
    ds.delete_sessions(_qdrant(), [session_id])
    log.info(f"[{g.api_key['name']}] sessione {session_id} chiusa")
    return "", 204


@bp.route("/sessions/<session_id>/documents", methods=["POST"])
def upload_document(session_id):
    max_bytes = settings.convert_max_upload_mb * 1024 * 1024
    if request.content_length and request.content_length > max_bytes:
        raise ApiError(413, f"Documento troppo grande (massimo {settings.convert_max_upload_mb} MB)")
    upload = request.files.get("file")
    if not upload or not upload.filename:
        raise ApiError(400, "Manca il documento (campo 'file')")
    file_name = pathlib.PurePosixPath(upload.filename.replace("\\", "/")).name[:255]
    extension = pathlib.PurePosixPath(file_name).suffix.lower()
    if extension not in CONVERT_EXTENSIONS:
        raise ApiError(415, f"Formato non gestito. Ammessi: {', '.join(sorted(CONVERT_EXTENSIONS))}")

    with Db() as db:
        _open_session(db, session_id)
        docs = [d for d in _documents(db, session_id) if d["status"] != ds.FAILED]
        if len(docs) >= settings.session_max_documents:
            raise ApiError(409, f"La sessione ha già {settings.session_max_documents} documenti: togline uno.",
                           code="max_documents")
        used = sum(d["tokens"] or 0 for d in docs)
        if used >= settings.session_max_tokens:
            raise ApiError(409, "Lo spazio della sessione è esaurito: togli un documento.", code="max_tokens")

        doc_id = uuid.uuid4().hex
        folder = ds.document_folder(session_id, doc_id)
        folder.mkdir(parents=True)
        try:
            input_file = f"input{extension}"
            upload.save(folder / input_file)
            if (folder / input_file).stat().st_size > max_bytes:
                raise ApiError(413, f"Documento troppo grande (massimo {settings.convert_max_upload_mb} MB)")
            (folder / "request.json").write_text(json.dumps({"file_name": file_name, "input_file": input_file}),
                                                 encoding="utf-8")
            db.execute("INSERT INTO session_documents (id, session_id, file_name, status, created_at) "
                       "VALUES (%s, %s, %s, %s, %s)", (doc_id, session_id, file_name, ds.QUEUED, ds.utcnow()))
            db.commit()
            try:
                ex.publish_to_queue(CONVERSION_QUEUE, {"session_id": session_id, "doc_id": doc_id},
                                    CONVERSION_QUEUE_ARGS)
            except Exception as e:
                log.error(f"Pubblicazione del documento di sessione fallita: {e}", exc_info=True)
                db.execute("DELETE FROM session_documents WHERE id = %s", (doc_id,))
                db.commit()
                raise ApiError(503, "Servizio di conversione non raggiungibile. Riprova.")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
    log.info(f"[{g.api_key['name']}] sessione {session_id}: documento {doc_id} caricato ({extension})")
    return jsonify({"doc_id": doc_id, "file_name": file_name, "status": ds.QUEUED}), 202


@bp.route("/sessions/<session_id>/documents/<doc_id>", methods=["GET"])
def get_session_document(session_id, doc_id):
    with Db() as db:
        _open_session(db, session_id)
        d = _document(db, session_id, doc_id)
    return jsonify(_document_view(d))


@bp.route("/sessions/<session_id>/documents/<doc_id>", methods=["DELETE"])
def delete_session_document(session_id, doc_id):
    with Db() as db:
        _open_session(db, session_id)
        _document(db, session_id, doc_id)
    ds.delete_document(_qdrant(), session_id, doc_id)
    return "", 204


@bp.route("/sessions/<session_id>/documents/<doc_id>/sections", methods=["GET"])
def get_sections(session_id, doc_id):
    try:
        start = max(0, int(request.args.get("from", 0)))
        limit = min(MAX_SECTIONS_PAGE, max(1, int(request.args.get("limit", 20))))
    except ValueError:
        raise ApiError(400, "from e limit devono essere numeri")
    with Db() as db:
        _open_session(db, session_id)
        d = _document(db, session_id, doc_id)
        if d["status"] != ds.READY:
            raise ApiError(409, "Documento non ancora pronto", doc_status=d["status"])
        rows = db.all("SELECT parent_index, content, metadata FROM session_parents "
                      "WHERE doc_id = %s AND session_id = %s AND parent_index >= %s "
                      "ORDER BY parent_index LIMIT %s", (doc_id, session_id, start, limit))
    return jsonify({
        "doc_id": doc_id, "file_name": d["file_name"], "total": d["sections"] or 0, "from": start,
        "sections": [{"index": r["parent_index"], "content": r["content"],
                      "headings": _parse_json(r["metadata"])} for r in rows],
    })


@bp.route("/sessions/<session_id>/search", methods=["POST"])
def search_session(session_id):
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ApiError(400, "Manca la domanda (query)")
    if len(query) > MAX_QUERY_CHARS:
        raise ApiError(400, f"Domanda troppo lunga (massimo {MAX_QUERY_CHARS} caratteri)")
    try:
        top_k = min(MAX_TOP_K, max(1, int(body.get("top_k", 8))))
    except (TypeError, ValueError):
        raise ApiError(400, "top_k deve essere un numero")
    requested = body.get("doc_ids")
    if requested is not None and (not isinstance(requested, list) or not all(isinstance(x, str) for x in requested)):
        raise ApiError(400, "doc_ids deve essere una lista di identificativi")

    with Db() as db:
        _open_session(db, session_id)
        docs = {d["id"]: d for d in _documents(db, session_id)}
    if requested is not None:
        unknown = [x for x in requested if x not in docs]
        if unknown:
            raise ApiError(404, "Documento non trovato", doc_ids=unknown)
        selected = requested
    else:
        selected = list(docs)
    ready = [x for x in selected if docs[x]["status"] == ds.READY]
    not_ready = [x for x in selected if docs[x]["status"] != ds.READY]
    if not ready:
        raise ApiError(409, "Nessuno dei documenti scelti è pronto", not_ready=not_ready)

    sections, reranker = session_search.search(_qdrant(), session_id, g.api_key["id"], ready,
                                               query.strip(), top_k)
    passages = []
    if sections:
        with Db() as db:
            for s in sections:
                row = db.one("SELECT content, metadata FROM session_parents "
                             "WHERE doc_id = %s AND session_id = %s AND parent_index = %s",
                             (s["doc_id"], session_id, s["parent_index"]))
                if row:
                    passages.append({**s, "file_name": docs[s["doc_id"]]["file_name"],
                                     "content": row["content"], "headings": _parse_json(row["metadata"])})
    return jsonify({"passages": passages, "reranker": reranker, "not_ready": not_ready})
