"""
Servizio di REVISIONE documentale.

Permette a revisori autenticati di:
  - cercare i documenti indicizzati nel RAG (tabella parent_documents);
  - leggere il testo completo prodotto da converter/OCR e i metadati;
  - visualizzare il file originale archiviato in DATA_FOLDER/processed;
  - correggere il testo: il nuovo .md viene rimesso nella coda
    "da-indicizzare" e l'ingest (già idempotente) rigenera parent e chunk;
  - aggiungere/modificare/rimuovere metadati: aggiornati direttamente su
    MySQL e sul payload Qdrant, senza ricalcolare gli embedding;
  - marcare lo stato di revisione e consultare lo storico delle modifiche.

Registrato in app.py come blueprint sotto /review: un solo punto di accesso
al sistema. Le rotte /review/* NON usano la API key della chat ma i token di
sessione dei revisori (utenti con login, tabella review_users): app.py le
esclude dal controllo della API key. Ogni utente può avere più ruoli e ogni
endpoint richiede un permesso specifico: vedi review_permissions.py.

Identità di un documento: la terna (source, topic_id, sub_topic_id), la
stessa usata dall'ingest per l'idempotenza.
"""

import hashlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import uuid
from datetime import datetime
from functools import wraps

import pika
from flask import Blueprint, g, jsonify, request, send_file
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from qdrant_client import QdrantClient, models
from werkzeug.security import check_password_hash

from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection
from common.review_permissions import (PERM_METADATA, PERM_READ, PERM_STATUS, PERM_TEXT,
                                PERM_USERS, ROLE_PERMISSIONS, permissions_for)

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

log = logging.getLogger("review_api")

QDRANT_COLLECTION = "document_chunks"
INGEST_QUEUE = "da-indicizzare"

DATA_FOLDER = pathlib.Path(settings.data_folder)
PROCESSED_FOLDER = DATA_FOLDER / "processed"
WATCH_FOLDER = DATA_FOLDER / "watch"
INGESTION_ERROR_FOLDER = DATA_FOLDER / "ingestion" / "error"

# Chiavi di metadato generate dal MarkdownHeaderTextSplitter dell'ingest:
# variano da parent a parent e non sono metadati "del documento", quindi non
# sono modificabili e vanno preservate quando si riscrivono i metadati.
HEADER_KEY_RE = re.compile(r"^Header \d+$")
METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-. ]{0,63}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MAX_CONTENT_CHARS = 5_000_000
REVIEW_STATUSES = {"da_revisionare", "in_revisione", "revisionato"}

# --- Impostazioni (variabili d'ambiente) ------------------------------------
# REVIEW_SECRET_KEY firma i token di sessione: con una chiave assente o debole
# chiunque potrebbe forgiarne uno. Se manca, le rotte /review rispondono 503
# ma il resto dell'API (chat) continua a funzionare normalmente.
REVIEW_SECRET_KEY = os.environ.get("REVIEW_SECRET_KEY", "").strip()
REVIEW_ENABLED = len(REVIEW_SECRET_KEY) >= 32
SESSION_HOURS = int(os.environ.get("REVIEW_SESSION_HOURS") or 10)
FILE_TOKEN_MINUTES = int(os.environ.get("REVIEW_FILE_TOKEN_MINUTES") or 30)

bp = Blueprint("review", __name__, url_prefix="/review")

db_handler = MySQLLogHandler()
db_handler.setLevel(logging.WARNING)
log.addHandler(db_handler)

if REVIEW_ENABLED:
    _auth_serializer = URLSafeTimedSerializer(REVIEW_SECRET_KEY, salt="review-auth")
    _file_serializer = URLSafeTimedSerializer(REVIEW_SECRET_KEY, salt="review-file")
else:
    _auth_serializer = _file_serializer = None
    log.error("REVIEW_SECRET_KEY mancante o più corta di 32 caratteri: endpoint /review disabilitati.")

_qdrant = None


def qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)
    return _qdrant


class ApiError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


@bp.errorhandler(ApiError)
def _handle_api_error(e: ApiError):
    return jsonify({"error": e.message, **e.extra}), e.status


@bp.errorhandler(Exception)
def _handle_unexpected(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    log.error(f"Errore non gestito su {request.path}: {e}", exc_info=True)
    return jsonify({"error": "Errore interno del server"}), 500


# ==============================================================================
# DATABASE
# ==============================================================================

class Db:
    """Context manager su una connessione del pool condiviso."""

    def __enter__(self):
        self.conn = get_db_connection()
        if not self.conn:
            raise ApiError(503, "Database non disponibile")
        self.cur = self.conn.cursor(dictionary=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
            self.cur.close()
        finally:
            try:
                if self.conn.is_connected():
                    self.conn.close()
            except Exception:
                pass
        return False

    def all(self, sql, params=()):
        self.cur.execute(sql, params)
        return self.cur.fetchall()

    def one(self, sql, params=()):
        self.cur.execute(sql, params)
        row = self.cur.fetchone()
        # Consuma eventuali righe residue (mysql-connector lo pretende).
        self.cur.fetchall()
        return row

    def execute(self, sql, params=()):
        self.cur.execute(sql, params)
        return self.cur.rowcount

    def commit(self):
        self.conn.commit()


def _parse_json(value):
    if value is None:
        return {}
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _iso(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ==============================================================================
# AUTENTICAZIONE
# ==============================================================================

def _load_user(db, uid, token_version):
    """
    Utente attivo con ruoli e permessi, oppure 401. I ruoli si rileggono a
    ogni richiesta: una modifica ai ruoli vale subito, senza nuovo login.
    """
    user = db.one(
        "SELECT id, username, display_name, active, token_version "
        "FROM review_users WHERE id = %s",
        (uid,),
    )
    # token_version permette di invalidare tutte le sessioni di un utente
    # (cambio password, disattivazione) senza gestire una blacklist.
    if not user or not user["active"] or user["token_version"] != token_version:
        raise ApiError(401, "Utente non abilitato")
    _attach_roles(db, user)
    return user


def _attach_roles(db, user):
    rows = db.all("SELECT role FROM review_user_roles WHERE user_id = %s", (user["id"],))
    user["roles"] = sorted(r["role"] for r in rows)
    unknown = [r for r in user["roles"] if r not in ROLE_PERMISSIONS]
    if unknown:
        log.warning(f"Utente '{user['username']}': ruoli sconosciuti ignorati {unknown}")
    user["permissions"] = permissions_for(user["roles"])


def require_user(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise ApiError(401, "Autenticazione richiesta")
        try:
            data = _auth_serializer.loads(header[7:], max_age=SESSION_HOURS * 3600)
        except SignatureExpired:
            raise ApiError(401, "Sessione scaduta, effettua di nuovo l'accesso")
        except BadSignature:
            raise ApiError(401, "Token non valido")

        with Db() as db:
            g.user = _load_user(db, data.get("uid"), data.get("tv"))
        return fn(*args, **kwargs)
    return wrapper


def _check_permission(user, permission):
    if permission not in user["permissions"]:
        log.warning(f"Accesso negato a '{user['username']}' su {request.method} {request.path}: "
                    f"manca il permesso '{permission}'")
        raise ApiError(403, "Il tuo profilo non è abilitato a questa operazione")


def require_permission(permission):
    """Autentica l'utente e verifica che i suoi ruoli concedano `permission`."""
    def decorator(fn):
        @wraps(fn)
        @require_user
        def wrapper(*args, **kwargs):
            _check_permission(g.user, permission)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def _public_user(user):
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name") or user["username"],
        "roles": user["roles"],
        "permissions": sorted(user["permissions"]),
    }


@bp.before_request
def _check_enabled():
    if request.method == "OPTIONS":
        return None
    if not REVIEW_ENABLED:
        return jsonify({"error": "Funzione di revisione non configurata (REVIEW_SECRET_KEY)"}), 503
    return None


@bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or not password:
        raise ApiError(400, "Nome utente e password obbligatori")

    with Db() as db:
        user = db.one(
            "SELECT id, username, display_name, active, password_hash, token_version "
            "FROM review_users WHERE username = %s",
            (username,),
        )
        if not user or not user["active"] or not check_password_hash(user["password_hash"], password):
            log.warning(f"Login fallito per l'utente '{username}' da {request.remote_addr}")
            raise ApiError(401, "Credenziali non valide")
        _attach_roles(db, user)
        db.execute("UPDATE review_users SET last_login_at = NOW() WHERE id = %s", (user["id"],))
        db.commit()

    token = _auth_serializer.dumps({"uid": user["id"], "tv": user["token_version"]})
    return jsonify({
        "token": token,
        "expires_in": SESSION_HOURS * 3600,
        "user": _public_user(user),
    })


@bp.route("/auth/me", methods=["GET"])
@require_user
def me():
    return jsonify({"user": _public_user(g.user)})


@bp.route("/auth/logout", methods=["POST"])
@require_user
def logout():
    """
    Chiude la sessione invalidando il token anche lato server: senza questo
    un token copiato resterebbe valido fino alla scadenza. Si incrementa
    token_version, quindi si chiudono tutte le sessioni dell'utente (altre
    schede o postazioni) e i link ai file originali già rilasciati.
    """
    with Db() as db:
        db.execute("UPDATE review_users SET token_version = token_version + 1 WHERE id = %s",
                   (g.user["id"],))
        db.commit()
    log.info(f"Logout di '{g.user['username']}'.")
    return jsonify({"status": "logged_out"})


# ==============================================================================
# RISOLUZIONE DEI FILE SU DISCO
# ==============================================================================

def _safe_under(root: pathlib.Path, *parts: str) -> pathlib.Path:
    """Costruisce un path sotto root rifiutando qualunque traversal."""
    candidate = root.joinpath(*parts).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ApiError(400, "Percorso non valido")
    return candidate


def _original_file_path(topic_id, sub_topic_id, file_name) -> pathlib.Path | None:
    """
    Il converter archivia l'allegato in processed/<cartella del manifest>/<nome>.
    La cartella del manifest è topic/sub_topic[/sottocartelle], e le eventuali
    sottocartelle coincidono con il prefisso di file_name (manifest di
    direct.py, es. "preliminari/gennaio/test.pdf"); per Sicr@Web file_name è
    un nome semplice. In entrambi i casi: processed/topic/sub/file_name.
    """
    if not file_name:
        return None
    path = _safe_under(PROCESSED_FOLDER, topic_id, sub_topic_id, file_name)
    return path if path.is_file() else None


def _find_text_package(source, topic_id, sub_topic_id, file_name):
    """
    Trova il pacchetto (.md/.txt + .json manifest) che l'ingest ha archiviato
    in processed/ per questo documento. Il nome è "<stem>.md" (direct) oppure
    "<stem_manifest>_<stem>.md" (Sicr@Web), sempre nella stessa cartella del
    file originale: si cercano i candidati lì e si conferma con il campo
    "source" del manifest, che è univoco.

    Ritorna (text_path, manifest_path, manifest_dict) oppure (None, None, None).
    """
    if not file_name:
        return None, None, None
    rel = pathlib.PurePosixPath(file_name)
    folder = _safe_under(PROCESSED_FOLDER, topic_id, sub_topic_id, *rel.parent.parts)
    if not folder.is_dir():
        return None, None, None

    stem = rel.stem
    candidates = []
    for ext in (".md", ".txt"):
        exact = folder / f"{stem}{ext}"
        if exact.is_file():
            candidates.append(exact)
        candidates.extend(sorted(p for p in folder.glob(f"*_{glob_escape(stem)}{ext}") if p != exact))

    for text_path in candidates:
        manifest_path = text_path.with_suffix(".json")
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("source") == source:
            return text_path, manifest_path, manifest
    return None, None, None


def glob_escape(value: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", value)


# ==============================================================================
# CARICAMENTO DOCUMENTO
# ==============================================================================

def _doc_key_from_request(data=None):
    src = data if data is not None else request.args
    source = (src.get("source") or "").strip()
    topic_id = (src.get("topic_id") or "").strip()
    sub_topic_id = (src.get("sub_topic_id") or "").strip()
    if not source or not topic_id or not sub_topic_id:
        raise ApiError(400, "source, topic_id e sub_topic_id sono obbligatori")
    return source, topic_id, sub_topic_id


def _load_parents(db, source, topic_id, sub_topic_id):
    return db.all(
        "SELECT id, file_name, parent_index, content, metadata, created_at, updated_at "
        "FROM parent_documents WHERE source = %s AND topic_id = %s AND sub_topic_id = %s "
        "ORDER BY parent_index",
        (source, topic_id, sub_topic_id),
    )


def _document_metadata(parents):
    """Metadati del documento = metadati del parent 0 senza le chiavi Header N."""
    if not parents:
        return {}
    meta = _parse_json(parents[0]["metadata"])
    return {k: v for k, v in meta.items() if not HEADER_KEY_RE.match(k)}


def _reindex_state(db, source, topic_id, sub_topic_id, parents):
    """
    Stato dell'ultima correzione del testo:
      - "pending": inviata all'ingest, i parent non sono ancora stati rigenerati;
      - "error":   l'ingest ha spostato il pacchetto in ingestion/error;
      - None:      nessuna correzione in sospeso.
    I parent vengono cancellati e reinseriti dall'ingest, quindi il loro
    created_at più vecchio indica l'ultima indicizzazione completata.
    """
    last = db.one(
        "SELECT id, created_at, new_value, details FROM review_audit "
        "WHERE source = %s AND topic_id = %s AND sub_topic_id = %s AND action = 'content' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (source, topic_id, sub_topic_id),
    )
    if not last:
        return None, None
    indexed_at = min((p["created_at"] for p in parents), default=None)
    if indexed_at is not None and indexed_at >= last["created_at"]:
        return None, None

    details = _parse_json(last["details"])
    watch_rel = details.get("watch_manifest")
    if watch_rel:
        err_json = INGESTION_ERROR_FOLDER / watch_rel
        if err_json.is_file():
            try:
                err = json.loads(err_json.read_text(encoding="utf-8")).get("_ingestion_error")
            except Exception:
                err = "errore sconosciuto"
            return {"state": "error", "since": _iso(last["created_at"]), "error": err}, last
    return {"state": "pending", "since": _iso(last["created_at"])}, last


def _load_document(db, source, topic_id, sub_topic_id, include_content=True):
    parents = _load_parents(db, source, topic_id, sub_topic_id)
    if not parents:
        raise ApiError(404, "Documento non trovato")

    file_name = parents[0]["file_name"]
    text_path, manifest_path, manifest = _find_text_package(source, topic_id, sub_topic_id, file_name)
    reindex, last_audit = _reindex_state(db, source, topic_id, sub_topic_id, parents)

    content = None
    content_origin = None
    if include_content:
        if reindex and last_audit and last_audit["new_value"] is not None:
            # Correzione già inviata ma non ancora indicizzata: si mostra la
            # versione corretta, altrimenti il revisore vedrebbe sparire il
            # proprio lavoro fino al termine dell'ingest.
            content = last_audit["new_value"]
            content_origin = "pending_review"
        elif text_path:
            content = text_path.read_text(encoding="utf-8", errors="replace")
            content_origin = "file"
        else:
            # Pacchetto non trovato su disco (es. dati migrati): si ricompone
            # il testo dai parent. L'ingest li divide senza overlap, quindi la
            # concatenazione è fedele salvo gli spazi ai punti di taglio.
            content = "\n\n".join(p["content"] for p in parents)
            content_origin = "parents"

    original = _original_file_path(topic_id, sub_topic_id, file_name)
    status = db.one(
        "SELECT status, note, updated_by, updated_at FROM review_status "
        "WHERE source = %s AND topic_id = %s AND sub_topic_id = %s",
        (source, topic_id, sub_topic_id),
    )

    doc = {
        "source": source,
        "topic_id": topic_id,
        "sub_topic_id": sub_topic_id,
        "file_name": file_name,
        "parent_count": len(parents),
        "indexed_at": _iso(min(p["created_at"] for p in parents)),
        "metadata": _document_metadata(parents),
        "protected_keys": sorted(settings.protected_keys),
        "original_file": None,
        "review_status": {
            "status": status["status"] if status else "da_revisionare",
            "note": status["note"] if status else None,
            "updated_by": status["updated_by"] if status else None,
            "updated_at": _iso(status["updated_at"]) if status else None,
        },
        "reindex": reindex,
        "has_text_package": text_path is not None,
    }
    if original:
        mime, _ = mimetypes.guess_type(original.name)
        doc["original_file"] = {
            "name": original.name,
            "size": original.stat().st_size,
            "mime_type": mime or "application/octet-stream",
        }
    if include_content:
        doc["content"] = content
        doc["content_origin"] = content_origin
        doc["content_hash"] = _sha256(content)
    return doc, parents, (text_path, manifest_path, manifest)


def _audit(db, action, source, topic_id, sub_topic_id, old_value, new_value, note=None, details=None):
    db.execute(
        "INSERT INTO review_audit (user_id, username, action, source, topic_id, sub_topic_id, "
        "old_value, new_value, note, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            g.user["id"], g.user["username"], action, source, topic_id, sub_topic_id,
            old_value, new_value, note, json.dumps(details or {}, ensure_ascii=False),
        ),
    )


def _set_status(db, source, topic_id, sub_topic_id, status, note=None):
    db.execute(
        "INSERT INTO review_status (source, topic_id, sub_topic_id, status, note, updated_by) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON DUPLICATE KEY UPDATE status = VALUES(status), note = VALUES(note), "
        "updated_by = VALUES(updated_by), updated_at = CURRENT_TIMESTAMP",
        (source, topic_id, sub_topic_id, status, note, g.user["username"]),
    )


# ==============================================================================
# ENDPOINT: NAVIGAZIONE
# ==============================================================================

@bp.route("/topics", methods=["GET"])
@require_permission(PERM_READ)
def list_topics():
    with Db() as db:
        topics = db.all("SELECT topic_id, description FROM topics ORDER BY topic_id")
        subs = db.all(
            "SELECT topic_id, sub_topic_id, description FROM sub_topics ORDER BY topic_id, sub_topic_id"
        )
    by_topic = {}
    for s in subs:
        by_topic.setdefault(s["topic_id"], []).append(
            {"id": s["sub_topic_id"], "description": s["description"]}
        )
    return jsonify({
        "topics": [
            {"id": t["topic_id"], "description": t["description"], "sub_topics": by_topic.get(t["topic_id"], [])}
            for t in topics
        ]
    })


@bp.route("/documents", methods=["GET"])
@require_permission(PERM_READ)
def list_documents():
    topic_id = request.args.get("topic_id", "").strip()
    sub_topic_id = request.args.get("sub_topic_id", "").strip()
    query = request.args.get("q", "").strip()
    status = request.args.get("status", "").strip()
    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 25))))
    except ValueError:
        raise ApiError(400, "page e page_size devono essere numeri")

    # Una riga per documento: il parent 0 esiste sempre e porta i metadati.
    where = ["p.parent_index = 0"]
    params = []
    if topic_id:
        where.append("p.topic_id = %s")
        params.append(topic_id)
    if sub_topic_id:
        where.append("p.sub_topic_id = %s")
        params.append(sub_topic_id)
    if query:
        like = f"%{query}%"
        where.append("(p.source LIKE %s OR p.file_name LIKE %s OR CAST(p.metadata AS CHAR) LIKE %s)")
        params += [like, like, like]
    if status:
        if status not in REVIEW_STATUSES:
            raise ApiError(400, "Stato non valido")
        if status == "da_revisionare":
            where.append("(s.status IS NULL OR s.status = 'da_revisionare')")
        else:
            where.append("s.status = %s")
            params.append(status)

    base = (
        "FROM parent_documents p "
        "LEFT JOIN review_status s ON s.source = p.source AND s.topic_id = p.topic_id "
        "AND s.sub_topic_id = p.sub_topic_id "
        "WHERE " + " AND ".join(where)
    )

    with Db() as db:
        total = db.one(f"SELECT COUNT(*) AS n {base}", tuple(params))["n"]
        rows = db.all(
            f"SELECT p.source, p.topic_id, p.sub_topic_id, p.file_name, p.metadata, p.created_at, "
            f"s.status, s.updated_by, s.updated_at AS status_updated_at "
            f"{base} ORDER BY p.created_at DESC, p.source LIMIT %s OFFSET %s",
            tuple(params) + (page_size, (page - 1) * page_size),
        )

    items = []
    for r in rows:
        meta = _parse_json(r["metadata"])
        items.append({
            "source": r["source"],
            "topic_id": r["topic_id"],
            "sub_topic_id": r["sub_topic_id"],
            "file_name": r["file_name"],
            "title": meta.get("oggetto") or pathlib.PurePosixPath(r["file_name"] or r["source"]).name,
            "anno": meta.get("anno"),
            "numero": meta.get("numero"),
            "data": meta.get("data"),
            "indexed_at": _iso(r["created_at"]),
            "status": r["status"] or "da_revisionare",
            "status_updated_by": r["updated_by"],
            "status_updated_at": _iso(r["status_updated_at"]),
        })
    return jsonify({"items": items, "total": total, "page": page, "page_size": page_size})


@bp.route("/document", methods=["GET"])
@require_permission(PERM_READ)
def get_document():
    source, topic_id, sub_topic_id = _doc_key_from_request()
    with Db() as db:
        doc, _, _ = _load_document(db, source, topic_id, sub_topic_id)
    if doc["original_file"]:
        token = _file_serializer.dumps({
            "s": source, "t": topic_id, "st": sub_topic_id,
            "uid": g.user["id"], "tv": g.user["token_version"],
        })
        doc["original_file"]["token"] = token
    return jsonify(doc)


@bp.route("/document/file", methods=["GET"])
def get_document_file():
    """
    Il file originale viene aperto dal browser (iframe/tab), che non può
    inviare l'header Authorization: si usa un token firmato a breve scadenza
    rilasciato da GET /document a un utente già autenticato. L'utente viene
    comunque ricontrollato: un link ancora valido non apre il file a chi nel
    frattempo è stato disattivato o ha perso il permesso di lettura.
    """
    token = request.args.get("token", "")
    try:
        data = _file_serializer.loads(token, max_age=FILE_TOKEN_MINUTES * 60)
    except SignatureExpired:
        raise ApiError(401, "Link scaduto, ricarica il documento")
    except BadSignature:
        raise ApiError(401, "Link non valido")

    with Db() as db:
        _check_permission(_load_user(db, data.get("uid"), data.get("tv")), PERM_READ)
        row = db.one(
            "SELECT file_name FROM parent_documents WHERE source = %s AND topic_id = %s "
            "AND sub_topic_id = %s AND parent_index = 0 LIMIT 1",
            (data["s"], data["t"], data["st"]),
        )
    if not row:
        raise ApiError(404, "Documento non trovato")
    path = _original_file_path(data["t"], data["st"], row["file_name"])
    if not path:
        raise ApiError(404, "File originale non presente in archivio")

    download = request.args.get("download") == "1"
    mime, _ = mimetypes.guess_type(path.name)
    response = send_file(
        path, mimetype=mime or "application/octet-stream",
        as_attachment=download, download_name=path.name, max_age=0,
    )
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@bp.route("/document/history", methods=["GET"])
@require_permission(PERM_READ)
def get_history():
    source, topic_id, sub_topic_id = _doc_key_from_request()
    with Db() as db:
        rows = db.all(
            "SELECT id, created_at, username, action, note, details, "
            "CHAR_LENGTH(old_value) AS old_len, CHAR_LENGTH(new_value) AS new_len "
            "FROM review_audit WHERE source = %s AND topic_id = %s AND sub_topic_id = %s "
            "ORDER BY created_at DESC, id DESC LIMIT 200",
            (source, topic_id, sub_topic_id),
        )
    return jsonify({"items": [
        {
            "id": r["id"],
            "created_at": _iso(r["created_at"]),
            "username": r["username"],
            "action": r["action"],
            "note": r["note"],
            "details": _parse_json(r["details"]),
            "old_length": r["old_len"],
            "new_length": r["new_len"],
        }
        for r in rows
    ]})


@bp.route("/document/history/<int:audit_id>", methods=["GET"])
@require_permission(PERM_READ)
def get_history_entry(audit_id):
    with Db() as db:
        r = db.one(
            "SELECT id, created_at, username, action, source, topic_id, sub_topic_id, "
            "old_value, new_value, note, details FROM review_audit WHERE id = %s",
            (audit_id,),
        )
    if not r:
        raise ApiError(404, "Voce di storico non trovata")
    return jsonify({**r, "created_at": _iso(r["created_at"]), "details": _parse_json(r["details"])})


# ==============================================================================
# ENDPOINT: MODIFICHE
# ==============================================================================

def _publish_to_ingest(manifest_rel: str):
    params = pika.ConnectionParameters(
        host=settings.broker_host,
        port=settings.broker_port,
        credentials=pika.PlainCredentials(settings.broker_username, settings.broker_password),
        connection_attempts=3,
        retry_delay=2,
    )
    connection = pika.BlockingConnection(params)
    try:
        channel = connection.channel()
        channel.basic_publish(
            exchange="",
            routing_key=INGEST_QUEUE,
            body=json.dumps({"json_manifest_path": manifest_rel}).encode(),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
        )
    finally:
        connection.close()


@bp.route("/document/content", methods=["PUT"])
@require_permission(PERM_TEXT)
def update_content():
    """
    Salva il testo corretto e chiede all'ingest di re-indicizzare il documento.

    Il pacchetto (.md + manifest .json) viene scritto in watch/ nella stessa
    posizione relativa che ha in processed/: l'ingest, a fine lavoro, lo
    sposta in processed/ sovrascrivendo la versione precedente, cancella i
    vecchi parent/chunk (stesso source/topic/sub_topic) e crea i nuovi.

    Concorrenza ottimistica: il client invia l'hash del testo che ha caricato;
    se nel frattempo qualcun altro ha salvato, la richiesta viene rifiutata.
    """
    data = request.get_json(silent=True) or {}
    source, topic_id, sub_topic_id = _doc_key_from_request(data)
    new_content = data.get("content")
    base_hash = data.get("base_hash")
    note = (data.get("note") or "").strip() or None
    # "Segna come revisionato" cambia anche lo stato: serve il relativo permesso.
    if data.get("mark_reviewed"):
        _check_permission(g.user, PERM_STATUS)

    if not isinstance(new_content, str) or not new_content.strip():
        raise ApiError(400, "Il testo non può essere vuoto")
    if len(new_content) > MAX_CONTENT_CHARS:
        raise ApiError(413, "Testo troppo lungo")
    new_content = new_content.replace("\r\n", "\n")

    with Db() as db:
        doc, parents, (text_path, manifest_path, manifest) = _load_document(db, source, topic_id, sub_topic_id)
        if base_hash and base_hash != doc["content_hash"]:
            raise ApiError(409, "Il documento è stato modificato da un altro utente. Ricaricalo prima di salvare.",
                           current_hash=doc["content_hash"])
        if new_content == doc["content"]:
            raise ApiError(400, "Nessuna modifica al testo")

        file_name = doc["file_name"]
        # Nome e posizione del pacchetto: quelli esistenti se trovati,
        # altrimenti si ricostruiscono come farebbe converter.py.
        if text_path:
            rel_dir = text_path.parent.relative_to(PROCESSED_FOLDER / topic_id / sub_topic_id)
            stem = text_path.stem
        else:
            rel_dir = pathlib.PurePosixPath(file_name or "").parent
            stem = pathlib.PurePosixPath(file_name or source.replace("/", "_")).stem or uuid.uuid4().hex
            manifest = None

        new_manifest = dict(manifest or {})
        new_manifest["source"] = source
        new_manifest["files"] = new_manifest.get("files") or ([file_name] if file_name else [])
        # I metadati attuali su MySQL sono la verità: includono le eventuali
        # modifiche fatte dai revisori dopo l'ingestione originale.
        new_manifest["metadati"] = doc["metadata"]
        new_manifest.pop("_ingestion_error", None)
        if not new_manifest["files"]:
            raise ApiError(422, "Impossibile ricostruire il manifest: file_name mancante")

        watch_dir = _safe_under(WATCH_FOLDER, topic_id, sub_topic_id, *pathlib.PurePosixPath(str(rel_dir)).parts)
        watch_dir.mkdir(parents=True, exist_ok=True)
        watch_md = watch_dir / f"{stem}.md"
        watch_json = watch_dir / f"{stem}.json"
        for leftover in (watch_dir / f"{stem}.txt",):
            if leftover.exists():
                leftover.unlink()

        # Se l'ultimo tentativo era finito in errore, ne rimuoviamo il
        # pacchetto: lo stato "error" si ricalcolerebbe su un file vecchio.
        manifest_rel = str(watch_json.relative_to(WATCH_FOLDER))
        for stale in (INGESTION_ERROR_FOLDER / manifest_rel,
                      (INGESTION_ERROR_FOLDER / manifest_rel).with_suffix(".md")):
            if stale.exists():
                stale.unlink()

        # Scrittura atomica: prima il testo, poi il manifest (è il JSON che
        # fa partire l'ingest, che si aspetta di trovare già il .md).
        tmp_md = watch_md.with_suffix(".md.tmp")
        tmp_md.write_text(new_content, encoding="utf-8")
        tmp_md.replace(watch_md)
        tmp_json = watch_json.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(new_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_json.replace(watch_json)

        _audit(db, "content", source, topic_id, sub_topic_id, doc["content"], new_content, note,
               {"watch_manifest": manifest_rel, "content_origin": doc["content_origin"],
                "old_hash": doc["content_hash"], "new_hash": _sha256(new_content)})
        if data.get("mark_reviewed") and doc["review_status"]["status"] != "revisionato":
            _set_status(db, source, topic_id, sub_topic_id, "revisionato", note)
            _audit(db, "status", source, topic_id, sub_topic_id, doc["review_status"]["status"],
                   "revisionato", note, {"from": doc["review_status"]["status"], "to": "revisionato"})

        try:
            _publish_to_ingest(manifest_rel)
        except Exception as e:
            log.error(f"Pubblicazione su '{INGEST_QUEUE}' fallita per {source}: {e}", exc_info=True)
            for p in (watch_md, watch_json):
                if p.exists():
                    p.unlink()
            raise ApiError(503, "Coda di indicizzazione non raggiungibile: modifica non salvata. Riprova.")
        db.commit()

    log.info(f"Testo corretto da '{g.user['username']}' per {source} ({topic_id}/{sub_topic_id}): reindicizzazione richiesta.")
    with Db() as db:
        doc, _, _ = _load_document(db, source, topic_id, sub_topic_id)
    return jsonify({"status": "reindexing", "document": doc}), 202


def _validate_metadata(meta):
    if not isinstance(meta, dict):
        raise ApiError(400, "metadata deve essere un oggetto")
    clean = {}
    errors = []
    for key, value in meta.items():
        k = str(key).strip()
        if not METADATA_KEY_RE.match(k):
            errors.append(f"Nome non valido: '{key}' (max 64 caratteri: lettere, cifre, _ - . spazio)")
            continue
        if k in settings.protected_keys or HEADER_KEY_RE.match(k):
            errors.append(f"'{k}' è una chiave di sistema e non può essere modificata")
            continue
        if isinstance(value, list):
            if not all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
                errors.append(f"'{k}': le liste possono contenere solo valori semplici")
                continue
        elif not (isinstance(value, (str, int, float, bool)) or value is None):
            errors.append(f"'{k}': valore non supportato")
            continue
        if isinstance(value, str):
            value = value.strip()
            # I campi data sono filtrati dalla chat come stringhe ISO
            # (confronto lessicografico): un formato diverso li renderebbe
            # invisibili ai filtri temporali.
            if k.startswith("data") and value and not ISO_DATE_RE.match(value):
                errors.append(f"'{k}': le date devono essere nel formato AAAA-MM-GG")
                continue
        clean[k] = value
    if errors:
        raise ApiError(422, "Metadati non validi", details=errors)
    return clean


@bp.route("/document/metadata", methods=["PUT"])
@require_permission(PERM_METADATA)
def update_metadata():
    """
    Sostituisce l'insieme dei metadati del documento.

    I metadati vivono in due posti, entrambi aggiornati qui:
      - parent_documents.metadata (MySQL): per ogni parent si conservano le
        chiavi "Header N" proprie del parent e si sostituisce il resto;
      - payload dei chunk su Qdrant: set_payload per le chiavi nuove/cambiate,
        delete_payload per quelle rimosse. Nessun embedding da ricalcolare.
    Il manifest in processed/ viene allineato, così un'eventuale futura
    re-indicizzazione non riporta indietro i valori.
    """
    data = request.get_json(silent=True) or {}
    source, topic_id, sub_topic_id = _doc_key_from_request(data)
    new_meta = _validate_metadata(data.get("metadata"))
    note = (data.get("note") or "").strip() or None

    with Db() as db:
        doc, parents, (_, manifest_path, manifest) = _load_document(db, source, topic_id, sub_topic_id,
                                                                     include_content=False)
        if doc["reindex"] and doc["reindex"]["state"] == "pending":
            raise ApiError(409, "Il documento è in re-indicizzazione: attendi il completamento prima di modificare i metadati.")

        old_meta = doc["metadata"]
        if new_meta == old_meta:
            raise ApiError(400, "Nessuna modifica ai metadati")
        removed = sorted(set(old_meta) - set(new_meta))
        changed = {k: v for k, v in new_meta.items() if old_meta.get(k, object()) != v}

        for p in parents:
            current = _parse_json(p["metadata"])
            headers = {k: v for k, v in current.items() if HEADER_KEY_RE.match(k)}
            db.execute(
                "UPDATE parent_documents SET metadata = %s WHERE id = %s",
                (json.dumps({**headers, **new_meta}, ensure_ascii=False), p["id"]),
            )

        doc_filter = models.Filter(must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=source)),
            models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
            models.FieldCondition(key="sub_topic_id", match=models.MatchValue(value=sub_topic_id)),
        ])
        try:
            if changed:
                qdrant().set_payload(collection_name=QDRANT_COLLECTION, payload=changed,
                                     points=doc_filter, wait=True)
            if removed:
                qdrant().delete_payload(collection_name=QDRANT_COLLECTION, keys=removed,
                                        points=doc_filter, wait=True)
        except Exception as e:
            # MySQL non è ancora committato: si annulla tutto. Qdrant può
            # essere rimasto aggiornato a metà; il retry del revisore lo
            # riallinea, perché set/delete_payload sono idempotenti.
            log.error(f"Aggiornamento payload Qdrant fallito per {source}: {e}", exc_info=True)
            raise ApiError(503, "Archivio vettoriale non raggiungibile: metadati non salvati. Riprova.")

        _audit(db, "metadata", source, topic_id, sub_topic_id,
               json.dumps(old_meta, ensure_ascii=False), json.dumps(new_meta, ensure_ascii=False), note,
               {"changed": sorted(changed), "removed": removed,
                "added": sorted(set(new_meta) - set(old_meta))})
        db.commit()

    if manifest_path:
        try:
            manifest["metadati"] = new_meta
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(manifest_path)
        except Exception as e:
            log.warning(f"Manifest {manifest_path} non aggiornato: {e}")

    log.info(f"Metadati aggiornati da '{g.user['username']}' per {source}: "
             f"modificati={sorted(changed)} rimossi={removed}")
    with Db() as db:
        doc, _, _ = _load_document(db, source, topic_id, sub_topic_id)
    return jsonify({"status": "saved", "document": doc})


@bp.route("/document/status", methods=["PUT"])
@require_permission(PERM_STATUS)
def update_status():
    data = request.get_json(silent=True) or {}
    source, topic_id, sub_topic_id = _doc_key_from_request(data)
    status = data.get("status")
    if status not in REVIEW_STATUSES:
        raise ApiError(400, f"Stato non valido. Valori ammessi: {', '.join(sorted(REVIEW_STATUSES))}")
    note = (data.get("note") or "").strip() or None
    with Db() as db:
        exists = db.one(
            "SELECT 1 AS ok FROM parent_documents WHERE source = %s AND topic_id = %s "
            "AND sub_topic_id = %s LIMIT 1",
            (source, topic_id, sub_topic_id),
        )
        if not exists:
            raise ApiError(404, "Documento non trovato")
        old = db.one(
            "SELECT status FROM review_status WHERE source = %s AND topic_id = %s AND sub_topic_id = %s",
            (source, topic_id, sub_topic_id),
        )
        _set_status(db, source, topic_id, sub_topic_id, status, note)
        previous = old["status"] if old else "da_revisionare"
        _audit(db, "status", source, topic_id, sub_topic_id, previous, status, note,
               {"from": previous, "to": status})
        db.commit()
    return jsonify({"status": status})


# ==============================================================================
# ENDPOINT: AMMINISTRAZIONE UTENTI
# ==============================================================================

@bp.route("/users", methods=["GET"])
@require_permission(PERM_USERS)
def list_users():
    with Db() as db:
        rows = db.all(
            "SELECT id, username, display_name, active, created_at, last_login_at "
            "FROM review_users ORDER BY username"
        )
        roles = {}
        for r in db.all("SELECT user_id, role FROM review_user_roles ORDER BY role"):
            roles.setdefault(r["user_id"], []).append(r["role"])
    return jsonify({"items": [
        {**r, "active": bool(r["active"]), "created_at": _iso(r["created_at"]),
         "last_login_at": _iso(r["last_login_at"]), "roles": roles.get(r["id"], [])}
        for r in rows
    ]})
