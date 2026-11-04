"""
Amministrazione dal backend, usata dall'area "Amministrazione" dell'app di
revisione (rotte sotto /review/admin, con il token di sessione dell'utente):

  - utenti: creazione, ruoli, archivi, disattivazione, password, sblocco del login
  - archivi (topic) e sottocategorie
  - siti della chat: un nuovo sito è operativo appena il suo dominio punta al sistema
  - credenziali dei sistemi esterni

utils/manage_review_users.py e utils/manage_api_keys.py restano per il primo
amministratore e per lavorare da riga di comando.
"""

import base64
import binascii
import json
import logging
import pathlib
import re
from datetime import date

from flask import Blueprint, g, jsonify, request
from pydantic import BaseModel
from werkzeug.security import generate_password_hash

from common.config import settings
from common.review_permissions import PERM_ARCHIVES, PERM_USERS, ROLES, permissions_for
from common.site_home import buttons_from_db, clean_buttons, clean_html
from common.utility import new_external_key
from review_routes import ApiError, Db, _iso, require_permission
from site_routes import forget_site

log = logging.getLogger("ADMIN")

bp = Blueprint("admin", __name__, url_prefix="/admin")

MIN_PASSWORD_LENGTH = 10
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@\-]{2,100}$")
# Finiscono nei percorsi delle cartelle dei documenti.
ARCHIVE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,99}$")
KEY_NAME_RE = re.compile(r"^[A-Za-z0-9_.\- ]{1,100}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ICON_BYTES = 1024 * 1024
MAX_HOME_CHARS = 200_000
PROMPTS_DIR = pathlib.Path(__file__).resolve().parent / "prompts"


def _json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError(400, "Corpo JSON non valido")
    return data


def _text(data, key, *, required=False, max_len=255):
    value = data.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ApiError(422, f"Il campo '{key}' è obbligatorio")
        return None
    if not isinstance(value, str):
        raise ApiError(422, f"Il campo '{key}' deve essere un testo")
    value = value.strip()
    if len(value) > max_len:
        raise ApiError(422, f"Il campo '{key}' supera i {max_len} caratteri")
    return value


def _flag(data, key, default):
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ApiError(422, f"Il campo '{key}' deve essere vero o falso")
    return value


def _list(data, key):
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ApiError(422, f"Il campo '{key}' deve essere un elenco")
    return sorted(set(value))


def _check_topics_exist(db, topics):
    for topic in topics:
        if not db.one("SELECT 1 AS ok FROM topics WHERE topic_id = %s", (topic,)):
            raise ApiError(422, f"Archivio '{topic}' inesistente")


def _audit(message):
    log.info(f"{g.user['username']}: {message}")


# ==============================================================================
# UTENTI (l'elenco è GET /review/users)
# ==============================================================================

def _check_roles(roles):
    unknown = [r for r in roles if r not in ROLES]
    if unknown:
        raise ApiError(422, f"Ruoli sconosciuti: {', '.join(unknown)}")


def _check_password(password):
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ApiError(422, f"La password deve avere almeno {MIN_PASSWORD_LENGTH} caratteri")


def _set_user_links(db, user_id, roles, topics):
    db.execute("DELETE FROM review_user_roles WHERE user_id = %s", (user_id,))
    for role in roles:
        db.execute("INSERT INTO review_user_roles (user_id, role) VALUES (%s, %s)", (user_id, role))
    db.execute("DELETE FROM review_user_topics WHERE user_id = %s", (user_id,))
    for topic in topics:
        db.execute("INSERT INTO review_user_topics (user_id, topic_id) VALUES (%s, %s)", (user_id, topic))


def _check_user_admin_remains(db):
    """Senza un amministratore attivo l'area si chiuderebbe da sola."""
    rows = db.all("SELECT r.user_id, r.role FROM review_user_roles r "
                  "JOIN review_users u ON u.id = r.user_id WHERE u.active = TRUE")
    if not any(PERM_USERS in permissions_for([r["role"]]) for r in rows):
        raise ApiError(409, "Deve restare almeno un amministratore attivo")


@bp.route("/roles", methods=["GET"])
@require_permission(PERM_USERS)
def list_roles():
    return jsonify({"items": list(ROLES)})


@bp.route("/users", methods=["POST"])
@require_permission(PERM_USERS)
def create_user():
    data = _json_body()
    username = _text(data, "username", required=True, max_len=100)
    if not USERNAME_RE.match(username):
        raise ApiError(422, "Nome utente non valido: 2-100 caratteri fra lettere, cifre e _ . @ -")
    password = data.get("password")
    _check_password(password)
    roles, topics = _list(data, "roles"), _list(data, "topics")
    _check_roles(roles)
    with Db() as db:
        if db.one("SELECT 1 AS ok FROM review_users WHERE username = %s", (username,)):
            raise ApiError(409, f"L'utente '{username}' esiste già")
        _check_topics_exist(db, topics)
        db.execute("INSERT INTO review_users (username, display_name, password_hash) VALUES (%s, %s, %s)",
                   (username, _text(data, "display_name") or username, generate_password_hash(password)))
        user_id = db.cur.lastrowid
        _set_user_links(db, user_id, roles, topics)
        db.commit()
    _audit(f"creato l'utente '{username}' (ruoli {roles}, archivi {topics})")
    return jsonify({"id": user_id}), 201


@bp.route("/users/<int:user_id>", methods=["PUT"])
@require_permission(PERM_USERS)
def update_user(user_id):
    data = _json_body()
    roles, topics = _list(data, "roles"), _list(data, "topics")
    _check_roles(roles)
    active = _flag(data, "active", True)
    if user_id == g.user["id"] and not active:
        raise ApiError(409, "Non puoi disattivare il tuo utente")
    with Db() as db:
        user = db.one("SELECT username, active FROM review_users WHERE id = %s", (user_id,))
        if not user:
            raise ApiError(404, "Utente non trovato")
        _check_topics_exist(db, topics)
        # Disattivare invalida le sessioni aperte, come il cambio password.
        bump = 1 if bool(user["active"]) and not active else 0
        db.execute("UPDATE review_users SET display_name = %s, active = %s, token_version = token_version + %s "
                   "WHERE id = %s", (_text(data, "display_name") or user["username"], active, bump, user_id))
        _set_user_links(db, user_id, roles, topics)
        _check_user_admin_remains(db)
        db.commit()
    _audit(f"modificato l'utente '{user['username']}' (attivo {active}, ruoli {roles}, archivi {topics})")
    return jsonify({"ok": True})


@bp.route("/users/<int:user_id>/password", methods=["POST"])
@require_permission(PERM_USERS)
def set_user_password(user_id):
    password = _json_body().get("password")
    _check_password(password)
    with Db() as db:
        user = db.one("SELECT username FROM review_users WHERE id = %s", (user_id,))
        if not user:
            raise ApiError(404, "Utente non trovato")
        db.execute("UPDATE review_users SET password_hash = %s, token_version = token_version + 1 WHERE id = %s",
                   (generate_password_hash(password), user_id))
        db.commit()
    _audit(f"cambiata la password di '{user['username']}'")
    return jsonify({"ok": True})


@bp.route("/users/<int:user_id>/unlock", methods=["POST"])
@require_permission(PERM_USERS)
def unlock_user(user_id):
    with Db() as db:
        user = db.one("SELECT username FROM review_users WHERE id = %s", (user_id,))
        if not user:
            raise ApiError(404, "Utente non trovato")
        db.execute("DELETE FROM review_login_failures WHERE lkey = %s", (f"user:{user['username'].lower()}",))
        db.commit()
    _audit(f"sbloccato il login di '{user['username']}'")
    return jsonify({"ok": True})


# ==============================================================================
# ARCHIVI (TOPIC) E SOTTOCATEGORIE
# ==============================================================================

def _available_prompts():
    reserved = {"grader", "query_rewriter"}
    return sorted(p.name for p in PROMPTS_DIR.iterdir() if p.is_file() and p.name not in reserved)


def _topic_fields(data):
    prompt = _text(data, "prompt", max_len=100)
    if prompt and prompt not in _available_prompts():
        raise ApiError(422, f"Prompt '{prompt}' inesistente")
    return (_text(data, "description", required=True, max_len=2000), _text(data, "aliases", max_len=2000),
            prompt, _flag(data, "public_access", False))


def _int_or_none(data, key, default=None):
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ApiError(422, f"Il campo '{key}' deve essere un numero intero positivo")
    return value


def _sub_topic_fields(data):
    return (_text(data, "description", required=True), _text(data, "description_long") or "",
            _int_or_none(data, "chunk_size"), _int_or_none(data, "chunk_overlap", 50),
            _int_or_none(data, "parent_chunk_size", 1500), _flag(data, "use_markdown_splitter", True))


@bp.route("/topics", methods=["GET"])
@require_permission(PERM_ARCHIVES)
def list_topics():
    with Db() as db:
        topics = db.all("SELECT topic_id, description, aliases, prompt, public_access FROM topics ORDER BY topic_id")
        subs = db.all("SELECT topic_id, sub_topic_id, description, description_long, chunk_size, chunk_overlap, "
                      "parent_chunk_size, use_markdown_splitter FROM sub_topics ORDER BY topic_id, sub_topic_id")
    by_topic = {}
    for s in subs:
        by_topic.setdefault(s["topic_id"], []).append(
            {**s, "use_markdown_splitter": bool(s["use_markdown_splitter"])})
    return jsonify({"prompts": _available_prompts(), "items": [
        {**t, "public_access": bool(t["public_access"]), "sub_topics": by_topic.get(t["topic_id"], [])}
        for t in topics
    ]})


@bp.route("/topics/<topic_id>", methods=["PUT"])
@require_permission(PERM_ARCHIVES)
def update_topic(topic_id):
    fields = _topic_fields(_json_body())
    with Db() as db:
        if not db.one("SELECT 1 AS ok FROM topics WHERE topic_id = %s", (topic_id,)):
            raise ApiError(404, "Archivio non trovato")
        db.execute("UPDATE topics SET description = %s, aliases = %s, prompt = %s, public_access = %s "
                   "WHERE topic_id = %s", (*fields, topic_id))
        db.commit()
    _audit(f"modificato l'archivio '{topic_id}'")
    return jsonify({"ok": True})


@bp.route("/topics/<topic_id>/subtopics", methods=["POST"])
@require_permission(PERM_ARCHIVES)
def create_sub_topic(topic_id):
    data = _json_body()
    sub_topic_id = _text(data, "sub_topic_id", required=True, max_len=100)
    if not ARCHIVE_ID_RE.match(sub_topic_id):
        raise ApiError(422, "Identificativo non valido: lettere minuscole, cifre, _ e -")
    fields = _sub_topic_fields(data)
    with Db() as db:
        if not db.one("SELECT 1 AS ok FROM topics WHERE topic_id = %s", (topic_id,)):
            raise ApiError(404, "Archivio non trovato")
        if db.one("SELECT 1 AS ok FROM sub_topics WHERE topic_id = %s AND sub_topic_id = %s", (topic_id, sub_topic_id)):
            raise ApiError(409, f"La sottocategoria '{sub_topic_id}' esiste già")
        db.execute("INSERT INTO sub_topics (topic_id, sub_topic_id, description, description_long, chunk_size, "
                   "chunk_overlap, parent_chunk_size, use_markdown_splitter) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                   (topic_id, sub_topic_id, *fields))
        db.commit()
    _audit(f"creata la sottocategoria '{topic_id}/{sub_topic_id}'")
    return jsonify({"sub_topic_id": sub_topic_id}), 201


@bp.route("/topics/<topic_id>/subtopics/<sub_topic_id>", methods=["PUT"])
@require_permission(PERM_ARCHIVES)
def update_sub_topic(topic_id, sub_topic_id):
    fields = _sub_topic_fields(_json_body())
    with Db() as db:
        if not db.one("SELECT 1 AS ok FROM sub_topics WHERE topic_id = %s AND sub_topic_id = %s", (topic_id, sub_topic_id)):
            raise ApiError(404, "Sottocategoria non trovata")
        db.execute("UPDATE sub_topics SET description = %s, description_long = %s, chunk_size = %s, chunk_overlap = %s, "
                   "parent_chunk_size = %s, use_markdown_splitter = %s WHERE topic_id = %s AND sub_topic_id = %s",
                   (*fields, topic_id, sub_topic_id))
        db.commit()
    _audit(f"modificata la sottocategoria '{topic_id}/{sub_topic_id}'")
    return jsonify({"ok": True})


@bp.route("/topics/<topic_id>/subtopics/<sub_topic_id>", methods=["DELETE"])
@require_permission(PERM_ARCHIVES)
def delete_sub_topic(topic_id, sub_topic_id):
    with Db() as db:
        if db.one("SELECT 1 AS ok FROM parent_documents WHERE topic_id = %s AND sub_topic_id = %s LIMIT 1",
                  (topic_id, sub_topic_id)):
            raise ApiError(409, "La sottocategoria contiene documenti indicizzati: non si può eliminare")
        if not db.execute("DELETE FROM sub_topics WHERE topic_id = %s AND sub_topic_id = %s", (topic_id, sub_topic_id)):
            raise ApiError(404, "Sottocategoria non trovata")
        db.commit()
    _audit(f"eliminata la sottocategoria '{topic_id}/{sub_topic_id}'")
    return jsonify({"ok": True})


# ==============================================================================
# SITI DELLA CHAT
# ==============================================================================
# Si creano e si eliminano solo da riga di comando (utils/manage_sites.py): dominio e archivio non si cambiano da qui.

def _site_fields(data):
    try:
        buttons = clean_buttons(data.get("buttons"))
    except ValueError as e:
        raise ApiError(422, str(e))
    return (_text(data, "name", required=True), _flag(data, "allow_date_filter", False),
            _flag(data, "allow_subtopic_selection", True), _flag(data, "active", True),
            clean_html(_text(data, "welcome_it", max_len=MAX_HOME_CHARS)),
            clean_html(_text(data, "welcome_en", max_len=MAX_HOME_CHARS)), json.dumps(buttons, ensure_ascii=False))


class _Translation(BaseModel):
    html: str
    labels: list[str]


TRANSLATION_PROMPT = """Translate from Italian to English the home page of a public document archive.
- "html": keep every HTML tag and link exactly as it is, translate only the text.
- "labels": translate each button label, same number and same order.
Do not add anything. Names of people, institutions and places stay as they are.

Text to translate (JSON):
"""


@bp.route("/translate", methods=["POST"])
@require_permission(PERM_ARCHIVES)
def translate_home():
    """Traduce in inglese testo ed etichette della home, con l'LLM di sistema. Non salva nulla."""
    import llm_provider

    data = _json_body()
    source = clean_html(_text(data, "html", max_len=MAX_HOME_CHARS)) or ""
    labels = [str(label) for label in data.get("labels") or []]
    if not source and not labels:
        raise ApiError(422, "Non c'è nulla da tradurre")
    try:
        try:
            provider = llm_provider.get_llm_provider()
        except RuntimeError:  # il provider si inizializza nei worker; qui serve solo per questa rotta
            provider = llm_provider.init_llm_provider()
        raw = provider.generate_json(
            settings.answer_generator_model_name,
            TRANSLATION_PROMPT + json.dumps({"html": source, "labels": labels}, ensure_ascii=False),
            max_tokens=None, response_schema=_Translation)
        result = _Translation.model_validate_json(raw)
    except Exception as e:
        log.error(f"Traduzione della home fallita: {e}")
        raise ApiError(502, "Traduzione non riuscita: riprova tra poco")
    if len(result.labels) != len(labels):
        raise ApiError(502, "Traduzione non riuscita: riprova tra poco")
    return jsonify({"html": clean_html(result.html) or "", "labels": result.labels})


@bp.route("/sites", methods=["GET"])
@require_permission(PERM_ARCHIVES)
def list_sites():
    with Db() as db:
        rows = db.all("SELECT hostname, topic_id, name, allow_date_filter, allow_subtopic_selection, active, welcome_it, welcome_en, "
                      "home_buttons, "
                      "icon IS NOT NULL AS has_icon FROM chat_sites ORDER BY hostname")
    return jsonify({"items": [
        {**r, "allow_date_filter": bool(r["allow_date_filter"]),
         "allow_subtopic_selection": bool(r["allow_subtopic_selection"]), "active": bool(r["active"]),
         "has_icon": bool(r["has_icon"]), "buttons": buttons_from_db(r.pop("home_buttons"))} for r in rows
    ]})


@bp.route("/sites/<hostname>", methods=["PUT"])
@require_permission(PERM_ARCHIVES)
def update_site(hostname):
    data = _json_body()
    with Db() as db:
        if not db.one("SELECT 1 AS ok FROM chat_sites WHERE hostname = %s", (hostname,)):
            raise ApiError(404, "Sito non trovato")
        db.execute("UPDATE chat_sites SET name = %s, allow_date_filter = %s, allow_subtopic_selection = %s, active = %s, "
                   "welcome_it = %s, welcome_en = %s, home_buttons = %s WHERE hostname = %s", (*_site_fields(data), hostname))
        db.commit()
    forget_site(hostname)
    _audit(f"modificato il sito '{hostname}'")
    return jsonify({"ok": True})


@bp.route("/sites/<hostname>/icon", methods=["PUT"])
@require_permission(PERM_ARCHIVES)
def set_site_icon(hostname):
    try:
        icon = base64.b64decode(_json_body().get("png") or "", validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise ApiError(422, "Immagine non valida")
    if not icon.startswith(PNG_SIGNATURE):
        raise ApiError(422, "L'icona deve essere un file PNG")
    if len(icon) > MAX_ICON_BYTES:
        raise ApiError(413, "L'icona supera 1 MB")
    with Db() as db:
        if not db.one("SELECT 1 AS ok FROM chat_sites WHERE hostname = %s", (hostname,)):
            raise ApiError(404, "Sito non trovato")
        db.execute("UPDATE chat_sites SET icon = %s WHERE hostname = %s", (icon, hostname))
        db.commit()
    forget_site(hostname)
    _audit(f"cambiata l'icona del sito '{hostname}'")
    return jsonify({"ok": True})


@bp.route("/sites/<hostname>/icon", methods=["DELETE"])
@require_permission(PERM_ARCHIVES)
def delete_site_icon(hostname):
    with Db() as db:
        db.execute("UPDATE chat_sites SET icon = NULL WHERE hostname = %s", (hostname,))
        db.commit()
    forget_site(hostname)
    _audit(f"tolta l'icona del sito '{hostname}'")
    return jsonify({"ok": True})


# ==============================================================================
# CREDENZIALI DEI SISTEMI ESTERNI
# ==============================================================================

def _expiry(value):
    if value in (None, ""):
        return None
    try:
        return f"{date.fromisoformat(value).isoformat()} 23:59:59"
    except (TypeError, ValueError):
        raise ApiError(422, "La scadenza deve essere una data AAAA-MM-GG")


def _set_key_topics(db, key_id, topics):
    db.execute("DELETE FROM external_api_key_topics WHERE key_id = %s", (key_id,))
    for topic in topics:
        db.execute("INSERT INTO external_api_key_topics (key_id, topic_id) VALUES (%s, %s)", (key_id, topic))


@bp.route("/keys", methods=["GET"])
@require_permission(PERM_ARCHIVES)
def list_keys():
    with Db() as db:
        rows = db.all("SELECT id, name, key_prefix, active, created_at, expires_at, last_used_at "
                      "FROM external_api_keys ORDER BY name")
        topics = {}
        for r in db.all("SELECT key_id, topic_id FROM external_api_key_topics ORDER BY topic_id"):
            topics.setdefault(r["key_id"], []).append(r["topic_id"])
    return jsonify({"items": [
        {**r, "active": bool(r["active"]), "created_at": _iso(r["created_at"]),
         "expires_at": _iso(r["expires_at"]), "last_used_at": _iso(r["last_used_at"]),
         "topics": topics.get(r["id"], [])} for r in rows
    ]})


@bp.route("/keys", methods=["POST"])
@require_permission(PERM_ARCHIVES)
def create_key():
    data = _json_body()
    name = _text(data, "name", required=True, max_len=100)
    if not KEY_NAME_RE.match(name):
        raise ApiError(422, "Nome non valido: lettere, cifre, spazi e _ . -")
    topics, expires = _list(data, "topics"), _expiry(data.get("expires"))
    key, key_hash, prefix = new_external_key()
    with Db() as db:
        if db.one("SELECT 1 AS ok FROM external_api_keys WHERE name = %s", (name,)):
            raise ApiError(409, f"La credenziale '{name}' esiste già")
        _check_topics_exist(db, topics)
        db.execute("INSERT INTO external_api_keys (name, key_hash, key_prefix, expires_at) VALUES (%s, %s, %s, %s)",
                   (name, key_hash, prefix, expires))
        key_id = db.cur.lastrowid
        _set_key_topics(db, key_id, topics)
        db.commit()
    _audit(f"creata la credenziale '{name}' (archivi {topics})")
    # La chiave si mostra solo qui: nel database resta l'hash.
    return jsonify({"id": key_id, "key": key}), 201


@bp.route("/keys/<int:key_id>", methods=["PUT"])
@require_permission(PERM_ARCHIVES)
def update_key(key_id):
    data = _json_body()
    topics, expires, active = _list(data, "topics"), _expiry(data.get("expires")), _flag(data, "active", True)
    with Db() as db:
        row = db.one("SELECT name FROM external_api_keys WHERE id = %s", (key_id,))
        if not row:
            raise ApiError(404, "Credenziale non trovata")
        _check_topics_exist(db, topics)
        db.execute("UPDATE external_api_keys SET active = %s, expires_at = %s WHERE id = %s", (active, expires, key_id))
        _set_key_topics(db, key_id, topics)
        db.commit()
    _audit(f"modificata la credenziale '{row['name']}' (attiva {active}, archivi {topics})")
    return jsonify({"ok": True})


@bp.route("/keys/<int:key_id>/rotate", methods=["POST"])
@require_permission(PERM_ARCHIVES)
def rotate_key(key_id):
    key, key_hash, prefix = new_external_key()
    with Db() as db:
        row = db.one("SELECT name FROM external_api_keys WHERE id = %s", (key_id,))
        if not row:
            raise ApiError(404, "Credenziale non trovata")
        db.execute("UPDATE external_api_keys SET key_hash = %s, key_prefix = %s WHERE id = %s", (key_hash, prefix, key_id))
        db.commit()
    _audit(f"rigenerata la credenziale '{row['name']}'")
    return jsonify({"key": key})


@bp.route("/keys/<int:key_id>", methods=["DELETE"])
@require_permission(PERM_ARCHIVES)
def delete_key(key_id):
    with Db() as db:
        row = db.one("SELECT name FROM external_api_keys WHERE id = %s", (key_id,))
        if not row:
            raise ApiError(404, "Credenziale non trovata")
        db.execute("DELETE FROM external_api_keys WHERE id = %s", (key_id,))
        db.commit()
    _audit(f"eliminata la credenziale '{row['name']}'")
    return jsonify({"ok": True})
