"""
Download dei documenti originali: unico punto di accesso, usato sia dalle
chat (attiamministrativi, archiviogreenlees) sia dalla revisione.

    GET /files/<topic_id>/<sub_topic_id>/<file_name>[?download=1]

Chi può scaricare dipende dall'archivio (tabella topics, colonna
public_access):
  - archivio pubblico: chiunque, senza autenticazione;
  - archivio riservato: solo un utente autenticato con il permesso
    documenti.lettura e l'archivio assegnato (le stesse regole della
    revisione). L'utente si riconosce dal cookie di sessione impostato al
    login (i link aperti dal browser non possono inviare header) oppure
    dall'header Authorization.

Si servono solo i file originali dei documenti indicizzati: il nome deve
corrispondere a un documento in parent_documents. Il resto di processed/
(testi estratti .md, manifest .json) non è mai raggiungibile.

Con FILES_X_ACCEL_PREFIX impostato l'API si limita ad autorizzare: risponde
con l'header X-Accel-Redirect e il file lo invia nginx da una location
internal, senza occupare i worker Python con file grandi. Senza, il file lo
invia l'API (sviluppo locale senza nginx).
"""

import html
import logging
import mimetypes
import time
from urllib.parse import quote

from flask import Blueprint, Response, g, jsonify, request, send_file

from common.config import settings
from common.review_permissions import PERM_READ
from review_routes import (PROCESSED_FOLDER, SESSION_COOKIE, ApiError, Db, RequestLogContext,
                           check_permission, original_file_path, topic_allowed, user_from_token)

log = logging.getLogger("files_api")
log.setLevel(getattr(logging, settings.log_level, logging.INFO))
log.addFilter(RequestLogContext())

bp = Blueprint("files", __name__, url_prefix="/files")

_PAGE = """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f3f4f8;color:#1b1c20;margin:0;
display:grid;place-items:center;min-height:100vh}}
main{{background:#fff;border-radius:12px;padding:32px;max-width:440px;margin:16px;
box-shadow:0 1px 4px rgba(0,0,0,.15)}}
h1{{font-size:1.3rem;margin-top:0}} a{{color:#1f5c99}}
</style></head>
<body><main><h1>{title}</h1><p>{message}</p>{action}</main></body></html>"""

_TITLES = {
    401: "Documento riservato",
    403: "Accesso non consentito",
    404: "Documento non trovato",
}


def _wants_html() -> bool:
    """Un link aperto dal browser riceve una pagina, un client una risposta JSON."""
    return request.accept_mimetypes.best_match(["application/json", "text/html"]) == "text/html"


@bp.errorhandler(ApiError)
def _handle_error(e: ApiError):
    # 401/403 sono già registrati con il motivo nel punto in cui nascono.
    level = logging.WARNING if e.status >= 500 else logging.DEBUG if e.status in (401, 403) else logging.INFO
    log.log(level, f"{request.method} {request.path} rifiutata con {e.status}: {e.message}")
    if not _wants_html():
        return jsonify({"error": e.message}), e.status
    action = ""
    if e.status == 401:
        action = (f'<p><a href="{html.escape(settings.files_login_url)}">Accedi</a> '
                  f'e poi riapri il link del documento.</p>')
    page = _PAGE.format(title=_TITLES.get(e.status, "Errore"), message=html.escape(e.message), action=action)
    return Response(page, status=e.status, mimetype="text/html")


@bp.errorhandler(Exception)
def _handle_unexpected(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    log.error(f"Errore non gestito su {request.path}: {e}", exc_info=True)
    return _handle_error(ApiError(500, "Errore interno del server"))


@bp.before_request
def _start():
    g.files_started = time.perf_counter()


@bp.after_request
def _log_duration(response):
    started = g.get("files_started")
    if started is not None:
        log.debug(f"{request.method} {request.path} -> {response.status_code} "
                  f"in {time.perf_counter() - started:.3f}s")
    return response


def _current_user():
    """Utente della richiesta (header o cookie di sessione), None se anonima."""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return user_from_token(header[7:])
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        return user_from_token(token)
    return None


@bp.route("/<topic_id>/<sub_topic_id>/<path:file_name>", methods=["GET"])
def get_file(topic_id, sub_topic_id, file_name):
    download = request.args.get("download") == "1"

    with Db() as db:
        topic = db.one("SELECT public_access FROM topics WHERE topic_id = %s", (topic_id,))
    if not topic:
        raise ApiError(404, "Archivio inesistente")
    public = bool(topic["public_access"])

    # L'autorizzazione precede la ricerca del documento: su un archivio
    # riservato un anonimo non deve poter scoprire quali file esistono.
    if public:
        log.debug(f"Archivio '{topic_id}' pubblico: accesso libero")
    else:
        user = _current_user()
        if user is None:
            log.info(f"Documento dell'archivio riservato '{topic_id}' richiesto senza autenticazione "
                     f"da {request.remote_addr}")
            raise ApiError(401, "Questo documento è riservato: accedi per consultarlo.")
        g.user = user
        check_permission(user, PERM_READ)
        if not topic_allowed(user, topic_id):
            log.warning(f"Accesso negato a '{user['username']}' al documento {file_name}: "
                        f"archivio '{topic_id}' non assegnato")
            raise ApiError(403, "Non sei abilitato a consultare i documenti di questo archivio.")
        log.debug(f"Archivio '{topic_id}' riservato: accesso consentito a '{user['username']}'")

    with Db() as db:
        row = db.one(
            "SELECT 1 AS ok FROM parent_documents WHERE topic_id = %s AND sub_topic_id = %s "
            "AND file_name = %s LIMIT 1",
            (topic_id, sub_topic_id, file_name),
        )
    if not row:
        log.info(f"{topic_id}/{sub_topic_id}/{file_name} non corrisponde a nessun documento indicizzato")
        raise ApiError(404, "Documento non trovato")
    path = original_file_path(topic_id, sub_topic_id, file_name)
    if not path:
        raise ApiError(404, "Il file originale non è presente in archivio")

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    disposition = f"{'attachment' if download else 'inline'}; filename*=UTF-8''{quote(path.name)}"
    cache = "public, max-age=3600" if public else "private, no-store"

    if settings.files_x_accel_prefix:
        rel = path.relative_to(PROCESSED_FOLDER.resolve()).as_posix()
        target = settings.files_x_accel_prefix.rstrip("/") + "/" + quote(rel)
        response = Response(status=200, mimetype=mime)
        response.headers["X-Accel-Redirect"] = target
        log.debug(f"File {rel} affidato a nginx: X-Accel-Redirect {target}")
    else:
        response = send_file(path, mimetype=mime, max_age=0, conditional=True)
        log.debug(f"File {path} inviato dall'API ({path.stat().st_size} byte)")
    response.headers["Content-Disposition"] = disposition
    response.headers["Cache-Control"] = cache
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
