"""
Siti della chat (tabella chat_sites). Il frontend è unico: chiede al backend chi
è in base al nome di dominio con cui è stato raggiunto.

    GET /site        nome, archivio e home page del sito (pubblica)
    GET /site/icon   favicon del sito (pubblica)

Un sito attivo autorizza le rotte della chat per il proprio archivio, al posto
della chiave che prima era scritta nel frontend e quindi leggibile da chiunque.
"""

import logging
import time

from flask import Blueprint, Response, jsonify, request

from common.config import settings
from common.db_logger import get_db_connection
from common.site_home import buttons_from_db

log = logging.getLogger("SITE")

bp = Blueprint("site", __name__, url_prefix="/site")

_FIELDS = "hostname, topic_id, name, allow_date_filter, welcome_it, welcome_en, home_buttons, icon IS NOT NULL AS has_icon"

# hostname -> (scadenza, sito o None). Ogni richiesta della chat passa da qui;
# con più processi una modifica dall'amministrazione arriva entro la durata della cache.
_cache = {}


def request_hostname():
    return request.host.split(":")[0].strip().lower()


def forget_site(hostname=None):
    if hostname is None:
        _cache.clear()
    else:
        _cache.pop(hostname, None)


def site_for_host(hostname):
    """Il sito attivo con questo dominio, oppure None. Un database non raggiungibile non si mette in cache."""
    cached = _cache.get(hostname)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT {_FIELDS} FROM chat_sites WHERE hostname = %s AND active = TRUE", (hostname,))
        site = cursor.fetchone()
        cursor.close()
    except Exception as e:
        log.error(f"Lettura del sito '{hostname}' fallita: {e}")
        return None
    finally:
        conn.close()
    _cache[hostname] = (time.monotonic() + settings.site_cache_seconds, site)
    return site


@bp.route("", methods=["GET"])
def get_site():
    site = site_for_host(request_hostname())
    if not site:
        return jsonify({"error": "Sito non configurato"}), 404
    return jsonify({
        "name": site["name"],
        "topic_id": site["topic_id"],
        "allow_date_filter": bool(site["allow_date_filter"]),
        "welcome": {"it": site["welcome_it"], "en": site["welcome_en"]},
        "buttons": buttons_from_db(site["home_buttons"]),
        "has_icon": bool(site["has_icon"]),
    })


@bp.route("/icon", methods=["GET"])
def get_icon():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database non disponibile"}), 503
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT icon FROM chat_sites WHERE hostname = %s AND active = TRUE", (request_hostname(),))
        row = cursor.fetchone()
        cursor.close()
    finally:
        conn.close()
    if not row or not row[0]:
        return jsonify({"error": "Icona non presente"}), 404
    return Response(bytes(row[0]), mimetype="image/png", headers={"Cache-Control": "public, max-age=300"})
