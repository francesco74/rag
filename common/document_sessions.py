"""
Sessioni di documenti delle API esterne (api/src/external_sessions.py).

Una sessione è un contenitore temporaneo di documenti caricati da un sistema
esterno (atti-pa) per una conversazione. Il percorso di un documento:

    API          salva il file in sessions/<sid>/<did>/, stato queued,
                 pubblica {session_id, doc_id} sulla coda delle conversioni
    converter    converting -> testo in text.md, stato indexing,
                 pubblica sulla coda SESSION_INDEX_QUEUE
    ingestione   sezioni in MySQL (session_parents), pezzi in Qdrant
                 (session_chunks), stato ready; cancella la cartella

Errori in ogni passo: stato failed con il motivo, cartella cancellata.
La scadenza la decide sessions.expires_at in MySQL (si rinnova a ogni uso).
In Qdrant expires_at è fisso (creazione + SESSION_MAX_HOURS) e serve solo a
togliere i punti rimasti senza sessione. I log non contengono mai il testo.

qdrant_client si importa solo nelle funzioni che lo usano: il converter, che non
lo ha, usa solo stati e cartelle.
"""

import logging
import re
import shutil
from datetime import datetime, timedelta, timezone

from common.config import settings
from common.db_logger import get_db_connection

log = logging.getLogger("SESSIONS")

SESSION_COLLECTION = "session_chunks"
SESSION_INDEX_QUEUE = "da-indicizzare-sessione"
SESSION_INDEX_QUEUE_ARGS = {"x-queue-type": "quorum"}

QUEUED = "queued"
CONVERTING = "converting"
INDEXING = "indexing"
READY = "ready"
FAILED = "failed"

ID_RE = re.compile(r"^[0-9a-f]{32}$")


def utcnow() -> datetime:
    """Ora UTC senza fuso, come si salva nelle colonne DATETIME."""
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def max_expiry(created_at: datetime) -> datetime:
    return created_at + timedelta(hours=settings.session_max_hours)


def next_expiry(created_at: datetime, now: datetime | None = None) -> datetime:
    """Scadenza scorrevole: ultimo uso + SESSION_TTL_MINUTES, mai oltre il tetto dalla creazione."""
    now = now or utcnow()
    return min(now + timedelta(minutes=settings.session_ttl_minutes), max_expiry(created_at))


def epoch(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def iso(dt: datetime | None) -> str | None:
    return dt.replace(microsecond=0).isoformat() + "Z" if dt else None


def sessions_folder():
    return settings.data_folder / "sessions"


def document_folder(session_id: str, doc_id: str):
    if not (ID_RE.match(session_id) and ID_RE.match(doc_id)):
        raise ValueError("Identificativo di sessione o di documento non valido")
    return sessions_folder() / session_id / doc_id


# ------------------------------------------------------------------------------
# MySQL
# ------------------------------------------------------------------------------

def _run(sql, params=(), fetch=None):
    conn = get_db_connection()
    if not conn:
        raise ConnectionError("Database non raggiungibile")
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params)
        if fetch == "one":
            result = cursor.fetchone()
        elif fetch == "all":
            result = cursor.fetchall()
        else:
            result = cursor.rowcount
        conn.commit()
        return result
    finally:
        conn.close()


def get_document(doc_id: str):
    """Documento con i dati della sua sessione; None se non esiste più."""
    return _run(
        "SELECT d.id, d.session_id, d.file_name, d.status, d.tokens, s.key_id, s.created_at "
        "FROM session_documents d JOIN sessions s ON s.id = d.session_id WHERE d.id = %s",
        (doc_id,), fetch="one")


def set_status(doc_id: str, status: str, error: str | None = None, tokens: int | None = None,
               sections: int | None = None) -> bool:
    """Aggiorna lo stato; False se il documento non c'è più (sessione o documento cancellati)."""
    fields, params = ["status = %s", "error = %s"], [status, (error or None) and str(error)[:1000]]
    if tokens is not None:
        fields.append("tokens = %s")
        params.append(tokens)
    if sections is not None:
        fields.append("sections = %s")
        params.append(sections)
    return _run(f"UPDATE session_documents SET {', '.join(fields)} WHERE id = %s",
                tuple(params) + (doc_id,)) > 0


def session_tokens(session_id: str, exclude_doc_id: str | None = None) -> int:
    """Token dei documenti della sessione non falliti (stima del testo convertito)."""
    row = _run("SELECT COALESCE(SUM(tokens), 0) AS n FROM session_documents "
               "WHERE session_id = %s AND status <> %s AND id <> %s",
               (session_id, FAILED, exclude_doc_id or ""), fetch="one")
    return int(row["n"] or 0)


# ------------------------------------------------------------------------------
# Cancellazione
# ------------------------------------------------------------------------------

def _qdrant_delete(client, conditions):
    from qdrant_client import models
    try:
        client.delete(collection_name=SESSION_COLLECTION, wait=True,
                      points_selector=models.FilterSelector(filter=models.Filter(must=conditions)))
    except Exception as e:
        # La collection può mancare (installazione senza setup_qdrant): niente da cancellare.
        log.warning(f"Cancellazione dei punti di sessione non riuscita: {e}")


def delete_document(client, session_id: str, doc_id: str):
    """Toglie un documento: punti Qdrant, sezioni, riga e cartella."""
    from qdrant_client import models
    _qdrant_delete(client, [
        models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
    ])
    _run("DELETE FROM session_parents WHERE doc_id = %s", (doc_id,))
    _run("DELETE FROM session_documents WHERE id = %s", (doc_id,))
    shutil.rmtree(document_folder(session_id, doc_id), ignore_errors=True)
    log.info(f"Sessione {session_id}: documento {doc_id} cancellato.")


def delete_sessions(client, session_ids):
    """Chiude le sessioni e cancella tutto quello che contengono."""
    from qdrant_client import models
    session_ids = [s for s in session_ids if ID_RE.match(s)]
    if not session_ids:
        return
    _qdrant_delete(client, [models.FieldCondition(key="session_id", match=models.MatchAny(any=session_ids))])
    placeholders = ", ".join(["%s"] * len(session_ids))
    _run(f"DELETE FROM session_parents WHERE session_id IN ({placeholders})", tuple(session_ids))
    _run(f"DELETE FROM session_documents WHERE session_id IN ({placeholders})", tuple(session_ids))
    _run(f"DELETE FROM sessions WHERE id IN ({placeholders})", tuple(session_ids))
    for sid in session_ids:
        shutil.rmtree(sessions_folder() / sid, ignore_errors=True)
    log.info(f"Sessioni cancellate: {len(session_ids)}.")


def prune_expired(client) -> int:
    """Cancella le sessioni scadute e i punti Qdrant rimasti senza sessione. Restituisce quante sessioni."""
    from qdrant_client import models
    now = utcnow()
    rows = _run("SELECT id FROM sessions WHERE expires_at < %s", (now,), fetch="all")
    expired = [r["id"] for r in rows]
    for i in range(0, len(expired), 100):
        delete_sessions(client, expired[i:i + 100])
    _qdrant_delete(client, [models.FieldCondition(key="expires_at", range=models.Range(lt=epoch(now)))])
    # Cartelle senza sessione (sessione cancellata mentre il converter scriveva).
    folder = sessions_folder()
    if folder.is_dir():
        alive = {r["id"] for r in _run("SELECT id FROM sessions", fetch="all")}
        for d in folder.iterdir():
            if d.is_dir() and d.name not in alive:
                shutil.rmtree(d, ignore_errors=True)
    return len(expired)


def insert_parents(rows):
    """rows: (session_id, doc_id, parent_index, content, metadata_json). Sezioni di un documento."""
    conn = get_db_connection()
    if not conn:
        raise ConnectionError("Database non raggiungibile")
    try:
        cursor = conn.cursor()
        sql = ("INSERT INTO session_parents (session_id, doc_id, parent_index, content, metadata) "
               "VALUES (%s, %s, %s, %s, %s)")
        for i in range(0, len(rows), 200):
            cursor.executemany(sql, rows[i:i + 200])
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def delete_parents(doc_id: str):
    _run("DELETE FROM session_parents WHERE doc_id = %s", (doc_id,))
