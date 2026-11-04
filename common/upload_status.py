"""Stato dei file caricati dal sito di amministrazione (tabella document_uploads)."""

import logging

from common.db_logger import get_db_connection

log = logging.getLogger("UPLOADS")

QUEUED = "in_coda"
NOT_SENT = "da_inviare"
PROCESSING = "in_lavorazione"
INDEXED = "indicizzato"
FAILED = "errore"


def mark_upload(source, status, error=None):
    """
    Aggiorna lo stato del file con questo source. Non fa nulla per i documenti che
    non vengono dal caricamento (Sicr@Web, direct.py). Un errore qui non deve
    fermare l'elaborazione del documento: si registra e si prosegue.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            log.warning(f"Stato '{status}' non registrato per {source}: database non raggiungibile")
            return
        cursor = conn.cursor()
        cursor.execute("UPDATE document_uploads SET status = %s, error = %s WHERE source = %s",
                       (status, str(error)[:1000] if error else None, source))
        conn.commit()
        cursor.close()
    except Exception as e:
        log.warning(f"Stato '{status}' non registrato per {source}: {e}")
    finally:
        if conn:
            conn.close()


def record_ocr(source, topic_id, sub_topic_id, ocr_pages, total_pages):
    """Quante pagine del documento vengono dall'OCR: un errore qui non deve fermare l'elaborazione."""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            log.warning(f"Pagine OCR non registrate per {source}: database non raggiungibile")
            return
        cursor = conn.cursor()
        cursor.execute("INSERT INTO document_ocr (source, topic_id, sub_topic_id, ocr_pages, total_pages) "
                       "VALUES (%s, %s, %s, %s, %s) "
                       "ON DUPLICATE KEY UPDATE ocr_pages = VALUES(ocr_pages), total_pages = VALUES(total_pages)",
                       (source, topic_id, sub_topic_id, ocr_pages, total_pages))
        conn.commit()
        cursor.close()
    except Exception as e:
        log.warning(f"Pagine OCR non registrate per {source}: {e}")
    finally:
        if conn:
            conn.close()
