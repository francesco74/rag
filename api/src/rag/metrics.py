"""Registrazione delle metriche e dei fallimenti (rag_metrics, chat_feedback)."""

import json
import logging

from common.db_logger import get_db_connection

log = logging.getLogger("WORKER")


def log_automatic_negative_feedback(query, answer, topic_id, history, task_id="UNKNOWN"):
    """
    Salva automaticamente nel database le query che non hanno trovato risposte
    valide o che sono state scartate dal Grader.
    """
    conn = get_db_connection()
    if not conn:
        log.error(f"[{task_id}] [AUTO_FEEDBACK] Impossibile salvare: DB pool non disponibile.")
        return
        
    try:
        # Serializziamo la cronologia come fatto sul gateway Flask
        history_json = json.dumps(history) if history else "[]"
        
        sql = """
            INSERT INTO chat_feedback 
            (user_query, ai_response, topic_id, rating, chat_history, comment) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        with conn.cursor() as cursor:
            cursor.execute(sql, (
                query,
                answer,
                topic_id,
                0,  # Rating 0 per identificare i fallimenti automatici del sistema
                history_json,
                "Auto-Log: Risposta non trovata o non soddisfacente (Grader/Self-Correction Fail)"
            ))
        conn.commit()
        log.info(f"[{task_id}] [AUTO_FEEDBACK] ✓ Query non risposta registrata con successo nel DB.")
        
    except Exception as e:
        log.error(f"[{task_id}] [AUTO_FEEDBACK] ✗ Errore durante il salvataggio a DB: {e}")
        conn.rollback()
    finally:
        conn.close()


def log_rag_metrics(task_id, topic_id, query, cache_hit, is_satisfactory,
                     total_attempts, duration_ms, attempts_metrics):
    """
    Salva su MySQL (tabella rag_metrics, vedi create_rag_metrics_table.sql) una
    riga di sintesi per ogni task RAG processato: quanti tentativi sono serviti,
    se è finito in cache, e il dettaglio per-tentativo (parametri di recall,
    lambda MMR, quanti candidati/penalizzati) come JSON in attempts_detail.

    Non blocca mai il flusso principale: un fallimento qui viene solo loggato,
    esattamente come già fatto per la cache semantica e il feedback automatico.
    """
    conn = get_db_connection()
    if not conn:
        log.error(f"[{task_id}] [RAG_METRICS] Impossibile salvare: DB pool non disponibile.")
        return

    try:
        attempts_json = json.dumps(attempts_metrics, default=str)

        sql = """
            INSERT INTO rag_metrics
            (task_id, topic_id, query_preview, cache_hit, is_satisfactory,
             total_attempts, duration_ms, attempts_detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        with conn.cursor() as cursor:
            cursor.execute(sql, (
                task_id,
                topic_id,
                query[:255] if query else None,
                cache_hit,
                is_satisfactory,
                total_attempts,
                duration_ms,
                attempts_json
            ))
        conn.commit()
        log.debug(f"[{task_id}] [RAG_METRICS] ✓ Metriche salvate (attempts={total_attempts}, cache_hit={cache_hit}).")

    except Exception as e:
        log.error(f"[{task_id}] [RAG_METRICS] ✗ Errore durante il salvataggio a DB: {e}")
        conn.rollback()
    finally:
        conn.close()
