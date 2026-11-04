"""Template dei prompt (cartella prompts/ e prompt per topic su MySQL)."""

import logging
import os

from common.db_logger import get_db_connection

log = logging.getLogger("WORKER")


# api/src/prompts: la cartella è accanto al pacchetto rag/, non dentro.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts')


def load_prompt_template(filename):
    """Load a prompt template from the prompts directory."""
    try:
        file_path = os.path.join(PROMPTS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        log.error(f"Prompt template '{filename}' not found. Using fallback.")
        return "{context_str}\n\nQuestion: {query}\n\nAnswer:"
    except Exception as e:
        log.error(f"Error loading prompt {filename}: {e}")
        return "{context_str}\n\nQuestion: {query}\n\nAnswer:"


def get_topic_prompt(topic_id):
    """Recupera il template del prompt associato al topic. Ritorna 'general' come fallback."""
    conn = get_db_connection()
    if not conn:
        return 'general'
    
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT prompt FROM topics WHERE topic_id = %s", (topic_id,))
            row = cursor.fetchone()
            # Se il campo prompt è NULL o vuoto, restituisce 'general'
            return row['prompt'] if row and row.get('prompt') else 'general'
    except Exception as e:
        log.error(f"Errore durante il recupero del prompt per il topic '{topic_id}': {e}")
        return 'general'
    finally:
        conn.close()
