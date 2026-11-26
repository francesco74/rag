"""Template dei prompt (cartella prompts/ e prompt per topic su MySQL).

Il campo topics.prompt indica il file in prompts/ (general se vuoto o inesistente). Ogni
prompt può avere un grader suo, prompts/<nome>_grader; senza, vale prompts/grader."""

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


DEFAULT_PROMPT = "general"
DEFAULT_GRADER = "grader"
GRADER_SUFFIX = "_grader"


def prompt_exists(name):
    """True se prompts/<name> è un file (nome semplice, senza percorsi)."""
    return bool(name) and os.path.basename(name) == name and os.path.isfile(os.path.join(PROMPTS_DIR, name))


def grader_name(prompt_name):
    """Il grader del prompt: prompts/<prompt>_grader se esiste, altrimenti prompts/grader."""
    candidate = f"{prompt_name}{GRADER_SUFFIX}" if prompt_name else ""
    return candidate if prompt_exists(candidate) else DEFAULT_GRADER


def get_topic_prompt(topic_id):
    """Recupera il template del prompt associato al topic. Ritorna 'general' come fallback."""
    conn = get_db_connection()
    if not conn:
        return DEFAULT_PROMPT

    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT prompt FROM topics WHERE topic_id = %s", (topic_id,))
            row = cursor.fetchone()
            # Se il campo prompt è NULL, vuoto o indica un file che non c'è: general
            name = row['prompt'] if row and row.get('prompt') else DEFAULT_PROMPT
            if not prompt_exists(name):
                log.warning(f"Topic '{topic_id}': prompt '{name}' inesistente, uso '{DEFAULT_PROMPT}'.")
                return DEFAULT_PROMPT
            return name
    except Exception as e:
        log.error(f"Errore durante il recupero del prompt per il topic '{topic_id}': {e}")
        return DEFAULT_PROMPT
    finally:
        conn.close()
