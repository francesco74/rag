"""Parsing robusto del JSON restituito dagli LLM."""

import json
import logging
import re

log = logging.getLogger("WORKER")


# Backslash NON seguito da un escape JSON valido (" \ / b f n r t u).
# Es. "\..." o "\'" -> vengono raddoppiati (\\... , \\') diventando un
# backslash letterale valido all'interno della stringa JSON.
_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def safe_json_parse(raw_text: str, task_id: str = "UNKNOWN") -> dict:
    """
    Estrae robustamente il JSON con log intermedi per capire ESATTAMENTE
    dove e perché il parser fallisce.
    """
    raw_text = raw_text.strip()
    # Logghiamo l'inizio del parsing (limitato ai primi 300 caratteri per non intasare i log)
    log.debug(f"[{task_id}] [JSON_PARSE] Inizio estrazione. Testo grezzo ricevuto: \n{raw_text[:300]}...")
    
    # 1. Tentativo standard
    try:
        parsed_data = json.loads(raw_text)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 1 (Standard Parse).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 1 fallito: {e}. Passo al Livello 1b (Raw Decode).")
 
    # 1b. Tentativo con raw_decode: parsa il primo oggetto JSON valido e ignora
    # eventuale "extra data" successiva (es. una '}' spuria aggiunta dal modello
    # dopo la chiusura corretta dell'oggetto).
    try:
        decoder = json.JSONDecoder()
        parsed_data, _ = decoder.raw_decode(raw_text)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 1b (raw_decode, extra data ignorata).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 1b fallito: {e}. Passo al Livello 2 (Markdown Strip).")
 
    # 2. Tentativo con pulizia Markdown esplicita
    clean_text = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.MULTILINE).strip()
    try:
        parsed_data = json.loads(clean_text)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 2 (Markdown rimosso).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 2 fallito: {e}. Passo al Livello 3 (Brute Force Regex).")
 
    # 2b. Tentativo di riparazione stringa troncata
    try:
        repaired = clean_text.rstrip() + '"}'
        parsed_data = json.loads(repaired)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 2b (Stringa riparata).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 2b fallito: {e}. Passo al Livello 2c (Escape Sanitize).")

    # 2c. Tentativo con sanificazione degli escape JSON non validi.
    # Con risposte HTML lunghe e piene di virgolette annidate, il modello
    # occasionalmente produce un backslash "orfano" (es. \... per abbreviare
    # un titolo già citato, o \' per un apostrofo) che non fa parte di
    # nessuna sequenza di escape JSON valida (\" \\ \/ \b \f \n \r \t \uXXXX).
    # json.loads() rifiuta l'intero payload con "Invalid \escape", anche se
    # il resto del JSON è perfettamente valido e la risposta è completa e
    # semanticamente corretta — un peccato buttarla via per un carattere.
    # Qui raddoppiamo ogni backslash orfano trasformandolo in un backslash
    # letterale valido, senza toccare gli escape già corretti.
    try:
        sanitized = _sanitize_invalid_json_escapes(clean_text)
        parsed_data = json.loads(sanitized)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 2c (Escape sanificati).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 2c fallito: {e}. Passo al Livello 3 (Brute Force Regex).")

    # 3. Tentativo "Forza Bruta": Cerca tutto ciò che è tra parentesi graffe
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed_data = json.loads(candidate)
            log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 3 (Regex Regex Brute Force).")
            return parsed_data
        except json.JSONDecodeError as e:
            log.debug(f"[{task_id}] [JSON_PARSE] Livello 3 fallito: {e}. Passo al Livello 3b (Brute Force + Escape Sanitize).")

        # 3b. Stessa sanificazione, applicata al blocco estratto in Livello 3:
        # copre il caso in cui l'escape invalido conviva con extra data attorno
        # alle graffe (markdown residuo, testo introduttivo del modello, ecc.).
        try:
            sanitized = _sanitize_invalid_json_escapes(candidate)
            parsed_data = json.loads(sanitized)
            log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 3b (Regex + Escape sanificati).")
            return parsed_data
        except json.JSONDecodeError as e:
            log.debug(f"[{task_id}] [JSON_PARSE] Livello 3b fallito: {e}.")

    # Fallimento totale
    log.error(f"[{task_id}] [JSON_PARSE] ✗ Fallimento totale. Impossibile estrarre JSON. Testo originale:\n{raw_text}")
    raise ValueError("Impossibile estrarre un JSON valido dal testo fornito.")


def _sanitize_invalid_json_escapes(text: str) -> str:
    """Ripara backslash "orfani" che rompono json.loads() senza toccare
    gli escape già validi (incluse le sequenze \\uXXXX)."""
    return _INVALID_JSON_ESCAPE_RE.sub(r'\\\\', text)
