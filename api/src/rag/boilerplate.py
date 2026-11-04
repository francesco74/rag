"""Frasi ripetute (intestazioni, piè di pagina) da togliere prima del rerank."""

import logging
import re
import time

import numpy as np
from qdrant_client import models

from common.config import settings
from common.db_logger import get_db_connection
from rag import state
from rag.state import BOILERPLATE_COLLECTION

log = logging.getLogger("WORKER")


# Cache in-processo delle frasi di boilerplate attive, per topic. Evita una
# query MySQL a ogni richiesta: le frasi cambiano una volta a notte (script
# detect_boilerplate.py) e su approvazione manuale, quindi un TTL breve è
# ampiamente sufficiente. Struttura: {topic_id: (scadenza_epoch, [frasi...])}.
_BOILERPLATE_CACHE = {}


_BOILERPLATE_TTL_SECONDS = 300


def load_active_boilerplate(topic_id: str, sub_topic_ids):
    """Frasi di boilerplate ATTIVE per (topic, sub_topic), con cache a TTL.

    Ritorna {sub_topic_id: [(phrase, vector), ...]} oppure None se non ce ne
    sono. MySQL è la fonte di verità sull'active (rispetta la revisione umana);
    Qdrant fornisce i vettori, recuperati PER phrase_hash — un join esatto che
    non dipende dall'identità testuale delle stringhe fra i due sistemi.

    Una frase attiva in MySQL ma priva di vettore in Qdrant (es. sync notturna
    non ancora rigirata) non si perde: entra nella mappa con vettore None, e la
    guardia ricadrà su di lei col criterio lessicale.

    Fail-soft: qualunque errore -> None, e il reranker lavora sul testo grezzo.
    """
    if not sub_topic_ids:
        return None

    # Chiave di cache: topic + insieme ordinato dei sub_topic interrogati.
    cache_key = (topic_id, tuple(sorted(sub_topic_ids)))
    now = time.time()
    cached = _BOILERPLATE_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    # --- 1. MySQL: quali frasi (active=TRUE) e il loro hash ------------------
    rows = []
    try:
        conn = get_db_connection()
        if conn is None:
            raise RuntimeError("connessione MySQL non disponibile")
        try:
            with conn.cursor(dictionary=True) as cursor:
                placeholders = ",".join(["%s"] * len(sub_topic_ids))
                cursor.execute(
                    f"SELECT phrase, sub_topic_id, phrase_hash "
                    f"FROM boilerplate_phrases "
                    f"WHERE topic_id = %s AND active = TRUE "
                    f"AND sub_topic_id IN ({placeholders})",
                    (topic_id, *sub_topic_ids),
                )
                rows = [r for r in cursor.fetchall() if r.get("phrase")]

                log.debug(f"Caricate {len(rows)} frasi boilerplate attive da MySQL per topic '{topic_id}' ")
        finally:
            conn.close()
    except Exception as e:
        log.warning(f"Caricamento boilerplate MySQL per '{topic_id}' fallito ({e}): "
                    f"il reranker procede sul testo grezzo.")
        _BOILERPLATE_CACHE[cache_key] = (now + 30, None)
        return None

    if not rows:
        _BOILERPLATE_CACHE[cache_key] = (now + _BOILERPLATE_TTL_SECONDS, None)
        return None

    # --- 2. Qdrant: i vettori delle frasi attive, recuperati per hash --------
    wanted_hashes = [r["phrase_hash"] for r in rows]
    vec_by_hash = {}
    try:
        scrolled, _ = state.qdrant_client.scroll(
            collection_name=BOILERPLATE_COLLECTION,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                models.FieldCondition(key="phrase_hash", match=models.MatchAny(any=wanted_hashes)),
            ]),
            with_vectors=True,
            with_payload=["phrase_hash"],
            limit=len(wanted_hashes) + 16,
        )
        for p in scrolled:
            h = (p.payload or {}).get("phrase_hash")
            if h and p.vector is not None:
                vec_by_hash[h] = np.asarray(p.vector, dtype=np.float32)
    except Exception as e:
        # Qdrant assente o collection non ancora popolata: si prosegue coi soli
        # vettori mancanti (tutti None) -> guardia lessicale. Non è un errore
        # fatale, è un degrado.
        log.warning(f"Recupero vettori boilerplate da Qdrant fallito ({e}): "
                    f"guardia lessicale per questo topic.")

    # --- 3. Raggruppa per sub_topic, agganciando il vettore quando c'è -------
    by_subtopic = {}
    for r in rows:
        st = r["sub_topic_id"]
        vec = vec_by_hash.get(r["phrase_hash"])   # None se mancante
        by_subtopic.setdefault(st, []).append((r["phrase"], vec))

    _BOILERPLATE_CACHE[cache_key] = (now + _BOILERPLATE_TTL_SECONDS, by_subtopic)
    return by_subtopic


def build_removal_by_subtopic(boilerplate_map, query_str, query_vector):
    """Decide UNA VOLTA per query quali frasi rimuovere, per ciascun sub_topic.

    Ritorna {sub_topic_id: compiled_regex} dove la regex contiene SOLO le frasi
    da rimuovere (quelle NON pertinenti alla query). Le frasi pertinenti non
    entrano nella regex, quindi restano nei chunk.

    Guardia PRIMARIA semantica (vettore query e vettore frase presenti): tiene
    la frase se la similarità coseno supera settings.boilerplate_similarity_
    threshold. FALLBACK lessicale (manca un vettore): tiene la frase se
    condivide un termine significativo con la query.

    Il confronto è per sub_topic perché il boilerplate di una serie non deve
    toccare i chunk di un'altra; ed è per query, non per chunk, perché la
    decisione dipende solo da query e frasi (i 200 chunk del gruppo ricevono
    tutti la stessa regex)."""
    if not boilerplate_map:
        return {}

    qv = None
    if query_vector is not None:
        qv = np.asarray(query_vector, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
    thr = settings.boilerplate_similarity_threshold
    q_terms = _significant_terms(query_str)

    removal = {}
    for st, phrase_vecs in boilerplate_map.items():
        to_remove = []
        kept = 0
        for phrase, pv in phrase_vecs:
            pertinent = False
            if qv is not None and pv is not None:
                sim = float(qv @ (pv / (np.linalg.norm(pv) + 1e-9)))
                pertinent = sim >= thr
            else:
                # Fallback lessicale: pertinente se condivide un termine.
                pertinent = bool(_significant_terms(phrase) & q_terms)
            if pertinent:
                kept += 1
            else:
                to_remove.append(phrase)

        if to_remove:
            ordered = sorted(set(to_remove), key=len, reverse=True)
            removal[st] = re.compile("|".join(re.escape(p) for p in ordered), re.IGNORECASE)
        log.debug(f"[BOILERPLATE] sub_topic '{st}': rimuovo {len(to_remove)}, "
                  f"tengo {kept} (pertinenti alla query).")

    return removal


_STOPWORDS = frozenset("""
il lo la i gli le un uno una di del dello della dei degli delle da dal dallo
dalla in nel nello nella con su sul sullo sulla per tra fra e ed o od a ad al
allo alla ai agli alle che chi cui non come più meno anche se ma però quindi
the a an of to in on for and or with by from at as is are be this that
""".split())


def _significant_terms(text: str) -> set:
    """Termini di un testo, minuscoli, esclusi stopword e token troppo corti.
    Un identificatore come '267/2000' o 'd.lgs' resta un termine significativo."""
    toks = re.findall(r"\w+(?:['\-./]\w+)*", (text or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}
