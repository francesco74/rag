"""
Ricerca nei documenti di una sessione (collection session_chunks), per le API esterne.

Gira nell'API e non nel worker Celery, che lavora un messaggio alla volta e farebbe
aspettare la chat. Passi:
  1. ricerca vettoriale sui pezzi della sessione e dei documenti scelti;
  2. ricerca per parole (indice full-text su content) per i termini rari, i numeri
     di articolo, le sigle, che la ricerca vettoriale tende a perdere;
  3. riordino: cross-encoder su vLLM con RERANK_PROVIDER=vllm, altrimenti quello
     lessicale (lexical_reranker.py), che è anche il ripiego se vLLM non risponde;
     con onnx si usa il lessicale per non caricare il modello nel processo dell'API;
  4. una sola voce per sezione (parent), con il testo della sezione intera.

Il filtro su sessione e chiave lo mette sempre il server.
"""

import logging
import math
import threading

from qdrant_client import models

from common.config import settings
from common.document_sessions import SESSION_COLLECTION
from common.utility import normalize_ws
from lexical_reranker import STOPWORDS, WORD_RE, LexicalReranker
from reranker import VLLMReranker

log = logging.getLogger("external_api")

CANDIDATES = 40          # pezzi dalla ricerca vettoriale
KEYWORD_CANDIDATES = 40  # pezzi dalla ricerca per parole
KEYWORD_TERMS = 6        # parole della domanda usate nella ricerca per parole

_lock = threading.Lock()
_embedding_ready = False
_model_reranker = None
_lexical = LexicalReranker()


def _embed_query(query: str):
    global _embedding_ready
    from common.embedding import embed_query, init_embedding
    if not _embedding_ready:
        with _lock:
            if not _embedding_ready:
                init_embedding()
                _embedding_ready = True
    return embed_query(query)


def _reranker():
    global _model_reranker
    if settings.rerank_provider != "vllm" or not settings.rerank_model_name:
        return None
    if _model_reranker is None:
        with _lock:
            if _model_reranker is None:
                _model_reranker = VLLMReranker(
                    base_url=settings.local_vllm_rerank_base_url or settings.local_vllm_base_url,
                    model=settings.rerank_model_name, api_key=settings.local_vllm_api_key,
                    max_length=settings.rerank_max_length, timeout=settings.rerank_timeout_seconds)
    return _model_reranker


def keyword_terms(query: str):
    """Parole della domanda per la ricerca full-text: le più lunghe, senza parole vuote."""
    words = [w for w in dict.fromkeys(WORD_RE.findall(query.lower()))
             if w not in STOPWORDS and (len(w) >= 4 or any(ch.isdigit() for ch in w))]
    return sorted(words, key=len, reverse=True)[:KEYWORD_TERMS]


def rerank(query, candidates):
    """Ordina i candidati; restituisce (lista di (candidato, punteggio 0..1), nome del riordino)."""
    docs = [normalize_ws(c.payload.get("content", ""))[:settings.rerank_truncate] for c in candidates]
    vector_scores = [getattr(c, "score", None) for c in candidates]
    model = _reranker()
    used = "lessicale"
    results = None
    if model:
        try:
            results = model.rerank(query, docs)
            used = "vllm"
        except Exception as e:
            log.warning(f"[RERANK_FALLBACK] Reranker vLLM in errore sulla ricerca di sessione: {e}. "
                        "Uso il riordino lessicale.")
    if results is None:
        results = _lexical.rerank(query, docs, vector_scores)
    ordered = []
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        prob = r.probability
        if prob is None:   # logit: stessa conversione del motore della chat
            prob = 1 / (1 + math.exp(-max(-100.0, r.score)))
        ordered.append((candidates[r.index], prob))
    return ordered, used


def search(client, session_id: str, key_id: int, doc_ids, query: str, top_k: int):
    """
    Sezioni pertinenti: lista di {doc_id, parent_index, score} in ordine, al massimo top_k,
    e il nome del riordino usato. Il testo delle sezioni lo aggiunge il chiamante (MySQL).
    """
    must = [
        models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
        models.FieldCondition(key="key_id", match=models.MatchValue(value=key_id)),
        models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))),
    ]
    vector = _embed_query(query)
    hits = client.query_points(collection_name=SESSION_COLLECTION, query=vector,
                               query_filter=models.Filter(must=must), limit=CANDIDATES,
                               with_payload=True).points
    candidates = {h.id: h for h in hits}
    words = keyword_terms(query)
    if words:
        try:
            records, _ = client.scroll(
                collection_name=SESSION_COLLECTION, limit=KEYWORD_CANDIDATES, with_payload=True,
                scroll_filter=models.Filter(must=must, should=[
                    models.FieldCondition(key="content", match=models.MatchText(text=w)) for w in words]))
            for r in records:
                candidates.setdefault(r.id, r)
        except Exception as e:
            log.warning(f"Ricerca per parole nella sessione {session_id} non riuscita: {e}")
    if not candidates:
        return [], None
    ordered, used = rerank(query, list(candidates.values()))
    sections, seen = [], set()
    for point, score in ordered:
        key = (point.payload["doc_id"], point.payload["parent_index"])
        if key in seen:
            continue
        seen.add(key)
        sections.append({"doc_id": key[0], "parent_index": key[1], "score": round(float(score), 4)})
        if len(sections) >= top_k:
            break
    log.info(f"Ricerca nella sessione {session_id}: {len(candidates)} candidati, "
             f"{len(sections)} sezioni, riordino {used}.")
    return sections, used

