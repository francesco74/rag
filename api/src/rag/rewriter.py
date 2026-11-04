"""Riscrittura della domanda: standalone query, sotto-query, keyword, coverage."""

import logging
from datetime import date

from common.config import settings
from llm_provider import get_llm_provider
from rag.concepts import format_verified_concepts
from rag.jsonparse import safe_json_parse
from rag.prompts import load_prompt_template

log = logging.getLogger("WORKER")


def parse_date_range(raw, task_id="UNKNOWN"):
    """
    Periodo degli atti indicato nella domanda, dal campo date_range del
    rewriter ({"from": "AAAA-MM-GG" | null, "to": "AAAA-MM-GG" | null}), nel
    formato del filtro sulla data ({"gte": ..., "lte": ...}). None se assente
    o non valido: un periodo sbagliato escluderebbe atti pertinenti, quindi
    nel dubbio si cerca senza filtro.
    """
    if not isinstance(raw, dict):
        return None
    out = {}
    for src, dst in (("from", "gte"), ("to", "lte")):
        value = raw.get(src)
        if value in (None, ""):
            continue
        try:
            out[dst] = date.fromisoformat(str(value).strip()[:10]).isoformat()
        except ValueError:
            log.warning(f"[{task_id}] [REWRITER] date_range ignorato: '{src}'={value!r} non è una data AAAA-MM-GG")
            return None
    if "gte" in out and "lte" in out and out["gte"] > out["lte"]:
        log.warning(f"[{task_id}] [REWRITER] date_range ignorato: inizio {out['gte']} dopo la fine {out['lte']}")
        return None
    return out or None


def transform_query(history, query, task_id="UNKNOWN", verified_concepts=None):
    """
    Transform conversational query into standalone query + search facets + keywords.

    verified_concepts: lista di concetti dal dizionario Qdrant (output di
    get_concepts_by_similarity). Se presenti, vengono iniettati nel prompt
    come vocabolario autoritativo: il rewriter li usa direttamente per la
    decomposizione geografica (comuni verificati invece che indovinati) e
    per l'espansione tematica, in un'unica chiamata — eliminando la vecchia
    catena rewrite -> lookup -> seconda chiamata di espansione -> 
    riconciliazione in Python.
    """
    log.info(f"[{task_id}] [REWRITER] Avvio analisi query: '{query}'")

    if not history:
        log.debug(f"[{task_id}] [REWRITER] Nessuna history fornita.")
        history_str = ""
    else:
        history_str = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('text', '')}"
            for msg in history
        ])
        log.debug(f"[{task_id}] [REWRITER] History iniettata (elementi: {len(history)})")

    verified_concepts_str = format_verified_concepts(verified_concepts or [])
    if verified_concepts_str:
        log.debug(f"[{task_id}] [REWRITER] Concetti verificati iniettati:\n{verified_concepts_str}")

    prompt = load_prompt_template("query_rewriter").format(
        history_str=history_str,
        query=query,
        max_sub_queries=settings.max_sub_queries,
        verified_concepts_str=verified_concepts_str,
        today=date.today().isoformat(),
    )

    try:
        log.debug(f"[{task_id}] [REWRITER] Chiamata al provider LLM in corso...")
        raw = get_llm_provider().generate_json(
            settings.query_rewriter_model_name, prompt, temperature=0.1
        )
        log.debug(f"[{task_id}] [REWRITER] Risposta grezza:\n{raw}")

        data = safe_json_parse(raw, task_id)
        standalone = data.get("standalone_query", query)
        searches   = data.get("search_queries", [query])
        keywords   = data.get("keywords", [])
        # Entità distinte che la risposta deve coprire (comuni, enti, oggetti).
        # Vuota quando la domanda non è aggregativa. Il .get con default
        # garantisce la retrocompatibilità con prompt di rewriter più vecchi.
        coverage   = data.get("coverage_targets", []) or []
        # Periodo degli atti chiesto dall'utente ("fatti nel 2025"), applicato
        # dal worker come filtro sulla data: None se la domanda non ne ha.
        date_range = parse_date_range(data.get("date_range"), task_id)

        log.info(
            f"[{task_id}] [REWRITER] ✓ Query processata. "
            f"Standalone: '{standalone}' | Facets: {len(searches)} | Keywords: {len(keywords)} | "
            f"Coverage: {len(coverage)}"
            + (f" | Periodo: {date_range.get('gte', '…')} → {date_range.get('lte', '…')}" if date_range else "")
        )

        log.debug(f"Standalone query: {standalone}")
        log.debug(f"Query individuate: {searches}")
        log.debug(f"Keywords: {keywords}")
        log.debug(f"Coverage targets: {coverage}")
        return {
            "standalone_query": standalone,
            "search_queries": searches,
            "keywords": keywords,
            "coverage_targets": coverage,
            "date_range": date_range,
        }

    except Exception as e:
        log.warning(
            f"[{task_id}] [REWRITER] ✗ Fallimento critico: {e}. Attivazione fallback (query raw).",
            exc_info=True
        )
        return {"standalone_query": query, "search_queries": [query], "keywords": [], "coverage_targets": [],
                "date_range": None}
