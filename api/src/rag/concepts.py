"""Dizionario dei concetti: ricerca per nome e per similarità."""

import logging
import re

from common.config import settings
from rag import state
from rag.state import CONCEPT_COLLECTION

log = logging.getLogger("WORKER")


def format_verified_concepts(concepts):
    """
    Formatta i concetti trovati nel dizionario per l'iniezione nel prompt
    del rewriter, nel formato "Concept: alias1, alias2, ...", una riga per
    concetto. Stringa vuota se non ci sono concetti (il prompt gestisce il
    caso esplicitamente).
    """
    if not concepts:
        return ""
    return "\n".join(
        f"{c['concept']}: {', '.join(c['aliases'])}"
        for c in concepts if c.get("concept") and c.get("aliases")
    )


def collect_concept_aliases(concepts):
    """Unione (deduplicata) di tutti gli alias dei concetti trovati, per le keyword testuali."""
    seen = set()
    out = []
    for c in concepts:
        for a in c.get("aliases", []):
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def get_concepts_by_similarity(query_vector, threshold=settings.qdrant_concept_threshold, task_id="UNKNOWN"):
    """
    Ricerca UNICA sul dizionario concettuale: ritorna TUTTI i concetti sopra
    soglia (fino a settings.qdrant_concept_max_hits), non solo il migliore.

    Sostituisce la vecchia get_aliases_by_concept (limit=1, poi limit=1 per
    categoria): il problema non era la mancanza di categorie ma il limite a
    un solo hit — una query che tocca più concetti insieme ("lavoro agile in
    Garfagnana") li recupera ora tutti in un colpo solo, quanti siano,
    generalizzando automaticamente a qualunque numero di concetti presenti
    nel dizionario, senza tassonomia da mantenere nel codice.

    Ritorna una lista di dict: [{"concept": ..., "aliases": [...],
    "category": ..., "score": ...}], ordinata per score decrescente (ordine
    nativo di Qdrant). Lista vuota su miss o errore, mai eccezioni.
    """
    if not state.qdrant_client:
        return []

    max_hits = getattr(settings, "qdrant_concept_max_hits", 5)

    try:
        hits = state.qdrant_client.query_points(
            collection_name=CONCEPT_COLLECTION,
            query=query_vector,
            limit=max_hits,
            score_threshold=threshold
        ).points

        concepts = []
        for hit in hits:
            payload = hit.payload or {}
            concepts.append({
                "concept": payload.get("concept", ""),
                "aliases": payload.get("aliases", []),
                "category": payload.get("category", ""),
                "score": hit.score,
            })

        if concepts:
            summary = ", ".join(f"'{c['concept']}' ({c['score']:.3f})" for c in concepts)
            log.info(f"[{task_id}] [CONCEPT HIT] {len(concepts)} concetti sopra soglia (max {max_hits}): {summary}")
        else:
            log.debug(f"[{task_id}] [CONCEPT MISS] Nessun concetto sopra soglia {threshold}.")
        return concepts
    except Exception as e:
        log.error(f"[{task_id}] Errore lookup concettuale su Qdrant: {e}")
        return []


def get_concepts_by_name(text, task_id="UNKNOWN"):
    """
    Concetti il cui NOME compare letteralmente nel testo (parola intera, senza
    distinzione di maiuscole): "atti sulle scuole della Versilia" -> Versilia.

    Il confronto per similarità (get_concepts_by_similarity) misura il
    significato dell'intera frase: una domanda che cita la Versilia insieme ad
    atti, lavori, anni e magari ai turni precedenti della conversazione resta
    sotto soglia anche se il nome del concetto c'è, scritto tale e quale. Qui
    il nome basta. Gli alias NON fanno scattare il concetto: "scuole di
    Viareggio" non deve allargarsi a tutti i comuni della Versilia.

    Il dizionario è piccolo (decine di voci): si legge tutto a ogni domanda.
    Lista vuota su errore, mai eccezioni.
    """
    if not state.qdrant_client or not text:
        return []
    try:
        lowered = text.lower()
        found, offset = [], None
        while True:
            points, offset = state.qdrant_client.scroll(
                collection_name=CONCEPT_COLLECTION, limit=256, offset=offset,
                with_payload=True, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                name = (payload.get("concept") or "").strip()
                if name and re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", lowered):
                    found.append({
                        "concept": name,
                        "aliases": payload.get("aliases", []),
                        "category": payload.get("category", ""),
                        "score": 1.0,
                    })
            if offset is None:
                break
        if found:
            log.info(f"[{task_id}] [CONCEPT NAME] Concetti citati per nome: {', '.join(c['concept'] for c in found)}")
        return found
    except Exception as e:
        log.error(f"[{task_id}] Errore lookup dei concetti per nome: {e}")
        return []


def find_concepts(query, concept_vector, task_id="UNKNOWN"):
    """
    Concetti del dizionario per la domanda: prima quelli citati per nome nella
    domanda corrente (sicuri), poi quelli trovati per similarità sul vettore
    di lookup (domanda + ultimi turni), senza doppioni.
    """
    by_name = get_concepts_by_name(query, task_id=task_id)
    by_similarity = get_concepts_by_similarity(concept_vector, task_id=task_id) if concept_vector is not None else []
    names = {c["concept"] for c in by_name}
    return by_name + [c for c in by_similarity if c["concept"] not in names]
