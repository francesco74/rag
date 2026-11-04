"""Ricerca testuale a combinazioni di parole chiave."""

import logging
import math
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import models

from common.config import settings
from rag import state
from rag.state import QDRANT_COLLECTION

log = logging.getLogger("WORKER")


# Gruppi di parole chiave combinati al massimo (i più rari): 2^6 - 1 = 63 conteggi.
KEYWORD_MAX_GROUPS = 6


def build_keyword_groups(keywords, concepts=None):
    """
    Raggruppa le parole chiave per la ricerca testuale a combinazioni.

    Gli alias di uno stesso concetto del dizionario sono ALTERNATIVE, non
    requisiti: "Barga" è un comune della Garfagnana, e un pezzo che cita sia
    "Garfagnana" sia "Barga" non è più pertinente di uno che cita solo
    "Barga" (tipicamente è un DUP che elenca tutti i comuni). Ogni concetto
    diventa quindi UN gruppo in OR (nome + alias); ogni altra parola chiave è
    un gruppo a sé. Le parole chiave che coincidono con il nome o un alias di
    un concetto confluiscono nel suo gruppo.

    Ritorna [{"label", "terms", "geo"}], con i termini in minuscolo.
    """
    groups, by_term = [], {}
    for c in concepts or []:
        name = (c.get("concept") or "").strip()
        terms = [t for t in dict.fromkeys([name.lower()] + [a.lower().strip() for a in c.get("aliases", [])])
                 if len(t) > 2]
        if not terms or any(t in by_term for t in terms):
            continue
        group = {"label": name or terms[0], "terms": terms, "geo": c.get("category") == "geografico"}
        groups.append(group)
        for t in terms:
            by_term[t] = group
    for k in keywords or []:
        w = (k or "").lower().strip()
        if len(w) <= 2 or w in by_term:
            continue
        group = {"label": w, "terms": [w], "geo": False}
        groups.append(group)
        by_term[w] = group
    return groups


def plan_keyword_combinations(n_groups, count_fn, max_hits, max_workers=8):
    """
    Conta i pezzi di ogni combinazione in AND dei gruppi (indici 0..n-1) e
    sceglie quelle da usare.

    Livello per livello, dalle singole alle combinazioni più ampie; una
    combinazione che contiene una sotto-combinazione a zero risultati è zero
    anch'essa e non si conta. count_fn(tuple_di_indici) -> int, o None se il
    conteggio fallisce (la combinazione si scarta).

    Ritorna (usate, counts):
      - usate: [(combinazione, conteggio)] con 0 < conteggio <= max_hits,
        prima i livelli più alti (più gruppi in AND), poi le più rare;
        se nessuna è sotto soglia, la sola combinazione più ampia e più rara
        (ripiego: la più specifica disponibile);
      - counts: tutti i conteggi calcolati, per il log.
    """
    from itertools import combinations
    counts = {}
    for level in range(1, n_groups + 1):
        todo = [combo for combo in combinations(range(n_groups), level)
                if all(counts.get(sub) for sub in combinations(combo, level - 1))] if level > 1 \
            else [(i,) for i in range(n_groups)]
        if not todo:
            break
        with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
            for combo, n in zip(todo, pool.map(count_fn, todo)):
                counts[combo] = n or 0
    found = [(combo, n) for combo, n in counts.items() if n > 0]
    found.sort(key=lambda item: (-len(item[0]), item[1]))
    used = [(combo, n) for combo, n in found if n <= max_hits]
    if not used and found:
        used = found[:1]
    return used, counts


def keyword_search(keywords, concepts, must_conditions, needs_vectors, task_id="UNKNOWN"):
    """
    Ricerca testuale a COMBINAZIONI di parole chiave.

    Lo scroll di Qdrant non ordina per pertinenza: dentro un filtro
    restituisce i punti in ordine di id (UUID casuali). L'ordine lo
    danno quindi le combinazioni:
      1. parole chiave e concetti diventano gruppi (alias di un
         concetto in OR, vedi build_keyword_groups);
      2. si conta ogni combinazione in AND dei gruppi (pochi ms l'una,
         indice testuale su "content");
      3. si riempiono i posti prima con le combinazioni con più gruppi,
         poi, a parità, con le più rare. Una combinazione con più di
         KEYWORD_MAX_HITS pezzi è troppo generica (ne uscirebbe un
         campione a caso) e si salta: "lavori" da sola no, "lavori +
         scuole + versilia" sì. Se nessuna è sotto soglia, si usa la più
         specifica;
      4. in una combinazione con un concetto geografico i posti si
         dividono fra i comuni, perché uno solo non li prenda tutti.
    """
    groups = build_keyword_groups(keywords, concepts)
    if not groups:
        return []
    try:
        target_size = settings.qdrant_syntactic_size

        def term_cond(term):
            return models.FieldCondition(key="content", match=models.MatchText(text=term))

        def group_cond(group):
            if len(group["terms"]) == 1:
                return term_cond(group["terms"][0])
            return models.Filter(should=[term_cond(t) for t in group["terms"]])

        def count_combo(combo):
            try:
                return state.qdrant_client.count(
                    collection_name=QDRANT_COLLECTION,
                    count_filter=models.Filter(must=must_conditions + [group_cond(groups[i]) for i in combo]),
                    exact=True,
                ).count
            except Exception as e:
                log.warning(f"Conteggio parole chiave {[groups[i]['label'] for i in combo]} fallito: {e}")
                return None

        # I gruppi più rari, al massimo KEYWORD_MAX_GROUPS
        single = {i: count_combo((i,)) for i in range(len(groups))}
        kept = sorted((i for i in single if single[i]), key=lambda i: single[i])[:KEYWORD_MAX_GROUPS]
        dropped = [groups[i]["label"] for i in single if i not in kept]
        groups = [groups[i] for i in kept]
        if not groups:
            log.info(f"[{task_id}] [KEYWORD] Nessun risultato per le parole chiave: {', '.join(dropped)}")
            return []

        used, counts = plan_keyword_combinations(len(groups), count_combo, settings.keyword_max_hits)
        label = lambda combo: " + ".join(groups[i]["label"] for i in combo)
        too_wide = sum(1 for n in counts.values() if n > settings.keyword_max_hits)
        groups_str = ", ".join(f"{g['label']}({counts[(i,)]})" for i, g in enumerate(groups))
        log.info(
            f"[{task_id}] [KEYWORD] Gruppi: {groups_str}"
            + (f" | senza risultati o oltre {KEYWORD_MAX_GROUPS} gruppi: {', '.join(dropped)}" if dropped else "")
        )
        log.info(
            f"[{task_id}] [KEYWORD] Combinazioni usate, in ordine: "
            f"{', '.join(f'{label(c)}({n})' for c, n in used) or 'nessuna'}"
            f" | troppo generiche (> {settings.keyword_max_hits}): {too_wide}"
        )

        seen_hashes: set = set()
        unique_hits = []

        def scroll_into(conds, limit):
            """Aggiunge fino a `limit` pezzi nuovi (senza duplicati) che soddisfano conds."""
            added, next_offset, iterations = 0, None, 0
            while added < limit and len(unique_hits) < target_size and iterations < 5:
                res, next_offset = state.qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=models.Filter(must=must_conditions + conds),
                    limit=target_size * 2,  # batch ampio per compensare i duplicati scartati
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=needs_vectors  # centroid-per-parent: solo se MMR è attivo
                )
                iterations += 1
                if not res:
                    break
                for point in res:
                    h = point.payload.get("content_hash") if point.payload else None
                    if h and h not in seen_hashes:
                        seen_hashes.add(h)
                        unique_hits.append(point)
                        added += 1
                        if added >= limit or len(unique_hits) >= target_size:
                            break
                if next_offset is None:
                    break
            return added

        for combo, _ in used:
            if len(unique_hits) >= target_size:
                break
            before = len(unique_hits)
            geo = next((i for i in combo if groups[i]["geo"] and len(groups[i]["terms"]) > 1), None)
            if geo is None:
                scroll_into([group_cond(groups[i]) for i in combo], target_size)
            else:
                # Posti divisi fra i luoghi del concetto geografico
                others = [group_cond(groups[i]) for i in combo if i != geo]
                places = groups[geo]["terms"]
                share = max(1, math.ceil((target_size - len(unique_hits)) / len(places)))
                for term in places:
                    scroll_into(others + [term_cond(term)], share)
            log.debug(f"[{task_id}] [KEYWORD] '{label(combo)}': +{len(unique_hits) - before} pezzi.")

        log.debug(f"Keyword search returned {len(unique_hits)} unique hits (deduped).")
        return unique_hits
    except Exception as e:
        log.error(f"Keyword search failed: {e}")
        return []
