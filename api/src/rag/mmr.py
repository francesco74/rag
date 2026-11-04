"""Diversificazione MMR dei parent nella redistribuzione degli slot liberi."""

import logging

import numpy as np

log = logging.getLogger("WORKER")

# Applicato SOLO in fase di redistribuzione degli slot liberi (vedi retrieve_chunks,
# step 5). La quota garantita per sottoquery non viene mai toccata da MMR: questo
# limita il rischio di escludere un parent rilevante ma "solo" per somiglianza
# vettoriale con un altro (vedi discussione: il vettore-proxy, un centroid dei
# chunk recuperati, non rappresenta l'intero parent).


MMR_LOG_TAG = "[MMR]"


def cosine(v1, v2):
    """Similarità coseno tra due vettori. Ritorna 0.0 se uno dei due è nullo."""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0


def select_with_mmr(candidates, selected_vectors, lam, sim_threshold, k, task_id="UNKNOWN"):
    """
    Seleziona fino a k parent tra 'candidates' massimizzando rilevanza e penalizzando
    la ridondanza rispetto ai parent già selezionati (already_selected, quota garantita).

    candidates: list di dict {"pid": str, "score": float, "vector": np.ndarray}
    selected_vectors: dict {pid: vector} dei parent GIÀ in quota (fissi, non rimovibili,
                      usati solo per calcolare la penalità di similarità)
    lam: peso rilevanza vs diversità. lam=1.0 -> comportamento identico al ranking puro.
    sim_threshold: la penalità scatta solo se la similarità supera questa soglia
                   (evita di penalizzare parent solo "un po'" simili ma complementari)
    k: numero di slot da riempire

    Ritorna: (picked_pids, debug_info) dove debug_info è una lista di dict utile al log
    (pid, score, max_sim_to_selected, penalizzato: bool)
    """
    debug_info = []

    if lam >= 1.0:
        # Nessuna penalità: puro ranking per score, identico al comportamento pre-MMR.
        ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
        picked = [c["pid"] for c in ordered[:k]]
        log.debug(f"[{task_id}] {MMR_LOG_TAG} lam=1.0 -> bypass, ranking puro su {len(candidates)} candidati.")
        return picked, debug_info

    picked = []
    pool = list(candidates)
    current_vectors = dict(selected_vectors)  # cresce ad ogni pick

    log.info(
        f"[{task_id}] {MMR_LOG_TAG} Avvio selezione: {len(candidates)} candidati, "
        f"{len(selected_vectors)} già selezionati (quota), lam={lam}, "
        f"sim_threshold={sim_threshold}, slot da riempire={k}."
    )

    while pool and len(picked) < k:
        best_item, best_mmr, best_max_sim, best_penalized = None, float("-inf"), 0.0, False

        for item in pool:
            valid_selected_vectors = [v for v in current_vectors.values() if v is not None]
            if valid_selected_vectors and item.get("vector") is not None:
                max_sim = max(cosine(item["vector"], v) for v in valid_selected_vectors)
            else:
                max_sim = 0.0  # nessun vettore disponibile: non penalizzabile per similarità
            penalized = max_sim >= sim_threshold
            penalty = max_sim if penalized else 0.0
            mmr_score = lam * item["score"] - (1 - lam) * penalty

            if mmr_score > best_mmr:
                best_mmr, best_item = mmr_score, item
                best_max_sim, best_penalized = max_sim, penalized

        picked.append(best_item["pid"])
        current_vectors[best_item["pid"]] = best_item["vector"]
        pool.remove(best_item)

        debug_info.append({
            "pid": best_item["pid"],
            "score": round(best_item["score"], 4),
            "max_sim_to_selected": round(best_max_sim, 4),
            "penalized": best_penalized,
            "mmr_score": round(best_mmr, 4),
        })

        log.debug(
            f"[{task_id}] {MMR_LOG_TAG} Scelto parent_id={best_item['pid']} "
            f"(score={best_item['score']:.4f}, max_sim_to_selected={best_max_sim:.4f}, "
            f"penalizzato={'sì' if best_penalized else 'no'}, mmr_score={best_mmr:.4f})."
        )

    n_penalized = sum(1 for d in debug_info if d["penalized"])
    log.info(
        f"[{task_id}] {MMR_LOG_TAG} Selezione completata: {len(picked)}/{k} slot riempiti, "
        f"{n_penalized} scelti nonostante penalità di ridondanza (sopra soglia {sim_threshold})."
    )

    return picked, debug_info
