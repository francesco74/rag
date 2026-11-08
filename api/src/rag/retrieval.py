"""Recupero dei chunk: ricerca vettoriale e testuale, rerank, selezione dei parent."""

import hashlib
import logging
import math
import time
from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np
from qdrant_client import models

from common.config import settings
from common.db_logger import get_db_connection
from common.utility import clean_ocr_text, normalize_ws
from rag import state
from rag.boilerplate import build_removal_by_subtopic, load_active_boilerplate
from rag.context import document_context_metadata, SNIPPETS_PER_PARENT
from rag.keywords import keyword_search
from rag.mmr import MMR_LOG_TAG, select_with_mmr
from rag.state import QDRANT_COLLECTION
from reranker import ONNXReranker, VLLMReranker

log = logging.getLogger("WORKER")


_RERANKER_POOL: list = []


_POOL_SIZE = settings.reranker_pool_size


def safe_sigmoid(x):
    """
    Converte i logit in probabilità (0-1) prevenendo OverflowError in Python.
    """
    if x < -100:  # Qualsiasi logit sotto -100 è di fatto 0 probabilità
        return 0.0
    return 1 / (1 + math.exp(-x))


# Tentativo 0: comportamento attuale (recall di base, MMR di fatto disattivato).
# Tentativo 1: allarga la RECALL (più candidati, soglia più permissiva), MMR ancora spento
#              -> "forse non ho pescato abbastanza materiale, allargo la rete".
# Tentativo 2: recall ancora più larga + MMR attivo (lam<1)
#              -> "ho materiale ma è probabilmente un cluster semantico ripetitivo,
#                  forzo diversità per esplorare angolazioni diverse".
RETRY_PROFILES = {
    0: {"size_mult": 1.0, "threshold_mult": 1.0, "mmr_lambda": 1.0},
    1: {"size_mult": 1.5, "threshold_mult": 0.8, "mmr_lambda": 1.0},
    2: {"size_mult": 2.0, "threshold_mult": 0.7, "mmr_lambda": 0.8},
}


def profile_kwargs(attempt, task_id="UNKNOWN"):
    """Traduce il numero di tentativo in parametri concreti per retrieve_chunks."""
    profile = RETRY_PROFILES.get(attempt, RETRY_PROFILES[max(RETRY_PROFILES)])
    kwargs = {
        "semantic_size": int(settings.qdrant_semantic_size * profile["size_mult"]),
        "semantic_threshold": settings.qdrant_semantic_threshold * profile["threshold_mult"],
        "mmr_lambda": profile["mmr_lambda"],
    }
    log.info(
        f"[{task_id}] [RETRY_PROFILE] Tentativo {attempt} -> "
        f"semantic_size={kwargs['semantic_size']} "
        f"(x{profile['size_mult']}), semantic_threshold={kwargs['semantic_threshold']:.3f} "
        f"(x{profile['threshold_mult']}), mmr_lambda={kwargs['mmr_lambda']}."
    )
    return kwargs


def retrieve_chunks(search_queries, vectors_list, keywords, topic_id, selected_sub_topics,
                     metadata_filters, include_undated=True,
                     semantic_size=None, semantic_threshold=None, mmr_lambda=1.0,
                     exclude_parent_ids=None,
                     query_semantic_vector=None,
                     rerank_size=None,
                     concepts=None,
                     task_id="UNKNOWN"):
    """
    Recupera, fonde e riordina i chunk dal Vector DB in modo sicuro.

    concepts: concetti del dizionario trovati per la domanda; nella ricerca
        testuale i loro alias valgono come alternative (vedi build_keyword_groups).

    semantic_size / semantic_threshold: sovrascrivono i default di settings, usati
        dai profili di retry per allargare progressivamente la recall.
    rerank_size: passaggi tenuti dopo il rerank per ogni sotto-domanda; default
        settings.rerank_size, settings.rerank_max_size con la ricerca approfondita.
    mmr_lambda: peso rilevanza/diversità nella redistribuzione degli slot liberi.
        1.0 = comportamento identico a prima dell'introduzione di MMR.
    exclude_parent_ids: set di parent_id da escludere del tutto dalla selezione
        (usato dall'ultimo tentativo di fallback, per non riproporre parent
        già passati al generatore in un tentativo precedente e già falliti).
        L'esclusione avviene PRIMA della quota/MMR, così il budget di parent
        non viene sprecato su candidati che verrebbero comunque scartati.
    """
    if not state.qdrant_client:
        log.error("Qdrant client not available! Retrieval aborted.")
        return [], [], {}

    # Boilerplate attivo per (topic, sub_topic), caricato una volta per
    # l'intero retrieval. La decisione "quali frasi rimuovere" dipende solo da
    # query e frasi, non dai chunk: si prende UNA volta qui, non per ogni chunk.
    # removal_by_subtopic: {sub_topic_id: regex delle frasi da togliere}.
    boilerplate_map = load_active_boilerplate(topic_id, selected_sub_topics)
    removal_by_subtopic = build_removal_by_subtopic(
        boilerplate_map, " ".join(search_queries), query_semantic_vector
    )

    semantic_size = semantic_size or settings.qdrant_semantic_size
    rerank_size = rerank_size or settings.rerank_size
    semantic_threshold = semantic_threshold if semantic_threshold is not None else settings.qdrant_semantic_threshold

    # I vettori dei chunk servono SOLO a costruire i centroid-per-parent usati
    # da MMR. Con mmr_lambda=1.0 (il profilo del tentativo 0, cioè il caso
    # normale) select_with_mmr va in bypass e i centroidi vengono buttati senza
    # essere mai letti: li si pagava due volte, in trasferimento di rete e in
    # calcolo. Ogni gruppo trasporta un embedding gemini da 3072 dimensioni
    # serializzato in JSON — su 30 gruppi × 6 sottoquery è il motivo per cui le
    # search sfioravano il timeout.
    needs_vectors = mmr_lambda < 1.0

    log.info(f"=== Inizio Retrieval per topic '{topic_id}' ===")
    log.info(
        f"[{task_id}] [RETRIEVAL_PARAMS] semantic_size={semantic_size}, "
        f"rerank_size={rerank_size}, "
        f"semantic_threshold={semantic_threshold:.3f}, mmr_lambda={mmr_lambda}, "
        f"with_vectors={needs_vectors}."
    )
    log.debug(f"Search Queries: {search_queries} | Keywords: {keywords}")
    
    try:
        must_conditions = [models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))]
        if selected_sub_topics:
            must_conditions.append(models.FieldCondition(key="sub_topic_id", match=models.MatchAny(any=selected_sub_topics)))

        # === INIEZIONE FILTRI METADATI ===
        if metadata_filters:
            for key, value in metadata_filters.items():
                
                # Ricerca full-text tokenizzata: {"oggetto__text": "disposizioni generali"}
                if key.endswith("__text"):
                    real_key = key[:-6]  # rimuove "__text"
                    must_conditions.append(models.FieldCondition(
                        key=real_key,
                        match=models.MatchText(text=str(value))
                    ))
                
                # Filtro range: {"data": {"gte": "2024-01-01", "lte": "2024-06-30"}}
                # oppure numerico: {"prezzo": {"gte": 10, "lte": 100}}
                # NB: i metadati sono dinamici (dipendono dalla fonte di ingestion),
                # quindi un documento può legittimamente non avere questo campo.
                # In tal caso lo includiamo comunque (non escludiamo per "assenza
                # di dato", solo per "dato fuori range"): range_condition è
                # soddisfatta OPPURE il campo non esiste affatto.
                elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                    range_values = [v for v in (value.get("gte"), value.get("lte")) if v is not None]
                    is_date_range = any(isinstance(v, str) for v in range_values)

                    if is_date_range:
                        # models.Range accetta solo float: per stringhe (es. date
                        # ISO "YYYY-MM-DD") serve DatetimeRange, altrimenti Qdrant
                        # solleva un errore di validazione.
                        range_condition = models.FieldCondition(
                            key=key,
                            range=models.DatetimeRange(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
                        )
                    else:
                        range_condition = models.FieldCondition(
                            key=key,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
                        )

                    if include_undated:
                        # Il documento passa se rientra nel range OPPURE se il
                        # campo non esiste affatto (comportamento di default:
                        # non escludiamo per "assenza di dato").
                        must_conditions.append(models.Filter(
                            should=[
                                range_condition,
                                models.IsEmptyCondition(is_empty=models.PayloadField(key=key))
                            ]
                        ))
                    else:
                        # L'utente ha chiesto di escludere i documenti privi
                        # del campo: nessuna clausola di "should", il documento
                        # deve soddisfare strettamente il range.
                        must_conditions.append(range_condition)
                
                # Filtro multi-valore: {"stato": ["approvato", "pubblicato"]}
                elif isinstance(value, list):
                    must_conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    ))
                
                # Filtro esatto: {"numero_atto": "123"}
                else:
                    must_conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ))

        # ==========================================================
        # 1. FUNZIONI DI RICERCA (Con Exception Handling Interno)
        # ==========================================================
        def safe_vector_search(vector, idx):
            try:
                res = state.qdrant_client.query_points_groups(
                    collection_name=QDRANT_COLLECTION,
                    query=vector,
                    group_by="content_hash",
                    group_size=1,
                    limit=semantic_size,
                    score_threshold=semantic_threshold,
                    query_filter=models.Filter(must=must_conditions),
                    # Con la copia int8 (setup_qdrant.collection_params): ricerca sulla copia
                    # compressa, poi i migliori oversampling x limite riordinati con i vettori
                    # pieni. Senza quantizzazione Qdrant ignora questi parametri.
                    search_params=models.SearchParams(quantization=models.QuantizationSearchParams(
                        rescore=True, oversampling=settings.qdrant_oversampling)),
                    with_vectors=needs_vectors  # centroid-per-parent: solo se MMR è attivo
                )
                points = [group.hits[0] for group in res.groups if group.hits]
                log.debug(f"Vector search #{idx} returned {len(points)} grouped hits.")
                return points
            except Exception as e:
                log.error(f"Vector search #{idx} failed: {e}")
                return []
            
        def safe_keyword_search():
            return keyword_search(keywords, concepts, must_conditions, needs_vectors, task_id)

        # ==========================================================
        # ESECUZIONE PARALLELA 
        # ==========================================================
        hits_per_query = [[] for _ in search_queries]
        search_started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=min(10, len(vectors_list) + 1)) as executor:
            future_kw = executor.submit(safe_keyword_search)
            future_map = {
                executor.submit(safe_vector_search, vec, idx): idx
                for idx, vec in enumerate(vectors_list)
            }
            keyword_hits = future_kw.result()
            for future, idx in future_map.items():
                try:
                    hits_per_query[idx] = future.result()
                except Exception as e:
                    log.error(f"Error resolving vector search future #{idx}: {e}")
        t_search = time.perf_counter() - search_started
 
        # Keyword hits come candidati extra della query principale
        if keyword_hits:
            kw_ids = {h.id for h in hits_per_query[0]}
            hits_per_query[0] = hits_per_query[0] + [h for h in keyword_hits if h.id not in kw_ids]

        # ==========================================================
        # 3. RERANK PER-QUERY — salva top_chunks per riuso nella redistribuzione
        #
        # Ogni sotto-query pesca dal pool di istanze ONNXReranker (get_reranker_pool),
        # una per thread concorrente invece di condividerne una sola. Prima, tutti i
        # thread del ThreadPoolExecutor sotto chiamavano .rerank() sulla STESSA
        # istanza globale (get_reranker()): la sessione ONNX è deliberatamente
        # ORT_SEQUENTIAL (per evitare deadlock in Celery), quindi 2 thread paralleli
        # si contendevano lo stesso budget di calcolo invece di raddoppiarlo.
        # ==========================================================
        pool = get_reranker_pool()
        n_queries = len(search_queries)
        top_chunks_per_query: list[list] = [[] for _ in range(n_queries)]

        # Giudizio del reranker su ogni chunk, come probabilità 0..1, per
        # sottoquery: lo stesso chunk può valere diversamente per domande
        # diverse. Usato come rilevanza dei parent (quote di contesto, MMR,
        # ordine dei core) al posto della similarità vettoriale di Qdrant,
        # che varia poco tra un chunk e l'altro e non coglie la pertinenza.
        rerank_probs: dict[tuple[int, str], float] = {}

        def process_single_rerank(q_idx, query_str, candidates):
            if not candidates:
                log.debug(f"Sottoquery #{q_idx} '{query_str[:40]}': nessun candidato.")
                return []
            
            # De-duplica i chunk per id in modo più efficiente
            seen_chunk_ids = {c.id: c for c in candidates}
            unique_candidates = list(seen_chunk_ids.values())

            # De-duplica cross duplicati
            n_before_content_dedup = len(unique_candidates)
            seen_content_hashes = {}
            deduped_candidates = []
            for c in unique_candidates:
                h = c.payload.get("content_hash") if c.payload else None
                if not h:
                    content = (c.payload.get("content", "") if c.payload else "")
                    h = hashlib.md5(" ".join(content.lower().split()).encode()).hexdigest()
                if h not in seen_content_hashes:
                    seen_content_hashes[h] = True
                    deduped_candidates.append(c)
            unique_candidates = deduped_candidates

            if len(unique_candidates) < n_before_content_dedup:
                log.debug(
                    f"Sottoquery #{q_idx} '{query_str[:40]}': "
                    f"{n_before_content_dedup} → {len(unique_candidates)} candidati "
                    f"(rimossi {n_before_content_dedup - len(unique_candidates)} duplicati)."
                )

            # Rerank effettivo — istanza dedicata dal pool per questo q_idx,
            # non condivisa con gli altri thread concorrenti.
            reranker_instance = pool[q_idx % len(pool)] if pool else None
            if reranker_instance:
                # Preparazione del testo passato al reranker, in quest'ordine:
                #   1. normalizzazione whitespace/entità HTML (&nbsp;), PRIMA del
                #      taglio: un chunk pieno di indentazione sprecherebbe la
                #      finestra del cross-encoder in spazi invece che in testo;
                #   2. rimozione del boilerplate attivo, protetta dalla guardia
                #      lessicale (una frase presente nella query non si tocca);
                #   3. troncamento a rerank_truncate.
                # Tutto ciò riguarda SOLO il testo per il ranking: il contenuto
                # renderizzato per l'utente e per il generatore resta integro.
                docs_content = []
                for c in unique_candidates:
                    raw = normalize_ws(c.payload.get("content", "") if c.payload else "")
                    # Rimuove SOLO il boilerplate del sub_topic DI QUESTO chunk:
                    # la formula delle determine non tocca i chunk dei decreti.
                    # La decisione di pertinenza è già stata presa (regex pronta);
                    # qui è pura sostituzione, nessun calcolo per chunk.
                    st = c.payload.get("sub_topic_id") if c.payload else None
                    rx = removal_by_subtopic.get(st) if st else None
                    txt = rx.sub(" ", raw) if rx else raw
                    docs_content.append(txt[:settings.rerank_truncate])
                try:
                    reranked = reranker_instance.rerank(query_str, docs_content)
                    reranked.sort(key=lambda x: x.score, reverse=True)
                    top_chunks = []
                    for res in reranked[:rerank_size]:
                        prob = getattr(res, "probability", None)
                        if prob is None:
                            prob = safe_sigmoid(res.score)
                        if prob >= settings.min_prob_threshold or len(top_chunks) < 2:
                            chunk = unique_candidates[res.index]
                            top_chunks.append(chunk)
                            rerank_probs[(q_idx, str(chunk.id))] = prob
                        else:
                            break
                    return top_chunks
                except Exception as e:
                    log.error(f"Reranker failed for query #{q_idx}: {e}. Fallback.")
                    
            unique_candidates.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            return unique_candidates[:rerank_size]
        
        # Il limite serve a non contendersi i core del pod; vLLM lavora altrove.
        max_workers = n_queries if settings.rerank_provider == "vllm" else min(n_queries, settings.max_reranker_thread)
        rerank_started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_rerank, q_idx, q_str, cands): q_idx
                for q_idx, (q_str, cands) in enumerate(zip(search_queries, hits_per_query))
            }
            
            for future in as_completed(future_to_idx):
                q_idx = future_to_idx[future]
                try:
                    top_chunks_per_query[q_idx] = future.result()
                    log.debug(f"Sottoquery #{q_idx}: rerank completato con {len(top_chunks_per_query[q_idx])} chunk finali.")
                except Exception as e:
                    log.error(f"Errore critico nel thread di rerank per la query #{q_idx}: {e}", exc_info=True)
                    top_chunks_per_query[q_idx] = []
        t_rerank = time.perf_counter() - rerank_started


        # ==========================================================
        # 3bis. CENTROID PER PARENT (vettore-proxy per MMR)
        #
        # Il parent non ha un proprio embedding: usiamo la media (centroid) dei
        # vettori di tutti i chunk recuperati per quel parent, tra tutte le
        # sottoquery. È più rappresentativo di un singolo "chunk vincitore"
        # perché riflette tutto ciò che il retrieval ha effettivamente trovato
        # di quel documento, non solo il suo frammento più rilevante.
        # ==========================================================
        parent_chunk_vectors: dict[str, list] = {}
        parent_best_score: dict[str, float] = {}
        # Tutte le sottoquery che hanno recuperato quel parent (non solo quella
        # che se lo aggiudica in quota): serve per l'header del contesto, dove
        # dichiariamo al modello la PROVENIENZA della ricerca.
        parent_retrieved_by: dict[str, set] = {}
        n_score_fallback = 0  # chunk senza giudizio del reranker (vedi sotto)
        # Miglior probabilità del reranker per parent e sottoquery che l'ha
        # data: serve ai posti extra per pertinenza (passo 5ter), sempre
        # calcolata qualunque sia CONTEXT_SCORE_SOURCE.
        parent_rerank_best: dict[str, tuple[float, int]] = {}
        # NEW: fino a SNIPPETS_PER_PARENT child (contenuto + score) per parent,
        # usati da generate_answer per il windowing invece del troncamento cieco.
        parent_top_snippets: dict[str, list] = {}

        exclude_parent_ids = exclude_parent_ids or set()
        excluded_count = 0

        for q_idx, chunks in enumerate(top_chunks_per_query):
            for chunk in chunks:
                pid = chunk.payload.get("parent_id") if chunk.payload else None
                if not pid:
                    continue
                if pid in exclude_parent_ids:
                    excluded_count += 1
                    continue
                parent_retrieved_by.setdefault(pid, set()).add(q_idx)
                vector_score = getattr(chunk, "score", 0.0) or 0.0
                rerank_prob = rerank_probs.get((q_idx, str(chunk.id)))
                if rerank_prob is not None and rerank_prob > parent_rerank_best.get(pid, (-1.0, 0))[0]:
                    parent_rerank_best[pid] = (rerank_prob, q_idx)
                if settings.context_score_source == "rerank":
                    # Se il reranker non era disponibile (fallback) si usa la
                    # similarità vettoriale: stessa scala 0..1, nessun buco.
                    score = rerank_probs.get((q_idx, str(chunk.id)))
                    if score is None:
                        score = vector_score
                        n_score_fallback += 1
                else:
                    score = vector_score
                if pid not in parent_best_score or score > parent_best_score[pid]:
                    parent_best_score[pid] = score

                if needs_vectors:
                    vec = getattr(chunk, "vector", None)
                    if vec is not None:
                        parent_chunk_vectors.setdefault(pid, []).append(np.asarray(vec))

                chunk_content = chunk.payload.get("content", "") if chunk.payload else ""
                if chunk_content:
                    snippets = parent_top_snippets.setdefault(pid, [])
                    snippets.append({
                        "content": chunk_content,
                        "score": score,
                        # Popolati dal backfill (backfill_chunk_offsets.py) o
                        # dall'ingestion futura; None se assenti — in quel caso
                        # compute_core_spans ripiega sulla ricerca testuale.
                        "start_char": chunk.payload.get("start_char"),
                        "end_char": chunk.payload.get("end_char"),
                    })
                    snippets.sort(key=lambda s: s["score"], reverse=True)
                    del snippets[SNIPPETS_PER_PARENT:]

        log.debug(
            f"[{task_id}] [SCORE] Rilevanza dei parent da '{settings.context_score_source}'"
            + (f" ({n_score_fallback} chunk senza giudizio del reranker: similarità vettoriale)"
               if n_score_fallback else "")
            + f". Migliori: {sorted((round(v, 3) for v in parent_best_score.values()), reverse=True)[:8]}"
        )

        if exclude_parent_ids:
            log.info(f"[{task_id}] [EXCLUDE] {excluded_count} chunk scartati (parent già usati in tentativi precedenti: {len(exclude_parent_ids)}).")

        if needs_vectors:
            parent_centroids = {
                pid: np.mean(vecs, axis=0) for pid, vecs in parent_chunk_vectors.items()
            }
            n_missing_vectors = sum(1 for pid in parent_best_score if pid not in parent_centroids)
            log.info(
                f"[{task_id}] {MMR_LOG_TAG} Centroid calcolati per {len(parent_centroids)} parent "
                f"(su {len(parent_best_score)} parent candidati; {n_missing_vectors} senza vettore disponibile)."
            )
        else:
            # Dizionario vuoto e non None: la redistribuzione più sotto filtra
            # già i pid privi di centroid ("if pid in parent_centroids"), quindi
            # con MMR in bypass il ramo degrada da solo a ranking puro senza
            # bisogno di un secondo controllo su mmr_lambda.
            parent_centroids = {}
            log.debug(
                f"[{task_id}] {MMR_LOG_TAG} mmr_lambda={mmr_lambda}: centroidi non calcolati "
                f"(vettori non richiesti a Qdrant)."
            )

        # ==========================================================
        # 4. QUOTA GARANTITA PER SOTTOQUERY
        #
        # Ogni sottoquery contribuisce al massimo PARENTS_PER_QUERY
        # parent distinti. Si scende la lista rerankkata e si prendono
        # i primi PARENTS_PER_QUERY parent_id unici incontrati.
        # I parent già selezionati da sottoquery precedenti non vengono
        # ricontati: ogni slot è un documento NUOVO nel pool finale.
        # ==========================================================
        quota_per_query: list[list[str]] = []
        already_selected: set[str] = set()
 
        for q_idx, (query_str, top_chunks) in enumerate(zip(search_queries, top_chunks_per_query)):
            selected_for_query: list[str] = []
            seen_parents_this_query: set[str] = set()
 
            for chunk in top_chunks:
                pid = chunk.payload.get("parent_id") if chunk.payload else None
                if not pid or pid in seen_parents_this_query:
                    continue
                seen_parents_this_query.add(pid)
                if pid not in already_selected:
                    already_selected.add(pid)
                    selected_for_query.append(pid)
                    if len(selected_for_query) >= settings.parents_per_query:
                        break
 
            quota_per_query.append(selected_for_query)
            log.info(
                f"Sottoquery #{q_idx} '{query_str[:40]}': "
                f"{len(top_chunks)} chunk → {len(selected_for_query)} parent in quota."
            )
 
        # ==========================================================
        # 5. REDISTRIBUZIONE SLOT LIBERI
        #
        # Se alcune sottoquery hanno trovato 0 risultati (o meno di
        # PARENTS_PER_QUERY), i loro slot vengono offerti alle
        # sottoquery con più materiale disponibile, in round-robin.
        # Questo evita di sprecare budget quando, ad esempio, 4 comuni
        # su 5 non hanno documenti e uno solo ha molti risultati.
        # ==========================================================
        total_budget = settings.parents_per_query * n_queries
        slots_free = total_budget - len(already_selected)

        # Inizializzati qui così restano definiti (a 0/vuoto) anche se non
        # scatta la redistribuzione — servono per le metriche finali.
        n_mmr_candidates = 0
        n_mmr_penalized = 0

        if slots_free > 0:
            log.info(f"Redistribuzione: {slots_free} slot liberi su {total_budget} totali.")

            # Riserve per ogni query: parent validi oltre la quota, nell'ordine del reranker.
            # pid_owner_query serve per sapere a quale sottoquery "restituire" il pid dopo
            # la selezione MMR, così quota_per_query resta coerente con la struttura precedente.
            reserve_per_query: list[list[str]] = []
            pid_owner_query: dict[str, int] = {}
            for q_idx, top_chunks in enumerate(top_chunks_per_query):
                reserve: list[str] = []
                seen_parents: set[str] = set()
                for chunk in top_chunks:
                    pid = chunk.payload.get("parent_id") if chunk.payload else None
                    if not pid or pid in seen_parents:
                        continue
                    seen_parents.add(pid)
                    if pid not in already_selected:
                        reserve.append(pid)
                        pid_owner_query.setdefault(pid, q_idx)
                reserve_per_query.append(reserve)

            # Pool unico di candidati (deduplicato) per la selezione MMR globale.
            # Solo i pid con centroid disponibile entrano nella valutazione MMR;
            # quelli senza vettore (raro: mismatch nel recupero with_vectors) vengono
            # solo loggati, non selezionabili da MMR (con lam=1.0 il problema non si pone,
            # perché MMR non serve nemmeno il vettore per ordinare per score).
            seen_reserve: set[str] = set()
            reserve_pool = []
            n_skipped_no_vector = 0
            for reserve in reserve_per_query:
                for pid in reserve:
                    if pid in seen_reserve:
                        continue
                    seen_reserve.add(pid)
                    if pid in parent_centroids:
                        reserve_pool.append({
                            "pid": pid,
                            "score": parent_best_score.get(pid, 0.0),
                            "vector": parent_centroids[pid],
                        })
                    else:
                        n_skipped_no_vector += 1
                        # Fallback: se manca il vettore includiamo comunque il pid con score,
                        # così anche con lam<1 non lo perdiamo (verrà solo trattato come non
                        # comparabile: nessuna penalità di similarità applicabile su di lui).
                        reserve_pool.append({
                            "pid": pid,
                            "score": parent_best_score.get(pid, 0.0),
                            "vector": None,
                        })

            # Il WARNING ha senso solo se i vettori li AVEVAMO chiesti e non sono
            # arrivati (anomalia vera). Con MMR in bypass non li abbiamo chiesti
            # affatto: segnalarlo come mancanza produrrebbe un allarme su ogni
            # singolo candidato di ogni richiesta del caso normale, che è il modo
            # più rapido per insegnare a chi legge i log a ignorare i WARNING.
            if n_skipped_no_vector and needs_vectors:
                log.warning(
                    f"[{task_id}] {MMR_LOG_TAG} {n_skipped_no_vector} parent candidati alla "
                    f"redistribuzione senza vettore disponibile (inclusi comunque, senza "
                    f"possibilità di essere penalizzati per similarità)."
                )

            already_selected_vectors = {
                pid: parent_centroids[pid] for pid in already_selected if pid in parent_centroids
            }

            picked_pids, mmr_debug = select_with_mmr(
                reserve_pool,
                already_selected_vectors,
                lam=mmr_lambda,
                sim_threshold=settings.mmr_similarity_threshold,
                k=slots_free,
                task_id=task_id,
            )

            redistributed = 0
            for pid in picked_pids:
                q_idx = pid_owner_query.get(pid)
                if q_idx is None:
                    log.warning(f"[{task_id}] {MMR_LOG_TAG} pid={pid} scelto ma senza sottoquery proprietaria: skip.")
                    continue
                quota_per_query[q_idx].append(pid)
                already_selected.add(pid)
                redistributed += 1

            log.info(
                f"[{task_id}] Redistribuzione completata: {redistributed} parent aggiunti "
                f"(mmr_lambda={mmr_lambda}, candidati valutati={len(reserve_pool)})."
            )
            if mmr_debug:
                log.debug(f"[{task_id}] {MMR_LOG_TAG} Dettaglio scelte: {mmr_debug}")

            n_mmr_candidates = len(reserve_pool)
            n_mmr_penalized = sum(1 for d in mmr_debug if d["penalized"])
 
        # ==========================================================
        # 5ter. POSTI EXTRA PER PERTINENZA
        #
        # La quota (PARENTS_PER_QUERY × sottoquery) dipende da quante
        # sottoquery ha prodotto il rewriter, non da quanti documenti
        # pertinenti esistono: con una sola sottoquery si prendevano 4
        # parent anche se il reranker ne giudicava pertinenti 8, e una
        # domanda "quali atti..." riceveva un elenco incompleto.
        # Si aggiungono quindi i parent non ancora scelti con probabilità
        # del reranker >= EXTRA_PARENTS_MIN_PROB, dal più pertinente, fino
        # al tetto MAX_SUB_QUERIES × PARENTS_PER_QUERY: lo stesso numero di
        # documenti a cui può arrivare la ricerca più ampia. Ogni extra va
        # alla sottoquery che lo ha giudicato meglio (quote di contesto).
        # ==========================================================
        n_extra_parents = 0
        max_total_parents = settings.max_sub_queries * settings.parents_per_query
        if settings.extra_parents and len(already_selected) < max_total_parents:
            extra_candidates = sorted(
                ((prob, pid, q) for pid, (prob, q) in parent_rerank_best.items()
                 if pid not in already_selected and prob >= settings.extra_parents_min_prob),
                reverse=True,
            )
            room = max_total_parents - len(already_selected)
            for prob, pid, q in extra_candidates[:room]:
                quota_per_query[q].append(pid)
                already_selected.add(pid)
                n_extra_parents += 1
            if n_extra_parents or extra_candidates:
                log.info(
                    f"[{task_id}] [EXTRA] {n_extra_parents} parent aggiunti per pertinenza "
                    f"(probabilità >= {settings.extra_parents_min_prob}, candidati {len(extra_candidates)}, "
                    f"tetto {max_total_parents}): "
                    f"{[round(p, 3) for p, _, _ in extra_candidates[:room]]}"
                )

        # ==========================================================
        # 5bis. STATISTICHE DI RETRIEVAL (per la tabella rag_metrics)
        # ==========================================================
        retrieval_stats = {
            "semantic_size": semantic_size,
            "rerank_size": rerank_size,
            "semantic_threshold": round(semantic_threshold, 4),
            "mmr_lambda": mmr_lambda,
            "n_parent_candidates": len(parent_best_score),
            "n_mmr_candidates": n_mmr_candidates,
            "n_mmr_penalized": n_mmr_penalized,
            "n_parents_selected": len(already_selected),
            "n_extra_parents": n_extra_parents,
            # Tempi (secondi) per il riepilogo della richiesta: ricerca
            # vettoriale + testuale in parallelo, poi rerank di tutte le sotto-query.
            "t_search_s": round(t_search, 3),
            "t_rerank_s": round(t_rerank, 3),
            "n_candidates": sum(len(h) for h in hits_per_query),
        }

        # ==========================================================
        # 6. POOL FINALE — ordine ROUND-ROBIN tra sottoquery
        #
        # PRIMA: concatenazione a blocchi (tutti i parent della sottoquery 0,
        # poi quelli della 1, ...). Quando il budget del contesto si esauriva,
        # a cadere era sempre la coda — cioè l'ULTIMO comune per intero. E la
        # sottoquery 0 era privilegiata due volte, essendo la standalone_query
        # e ricevendo in appendice tutti i keyword_hits.
        #
        # ORA: rank 0 di ogni sottoquery, poi rank 1, ecc. Se qualcosa deve
        # cadere, cade la coda di TUTTE le sottoquery. Serve anche al modello,
        # che vede subito la diversità dei territori invece di una lunga
        # sequenza omogenea sul primo.
        # ==========================================================
        final_parent_ids_ordered: list[str] = []
        seen_final: set[str] = set()
        # Sottoquery "proprietaria" di ogni parent, per l'equità nell'allocatore.
        pid_to_query_idx: dict[str, int] = {}

        max_depth = max((len(q) for q in quota_per_query), default=0)
        for rank in range(max_depth):
            for q_idx, selected_pids in enumerate(quota_per_query):
                if rank >= len(selected_pids):
                    continue
                pid = selected_pids[rank]
                if pid not in seen_final:
                    final_parent_ids_ordered.append(pid)
                    seen_final.add(pid)
                    pid_to_query_idx[pid] = q_idx
 
        if not final_parent_ids_ordered:
            log.warning("Nessun parent_id estratto da nessuna sottoquery.")
            return [], [], retrieval_stats
 
        log.info(
            f"Pool finale: {len(final_parent_ids_ordered)} parent distinti "
            f"(budget={settings.parents_per_query}×{n_queries}={total_budget}"
            f"{f', +{n_extra_parents} per pertinenza' if n_extra_parents else ''})."
        )
 
 
        # ==========================================================
        # 7. RECUPERO PARENT DOCUMENTS (dal DB)
        # ==========================================================
        parent_records = []
        conn = get_db_connection()
        if not conn:
            return [], [], retrieval_stats
 
        try:
            log.debug(f"Parent ID da recuperare da DB: {final_parent_ids_ordered}")

            format_strings = ','.join(['%s'] * len(final_parent_ids_ordered))
            sql = f"""
                SELECT id, topic_id, sub_topic_id, source, file_name, content, metadata
                FROM parent_documents
                WHERE id IN ({format_strings})
            """
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, tuple(final_parent_ids_ordered))
                parent_records = cursor.fetchall()

            log.debug(f"Record restituiti dal DB: {len(parent_records)}")
            if not parent_records:
                log.error(
                    f"MySQL ha restituito 0 record per {len(final_parent_ids_ordered)} parent_id. "
                    f"Primo ID cercato: {final_parent_ids_ordered[0] if final_parent_ids_ordered else 'N/A'}. "
                    f"Verificare tipo colonna id e sincronizzazione Qdrant↔MySQL."
                )
        finally:
            conn.close()
 
        # Riordina i parent nell'ordine in cui sono stati aggiunti
        # (rispetta la priorità per sottoquery)
        pid_to_record = {r['id']: r for r in parent_records}
        parent_records = [pid_to_record[pid] for pid in final_parent_ids_ordered if pid in pid_to_record]
 
 
        # ==========================================================
        # 8. FORMATTAZIONE OUTPUT
        # ==========================================================
        rich_context = []
        unique_sources_map = {}
 
        for p_doc in parent_records:
            source = p_doc.get("source", "Fonte_Sconosciuta")
            file_name = p_doc.get("file_name", "File_Sconosciuto")
            sub_topic_id = p_doc.get("sub_topic_id", "")
            # Rientri e allineamenti dell'OCR (&nbsp; a decine) sono token
            # sprecati per l'LLM: si tolgono qui, prima di misurare e ritagliare
            # i passaggi, così lo spazio recuperato va ad altri documenti.
            content = clean_ocr_text(p_doc.get("content", "")).strip()

            # Data del documento (per "Date:") e metadati da mostrare al
            # generatore, dal JSON dei metadati salvato in MySQL.
            doc_date, doc_metadata = document_context_metadata(p_doc.get("metadata"))

            if content:
                pid = p_doc.get("id")
                # Pezzi ripuliti come il parent, così si ritrovano nel testo
                # pulito; gli offset si riferiscono al testo originale e non
                # valgono più (resta la ricerca testuale, tollerante agli spazi).
                matched_snippets = [
                    {**{k: v for k, v in s.items() if k not in ("start_char", "end_char")},
                     "content": clean_ocr_text(s["content"])}
                    for s in parent_top_snippets.get(pid, []) if s.get("content")
                ]
                q_idx = pid_to_query_idx.get(pid, 0)
                # Testi delle sottoquery che hanno recuperato questo parent:
                # è PROVENIENZA della ricerca, non attribuzione del contenuto.
                retrieved_by = [
                    search_queries[i]
                    for i in sorted(parent_retrieved_by.get(pid, {q_idx}))
                    if i < len(search_queries)
                ]
                rich_context.append({
                    "parent_id": pid,
                    "content": content,
                    "source": source,
                    "date": doc_date,
                    "metadata": doc_metadata,
                    "file_name": file_name,
                    "sub_topic": sub_topic_id,
                    "matched_snippets": matched_snippets,
                    # Chi ha "vinto" questo parent in quota: è l'asse su cui
                    # l'allocatore garantisce l'equità del contesto.
                    "query_idx": q_idx,
                    "query_text": search_queries[q_idx] if q_idx < len(search_queries) else "",
                    "retrieved_by": retrieved_by,
                    "score": parent_best_score.get(pid, 0.0),
                })
                if source not in unique_sources_map:
                    unique_sources_map[source] = {
                        "source": source, "sub_topic": sub_topic_id, "file_name": file_name,
                        "date": doc_date, "parent_id": p_doc.get("id"),
                    }
 
        log.info(f"=== Retrieval completata. Parent al generatore: {len(rich_context)} ===")
        retrieval_stats["n_parents_final"] = len(rich_context)
        return rich_context, list(unique_sources_map.values()), retrieval_stats
 
    except Exception as e:
        log.critical(f"Errore critico nel Retrieval: {e}", exc_info=True)
        return [], [], {}


def get_reranker_pool():
    """
    Pool di istanze ONNXReranker indipendenti, una per thread concorrente
    (settings.reranker_pool_size). Ogni sotto-query pesca la propria istanza
    via round-robin su q_idx (vedi process_single_rerank in retrieve_chunks),
    invece di contendersi un'unica sessione condivisa tra tutti i thread.

    num_threads per istanza è calcolato dividendo i core disponibili per la
    dimensione del pool. I core disponibili vengono letti da
    settings.cpu_limit (il resources.limits.cpu del pod, es. "2"), NON da
    os.cpu_count(): quest'ultimo, in un container Kubernetes, riflette i
    core del nodo host — i CPU limit di Kubernetes si applicano via CFS
    quota/period, non modificano l'affinity mask che os.cpu_count() legge.
    Su un pod limitato a 2 core ma su un nodo da 8, os.cpu_count() avrebbe
    fatto sovrastimare di 4 volte i thread assegnabili per istanza.
    Se settings.cpu_limit non è impostato esplicitamente (CPU_LIMIT), il
    fallback su os.cpu_count() avviene già in config.py — qui arriva sempre
    un intero valido. Da preferire comunque sempre CPU_LIMIT esplicito in
    un pod Kubernetes, dove os.cpu_count() riflette i core del nodo host.

    I core disponibili vengono inoltre divisi per settings.celery_concurrency
    (deve combaciare col --concurrency del comando Celery nel deployment):
    i core del pod si dividono tra TUTTI i processi Celery concorrenti, non
    solo tra le istanze del pool all'interno di un singolo processo. Senza
    questo, ogni processo dimensionerebbe il proprio pool assumendo di avere
    tutti i core del pod per sé, sovrasottoscrivendo quando più processi
    fanno reranking nello stesso momento.
    """
    global _RERANKER_POOL
    if not _RERANKER_POOL and settings.rerank_provider == "vllm":
        if not settings.rerank_model_name:
            raise ValueError("RERANK_PROVIDER='vllm' richiede RERANK_MODEL_NAME.")
        _RERANKER_POOL.append(VLLMReranker(
            base_url=settings.local_vllm_rerank_base_url or settings.local_vllm_base_url,
            model=settings.rerank_model_name,
            api_key=settings.local_vllm_api_key,
            max_length=settings.rerank_max_length,
            timeout=settings.rerank_timeout_seconds,
        ))
    if not _RERANKER_POOL:
        total_cores = settings.cpu_limit
        cores_per_process = max(1, total_cores // max(1, settings.celery_concurrency))
        num_threads = max(1, cores_per_process // max(1, _POOL_SIZE))
        log.info(
            f"Core totali pod: {total_cores}, concorrenza Celery: {settings.celery_concurrency} "
            f"-> {cores_per_process} core per processo, {num_threads} thread per istanza del pool."
        )
        for i in range(_POOL_SIZE):
            log.info(f"Initializing ONNX Reranker pool instance {i+1}/{_POOL_SIZE} (num_threads={num_threads})...")
            instance = ONNXReranker(
                model_folder=settings.onnx_model_cache_path,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
                num_threads=num_threads
            )
            _RERANKER_POOL.append(instance)
    return _RERANKER_POOL


def get_all_sub_topics(topic_id):
    """Recupera tutti i sub_topic_id associati a un topic dal database."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT sub_topic_id FROM sub_topics WHERE topic_id = %s", (topic_id,))
            rows = cursor.fetchall()
            return [row['sub_topic_id'] for row in rows]
    except Exception as e:
        log.error(f"Errore recupero tutti i sub-topics per '{topic_id}': {e}")
        return []
    finally:
        conn.close()
