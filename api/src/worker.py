"""
Worker Celery della chat: configurazione di Celery, inizializzazione dei
client dopo il fork, task `rag_queue` (process_rag_query) e manutenzione
della cache semantica.

Il motore vero e proprio sta nel pacchetto `rag/`:
    concepts    dizionario dei concetti (per nome e per similarità)
    rewriter    riscrittura della domanda e scomposizione in sotto-query
    retrieval   ricerca vettoriale + testuale, rerank, selezione dei parent
    keywords    ricerca testuale a combinazioni di parole chiave
    boilerplate frasi ripetute da togliere prima del rerank
    mmr         diversificazione dei parent negli slot liberi
    context     composizione del contesto (core, budget, finestre, metadati)
    generation  generazione della risposta, citazioni, Grader
    cache       cache semantica
    metrics     rag_metrics e auto-log dei fallimenti
    prompts     template dei prompt
    jsonparse   parsing robusto del JSON degli LLM
    state       client Qdrant condiviso (creato qui, dopo il fork)

Celery si avvia con `-A worker.celery_app`; l'API invia i task per nome.
"""

import os
import logging
import time
from celery import Celery
from celery.signals import setup_logging
from celery.signals import worker_process_init, worker_shutdown
from celery.schedules import crontab
from qdrant_client import QdrantClient, models

# Provider-agnostic LLM adapter
from llm_provider import init_llm_provider

# Embedding (Gemini only) — estratto in embedding.py
from common.embedding import (
    init_embedding,
    embed_query,
    embed_queries_batch,
    embed_for_semantic_query,
)

from common.config import settings
from common.db_logger import init_db_pool
from common.utility import quiet_http_loggers

from rag import state
from rag.state import CACHE_COLLECTION
from rag.cache import (check_semantic_cache, generate_filters_key, generate_sub_topics_key,
                       save_to_semantic_cache)
from rag.concepts import collect_concept_aliases, find_concepts, get_concepts_by_name
from rag.generation import evaluate_generation, generate_answer
from rag.metrics import log_automatic_negative_feedback, log_rag_metrics
from rag.retrieval import get_all_sub_topics, get_reranker_pool, profile_kwargs, retrieve_chunks
from rag.rewriter import transform_query
from rag.timing import PhaseTimer

# ==============================================================================
# 1. CONFIGURATION & LOGGING
# ==============================================================================

@setup_logging.connect
def configure_worker_logging(*args, **kwargs):
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format='%(asctime)s - [WORKER-%(process)d] - %(levelname)s - %(message)s',
        force=True  # Svuota e sovrascrive gli handler precedentemente configurati da Celery
    )
    quiet_http_loggers()

log = logging.getLogger("WORKER")


MAX_AGE_SECONDS = 86400  # 24 hours
_RERANKER_POOL: list = []

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_MODE"] = "PARALLEL"


# ==============================================================================
# 2. CELERY INITIALIZATION
# ==============================================================================
redis_conn_string = f"redis://{settings.redis_host}:{settings.redis_port}/0"
celery_app = Celery('rag_queue', broker=redis_conn_string, backend=redis_conn_string)
celery_app.conf.update(
    result_expires=3600,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,   
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_max_memory_per_child=2400000, # 2GB limit
    worker_hijack_root_logger=False,
    # Secondi che Celery concede a un processo appena forkato per segnalare
    # "UP". Il default è 4.0 e NON basta più: init_worker_process carica il
    # pool ONNX (tokenizer + sessione per ciascuna istanza, ~6.9s misurati),
    # quindi ogni processo di rimpiazzo veniva ucciso con SIGKILL a 3.97s
    # esatti e il worker entrava in crash loop infinito — visibile nei log
    # come "Timed out waiting for UP message from <ForkProcess(...)>".
    # Va tenuto sopra il tempo di init peggiore, con margine: è un limite di
    # sicurezza contro i processi nati morti, non una manopola di performance.
    worker_proc_alive_timeout=60.0,
)

# ==============================================================================
# PROCESS-SAFE INITIALIZATION 
# ==============================================================================
# Globals assigned strictly AFTER the fork
db_pool = None

@worker_process_init.connect
def init_worker_process(**kwargs):
    log.info("Initializing Worker Resources (Post-Fork)...")

    try:
        init_db_pool()  # Inizializza il pool globalmente

        state.qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            # Il default della libreria è 5s. Le vector search con group_by
            # impiegavano 4,1-4,9s: si viaggiava sul filo, e la prima che
            # sforava perdeva l'INTERA sottoquery (safe_vector_search cattura
            # l'eccezione e restituisce una lista vuota, senza retry), con il
            # comune corrispondente assente dalla risposta finale.
            timeout=10,
        )
        init_embedding()
        init_llm_provider()

        # Il pool ONNX viene costruito QUI, non alla prima query. Era lazy
        # dentro retrieve_chunks: i ~6,7s di caricamento (tokenizer + sessione,
        # per ogni istanza, in serie) finivano dentro la latenza percepita
        # dall'utente che aveva la sfortuna di essere il primo a interrogare il
        # processo appena forkato. Post-fork è il punto giusto: prima del fork
        # la sessione ONNX non sarebbe ereditabile in sicurezza.
        get_reranker_pool()

        log.info("✓ Resources successfully initialized for this process.")
    except Exception as e:
        log.critical(f"✗ Failed to initialize worker resources: {e}")
        raise


# ==============================================================================
# 3. MANUTENZIONE DELLA CACHE SEMANTICA
# ==============================================================================


@celery_app.task(name="prune_semantic_cache")
def prune_semantic_cache():
    """Delete cache entries older than MAX_AGE_SECONDS."""
    if not state.qdrant_client:
        log.warning("Qdrant client not available for cache pruning.")
        return
    
    try:
        cutoff_time = time.time() - MAX_AGE_SECONDS
        
        state.qdrant_client.delete(
            collection_name=CACHE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(lt=cutoff_time)
                        )
                    ]
                )
            )
        )
        
        log.info(f"✓ Cache pruned (entries older than {MAX_AGE_SECONDS}s removed).")
        
    except Exception as e:
        log.error(f"Cache pruning failed: {e}")


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup scheduled tasks."""
    sender.add_periodic_task(
        crontab(minute=0),
        prune_semantic_cache.s(),
        name='hourly-cache-cleanup'
    )
    log.info("✓ Scheduled task: Cache cleanup every hour")


# ==============================================================================
# 4. MAIN CELERY TASK
# ==============================================================================

@celery_app.task(bind=True, name="rag_queue")
def process_rag_query(self, query, history, topic_id, selected_sub_topics=None, metadata_filters=None,
                      include_undated=True, deep_search=False):
    """
    Task della chat. Il lavoro è in _run_rag_query; qui si misurano le fasi e
    si scrive, comunque vada (anche su errore o eccezione), una sola riga di
    riepilogo nel log:
        [SUMMARY] esito=ok tentativi=1 parent=24 totale=31.0s | concetti 0.5s · riscrittura 1.1s · …
    """
    phases = PhaseTimer()
    result = None
    try:
        result = _run_rag_query(self.request.id, query, history, topic_id, selected_sub_topics,
                                metadata_filters, include_undated, deep_search, phases)
        return result
    finally:
        if result is None:
            outcome = "eccezione"
        elif result.get("status") == "failed":
            outcome = f"errore ({result.get('error', '?')})"
        elif result.get("cached"):
            outcome = "cache"
        else:
            outcome = phases.info.get("esito", "ok")
        log.info(f"[{self.request.id}] [SUMMARY] {phases.summary(outcome)}")


def _run_rag_query(task_id, query, history, topic_id, selected_sub_topics, metadata_filters,
                   include_undated, deep_search, phases):
    """Main RAG processing pipeline: JSON Mode, Multi-Query, and Defensive Error Handling."""
    start_time = time.time()
    log.info(f"[{task_id}] Task started: '{query[:50]}...' su topic: {topic_id}")

    # Ricerca approfondita (opzione della chat): più passaggi per sotto-domanda.
    # max(): se RERANK_SIZE fosse configurata più alta, l'opzione non deve ridurla.
    rerank_size = (max(settings.rerank_size, settings.rerank_max_size)
                   if deep_search else settings.rerank_size)
    if deep_search:
        log.info(f"[{task_id}] Ricerca approfondita: rerank_size={rerank_size} "
                 f"(normale: {settings.rerank_size})")

    # Accumula, per ciascun tentativo di retrieval, i parametri usati e le stats
    # ritornate da retrieve_chunks. Alla fine viene salvato in rag_metrics
    # (vedi log_rag_metrics) come JSON, per analisi a posteriori sui profili.
    attempts_metrics: list[dict] = []

    # ==========================================================
    # SETUP & VALIDATION
    # ==========================================================
    try:
        if not selected_sub_topics:
            selected_sub_topics = get_all_sub_topics(topic_id)
            if not selected_sub_topics:
                log.error(f"[{task_id}] Config Error: Nessun sub-topic per il topic '{topic_id}'.")
                return {
                    "error": "Configuration Error",
                    "message": "Il topic selezionato non è configurato correttamente.",
                    "status": "failed"
                }
        st_key = generate_sub_topics_key(selected_sub_topics)

    except Exception as e:
        log.error(f"[{task_id}] Initialization DB failed: {e}", exc_info=True)
        return {"error": "Internal Error", "message": "Errore di inizializzazione.", "status": "failed"}

    # ==========================================================
    # QUERY TRANSFORMATION & MULTI-QUERY EXPANSION
    # (concept-first: il dizionario Qdrant viene interrogato PRIMA del
    # rewriter, e i concetti trovati vengono passati come vocabolario
    # autoritativo dentro l'unica chiamata LLM che decide tutto)
    # ==========================================================
    # Concetti dell'ultimo giro di riformulazione: servono anche alla ricerca
    # testuale (alias in alternativa, vedi build_keyword_groups).
    active_concepts = []
    question_dates = None
    try:
        # 1. Lookup concettuale sul testo grezzo (ultimi turni + query corrente):
        #    nomi di zone e concetti tecnici sono quasi sempre espliciti, quindi
        #    non serve la risoluzione dei pronomi per trovarli. Nota: euristica —
        #    su follow-up molto impliciti ("e lì invece?") il lookup può mancare
        #    il concetto; in quel caso il rewriter ripiega sulle sue regole
        #    generali, come da prompt.
        lookup_text = query
        if history:
            recent_turns = [msg.get("text", "") for msg in history[-2:]]
            lookup_text = " ".join(recent_turns + [query])

        verified_concepts = []
        concept_vector = None 
        try:
            # Dizionario concettuale: usa lookup_text (query + ultimi turni),
            # perché il contesto conversazionale aiuta ad agganciare i concetti
            # su domande di follow-up brevi ("e quelle del 2024?").
            concept_vector = phases.run("concetti", embed_for_semantic_query, lookup_text)
            verified_concepts = phases.run("concetti", find_concepts, query, concept_vector, task_id=task_id)
        except Exception as ce:
            log.warning(f"[{task_id}] Lookup concettuale fallito ({ce}): il rewriter procede senza concetti verificati.")
            # L'embedding può fallire (rete, quota): i concetti citati per nome restano validi
            verified_concepts = get_concepts_by_name(query, task_id=task_id)

        # 2. Unica chiamata: rewrite + decomposizione + espansione insieme,
        #    con i concetti verificati come dato di fatto nel prompt.
        rewritten_data = phases.run("riscrittura", transform_query, history, query, task_id,
                                    verified_concepts=verified_concepts)
        active_concepts = verified_concepts
        question_dates = rewritten_data.get("date_range")

        standalone_query = rewritten_data.get("standalone_query", query)
        primary_vector = phases.run("embedding", embed_query, standalone_query)

        extracted_keywords = rewritten_data.get("keywords", [])
        # Fallback testuale di sicurezza
        if not extracted_keywords:
            log.debug(f"[{task_id}] Nessuna keyword restituita, attivo fallback testuale in Python.")
            extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]

        search_queries = rewritten_data.get("search_queries", [standalone_query])
        coverage_targets = rewritten_data.get("coverage_targets", [])
        db_aliases = collect_concept_aliases(verified_concepts)

        if standalone_query not in search_queries:
                    search_queries.insert(0, standalone_query)
        vectors_list = phases.run("embedding", embed_queries_batch, search_queries)
        all_keywords = list(set(extracted_keywords + db_aliases))
            
    except Exception as e:
        log.warning(f"[{task_id}] [MAIN_TASK] Pipeline di trasformazione caduta: {e}. Uso raw query.")
        standalone_query, search_queries, all_keywords = query, [query], []
        coverage_targets = []
        lookup_text = query  # garantito anche nel fallback: usato dal lookup concettuale in retry
        primary_vector = phases.run("embedding", embed_query, query)  # fallback sul raw query
        vectors_list = [primary_vector]

    # ==========================================================
    # PERIODO CHIESTO NELLA DOMANDA ("atti fatti nel 2025")
    #
    # Il rewriter lo restituisce in date_range; si applica come filtro sulla
    # data del documento, lo stesso dei Filtri della chat, includendo sempre
    # gli atti senza data (un atto non datato non va escluso per un'ipotesi).
    # Se l'utente ha già impostato un periodo nei Filtri vale il suo. Il
    # periodo resta quello della domanda originale anche nei tentativi
    # successivi; entra nella chiave della cache semantica.
    # ==========================================================
    if question_dates:
        period = f"{question_dates.get('gte', '…')}→{question_dates.get('lte', '…')}"
        if (metadata_filters or {}).get("data"):
            log.info(f"[{task_id}] [DATE] Periodo nella domanda {period} non applicato: "
                     f"vale quello impostato nei Filtri ({metadata_filters['data']})")
        else:
            metadata_filters = {**(metadata_filters or {}), "data": question_dates}
            include_undated = True
            phases.info["periodo"] = period
            log.info(f"[{task_id}] [DATE] Periodo dalla domanda applicato come filtro: {period} "
                     f"(atti senza data inclusi)")

    # ==========================================================
    # SEMANTIC CACHE CHECK
    # ==========================================================
    try:
        # La ricerca approfondita ha la sua voce in cache (una risposta normale
        # non deve servirla). Solo se attiva: le chiavi delle ricerche normali
        # restano quelle di prima e la cache esistente resta valida.
        filters_key = generate_filters_key({
            **(metadata_filters or {}),
            "_include_undated": include_undated,
            **({"_deep_search": True} if deep_search else {}),
        })

        cached = phases.run("cache", check_semantic_cache, primary_vector, topic_id, st_key, filters_key)
        if cached:
            log.info(f"[{task_id}] Cache HIT. Returning cached response.")
            log_rag_metrics(
                task_id=task_id, topic_id=topic_id, query=query,
                cache_hit=True, is_satisfactory=True,
                total_attempts=0, duration_ms=int((time.time() - start_time) * 1000),
                attempts_metrics=[]
            )
            return {
                "answer": cached['answer'],
                "sources": cached['sources'],
                "topic": cached['topic_id'],
                "cached": True,
                "status": "success"
            }
    except Exception as e:
        log.warning(f"[{task_id}] Cache check failed (non-blocking): {e}")

    # ==========================================================
    # RETRIEVAL (Parallel Vector + Keyword)
    # ==========================================================
    try:
        context, sources, retrieval_stats = phases.run("ricerca", retrieve_chunks,
            search_queries, 
            vectors_list, 
            all_keywords, 
            topic_id, 
            selected_sub_topics,
            metadata_filters,
            include_undated,
            query_semantic_vector=concept_vector,
            rerank_size=rerank_size,
            concepts=active_concepts,
            task_id=task_id,
            **profile_kwargs(0, task_id=task_id)
        )
        attempts_metrics.append({"attempt": 0, "standalone_query": standalone_query, **retrieval_stats})
        phases.note_retrieval(retrieval_stats)

        # Parent già MOSTRATI al generatore, accumulati tra i tentativi: usato
        # dall'ultimo fallback (dopo max_model_retries) per non riproporre
        # parent già passati e già giudicati insoddisfacenti.
        #
        # PRIMA veniva derivato da `sources`, che è deduplicata per `source` e
        # conserva un solo parent_id per documento: se un atto contribuiva con
        # 3 parent ne veniva registrato 1, e gli altri 2 rientravano indisturbati
        # nel contesto del fallback, vanificandone in buona parte lo scopo.
        # ORA viene popolato da generate_answer con i parent REALMENTE finiti nel
        # prompt: un parent recuperato ma mai mostrato non è stato giudicato da
        # nessuno e resta legittimamente candidabile.
        used_parent_ids = set()
    except Exception as e:
        log.error(f"[{task_id}] Vector DB Retrieval failed: {e}", exc_info=True)
        return {"error": "Database Error", "message": "Errore durante il recupero dei documenti.", "status": "failed"}

    # ==========================================================
    # SELF-CORRECTION LOOP (Generation & Grading)
    # ==========================================================
    attempt = 0
    answer = None
    is_satisfactory = False
    # Inizializzati qui: restano definiti anche se il loop esce subito per
    # contesto vuoto, e vengono letti dopo il loop per applicare la policy.
    fail_reason = "ok"
    missing_targets = []

    while attempt < settings.max_model_retries and not is_satisfactory:
        if not context:
            answer = "<p>Non sono riuscito a trovare la risposta nei documenti che ho analizzato.</p>"
            log.debug(f"[{task_id}] Context empty. Breaking loop.")
            break
        else:
            log.debug(f"Contesto da passare alla generazione: {str(context)[:1500]}")

        # A. Generazione (JSON Mode)
        try:
            gen_result = phases.run("generazione", generate_answer,
                standalone_query, context, topic_id,
                search_queries=search_queries,
                coverage_targets=coverage_targets,
                task_id=task_id,
            )
            is_found = gen_result.get("is_found", False)
            answer = gen_result.get("answer", "")

            # Le fonti mostrate all'utente sono SOLO quelle realmente entrate nel
            # prompt: prima si elencavano tutti i parent recuperati, comprese le
            # fonti scartate per esaurimento budget e mai lette dal modello.
            sources = gen_result.get("used_sources", sources)
            used_parent_ids |= gen_result.get("used_parent_ids", set())

            log.debug(f"Risposta generata: {answer[:500]}... | is_found: {is_found}")
        except Exception as e:
            log.error(f"[{task_id}] Answer generation API failed: {e}", exc_info=True)
            return {"error": "Generation Failed", "message": "Impossibile elaborare la risposta.", "status": "failed"}

        # B. Valutazione unificata (fail-fast + grader + policy)
        is_satisfactory, fail_reason, missing_targets = phases.run("grader", evaluate_generation,
            gen_result, standalone_query, coverage_targets=coverage_targets, task_id=task_id
        )
        if attempts_metrics:
            attempts_metrics[-1]["grader_reason"] = fail_reason
            attempts_metrics[-1]["missing_targets"] = missing_targets

        # D. Gestione Retry Fallimento
        if not is_satisfactory:
            attempt += 1
            if attempt < settings.max_model_retries:
                log.warning(f"[{task_id}] Answer unsatisfactory. Retrying ({attempt}/{settings.max_model_retries}) with full pipeline...")
                try:
                    # 1. Iniettiamo un "falso" messaggio di sistema nella history per forzare l'LLM a cambiare approccio.
                    #    Il messaggio dipende dal MOTIVO del fallimento: dire al
                    #    rewriter che "la ricerca non ha prodotto documenti validi"
                    #    quando invece i documenti c'erano ma erano insufficienti su
                    #    un aspetto lo spinge ad allontanarsi dalla formulazione
                    #    originale, mentre servirebbe restare sul punto e scavare.
                    if fail_reason == "general_knowledge":
                        retry_instruction = (
                            f"La risposta precedente per '{standalone_query}' ha dovuto ricorrere a conoscenza "
                            f"esterna perché i documenti recuperati non coprivano tutti gli aspetti richiesti. "
                            f"Mantieni la stessa intenzione e lo stesso ambito, ma cerca documenti più specifici "
                            f"sugli aspetti rimasti scoperti."
                        )
                    elif fail_reason == "grounding":
                        retry_instruction = (
                            f"La risposta precedente per '{standalone_query}' conteneva affermazioni non "
                            f"sostenute dai documenti recuperati. Mantieni l'intenzione originale ma cerca "
                            f"documenti che trattino l'argomento in modo più diretto e verificabile."
                        )
                    else:
                        retry_instruction = (
                            f"La ricerca precedente per '{standalone_query}' non ha prodotto documenti validi. "
                            f"Riformula completamente la query usando sinonimi o concetti più ampi per esplorare "
                            f"un'angolazione semantica diversa."
                        )

                    retry_history = history.copy() if history else []
                    retry_history.append({
                        "role": "user",
                        "text": retry_instruction
                    })
                    
                    # 2. Lookup concettuale anche in retry, ma sul lookup_text
                    #    GREZZO del primo giro (history + query originale
                    #    dell'utente), NON sulla standalone_query del tentativo
                    #    fallito: se il fallimento era dovuto proprio a una
                    #    riformulazione distorta, ripartire da quella
                    #    propagherebbe l'errore anche nel lookup. Il testo
                    #    originale dell'utente è immune da errori di rewrite.
                    retry_concepts = []
                    retry_concept_vector = None
                    try:
                        retry_concept_vector = phases.run("concetti", embed_for_semantic_query, lookup_text)
                        retry_concepts = phases.run("concetti", find_concepts, query, retry_concept_vector,
                                                    task_id=f"{task_id}-RETRY")
                    except Exception as ce:
                        log.warning(f"[{task_id}-RETRY] Lookup concettuale fallito ({ce}): retry senza concetti verificati.")
                        retry_concepts = get_concepts_by_name(query, task_id=f"{task_id}-RETRY")

                    retry_data = phases.run("riscrittura", transform_query, retry_history, query, f"{task_id}-RETRY",
                                            verified_concepts=retry_concepts)
                    active_concepts = retry_concepts
                    
                    standalone_query = retry_data.get("standalone_query", query)
                    extracted_keywords = retry_data.get("keywords", [])
                    
                    if not extracted_keywords:
                        extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]

                    search_queries = retry_data.get("search_queries", [standalone_query])
                    coverage_targets = retry_data.get("coverage_targets", coverage_targets)
                    extracted_keywords = list(set(extracted_keywords + collect_concept_aliases(retry_concepts)))

                    if standalone_query not in search_queries:
                        search_queries.insert(0, standalone_query)
                    
                    # 3. Rieseguiamo il Batch Embedding e il Retrieval Multi-Query
                    vectors_list = phases.run("embedding", embed_queries_batch, search_queries)
                    context, sources, retrieval_stats = phases.run("ricerca", retrieve_chunks,
                        search_queries, 
                        vectors_list, 
                        extracted_keywords, 
                        topic_id, 
                        selected_sub_topics,
                        metadata_filters,
                        include_undated,
                        query_semantic_vector=retry_concept_vector,
                        rerank_size=rerank_size,
                        concepts=active_concepts,
                        task_id=task_id,
                        **profile_kwargs(attempt, task_id=task_id)
                    )
                    attempts_metrics.append({"attempt": attempt, "standalone_query": standalone_query, **retrieval_stats})
                    phases.note_retrieval(retrieval_stats)
                    # used_parent_ids NON viene più aggiornato qui: lo popola
                    # generate_answer col contenuto effettivo del prompt.
                except Exception as e:
                    log.error(f"[{task_id}] Retry infrastructure failed: {e}", exc_info=True)
                    break  # Usciamo usando l'ultima answer generata
            else:
                log.warning(f"[{task_id}] Max retries esauriti. Tento un ultimo fallback senza riformulazione, escludendo {len(used_parent_ids)} parent già usati.")
                try:
                    final_context, final_sources, final_stats = phases.run("ricerca", retrieve_chunks,
                        search_queries,
                        vectors_list,
                        extracted_keywords,
                        topic_id,
                        selected_sub_topics,
                        metadata_filters,
                        include_undated,
                        exclude_parent_ids=used_parent_ids,
                        query_semantic_vector=concept_vector,
                        rerank_size=rerank_size,
                        concepts=active_concepts,
                        task_id=f"{task_id}-FALLBACK",
                        **profile_kwargs(settings.max_model_retries - 1, task_id=task_id)
                    )
                    attempts_metrics.append({"attempt": "fallback", "standalone_query": standalone_query, **final_stats})
                    phases.note_retrieval(final_stats)

                    if not final_context:
                        log.info(f"[{task_id}] [FALLBACK] Nessun parent nuovo trovato dopo l'esclusione. Nessun ulteriore tentativo possibile.")
                    else:
                        context, sources = final_context, final_sources
                        gen_result = phases.run("generazione", generate_answer,
                            standalone_query, context, topic_id,
                            search_queries=search_queries,
                            coverage_targets=coverage_targets,
                            task_id=f"{task_id}-FALLBACK",
                        )
                        is_found = gen_result.get("is_found", False)
                        answer = gen_result.get("answer", "")
                        sources = gen_result.get("used_sources", sources)
                        used_parent_ids |= gen_result.get("used_parent_ids", set())
                        log.debug(f"[{task_id}] [FALLBACK] Risposta generata: {answer[:500]}... | is_found: {is_found}")

                        is_satisfactory, fail_reason, missing_targets = phases.run("grader", evaluate_generation,
                            gen_result, standalone_query,
                            coverage_targets=coverage_targets,
                            task_id=f"{task_id}-FALLBACK",
                        )
                        if attempts_metrics:
                            attempts_metrics[-1]["grader_reason"] = fail_reason
                            attempts_metrics[-1]["missing_targets"] = missing_targets

                        log.info(f"[{task_id}] [FALLBACK] Esito ultimo tentativo: is_satisfactory={is_satisfactory} (reason={fail_reason}).")
                except Exception as fe:
                    log.error(f"[{task_id}] [FALLBACK] Ultimo tentativo fallito: {fe}", exc_info=True)

    # ==========================================================
    # APPLICAZIONE DELLA POLICY SULL'OUTPUT FINALE
    #
    # Con ALLOW_GENERAL_KNOWLEDGE=false, esaurire i tentativi lasciando in piedi
    # una risposta basata su conoscenza generale significherebbe applicare la
    # policy al giudizio ma non a ciò che l'utente legge: is_satisfactory=False
    # e, in mano al cittadino, esattamente la risposta che la policy vieta.
    # ==========================================================
    if not settings.allow_general_knowledge and fail_reason == "general_knowledge":
        log.warning(
            f"[{task_id}] Tentativi esauriti con risposta ancora basata su conoscenza generale: "
            f"sostituita con il messaggio di risposta non trovata (ALLOW_GENERAL_KNOWLEDGE=false)."
        )
        answer = "<p>Non sono riuscito a trovare la risposta nei documenti che ho analizzato.</p>"

    # ==========================================================
    # FEEDBACK NEGATIVO AUTOMATICO
    # ==========================================================
    if not is_satisfactory:
        log.info(f"[{task_id}] [MAIN_TASK] Rilevato fallimento retrieval/grading. Avvio auto-logging.")
        # Usiamo la query originale dell'utente passata al task
        log_automatic_negative_feedback(
            query=query, 
            answer=answer, 
            topic_id=topic_id, 
            history=history, 
            task_id=task_id
        )

    # ==========================================================
    # CACHE SAVING & RETURN
    # ==========================================================
    try:
        # Evitiamo di mettere in cache risposte troppo brevi o palesemente vuote
        if answer and is_satisfactory and len(answer) > 20:
            phases.run("cache", save_to_semantic_cache,
                primary_vector, 
                standalone_query, 
                answer, 
                sources, 
                topic_id,
                st_key,
                filters_key
            )
    except Exception as e:
        log.warning(f"[{task_id}] Failed to save successful response to cache (non-blocking): {e}")

    duration = time.time() - start_time
    log.info(f"[{task_id}] ✓ Completed gracefully in {duration:.2f}s")
    phases.info["tentativi"] = len(attempts_metrics)
    phases.info["esito"] = "ok" if is_satisfactory else f"insoddisfacente ({fail_reason})"

    log_rag_metrics(
        task_id=task_id, topic_id=topic_id, query=query,
        cache_hit=False, is_satisfactory=is_satisfactory,
        total_attempts=len(attempts_metrics), duration_ms=int(duration * 1000),
        attempts_metrics=attempts_metrics
    )

    return {
        "answer": answer,
        "sources": sources,
        "topic": topic_id,
        "cached": False,
        "status": "success"
    }
    
# ==============================================================================
# 5. WORKER LIFECYCLE
# ==============================================================================

@worker_shutdown.connect
def cleanup_worker(**kwargs):
    """Gracefully close connections when the worker shuts down."""
    if state.qdrant_client:
        state.qdrant_client.close()
        log.info("Qdrant client closed.")