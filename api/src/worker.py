import os
import logging
import time
import uuid
from celery import Celery
from celery.signals import setup_logging
from celery.signals import worker_process_init, worker_shutdown
from celery.schedules import crontab
from mysql.connector import pooling
from qdrant_client import QdrantClient, models
from google import genai  # embedding only
from google.genai.errors import APIError
import math
import json, re
import hashlib
# Use the optimized reranker
from reranker import ONNXReranker, RerankResult

# Provider-agnostic LLM adapter
from llm_provider import init_llm_provider, get_llm_provider

from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool




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

log = logging.getLogger("rag_queue")

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

MAX_AGE_SECONDS = 86400  # 24 hours
MAX_MODEL_RETRIES = 2
_RERANKER_POOL: list = []
_POOL_SIZE = settings.reranker_pool_size

# Embedding retry (Gemini only — embedding stays on Google)
GEMINI_EMBEDDING_RETRY = retry(
    retry=retry_if_exception_type((APIError,)),
    wait=wait_random_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: log.warning(
        f"Embedding rate limit hit. Retrying in {retry_state.next_action.sleep}s..."
    )
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_MODE"] = "PARALLEL"

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
CONCEPT_COLLECTION = "conceptual_dictionary"

# ==============================================================================
# 2. CELERY INITIALIZATION
# =========================================", "rag_system")=====================================
redis_conn_string = f"redis://{settings.redis_host}:{settings.redis_port}/0"
celery_app = Celery('rag_queue', broker=redis_conn_string, backend=redis_conn_string)
celery_app.conf.update(
    result_expires=3600,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,   
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_max_memory_per_child=2000000, # 2GB limit
    worker_hijack_root_logger=False,
)

# ==============================================================================
# PROCESS-SAFE INITIALIZATION 
# ==============================================================================
# Globals assigned strictly AFTER the fork
db_pool = None
qdrant_client = None
_RERANKER_INSTANCE = None

EMBEDDING_MODEL = "gemini-embedding-001"

@worker_process_init.connect
def init_worker_process(**kwargs):
    global qdrant_client, embedding_client
    log.info("Initializing Worker Resources (Post-Fork)...")

    try:
        init_db_pool()  # Inizializza il pool globalmente

        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        embedding_client = genai.Client(api_key=settings.api_llm_key)
        init_llm_provider()

        log.info("✓ Resources successfully initialized for this process.")
    except Exception as e:
        log.critical(f"✗ Failed to initialize worker resources: {e}")
        raise


# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

def generate_filters_key(metadata_filters):
    """Genera una chiave univoca basata sui filtri applicati per non inquinare la cache."""
    if not metadata_filters: return "no_filters"
    # Ordina le chiavi per garantire che lo stesso set generi sempre lo stesso hash
    filter_str = json.dumps(metadata_filters, sort_keys=True)
    return hashlib.md5(filter_str.encode()).hexdigest()

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
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 1 fallito: {e}. Passo al Livello 2 (Markdown Strip).")
        
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
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 2b fallito: {e}. Passo al Livello 3 (Brute Force Regex).")

    # 3. Tentativo "Forza Bruta": Cerca tutto ciò che è tra parentesi graffe
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            parsed_data = json.loads(match.group(0))
            log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 3 (Regex Regex Brute Force).")
            return parsed_data
        except json.JSONDecodeError as e:
            log.debug(f"[{task_id}] [JSON_PARSE] Livello 3 fallito: {e}.")
            
    # Fallimento totale
    log.error(f"[{task_id}] [JSON_PARSE] ✗ Fallimento totale. Impossibile estrarre JSON. Testo originale:\n{raw_text}")
    raise ValueError("Impossibile estrarre un JSON valido dal testo fornito.")

def safe_sigmoid(x):
    """
    Converte i logit in probabilità (0-1) prevenendo OverflowError in Python.
    """
    if x < -100:  # Qualsiasi logit sotto -100 è di fatto 0 probabilità
        return 0.0
    return 1 / (1 + math.exp(-x))

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


# ==============================================================================
# 5. LLM API FUNCTIONS
# ==============================================================================

@GEMINI_EMBEDDING_RETRY
def embed_query(query):
    """Generate embedding for a single query (Gemini only)."""
    log.debug(f"Embedding query: '{query[:50]}...'")
    result = embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=dict(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
    )
    return result.embeddings[0].values


@GEMINI_EMBEDDING_RETRY
def embed_queries_batch(queries_list):
    """Generate embeddings for multiple queries in a single API call (Gemini only)."""
    result = embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=queries_list,
        config=dict(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
    )
    if isinstance(queries_list, str):
        return [result.embeddings[0].values]
    return [emb.values for emb in result.embeddings]

@GEMINI_EMBEDDING_RETRY
def embed_for_concept_lookup(query):
    """Embedding per confronto con il dizionario concettuale (task simmetrico)."""
    result = embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=dict(task_type="SEMANTIC_SIMILARITY", output_dimensionality=768)
    )
    return result.embeddings[0].values

def expand_standalone_with_aliases(standalone_query, aliases, task_id="UNKNOWN"):
    aliases_str = ", ".join(aliases)
    
    prompt = load_prompt_template("expand_with_alias").format(
        aliases_str=aliases_str,
        standalone_query=standalone_query,
        max_sub_queries=settings.max_sub_queries
    )
    
    try:
        raw = get_llm_provider().generate_json(settings.query_rewriter_model_name, prompt, temperature=0.2, max_tokens=settings.answer_max_tokens)
        data = safe_json_parse(raw, task_id)
        return data.get("search_queries", [standalone_query])
    except Exception as e:
        log.error(f"[{task_id}] Errore nell'espansione alias: {e}. Fallback su query standalone.")
        return [standalone_query]

def transform_query(history, query, task_id="UNKNOWN"):
    """Transform conversational query into standalone query + search facets + keywords."""
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

    prompt = load_prompt_template("query_rewriter").format(
        history_str=history_str,
        query=query,
        max_sub_queries=settings.max_sub_queries
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

        log.info(
            f"[{task_id}] [REWRITER] ✓ Query processata. "
            f"Standalone: '{standalone}' | Facets: {len(searches)} | Keywords: {len(keywords)}"
        )

        log.debug(f"Standalone query: {standalone}")
        log.debug(f"Query individuate: {searches}")
        log.debug(f"Keywords: {keywords}")
        return {"standalone_query": standalone, "search_queries": searches, "keywords": keywords}

    except Exception as e:
        log.warning(
            f"[{task_id}] [REWRITER] ✗ Fallimento critico: {e}. Attivazione fallback (query raw).",
            exc_info=True
        )
        return {"standalone_query": query, "search_queries": [query], "keywords": []}


def generate_answer(query, rich_context, topic_id):
    formatted_chunks = []
    curr_len = 0
    n_parents = len(rich_context)

    if n_parents == 0:
        context_str = ""
    else:
        theoretical_budget_per_parent = settings.max_context_chars // (settings.parents_per_query * settings.max_sub_queries)
        cap_per_parent = int(theoretical_budget_per_parent / settings.parent_budget_ratio)
        budget_per_parent = min(settings.max_context_chars // n_parents, cap_per_parent)

        log.debug(
            f"Budget per parent: {budget_per_parent} chars "
            f"(teorico={theoretical_budget_per_parent}, cap={cap_per_parent}, "
            f"parent effettivi={n_parents})."
        )

        for item in rich_context:
            content = item['content']
            if len(content) > budget_per_parent:
                log.debug(f"Parent '{item['source']}' troncato: {len(content)} → {budget_per_parent} chars.")
                content = content[:budget_per_parent]

            chunk = f"[Source: {item['source']}]\n{content}\n\n"

            if curr_len + len(chunk) > settings.max_context_chars:
                log.warning(
                    f"Budget complessivo raggiunto ({curr_len}/{settings.max_context_chars} chars). "
                    f"{n_parents - len(formatted_chunks)} parent scartati."
                )
                break

            formatted_chunks.append(chunk)
            curr_len += len(chunk)

        context_str = "".join(formatted_chunks)

    log.info(
        f"Contesto finale: {len(formatted_chunks)}/{n_parents} parent, "
        f"{curr_len}/{settings.max_context_chars} chars utilizzati."
    )

    prompt_file  = get_topic_prompt(topic_id)
    prompt_tmpl  = load_prompt_template(prompt_file)
    prompt       = prompt_tmpl.format(context_str=context_str, query=query)

    log.info(f"Generating structured answer for topic '{topic_id}'")
    log.debug(f"Context size: {len(context_str)} chars")

    raw = get_llm_provider().generate_json(settings.answer_generator_model_name, prompt, max_tokens=settings.answer_max_tokens, thinking_level=settings.answer_thinking_level )
    log.debug(f"Raw response:\n{raw[:500]}...")

    try:
        result_data = json.loads(raw.strip())
        return {
            "is_found": bool(result_data.get("is_found", True)),
            "answer":   str(result_data.get("answer", ""))
        }
    except json.JSONDecodeError as e:
        log.error(f"Generazione JSON fallita: {e}. Output grezzo: {raw}")
        return {"is_found": True, "answer": raw.strip()}
    

def grade_answer(query, context_snippet, answer):
    """Ask the grader model whether the answer is satisfactory. Returns True/False."""
    grader_tmpl   = load_prompt_template("grader")
    grader_prompt = grader_tmpl.format(
        query=query,
        context_snippet=context_snippet,
        answer=answer
    )
    raw = get_llm_provider().generate_text(
        settings.grader_model_name, grader_prompt, temperature=0.0, max_tokens=5
    )
    result = raw.strip().upper()
    log.info(f"Grader response: '{result}'")
    return result.startswith("YES")


# ==============================================================================
# 6. SEMANTIC CACHE FUNCTIONS
# ==============================================================================

def check_semantic_cache(query_vector, topic_id, sub_topics_key, filters_key):
    """Check if similar query exists in cache."""
    if not qdrant_client:
        return None
    
    try:
        hits = qdrant_client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_vector,
            limit=1,
            score_threshold=0.97,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                    models.FieldCondition(key="sub_topics_key", match=models.MatchValue(value=sub_topics_key)),
                    models.FieldCondition(key="filters_key", match=models.MatchValue(value=filters_key)) 
                ]
            )
        ).points
        
        if hits:
            log.info(f"✓ Cache HIT (similarity: {hits[0].score:.3f}) for topic '{topic_id}' & sub_topics '{sub_topics_key}'")
            return hits[0].payload
        
        log.debug("Cache MISS")
        return None
        
    except Exception as e:
        log.error(f"Cache check failed: {e}")
        return None

def save_to_semantic_cache(query_vector, original_query, answer, sources, topic_id, sub_topics_key, filters_key):
    """Salva la risposta in cache includendo la chiave di combinazione dei sub-topic."""
    if not qdrant_client:
        return
    
    try:
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={
                "original_query": original_query,
                "answer": answer,
                "sources": sources,
                "topic_id": topic_id,
                "sub_topics_key": sub_topics_key, 
                "filters_key": filters_key,
                "timestamp": time.time()
            }
        )
        qdrant_client.upsert(collection_name=CACHE_COLLECTION, points=[point])
        log.debug("Cache entry saved.")
        
    except Exception as e:
        log.error(f"Cache save failed: {e}")

def get_topic_prompt(topic_id):
    """Recupera il template del prompt associato al topic. Ritorna 'general' come fallback."""
    conn = get_db_connection()
    if not conn:
        return 'general'
    
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT prompt FROM topics WHERE topic_id = %s", (topic_id,))
            row = cursor.fetchone()
            # Se il campo prompt è NULL o vuoto, restituisce 'general'
            return row['prompt'] if row and row.get('prompt') else 'general'
    except Exception as e:
        log.error(f"Errore durante il recupero del prompt per il topic '{topic_id}': {e}")
        return 'general'
    finally:
        conn.close()

def generate_sub_topics_key(selected_sub_topics):
   return ",".join(sorted(selected_sub_topics))


@celery_app.task(name="prune_semantic_cache")
def prune_semantic_cache():
    """Delete cache entries older than MAX_AGE_SECONDS."""
    if not qdrant_client:
        log.warning("Qdrant client not available for cache pruning.")
        return
    
    try:
        cutoff_time = time.time() - MAX_AGE_SECONDS
        
        qdrant_client.delete(
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
# 7. DOCUMENT RETRIEVAL (With OPTIMIZED Reranking)
# ==============================================================================
def retrieve_chunks(search_queries, vectors_list, keywords, topic_id, selected_sub_topics, metadata_filters):
    """
    Recupera, fonde e riordina i chunk dal Vector DB in modo sicuro.
    """
    if not qdrant_client:
        log.error("Qdrant client not available! Retrieval aborted.")
        return [], []
    
    log.info(f"=== Inizio Retrieval per topic '{topic_id}' ===")
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
                
                # Filtro range: {"data": {"gte": 1700000000, "lte": 1800000000}}
                elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                    must_conditions.append(models.FieldCondition(
                        key=key,
                        range=models.Range(
                            gte=value.get("gte"),
                            lte=value.get("lte")
                        )
                    ))
                
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
                res = qdrant_client.query_points_groups(
                    collection_name=QDRANT_COLLECTION,
                    query=vector,
                    group_by="content_hash",
                    group_size=1,
                    limit=settings.qdrant_semantic_size,
                    score_threshold=settings.qdrant_semantic_threshold,
                    query_filter=models.Filter(must=must_conditions)
                )
                points = [group.hits[0] for group in res.groups if group.hits]
                log.debug(f"Vector search #{idx} returned {len(points)} grouped hits.")
                return points
            except Exception as e:
                log.error(f"Vector search #{idx} failed: {e}")
                return []
            
        def safe_keyword_search():
            if not keywords: return []
            try:
                clean_kw = [w.lower().strip() for w in keywords if len(w.strip()) > 2]
                if not clean_kw: return []

                should_cond = [models.FieldCondition(key="content", match=models.MatchText(text=w)) for w in clean_kw]
                target_size = settings.qdrant_syntactic_size

                seen_hashes: set = set()
                unique_hits = []
                next_offset = None
                max_scroll_iterations = 5  # safety: evita loop infiniti se il corpus è quasi tutto duplicato
                iterations = 0

                while len(unique_hits) < target_size and iterations < max_scroll_iterations:
                    log.debug(f"Scroll {iterations} di {max_scroll_iterations}")

                    res, next_offset = qdrant_client.scroll(
                        collection_name=QDRANT_COLLECTION,
                        scroll_filter=models.Filter(must=must_conditions + [models.Filter(should=should_cond)]),
                        limit=target_size * 2,  # batch ampio per compensare i duplicati scartati
                        offset=next_offset,
                        with_payload=True
                    )
                    iterations += 1

                    if not res:
                        break

                    for point in res:
                        h = point.payload.get("content_hash") if point.payload else None
                        if h and h not in seen_hashes:
                            seen_hashes.add(h)
                            unique_hits.append(point)
                            if len(unique_hits) >= target_size:
                                break

                    if next_offset is None:
                        break  # esauriti i punti nella collection per questo filtro

                log.debug(f"Keyword search returned {len(unique_hits)} unique hits (deduped, {iterations} scroll iterations).")
                return unique_hits
            except Exception as e:
                log.error(f"Keyword search failed: {e}")
                return []

        # ==========================================================
        # ESECUZIONE PARALLELA 
        # ==========================================================
        hits_per_query = [[] for _ in search_queries]
        
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
 
        # Keyword hits come candidati extra della query principale
        if keyword_hits:
            kw_ids = {h.id for h in hits_per_query[0]}
            hits_per_query[0] = hits_per_query[0] + [h for h in keyword_hits if h.id not in kw_ids]

        # ==========================================================
        # 3. RERANK PER-QUERY — salva top_chunks per riuso nella redistribuzione
        # ==========================================================
        reranker = get_reranker()
        n_queries = len(search_queries)
        top_chunks_per_query: list[list] = [[] for _ in range(n_queries)]

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

            # Rerank effettivo
            if reranker:
                docs_content = [
                    c.payload.get("content", "")[:settings.rerank_truncate]
                    for c in unique_candidates
                ]
                try:
                    reranked = reranker.rerank(query_str, docs_content)
                    reranked.sort(key=lambda x: x.score, reverse=True)
                    top_chunks = []
                    for res in reranked[:settings.rerank_size]:
                        prob = safe_sigmoid(res.score)
                        if prob >= settings.min_prob_threshold or len(top_chunks) < 2:
                            top_chunks.append(unique_candidates[res.index])
                        else:
                            break
                    return top_chunks
                except Exception as e:
                    log.error(f"Reranker failed for query #{q_idx}: {e}. Fallback.")
                    
            unique_candidates.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            return unique_candidates[:settings.rerank_size]
        
        max_workers = min(n_queries, settings.max_reranker_thread)
        
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
 
        if slots_free > 0:
            log.info(f"Redistribuzione: {slots_free} slot liberi su {total_budget} totali.")
 
            # Riserve per ogni query: parent validi oltre la quota, nell'ordine del reranker
            reserve_per_query: list[list[str]] = []
            for top_chunks in top_chunks_per_query:
                reserve: list[str] = []
                seen_parents: set[str] = set()
                for chunk in top_chunks:
                    pid = chunk.payload.get("parent_id") if chunk.payload else None
                    if not pid or pid in seen_parents:
                        continue
                    seen_parents.add(pid)
                    if pid not in already_selected:
                        reserve.append(pid)
                reserve_per_query.append(reserve)
 
            # Round-robin tra le sottoquery finché slot esauriti o riserve vuote
            redistributed = 0
            changed = True
            while slots_free > 0 and changed:
                changed = False
                for q_idx, reserve in enumerate(reserve_per_query):
                    if slots_free == 0:
                        break
                    if not reserve:
                        continue
                    pid = reserve.pop(0)
                    if pid in already_selected:
                        continue
                    quota_per_query[q_idx].append(pid)
                    already_selected.add(pid)
                    slots_free -= 1
                    redistributed += 1
                    changed = True
 
            log.info(f"Redistribuzione completata: {redistributed} parent aggiunti.")
 
        # ==========================================================
        # 6. POOL FINALE — ordine FIFO per sottoquery
        # ==========================================================
        final_parent_ids_ordered: list[str] = []
        seen_final: set[str] = set()
 
        for selected_pids in quota_per_query:
            for pid in selected_pids:
                if pid not in seen_final:
                    final_parent_ids_ordered.append(pid)
                    seen_final.add(pid)
 
        if not final_parent_ids_ordered:
            log.warning("Nessun parent_id estratto da nessuna sottoquery.")
            return [], []
 
        log.info(
            f"Pool finale: {len(final_parent_ids_ordered)} parent distinti "
            f"(budget={settings.parents_per_query}×{n_queries}={total_budget})."
        )
 
 
        # ==========================================================
        # 7. RECUPERO PARENT DOCUMENTS (dal DB)
        # ==========================================================
        parent_records = []
        conn = get_db_connection()
        if not conn:
            return [], []
 
        try:
            log.debug(f"Parent ID da recuperare da DB: {final_parent_ids_ordered}")

            format_strings = ','.join(['%s'] * len(final_parent_ids_ordered))
            sql = f"""
                SELECT id, topic_id, sub_topic_id, source, content, metadata
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
            sub_topic_id = p_doc.get("sub_topic_id", "")
            content = p_doc.get("content", "").strip()
 
            if content:
                rich_context.append({"content": content, "source": source})
                if source not in unique_sources_map:
                    unique_sources_map[source] = {"file": source, "sub_topic": sub_topic_id}
 
        log.info(f"=== Retrieval completata. Parent al generatore: {len(rich_context)} ===")
        return rich_context, list(unique_sources_map.values())
 
    except Exception as e:
        log.critical(f"Errore critico nel Retrieval: {e}", exc_info=True)
        return [], []

def get_reranker_pool():
    global _RERANKER_POOL
    if not _RERANKER_POOL:
        for i in range(_POOL_SIZE):
            log.info(f"Initializing ONNX Reranker pool instance {i+1}/{_POOL_SIZE}...")
            instance = ONNXReranker(
                model_folder=settings.onnx_model_cache_path,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
                num_threads=2
            )
            _RERANKER_POOL.append(instance)
    return _RERANKER_POOL

def rerank_subquery(args):
    pool_idx, query_str, docs_content = args
    pool = get_reranker_pool()
    reranker_instance = pool[pool_idx % len(pool)]
    return reranker_instance.rerank(query_str, docs_content)


def get_reranker():
    """
    Lazy loader for the Reranker.
    Ensures initialization happens INSIDE the worker process, avoiding deadlocks.
    """
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        try:
            log.info("Initializing ONNX Reranker (Lazy Load)...")
            num_threads = min(os.cpu_count() or 2, settings.max_reranker_thread) 
            log.info(f"Using {num_threads} threads...")

            _RERANKER_INSTANCE = ONNXReranker(
                model_folder=settings.onnx_model_cache_path,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
                num_threads=num_threads
            )
            log.info("✓ Reranker initialized successfully.")
        except Exception as e:
            log.error(f"✗ Failed to lazy load Reranker: {e}")
            _RERANKER_INSTANCE = False # Mark as failed so we don't retry every time
            
    return _RERANKER_INSTANCE if _RERANKER_INSTANCE is not False else None



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

def get_aliases_by_concept(query_vector, threshold=settings.qdrant_concept_threshold):
    """
    Cerca nella collezione Qdrant se la query esprime un concetto 
    mappato nel nostro dizionario semantico.
    """
    if not qdrant_client:
        return []
        
    try:
        hits = qdrant_client.query_points(
            collection_name=CONCEPT_COLLECTION,
            query=query_vector,
            limit=1,
            score_threshold=threshold
        ).points
        
        if hits:
            aliases = hits[0].payload.get("aliases", [])
            log.info(f"[CONCEPT HIT] Rilevato concetto '{hits[0].payload.get('concept')}' (score: {hits[0].score:.3f}) -> Alias: {aliases}")
            return aliases
        else:
            log.debug(f"[CONCEPT MISS]")
        return []
    except Exception as e:
        log.error(f"Errore lookup concettuale su Qdrant: {e}")
        return []


def log_automatic_negative_feedback(query, answer, topic_id, history, task_id="UNKNOWN"):
    """
    Salva automaticamente nel database le query che non hanno trovato risposte
    valide o che sono state scartate dal Grader.
    """
    conn = get_db_connection()
    if not conn:
        log.error(f"[{task_id}] [AUTO_FEEDBACK] Impossibile salvare: DB pool non disponibile.")
        return
        
    try:
        # Serializziamo la cronologia come fatto sul gateway Flask
        history_json = json.dumps(history) if history else "[]"
        
        sql = """
            INSERT INTO chat_feedback 
            (user_query, ai_response, topic_id, rating, chat_history, comment) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        with conn.cursor() as cursor:
            cursor.execute(sql, (
                query,
                answer,
                topic_id,
                0,  # Rating 0 per identificare i fallimenti automatici del sistema
                history_json,
                "Auto-Log: Risposta non trovata o non soddisfacente (Grader/Self-Correction Fail)"
            ))
        conn.commit()
        log.info(f"[{task_id}] [AUTO_FEEDBACK] ✓ Query non risposta registrata con successo nel DB.")
        
    except Exception as e:
        log.error(f"[{task_id}] [AUTO_FEEDBACK] ✗ Errore durante il salvataggio a DB: {e}")
        conn.rollback()
    finally:
        conn.close()



# ==============================================================================
# 8. MAIN CELERY TASK
# ==============================================================================

@celery_app.task(bind=True, name="rag_queue")
def process_rag_query(self, query, history, topic_id, selected_sub_topics=None, metadata_filters=None):
    """Main RAG processing pipeline: JSON Mode, Multi-Query, and Defensive Error Handling."""
    start_time = time.time()
    task_id = self.request.id
    log.info(f"[{task_id}] Task started: '{query[:50]}...' su topic: {topic_id}")

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
    # ==========================================================
    try:
        # Nota l'aggiunta di task_id qui
        rewritten_data = transform_query(history, query, task_id)
        
        standalone_query = rewritten_data.get("standalone_query", query)
        # Generiamo l'embedding della sola query standalone per capire il concetto
        primary_vector = embed_query(standalone_query)

        extracted_keywords = rewritten_data.get("keywords", [])
        # Fallback testuale di sicurezza
        if not extracted_keywords:
            log.debug(f"[{task_id}] Nessuna keyword restituita, attivo fallback testuale in Python.")
            extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]

        # Verifichiamo SEMANTICAMENTE se la query esprime un concetto nel dizionario
        concept_vector = embed_for_concept_lookup(standalone_query)
        db_aliases = get_aliases_by_concept(concept_vector)
        
        if db_aliases:
            # Se c'è un concetto corrispondente, generiamo le sotto-query dedicate
            search_queries = expand_standalone_with_aliases(standalone_query, db_aliases, task_id)
            if not search_queries:
                search_queries = [standalone_query]
        else:
            # Altrimenti proseguiamo con il flusso nativo dell'LLM
            search_queries = rewritten_data.get("search_queries", [standalone_query])
        
        if standalone_query not in search_queries:
                    search_queries.insert(0, standalone_query)
        vectors_list = embed_queries_batch(search_queries) 
        all_keywords = list(set(extracted_keywords + db_aliases))
            
    except Exception as e:
        log.warning(f"[{task_id}] [MAIN_TASK] Pipeline di trasformazione caduta: {e}. Uso raw query.")
        standalone_query, search_queries, all_keywords = query, [query], []
        primary_vector = embed_query(query)  # fallback sul raw query
        vectors_list = [primary_vector]

    # ==========================================================
    # SEMANTIC CACHE CHECK
    # ==========================================================
    try:
        filters_key = generate_filters_key(metadata_filters)

        cached = check_semantic_cache(primary_vector, topic_id, st_key, filters_key)
        if cached:
            log.info(f"[{task_id}] Cache HIT. Returning cached response.")
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
        context, sources = retrieve_chunks(
            search_queries, 
            vectors_list, 
            all_keywords, 
            topic_id, 
            selected_sub_topics,
            metadata_filters
        )
    except Exception as e:
        log.error(f"[{task_id}] Vector DB Retrieval failed: {e}", exc_info=True)
        return {"error": "Database Error", "message": "Errore durante il recupero dei documenti.", "status": "failed"}

    # ==========================================================
    # SELF-CORRECTION LOOP (Generation & Grading)
    # ==========================================================
    attempt = 0
    answer = None
    is_satisfactory = False

    while attempt < MAX_MODEL_RETRIES and not is_satisfactory:
        if not context:
            answer = "<p>Non sono riuscito a trovare la risposta nei documenti che ho analizzato.</p>"
            log.debug(f"[{task_id}] Context empty. Breaking loop.")
            break
        else:
            log.debug(f"Contesto da passare alla generazione: {context[:1500]}")

        # A. Generazione (JSON Mode)
        try:
            gen_result = generate_answer(standalone_query, context, topic_id)
            is_found = gen_result.get("is_found", False)
            answer = gen_result.get("answer", "")

            log.debug(f"Risposta generata: {answer[:500]}... | is_found: {is_found}")
        except Exception as e:
            log.error(f"[{task_id}] Answer generation API failed: {e}", exc_info=True)
            return {"error": "Generation Failed", "message": "Impossibile elaborare la risposta.", "status": "failed"}

        # B. Logica Fail-Fast
        if not is_found:
            log.info(f"[{task_id}] Model explicitly flagged is_found=False. Skipping Grader.")
            log.debug(f"[{task_id}] Model answer: {answer}")
            is_satisfactory = False

        elif gen_result.get("is_general_knowledge"):
            if settings.allow_general_knowledge:
                log.info(f"[{task_id}] Answer based on general knowledge (allowed). Skipping Grader.")
                is_satisfactory = True
            else:
                # Il modello ha sforato i limiti: forza il retry senza sprecare una chiamata al grader
                log.warning(f"[{task_id}] Answer based on general knowledge but ALLOW_GENERAL_KNOWLEDGE=false. Forcing retry.")
                is_satisfactory = False

        else:
            # Risposta ancorata al contesto: valutazione normale
            try:
                if isinstance(context, list):
                    context_snippet = "\n\n".join(
                        f"[{item.get('source', '')}]\n{item.get('content', '')}"
                        for item in context
                    )
                else:
                    context_snippet = str(context)
                is_satisfactory = grade_answer(standalone_query, context_snippet, answer)
            except Exception as e:
                log.error(f"[{task_id}] Grader LLM failed (non-blocking): {e}. Defaulting to YES.")
                is_satisfactory = True

        # D. Gestione Retry Fallimento
        if not is_satisfactory:
            attempt += 1
            if attempt < MAX_MODEL_RETRIES:
                log.warning(f"[{task_id}] Answer unsatisfactory. Retrying ({attempt}/{MAX_MODEL_RETRIES}) with full pipeline...")
                try:
                    # 1. Iniettiamo un "falso" messaggio di sistema nella history per forzare l'LLM a cambiare approccio
                    retry_history = history.copy() if history else []
                    retry_history.append({
                        "role": "user", 
                        "text": f"La ricerca precedente per '{standalone_query}' non ha prodotto documenti validi. Riformula completamente la query usando sinonimi o concetti più ampi per esplorare un'angolazione semantica diversa."
                    })
                    
                    # 2. Riusiamo la funzione strutturata (JSON Mode + Facets + Keywords)
                    retry_data = transform_query(retry_history, query, f"{task_id}-RETRY")
                    
                    standalone_query = retry_data.get("standalone_query", query)
                    search_queries = retry_data.get("search_queries", [standalone_query])
                    extracted_keywords = retry_data.get("keywords", [])
                    
                    if not extracted_keywords:
                        extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]
                    if standalone_query not in search_queries:
                        search_queries.insert(0, standalone_query)
                    
                    # 3. Rieseguiamo il Batch Embedding e il Retrieval Multi-Query
                    vectors_list = embed_queries_batch(search_queries)
                    context, sources = retrieve_chunks(
                        search_queries, 
                        vectors_list, 
                        extracted_keywords, 
                        topic_id, 
                        selected_sub_topics,
                        metadata_filters
                    )
                except Exception as e:
                    log.error(f"[{task_id}] Retry infrastructure failed: {e}", exc_info=True)
                    break  # Usciamo usando l'ultima answer generata
            else:
                log.warning(f"[{task_id}] Max retries exhausted. Returning best effort answer.")

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
        # Evitiamo di cacchare risposte troppo brevi o palesemente vuote
        if answer and is_satisfactory and len(answer) > 20:
            save_to_semantic_cache(
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
    
    return {
        "answer": answer,
        "sources": sources,
        "topic": topic_id,
        "cached": False,
        "status": "success"
    }
    
# ==============================================================================
# 9. WORKER LIFECYCLE
# ==============================================================================

@worker_shutdown.connect
def cleanup_worker(**kwargs):
    """Gracefully close connections when the worker shuts down."""
    if qdrant_client:
        qdrant_client.close()
        log.info("Qdrant client closed.")