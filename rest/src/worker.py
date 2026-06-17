import os
import logging
import time
import uuid
from celery import Celery
from celery.signals import worker_process_init, worker_shutdown
from celery.schedules import crontab
from mysql.connector import pooling
from qdrant_client import QdrantClient, models
import google.generativeai as genai
import math
import json, re
import hashlib

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Use the optimized reranker
from reranker import ONNXReranker, RerankResult

from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

load_dotenv()


# ==============================================================================
# 1. CONFIGURATION & LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [WORKER-%(process)d] - %(levelname)s - %(message)s'
)
log = logging.getLogger("rag_queue")

# Load Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

# Reranking Configuration (OPTIMIZED)
RERANK_SIZE = int(os.environ.get("RERANK_SIZE", 25))
RERANK_TRUNCATE = int(os.environ.get("RERANK_TRUNCATE", 1200))
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", 32))  
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", 512))

ONNX_MODEL_CACHE_PATH = os.environ.get(
    "RERANKER_MODEL_PATH", 
    "./model_cache/mmarco-mMiniLMv2-L12-H384-v1"
)

QDRANT_SYNTATIC_SIZE = int(os.environ.get("QDRANT_SYNTATIC_SIZE", 20))
QDRANT_SEMANTIC_SIZE = int(os.environ.get("QDRANT_SEMANTIC_SIZE", 30))
QDRANT_THRESHOLD = float(os.environ.get("QDRANT_THRESHOLD", 0.60))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", 30000)) 

MIN_PROB_THRESHOLD = float(os.environ.get("MIN_PROB_THRESHOLD", 0.02))

MAX_AGE_SECONDS = 86400  # 24 hours
MAX_MODEL_RETRIES = 2

# Gemini Retry Configuration
GEMINI_RETRY = retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_random_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: log.warning(
        f"Rate limit hit. Retrying in {retry_state.next_action.sleep}s..."
    )
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_MODE"] = "PARALLEL"

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"

# ==============================================================================
# 2. CELERY INITIALIZATION
# ==============================================================================
celery_app = Celery('rag_queue', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    result_expires=3600,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,   
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_max_memory_per_child=2000000, # 2GB limit
)

# ==============================================================================
# PROCESS-SAFE INITIALIZATION 
# ==============================================================================
# Globals assigned strictly AFTER the fork
db_pool = None
qdrant_client = None
_RERANKER_INSTANCE = None

EMBEDDING_MODEL = "gemini-embedding-001"
TRANSFORM_MODEL = None
OCR_MODEL_NAME = None
GENERATOR_MODEL = None
TRANSFORM_MODEL_NAME = None

@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Initializes network connections and models ONLY after Celery forks the process.
    Prevents Socket Corruption and BrokenPipeErrors.
    """
    global db_pool, qdrant_client, TRANSFORM_MODEL, GENERATOR_MODEL,  OCR_MODEL_NAME, TRANSFORM_MODEL_NAME
    log.info("Initializing Worker Resources (Post-Fork)...")

    try:
        db_pool = pooling.MySQLConnectionPool(
            pool_name=f"worker_pool_{os.getpid()}",
            pool_size=3,
            pool_reset_session=True,
            host=os.environ.get("DB_HOST", "localhost"),
            user=os.environ.get("DB_USER", "root"),
            password=os.environ.get("DB_PASS", "password"),
            database=os.environ.get("DB_NAME", "rag_system")
        )
        qdrant_client = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"), 
            port=int(os.environ.get("QDRANT_PORT", 6333))
        )
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        # Instantiate models here
        OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "gemini-3.5-flash")
        TRANSFORM_MODEL_NAME = os.environ.get("TRANSFORM_MODEL_NAME", "gemini-3.1-flash-lite")
        TRANSFORM_MODEL = genai.GenerativeModel(TRANSFORM_MODEL_NAME)
        GENERATOR_MODEL = genai.GenerativeModel(OCR_MODEL_NAME)
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

def get_db_connection():
    """Get a connection from the pool and ensure it is alive."""
    if not db_pool:
        log.error("DB pool not initialized!")
        return None
    try:
        conn = db_pool.get_connection()
        # FIX: Pragmatic check to re-establish dropped connections
        conn.ping(reconnect=True, attempts=2, delay=1) 
        return conn
    except Exception as e:
        log.error(f"Error getting connection from pool: {e}")
        return None


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
# 5. GEMINI API FUNCTIONS (With Retry Logic)
# ==============================================================================

@GEMINI_RETRY
def embed_query(query):
    """Generate embedding for a query with retry logic."""
    log.debug(f"Embedding query: '{query[:50]}...'")
    
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768
    )
    return result['embedding']


@GEMINI_RETRY
def transform_query(history, query, task_id="UNKNOWN"):
    """Transform conversational query into standalone query via JSON Mode with full tracing."""
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
        query=query
    )
    
    try:
        log.debug(f"[{task_id}] [REWRITER] Chiamata a Gemini API in corso...")
        
        response = TRANSFORM_MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        
        # Logghiamo l'intero testo restituito da Gemini PRIMA di parsarlo
        log.debug(f"[{task_id}] [REWRITER] Risposta grezza Gemini:\n{response.text}")
        
        # Uso del parser robusto passandogli il task_id
        data = safe_json_parse(response.text, task_id)
        
        standalone = data.get("standalone_query", query)
        searches = data.get("search_queries", [query])
        keywords = data.get("keywords", [])
        
        log.info(f"[{task_id}] [REWRITER] ✓ Query processata. Standalone: '{standalone}' | Facets: {len(searches)} | Keywords: {len(keywords)}")
        
        return {
            "standalone_query": standalone,
            "search_queries": searches,
            "keywords": keywords
        }
        
    except Exception as e:
        # Se cade qui, o l'API è down o safe_json_parse ha sollevato il ValueError estremo
        log.warning(f"[{task_id}] [REWRITER] ✗ Fallimento critico: {e}. Attivazione fallback (query raw).", exc_info=True)
        return {
            "standalone_query": query,
            "search_queries": [query],
            "keywords": []
        }

@GEMINI_RETRY
def embed_queries_batch(queries_list):
    """Generate embeddings for MULTIPLE queries in a single API call."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=queries_list,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768
    )
    # result['embedding'] will be a list of vectors if input is a list
    return result['embedding'] if isinstance(queries_list, list) else [result['embedding']]


@GEMINI_RETRY
def generate_answer(query, rich_context, topic_id):
    """Generate final answer using retrieved context, returning a structured dict."""
    formatted_chunks = []
    curr_len = 0
    
    for item in rich_context:
        chunk = f"[Source: {item['source']}]\n{item['content']}\n\n"
        if curr_len + len(chunk) < MAX_CONTEXT_CHARS:
            formatted_chunks.append(chunk)
            curr_len += len(chunk)
        else:
            break
    
    context_str = "".join(formatted_chunks)
    prompt_file = get_topic_prompt(topic_id)
    
    prompt_tmpl = load_prompt_template(prompt_file)
    prompt = prompt_tmpl.format(context_str=context_str, query=query)
    
    log.info(f"Generating structured answer for topic '{topic_id}'")
    log.debug(f"Context size: {len(context_str)} chars")
    
    # 1. Chiediamo ESPLICITAMENTE il JSON tramite la configurazione
    response = GENERATOR_MODEL.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )
    
    # 2. Parsing sicuro del JSON
    try:
        raw_text = response.text.strip()
        result_data = json.loads(raw_text)
        
        # Garantiamo che restituisca sempre le chiavi attese
        return {
            "is_found": bool(result_data.get("is_found", True)),
            "answer": str(result_data.get("answer", ""))
        }
    except json.JSONDecodeError as e:
        log.error(f"Generazione JSON fallita: {e}. Output grezzo: {response.text}")
        # Fallback difensivo in caso di errore del modello
        return {
            "is_found": True,
            "answer": response.text.strip()
        }


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
        fused_hits = {}
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
                res = qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vector, 
                    limit=QDRANT_SEMANTIC_SIZE,
                    score_threshold=QDRANT_THRESHOLD,
                    query_filter=models.Filter(must=must_conditions)
                )
                log.debug(f"Vector search #{idx} returned {len(res.points)} hits.")
                return res.points
            except Exception as e:
                log.error(f"Vector search #{idx} failed: {e}")
                return []
            
        def safe_keyword_search():
            if not keywords: return []
            try:
                # Normalizzazione rigorosa delle keyword
                clean_kw = [w.lower().strip() for w in keywords if len(w.strip()) > 2]
                if not clean_kw: return []
                
                should_cond = [models.FieldCondition(key="content", match=models.MatchText(text=w)) for w in clean_kw]
                res = qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=models.Filter(must=must_conditions + [models.Filter(should=should_cond)]),
                    limit=QDRANT_SYNTATIC_SIZE * 2,
                    with_payload=True
                )
                log.debug(f"Keyword search returned {len(res[0])} hits.")
                return res[0]
            except Exception as e:
                log.error(f"Keyword search failed: {e}")
                return []

        # ==========================================================
        # 2. ESECUZIONE PARALLELA 
        # ==========================================================
        with ThreadPoolExecutor(max_workers=min(10, len(vectors_list) + 1)) as executor:
            futures = []
            
            # Lancio ricerca testuale
            futures.append(executor.submit(safe_keyword_search))
            
            # Lancio ricerche vettoriali
            for idx, vec in enumerate(vectors_list):
                futures.append(executor.submit(safe_vector_search, vec, idx))
                
            # Aggregazione e De-duplicazione
            for future in as_completed(futures):
                try:
                    hits = future.result()
                    for hit in hits:
                        if hit.id not in fused_hits: 
                            fused_hits[hit.id] = hit
                except Exception as e:
                    log.error(f"Error resolving search future: {e}", exc_info=True)

        candidates = list(fused_hits.values())
        if not candidates: 
            log.warning("Nessun Child Chunk trovato da nessuna delle query.")
            return [], []
            
        log.info(f"Ricerca parallela completata. Trovati {len(candidates)} candidati unici (Child).")

        # ==========================================================
        # 3. RERANKING LOCALE (Con Graceful Degradation)
        # ==========================================================
        top_child_docs = []
        
        try:
            reranker = get_reranker()
            if reranker and candidates:
                log.debug("Inizio Reranking locale...")
                # Usa la query originale per valutare la coerenza
                eval_query = search_queries[0] 
                docs_content = [c.payload.get("content", "")[:RERANK_TRUNCATE] for c in candidates]
                
                reranked_results = reranker.rerank(eval_query, docs_content)
                reranked_results.sort(key=lambda x: x.score, reverse=True)

                for res in reranked_results[:RERANK_SIZE]:
                    prob = safe_sigmoid(res.score)
                    if prob >= MIN_PROB_THRESHOLD or len(top_child_docs) < 2:
                        top_child_docs.append(candidates[res.index])
                    else:
                        break # Soglia minima non raggiunta, interrompiamo (sono già ordinati)
                
                log.info(f"Reranking completato. Sopravvissuti {len(top_child_docs)} Child validi.")
            else:
                log.warning("Reranker non disponibile. Fallback sui risultati grezzi di Qdrant.")
                # Ordinamento per score Qdrant (approssimativo) e taglio a RERANK_SIZE
                candidates.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
                top_child_docs = candidates[:RERANK_SIZE]
                
        except Exception as e:
            log.error(f"Errore fatale durante il Reranking: {e}. Fallback sui risultati grezzi.", exc_info=True)
            top_child_docs = candidates[:RERANK_SIZE]

        # ==========================================================
        # 4. RECUPERO PARENT DOCUMENTS
        # ==========================================================
        try:
            # Estrazione sicura dei parent_id
            parent_ids = list({
                doc.payload.get("parent_id") 
                for doc in top_child_docs 
                if doc.payload and doc.payload.get("parent_id")
            })

            if not parent_ids:
                log.error("I Child document non contengono alcun 'parent_id' nel payload.")
                return [], []

            log.debug(f"Recupero batch di {len(parent_ids)} Parent Documents da MySql...")
            parent_records = []
            conn = get_db_connection()
            if not conn:
                log.error("Connessione al database MySQL fallita durante il retrieval dei parent.")
                return [], []
            
            try:
                # Creazione di una query con numero variabile di segnaposto (%s) per prevenire SQL Injection
                format_strings = ','.join(['%s'] * len(parent_ids))
                query = f"""
                    SELECT id, topic_id, sub_topic_id, source, content, metadata 
                    FROM parent_documents 
                    WHERE id IN ({format_strings})
                """

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, tuple(parent_ids))
                    db_parents = cursor.fetchall()
                    
                    # Convertiamo i record MySQL nel formato dizionario che il blocco successivo (FORMATTAZIONE OUTPUT) si aspetta.
                    # Simuliamo la struttura "payload" di Qdrant per non rompere la logica successiva.
                    for row in db_parents:
                        parent_records.append({
                            "payload": {
                                "source": row.get("source"),
                                "sub_topic_id": row.get("sub_topic_id"),
                                "content": row.get("content")
                                # Se in futuro serve usare i metadati:
                                # "metadata": json.loads(row.get("metadata")) if row.get("metadata") else {}
                            }
                        })
                        
            finally:
                conn.close()

        except Exception as e:
            log.error(f"Errore recupero Parent Documents: {e}", exc_info=True)
            return [], []

        # ==========================================================
        # 5. FORMATTAZIONE OUTPUT
        # ==========================================================
        rich_context = []
        unique_sources_map = {}

        for p_doc in parent_records:
            if not p_doc.payload: continue
            
            source = p_doc.payload.get("source", "Fonte_Sconosciuta")
            sub_topic_id = p_doc.payload.get("sub_topic_id", "")
            content = p_doc.payload.get("content", "").strip()
            
            if content:
                rich_context.append({"content": content, "source": source})
                
                if source not in unique_sources_map:
                    unique_sources_map[source] = {"file": source, "sub_topic": sub_topic_id}

        log.info(f"=== Retrieval terminata con successo. Parent passati al generatore: {len(rich_context)} ===")
        return rich_context, list(unique_sources_map.values())
        
    except Exception as e:
        log.critical(f"Errore critico imprevisto nel Retrieval: {e}", exc_info=True)
        return [], []


def get_reranker():
    """
    Lazy loader for the Reranker.
    Ensures initialization happens INSIDE the worker process, avoiding deadlocks.
    """
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        try:
            log.info("Initializing ONNX Reranker (Lazy Load)...")
            num_threads = os.cpu_count() or 4

            # CRITICAL FIX 3: num_threads=1
            # We want the WORKER to be the unit of parallelism, not the matrix math.
            # This prevents 32 threads fighting for resources inside one worker.
            _RERANKER_INSTANCE = ONNXReranker(
                model_folder=ONNX_MODEL_CACHE_PATH,
                batch_size=RERANK_BATCH_SIZE,
                max_length=RERANK_MAX_LENGTH,
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
    # 1. SETUP & VALIDATION
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
    # 2. QUERY TRANSFORMATION & MULTI-QUERY EXPANSION
    # ==========================================================
    try:
        # Nota l'aggiunta di task_id qui
        rewritten_data = transform_query(history, query, task_id)
        
        standalone_query = rewritten_data.get("standalone_query", query)
        search_queries = rewritten_data.get("search_queries", [standalone_query])
        extracted_keywords = rewritten_data.get("keywords", [])
        
        # Fallback testuale di sicurezza
        if not extracted_keywords:
            log.debug(f"[{task_id}] Nessuna keyword restituita, attivo fallback testuale in Python.")
            extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]

        if standalone_query not in search_queries:
            search_queries.insert(0, standalone_query)
            
    except Exception as e:
        log.warning(f"[{task_id}] [MAIN_TASK] Pipeline di trasformazione caduta: {e}. Uso raw query.")
        standalone_query, search_queries, extracted_keywords = query, [query], []

    # ==========================================================
    # 3. BATCH EMBEDDING
    # ==========================================================
    try:
        vectors_list = embed_queries_batch(search_queries)
        # Vettore primario usato per la Semantic Cache
        primary_query_vector = vectors_list[0] 
    except Exception as e:
        log.error(f"[{task_id}] AI Embedding service failed: {e}", exc_info=True)
        return {"error": "AI Service Unavailable", "message": "Servizio momentaneamente sovraccarico.", "status": "failed"}

    # ==========================================================
    # 4. SEMANTIC CACHE CHECK
    # ==========================================================
    try:
        filters_key = generate_filters_key(metadata_filters)

        cached = check_semantic_cache(primary_query_vector, topic_id, st_key, filters_key)
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
    # 5. RETRIEVAL (Parallel Vector + Keyword)
    # ==========================================================
    try:
        context, sources = retrieve_chunks(
            search_queries, 
            vectors_list, 
            extracted_keywords, 
            topic_id, 
            selected_sub_topics,
            metadata_filters
        )
    except Exception as e:
        log.error(f"[{task_id}] Vector DB Retrieval failed: {e}", exc_info=True)
        return {"error": "Database Error", "message": "Errore durante il recupero dei documenti.", "status": "failed"}

    # ==========================================================
    # 6. SELF-CORRECTION LOOP (Generation & Grading)
    # ==========================================================
    attempt = 0
    answer = None
    is_satisfactory = False

    while attempt < MAX_MODEL_RETRIES and not is_satisfactory:
        if not context:
            answer = "<p>Non sono riuscito a trovare la risposta nei documenti che ho analizzato.</p>"
            log.debug(f"[{task_id}] Context empty. Breaking loop.")
            break

        # A. Generazione (JSON Mode)
        try:
            gen_result = generate_answer(standalone_query, context, topic_id)
            is_found = gen_result.get("is_found", False)
            answer = gen_result.get("answer", "")
        except Exception as e:
            log.error(f"[{task_id}] Answer generation API failed: {e}", exc_info=True)
            return {"error": "Generation Failed", "message": "Impossibile elaborare la risposta.", "status": "failed"}

        # B. Logica Fail-Fast
        if not is_found:
            log.info(f"[{task_id}] Model explicitly flagged is_found=False. Skipping Grader.")
            log.debug(f"[{task_id}] Model answer: {answer}")
            is_satisfactory = False
        else:
            # C. Grader LLM
            try:
                context_snippet = context if isinstance(context, str) else str(context)
                grader_tmpl = load_prompt_template("grader")
                grader_prompt = grader_tmpl.format(query=standalone_query, context_snippet=context_snippet, answer=answer)
                
                grade_response = TRANSFORM_MODEL.generate_content(
                    grader_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=5)
                ).text.strip().upper()
                
                log.info(f"[{task_id}] Grader response: '{grade_response}'")
                is_satisfactory = grade_response.startswith("YES")
                
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
    # 7. CACHE SAVING & RETURN
    # ==========================================================
    try:
        # Evitiamo di cacchare risposte troppo brevi o palesemente vuote
        if answer and is_satisfactory and len(answer) > 20:
            save_to_semantic_cache(
                primary_query_vector, 
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

