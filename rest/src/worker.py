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
RERANK_THRESHOLD = 0.40

ONNX_MODEL_CACHE_PATH = os.environ.get(
    "RERANKER_MODEL_PATH", 
    "./model_cache/mmarco-mMiniLMv2-L12-H384-v1"
)

QDRANT_SYNTATIC_SIZE = 20
QDRANT_SEMANTIC_SIZE = 30
QDRANT_THRESHOLD = 0.60
MAX_CONTEXT_CHARS = 30000 

MAX_AGE_SECONDS = 86400  # 24 hours

# Gemini Retry Configuration
GEMINI_RETRY = retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: log.warning(
        f"Rate limit hit. Retrying in {retry_state.next_action.sleep}s..."
    )
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_MODE"] = "PARALLEL"

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
PARENT_COLLECTION = "parent_documents"

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
# PROCESS-SAFE INITIALIZATION (THE CRITICAL FIX)
# ==============================================================================
# Globals assigned strictly AFTER the fork
db_pool = None
qdrant_client = None
_RERANKER_INSTANCE = None

EMBEDDING_MODEL = "gemini-embedding-001"
TRANSFORM_MODEL = None
GENERATOR_MODEL = None

@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Initializes network connections and models ONLY after Celery forks the process.
    Prevents Socket Corruption and BrokenPipeErrors.
    """
    global db_pool, qdrant_client, TRANSFORM_MODEL, GENERATOR_MODEL
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
        OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "gemini-3-flash-preview")
        TRANSFORM_MODEL = genai.GenerativeModel(OCR_MODEL_NAME)
        GENERATOR_MODEL = genai.GenerativeModel(OCR_MODEL_NAME)
        log.info("✓ Resources successfully initialized for this process.")
    except Exception as e:
        log.critical(f"✗ Failed to initialize worker resources: {e}")
        raise


# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

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
def transform_query(history, query):
    """Transform conversational query into standalone query."""
    if not history:
        log.debug("No history provided, using original query.")
        return query
    
    history_str = "\n".join([
        f"{msg.get('role', 'user')}: {msg.get('text', '')}" 
        for msg in history
    ])
    
    prompt = load_prompt_template("query_rewriter").format(
        history_str=history_str, 
        query=query
    )
    
    log.debug(f"Transforming query with history context...")
    
    response = TRANSFORM_MODEL.generate_content(prompt)
    transformed = response.text.strip()
    
    log.debug(f"Query transformed: '{query}' → '{transformed}'")
    return transformed


@GEMINI_RETRY
def extract_keywords_with_ai(query):
    """Extract significant keywords using AI."""
    prompt = load_prompt_template("extract_keywords").format(query=query)
    log.info(f"input prompt'{prompt}'")
    
    response = TRANSFORM_MODEL.generate_content(prompt)
    cleaned_text = response.text.strip()
    
    log.info(f"AI extracted keywords: '{cleaned_text}'")
    
    keywords = cleaned_text.replace(",", " ").replace(".", "").split()
    keywords = [w.lower() for w in keywords if len(w) > 2]
    
    return list(dict.fromkeys(keywords))


@GEMINI_RETRY
def generate_answer(query, rich_context, topic_id):
    """Generate final answer using retrieved context."""
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
    
    # 2. Recupero del Prompt 
    prompt_file = get_topic_prompt(topic_id)
    
    prompt_tmpl = load_prompt_template(prompt_file)
    prompt = prompt_tmpl.format(context_str=context_str, query=query)
    
    log.info(f"Generating answer for topic '{topic_id}' (prompt file: {prompt_file})")
    log.debug(f"Context size: {len(context_str)} chars, Chunks: {len(rich_context)}")
    
    log.debug(f"Context... {context_str[:200]}... ")
    
    response = GENERATOR_MODEL.generate_content(prompt)
    return response.text.strip()


# ==============================================================================
# 6. SEMANTIC CACHE FUNCTIONS
# ==============================================================================

def check_semantic_cache(query_vector, topic_id, sub_topics_key):
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
                    models.FieldCondition(key="sub_topics_key", match=models.MatchValue(value=sub_topics_key))
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

def save_to_semantic_cache(query_vector, original_query, answer, sources, topic_id, sub_topics_key):
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
def retrieve_chunks(query, vector, topic_id, selected_sub_topics):
    if not qdrant_client:
        log.error("Qdrant client not available!")
        return [], []
    
    try:
        log.info(f"=== Inizio Retrieval (Parent-Child) per topic '{topic_id}' ===")
        fused_hits = {}
        
        # ==================================================================
        # 1. PARALLEL SEARCH (Vector + Keyword)
        # ==================================================================
        must_conditions = [models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))]
        if selected_sub_topics:
            must_conditions.append(models.FieldCondition(key="sub_topic_id", match=models.MatchAny(any=selected_sub_topics)))

        def vector_search():
            res = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector, 
                limit=QDRANT_SEMANTIC_SIZE,
                score_threshold=QDRANT_THRESHOLD,
                query_filter=models.Filter(must=must_conditions)
            )
            log.info(f"Child Vector search: {len(res.points)} hits")
            return res.points
        
        def keyword_search():
            try:
                keywords = extract_keywords_with_ai(query)
                log.info(f"Keywords estratte per Child Search: {keywords}")
            except Exception as e:
                log.warning(f"AI keyword extraction fallita: {e}. Uso fallback.")
                keywords = [w.lower() for w in query.split() if len(w) > 3]
            
            if not keywords: 
                return []
            
            should_cond = [models.FieldCondition(key="content", match=models.MatchText(text=w)) for w in keywords]
            k_res = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=models.Filter(must=must_conditions + [models.Filter(should=should_cond)]),
                limit=QDRANT_SYNTATIC_SIZE * 2,
                with_payload=True
            )
            hits = k_res[0]
            log.info(f"Child Keyword search: {len(hits)} hits validi")
            return hits
        
        # Esecuzione parallela delle ricerche sui Child
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(vector_search): "vector", executor.submit(keyword_search): "keyword"}
            for future in as_completed(futures):
                try:
                    for hit in future.result():
                        if hit.id not in fused_hits: fused_hits[hit.id] = hit
                except Exception as e:
                    log.error(f"Ricerca {futures[future]} fallita: {e}")
                    
        candidates = list(fused_hits.values())
        if not candidates: 
            log.warning("Nessun Child Chunk trovato.")
            return [], []

        # ==================================================================
        # 2. RERANKING SUI CHILD CHUNKS
        # ==================================================================
        top_child_docs = []
        reranker = get_reranker()
        
        if reranker and candidates:
            log.info(f"Reranking di {len(candidates)} candidati (Child)...")
            docs_content = [c.payload.get("content", "")[:RERANK_TRUNCATE] for c in candidates]
            try:
                reranked_results = reranker.rerank(query, docs_content)
                #top_child_docs = [candidates[res.index] for res in reranked_results[:RERANK_SIZE]]
                top_child_docs = [
                    candidates[res.index] 
                    for res in reranked_results 
                        if res.score >= RERANK_THRESHOLD  
                ][:RERANK_SIZE] 
                
                log.info(f"Reranking completato. Selezionati {len(top_child_docs)} Child migliori.")
            except Exception as e:
                log.error(f"Reranking fallito: {e}")
                top_child_docs = candidates[:RERANK_SIZE]
        else:
            top_child_docs = candidates[:RERANK_SIZE]

        # ==================================================================
        # 3. RECUPERO DEI PARENT DOCUMENTS (Il "Contesto Vero")
        # ==================================================================
        # Estraiamo i parent_id unici dai Child migliori
        parent_ids = list(set([
            doc.payload.get("parent_id") for doc in top_child_docs 
            if doc.payload and doc.payload.get("parent_id")
        ]))

        if not parent_ids:
            log.error("Errore critico: i Child non hanno parent_id nel payload!")
            return [], []

        log.info(f"Recupero di {len(parent_ids)} Parent Documents dalla collezione '{PARENT_COLLECTION}'...")
        
        # Recupero batch dei Parent tramite ID
        parent_records = qdrant_client.retrieve(
            collection_name=PARENT_COLLECTION,
            ids=parent_ids,
            with_payload=True
        )

        # ==================================================================
        # 4. ORGANIZZAZIONE RISULTATI PER IL GENERATOR
        # ==================================================================
        rich_context = []
        unique_sources_map = {}

        for p_doc in parent_records:
            source = p_doc.payload.get("source", "Fonte sconosciuta")
            sub_topic_id = p_doc.payload.get("sub_topic_id", "")
            content = p_doc.payload.get("content", "").strip()
            
            if content:
                rich_context.append({
                    "content": content,
                    "source": source
                })
                
                if source not in unique_sources_map:
                    unique_sources_map[source] = {"file": source, "sub_topic": sub_topic_id}

        log.info(f"=== Retrieval terminata. Inviati {len(rich_context)} Parent Documents a Gemini. ===")
        return rich_context, list(unique_sources_map.values())
        
    except Exception as e:
        log.error(f"Errore durante il processo di retrieval: {e}", exc_info=True)
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
def process_rag_query(self, query, history, topic_id, selected_sub_topics=None):
    """Main RAG processing pipeline with optimized reranking."""
    start_time = time.time()
    task_id = self.request.id
    log.info(f"[{task_id}] Task started: '{query[:50]}...' su topic: {topic_id}")

    if not selected_sub_topics:
        selected_sub_topics = get_all_sub_topics(topic_id)
        if not selected_sub_topics:
            log.error(f"[{task_id}] Nessun sub-topic trovato o configurato per il topic '{topic_id}'.")
            return {
                "error": "Configuration Error",
                "message": "Il topic selezionato non contiene documenti o non è configurato correttamente.",
                "status": "failed"
            }
        
    try:
        # 1. Query Transformation
        try:
            standalone_query = transform_query(history, query)
        except Exception as e:
            log.warning(f"Query transformation failed: {e}")
            standalone_query = query
        
        # 2. Embedding
        try:
            query_vector = embed_query(standalone_query)
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            return {
                "error": "AI Service Unavailable",
                "message": "The AI service is currently overloaded. Please try again.",
                "status": "failed"
            }
        
        if not query_vector:
            return {
                "error": "Embedding Failed",
                "message": "Unable to process your question. Please try rephrasing.",
                "status": "failed"
            }
        
        # 3. Cache Check
        st_key = generate_sub_topics_key(selected_sub_topics)
        cached = check_semantic_cache(query_vector, topic_id, st_key)
        if cached:
            log.info(f"[{task_id}] Returning cached response.")
            return {
                "answer": cached['answer'],
                "sources": cached['sources'],
                "topic": cached['topic_id'],
                "cached": True,
                "status": "success"
            }
        
        # 4. Document Retrieval (Saltiamo totalmente la logica LLM di routing)
        context, sources = retrieve_chunks(
            standalone_query, 
            query_vector, 
            topic_id, 
            selected_sub_topics
        )

        if not context:
            return {
                "error": "No Relevant Documents",
                "message": f"I couldn't find specific information to answer your question.",
                "topic": topic_id,
                "status": "failed"
            }
        
        # 5. Answer Generation
        try:
            answer = generate_answer(standalone_query, context, topic_id)
        except Exception as e:
            log.error(f"Answer generation failed: {e}")
            return {
                "error": "Generation Failed",
                "message": "I found documents but couldn't generate an answer.",
                "status": "failed"
            }
        
        # 6. Cache Valid Responses
        if answer and len(answer) > 20 and "could not find" not in answer.lower():
            save_to_semantic_cache(
                query_vector, 
                standalone_query, 
                answer, 
                sources, 
                topic_id,
                st_key
            )
        
        duration = time.time() - start_time
        log.info(f"[{task_id}] ✓ Completed in {duration:.2f}s")
        
        return {
            "answer": answer,
            "sources": sources,
            "topic": topic_id,
            "cached": False,
            "status": "success"
        }
    
    except Exception as e:
        log.critical(f"[{task_id}] ✗ Unhandled exception: {e}", exc_info=True)
        return {
            "error": "Internal Processing Error",
            "message": "An unexpected error occurred.",
            "status": "failed"
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

