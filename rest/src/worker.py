import os
import logging
import time
import uuid
from celery import Celery
from celery.signals import worker_process_init, worker_shutdown
from celery.schedules import crontab
from celery.signals import worker_shutdown
from mysql.connector import pooling
from qdrant_client import QdrantClient, models
import google.generativeai as genai
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


QDRANT_SYNTATIC_SIZE = 20
QDRANT_SEMANTIC_SIZE = 30
MAX_CONTEXT_CHARS = 30000 

FORCE_TOPIC = os.environ.get("TOPIC", None)
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


# ==============================================================================
# 2. CELERY INITIALIZATION
# ==============================================================================
celery_app = Celery('rag_queue', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    result_expires=3600,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,   
    task_acks_late=True,
    task_reject_on_worker_lost=True
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
ROUTER_MODEL = None
GENERATOR_MODEL = None

@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Initializes network connections and models ONLY after Celery forks the process.
    Prevents Socket Corruption and BrokenPipeErrors.
    """
    global db_pool, qdrant_client, TRANSFORM_MODEL, ROUTER_MODEL, GENERATOR_MODEL
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
        TRANSFORM_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        ROUTER_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        GENERATOR_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        log.info("✓ Resources successfully initialized for this process.")
    except Exception as e:
        log.critical(f"✗ Failed to initialize worker resources: {e}")
        raise

# ==============================================================================
# 3. GLOBAL MODEL & DB INITIALIZATION
# ==============================================================================

log.info("Initializing Worker Resources...")

# --- A. Database Connection Pool ---
db_pool = None
try:
    db_config = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "user": os.environ.get("DB_USER", "root"),
        "password": os.environ.get("DB_PASS", "password"),
        "database": os.environ.get("DB_NAME", "rag_system")
    }

    db_pool = pooling.MySQLConnectionPool(
        pool_name="worker_pool",
        pool_size=5,
        pool_reset_session=True,
        **db_config
    )
    log.info("✓ MySQL Connection Pool created.")
except Exception as e:
    log.critical(f"✗ Failed to create DB Pool: {e}")
    raise

# --- B. Qdrant Client ---
qdrant_client = None
try:
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    QDRANT_COLLECTION = "document_chunks"
    CACHE_COLLECTION = "semantic_cache"
    
    qdrant_client.get_collections()
    log.info(f"✓ Qdrant client connected to {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    log.critical(f"✗ Failed to connect to Qdrant: {e}")
    raise

# --- C. Google Gemini ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set!")
    
    log.info(f"Using GOOGLE_API_KEY: {GOOGLE_API_KEY[:8]}***")

    genai.configure(api_key=GOOGLE_API_KEY)
    
    EMBEDDING_MODEL = "gemini-embedding-001"
    TRANSFORM_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    ROUTER_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    GENERATOR_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    
    log.info("✓ Gemini models configured.")
except Exception as e:
    log.critical(f"✗ Failed to initialize Gemini: {e}")
    raise

# --- D. OPTIMIZED Cross-Encoder Reranker ---
#ONNX_MODEL_CACHE_PATH = "./model_cache/bge-reranker-v2-m3-ONNX-int8"
ONNX_MODEL_CACHE_PATH = "./model_cache/mmarco-mMiniLMv2-L12-H384-v1"
_RERANKER_INSTANCE = None # Singleton holder for the reranker, initialized lazily inside the worker process to avoid deadlocks.

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

def get_db_connection():
    """Get a connection from the pool."""
    if not db_pool:
        log.error("DB pool not initialized!")
        return None
    try:
        return db_pool.get_connection()
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
def route_query_to_topic(standalone_query, topics):
    """Route query to most relevant topic using AI."""
    topic_list_str = "\n".join([
        f"- {t['topic_id']}: {t['description']}" 
        for t in topics
    ])
    
    prompt = load_prompt_template("topic_finder").format(
        topic_string=topic_list_str, 
        standalone_query=standalone_query
    )
    
    log.debug("Routing query to topic...")
    
    response = ROUTER_MODEL.generate_content(prompt)
    topic_id = response.text.strip().replace("`", "").replace("'", "").replace('"', "")
    
    valid_topics = [t['topic_id'] for t in topics]
    if topic_id in valid_topics:
        log.info(f"Query routed to topic: '{topic_id}'")
        return topic_id
    
    log.warning(f"AI returned invalid topic '{topic_id}'. Available: {valid_topics}")
    return None


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
def generate_answer(query, rich_context, topic_id, topics_list):
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
    
    topic_details = next(
        (t for t in topics_list if t['topic_id'] == topic_id), 
        None
    )
    
    if topic_details:
        prompt_file = topic_details.get('prompt') or 'general'
    else:
        prompt_file = 'general'
    
    prompt_tmpl = load_prompt_template(prompt_file)
    prompt = prompt_tmpl.format(context_str=context_str, query=query)
    
    log.info(f"Generating answer (prompt: {len(prompt)} chars, "
             f"chunks: {len(rich_context)})")
    
    log.debug(f"Context... {context_str}... ")
    
    response = GENERATOR_MODEL.generate_content(prompt)
    return response.text.strip()


# ==============================================================================
# 6. SEMANTIC CACHE FUNCTIONS
# ==============================================================================

def check_semantic_cache(query_vector):
    """Check if similar query exists in cache."""
    if not qdrant_client:
        return None
    
    try:
        hits = qdrant_client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_vector,
            limit=1,
            score_threshold=0.97
        ).points
        
        if hits:
            log.info(f"✓ Cache HIT (similarity: {hits[0].score:.3f})")
            return hits[0].payload
        
        log.debug("Cache MISS")
        return None
        
    except Exception as e:
        log.error(f"Cache check failed: {e}")
        return None


def save_to_semantic_cache(query_vector, original_query, answer, sources, topic_id):
    """Save successful response to semantic cache."""
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
                "topic": topic_id,
                "timestamp": time.time()
            }
        )

        qdrant_client.upsert(collection_name=CACHE_COLLECTION, points=[point])
        log.debug("Cache entry saved.")
        
    except Exception as e:
        log.error(f"Cache save failed: {e}")


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

def retrieve_chunks(query, vector, topic_id):
    """
    Hybrid retrieval with OPTIMIZED reranking.
    
    Expected: 94s → ~5-10s for 59 documents
    """
    if not qdrant_client:
        log.error("Qdrant client not available!")
        return [], []
    
    try:
        log.info(f"Retrieving chunks for topic '{topic_id}'...")
        fused_hits = {}
        
        # ==================================================================
        # PARALLEL SEARCH
        # ==================================================================
        def vector_search():
            """Semantic vector search."""
            res = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=QDRANT_SEMANTIC_SIZE,
                query_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="topic_id", 
                        match=models.MatchValue(value=topic_id)
                    )]
                )
            )
            log.info(f"Vector search: {len(res.points)} hits")
            return res.points
        
        def keyword_search():
            """Keyword search with AI extraction."""
            try:
                keywords = extract_keywords_with_ai(query)
                log.info(f"Keywords extracted: {keywords}")
            except Exception as e:
                log.warning(f"AI keyword extraction failed: {e}. Using fallback.")
                keywords = [w.lower() for w in query.split() if len(w) > 3]
                keywords = list(dict.fromkeys(keywords))
            
            if not keywords:
                log.info("No valid keywords - skipping keyword search.")
                return []
            
            min_match = 1
            
            log.debug(f"Min-match threshold: {min_match}/{len(keywords)}")
            
            should_cond = [
                models.FieldCondition(key="content", match=models.MatchText(text=w))
                for w in keywords
            ]
            
            k_res = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        # Condition 1: Must match the topic exactly
                        models.FieldCondition(
                            key="topic_id", 
                            match=models.MatchValue(value=topic_id)
                        ),
                        # Condition 2: Must match AT LEAST ONE of the keywords
                        models.Filter(
                            should=should_cond
                        )
                    ]
                ),
                limit=QDRANT_SYNTATIC_SIZE * 2,
                with_payload=True
            )
            
            k_hits_raw = k_res[0]
            valid_hits = []
            
            for hit in k_hits_raw:
                content = hit.payload.get("content", "").lower()
                matches = sum(1 for w in keywords if w in content)
                
                if matches >= min_match:
                    valid_hits.append(hit)
            
            log.info(f"Keyword search: {len(k_hits_raw)} raw → {len(valid_hits)} valid")
            return valid_hits
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(vector_search): "vector", executor.submit(keyword_search): "keyword"}
            for future in as_completed(futures):
                try:
                    for hit in future.result():
                        if hit.id not in fused_hits: fused_hits[hit.id] = hit
                except Exception as e:
                    log.error(f"{futures[future]} search failed: {e}")
        
        candidates = list(fused_hits.values())
        
        if not candidates:
            log.warning("No documents found in vector or keyword search.")
            return [], []
        
        log.info(f"Total unique candidates: {len(candidates)}")
        
        top_docs = []
        reranker = get_reranker()
        
        if reranker and candidates:
            log.info(f"Reranking {len(candidates)} candidates...")
            docs_content = [c.payload.get("content", "")[:RERANK_TRUNCATE] for c in candidates]
            
            try:
                start_rr = time.time()
                
                # Run reranking (Lazy loaded instance)
                reranked_results = reranker.rerank(query, docs_content)
                
                for res in reranked_results[:RERANK_SIZE]:
                    top_docs.append(candidates[res.index])
                
                elapsed = time.time() - start_rr
                log.info(f"⚡ Reranking completed in {elapsed:.2f}s")
                
            except Exception as e:
                log.error(f"Reranking failed: {e}", exc_info=True)
                top_docs = candidates[:RERANK_SIZE]
        else:
            log.warning("Reranker not initialized or unavailable.")
            top_docs = candidates[:RERANK_SIZE]

        # ==================================================================
        # FORMAT OUTPUT
        # ==================================================================
        rich_context = [
            {
                "content": d.payload.get("content", ""),
                "source": d.payload.get("source", "")
            }
            for d in top_docs
        ]
        
        unique_sources = list({
            d['source']: {"file": d['source']} 
            for d in rich_context
        }.values())
        
        return rich_context, unique_sources
        
    except Exception as e:
        log.error(f"Retrieval process failed: {e}", exc_info=True)
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

# ==============================================================================
# 8. MAIN CELERY TASK
# ==============================================================================

@celery_app.task(bind=True, name="rag_queue")
def process_rag_query(self, query, history):
    """Main RAG processing pipeline with optimized reranking."""
    start_time = time.time()
    task_id = self.request.id
    log.info(f"[{task_id}] Task started: '{query[:50]}...'")
    
    conn = None
    
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
        cached = check_semantic_cache(query_vector)
        if cached:
            log.info(f"[{task_id}] Returning cached response.")
            return {
                "answer": cached['answer'],
                "sources": cached['sources'],
                "topic": cached['topic'],
                "cached": True,
                "status": "success"
            }
        
        # 4. Load Topics
        conn = get_db_connection()
        if not conn:
            raise ValueError("Database connection failed")
        
        topics = []
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT topic_id, description, aliases, prompt FROM topics")
            topics = cursor.fetchall()
        
        if not topics:
            log.error("No topics configured!")
            return {
                "error": "System Configuration Error",
                "message": "The system is not properly configured.",
                "status": "failed"
            }
        
        # 5. Topic Routing\
        topic_id = FORCE_TOPIC
        
        if not topic_id:
            try:
                topic_id = route_query_to_topic(standalone_query, topics)
            except Exception as e:
                log.error(f"Routing failed: {e}")
        else:
            log.debug(f"Forced topic: {topic_id}")
        
        if not topic_id:
            return {
                "error": "No Matching Topic",
                "message": "I couldn't identify a relevant topic for your question.",
                "status": "failed"
            }
        
        log.info(f"[{task_id}] Routed to topic: '{topic_id}'")
        
        # 6. Document Retrieval (WITH OPTIMIZED RERANKING!)
        context, sources = retrieve_chunks(standalone_query, query_vector, topic_id)
        
        if not context:
            return {
                "error": "No Relevant Documents",
                "message": f"I couldn't find specific information to answer your question.",
                "topic": topic_id,
                "status": "failed"
            }
        
        # 7. Answer Generation
        try:
            answer = generate_answer(standalone_query, context, topic_id, topics)
        except Exception as e:
            log.error(f"Answer generation failed: {e}")
            return {
                "error": "Generation Failed",
                "message": "I found documents but couldn't generate an answer.",
                "status": "failed"
            }
        
        # 8. Cache Valid Responses
        if answer and len(answer) > 20 and "could not find" not in answer.lower():
            save_to_semantic_cache(
                query_vector, 
                standalone_query, 
                answer, 
                sources, 
                topic_id
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
    
    finally:
        if conn and conn.is_connected():
            conn.close()


# ==============================================================================
# 9. WORKER LIFECYCLE
# ==============================================================================

def cleanup_worker(**kwargs):
    if qdrant_client: qdrant_client.close()

