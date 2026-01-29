import os
import logging
import time
import uuid
from celery import Celery
from mysql.connector import pooling
from qdrant_client import QdrantClient, models
import google.generativeai as genai
from sentence_transformers.cross_encoder import CrossEncoder
import random

from dotenv import load_dotenv

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

# Configure Logging for the Worker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [WORKER] - %(levelname)s - %(message)s'
)
log = logging.getLogger("rag_queue")

# Load Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

# Constants
RERANK_SIZE = 15
QDRANT_SIZE = 20
MAX_CONTEXT_CHARS = 30000 

FORCE_TOPIC = os.environ.get("TOPIC", None)

MAX_AGE_SECONDS=86400

GEMINI_RETRY = retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: log.warning(f"Rate limit hit. Retrying in {retry_state.next_action.sleep}s...")
)



# ==============================================================================
# 2. CELERY INITIALIZATION
# ==============================================================================
celery_app = Celery('rag_queue', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    result_expires=3600,  # Results are deleted from Redis after 1 hour (3600 seconds)
    worker_concurrency=1 
)

# ==============================================================================
# 3. GLOBAL MODEL & DB INITIALIZATION (Runs ONCE on startup)
# ==============================================================================

log.info("Initializing Worker Resources...")

# --- A. Database Connection Pool ---
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
    log.info("MySQL Connection Pool created.")
except Exception as e:
    log.critical(f"Failed to create DB Pool: {e}")
    db_pool = None

# --- B. Qdrant Client ---
try:
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    QDRANT_COLLECTION = "document_chunks"
    CACHE_COLLECTION = "semantic_cache"
    log.info(f"Qdrant client connected to {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    log.critical(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

# --- C. Google Gemini ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        log.warning("GOOGLE_API_KEY not set!")
    else:
        log.debug("Using GOOGLE_API_KEY {GOOGLE_API_KEY}")

        genai.configure(api_key=GOOGLE_API_KEY)
        # Initialize specialized models
        EMBEDDING_MODEL = "gemini-embedding-001"
        TRANSFORM_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        ROUTER_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        GENERATOR_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        log.info("Gemini models configured.")
except Exception as e:
    log.critical(f"Failed to initialize Gemini: {e}")

# --- D. Cross-Encoder (The Heavy Lifter) ---
RERANKER_MODEL = None
try:
    log.info("Loading Cross-Encoder model...")
    # FORCE CPU DEVICE
    RERANKER_MODEL = CrossEncoder(
        'BAAI/bge-reranker-v2-m3', 
        max_length=1024,
        device="cpu",  # <--- CRITICAL CHANGE
        automodel_args={"torch_dtype": "auto"} 
    )
    log.info("Cross-Encoder loaded successfully on CPU.")
except Exception as e:
    log.critical(f"Failed to load Cross-Encoder: {e}")

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

def get_db_connection():
    if not db_pool: return None
    try:
        return db_pool.get_connection()
    except Exception as e:
        log.error(f"Error getting connection from pool: {e}")
        return None

def load_prompt_template(filename):
    try:
        file_path = os.path.join(PROMPTS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        log.error(f"Error loading prompt {filename}: {e}")
        return "{context_str}\n{query}"



@GEMINI_RETRY
def embed_query(query):
    log.info(f"Gemini EMBEDDING request: '{query}'")
    
    # Removed try/except so Tenacity can catch RateLimit errors
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768
    )
    return result['embedding']

@GEMINI_RETRY
def transform_query(history, query):
    if not history: return query
        
    history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('text', '')}" for msg in history])
    prompt = load_prompt_template("query_rewriter").format(history_str=history_str, query=query)
    
    # Log the full prompt
    log.info(f"Gemini TRANSFORM request:\n{prompt}")
    
    response = TRANSFORM_MODEL.generate_content(prompt)
    log.info(f"Gemini TRANSFORM response: '{response.text.strip()}'")
    return response.text.strip()

@GEMINI_RETRY
def route_query_to_topic(standalone_query, topics):
    topic_list_str = "\n".join([f"- {t['topic_id']}: {t['description']}" for t in topics])
    prompt = load_prompt_template("topic_finder").format(topic_string=topic_list_str, standalone_query=standalone_query)
    
    # Log the full prompt
    log.info(f"Gemini ROUTING request:\n{prompt}")
    
    response = ROUTER_MODEL.generate_content(prompt)
    topic_id = response.text.strip().replace("`", "").replace("'", "")
    
    log.info(f"Gemini ROUTING response: '{topic_id}'")
    
    if topic_id in [t['topic_id'] for t in topics]:
        return topic_id
    return None

def check_semantic_cache(query_vector):
    if not qdrant_client: return None
    try:
        hits = qdrant_client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_vector,
            limit=1,
            score_threshold=0.95
        ).points
        if hits:
            log.info("Cache HIT.")
            return hits[0].payload
    except Exception:
        pass
    return None

def prune_semantic_cache():
    """
    Deletes cache entries older than max_age_seconds (Default: 24 hours).
    """
    if not qdrant_client: return

    try:
        # Calculate the cutoff timestamp (Current Time - Max Age)
        cutoff_time = time.time() - MAX_AGE_SECONDS
        
        # Qdrant Delete Operation
        qdrant_client.delete(
            collection_name=CACHE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(lt=cutoff_time) # "Less Than" cutoff
                        )
                    ]
                )
            )
        )
        log.info(f"Cache Pruned: Deleted entries older than {MAX_AGE_SECONDS}s.")
    except Exception as e:
        log.error(f"Failed to prune cache: {e}")

def save_to_semantic_cache(query_vector, original_query, answer, sources, topic_id):
    if not qdrant_client: return
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

        if random.random() < 0.05:
            log.info("Triggering background cache cleanup...")
            prune_semantic_cache()
     
    except Exception as e:
        log.error(f"Cache save/prune failed: {e}")

def retrieve_chunks(query, vector, topic_id):
    """
    Hybrid Search with Client-Side 'Min-Match' Filtering.
    This bypasses the Pydantic ValidationError completely.
    """
    if not qdrant_client: 
        log.error("Qdrant client not available.")
        return [], []
    
    try:
        log.info(f"Retrieving chunks for topic '{topic_id}'...")
        fused_hits = {}
        
        # ======================================================================
        # 1. Vector Search (Semantic)
        # ======================================================================
        v_res = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=QDRANT_SIZE,
            query_filter=models.Filter(must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))])
        )
        
        v_count = len(v_res.points)
        log.info(f"Vector Search found {v_count} hits.")
        for hit in v_res.points: fused_hits[hit.id] = hit

        # ======================================================================
        # 2. Keyword Search (Manual Filtering)
        # ======================================================================
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        
        if keywords:
            # Calculate threshold (e.g., need 2 matches out of 4 keywords)
            min_match = max(1, len(keywords) // 2) if len(keywords) >= 4 else 1
            log.info(f"Keyword Search: {keywords} (Looking for >= {min_match} matches)")
            
            should_cond = [models.FieldCondition(key="content", match=models.MatchText(text=w)) for w in keywords]
            
            # --- SAFE QUERY: Ask for ANY match (OR logic) ---
            k_res = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))],
                    should=should_cond
                    # min_should_match removed to prevent Pydantic error
                ),
                # Fetch double the size to ensure we have enough candidates after filtering
                limit=QDRANT_SIZE * 2, 
                with_payload=True
            )
            
            k_hits_raw = k_res[0]
            k_valid_count = 0
            new_hits = 0
            
            # --- MANUAL FILTERING (The "Min Match" Logic) ---
            for hit in k_hits_raw:
                content = hit.payload.get("content", "").lower()
                
                # Count how many keywords appear in this document
                matches = sum(1 for w in keywords if w in content)
                
                # Strict Check: Only keep if matches >= min_match
                if matches >= min_match:
                    k_valid_count += 1
                    if hit.id not in fused_hits: 
                        fused_hits[hit.id] = hit
                        new_hits += 1
            
            log.info(f"Keyword Search: {len(k_hits_raw)} raw hits -> {k_valid_count} valid ({new_hits} unique added).")
        else:
            log.info("Skipping Keyword Search (no valid keywords).")

        candidates = list(fused_hits.values())
        if not candidates: 
            log.warning("No documents found in Vector or Keyword search.")
            return [], []

        log.info(f"Total Unique Candidates before Re-ranking: {len(candidates)}")

        # ======================================================================
        # 3. Re-ranking
        # ======================================================================
        if RERANKER_MODEL:
            log.info("Re-ranking candidates...")
            pairs = [(query, doc.payload.get('content', '')) for doc in candidates]
            scores = RERANKER_MODEL.predict(pairs)
            scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:RERANK_SIZE]]
        else:
            top_docs = candidates[:RERANK_SIZE]

        # 4. Format
        rich_context = [{"content": d.payload.get("content", ""), "source": d.payload.get("source", "")} for d in top_docs]
        unique_sources = list({d['source']: {"file": d['source']} for d in rich_context}.values())
        
        return rich_context, unique_sources
        
    except Exception as e:
        log.error(f"Retrieval process failed: {e}", exc_info=True)
        return [], []

@GEMINI_RETRY
def generate_answer(query, rich_context, topic_id, topics_list):
    # Construct Context String
    formatted_chunks = []
    curr_len = 0
    for item in rich_context:
        chunk = f"[Source: {item['source']}]\n{item['content']}\n\n"
        if curr_len + len(chunk) < MAX_CONTEXT_CHARS:
            formatted_chunks.append(chunk)
            curr_len += len(chunk)
        else:
            break
    
    topic_details = next((t for t in topics_list if t['topic_id'] == topic_id), None)
    prompt_file = topic_details['prompt'] if topic_details and topic_details.get('prompt') else "general"
    
    prompt_tmpl = load_prompt_template(prompt_file)
    prompt = prompt_tmpl.format(context_str="".join(formatted_chunks), query=query)
    
    # Log the prompt (Truncated if too massive, to keep logs readable)
    log.info(f"Gemini GENERATION request (Prompt Length: {len(prompt)} chars).")
    log.debug(f"Full Prompt Preview:\n{prompt}...[truncated]...")
    
    response = GENERATOR_MODEL.generate_content(prompt)
    return response.text.strip()

# ==============================================================================
# 5. THE MAIN CELERY TASK
# ==============================================================================

@celery_app.task(bind=True, name="rag_queue")
# ==============================================================================
# 5. THE MAIN CELERY TASK
# ==============================================================================

@celery_app.task(bind=True, name="rag_queue")
def process_rag_query(self, query, history):
    """
    The main entry point for the worker.
    """
    start_time = time.time()
    log.info(f"Task received. Query: {query[:30]}...")
    
    conn = None
    try:
        # ----------------------------------------------------------------------
        # 1. Transform Query
        # ----------------------------------------------------------------------
        try:
            # This calls the @GEMINI_RETRY decorated function
            # If it hits RateLimit, it will retry 6 times automatically.
            standalone_query = transform_query(history, query)
        except Exception as e:
            # If retries fail completely (e.g., after 60s), fall back to original query
            log.warning(f"Query Transformation failed after retries: {e}")
            standalone_query = query
        
        # ----------------------------------------------------------------------
        # 2. Embed Query
        # ----------------------------------------------------------------------
        try:
            query_vector = embed_query(standalone_query)
        except Exception as e:
            # Embedding is critical. If this fails, we cannot proceed.
            log.error(f"Embedding failed after retries: {e}")
            return {
                "error": "AI Service Unavailable", 
                "message": "The AI engine is currently overloaded. Please try again in a minute.",
                "status": "failed"
            }
        
        if not query_vector:
            return {"error": "Embedding returned None", "status": "failed"}

        # ----------------------------------------------------------------------
        # 3. Cache Check
        # ----------------------------------------------------------------------
        cached = check_semantic_cache(query_vector)
        if cached:
            log.info("Returning cached response.")
            return {
                "answer": cached['answer'],
                "sources": cached['sources'],
                "topic": cached['topic'],
                "cached": True,
                "status": "success"
            }

        # ----------------------------------------------------------------------
        # 4. DB Operations (Get Topics)
        # ----------------------------------------------------------------------
        conn = get_db_connection()
        if not conn: raise ValueError("DB Connection failed")
        
        topics = []
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT topic_id, description, aliases, prompt FROM topics")
            topics = cursor.fetchall()
        
        if not topics:
             log.warning("No topics found in DB.")
             return {"error": "System Configuration Error", "message": "No topics configured.", "status": "failed"}

        # ----------------------------------------------------------------------
        # 5. Route Query
        # ----------------------------------------------------------------------
        topic_id = FORCE_TOPIC
        try:
            if not topic_id:
                topic_id = route_query_to_topic(standalone_query, topics)
        except Exception as e:
            log.error(f"Routing failed after retries: {e}")
            
        if not topic_id:
            # Fallback strategy: If routing fails, maybe default to a 'general' topic or fail gracefully
            return {
                "error": "No matching topic found",
                "message": "I couldn't identify a relevant topic for your question based on the available documents.",
                "status": "failed"
            }
        else:
            log.debug(f"Query routed to topic: {topic_id}")

        # ----------------------------------------------------------------------
        # 6. Retrieve Documents (Hybrid Search)
        # ----------------------------------------------------------------------
        context, sources = retrieve_chunks(standalone_query, query_vector, topic_id)
        if not context:
            return {
                "error": "No context found",
                "message": f"I identified the topic as '{topic_id}', but I couldn't find specific documents matching your query.",
                "topic": topic_id,
                "status": "failed"
            }

        # ----------------------------------------------------------------------
        # 7. Generate Answer
        # ----------------------------------------------------------------------
        try:
            answer = generate_answer(standalone_query, context, topic_id, topics)
        except Exception as e:
             log.error(f"Generation failed after retries: {e}")
             return {
                 "error": "AI Generation Failed", 
                 "message": "I found the documents, but the AI service is currently overloaded and couldn't generate an answer.", 
                 "status": "failed"
             }
        
        # ----------------------------------------------------------------------
        # 8. Save to Cache
        # ----------------------------------------------------------------------
        # Only cache valid answers (avoid caching "I don't know" or empty responses)
        if answer and len(answer) > 10 and "could not find" not in answer.lower():
            save_to_semantic_cache(query_vector, standalone_query, answer, sources, topic_id)

        duration = time.time() - start_time
        log.info(f"Task completed successfully in {duration:.2f}s")
        
        return {
            "answer": answer,
            "sources": sources,
            "topic": topic_id,
            "status": "success"
        }

    except Exception as e:
        log.critical(f"Unhandled Worker Exception: {e}", exc_info=True)
        return {
            "error": "Internal Processing Error",
            "message": "An unexpected error occurred in the AI worker.",
            "status": "failed"
        }
    finally:
        if conn and conn.is_connected():
            conn.close()