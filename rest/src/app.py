import os
import logging
import json
import time
import uuid
from dotenv import load_dotenv

# --- Flask and DB Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from qdrant_client import QdrantClient, models

# --- Google AI ---
import google.generativeai as genai

# --- Re-ranking ---
from sentence_transformers.cross_encoder import CrossEncoder

# Define the prompts directory path relative to this script
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

load_dotenv()

# Set log level from environment variable, default to INFO
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Log to console
)
log = logging.getLogger("app")

log.info(f"Logging level set to {log_level}")
# ==============================================================================
# 2. INITIALIZE CLIENTS AND CONSTANTS
# ==============================================================================

RERANK_SIZE = 15
QDRANT_SIZE = 20

# --- CONFIGURATION ---
    # 30k chars is approx 7.5k tokens. Safe for Gemini Flash (1M window) 
    # but prevents massive payloads that slow down response.
MAX_CONTEXT_CHARS = 30000 


# --- App Initialization ---
app = Flask(__name__)
CORS(app)
log.info("CORS enabled for all routes.")

# --- Database Config ---
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME", "rag_system")
}
log.debug(f"MySQL configured for host: {db_config['host']}")

# --- Qdrant Client ---
try:
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    QDRANT_COLLECTION = "document_chunks"
    CACHE_COLLECTION = "semantic_cache"
    log.info(f"Qdrant client initialized for {QDRANT_HOST}:{QDRANT_PORT}.")
except Exception as e:
    log.critical(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

# --- Google Gemini AI ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in .env file")
    genai.configure(api_key=GOOGLE_API_KEY)
    
    EMBEDDING_MODEL = "text-embedding-004"
    TRANSFORM_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    ROUTER_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    GENERATOR_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    log.info("Google Gemini clients initialized with gemini-2.5-flash.")
except Exception as e:
    log.critical(f"Failed to initialize Gemini: {e}")

# --- Re-ranker Model ---
try:
    # This is a small, fast, and very effective model
    RERANKER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    log.info("Cross-encoder re-ranker model loaded.")
except Exception as e:
    log.critical(f"Failed to load cross-encoder model: {e}")
    RERANKER_MODEL = None

# ==============================================================================
# 3. HELPER FUNCTIONS (Database, Logging, AI)
# ==============================================================================

def load_prompt_template(filename):
    """Reads a text file from the prompts directory."""
    try:
        file_path = os.path.join(PROMPTS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        log.error(f"Error loading prompt file {filename}: {e}")
        # Fallback to a safe default if file is missing
        return "Context: {context_str}\nQuery: {query}\nAnswer:"

# ==============================================================================
# 1. LOAD CONFIGURATION AND SET UP LOGGING
# ==============================================================================


def init_cache_collection():
    """Ensures the cache collection exists in Qdrant."""
    if not qdrant_client: return
    try:
        if not qdrant_client.collection_exists(CACHE_COLLECTION):
            qdrant_client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                )
            )
            log.info(f"Created semantic cache collection: {CACHE_COLLECTION}")
    except Exception as e:
        log.error(f"Failed to init cache collection: {e}")

init_cache_collection()

def check_semantic_cache(query_vector, similarity_threshold=0.95):
    """Checks Qdrant for a similar past query."""
    if not qdrant_client: return None
    
    try:
        response = qdrant_client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_vector, # Parameter is now 'query'
            limit=1,
            score_threshold=similarity_threshold
        )
        
        hits = response.points
        
        if hits:
            cached_response = hits[0].payload
            log.info(f"Cache HIT! Similarity: {hits[0].score}")
            return cached_response
        
        log.debug("Cache MISS.")
        return None
    except Exception as e:
        log.error(f"Cache lookup failed: {e}")
        return None

def save_to_semantic_cache(query_vector, original_query, answer, sources, topic_id):
    """Saves a query and its answer to the cache."""
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
        
        qdrant_client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[point]
        )
        log.debug("Saved response to semantic cache.")
    except Exception as e:
        log.error(f"Failed to save to cache: {e}")

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        log.debug("New MySQL connection established.")
        return conn
    except mysql.connector.Error as err:
        log.error(f"MySQL Connection Error: {err}")
        return None

def log_failed_query(query, failure_type, topic_id=None):
    log.warning(f"Logging failed query. Type: {failure_type}, Query: {query[:100]}...")
    db_conn = None
    try:
        db_conn = get_db_connection()
        if not db_conn:
            log.error("Cannot log failed query, DB connection failed.")
            return

        with db_conn.cursor() as cursor:
            sql = """
            INSERT INTO failed_queries (standalone_query, failure_type, topic_id_routed)
            VALUES (%s, %s, %s)
            """
            val = (query, failure_type, topic_id)
            cursor.execute(sql, val)
            db_conn.commit()
        log.info("Failed query successfully logged to database.")
    except Exception as e:
        log.error(f"Failed to log query to DB: {e}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

def get_inference_topics(db_conn):
    """
    Fetches a simple list of topic_ids where inference = 1.
    Returns: list of strings (e.g., ['hr_policy', 'it_support'])
    """
    topic_ids = []
    
    if not db_conn:
        log.error("Database connection missing for get_inference_topics.")
        return []

    try:
        # We use a standard cursor (not dictionary) to get tuples
        with db_conn.cursor() as cursor:
            sql = "SELECT topic_id FROM topics WHERE inference = 1"
            cursor.execute(sql)
            results = cursor.fetchall()
            
            # results looks like [('topic_a',), ('topic_b',)]
            # We flatten this into a simple list of strings
            topic_ids = [row[0] for row in results]
            
        log.debug(f"Fetched {len(topic_ids)} active inference topics.")
        return topic_ids

    except Exception as e:
        log.error(f"Failed to fetch inference topics: {e}")
        return []

def transform_query(history, query):
    if not history:
        log.debug("No chat history. Using query as-is.")
        return query

    history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('text', '')}" for msg in history])
    
    prompt_template = load_prompt_template("query_rewriter")
    
    try:
        prompt = prompt_template.format(
            history_str=history_str, 
            query=query
        )
    except KeyError as e:
        log.error(f"Prompt template missing key: {e}")
        return "<p>Error in prompt configuration.</p>"
    
    log.debug(f"Sending prompt to AI Query Transformer:\n{prompt}")
    try:
        response = TRANSFORM_MODEL.generate_content(prompt)
        standalone_query = response.text.strip()
        log.info(f"Transformed query: '{standalone_query}'")
        return standalone_query
    except Exception as e:
        log.error(f"Gemini query transformation failed: {e}")
        return query

def get_all_topics(db_conn):
    topics = []
    try:
        with db_conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT topic_id, description, aliases FROM topics")
            topics = cursor.fetchall()
        log.debug(f"Fetched {len(topics)} topics from MySQL.")
    except Exception as e:
        log.error(f"Failed to fetch topics: {e}")
    return topics

def route_query_to_topic(standalone_query, topics):
    topic_list_str = []
    for t in topics:
        entry = f"- {t['topic_id']}: {t['description']}"
        if t.get('aliases'):
            entry += f" (Aliases: {t['aliases']})"
        topic_list_str.append(entry)
    
    topic_string = '\n'.join(topic_list_str)

    prompt_template = load_prompt_template("topic_finder")
    try:
        prompt = prompt_template.format(
            topic_string=topic_string, 
            standalone_query=standalone_query,
        )
    except KeyError as e:
        log.error(f"Prompt template missing key: {e}")
        return "<p>Error in prompt configuration.</p>"
    
    
    log.debug(f"Sending prompt to AI Router:\n{prompt}")
    try:
        response = ROUTER_MODEL.generate_content(prompt)
        topic_id = response.text.strip()
        log.debug(f"AI Router raw response: '{topic_id}'")
        
        if topic_id in [t['topic_id'] for t in topics]:
            log.info(f"Query routed to topic: {topic_id}")
            return topic_id
        else:
            log.warning(f"Query '{standalone_query[:50]}...' did not match any topic. Router responded: {topic_id}")
            return None
    except Exception as e:
        log.error(f"Gemini routing call failed: {e}")
        return None

def embed_query(query):
    log.debug(f"Embedding query: '{query[:50]}...'")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        log.debug("Query successfully embedded.")
        return result['embedding']
    except Exception as e:
        log.error(f"Gemini embedding call failed: {e}")
        return None

def rerank_documents(query, documents, top_n=5):
    if not RERANKER_MODEL:
        log.error("Re-ranker model not loaded. Skipping re-ranking.")
        return documents[:top_n]

    if not documents:
        log.debug("No documents to re-rank.")
        return []

    log.debug(f"Re-ranking {len(documents)} documents...")
    
    pairs = [(query, doc.payload['content']) for doc in documents]
    
    scores = RERANKER_MODEL.predict(pairs)
    
    scored_docs = list(zip(scores, documents))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    top_docs = [doc for score, doc in scored_docs[:top_n]]
    
    log.debug(f"Re-ranking complete. Top 5 scores: {[score for score, doc in scored_docs[:top_n]]}")
    return top_docs

def retrieve_chunks(query, vector, topic_id):
    """
    Retrieves, fuses, and re-ranks documents. 
    Returns a list of dictionaries: {'content': str, 'source': str}
    """
    if not qdrant_client:
        log.error("Qdrant client not available.")
        return [], []

    log.debug(f"Performing Hybrid Query for topic_id='{topic_id}'")
    
    try:
        fused_hits = {}
        
        # 1. Vector Search
        vector_response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=QDRANT_SIZE,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))]
            )
        )
        for hit in vector_response.points:
            fused_hits[hit.id] = hit

        # 2. Keyword Search (Filter)
        keyword_response = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                    models.FieldCondition(key="content", match=models.MatchText(text=query))
                ]
            ),
            limit=QDRANT_SIZE,
            with_payload=True,
            with_vectors=False
        )
        for hit in keyword_response[0]: 
            if hit.id not in fused_hits:
                fused_hits[hit.id] = hit
    
        candidates = list(fused_hits.values())
        if not candidates:
            return [], []
            
        # 3. Re-ranking
        # We handle re-ranking safely; if it fails, we fall back to the original order
        try:
            if RERANKER_MODEL:
                pairs = [(query, doc.payload.get('content', '')) for doc in candidates]
                scores = RERANKER_MODEL.predict(pairs)
                # Sort by score descending
                scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
                top_docs = [doc for score, doc in scored_docs[:RERANK_SIZE]] # Take top 15 candidates
            else:
                top_docs = candidates[:RERANK_SIZE]
        except Exception as e:
            log.error(f"Re-ranking error: {e}")
            top_docs = candidates[:RERANK_SIZE]

        # 4. Format for Output (Rich Objects)
        rich_context = []
        unique_sources = []
        seen_files = set()

        for doc in top_docs:
            payload = doc.payload
            content = payload.get("content", "").strip()
            source_file = payload.get("source_file", "Unknown File")
            
            # Add to context list
            rich_context.append({
                "content": content,
                "source": source_file
            })

            # Add to sources list (for the UI)
            if source_file not in seen_files:
                unique_sources.append({"file": source_file})
                seen_files.add(source_file)
        
        log.info(f"Retrieved {len(rich_context)} chunks from {len(unique_sources)} files.")
        log.debug(f"retrived text: {rich_context} ")
        return rich_context, unique_sources

    except Exception as e:
        log.error(f"Search pipeline failed: {e}", exc_info=True)
        return [], []

def generate_answer(query, rich_context, topic_id):
    """
    Constructs prompt with Source Headers and enforces Character Limit.
    """
    formatted_chunks = []
    current_char_count = 0
    
    # 1. Format and Limit Context
    for item in rich_context:
        # Create a header like: [Source: employee_handbook.pdf]
        # This allows the LLM to cite the specific file.
        chunk_str = f"[Source: {item['source']}]\n{item['content']}\n\n"
        
        chunk_len = len(chunk_str)
        
        if current_char_count + chunk_len < MAX_CONTEXT_CHARS:
            formatted_chunks.append(chunk_str)
            current_char_count += chunk_len
        else:
            log.info(f"Context limit ({MAX_CONTEXT_CHARS}) reached. Dropped remaining chunks.")
            break
            
    final_context_str = "".join(formatted_chunks)
    
    # 2. Prepare Prompt
    prompt_template = load_prompt_template("generate_answer")
    
    # Fetch extra topic info if needed
    db_conn = get_db_connection()
    inference_topics = get_inference_topics(db_conn)
    inference_topics_str = "- " + "\n- ".join(inference_topics)
    if db_conn: db_conn.close()

    try:
        prompt = prompt_template.format(
            context_str=final_context_str, 
            query=query,
            inference_topics=inference_topics_str,
            topic=topic_id
        )
    except KeyError as e:
        log.error(f"Prompt template missing key: {e}")
        return "Error in prompt configuration."
    
    log.info(f"Sending prompt to AI (Length: {len(prompt)} chars)")
    log.debug(f"Prompt:\n\n{prompt}")
    
    # 3. Generate
    try:
        response = GENERATOR_MODEL.generate_content(prompt)
        answer = response.text.strip()
        
        if "could not find an answer" in answer.lower():
            log_failed_query(query, "NO_ANSWER_IN_CONTEXT", topic_id)
            
        return answer
    except Exception as e:
        log.error(f"Gemini generation call failed: {e}")
        return "I'm sorry, but I encountered an error while generating a response."
    

# ==============================================================================
# 4. FLASK API ENDPOINT
# ==============================================================================

@app.route("/chat", methods=["POST"])
def chat_handler():
    log.info("Received new /chat request.")
    
    try:
        data = request.json
        log.debug(f"Raw request data: {data}")
    except Exception as e:
        log.warning(f"Bad request: Could not parse JSON. Error: {e}")
        return jsonify({"error": "Bad Request", "message": "<p>Invalid JSON format</p>"}), 400

    query = data.get("query")
    history = data.get("history", [])
    
    if not query:
        log.warning("Bad request: 'query' not found in JSON.")
        return jsonify({"error": "Bad Request", "message": "<p>JSON body must contain 'query'</p>"}), 400
    
    log.info(f"Processing query: '{query}' with {len(history)} history messages.")

    db_conn = None
    try:
        db_conn = get_db_connection()
        if not db_conn:
            return jsonify({"error": "Internal Server Error", "message": "<p>Could not connect to database</p>"}), 500

        standalone_query = transform_query(history, query)
        
        query_vector = embed_query(standalone_query)
        if not query_vector:
            return jsonify({"error": "Internal Server Error", "message": "<p>Failed to process query (embedding).</p>"}), 500

        cached_response = check_semantic_cache(query_vector)
        if cached_response:
            return jsonify({
                "answer": cached_response['answer'],
                "sources": cached_response['sources'],
                "topic": cached_response['topic'],
                "cached": True
            }), 200

        topics = get_all_topics(db_conn)
        if not topics:
            log.error("No topics found in database. Ingestion pipeline must be run first.")
            return jsonify({"error": "Internal Server Error", "message": "<p>System not configured, no topics found.</p>"}), 500

        topic_id = route_query_to_topic(standalone_query, topics)
        if not topic_id:
            log.warning(f"Routing failed for query: '{standalone_query}'")
            log_failed_query(standalone_query, "NO_TOPIC_FOUND")
            return jsonify({
                "error": "No matching topic found",
                "message": "<p>I'm sorry, I don't have any documents related to that topic. Please rephrase your question.</p>"
            }), 404

        context, sources = retrieve_chunks(standalone_query, query_vector, topic_id)
        if not context:
            log.warning(f"No context found in Qdrant for topic '{topic_id}' and query '{standalone_query}'")
            log_failed_query(standalone_query, "NO_ANSWER_FOUND", topic_id)
            return jsonify({
                "error": "Answer not found in context",
                "message": f"<p>I found documents for '{topic_id}', but I could not find a specific answer to your question in them.</p>",
                "topic": topic_id
            }), 404

        answer = generate_answer(standalone_query, context, topic_id)
        
        if "could not find an answer" not in answer.lower():
            save_to_semantic_cache(query_vector, standalone_query, answer, sources, topic_id)
        
        response_payload = {
            "answer": answer,
            "sources": sources,
            "topic": topic_id
        }
        log.debug(f"Sending final response: {response_payload}")
        log.info(f"Successfully answered query for topic {topic_id}.")
        return jsonify(response_payload), 200

    except Exception as e:
        log.error(f"Unhandled exception in /chat: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": f"<p>{str(e)}</p>"}), 500
        
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            log.debug("MySQL connection closed.")

# --- NEW: Feedback Endpoint ---
@app.route("/feedback", methods=["POST"])
def feedback_handler():
    log.info("Received new /feedback request.")
    
    try:
        data = request.json
        # Extract data
        query = data.get("query")
        answer = data.get("answer")
        topic_id = data.get("topic_id")
        rating = data.get("rating") # 1 (like) or -1 (dislike)
        history = data.get("history", []) # Contextual history
        
        if not query or not answer or rating is None:
            log.warning("Bad feedback request: Missing fields.")
            return jsonify({"error": "Bad Request", "message": "Missing required fields"}), 400

        # Prepare history as string for DB storage
        history_json = json.dumps(history)

        db_conn = get_db_connection()
        if not db_conn:
             return jsonify({"error": "Internal Server Error", "message": "Database connection failed"}), 500

        try:
            with db_conn.cursor() as cursor:
                sql = """
                INSERT INTO chat_feedback (user_query, ai_response, topic_id, rating, chat_history)
                VALUES (%s, %s, %s, %s, %s)
                """
                val = (query, answer, topic_id, rating, history_json)
                cursor.execute(sql, val)
                db_conn.commit()
            
            log.info(f"Feedback saved. Rating: {rating} for topic: {topic_id}")
            return jsonify({"status": "success", "message": "Feedback received"}), 200

        except Exception as e:
            log.error(f"Database error saving feedback: {e}")
            return jsonify({"error": "Internal Server Error", "message": "Failed to save feedback"}), 500
        finally:
            if db_conn and db_conn.is_connected():
                db_conn.close()

    except Exception as e:
        log.error(f"Unhandled exception in /feedback: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
# --- END NEW ---

if __name__ == "__main__":
    log.info("Starting Flask RAG API server...")
    app.run(host="0.0.0.0", port=5000)