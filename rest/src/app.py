import os
import logging
import json
from dotenv import load_dotenv

# --- Flask and DB Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import mysql.connector
from qdrant_client import QdrantClient, models

# --- Google AI ---
import google.generativeai as genai

# --- Re-ranking ---
from sentence_transformers.cross_encoder import CrossEncoder

# ==============================================================================
# 1. LOAD CONFIGURATION AND SET UP LOGGING
# ==============================================================================
load_dotenv()

# Set log level from environment variable, default to INFO
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Log to console
)
log = logging.getLogger("app")  # Give the logger a name

log.info(f"Logging level set to {log_level}")

# ==============================================================================
# 2. INITIALIZE CLIENTS AND CONSTANTS
# ==============================================================================

# --- App Initialization ---
app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes
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

def get_db_connection():
    """Establishes a new MySQL connection."""
    try:
        conn = mysql.connector.connect(**db_config)
        log.debug("New MySQL connection established.")
        return conn
    except mysql.connector.Error as err:
        log.error(f"MySQL Connection Error: {err}")
        return None

def log_failed_query(query, failure_type, topic_id=None):
    """Logs a failed query to the MySQL database."""
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

def transform_query(history, query):
    """(LLM Call 1) Transform conversational query to standalone query."""
    if not history:
        log.debug("No chat history. Using query as-is.")
        return query

    # Format history
    history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('text', '')}" for msg in history])
    
    prompt = f"""
    <role>
    You are a query re-writer.
    </role>
    <instructions>
    Look at the <chat_history> (which is in plain text) and the user's <last_query>.
    Your goal is to re-write the <last_query> as a single, standalone query that is optimal for a vector database search.
    - If the <last_query> is already a good standalone query, just return it.
    - If the <last_query> is conversational (e.g., "Why?", "Tell me more"), use the <chat_history> to create a new, self-contained query.
    - Respond with ONLY the re-written query and nothing else.
    </instructions>
    
    <chat_history>
    {history_str}
    </chat_history>
    
    <last_query>
    {query}
    </last_query>

    Standalone Query:
    """
    
    log.debug(f"Sending prompt to AI Query Transformer:\n{prompt}")
    try:
        response = TRANSFORM_MODEL.generate_content(prompt)
        standalone_query = response.text.strip()
        log.info(f"Transformed query: '{standalone_query}'")
        return standalone_query
    except Exception as e:
        log.error(f"Gemini query transformation failed: {e}")
        return query # Fallback to original query

def get_all_topics(db_conn):
    """Fetches all topics and their aliases from the MySQL topic registry."""
    topics = []
    try:
        with db_conn.cursor(dictionary=True) as cursor:
            # Fetch topic, description, and aliases
            cursor.execute("SELECT topic_id, description, aliases FROM topics")
            topics = cursor.fetchall()
        log.debug(f"Fetched {len(topics)} topics from MySQL.")
    except Exception as e:
        log.error(f"Failed to fetch topics: {e}")
    return topics

def route_query_to_topic(standalone_query, topics):
    """(LLM Call 2) Use Gemini to classify the standalone query."""
    
    # Format the topic list with aliases
    topic_list_str = []
    for t in topics:
        entry = f"- {t['topic_id']}: {t['description']}"
        if t.get('aliases'):
            entry += f" (Aliases: {t['aliases']})"
        topic_list_str.append(entry)
    
    topic_string = '\n'.join(topic_list_str)

    prompt = f"""
    <role>
    You are a topic finder.
    </role>
    <instructions>
    Given the user <query>, which of the following <topics> is it about?
    - Analyze the query, description, and aliases.
    - Respond with ONLY the topic_id.
    - If none match, respond with 'NONE'.
    </instructions>

    <topics>
    {topic_string}
    </topics>
    
    <query>
    {standalone_query}
    </query>

    Topic:
    """
    
    log.debug(f"Sending prompt to AI Router:\n{prompt}")
    try:
        response = ROUTER_MODEL.generate_content(prompt)
        topic_id = response.text.strip()
        log.debug(f"AI Router raw response: '{topic_id}'")
        
        # Validate response
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
    """Embed the user's query for vector search."""
    log.debug(f"Embedding query: '{query[:50]}...'")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY" # Query, not Document
        )
        log.debug("Query successfully embedded.")
        return result['embedding']
    except Exception as e:
        log.error(f"Gemini embedding call failed: {e}")
        return None

def rerank_documents(query, documents, top_n=5):
    """Re-ranks a list of documents against a query."""
    if not RERANKER_MODEL:
        log.error("Re-ranker model not loaded. Skipping re-ranking.")
        # Fallback: just return the first N documents
        return documents[:top_n]

    if not documents:
        log.debug("No documents to re-rank.")
        return []

    log.debug(f"Re-ranking {len(documents)} documents...")
    
    # The cross-encoder expects pairs of [query, document_content]
    pairs = [(query, doc.payload['content']) for doc in documents]
    
    # Predict scores
    scores = RERANKER_MODEL.predict(pairs)
    
    # Combine documents with their new scores
    scored_docs = list(zip(scores, documents))
    
    # Sort by score in descending order
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return the top_n documents (not the scores, just the docs)
    top_docs = [doc for score, doc in scored_docs[:top_n]]
    
    log.debug(f"Re-ranking complete. Top 5 scores: {[score for score, doc in scored_docs[:top_n]]}")
    return top_docs

# ---
# THIS IS THE FULLY CORRECTED FUNCTION
# ---
def retrieve_chunks(query, vector, topic_id):
    """Full Hybrid Search (2-step) + Re-ranking pipeline."""
    if not qdrant_client:
        log.error("Qdrant client not available.")
        return [], []

    log.debug(f"Performing 2-Step Hybrid Query for topic_id='{topic_id}'")
    
    try:
        fused_hits = {}
        
        # 1. Vector Search
        log.debug("Step 1: Vector Search")
        vector_hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=20,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))
                ]
            )
        )
        for hit in vector_hits:
            fused_hits[hit.id] = hit
        log.debug(f"Vector search found {len(vector_hits)} hits.")

        # ---
        # 2. Keyword Search
        # This is the most reliable way to do a keyword search.
        # We use a `Must` condition with a `MatchText` filter.
        # This is a *filter*, not a *search*, but it will find exact keywords.
        # ---
        log.debug("Step 2: Keyword Search (Filter)")
        keyword_hits = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                    models.FieldCondition(key="content", match=models.MatchText(text=query))
                ]
            ),
            limit=20, # Get 20 keyword results
            with_vectors=False,
            with_payload=True
        )[0] # scroll returns (points, next_offset)
        
        for hit in keyword_hits:
            fused_hits[hit.id] = hit # Add/overwrite in our dictionary to de-duplicate
        log.debug(f"Keyword filter found {len(keyword_hits)} hits.")
    
        # 3. Fusion
        fused_documents = list(fused_hits.values())
        log.debug(f"Hybrid search retrieved {len(fused_documents)} unique candidates.")
        
        if not fused_documents:
            return [], []
            
        # 4. Fine-Grained Re-ranking
        reranked_docs = rerank_documents(query, fused_documents, top_n=7)

        # 5. Format for output
        context_chunks = []
        sources = []
        for doc in reranked_docs:
            payload = doc.payload
            context_chunks.append(payload["content"])
            # --- THIS IS THE FIX ---
            # We only append the file, not the page/chunk index,
            # so that de-duplication works correctly.
            sources.append({
                "file": payload["source_file"]
            })
            # --- END FIX ---
            log.debug(f"  > Finalist chunk from {payload['source_file']}")
                
        # De-duplicate sources
        # This will now correctly de-duplicate based on filename only
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        
        log.debug(f"Retrieved {len(context_chunks)} re-ranked chunks. Unique sources: {len(unique_sources)}")
        return context_chunks, unique_sources

    except Exception as e:
        log.error(f"Qdrant hybrid search failed: {e}", exc_info=True)
        return [], []
# ---
# END OF CORRECTED FUNCTION
# ---

def generate_answer(query, context, topic_id):
    """(LLM Call 3) Use Gemini to generate an answer from context."""
    context_str = "\n\n".join(context)
    
    prompt = f"""
    <role>
    You are a helpful assistant.
    </role>
    <instructions>
    - Answer the user's <query> based ONLY on the provided <context>.
    
    - **IMPORTANT RULE:** If a piece of context appears to be a Table of Contents, an Index, or a reference to another page (e.g., '...see page 25'), you MUST ignore that piece of context and find the answer in the *other* context passages.
    - If the *only* context you are given is a Table of Contents or page references, you MUST act as if you found no answer.

    - Format your answer in simple HTML (using <p>, <ul>, <li>, and <b> tags).
    - Do NOT use markdown.
    - If the answer is not in the context (or you were forced to ignore it), you MUST respond with only:
    "<p>I could not find an answer in the provided documents."
    - Do NOT repeat the query in your answer.
    - Begin your response *directly* with the answer (e.g., "<p>The answer is...").
    </instructions>
    
    <context>
    {context_str}
    </context>
    
    <query>
    {query}
    </query>

    Answer:
    """
    
    log.debug(f"Sending prompt to AI Generator (Context length: {len(context_str)} chars)")
    log.debug(f"{context_str}")
    
    try:
        response = GENERATOR_MODEL.generate_content(prompt)
        answer = response.text.strip()
        log.debug(f"AI Generator raw response: '{answer[:100]}...'")

        # Check if AI failed to find answer
        if "could not find an answer" in answer.lower():
            log_failed_query(query, "NO_ANSWER_IN_CONTEXT", topic_id)
            
        return answer
    except Exception as e:
        log.error(f"Gemini generation call failed: {e}")
        return "<p>I'm sorry, but I encountered an error while generating a response.</p>"

# ==============================================================================
# 4. FLASK API ENDPOINT
# ==============================================================================

@app.route("/chat", methods=["POST"])
def chat_handler():
    log.info("Received new /chat request.")
    
    # --- 1. Get Query and History ---
    try:
        data = request.json
        log.debug(f"Raw request data: {data}")
    except Exception as e:
        log.warning(f"Bad request: Could not parse JSON. Error: {e}")
        return jsonify({"error": "Bad Request", "message": "<p>Invalid JSON format</p>"}), 400

    query = data.get("query")
    history = data.get("history", []) # Default to empty list
    
    if not query:
        log.warning("Bad request: 'query' not found in JSON.")
        return jsonify({"error": "Bad Request", "message": "<p>JSON body must contain 'query'</p>"}), 400
    
    log.info(f"Processing query: '{query}' with {len(history)} history messages.")

    db_conn = None
    try:
        # --- 2. Get DB Connection ---
        db_conn = get_db_connection()
        if not db_conn:
            return jsonify({"error": "Internal Server Error", "message": "<p>Could not connect to database</p>"}), 500

        # --- 3. Step A: Transform Query ---
        standalone_query = transform_query(history, query)

        # --- 4. Step B: Route Query ---
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

        # --- 5. Step C: Retrieve Chunks ---
        query_vector = embed_query(standalone_query)
        if not query_vector:
            return jsonify({"error": "Internal Server Error", "message": "<p>Failed to process query (embedding).</p>"}), 500

        context, sources = retrieve_chunks(standalone_query, query_vector, topic_id)
        if not context:
            log.warning(f"No context found in Qdrant for topic '{topic_id}' and query '{standalone_query}'")
            log_failed_query(standalone_query, "NO_ANSWER_FOUND", topic_id)
            return jsonify({
                "error": "Answer not found in context",
                "message": f"<p>I found documents for '{topic_id}', but I could not find a specific answer to your question in them.</p>",
                "topic": topic_id
            }), 404

        # --- 6. Step D: Generate Answer ---
        answer = generate_answer(standalone_query, context, topic_id)
        
        # --- 7. Send Response ---
        response_payload = {
            "answer": answer,
            "sources": sources,
            "topic": topic_id
        }
        log.debug(f"Sending final response: {response_payload}")
        log.info(f"Successfully answered query for topic {topic_id}.")
        return jsonify(response_payload), 200

    except Exception as e:
        log.error(f"Unhandled exception in /chat: {e}", exc_info=True) # exc_info=True logs stack trace
        return jsonify({"error": "Internal Server Error", "message": f"<p>{str(e)}</p>"}), 500
        
    finally:
        # --- 8. Cleanup ---
        if db_conn and db_conn.is_connected():
            db_conn.close()
            log.debug("MySQL connection closed.")

# ==============================================================================
# 5. RUN THE APPLICATION
# ==============================================================================

if __name__ == "__main__":
    log.info("Starting Flask RAG API server...")
    app.run(host="0.0.0.0", port=5000)