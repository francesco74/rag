import os
import shutil
import time
import logging
import uuid
import pathlib
import mysql.connector
from dotenv import load_dotenv

# Image & PDF Processing
import fitz  # PyMuPDF
from PIL import Image

# Google Gemini (GenAI)
import google.generativeai as genai

# Vector DB
from qdrant_client import QdrantClient, models

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# 1. CONFIGURATION & SETUP
# ==============================================================================

# Load environment variables
load_dotenv()

# Base directory setup
BASE_DIR = pathlib.Path(__file__).parent.resolve()
OCR_LIMIT_SLEEP = 30

# Logging Configuration
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("IngestWorker")

# Define Folder Paths
WATCH_FOLDER = BASE_DIR / os.environ.get("WATCH_FOLDER", "./watch_folders/")
PROCESSED_FOLDER = BASE_DIR / os.environ.get("PROCESSED_FOLDER", "./processed_folders/")
ERROR_FOLDER = BASE_DIR / os.environ.get("ERROR_FOLDER", "./error_folders/")

# Create directories if they don't exist
for folder in [WATCH_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        log.info(f"Created directory: {folder}")

log.info(f"Monitoring: {WATCH_FOLDER}")

# --- Google AI Configuration ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing in .env file")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Models
    OCR_MODEL_NAME = "gemini-2.5-flash"  # Fast, cheap, multimodal (reads images)
    EMBEDDING_MODEL_NAME = "text-embedding-004"
    
    log.info(f"Google Clients initialized. OCR: {OCR_MODEL_NAME}, Embed: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    log.critical(f"Failed to initialize Google AI: {e}")
    exit(1)

# --- Qdrant Configuration ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "document_chunks"

try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    log.info(f"Qdrant client configured for {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    log.critical(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

# --- MySQL Configuration ---
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME", "rag_system")
}

# --- Text Splitter ---
# Uses tiktoken (cl100k_base) to match modern LLM tokenization
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", 
    chunk_size=800, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_db_connection():
    """Establishes a MySQL connection."""
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        log.error(f"MySQL Connection Error: {err}")
        return None

def extract_text_from_image(image_input, is_file_path=True):
    """
    Uses Gemini 1.5 Flash to extract text from an image.
    This handles handwriting, tables, and full pages effectively.
    """
    try:
        # Load image
        if is_file_path:
            img = Image.open(image_input)
        else:
            img = image_input  # It's already a PIL Image object

        log.debug(f"Sending image to Gemini ({OCR_MODEL_NAME}) for OCR...")

        # Prompt for pure extraction
        prompt = (
            "Transcribe the text in this image exactly as it appears. "
            "Maintain the structure (lists, headers, numbering). "
            "If text is illegible, write [unreadable]. "
            "Do not add any conversational text or markdown code blocks."
        )

        model = genai.GenerativeModel(OCR_MODEL_NAME)
        response = model.generate_content([prompt, img])
        
        text = response.text.strip()
        
        if not text:
            log.warning("Gemini returned empty text for image.")
            return []

        log.debug(f"Gemini extracted {len(text)} characters.")
        # Return format consistent with PDF extractor: [(page_num, text)]
        return [(1, text)]

    except Exception as e:
        log.error(f"Gemini OCR Failed: {e}")
        # Simple rate limit handling
        if "429" in str(e):
            log.warning(f"Rate limit hit. Sleeping for {OCR_LIMIT_SLEEP} seconds...")
            time.sleep(OCR_LIMIT_SLEEP)
        return []

def extract_text_from_pdf(file_path):
    """
    Extracts text from PDF. Checks for digital text first.
    If text is sparse (scanned), renders page to image and uses Gemini OCR.
    """
    log.debug(f"Processing PDF: {file_path}")
    text_by_page = []
    
    try:
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                # 1. Try extracting digital text
                page_text = page.get_text(sort=True).strip()
                
                # 2. Heuristic: If < 50 chars, assume it's a scanned image
                if len(page_text) < 50:
                    log.info(f"Page {page_num+1}: Text sparse ({len(page_text)} chars). Using Gemini Vision OCR.")
                    
                    # Render page to image at 200 DPI
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Send to Gemini
                    ocr_results = extract_text_from_image(img, is_file_path=False)
                    if ocr_results:
                        page_text = ocr_results[0][1]
                    else:
                        log.warning(f"Page {page_num+1}: OCR returned no text.")
                
                if page_text:
                    text_by_page.append((page_num + 1, page_text))
                    
        return text_by_page
    except Exception as e:
        log.error(f"Failed to read PDF {file_path}: {e}")
        return []

def register_topic_in_mysql(topic_id, db_conn):
    """Registers the topic in MySQL."""
    description = f"Documents related to {topic_id.replace('_', ' ').title()}"
    sql = """
    INSERT INTO topics (topic_id, description) 
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE description = VALUES(description)
    """
    try:
        with db_conn.cursor() as cursor:
            cursor.execute(sql, (topic_id, description))
            db_conn.commit()
        log.info(f"Topic '{topic_id}' synced in MySQL.")
        return True
    except mysql.connector.Error as err:
        log.error(f"MySQL Topic Register Error: {err}")
        return False

# ==============================================================================
# 3. CORE PROCESSING LOGIC
# ==============================================================================

def process_folder(topic_id, folder_path, db_conn):
    """Reads all files in a topic folder, chunks them, and indexes them."""
    log.info(f"--- Processing Topic: {topic_id} ---")
    
    all_chunks_to_embed = []
    all_metadata = []
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if not files:
        log.warning("Folder is empty.")
        return False

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        log.info(f"Reading file: {filename}")
        
        extracted_pages = []
        
        # Route by file type
        if filename.lower().endswith(".pdf"):
            extracted_pages = extract_text_from_pdf(file_path)
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            extracted_pages = extract_text_from_image(file_path, is_file_path=True)
        else:
            log.warning(f"Skipping unsupported file type: {filename}")
            continue

        if not extracted_pages:
            log.warning(f"No text found in {filename}")
            continue

        # Combine all text for "Whole Document" context
        full_text = "\n\n".join([txt for _, txt in extracted_pages])
        
        # Chunking
        chunks = text_splitter.split_text(full_text)
        log.info(f"Split {filename} into {len(chunks)} chunks.")
        log.debug(f"Full text:\n\n{full_text}")

        for i, chunk in enumerate(chunks):
            all_chunks_to_embed.append(chunk)
            all_metadata.append({
                "topic_id": topic_id,
                "source_file": filename,
                "chunk_index": i,
                "content": chunk
            })

    if not all_chunks_to_embed:
        log.warning(f"No chunks to index for topic {topic_id}.")
        return False

    # --- Embedding (Batch Process) ---
    log.info(f"Embedding {len(all_chunks_to_embed)} chunks...")
    embeddings = []
    batch_size = 100 # Gemini embedding limit
    
    try:
        for i in range(0, len(all_chunks_to_embed), batch_size):
            batch = all_chunks_to_embed[i:i+batch_size]
            result = genai.embed_content(
                model=EMBEDDING_MODEL_NAME,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.extend(result['embedding'])
            time.sleep(0.5) # Rate limit safety
    except Exception as e:
        log.error(f"Embedding failed: {e}")
        return False

    # --- Indexing (Qdrant) ---
    log.info(f"Upserting {len(embeddings)} vectors to Qdrant...")
    points = []
    for i in range(len(embeddings)):
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload=all_metadata[i]
        ))
    
    try:
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        log.info("Qdrant Indexing Complete.")
    except Exception as e:
        log.error(f"Qdrant upsert failed: {e}")
        return False

    # --- MySQL Registration ---
    return register_topic_in_mysql(topic_id, db_conn)

def initialize_qdrant_collection():
    """Create Qdrant collection if it doesn't exist."""
    if not qdrant_client: return
    try:
        if not qdrant_client.collection_exists(QDRANT_COLLECTION):
            log.info(f"Creating collection '{QDRANT_COLLECTION}'...")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            # Create indexes
            qdrant_client.create_payload_index(QDRANT_COLLECTION, "topic_id", models.PayloadSchemaType.KEYWORD)
            qdrant_client.create_payload_index(QDRANT_COLLECTION, "content", models.PayloadSchemaType.TEXT)
            log.info("Collection initialized.")
    except Exception as e:
        log.error(f"Qdrant Init Error: {e}")

def move_files(src_folder, base_dest_folder, topic_id):
    """Moves processed files to destination structure."""
    dest_folder = os.path.join(base_dest_folder, topic_id)
    os.makedirs(dest_folder, exist_ok=True)
    
    for filename in os.listdir(src_folder):
        src = os.path.join(src_folder, filename)
        dst = os.path.join(dest_folder, filename)
        if os.path.isfile(src):
            try:
                shutil.move(src, dst)
            except Exception as e:
                log.error(f"Failed to move {filename}: {e}")

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================

def main():
    log.info("--- Ingestion Service Started ---")
    
    if not qdrant_client:
        log.critical("No Qdrant connection. Exiting.")
        return

    initialize_qdrant_collection()

    while True:
        try:
            # Check DB connection
            db_conn = get_db_connection()
            if not db_conn:
                log.warning("Database unavailable. Retrying in 30s...")
                time.sleep(30)
                continue

            # Scan Watch Folder
            folders = [f for f in os.listdir(WATCH_FOLDER) if os.path.isdir(os.path.join(WATCH_FOLDER, f))]
            
            for topic_id in folders:
                folder_path = os.path.join(WATCH_FOLDER, topic_id)
                
                # Check if folder has files
                if not os.listdir(folder_path):
                    continue

                log.info(f"Found new topic folder: {topic_id}")
                
                success = process_folder(topic_id, folder_path, db_conn)
                
                if success:
                    log.info(f"Topic '{topic_id}' success. Moving to Processed.")
                    move_files(folder_path, PROCESSED_FOLDER, topic_id)
                    # Remove empty source dir
                    os.rmdir(folder_path)
                else:
                    log.error(f"Topic '{topic_id}' failed. Moving to Error.")
                    move_files(folder_path, ERROR_FOLDER, topic_id)
                    os.rmdir(folder_path)

            if db_conn.is_connected():
                db_conn.close()
            
            # Wait before next scan
            time.sleep(10)

        except KeyboardInterrupt:
            log.info("Stopping service...")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(30)

if __name__ == "__main__":
    main()