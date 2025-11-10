import os
import shutil
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
from qdrant_client import QdrantClient, models
import mysql.connector
import uuid
import time
import logging
from dotenv import load_dotenv
import pathlib

# --- NEW IMPORTS for TrOCR ---
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel # <-- UPDATED
# --- END NEW IMPORTS ---

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# 1. LOAD CONFIGURATION AND SET UP LOGGING
# ==============================================================================
load_dotenv()

# --- THIS IS THE KEY ---
# Get the absolute path of the directory where this script is located
# __file__ is a special variable that holds the path to the current script
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Set log level from environment variable, default to INFO
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Log to console
)
log = logging.getLogger(__name__)

log.info(f"Logging level set to {log_level}")
log.info(f"Script base directory set to: {BASE_DIR}")

# ==============================================================================
# 2. INITIALIZE CLIENTS AND CONSTANTS
# ==============================================================================

# --- Folder Paths (Now Absolute) ---
WATCH_FOLDER = "/data/watch_folders"
PROCESSED_FOLDER = "/data/processed_folders"
ERROR_FOLDER = "/data/error_folders"

# Log the final absolute paths
log.info(f"Monitoring watch folder: {WATCH_FOLDER}")
log.info(f"Moving to processed folder: {PROCESSED_FOLDER}")
log.info(f"Moving to error folder: {ERROR_FOLDER}")

# Create directories if they don't exist
os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(ERROR_FOLDER, exist_ok=True)


# --- Google AI Config ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in .env file")
    genai.configure(api_key=GOOGLE_API_KEY)
    EMBEDDING_MODEL = "text-embedding-004"
    log.info("Google Gemini clients initialized.")
except Exception as e:
    log.critical(f"Failed to initialize Gemini: {e}")
    exit(1) # Exit if no API key

# --- Qdrant Config ---
try:
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    QDRANT_COLLECTION = "document_chunks"
    log.info(f"Qdrant client configured for {QDRANT_HOST}:{QDRANT_PORT}.")
except Exception as e:
    log.critical(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

# --- MySQL Config ---
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME", "rag_system")
}
log.debug(f"MySQL configured for host: {db_config['host']}")

# --- Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, # Overlap helps keep context
    separators=["\n\n", "\n", ". ", " ", ""] # Tries to split on paragraphs first
)
log.info("LangChain RecursiveCharacterTextSplitter initialized.")

# --- TrOCR Model ---
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Initializing TrOCR model on device: {device}")
    # --- UPDATED ---
    trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    # --- END UPDATE ---
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(device)
    log.info("TrOCR model loaded successfully.")
except Exception as e:
    log.critical(f"Failed to load TrOCR model: {e}. Image OCR will fail.")
    trocr_processor = None
    trocr_model = None

# ==============================================================================
# 3. HELPER FUNCTIONS (Parsing, Chunking, DB)
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

def extract_text_from_image(image_input, is_file_path=True):
    """
    Extracts text from an image using TrOCR.
    Input can be a file path or a PIL.Image object.
    Returns a list of tuples: [(1, "full_text")]
    """
    if not trocr_model or not trocr_processor:
        log.error("TrOCR model is not loaded. Cannot process image.")
        return []

    log_debug_msg = f"Extracting text from Image: {image_input if is_file_path else 'PIL object'}"
    log.debug(log_debug_msg)
    
    try:
        if is_file_path:
            image = Image.open(image_input).convert("RGB")
        else:
            # It's already a PIL Image object from the PDF
            image = image_input.convert("RGB") # Ensure it's RGB
        
        # --- UPDATED ---
        # Process image with TrOCR
        # This is the original, correct call that will work
        # now that the processor is loaded correctly.
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values.to(device)
        # --- END UPDATE ---
        
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=2048)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Return in the same format as extract_text_from_pdf
        return [(1, generated_text.strip())] if generated_text.strip() else []
    except Exception as e:
        log_error_msg = f"Failed to extract Image: {image_input if is_file_path else 'PIL object'}"
        log.error(f"{log_error_msg}: {e}", exc_info=True)
        return []

# --- UPGRADED: PDF Function ---
def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF.
    - Checks for digital fonts. If found, extracts digital text.
    - If no fonts are found, renders the page as an image and uses TrOCR.
    Returns a list of tuples: [(page_num, "page_text"), ...]
    """
    log.debug(f"Extracting text from PDF: {file_path}")
    text_by_page = []
    
    try:
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = ""
                
                # --- HEURISTIC ---
                # Check if the page has any embedded fonts.
                # If it has fonts, it's a digital page.
                # If it has no fonts, it's a scanned image.
                fonts = page.get_fonts(full=False)
                
                if fonts:
                    # 1. Digital Page: Extract text normally.
                    # This will correctly handle short pages.
                    log.debug(f"Page {page_num+1}: Found {len(fonts)} digital fonts. Extracting digital text.")
                    page_text = page.get_text(sort=True).strip()
                else:
                    # 2. Scanned Page: Fallback to OCR
                    log.warning(f"Page {page_num+1}: No digital fonts found. Running OCR...")
                    
                    # Render page to image
                    pix = page.get_pixmap(dpi=200) # Render at 200 DPI for better OCR
                    mode = "RGB" if pix.n == 3 else "RGBA"
                    
                    # Convert to PIL Image
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    
                    # Call TrOCR function (passing the PIL object)
                    ocr_results = extract_text_from_image(img, is_file_path=False)
                    
                    if ocr_results:
                        ocr_text = ocr_results[0][1] # Get text from (1, "text")
                        log.debug(f"Page {page_num+1}: OCR extracted {len(ocr_text)} chars.")
                        log.debug(f"{ocr_text}")
                        page_text = ocr_text
                    else:
                        log.warning(f"Page {page_num+1}: OCR failed to extract text.")
                # --- END HEURISTIC ---
                
                if page_text:
                    text_by_page.append((page_num + 1, page_text))
                    
        return text_by_page
    except Exception as e:
        log.error(f"Failed to extract PDF {file_path}: {e}", exc_info=True)
        return []

def chunk_text(text):
    """Splits text using the recursive splitter."""
    return text_splitter.split_text(text)

def register_topic_in_mysql(topic_id, db_conn):
    """
    Adds/Updates the topic in the MySQL registry.
    It only inserts/updates the topic_id and description, leaving
    the 'aliases' column to be managed manually in the database.
    """
    log.debug(f"Registering topic '{topic_id}' in MySQL.")
    
    # Create default description from folder name
    description = f"Documents related to {topic_id.replace('_', ' ').title()}"
    
    sql = """
    INSERT INTO topics (topic_id, description) 
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE 
        description = VALUES(description)
    """
    val = (topic_id, description) 
    
    try:
        with db_conn.cursor() as cursor:
            cursor.execute(sql, val)
            db_conn.commit()
        log.info(f"Topic '{topic_id}' registered in MySQL (aliases column untouched).")
        return True
    except mysql.connector.Error as err:
        log.error(f"MySQL error registering topic '{topic_id}': {err}")
        return False

# ==============================================================================
# 4. CORE INGESTION LOGIC
# ==============================================================================

def process_folder(topic_id, folder_path, db_conn):
    """Processes all files in a single topic folder."""
    log.info(f"Processing topic: {topic_id}")
    
    all_chunks_to_embed = []
    all_metadata = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        text_pages = []
        
        if filename.lower().endswith(".pdf"):
            text_pages = extract_text_from_pdf(file_path) # Calls the upgraded function
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # Call TrOCR directly
            text_pages = extract_text_from_image(file_path, is_file_path=True)
        else:
            log.warning(f"Skipping unsupported file: {filename}")
            continue

        if not text_pages:
            log.warning(f"No text extracted from: {filename}")
            continue

        # --- "Whole Document" Strategy ---
        # 1. Concatenate all page text into one giant string
        full_document_text = "\n\n".join([page_text for page_num, page_text in text_pages])
        
        if not full_document_text:
            log.warning(f"No text extracted from {filename} after concatenation.")
            continue
            
        # 2. Chunk the *entire* document text at once
        chunks = chunk_text(full_document_text)
        
        # 3. Store chunks with a "chunk index" instead of a page number
        for i, chunk in enumerate(chunks):
            all_chunks_to_embed.append(chunk)
            all_metadata.append({
                "topic_id": topic_id,
                "source_file": filename,
                "page": i + 1, # This is now a "chunk index" not a page number
                "content": chunk
            })

    if not all_chunks_to_embed:
        log.warning(f"No text chunks generated for topic {topic_id}.")
        return False

    # --- Step 1: Embed Chunks (Google AI) ---
    log.info(f"Embedding {len(all_chunks_to_embed)} chunks for {topic_id}...")
    try:
        # Note: Gemini has a limit on batch embedding,
        # 1000 chunks is likely too many. We should batch this.
        
        embeddings = []
        batch_size = 100 # Gemini batch limit is 100 for text-embedding-004
        
        for i in range(0, len(all_chunks_to_embed), batch_size):
            batch_chunks = all_chunks_to_embed[i:i+batch_size]
            log.debug(f"Embedding batch {i//batch_size + 1}...")
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_chunks,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.extend(result['embedding'])
            time.sleep(1) # Simple rate limiting
            
    except Exception as e:
        log.error(f"Gemini embedding call failed: {e}")
        return False

    # --- Step 2: Index Chunks (Qdrant) ---
    log.debug(f"Preparing {len(embeddings)} points for Qdrant.")
    if len(embeddings) != len(all_metadata):
        log.error(f"Mismatch between embeddings ({len(embeddings)}) and metadata ({len(all_metadata)})")
        return False
        
    points = []
    for i in range(len(embeddings)):
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload=all_metadata[i]
        ))

    try:
        # --- FIX: Removed batch_size=128 ---
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True
        )
        # --- END FIX ---
# ... existingK code ...
    except Exception as e:
        log.error(f"Qdrant upsert failed: {e}")
        return False

    # --- Step 3: Register Topic (MySQL) ---
    return register_topic_in_mysql(topic_id, db_conn)


def initialize_qdrant_collection():
    """Ensures the Qdrant collection and indexes exist."""
    try:
        collection_exists = qdrant_client.collection_exists(
            collection_name=QDRANT_COLLECTION
        )
        
        if not collection_exists:
            log.info(f"Collection '{QDRANT_COLLECTION}' not found. Creating...")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=768,  # Size of Google's text-embedding-004
                    distance=models.Distance.COSINE
                ),
            )
            log.info(f"Collection '{QDRANT_COLLECTION}' created.")

        # Index for filtering
        qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="topic_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        log.info(f"Payload index for 'topic_id' ensured.")
        
        # Index for keyword search
        qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="content",
            field_schema=models.PayloadSchemaType.TEXT
        )
        log.info(f"Payload index for 'content' (keyword search) ensured.")

    except Exception as e:
        log.error(f"Error during Qdrant initialization: {e}", exc_info=True)


# ==============================================================================
# 5. MAIN EXECUTION LOOP
# ==============================================================================
def move_files_in_folder(src_folder_path, dest_base_folder, topic_id):
    """
    Moves all files from src_folder_path to a new subfolder in dest_base_folder,
    re-creating the topic folder structure.
    e.g., /watch_folder/topic_A/file.pdf -> /processed_folders/topic_A/file.pdf
    """
    dest_topic_folder = os.path.join(dest_base_folder, topic_id)
    
    # Ensure the destination topic folder exists
    try:
        os.makedirs(dest_topic_folder, exist_ok=True)
    except Exception as e:
        log.error(f"Could not create destination directory {dest_topic_folder}: {e}")
        return

    files_to_move = os.listdir(src_folder_path)
    if not files_to_move:
        log.warning(f"No files to move in {src_folder_path}.")
        return

    log.info(f"Moving {len(files_to_move)} file(s) from {src_folder_path} to {dest_topic_folder}...")

    for filename in files_to_move:
        src_file = os.path.join(src_folder_path, filename)
        dest_file = os.path.join(dest_topic_folder, filename)
        
        if os.path.isfile(src_file):
            try:
                shutil.move(src_file, dest_file)
                log.debug(f"Moved file {filename} to {dest_topic_folder}")
            except Exception as e:
                log.error(f"Failed to move file {src_file}: {e}")
        else:
            log.warning(f"Skipping non-file item: {filename} in {src_folder_path}")


def main():
    """Main loop to monitor the watch folder."""
    log.info("Starting ingestion pipeline... (Press Ctrl+C to stop)")
    
    if not qdrant_client:
        log.critical("Qdrant connection failed. Exiting.")
        return
        
    if not trocr_model or not trocr_processor:
        log.warning("TrOCR model not loaded. Scanned PDF/image processing will fail.")

    initialize_qdrant_collection()
    
    while True:
        try:
            db_conn = get_db_connection()
            if not db_conn:
                log.error("Failed to connect to MySQL. Retrying in 60s...")
                time.sleep(60)
                continue

            for folder_name in os.listdir(WATCH_FOLDER):
                folder_path = os.path.join(WATCH_FOLDER, folder_name)
                
                if os.path.isdir(folder_path):
                    topic_id = folder_name
                    
                    # Check if the folder is empty
                    if not os.listdir(folder_path):
                        log.debug(f"Topic folder {folder_name} is empty. Skipping.")
                        continue 

                    log.info(f"Found files in topic folder: {folder_name}")
                    
                    try:
                        success = process_folder(topic_id, folder_path, db_conn)
                        
                        if success:
                            log.info(f"Successfully processed topic '{topic_id}'. Moving files...")
                            move_files_in_folder(folder_path, PROCESSED_FOLDER, topic_id)
                        else:
                            log.error(f"Failed to process '{folder_name}'. Moving files to error folder.")
                            move_files_in_folder(folder_path, ERROR_FOLDER, topic_id)
                            
                    except Exception as e:
                        log.critical(f"!! CRITICAL ERROR processing {folder_name}: {e}", exc_info=True)
                        try:
                            log.error(f"Moving files to error folder due to critical error.")
                            move_files_in_folder(folder_path, ERROR_FOLDER, topic_id)
                        except Exception as move_e:
                            log.error(f"Could not move error files for {folder_name}: {move_e}")

            if db_conn and db_conn.is_connected():
                db_conn.close()
                log.debug("MySQL connection closed.")

            log.debug("Waiting for new files...")
            time.sleep(60) # Scan every 60 seconds

        except KeyboardInterrupt:
            log.info("Shutdown signal received. Exiting.")
            break
        except Exception as e:
            log.error(f"Unhandled error in main loop: {e}", exc_info=True)
            time.sleep(60) # Wait before retrying loop

if __name__ == "__main__":
    main()