import os
import shutil
import time
import logging
import uuid
import pathlib
import asyncio
import hashlib
import mysql.connector
import numpy as np
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

# Async & Resilience
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# ==============================================================================
# 1. CONFIGURATION & SETUP
# ==============================================================================

CROP_OCR_LIMIT = 50
VARIANCE_LIMIT = 15
SLEEP_TIME = 60

load_dotenv()

BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Logging Configuration
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("IngestWorker")

# Directories
WATCH_FOLDER = BASE_DIR / os.environ.get("WATCH_FOLDER", "./watch_folders/")
PROCESSED_FOLDER = BASE_DIR / os.environ.get("PROCESSED_FOLDER", "./processed_folders/")
ERROR_FOLDER = BASE_DIR / os.environ.get("ERROR_FOLDER", "./error_folders/")

for folder in [WATCH_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
    os.makedirs(folder, exist_ok=True)

log.info(f"Monitoring: {WATCH_FOLDER}")

# --- Google AI Configuration ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing.")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    OCR_MODEL_NAME = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME = "text-embedding-004"
    
    log.info(f"AI Clients Init: OCR={OCR_MODEL_NAME}, Embed={EMBEDDING_MODEL_NAME}")
except Exception as e:
    log.critical(f"Google AI Init Failed: {e}")
    exit(1)

# --- Qdrant Configuration ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "document_chunks"

try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    log.info(f"Qdrant Connected: {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    log.critical(f"Qdrant Init Failed: {e}")
    exit(1)

# --- MySQL Configuration ---
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME", "rag_system"),
    "connect_timeout": 10
}

# --- Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", 
    chunk_size=800, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ==============================================================================
# 2. ASYNC CONTROLS
# ==============================================================================

CONCURRENCY_LIMIT = asyncio.Semaphore(5)
GEMINI_LIMITER = AsyncLimiter(max_rate=60, time_period=60)

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        log.error(f"MySQL Connect Error: {err}")
        return None

def register_topic_safe(topic_id):
    conn = get_db_connection()
    if not conn: return False
    
    description = f"Documents related to {topic_id.replace('_', ' ').title()}"
    sql = """
    INSERT INTO topics (topic_id, description) 
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE description = VALUES(description)
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (topic_id, description))
            conn.commit()
        log.info(f"Topic '{topic_id}' synced to DB.")
        return True
    except Exception as e:
        log.error(f"MySQL Write Error: {e}")
        return False
    finally:
        conn.close()

def safe_move_file(src_path, dest_folder):
    """Safely moves a file, overwriting if it exists."""
    try:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)
        
        # If file exists in destination, remove it first to ensure overwrite
        if os.path.exists(dest_path):
            os.remove(dest_path)
            
        shutil.move(src_path, dest_path)
    except Exception as e:
        log.error(f"Failed to move {src_path} to {dest_folder}: {e}")

def clean_text(text):
    """Cleans text to improve RAG quality."""
    if not text: return ""
    # Fix hyphenated words at line breaks (e.g. "integ-\nration" -> "integration")
    text = text.replace("-\n", "")
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text

async def wait_for_file_stability(file_path, timeout=10):
    """Waits until file size stops changing (file is fully copied)."""
    start_time = time.time()
    last_size = -1
    
    while True:
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            return False # File disappeared

        if current_size == last_size:
            return True # Size hasn't changed in 1 second, it's stable
        
        last_size = current_size
        
        if time.time() - start_time > timeout:
            log.error(f"File {file_path} is unstable (still copying?)")
            return False
            
        await asyncio.sleep(1)

def extract_text_from_pdf_sync(file_path):
    text_by_page = []
    try:
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                rect = page.rect
                
                if rect.height > 200:
                    clip_rect = fitz.Rect(0, CROP_OCR_LIMIT, rect.width, rect.height - CROP_OCR_LIMIT)
                else:
                    clip_rect = rect

                # 1. Digital Text
                page_text = page.get_text(sort=True, clip=clip_rect).strip()
                
                # 2. Heuristic Check
                if len(page_text) < 50:
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip_rect)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    stat = np.array(img.convert('L'))
                    if stat.std() < VARIANCE_LIMIT:
                        log.debug(f"Page {page_num+1} seems blank. Skipping.")
                        continue

                    text_by_page.append((page_num + 1, img))
                else:
                    # Apply Text Cleaning here
                    cleaned_text = clean_text(page_text)
                    if cleaned_text:
                        text_by_page.append((page_num + 1, cleaned_text))
                    
        return text_by_page
    except Exception as e:
        log.error(f"PDF Read Error {file_path}: {e}")
        return []

# --- Async Gemini Wrappers ---

@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5), reraise=True)
async def async_ocr_generate(image_input):
    async with GEMINI_LIMITER:
        model = genai.GenerativeModel(OCR_MODEL_NAME)
        response = await model.generate_content_async(
            ["Transcribe this image exactly. Do not say 'Here is the text'.", image_input]
        )
        return clean_text(response.text.strip())

@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5), reraise=True)
async def async_embed_batch(batch_texts):
    async with GEMINI_LIMITER:
        result = await asyncio.to_thread(
            genai.embed_content,
            model=EMBEDDING_MODEL_NAME,
            content=batch_texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']

# ==============================================================================
# 4. CORE PIPELINE
# ==============================================================================

async def process_single_file_async(topic_id, file_path):
    async with CONCURRENCY_LIMIT:
        filename = os.path.basename(file_path)
        log.info(f"Processing: {filename}")

        if not await wait_for_file_stability(file_path):
            return False
        
        extracted_content = []
        
        try:
            # --- 1. EXTRACTION ---
            if filename.lower().endswith(".pdf"):
                raw_pages = await asyncio.to_thread(extract_text_from_pdf_sync, file_path)
                
                for page_num, item in raw_pages:
                    if isinstance(item, str):
                        extracted_content.append(item)
                    elif isinstance(item, Image.Image):
                        log.info(f"OCR scanning page {page_num} of {filename}...")
                        text = await async_ocr_generate(item)
                        if text: extracted_content.append(text)

            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img = await asyncio.to_thread(Image.open, file_path)
                text = await async_ocr_generate(img)
                if text: extracted_content.append(text)
            
            else:
                log.warning(f"Skipping unsupported type: {filename}")
                return False

            full_text = "\n\n".join([t for t in extracted_content if t])
            if not full_text:
                log.warning(f"No text extracted: {filename}")
                return False

            # --- 2. CHUNKING ---
            chunks = text_splitter.split_text(full_text)
            if not chunks: return False

            # --- 3. EMBEDDING ---
            all_vectors = []
            batch_size = 50 
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_vectors = await async_embed_batch(batch)
                all_vectors.extend(batch_vectors)

            # --- 4. UPSERTING ---
            points = []
            for i, vector in enumerate(all_vectors):
                unique_sig = f"{topic_id}_{filename}_{i}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_sig))

                points.append(models.PointStruct(
                    id=chunk_id, 
                    vector=vector,
                    payload={
                        "topic_id": topic_id,
                        "source_file": filename,
                        "chunk_index": i,
                        "content": chunks[i]
                    }
                ))

            UPSERT_BATCH_SIZE = 100
            for i in range(0, len(points), UPSERT_BATCH_SIZE):
                sub_points = points[i : i + UPSERT_BATCH_SIZE]
                await asyncio.to_thread(
                    qdrant_client.upsert,
                    collection_name=QDRANT_COLLECTION,
                    points=sub_points
                )
            
            log.info(f"✅ Indexed {len(points)} chunks: {filename}")
            return True

        except Exception as e:
            log.error(f"❌ Failed {filename}: {e}")
            return False

async def process_topic_folder_async(topic_id, folder_path):
    log.info(f"--- Topic: {topic_id} ---")
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files: return

    dest_proc = PROCESSED_FOLDER / topic_id
    dest_err = ERROR_FOLDER / topic_id
    os.makedirs(dest_proc, exist_ok=True)
    os.makedirs(dest_err, exist_ok=True)

    tasks = [process_single_file_async(topic_id, os.path.join(folder_path, f)) for f in files]
    results = await asyncio.gather(*tasks)

    success_count = 0
    for filename, success in zip(files, results):
        src = os.path.join(folder_path, filename)
        if success:
            safe_move_file(src, dest_proc)
            success_count += 1
        else:
            safe_move_file(src, dest_err)

    if success_count > 0:
        await asyncio.to_thread(register_topic_safe, topic_id)

    if not os.listdir(folder_path):
        os.rmdir(folder_path)

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def initialize_qdrant_collection():
    try:
        if not qdrant_client.collection_exists(QDRANT_COLLECTION):
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            qdrant_client.create_payload_index(QDRANT_COLLECTION, "topic_id", models.PayloadSchemaType.KEYWORD)
            # Added source_file index for better management later
            qdrant_client.create_payload_index(QDRANT_COLLECTION, "source_file", models.PayloadSchemaType.KEYWORD)
            log.info("Qdrant Collection Initialized.")
    except Exception as e:
        log.error(f"Qdrant Init Error: {e}")

async def main_loop():
    log.info("--- Async Ingestion Service Started ---")
    
    while True:
        try:
            folders = [f for f in os.listdir(WATCH_FOLDER) if os.path.isdir(os.path.join(WATCH_FOLDER, f))]
            
            if not folders:
                await asyncio.sleep(SLEEP_TIME)
                continue

            for topic_id in folders:
                await process_topic_folder_async(topic_id, os.path.join(WATCH_FOLDER, topic_id))
            
            await asyncio.sleep(SLEEP_TIME)

        except asyncio.CancelledError:
            log.info("Service stopping...")
            break
        except Exception as e:
            log.error(f"Main Loop Error: {e}", exc_info=True)
            await asyncio.sleep(10)

if __name__ == "__main__":
    initialize_qdrant_collection()
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass