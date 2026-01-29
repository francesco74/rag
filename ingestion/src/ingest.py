import os
import shutil
import time
import logging
import uuid
import pathlib
import asyncio
import mysql.connector
import numpy as np
import requests
import trafilatura
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv


# Image & PDF Processing
import fitz  # PyMuPDF
from PIL import Image

# Google Gemini (GenAI)
import google.generativeai as genai

# Vector DB
from qdrant_client import models
from qdrant_client import AsyncQdrantClient

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Async & Resilience
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable


# ==============================================================================
# 1. CONFIGURATION & LOGGING SETUP
# ==============================================================================

GEMINI_RETRY = retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: log.warning(f"Rate limit hit. Retrying in {retry_state.next_action.sleep}s...")
)

CROP_OCR_LIMIT = 50
VARIANCE_LIMIT = 15
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150

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

# --- DIRECTORY CREATION LOGIC ---
DATA_FOLDER = pathlib.Path(os.environ.get("DATA_FOLDER", str(BASE_DIR)))
WATCH_FOLDER = DATA_FOLDER / "watch"
PROCESSED_FOLDER = DATA_FOLDER / "processed"
ERROR_FOLDER = DATA_FOLDER / "error"

# Ensure all directories exist
for folder in [DATA_FOLDER, WATCH_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
    if not folder.exists():
        log.info(f"Creating directory: {folder}")
        folder.mkdir(parents=True, exist_ok=True)

log.info(f"System paths initialized. Monitoring: {WATCH_FOLDER}")

# --- Google AI Configuration ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    OCR_MODEL_NAME = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME = "gemini-embedding-001"
    log.debug(f"AI Models: OCR={OCR_MODEL_NAME}, Embed={EMBEDDING_MODEL_NAME}")
except Exception as e:
    log.critical(f"Google AI Init Failed: {e}"); exit(1)

# --- Qdrant Configuration ---
try:
    qdrant_client = AsyncQdrantClient(host=os.environ.get("QDRANT_HOST", "localhost"), port=int(os.environ.get("QDRANT_PORT", 6333)))
    QDRANT_COLLECTION = "document_chunks"
    CACHE_COLLECTION = "semantic_cache"
except Exception as e:
    log.critical(f"Qdrant Init Failed: {e}"); exit(1)

db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME", "rag_system"),
    "connect_timeout": 10
}

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

CONCURRENCY_LIMIT = asyncio.Semaphore(2)
GEMINI_LIMITER = AsyncLimiter(max_rate=10, time_period=60)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

async def initialize_qdrant_collection():
    try:
        if not await qdrant_client.collection_exists(QDRANT_COLLECTION):
            log.info(f"Initial setup: Creating Qdrant collection '{QDRANT_COLLECTION}'")
            await qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            
            # Create indexes for faster retrieval filtering
            await qdrant_client.create_payload_index(QDRANT_COLLECTION, "topic_id", models.PayloadSchemaType.KEYWORD)
            await qdrant_client.create_payload_index(QDRANT_COLLECTION, "source", models.PayloadSchemaType.KEYWORD)
            log.info("Qdrant Collection and Indexes Initialized.")

        # 2. Initialize Semantic Cache Collection
        # Note: CACHE_COLLECTION should be defined as "semantic_cache"
        if not await qdrant_client.collection_exists(CACHE_COLLECTION):
            log.info(f"Initial setup: Creating Cache collection '{CACHE_COLLECTION}'")
            await qdrant_client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=models.VectorParams(
                    size=768, 
                    distance=models.Distance.COSINE
                )
            )
            # Index topic in cache to allow topic-specific cache clearing if needed
            await qdrant_client.create_payload_index(CACHE_COLLECTION, "topic", models.PayloadSchemaType.KEYWORD)
            log.info(f"Cache collection '{CACHE_COLLECTION}' initialized.")
    except Exception as e:
        log.error(f"Qdrant Setup Error: {e}")

def get_db_connection():
    try: return mysql.connector.connect(**db_config)
    except Exception as e: 
        log.warning(f"Database connection failed: {e}")
        return None

def register_topic_safe(topic_id):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO topics (topic_id, description) VALUES (%s, %s) ON DUPLICATE KEY UPDATE topic_id=topic_id", 
                           (topic_id, f"Knowledge base for {topic_id}"))
            conn.commit()
        log.info(f"Topic metadata synced: {topic_id}")
        return True
    except Exception as e: 
        log.error(f"MySQL Registration Error: {e}")
        return False
    finally: conn.close()

def safe_move_file(src_path, dest_folder):
    try:
        src = pathlib.Path(src_path)
        dest = pathlib.Path(dest_folder) / src.name
        if dest.exists(): dest.unlink()
        shutil.move(str(src), str(dest))
        log.debug(f"Cleanup: Moved {src.name} to {dest.parent.name}")
    except Exception as e: log.error(f"File Management Error: {e}")

def clean_text(text):
    if not text: return ""
    return " ".join(text.replace("-\n", "").split())

def extract_text_from_pdf_sync(file_path):
    text_by_page = []
    log.debug(f"Starting PDF Extraction: {pathlib.Path(file_path).name}")
    try:
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                rect = page.rect
                clip_rect = fitz.Rect(0, CROP_OCR_LIMIT, rect.width, rect.height - CROP_OCR_LIMIT) if rect.height > 200 else rect
                page_text = page.get_text(sort=True, clip=clip_rect).strip()
                
                if len(page_text) < 50:
                    log.debug(f"Page {page_num+1} triggered OCR (low text density)")
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, clip=clip_rect)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    if np.array(img.convert('L')).std() > VARIANCE_LIMIT:
                        text_by_page.append((page_num + 1, img))
                else:
                    text_by_page.append((page_num + 1, clean_text(page_text)))
        return text_by_page
    except Exception as e:
        log.error(f"Critical PDF Read Error {file_path}: {e}"); return []

def download_linked_file(url, topic_folder):
    log.info(f"Scraper: Downloading linked asset -> {url}")
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            filename = os.path.basename(urlparse(url).path) or f"dl_{uuid.uuid4().hex[:8]}"
            save_path = topic_folder / filename
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            with open(str(save_path) + ".meta", "w", encoding="utf-8") as f: f.write(url)
            return filename
    except Exception as e: log.warning(f"Failed asset download: {e}")
    return None

def scrape_web_page(url, topic_folder):
    log.info(f"Scraper: Analyzing URL -> {url}")
    try:
        html = trafilatura.fetch_url(url)
        if not html: return None
        main_text = trafilatura.extract(html, include_comments=False)
        import re
        links = re.findall(r'href=[\'"]?([^\'" >]+)', html)
        for link in links:
            if link.lower().endswith('.pdf'):
                download_linked_file(urljoin(url, link), topic_folder)
        return main_text
    except Exception as e:
        log.error(f"Scraper Logic Error: {e}")
        return None

# ==============================================================================
# 3. ASYNC WRAPPERS (WITH 429 LOGGING)
# ==============================================================================

@GEMINI_RETRY
async def async_ocr_generate(image_input):
    async with GEMINI_LIMITER:
        log.debug("API Call: Gemini Vision OCR")
        model = genai.GenerativeModel(OCR_MODEL_NAME)
        response = await model.generate_content_async(["Transcribe this document image precisely:", image_input])
        return clean_text(response.text.strip())

@GEMINI_RETRY
async def async_embed_batch(batch_texts):
    if not batch_texts: return []
    async with GEMINI_LIMITER:
        log.debug(f"API Call: Embedding {len(batch_texts)} chunks")
        result = await asyncio.to_thread(genai.embed_content, model=EMBEDDING_MODEL_NAME, content=batch_texts, 
                                            task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768)
        return result['embedding']
        

# ==============================================================================
# 4. PIPELINE
# ==============================================================================

async def process_single_file_async(topic_id, file_path, root_folder):
    async with CONCURRENCY_LIMIT:
        file_path = pathlib.Path(file_path)
        filename = file_path.name
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")

        try:
            relative_path_str = str(file_path.relative_to(root_folder)).replace("\\", "/")
        except ValueError:
            relative_path_str = filename # Fallback if path logic fails
        
        origin_url = None
        if meta_path.exists():
            with open(meta_path, 'r') as f: origin_url = f.read().strip()
            meta_path.unlink()

        source_name = origin_url if origin_url else relative_path_str
        log.info(f"--- Processing: {source_name} ---")

        try:
            full_text = ""
            if filename.lower().endswith(".pdf"):
                pages = await asyncio.to_thread(extract_text_from_pdf_sync, str(file_path))
                texts = []
                for p_num, item in pages:
                    texts.append(item if isinstance(item, str) else await async_ocr_generate(item))
                full_text = "\n\n".join(texts)
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                full_text = await async_ocr_generate(Image.open(file_path))
            elif filename.lower().endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f: full_text = f.read()

            if not full_text: 
                log.warning(f"Skipping {filename}: No readable text.")
                return False
            
            # Creates 'file.jpg.ocr' containing the raw text
            ocr_path = file_path.with_suffix(file_path.suffix + ".ocr")
            with open(ocr_path, "w", encoding="utf-8") as f_ocr:
                f_ocr.write(full_text)
            log.debug(f"Saved OCR result to {ocr_path.name}")
            # -----------------------------------------

            chunks = text_splitter.split_text(full_text)
            log.debug(f"Document chunked: {len(chunks)} segments")
            
            all_vectors = []
            for i in range(0, len(chunks), 50):
                all_vectors.extend(await async_embed_batch(chunks[i:i+50]))

            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=vec, 
                    payload={
                        "topic_id": topic_id, 
                        "source": source_name, 
                        "chunk_index": idx, 
                        "content": chunks[idx]
                    }
                )
                for idx, vec in enumerate(all_vectors)
            ]
            
            # Before upserting loop
            log.info(f"Removing existing chunks for source: {source_name}")
            await qdrant_client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(  
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source_name)
                            )
                        ]
                    )
                )
            )
            
            for i in range(0, len(points), 100):
                await qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points[i:i+100]
                )
            
            log.info(f"Success: Indexed {filename}")
            return True
        except Exception as e:
            log.error(f"Process Failure {filename}: {e}", exc_info=True); return False

async def process_topic_folder_async(topic_id, folder_path):
    folder_path = pathlib.Path(folder_path)
    log.info(f"====== Topic Start: {topic_id} ======")
    
    # 1. URL Batch (PARALLELIZED)
    url_file = folder_path / "urls.txt"
    if url_file.exists():
        log.info(f"Scraping URLs from {url_file.name}")
        with open(url_file, 'r') as f:
            urls = [l.strip() for l in f if l.strip().startswith("http")]

        # Wrapper for parallel execution
        async def process_url(u):
            return u, await asyncio.to_thread(scrape_web_page, u, folder_path)

        # Run all scrapes concurrently
        url_tasks = [process_url(u) for u in urls]
        url_results = await asyncio.gather(*url_tasks)

        for url, text in url_results:
            if text:
                v_name = f"web_{uuid.uuid4().hex[:8]}.txt"
                v_path = folder_path / v_name
                with open(v_path, 'w', encoding='utf-8') as f_v: f_v.write(text)
                with open(str(v_path) + ".meta", 'w', encoding='utf-8') as f_m: f_m.write(url)
        
        await asyncio.to_thread(safe_move_file, str(url_file), PROCESSED_FOLDER / topic_id)

    # 2. File Batch
    all_files = [
        p for p in folder_path.rglob("*") 
        if p.is_file() and not p.name.endswith(".meta") and p.name != "urls.txt"
    ]
    
    if not all_files:
        log.info(f"No files found in {topic_id} (checked subdirectories).")
    else:
        log.info(f"Found {len(all_files)} files to ingest recursively.")
    
    tasks = [process_single_file_async(topic_id, p, folder_path) for p in all_files]
    results = await asyncio.gather(*tasks)

    for file_path_obj, success in zip(all_files, results):
        # Calculate relative path to keep structure (e.g., "sub/image.png")
        relative_path = file_path_obj.relative_to(folder_path)
        
        root_dest = PROCESSED_FOLDER if success else ERROR_FOLDER
        final_dest = root_dest / topic_id / relative_path
        
        # Ensure parent sub-folders exist in destination
        os.makedirs(final_dest.parent, exist_ok=True)
        
        # 1. Move Main File
        await asyncio.to_thread(safe_move_file, file_path_obj, final_dest.parent)

        # 2. Move Sibling OCR File (if it exists)
        # This looks for 'file.jpg.ocr' created in the previous step
        ocr_sibling = file_path_obj.with_suffix(file_path_obj.suffix + ".ocr")
        if ocr_sibling.exists():
            # It will land in processed/topic/sub/file.jpg.ocr
            await asyncio.to_thread(safe_move_file, ocr_sibling, final_dest.parent)
    
    if any(results): 
        # FIX: Syntax for to_thread
        await asyncio.to_thread(register_topic_safe, topic_id)
    
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass # Directory not empty

    log.info(f"====== Topic End: {topic_id} ======")

async def main_run():
    log.info("Ingestion Worker Active - One Shot Run")
    await initialize_qdrant_collection()

    folders = [f for f in os.listdir(WATCH_FOLDER) if (WATCH_FOLDER / f).is_dir()]
    if not folders:
        log.info("No work found in watch folder.")
        return

    for tid in folders: 
        await process_topic_folder_async(tid, WATCH_FOLDER / tid)
    log.info("Run finished.")

if __name__ == "__main__":
    asyncio.run(main_run())