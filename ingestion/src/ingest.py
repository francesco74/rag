import os
import shutil
import logging
import uuid
import pathlib
import asyncio
import mysql.connector
from PIL import Image
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

# Web & Networking
import requests
import trafilatura

# Document Processing
import fitz          # PyMuPDF
import pymupdf4llm   # Native PDF to Markdown extraction

# Google Gemini (GenAI)
import google.generativeai as genai

# Vector DB
from qdrant_client import models
from qdrant_client import AsyncQdrantClient

# Text Splitting
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

# Async & Resilience
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    RetryError
)
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from db_logger import MySQLLogHandler, get_db_connection


# ==============================================================================
# 1. CONFIGURATION & LOGGING SETUP
# ==============================================================================

load_dotenv()
BASE_DIR = pathlib.Path(__file__).parent.resolve()

log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper() # Set to DEBUG for deep tracing
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("IngestWorker")
db_handler = MySQLLogHandler()
db_handler.setFormatter(logging.Formatter(log_format))
log.addHandler(db_handler)
log.info("Starting Ingestion Worker Initialization...")

# --- DIRECTORY CREATION LOGIC ---
DATA_FOLDER = pathlib.Path(os.environ.get("DATA_FOLDER", str(BASE_DIR)))
WATCH_FOLDER = DATA_FOLDER / "watch"
PROCESSED_FOLDER = DATA_FOLDER / "processed"
ERROR_FOLDER = DATA_FOLDER / "error"

for folder in [DATA_FOLDER, WATCH_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
    if not folder.exists():
        log.info(f"Creating directory: {folder}")
        folder.mkdir(parents=True, exist_ok=True)

# --- Google AI Configuration ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    OCR_MODEL_NAME = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME = "gemini-embedding-001" 
    log.info(f"AI Models Configured: OCR='{OCR_MODEL_NAME}', Embeddings='{EMBEDDING_MODEL_NAME}', API Key {GOOGLE_API_KEY[:10]}...")
except Exception as e:
    log.critical(f"Google AI Init Failed: {e}")
    exit(1)

# --- Qdrant & DB Configuration ---
try:
    qdrant_client = AsyncQdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"), 
        port=int(os.environ.get("QDRANT_PORT", 6333))
    )
    QDRANT_COLLECTION = "document_chunks"
    CACHE_COLLECTION = "semantic_cache"
    log.info("Qdrant client initialized successfully.")
except Exception as e:
    log.critical(f"Qdrant Init Failed: {e}")
    exit(1)

db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME", "rag_system"),
    "connect_timeout": 10
}

# --- Resilience & Splitting Config ---
CROP_OCR_LIMIT = 0  # If I want to remove header/footer of documents
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150

log.debug(f"Chunking Config: Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}")

CONCURRENCY_LIMIT = asyncio.Semaphore(2)
GEMINI_LIMITER = AsyncLimiter(max_rate=10, time_period=60)

GEMINI_RETRY = retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_random_exponential(multiplier=2, min=10, max=80),
    stop=stop_after_attempt(20),
    before_sleep=lambda retry_state: log.warning(f"API Error Caught: {retry_state.outcome.exception()} | Retrying in {retry_state.next_action.sleep:.2f}s...")
)

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " ", ""]
)


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
# 2. CORE HELPER FUNCTIONS
# ==============================================================================

async def initialize_qdrant_collection():
    try:
        if not await qdrant_client.collection_exists(QDRANT_COLLECTION):
            log.info(f"Initial setup: Creating Qdrant collection '{QDRANT_COLLECTION}'")
            await qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            log.debug("Creating Qdrant payload indexes...")
            await qdrant_client.create_payload_index(QDRANT_COLLECTION, "topic_id", models.PayloadSchemaType.KEYWORD)
            await qdrant_client.create_payload_index(QDRANT_COLLECTION, "source", models.PayloadSchemaType.KEYWORD)
            await qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION, field_name="content",
                field_schema=models.TextIndexParams(type="text", tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True)
            )
        else:
            log.debug(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")

        if not await qdrant_client.collection_exists(CACHE_COLLECTION):
            log.info(f"Initial setup: Creating Cache collection '{CACHE_COLLECTION}'")
            await qdrant_client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            await qdrant_client.create_payload_index(CACHE_COLLECTION, "topic", models.PayloadSchemaType.KEYWORD)
    except Exception as e:
        log.error(f"Qdrant Setup Error: {e}")


def register_topic_safe(topic_id):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO topics (topic_id, description) VALUES (%s, %s) ON DUPLICATE KEY UPDATE topic_id=topic_id", 
                           (topic_id, f"Knowledge base for {topic_id}"))
            conn.commit()
            log.info(f"Successfully registered topic in MySQL: {topic_id}")
        return True
    except Exception as e: 
        log.error(f"MySQL Registration Error: {e}")
        return False
    finally: 
        conn.close()

def safe_move_file(src_path, dest_folder):
    try:
        src = pathlib.Path(src_path)
        dest = pathlib.Path(dest_folder) / src.name
        log.debug(f"Moving file from {src} to {dest}")
        if dest.exists(): dest.unlink()
        shutil.move(str(src), str(dest))
    except Exception as e: 
        log.error(f"File Move Error ({src_path}): {e}")

def clean_text(text):
    if not text: return ""
    return " ".join(text.replace("-\n", "").split())

# ==============================================================================
# 3. EXTRACTION & AI GENERATION
# ==============================================================================

@GEMINI_RETRY
async def async_ocr_generate(image_input, as_markdown=False):
    async with GEMINI_LIMITER:
        model = genai.GenerativeModel(OCR_MODEL_NAME)
        if as_markdown:
            log.debug("Calling Gemini Vision API (Markdown Mode)...")
            prompt = "Transcribe this document image. Format the output strictly as Markdown. Preserve all tables, headings, and lists exactly as they appear."
        else:
            log.debug("Calling Gemini Vision API (Raw Text Mode)...")
            prompt = "Transcribe this document image precisely as raw text:"
            
        response = await model.generate_content_async([prompt, image_input])
        text_length = len(response.text) if response.text else 0
        log.debug(f"Gemini API returned {text_length} characters.")
        return clean_text(response.text.strip())

async def extract_hybrid_markdown_from_pdf_async(file_path):
    log.info(f"Starting Hybrid PDF Extraction for: {pathlib.Path(file_path).name}")
    md_pages = []
    
    def evaluate_and_extract_page(path, p_num):
        log.debug(f"Evaluating page {p_num+1} of {path}...")
        with fitz.open(path) as doc:
            page = doc[p_num]
            rect = page.rect
            clip_rect = fitz.Rect(0, CROP_OCR_LIMIT, rect.width, rect.height - CROP_OCR_LIMIT) if rect.height > 200 else rect
            text = page.get_text(sort=True, clip=clip_rect).strip()
            
            if len(text) < 50:
                log.debug(f"Page {p_num+1}: Low text density detected ({len(text)} chars). Flagging for Vision OCR.")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, clip=clip_rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return {"type": "image", "content": img}
            else:
                log.debug(f"Page {p_num+1}: Sufficient text density. Extracting native Markdown.")
                page_md = pymupdf4llm.to_markdown(doc, pages=[p_num])
                return {"type": "markdown", "content": page_md}

    try:
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            log.info(f"PDF {pathlib.Path(file_path).name} contains {total_pages} total pages.")

        for page_num in range(total_pages):
            page_data = await asyncio.to_thread(evaluate_and_extract_page, str(file_path), page_num)
            
            if page_data["type"] == "image":
                gemini_md = await async_ocr_generate(page_data["content"], as_markdown=True)
                md_pages.append(gemini_md)
            else:
                md_pages.append(page_data["content"])

        final_md = "\n\n---\n\n".join(md_pages)
        log.info(f"Completed Hybrid PDF Extraction. Total generated Markdown size: {len(final_md)} characters.")
        return final_md
    except Exception as e:
        log.error(f"Critical Hybrid PDF Read Error {file_path}: {e}")
        return ""

@GEMINI_RETRY
async def async_embed_batch(batch_texts):
    if not batch_texts: return []
    async with GEMINI_LIMITER:
        log.debug(f"Calling Gemini Embeddings API for a batch of {len(batch_texts)} chunks...")
        result = await genai.embed_content_async(
            model=EMBEDDING_MODEL_NAME, 
            content=batch_texts, 
            task_type="RETRIEVAL_DOCUMENT", 
            output_dimensionality=768
        )
        embeddings = result['embedding']
        log.debug(f"Embeddings API successful. Received {len(embeddings)} vectors (Dimension: {len(embeddings[0]) if embeddings else 0}).")
        return embeddings


# ==============================================================================
# 4. PIPELINE ORCHESTRATION
# ==============================================================================

async def process_single_file_async(topic_id, file_path, root_folder):
    async with CONCURRENCY_LIMIT:
        file_path = pathlib.Path(file_path)
        filename = file_path.name
        log.info(f"--- Picking up file for processing: {filename} ---")
        
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")

        try:
            relative_path_str = str(file_path.relative_to(root_folder)).replace("\\", "/")
        except ValueError:
            relative_path_str = filename 
        
        origin_url = None
        if meta_path.exists():
            log.debug(f"Found metadata file: {meta_path.name}")
            with open(meta_path, 'r') as f: origin_url = f.read().strip()
            meta_path.unlink()

        source_name = origin_url if origin_url else relative_path_str
        log.debug(f"Assigned source identifier: {source_name}")

        try:
            full_text = ""
            is_markdown = False

            # --- ROUTING LOGIC ---
            if filename.lower().endswith(".pdf"):
                log.info(f"Routing '{filename}' to Hybrid PDF Extractor.")
                full_text = await extract_hybrid_markdown_from_pdf_async(file_path)
                is_markdown = True
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                log.info(f"Routing '{filename}' to Raw Image OCR.")
                full_text = await async_ocr_generate(Image.open(file_path), as_markdown=True)
                is_markdown = filename.lower().endswith(".md")
            elif filename.lower().endswith((".txt", ".md")):
                log.info(f"Routing '{filename}' to standard text read.")
                with open(file_path, 'r', encoding='utf-8') as f: full_text = f.read()
                is_markdown = filename.lower().endswith(".md")

            if not full_text: 
                log.warning(f"Aborting '{filename}': No readable text was extracted.")
                await finalize_file_move(file_path, root_folder, topic_id, success=False)
                return False
            
            # Save Artifact
            artifact_ext = ".md" if is_markdown else ".ocr"
            artifact_path = file_path.with_suffix(file_path.suffix + artifact_ext)
            log.debug(f"Writing extracted text artifact to {artifact_path.name}")
            with open(artifact_path, "w", encoding="utf-8") as f_art: 
                f_art.write(full_text)
            
            # --- TWO-PASS CHUNKING LOGIC ---
            log.debug("Initiating chunking process...")
            if is_markdown:
                md_header_splits = markdown_splitter.split_text(full_text)
                log.debug(f"Pass 1 (Markdown Split): Generated {len(md_header_splits)} logical sections.")
                final_chunks = text_splitter.split_documents(md_header_splits)
            else:
                final_chunks = text_splitter.create_documents([full_text])
            
            log.info(f"Final Chunking output: {len(final_chunks)} chunks ready for embedding.")
            batch_texts = [doc.page_content for doc in final_chunks]
            
            all_vectors = []
            for i in range(0, len(batch_texts), 50):
                log.debug(f"Processing embedding batch {i} to {i+50}...")
                all_vectors.extend(await async_embed_batch(batch_texts[i:i+50]))

            # --- QDRANT UPSERT WITH DYNAMIC METADATA ---
            log.debug("Constructing Qdrant PointStruct objects...")
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=vec, 
                    payload={
                        "topic_id": topic_id, 
                        "source": source_name, 
                        "chunk_index": idx, 
                        "content": final_chunks[idx].page_content,
                        **final_chunks[idx].metadata
                    }
                )
                for idx, vec in enumerate(all_vectors)
            ]
            
            log.info(f"Preparing to upsert {len(points)} points to Qdrant for source: {source_name}")
            
            # Cleanup old vectors
            await qdrant_client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(  
                    filter=models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_name))])
                )
            )
            log.debug("Cleared previous vectors for source (if any).")
            
            for i in range(0, len(points), 100):
                await qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+100])
            
            log.info(f"SUCCESS: Finished processing and indexing '{filename}'.")
            await finalize_file_move(file_path, root_folder, topic_id, success=True)
            return True

        except Exception as e:
            log.error(f"Process Failure for '{filename}': {e}", exc_info=True)
            await finalize_file_move(file_path, root_folder, topic_id, success=False)
            return False

async def finalize_file_move(file_path, root_folder, topic_id, success):
    try:
        try: 
            relative_path = file_path.relative_to(root_folder)
        except ValueError: 
            relative_path = pathlib.Path(file_path.name)

        dest_root = PROCESSED_FOLDER if success else ERROR_FOLDER
        final_dest = dest_root / topic_id / relative_path
        final_dest.parent.mkdir(parents=True, exist_ok=True)
        
        log.debug(f"Finalizing file: moving main file to {final_dest.parent}")
        await asyncio.to_thread(safe_move_file, file_path, final_dest.parent)

        for ext in [".ocr", ".md"]:
            artifact_sibling = file_path.with_suffix(file_path.suffix + ext)
            if artifact_sibling.exists():
                log.debug(f"Finalizing file: moving artifact {artifact_sibling.name} to {final_dest.parent}")
                await asyncio.to_thread(safe_move_file, artifact_sibling, final_dest.parent)
            
        status_tag = '[SUCCESS]' if success else '[ERROR]'
        log.info(f"File lifecycle complete: {status_tag} {file_path.name}")
    except Exception as e:
        log.error(f"Failed during finalize_file_move for {file_path.name}: {e}")

async def process_topic_folder_async(topic_id, folder_path):
    folder_path = pathlib.Path(folder_path)
    log.info(f"====== Topic Start: '{topic_id}' ======")
    
    # 1. URL Batch (PARALLELIZED)
    url_file = folder_path / "urls.txt"
    if url_file.exists():
        log.info(f"Scraping URLs from {url_file.name}")
        with open(url_file, 'r') as f:
            urls = [l.strip() for l in f if l.strip().startswith("http")]

        async def process_url(u):
            return u, await asyncio.to_thread(scrape_web_page, u, folder_path)

        url_tasks = [process_url(u) for u in urls]
        url_results = await asyncio.gather(*url_tasks)

        for url, text in url_results:
            if text:
                v_name = f"web_{uuid.uuid4().hex[:8]}.txt"
                v_path = folder_path / v_name
                with open(v_path, 'w', encoding='utf-8') as f_v: f_v.write(text)
                with open(str(v_path) + ".meta", 'w', encoding='utf-8') as f_m: f_m.write(url)
        
        # Move the urls.txt file to processed immediately after scraping
        await asyncio.to_thread(safe_move_file, str(url_file), PROCESSED_FOLDER / topic_id)

    # 2. File Batch
    all_files = [
        p for p in folder_path.rglob("*") 
        if p.is_file() and not p.name.endswith(".meta") and p.name != "urls.txt"
    ]
    
    if not all_files:
        log.info(f"No documents found to ingest in topic folder '{topic_id}'.")
    else:
        log.info(f"Topic '{topic_id}' has {len(all_files)} files ready for ingestion.")
    
    tasks = [process_single_file_async(topic_id, p, folder_path) for p in all_files]
    results = await asyncio.gather(*tasks)

    # 3. Final Metadata & Cleanup
    if any(results): 
        log.debug("At least one file succeeded. Syncing topic metadata to DB.")
        await asyncio.to_thread(register_topic_safe, topic_id)
    
    log.debug("Cleaning up empty directories in watch folder...")
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in dirs:
            try: os.rmdir(os.path.join(root, name))
            except OSError: pass

    log.info(f"====== Topic End: '{topic_id}' ======")

async def main_run():
    log.info("Ingestion Worker Active - Beginning One Shot Run")
    await initialize_qdrant_collection()

    folders = [f for f in os.listdir(WATCH_FOLDER) if (WATCH_FOLDER / f).is_dir()]
    if not folders:
        log.info("No topic folders found in watch directory. Exiting.")
        return

    log.info(f"Found {len(folders)} topic folders: {folders}")
    for tid in folders: 
        await process_topic_folder_async(tid, WATCH_FOLDER / tid)
    
    log.info("One Shot Run Finished Successfully.")

if __name__ == "__main__":
    asyncio.run(main_run())