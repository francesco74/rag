import os
import shutil
import logging
import uuid
import pathlib
import asyncio
from PIL import Image
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
import traceback

# Web & Networking
import httpx
import trafilatura


# Document Processing
import fitz          # PyMuPDF
import pymupdf4llm   # Native PDF to Markdown extraction

# Google Gemini (GenAI)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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

import io
from google.cloud import vision


# ==============================================================================
# 1. CONFIGURATION & LOGGING SETUP
# ==============================================================================

load_dotenv()
BASE_DIR = pathlib.Path(__file__).parent.resolve()

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
PARENT_COLLECTION = "parent_documents"

log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper() 
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
    OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "gemini-3-flash-preview")
    EMBEDDING_MODEL_NAME = "gemini-embedding-001" 
    log.info(f"AI Models Configured: OCR='{OCR_MODEL_NAME}', Embeddings='{EMBEDDING_MODEL_NAME}', API Key {GOOGLE_API_KEY[:10]}...")
except Exception as e:
    log.critical(f"Google AI Init Failed: {e}")
    exit(1)


try:
    vision_client = vision.ImageAnnotatorClient()
    log.info("Google Cloud Vision fallback client initialized successfully.")
except Exception as e:
    log.error(f"Google Cloud Vision Init Failed (Fallback disabled): {e}")
    vision_client = None

# --- Qdrant & DB Configuration ---
try:
    qdrant_client = AsyncQdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"), 
        port=int(os.environ.get("QDRANT_PORT", 6333)),
        timeout=60.0
    )
    
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
#text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#    encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " ", ""]
#)

http_client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)

async def download_linked_file_async(url, topic_folder):
    """Refactored to be non-blocking using httpx."""
    log.info(f"Scraper: Downloading linked asset -> {url}")
    try:
        async with http_client.stream("GET", url) as response:
            if response.status_code == 200:
                filename = os.path.basename(urlparse(url).path) or f"dl_{uuid.uuid4().hex[:8]}"
                save_path = topic_folder / filename
                
                # Offload blocking write to thread
                def _save():
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_bytes(): f.write(chunk)
                    with open(str(save_path) + ".meta", "w", encoding="utf-8") as f: f.write(url)
                
                await asyncio.to_thread(_save)
                return filename
    except Exception as e: log.warning(f"Failed asset download: {e}")
    return None

async def scrape_web_page_async(url, topic_folder):
    """Refactored to be non-blocking."""
    log.info(f"Scraper: Analyzing URL -> {url}")
    try:
        response = await http_client.get(url)
        html = response.text if response.status_code == 200 else None
        if not html: return None
        
        main_text = trafilatura.extract(html, include_comments=False)
        import re
        links = re.findall(r'href=[\'"]?([^\'" >]+\.pdf)', html, re.I)
        
        # Process downloads in parallel
        if links:
            tasks = [download_linked_file_async(urljoin(url, link), topic_folder) for link in links]
            await asyncio.gather(*tasks)
            
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
            await qdrant_client.create_payload_index(QDRANT_COLLECTION, "sub_topic_id", models.PayloadSchemaType.KEYWORD)
            await qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION, field_name="content",
                field_schema=models.TextIndexParams(type="text", tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True)
            )
        else:
            log.debug(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")

        if not await qdrant_client.collection_exists(PARENT_COLLECTION):
            log.info(f"Initial setup: Creating Parent collection '{PARENT_COLLECTION}'")
            await qdrant_client.create_collection(
                collection_name=PARENT_COLLECTION,
                # Usiamo una dimensione minima (es. 1) o la stessa (768) con vettori a zero. 
                # Manteniamo 768 per compatibilità di sistema.
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            # Indice utile se volessimo svuotare i parent per source
            await qdrant_client.create_payload_index(PARENT_COLLECTION, "source", models.PayloadSchemaType.KEYWORD)
        else:
            log.debug(f"Qdrant collection '{PARENT_COLLECTION}' already exists.")


        if not await qdrant_client.collection_exists(CACHE_COLLECTION):
            log.info(f"Initial setup: Creating Cache collection '{CACHE_COLLECTION}'")
            await qdrant_client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            await qdrant_client.create_payload_index(CACHE_COLLECTION, "topic_id", models.PayloadSchemaType.KEYWORD)
            await qdrant_client.create_payload_index(CACHE_COLLECTION, "sub_topics_key", models.PayloadSchemaType.KEYWORD)
    except Exception as e:
        log.error(f"Qdrant Setup Error: {e}")

def get_subtopic_config(topic_id, sub_topic_id):
    """Recupera la configurazione completa. Ritorna None se assente."""
    conn = get_db_connection()
    if not conn: return None
    try:
        with conn.cursor(dictionary=True) as cursor:
            # Estraiamo tutta la riga di configurazione
            cursor.execute(
                "SELECT chunk_size, chunk_overlap, parent_chunk_size FROM sub_topics WHERE topic_id = %s AND sub_topic_id = %s", 
                (topic_id, sub_topic_id)
            )
            return cursor.fetchone() 
    finally:
        conn.close()


def safe_move_file(src_path, dest_folder):
    """Kept as synchronous, but will be called via asyncio.to_thread in pipeline."""
    try:
        src = pathlib.Path(src_path)
        dest = pathlib.Path(dest_folder) / src.name
        if not src.exists(): return # Avoid error if already moved
        if dest.exists(): dest.unlink()
        shutil.move(str(src), str(dest))
    except Exception as e: 
        log.error(f"File Move Error ({src_path}): {e}")

def clean_text(text, as_markdown):

    if not text:
        result = ""
    else: 
        if as_markdown:
            # Se è markdown, uniamo solo le parole spezzate a fine riga ma preserviamo i \n
            result = text.replace("-\n", "")
        else: 
            result =  " ".join(text.replace("-\n", "").split())

    return result.strip()

# ==============================================================================
# 3. EXTRACTION & AI GENERATION
# ==============================================================================

def _cloud_vision_fallback(image: Image.Image) -> str:
    """Synchronous Cloud Vision API call, executed in a thread."""
    if not vision_client:
        raise ValueError("Cloud Vision client is not initialized. Cannot perform fallback.")
    
    log.info("Executing Google Cloud Vision document text detection...")
    
    # Convert PIL Image to bytes for Cloud Vision
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    vision_image = vision.Image(content=content)
    
    # Use document_text_detection for dense text/handwriting
    response = vision_client.document_text_detection(image=vision_image)
    
    if response.error.message:
        raise Exception(f"Cloud Vision API Error: {response.error.message}")
        
    return response.full_text_annotation.text


@GEMINI_RETRY
async def async_ocr_generate(image_input, as_markdown=False):
    async with GEMINI_LIMITER:
        model = genai.GenerativeModel(OCR_MODEL_NAME)
        
        if as_markdown:
            log.debug("Calling Gemini Vision API (Markdown Mode)...")
            prompt = "Transcribe the text in this image precisely. Format the output strictly as Markdown."
        else:
            log.debug("Calling Gemini Vision API (Raw Text Mode)...")
            prompt = "Transcribe the text in this image precisely as raw text."
            
        try:
            response = await model.generate_content_async(
                [prompt, image_input],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # DEFENSIVE CHECK: Catch the Gemini block
            if not response.candidates or not response.candidates[0].content.parts:
                reason = getattr(response.candidates[0], "finish_reason", "Unknown") if response.candidates else "Unknown"
                
                # Check if it is a Recitation (4) block. If so, trigger the fallback.
                if reason == 4 or str(reason) == "FinishReason.RECITATION":
                    log.warning("Gemini API blocked page (RECITATION). Engaging Cloud Vision API fallback...")
                    fallback_text = await asyncio.to_thread(_cloud_vision_fallback, image_input)
                    return clean_text(fallback_text.strip(), as_markdown)
                else:
                    # If it blocked for a different safety reason, bubble it up
                    raise ValueError(f"Gemini API Blocked Page. Exact Finish Reason: {reason}")
            
            return clean_text(response.text.strip(), as_markdown)
            
        except Exception as e:
            # Re-raise so the pipeline catches it and writes to the .log file
            raise ValueError(f"OCR Generation Failed: {e}")


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
                del pix 
                
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

                del page_data["content"]
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

async def process_single_file_async(topic_id, sub_topic_id, file_path, root_folder, chunk_size, chunk_overlap, parent_chunk_size):
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
            #meta_path.unlink()

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
                is_markdown = True
            elif filename.lower().endswith((".txt", ".md")):
                log.info(f"Routing '{filename}' to standard text read.")
                full_text = await asyncio.to_thread(file_path.read_text, encoding='utf-8')
                is_markdown = filename.lower().endswith(".md")

            if not full_text: 
                log.warning(f"Aborting '{filename}': No readable text was extracted.")

                # Write an explicit error log file
                log_file = file_path.with_suffix(".log")
                error_msg = "Ingestion Error: No readable text was extracted.\nPossible causes: The document is entirely blank, corrupt, or Gemini blocked the content due to copyright/safety filters."
                await asyncio.to_thread(log_file.write_text, error_msg, encoding="utf-8")

                await finalize_file_move(file_path, root_folder, topic_id, sub_topic_id, success=False)
                return False
            
            # Save Artifact
            artifact_ext = ".md" if is_markdown else ".ocr"
            artifact_path = file_path.with_suffix(file_path.suffix + artifact_ext)
            log.debug(f"Writing extracted text artifact to {artifact_path.name}")
            await asyncio.to_thread(artifact_path.write_text, full_text, encoding="utf-8")
            
            # --- PARENT / CHILD CHUNKING LOGIC ---
            log.debug(f"Initiating chunking process (Size: {chunk_size}, Overlap: {chunk_overlap})...")
            
            parent_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", 
                chunk_size=parent_chunk_size, 
                chunk_overlap=0, 
                separators=["\n\n", "\n", ". ", " "]
            )

            # Splitter per i Child (parametri dinamici dal DB per precisione vettoriale)
            child_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            # Generazione dei Parent Documents
            if is_markdown:
                md_header_splits = markdown_splitter.split_text(full_text)
                log.debug(f"Markdown Split: Generated {len(md_header_splits)} logical sections.")
                parent_docs = parent_text_splitter.split_documents(md_header_splits)
            else:
                parent_docs = parent_text_splitter.create_documents([full_text])
            
            log.info(f"Generated {len(parent_docs)} Parent Documents (Size: ~{parent_chunk_size} tokens).")

            parent_points = []
            all_child_points = []

            # --- 3. GENERAZIONE VETTORI E PAYLOAD GERARCHICO ---
            for p_idx, p_doc in enumerate(parent_docs):
                
                parent_id = str(uuid.uuid4())
                
                # A. Crea il punto per il Parent (Vettore fittizio, salviamo solo il payload testuale)
                parent_points.append(
                    models.PointStruct(
                        id=parent_id,
                        vector=[0.0] * 768, # Vettore dummy per soddisfare lo schema di Qdrant
                        payload={
                            "topic_id": topic_id,
                            "sub_topic_id": sub_topic_id,
                            "source": source_name,
                            "parent_index": p_idx,
                            "content": p_doc.page_content,
                            **p_doc.metadata
                        }
                    )
                )

                # B. Dividi questo Parent specifico in Child Chunks
                child_docs = child_text_splitter.create_documents([p_doc.page_content])
                batch_texts = [c.page_content for c in child_docs]
                
                if not batch_texts:
                    continue

                log.debug(f"Parent {p_idx+1}/{len(parent_docs)}: Generated {len(child_docs)} Child Chunks. Embedding...")
                
                # Vettorializza i Child in batch per rispettare i limiti API
                child_vectors = []
                for i in range(0, len(batch_texts), 50):
                    child_vectors.extend(await async_embed_batch(batch_texts[i:i+50]))

                # C. Crea i punti vettoriali per i Child, collegandoli al parent_id
                for c_idx, vec in enumerate(child_vectors):
                    all_child_points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload={
                                "parent_id": parent_id, # <-- IL COLLEGAMENTO CHIAVE
                                "topic_id": topic_id,
                                "sub_topic_id": sub_topic_id,
                                "source": source_name,
                                "child_index": c_idx,
                                "content": child_docs[c_idx].page_content,
                                **child_docs[c_idx].metadata
                            }
                        )
                    )
            
            # --- 4. QDRANT UPSERT ---
            log.info(f"Upsert phase: Preparing {len(parent_points)} Parents and {len(all_child_points)} Children for source: {source_name}")
            
            # Condizione di filtro per rimuovere i vecchi dati associati a questa source
            cleanup_filter = models.FilterSelector(  
                filter=models.Filter(must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=source_name)),
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                    models.FieldCondition(key="sub_topic_id", match=models.MatchValue(value=sub_topic_id))
                ])
            )

            # Cleanup vecchi vettori dalla collezione Child
            await qdrant_client.delete(collection_name=QDRANT_COLLECTION, points_selector=cleanup_filter)
            # Cleanup vecchi vettori dalla collezione Parent
            await qdrant_client.delete(collection_name=PARENT_COLLECTION, points_selector=cleanup_filter)
            log.debug("Cleared previous vectors for source (if any).")
            
            # Inserimento dei Parent in batch
            for i in range(0, len(parent_points), 100):
                await qdrant_client.upsert(collection_name=PARENT_COLLECTION, points=parent_points[i:i+100])

            # Inserimento dei Child in batch
            for i in range(0, len(all_child_points), 100):
                await qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=all_child_points[i:i+100])

            # Cleanup metadata file se esiste
            if meta_path.exists(): await asyncio.to_thread(meta_path.unlink)
            
            log.info(f"SUCCESS: Finished processing and indexing '{filename}'.")
            await finalize_file_move(file_path, root_folder, topic_id, sub_topic_id, success=True)
            return True

        except Exception as e:
            log.error(f"Process Failure for '{filename}': {e}", exc_info=True)

            log_file = file_path.with_suffix(".log")
            error_msg = f"Critical Ingestion Exception:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            await asyncio.to_thread(log_file.write_text, error_msg, encoding="utf-8")

            await finalize_file_move(file_path, root_folder, topic_id, sub_topic_id, success=False)
            return False
        
        
async def finalize_file_move(file_path, root_folder, topic_id, sub_topic_id, success):
    try:
        try: 
            relative_path = file_path.relative_to(root_folder)
        except ValueError: 
            relative_path = pathlib.Path(file_path.name)

        dest_root = PROCESSED_FOLDER if success else ERROR_FOLDER
        
        final_dest = dest_root / topic_id / sub_topic_id / relative_path
        
        final_dest.parent.mkdir(parents=True, exist_ok=True)
        
        log.debug(f"Finalizing file: moving main file to {final_dest.parent}")
        await asyncio.to_thread(safe_move_file, file_path, final_dest.parent)

        for ext in [".ocr", ".md"]:
            artifact_sibling = file_path.with_suffix(file_path.suffix + ext)
            if artifact_sibling.exists():
                log.debug(f"Finalizing file: moving artifact {artifact_sibling.name} to {final_dest.parent}")
                await asyncio.to_thread(safe_move_file, artifact_sibling, final_dest.parent)

        error_log_sibling = file_path.with_suffix(".log")
        if error_log_sibling.exists():
            await asyncio.to_thread(safe_move_file, error_log_sibling, final_dest.parent)
            
        status_tag = '[SUCCESS]' if success else '[ERROR]'
        log.info(f"File lifecycle complete: {status_tag} {file_path.name}")
    except Exception as e:
        log.error(f"Failed during finalize_file_move for {file_path.name}: {e}")

async def process_topic_folder_async(topic_id, sub_topic_id, folder_path): 
    folder_path = pathlib.Path(folder_path)
    log.info(f"====== Topic Start: '{topic_id}' -> Sub: '{sub_topic_id}' ======")
    
    # 1. Recupero Configurazione dal DB (FAIL-FAST)
    # Interroghiamo il DB per ottenere i parametri necessari allo splitting
    config = await asyncio.to_thread(get_subtopic_config, topic_id, sub_topic_id)
    
    if not config or not config.get('chunk_size'):
        log.error(f"CONFIG ERROR: Configurazione mancante per '{sub_topic_id}'. Chunk_size non trovato.")
        
        # Recuperiamo tutti i file presenti per spostarli nella cartella di errore
        all_files = [
            p for p in folder_path.rglob("*") 
            if p.is_file() and not p.name.endswith((".meta", ".ocr", ".md")) and p.name != "urls.txt"
        ]
        
        for p in all_files:
            log.warning(f"Spostamento file '{p.name}' in ERROR a causa di configurazione DB mancante.")
            # Scriviamo un file .log per l'utente/admin
            error_log = p.with_suffix(".log")
            await asyncio.to_thread(
                error_log.write_text, 
                f"Errore: chunk_size non configurato nel DB per il sub_topic_id '{sub_topic_id}'. "
                "Configura i parametri prima di tentare nuovamente l'ingestione.", 
                encoding="utf-8"
            )
            # Sposta fisicamente il file nella cartella ERROR specifica
            await finalize_file_move(p, folder_path, topic_id, sub_topic_id, success=False)
        
        log.error(f"====== Topic Aborted: '{sub_topic_id}' (Config Missing) ======")
        return

    # Estrazione parametri 
    chunk_size = config['chunk_size'] or 300  # Default a 300 se non specificato
    chunk_overlap = config.get('chunk_overlap') or 50  # Default a 50 se non specificato
    parent_size = config.get('parent_chunk_size') or 1500  # Fallback a 1500
    log.info(f"Configurazione attiva per '{sub_topic_id}': Size={chunk_size}, Overlap={chunk_overlap}")
    
    # 2. Gestione URL (Scraping parallelo)
    url_file = folder_path / "urls.txt"
    if url_file.exists():
        log.info(f"Scraper: Rilevato file {url_file.name}. Inizio estrazione URL.")
        with open(url_file, 'r') as f:
            urls = [l.strip() for l in f if l.strip().startswith("http")]

        if urls:
            log.debug(f"Scraper: Trovati {len(urls)} URL da processare.")
            # Eseguiamo lo scraping in parallelo
            url_tasks = [scrape_web_page_async(u, folder_path) for u in urls]
            url_results = await asyncio.gather(*url_tasks)

            for i, text in enumerate(url_results):
                if text:
                    v_name = f"web_{uuid.uuid4().hex[:8]}.txt"
                    v_path = folder_path / v_name
                    log.debug(f"Scraper: Salvataggio contenuto web in {v_name}")
                    with open(v_path, 'w', encoding='utf-8') as f_v: f_v.write(text)
                    with open(str(v_path) + ".meta", 'w', encoding='utf-8') as f_m: f_m.write(urls[i])
        
        # Sposta il file urls.txt nei processati dopo averlo letto
        await asyncio.to_thread(safe_move_file, str(url_file), PROCESSED_FOLDER / topic_id)

    # 3. Gestione File Batch
    # Scannerizziamo la cartella cercando file che non siano artifact (.meta, .ocr, .md)
    all_files_gen = (
        p for p in folder_path.rglob("*") 
        if p.is_file() and not p.name.endswith((".meta", ".ocr", ".md")) and p.name != "urls.txt"
    )
    
    results = []
    batch = []
    BATCH_LIMIT = 5  # Numero di file processati simultaneamente per non saturare la RAM

    log.info(f"Ingest: Inizio elaborazione batch file per '{sub_topic_id}'...")

    for p in all_files_gen:
        # Passiamo i parametri dinamici chunk_size e chunk_overlap
        batch.append(process_single_file_async(topic_id, sub_topic_id, p, folder_path, chunk_size, chunk_overlap, parent_size))
        
        if len(batch) >= BATCH_LIMIT:
            log.debug(f"Ingest: Esecuzione batch di {len(batch)} file...")
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            batch = [] 
            
    if batch:
        log.debug(f"Ingest: Esecuzione ultimo batch di {len(batch)} file...")
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    # 4. Aggiornamento Metadati e Pulizia
    if any(results): 
        log.info(f"Ingest: Batch completato. Successi: {sum(1 for r in results if r)}.")
        # Registra la presenza del topic se non esiste (logica generica)
        #await asyncio.to_thread(register_topic_safe, topic_id)
    else:
        log.warning(f"Ingest: Nessun file è stato processato con successo per '{sub_topic_id}'.")
    
    log.debug(f"Cleanup: Rimozione cartelle vuote in {folder_path}...")
    def _cleanup():
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in dirs:
                try: 
                    os.rmdir(os.path.join(root, name))
                except OSError: 
                    pass # Cartella non vuota, ignoriamo
    await asyncio.to_thread(_cleanup)

    log.info(f"====== Topic End: '{sub_topic_id}' (Processati: {len(results)}) ======")


async def main_run():
    log.info("Ingestion Worker Active - Beginning One Shot Run")
    try:
        await initialize_qdrant_collection()

        folders = [f for f in os.listdir(WATCH_FOLDER) if (WATCH_FOLDER / f).is_dir()]
        if not folders:
            log.info("No topic folders found in watch directory. Exiting.")
            return

        for tid in folders:
            topic_path = WATCH_FOLDER / tid
            sub_folders = [f for f in os.listdir(topic_path) if (topic_path / f).is_dir()]
            for sub_tid in sub_folders:
                # Rimossa la creazione automatica: register_sub_topic_safe(tid, sub_tid)
                # Ora l'ingestion pretende che il sub-topic esista e sia configurato nel DB.
                
                # Processa i file passando il sub_topic_id
                await process_topic_folder_async(tid, sub_tid, topic_path / sub_tid)
        
        log.info("One Shot Run Finished Successfully.")
    finally:
        # Ensures connection pool is closed to prevent OS socket leaks
        await http_client.aclose()

if __name__ == "__main__":
    asyncio.run(main_run())
