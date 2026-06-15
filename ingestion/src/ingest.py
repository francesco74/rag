import os
import shutil
import logging
import uuid
import pathlib
import asyncio
import json

# --- NUOVO SDK GOOGLE GENAI ---
from google import genai
from google.genai import types

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
    retry_if_exception_type
)
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from db_logger import MySQLLogHandler, get_db_connection
from dotenv import load_dotenv

load_dotenv()


# ==============================================================================
# 1. CONFIGURATION & LOGGING SETUP
# ==============================================================================

BASE_DIR = pathlib.Path(__file__).parent.resolve()

QDRANT_COLLECTION = "document_chunks"
PARENT_COLLECTION = "parent_documents"

PROTECTED_KEYS = {"topic_id", "sub_topic_id", "source", "parent_id", "content",
                  "parent_index", "child_index", "file_name"}

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper() 
logging.basicConfig(
    level=getattr(logging, log_level_str, logging.INFO),
    format='%(asctime)s - INGESTION - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
log = logging.getLogger("IngestWorker")

db_handler = MySQLLogHandler()
db_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
log.addHandler(db_handler)

# --- DIRECTORY LOGIC ---
DATA_FOLDER = pathlib.Path(os.environ.get("DATA_FOLDER", str(BASE_DIR)))
WATCH_FOLDER = DATA_FOLDER / "watch"
PROCESSED_FOLDER = DATA_FOLDER / "processed"
ERROR_FOLDER = DATA_FOLDER / "error"

for folder in [DATA_FOLDER, WATCH_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# --- AI & DB Init ---
# Inizializzazione pulita tramite il nuovo modulo google.genai
ai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Aggiornamento al modello Embedding di punta
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

# NOTA: La creazione delle collection è delegata a setup_qdrant.py
qdrant_client = AsyncQdrantClient(
    host=os.environ.get("QDRANT_HOST", "localhost"), 
    port=int(os.environ.get("QDRANT_PORT", 6333)),
    timeout=60.0
)

CONCURRENCY_LIMIT = asyncio.Semaphore(5)
GEMINI_LIMITER = AsyncLimiter(max_rate=100, time_period=60)

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

# ==============================================================================
# 2. CORE HELPER FUNCTIONS
# ==============================================================================

def get_subtopic_config(topic_id, sub_topic_id):
    conn = get_db_connection()
    if not conn: return None
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                "SELECT chunk_size, chunk_overlap, parent_chunk_size FROM sub_topics WHERE topic_id = %s AND sub_topic_id = %s", 
                (topic_id, sub_topic_id)
            )
            return cursor.fetchone() 
    finally:
        conn.close()

def safe_move_file(src_path, dest_folder):
    try:
        src = pathlib.Path(src_path)
        dest = pathlib.Path(dest_folder) / src.name
        if not src.exists(): return 
        if dest.exists(): dest.unlink()
        shutil.move(str(src), str(dest))
    except Exception as e: 
        log.error(f"File Move Error ({src_path}): {e}")

@retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_random_exponential(multiplier=2, min=10, max=80),
    stop=stop_after_attempt(20)
)
async def async_embed_batch(batch_texts):
    if not batch_texts: return []
    async with GEMINI_LIMITER:
        log.debug(f"Calling Embeddings API for a batch of {len(batch_texts)} chunks...")
        
        # Nuova sintassi google.genai con output_dimensionality a 768
        # per allinearsi al setup_qdrant.py
        response = await ai_client.aio.models.embed_content(
            model=EMBEDDING_MODEL_NAME, 
            contents=batch_texts, 
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        return [emb.values for emb in response.embeddings]

# ==============================================================================
# 3. PIPELINE ORCHESTRATION
# ==============================================================================

async def process_single_file_async(topic_id, sub_topic_id, json_path, root_folder, chunk_size, chunk_overlap, parent_chunk_size):
    async with CONCURRENCY_LIMIT:
        json_path = pathlib.Path(json_path)
        base_name = json_path.stem
        log.info(f"--- Trigger pacchetto rilevato dal JSON: {json_path.name} ---")

        # 1. Risoluzione deterministica del file di testo normalizzato (.md o .txt)
        text_file_path = json_path.with_suffix(".md")
        is_markdown = True
        
        if not text_file_path.exists():
            text_file_path = json_path.with_suffix(".txt")
            is_markdown = False
            
        if not text_file_path.exists():
            log.error(f"File di testo (.md/.txt) mancante per il manifest {json_path.name}. Ingestione abortita.")
            await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, success=False)
            return False

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
            
            source_name = raw_json.get("source")
            
            if not source_name or not source_name.strip():
                log.error(f"HARD STOP: Chiave 'source' mancante o vuota nel manifest {json_path.name}.")
                await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, success=False)
                if text_file_path.exists():
                    await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, success=False)
                return False

            # --- ESTRAZIONE NOME FILE
            files_array = raw_json.get("files", [])
            if not isinstance(files_array, list) or not files_array:
                log.error(f"HARD STOP: Array 'files' mancante o vuoto nel manifest {json_path.name}.")
                await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, success=False)
                if text_file_path.exists():
                    await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, success=False)
                return False
                
            file_name = str(files_array[0])
            # -----------------------------------------------------

            extra_metadata = raw_json.get("metadati", {})
            if not isinstance(extra_metadata, dict):
                extra_metadata = {}
                
        except Exception as e:
            log.error(f"Errore critico nel parsing del file JSON {json_path.name}: {e}")
            await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, success=False)
            if text_file_path.exists():
                await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, success=False)
            return False

        # 3. Lettura del testo normalizzato ed esecuzione del chunking
        try:
            full_text = await asyncio.to_thread(text_file_path.read_text, encoding='utf-8')
            if not full_text.strip(): 
                log.warning(f"Contenuto del file di testo vuoto per '{text_file_path.name}'. File scartato.")
                await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, success=False)
                await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, success=False)
                return False
            
            parent_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=parent_chunk_size, chunk_overlap=0, separators=["\n\n", "\n", ". ", " "]
            )
            child_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
            )

            if is_markdown:
                md_header_splits = markdown_splitter.split_text(full_text)
                parent_docs = parent_text_splitter.split_documents(md_header_splits)
            else:
                parent_docs = parent_text_splitter.create_documents([full_text])
            
            log.info(f"Generati {len(parent_docs)} Parent Documents (Token size: ~{parent_chunk_size}).")

            parent_points = []
            all_child_points = []

            def build_payload(base_dict, extra_meta):
                payload = base_dict.copy()
                for key, value in extra_meta.items():
                    if key in PROTECTED_KEYS:
                        log.warning(f"Metadato '{key}' nel JSON ignorato: chiave di sistema riservata.")
                        continue
                    payload[key] = value
                return payload
        
            # 4. Generazione dei vettori
            for p_idx, p_doc in enumerate(parent_docs):
                parent_id = str(uuid.uuid4())
                
                # --- INIEZIONE IN PARENT ---
                parent_base_payload = {
                    "topic_id": topic_id, 
                    "sub_topic_id": sub_topic_id, 
                    "source": source_name,
                    "file_name": file_name,
                    "parent_index": p_idx, 
                    "content": p_doc.page_content, 
                    **p_doc.metadata
                }
                
                parent_points.append(
                    models.PointStruct(
                        id=parent_id, vector=[0.0] * 768, payload=build_payload(parent_base_payload, extra_metadata)
                    )
                )

                child_docs = child_text_splitter.create_documents([p_doc.page_content])
                batch_texts = [c.page_content for c in child_docs]
                
                if not batch_texts:
                    continue

                log.debug(f"Parent {p_idx+1}/{len(parent_docs)}: Richiesta embedding per {len(child_docs)} Child Chunks...")
                
                child_vectors = []
                for i in range(0, len(batch_texts), 250):
                    child_vectors.extend(await async_embed_batch(batch_texts[i:i+250]))

                for c_idx, vec in enumerate(child_vectors):
                    
                    # --- INIEZIONE IN CHILD ---
                    child_base_payload = {
                        "parent_id": parent_id, 
                        "topic_id": topic_id, 
                        "sub_topic_id": sub_topic_id,
                        "source": source_name, 
                        "file_name": file_name, 
                        "child_index": c_idx, 
                        "content": child_docs[c_idx].page_content,
                        **child_docs[c_idx].metadata
                    }
                    
                    all_child_points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()), vector=vec, payload=build_payload(child_base_payload, extra_metadata)
                        )
                    )

            # 5. Upsert su Qdrant
            log.info(f"Fase Upsert Qdrant: Preparazione di {len(parent_points)} Parents e {len(all_child_points)} Children.")
            
            cleanup_filter = models.FilterSelector(  
                filter=models.Filter(must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=source_name)),
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                    models.FieldCondition(key="sub_topic_id", match=models.MatchValue(value=sub_topic_id))
                ])
            )

            await qdrant_client.delete(collection_name=QDRANT_COLLECTION, points_selector=cleanup_filter)
            await qdrant_client.delete(collection_name=PARENT_COLLECTION, points_selector=cleanup_filter)
            
            if parent_points:
                for i in range(0, len(parent_points), 100):
                    await qdrant_client.upsert(collection_name=PARENT_COLLECTION, points=parent_points[i:i+100])

            if all_child_points:
                for i in range(0, len(all_child_points), 100):
                    await qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=all_child_points[i:i+100])

            log.info(f"SUCCESS: Indicizzazione completata con successo per source '{source_name}'.")
            await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, success=True)
            await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, success=True)
            return True

        except Exception as e:
            log.error(f"Fallimento critico durante l'indicizzazione del pacchetto '{base_name}': {e}", exc_info=True)
            await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, success=False)
            await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, success=False)
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
        
        await asyncio.to_thread(safe_move_file, file_path, final_dest.parent)

        # RIMOSSO IL CONTROLLO SUL JSON SIBLING PER EVITARE RACE CONDITIONS
            
        status_tag = '[SUCCESS]' if success else '[ERROR]'
        log.info(f"File lifecycle complete: {status_tag} {file_path.name}")
    except Exception as e:
        log.error(f"Failed during finalize_file_move for {file_path.name}: {e}")

async def process_topic_folder_async(topic_id, sub_topic_id, folder_path): 
    folder_path = pathlib.Path(folder_path)
    log.info(f"====== Topic Start: '{topic_id}' -> Sub: '{sub_topic_id}' ======")
    
    config = await asyncio.to_thread(get_subtopic_config, topic_id, sub_topic_id)
    if not config or not config.get('chunk_size'):
        log.error(f"CONFIG ERROR: Configurazione mancante per '{sub_topic_id}'.")
        return

    chunk_size = config['chunk_size'] or 500
    chunk_overlap = config.get('chunk_overlap') or 100
    parent_size = config.get('parent_chunk_size') or 1500
    
    all_files_gen = (p for p in folder_path.rglob("*") if p.is_file() and p.name.endswith(".json"))
    
    results = []
    batch = []
    BATCH_LIMIT = 10 

    for json_file in all_files_gen:
        batch.append(process_single_file_async(topic_id, sub_topic_id, json_file, folder_path, chunk_size, chunk_overlap, parent_size))
        
        if len(batch) >= BATCH_LIMIT:
            results.extend(await asyncio.gather(*batch))
            batch = [] 
            
    if batch:
        results.extend(await asyncio.gather(*batch))

    log.info(f"====== Topic End: '{sub_topic_id}' (Processati: {len(results)}) ======")

async def main_run():
    log.info("Ingestion Worker Active - Beginning One Shot Run")
    folders = [f for f in os.listdir(WATCH_FOLDER) if (WATCH_FOLDER / f).is_dir()]
    
    for tid in folders:
        topic_path = WATCH_FOLDER / tid
        sub_folders = [f for f in os.listdir(topic_path) if (topic_path / f).is_dir()]
        for sub_tid in sub_folders:
            await process_topic_folder_async(tid, sub_tid, topic_path / sub_tid)
    
    log.info("One Shot Run Finished Successfully.")

if __name__ == "__main__":
    asyncio.run(main_run())