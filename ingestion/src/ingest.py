import os
import shutil
import logging
import uuid
import pathlib
import asyncio
import json
import hashlib

# --- EMBEDDING (centralizzato in embedding.py) ---
from common.embedding import init_embedding, embed_documents_batch

# --- VECTOR DB ---
from qdrant_client import models
from qdrant_client import AsyncQdrantClient

# --- TEXT SPLITTING ---
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

# --- ASYNC, RESILIENCE & MESSAGING ---
import aio_pika
from aiolimiter import AsyncLimiter
import aiormq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_exponential,
    retry_if_exception_type
)

from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool
from common.config import settings

# ==============================================================================
# 1. CONFIGURATION & LOGGING SETUP
# ==============================================================================

BASE_DIR = pathlib.Path(__file__).parent.resolve()

QDRANT_COLLECTION = "document_chunks"

init_db_pool()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format='%(asctime)s - INGESTION - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
log = logging.getLogger("IngestWorker")

# Aggiunta handler su database
db_handler = MySQLLogHandler()
db_handler.setLevel(logging.WARNING)
db_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
log.addHandler(db_handler)

# --- DIRECTORY LOGIC ---
DATA_FOLDER = settings.data_folder
WATCH_FOLDER = DATA_FOLDER / "watch"
PROCESSED_FOLDER = DATA_FOLDER / "processed"
ERROR_FOLDER = DATA_FOLDER / "ingestion" / "error"

for folder in [DATA_FOLDER, WATCH_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# --- AI & DB Init ---
# Client di embedding centralizzato in embedding.py (stesso SDK/modello del
# worker). Lo stesso oggetto client espone sia il path sync sia .aio per l'uso
# asincrono in embed_documents_batch.
init_embedding()

qdrant_client = AsyncQdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    # Alza il timeout complessivo a 2 minuti e configura httpx in modo specifico
    timeout=120.0
)

# Concurrency & Limiting
CONCURRENCY_LIMIT = asyncio.Semaphore(5)
# Rate limiter generico lato ingestion, NON specifico di un provider: con
# EMBEDDING_PROVIDER=gemini protegge la quota API di Gemini, ma lo stesso
# codice gira anche con openai/mistral (altra quota API) o local (nessuna
# quota reale — un modello self-hosted non ha rate limit esterno, il limite
# è solo throughput hardware). 1000/min resta un tetto di sicurezza
# ragionevole in tutti i casi, quindi non condizionato al provider attivo.
EMBEDDING_RATE_LIMITER = AsyncLimiter(max_rate=1000, time_period=60)

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

# ==============================================================================
# 2. CORE HELPER FUNCTIONS
# ==============================================================================

@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60), # Attese: 4s, 8s, 16s, 32s, 60s...
    stop=stop_after_attempt(5),                         # Massimo 5 tentativi
    reraise=True                                        # Rilancia l'errore se fallisce definitivamente
)
async def process_with_retry(payload: dict):
    """Esegue il job ritentando con backoff esponenziale in caso di eccezioni non previste."""
    log.info(f"Tentativo di elaborazione payload: {payload.get('json_manifest_path')}")
    await process_single_job(payload)

def get_subtopic_config(topic_id, sub_topic_id):
    """Recupera i parametri di chunking dal DB."""
    conn = get_db_connection()
    if not conn: 
        raise ConnectionError("Database temporaneamente non raggiungibile o pool non pronto.")
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                """SELECT chunk_size, chunk_overlap, parent_chunk_size, use_markdown_splitter 
                   FROM sub_topics 
                   WHERE topic_id = %s AND sub_topic_id = %s""", 
                (topic_id, sub_topic_id)
            )
            return cursor.fetchone() 
    finally:
        conn.close()

def safe_move_file(src_path, dest_folder):
    """Muove fisicamente il file, sovrascrivendo se necessario."""
    try:
        src = pathlib.Path(src_path)
        dest = pathlib.Path(dest_folder) / src.name
        if not src.exists(): return 
        if dest.exists(): dest.unlink()
        shutil.move(str(src), str(dest))
    except Exception as e: 
        log.error(f"File Move Error ({src_path}): {e}")

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(multiplier=2, min=10, max=80),
    stop=stop_after_attempt(20)
)
async def async_embed_batch(batch_texts):
    """Genera embeddings in batch con backoff esponenziale in caso di rate limit.

    La chiamata all'API (modello, RETRIEVAL_DOCUMENT, dimensioni) è centralizzata
    in embedding.embed_documents_batch; qui restano le politiche specifiche
    dell'ingestione: rate limiter globale e retry aggressivo."""
    if not batch_texts: return []
    async with EMBEDDING_RATE_LIMITER:
        log.debug(f"Calling Embeddings API for a batch of {len(batch_texts)} chunks...")
        return await embed_documents_batch(batch_texts)

async def finalize_file_move(file_path, root_folder, topic_id, sub_topic_id, error_msg: str = None):
    """Smista i file processati. Se error_msg è presente, lo inietta nel JSON."""
    success = error_msg is None
    try:
        try: 
            relative_path = file_path.relative_to(root_folder)
        except ValueError: 
            relative_path = pathlib.Path(file_path.name)

        dest_root = PROCESSED_FOLDER if success else ERROR_FOLDER
        final_dest = dest_root / topic_id / sub_topic_id / relative_path
        final_dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Se è un fallimento ed è il JSON, iniettiamo l'errore
        if not success and file_path.suffix.lower() == '.json' and file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Aggiungiamo il tracciato dell'errore
                data['_ingestion_error'] = str(error_msg)
                
                # Scriviamo direttamente nella destinazione per evitare race condition
                with open(final_dest, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                # Cancelliamo l'originale
                file_path.unlink()
            except Exception as e:
                log.error(f"Impossibile iniettare l'errore nel JSON {file_path.name}: {e}")
                await asyncio.to_thread(safe_move_file, file_path, final_dest.parent)
        else:
            # Spostamento standard per successi o per file non-JSON (.md, .txt)
            await asyncio.to_thread(safe_move_file, file_path, final_dest.parent)
            
        status_tag = '[SUCCESS]' if success else '[ERROR]'
        log.info(f"File lifecycle complete: {status_tag} {file_path.name}")
    except Exception as e:
        log.error(f"Failed during finalize_file_move for {file_path.name}: {e}")

def sync_upsert_parents_mysql(source_name, topic_id, sub_topic_id, parents_data):
    """
    Cancella i vecchi parent per idempotenza e inserisce i nuovi in batch frazionati.
    parents_data è una lista di tuple: (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata_json)
    """
    conn = get_db_connection()
    if not conn:
        raise Exception("Impossibile connettersi a MySQL per il salvataggio dei parent_documents.")
    
    try:
        with conn.cursor() as cursor:
            # 1. Cancellazione atomica preventiva (idempotenza)
            delete_query = """
                DELETE FROM parent_documents 
                WHERE source = %s AND topic_id = %s AND sub_topic_id = %s
            """
            cursor.execute(delete_query, (source_name, topic_id, sub_topic_id))
            
            # 2. Inserimento massivo dei nuovi parent suddiviso in chunk (Evita max_allowed_packet)
            if parents_data:
                insert_query = """
                    INSERT INTO parent_documents 
                    (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Sotto-chunk conservativi (es. 200 righe alla volta) poiché il campo 'content' può essere pesante
                MYSQL_BATCH_SIZE = 200 
                for i in range(0, len(parents_data), MYSQL_BATCH_SIZE):
                    chunk = parents_data[i:i + MYSQL_BATCH_SIZE]
                    cursor.executemany(insert_query, chunk)
        
        conn.commit()
    except Exception as e:
        log.error(f"Errore durante l'upsert dei parent in MySQL: {e}")
        # Gestione resiliente del rollback: evita di crashare se la connessione è già caduta
        try:
            if conn and conn.is_connected():
                conn.rollback()
        except Exception as rollback_err:
            log.warning(f"Impossibile eseguire il rollback (connessione persa): {rollback_err}")
        raise e
    finally:
        # Chiusura sicura senza far arrabbiare il session reset del pooler
        try:
            if conn and conn.is_connected():
                conn.close()
        except Exception as close_err:
            log.warning(f"Errore durante la chiusura della connessione MySQL: {close_err}")

# ==============================================================================
# 3. PIPELINE ORCHESTRATION (DOCUMENT PROCESSING)
# ==============================================================================

# Funzione di isolamento protetta da retry per gestire i ReadError/Timeout di Qdrant
# ==============================================================================
# Funzione retry Qdrant estratta a livello di modulo (non più ridefinita ad ogni chiamata)
# ==============================================================================

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    stop=stop_after_attempt(5),
    reraise=True
)
async def _qdrant_upsert_with_retry(points_batch):
    await qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points_batch
    )

# ==============================================================================
# 3. PIPELINE ORCHESTRATION (DOCUMENT PROCESSING)
# ==============================================================================

async def process_single_file_async(topic_id, sub_topic_id, json_path, root_folder, chunk_size, chunk_overlap, parent_chunk_size, use_markdown_splitter=True):
    async with CONCURRENCY_LIMIT:
        json_path = pathlib.Path(json_path)
        base_name = json_path.stem
        log.info(f"--- Trigger pacchetto rilevato dal JSON: {json_path.name} ---")

        text_file_path = json_path.with_suffix(".md")
        is_markdown = True

        if not text_file_path.exists():
            text_file_path = json_path.with_suffix(".txt")
            is_markdown = False

        if not text_file_path.exists():
            log.error(f"HARD STOP: File di testo mancante per {json_path.name}.")
            await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg="File di testo nativo (.md/.txt) mancante.")
            return False

        # --- Parsing del manifest JSON ---
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)

            source_name = raw_json.get("source")
            if not source_name or not source_name.strip():
                log.error(f"HARD STOP: Chiave 'source' mancante o vuota nel manifest {json_path.name}.")
                await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg="Chiave 'source' mancante o vuota nel manifest.")
                await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, error_msg="Chiave 'source' mancante o vuota nel manifest.")
                return False

            files_array = raw_json.get("files", [])
            if not isinstance(files_array, list) or not files_array:
                log.error(f"HARD STOP: Array 'files' mancante o vuoto nel manifest {json_path.name}.")
                await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg="Array 'files' mancante o vuoto nel manifest.")
                await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, error_msg="Array 'files' mancante o vuoto nel manifest.")
                return False

            file_name = str(files_array[0])

            extra_metadata = raw_json.get("metadati", {})
            if not isinstance(extra_metadata, dict):
                extra_metadata = {}

            conflicting_keys = set(extra_metadata.keys()) & settings.protected_keys
            if conflicting_keys:
                err = f"Chiavi riservate trovate nei metadati extra: {conflicting_keys}. Ingestione bloccata."
                log.error(f"HARD STOP: {err}")
                await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg=err)
                await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, error_msg=err)
                return False

        except Exception as e:
            log.error(f"Errore critico nel parsing JSON {json_path.name}: {e}")
            await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg=f"JSON Parsing Error: {str(e)}")
            await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, error_msg="JSON Parsing Error")
            return False

        # --- Pipeline di indicizzazione ---
        try:
            # Lettura del testo — necessaria tutta in RAM per lo splitting
            full_text = await asyncio.to_thread(text_file_path.read_text, encoding='utf-8')
            if not full_text.strip():
                log.warning(f"Contenuto file vuoto per '{text_file_path.name}'. Scartato.")
                await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg="Contenuto file vuoto.")
                await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id, error_msg="Contenuto file vuoto.")
                return False

            parent_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=parent_chunk_size, chunk_overlap=0,
                separators=["\n\n", "\n", ". ", " "]
            )
            child_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            if is_markdown and use_markdown_splitter:
                log.info("Applico MarkdownHeaderTextSplitter come da configurazione DB.")
                md_header_splits = await asyncio.to_thread(markdown_splitter.split_text, full_text)
                parent_docs = await asyncio.to_thread(parent_text_splitter.split_documents, md_header_splits)
                del md_header_splits
            elif is_markdown:
                parent_docs = await asyncio.to_thread(parent_text_splitter.create_documents, [full_text])
            else:
                parent_docs = await asyncio.to_thread(parent_text_splitter.create_documents, [full_text])

            # Libera subito la stringa originale: non serve più
            del full_text

            log.info(f"Chunking: {len(parent_docs)} Parent Documents generati.")

            def build_payload(base_dict, extra_meta):
                payload = base_dict.copy()
                for key, value in extra_meta.items():
                    if key in settings.protected_keys:
                        log.warning(f"Metadato '{key}' nel JSON ignorato: chiave di sistema riservata.")
                        continue
                    payload[key] = value
                return payload

            ingestion_id = str(uuid.uuid4())

            # ---------------------------------------------------------------
            # FIX MEMORIA: Processing per batch di parent
            # Invece di accumulare tutto il documento in RAM, embeddiamo e
            # scarichiamo su Qdrant ogni PARENT_BATCH_SIZE parent, poi buttiamo via.
            # I mysql_parents_data sono leggeri (no vettori) e li teniamo tutti.
            # ---------------------------------------------------------------
            PARENT_BATCH_SIZE = 50

            mysql_parents_data = []
            pending_child_points = []  # Buffer per flush intermedi su Qdrant

            for batch_start in range(0, len(parent_docs), PARENT_BATCH_SIZE):
                batch_parents = parent_docs[batch_start:batch_start + PARENT_BATCH_SIZE]
                batch_child_payloads = []
                batch_texts = []

                for p_idx_offset, p_doc in enumerate(batch_parents):
                    p_idx = batch_start + p_idx_offset
                    parent_id = str(uuid.uuid4())
                    merged_metadata = {**p_doc.metadata, **extra_metadata}

                    mysql_parents_data.append((
                        parent_id, topic_id, sub_topic_id, source_name, file_name,
                        p_idx, p_doc.page_content, json.dumps(merged_metadata)
                    ))

                    child_docs = await asyncio.to_thread(child_text_splitter.create_documents, [p_doc.page_content])
                    for c_idx, c_doc in enumerate(child_docs):
                        content_norm = " ".join(c_doc.page_content.lower().split())
                        content_hash = hashlib.md5(content_norm.encode()).hexdigest()

                        batch_child_payloads.append({
                            **c_doc.metadata,
                            "parent_id": parent_id,
                            "topic_id": topic_id,
                            "sub_topic_id": sub_topic_id,
                            "source": source_name,
                            "file_name": file_name,
                            "child_index": c_idx,
                            "content": c_doc.page_content,
                            "content_hash": content_hash, 
                            "_ingestion_id": ingestion_id
                        })
                        batch_texts.append(c_doc.page_content)

                # Embedding per questo batch
                batch_vectors = []
                if batch_texts:
                    for i in range(0, len(batch_texts), 100):
                        sub_batch = batch_texts[i:i + 100]
                        batch_vectors.extend(await async_embed_batch(sub_batch))

                # Assembla PointStruct e accoda al buffer
                for vec, payload in zip(batch_vectors, batch_child_payloads):
                    pending_child_points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload=build_payload(payload, extra_metadata)
                        )
                    )

                # Flush intermedio su Qdrant: scarica e libera RAM
                if len(pending_child_points) >= 500:
                    log.info(f"Qdrant flush intermedio: {len(pending_child_points)} punti (batch parent {batch_start}).")
                    for i in range(0, len(pending_child_points), 100):
                        await _qdrant_upsert_with_retry(pending_child_points[i:i + 100])
                    pending_child_points.clear()

                # Libera il batch corrente prima di passare al prossimo
                del batch_parents, batch_child_payloads, batch_texts, batch_vectors
                log.debug(f"Batch parent [{batch_start}:{batch_start + PARENT_BATCH_SIZE}] completato.")

            # Libera i parent_docs ora che abbiamo finito il loop
            total_parents = len(parent_docs)
            del parent_docs

            # --- Flush finale dei punti rimasti nel buffer ---
            if pending_child_points:
                log.info(f"Qdrant flush finale: {len(pending_child_points)} punti.")
                for i in range(0, len(pending_child_points), 100):
                    await _qdrant_upsert_with_retry(pending_child_points[i:i + 100])
                pending_child_points.clear()

            # --- Salvataggio Parent su MySQL ---
            log.info(f"MySQL Upsert: {len(mysql_parents_data)} Parent Documents.")
            await asyncio.to_thread(sync_upsert_parents_mysql, source_name, topic_id, sub_topic_id, mysql_parents_data)

            # --- Cleanup vecchi punti Qdrant (idempotenza) ---
            # Elimina i punti con stesso source/topic/subtopic ma ingestion_id diverso
            cleanup_filter = models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="source",       match=models.MatchValue(value=source_name)),
                        models.FieldCondition(key="topic_id",     match=models.MatchValue(value=topic_id)),
                        models.FieldCondition(key="sub_topic_id", match=models.MatchValue(value=sub_topic_id)),
                    ],
                    must_not=[
                        models.FieldCondition(key="_ingestion_id", match=models.MatchValue(value=ingestion_id)),
                    ]
                )
            )
            await qdrant_client.delete(collection_name=QDRANT_COLLECTION, points_selector=cleanup_filter)

            log.info(f"SUCCESS: Indicizzazione completata per source '{source_name}' — {total_parents} parent, ingestion_id={ingestion_id}.")
            await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id)
            await finalize_file_move(text_file_path, root_folder, topic_id, sub_topic_id)
            return True

        except Exception as e:
            log.error(f"Fallimento critico durante l'indicizzazione del pacchetto '{base_name}': {e}", exc_info=True)
            raise


# ==============================================================================
# 4. EVENT-DRIVEN ORCHESTRATION (RABBITMQ)
# ==============================================================================

async def process_single_job(payload: dict):
    """Analizza il payload di RabbitMQ ed avvia l'ingestione se il file è valido."""
    rel_path_str = payload.get("json_manifest_path")
    
    if not rel_path_str:
        log.error(f"Payload malformato. Chiave 'json_manifest_path' mancante: {payload}")
        return

    rel_path = pathlib.Path(rel_path_str)
    
    # Path Resolution Dinamica: topic_id e sub_topic_id sono SEMPRE i primi
    # due segmenti del path (rel_path.parts[0]/[1]), indipendentemente da
    # quante sottocartelle seguano prima del file (es.
    # "topic/sub/preliminari/gennaio/file.json" con direct.py, non solo
    # "topic/sub/file.json" come nella pipeline Sicr@Web). In precedenza si
    # usava rel_path.parent.parent.name (le ULTIME due cartelle prima del
    # file), che per path più profondi di 3 livelli restituiva le cartelle
    # sbagliate (es. "preliminari"/"gennaio" invece di "topic"/"sub").
    # finalize_file_move() più sotto è già agnostico alla profondità (usa
    # relative_to(root_folder)), quindi non necessita di modifiche.
    if len(rel_path.parts) < 3:
        log.error(f"Struttura path non supportata (attesa: topic/sub_topic/.../file.json). Ricevuta: {rel_path}")
        return

    topic_id = rel_path.parts[0]
    sub_topic_id = rel_path.parts[1]
    
    json_path = WATCH_FOLDER / rel_path
    root_folder = WATCH_FOLDER / topic_id / sub_topic_id

    if not json_path.exists():
        log.warning(f"File non trovato in watch dir: {json_path}. Probabilmente già processato o cancellato.")
        return

    # Recupero dinamico dei parametri via DB
    config = await asyncio.to_thread(get_subtopic_config, topic_id, sub_topic_id)
    if not config:
        log.error(f"CONFIG ERROR: Dati non trovati nel database per il sub_topic '{sub_topic_id}'. Ingestione annullata.")
        await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg="sub_topic non trovata in DB.")
        # Muove anche l'md/txt associato in errore se esiste
        text_md = json_path.with_suffix(".md")
        if text_md.exists(): await finalize_file_move(text_md, root_folder, topic_id, sub_topic_id, error_msg="sub_topic non trovata in DB.")
        text_txt = json_path.with_suffix(".txt")
        if text_txt.exists(): await finalize_file_move(text_txt, root_folder, topic_id, sub_topic_id, error_msg="sub_topic non trovata in DB.")
        return

    chunk_size = config.get('chunk_size') if config.get('chunk_size') is not None else 500
    chunk_overlap = config.get('chunk_overlap') if config.get('chunk_overlap') is not None else 100
    parent_size = config.get('parent_chunk_size') or 1500
    use_md_splitter = bool(config.get('use_markdown_splitter', True))

    log.info(f"Job Iniziato: {json_path.name} | Topic: {topic_id} | SubTopic: {sub_topic_id}")

    await process_single_file_async(
        topic_id=topic_id, 
        sub_topic_id=sub_topic_id, 
        json_path=json_path, 
        root_folder=root_folder, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        parent_chunk_size=parent_size,
        use_markdown_splitter=use_md_splitter
    )

    
async def on_message_received(message: aio_pika.IncomingMessage):
    """
    Gestisce il ciclo di vita del messaggio: Ack se ok, Reject (verso DLQ) se fallisce dopo i retry.
    """
    # ignore_processed=True ci permette di chiamare manualmente ack() o reject()
    async with message.process(ignore_processed=True):
        payload = {}
        try:
            payload = json.loads(message.body.decode())
            
            # Avviamo il processing con i retry
            await process_with_retry(payload)
            
            # Se arriviamo qui, l'elaborazione è andata a buon fine
            await message.ack()
            
        except json.JSONDecodeError as e:
            log.error(f"Decode JSON Fallito: {e}. Payload irrecuperabile, invio a DLQ.")
            await message.reject(requeue=False)

        except (aiormq.exceptions.ChannelInvalidStateError, aiormq.exceptions.AMQPConnectionError) as e:
            # Il canale è morto: RabbitMQ farà il redeliver da solo quando il canale/connessione
            # si chiude senza ack. Non tentare ack/reject, logga soltanto.
            log.warning(f"Canale invalido durante il processing di un messaggio: {e}. "
                        f"Il broker farà redeliver automaticamente.")
        except Exception as e:
            # Se siamo qui, tutti i 5 tentativi di Tenacity sono falliti.
            log.error(f"Fallimento definitivo dopo i retry. Spostamento in DLQ (rag_dlx). Errore: {e}", exc_info=True)
            
            rel_path_str = payload.get("json_manifest_path")
            if rel_path_str:
                rel_path = pathlib.Path(rel_path_str)
                # Stessa correzione applicata sopra: primi due segmenti, non
                # le ultime due cartelle (vedi commento in process_single_job).
                if len(rel_path.parts) < 3:
                    log.error(f"Struttura path non supportata in DLQ handler (attesa: topic/sub_topic/.../file.json). Ricevuta: {rel_path}")
                else:
                    topic_id = rel_path.parts[0]
                    sub_topic_id = rel_path.parts[1]
                    root_folder = WATCH_FOLDER / topic_id / sub_topic_id
                    json_path = WATCH_FOLDER / rel_path

                    # Sposta JSON e il file di testo associato
                    await finalize_file_move(json_path, root_folder, topic_id, sub_topic_id, error_msg=str(e))
                    for ext in [".md", ".txt"]:
                        text_path = json_path.with_suffix(ext)
                        if text_path.exists():
                            await finalize_file_move(text_path, root_folder, topic_id, sub_topic_id, error_msg=str(e))
            
            # Invio in Dead Letter Queue per ispezione
            await message.reject(requeue=False)

async def main_worker():
    """Worker principale per RabbitMQ."""
    log.info(f"Tentativo di connessione a RabbitMQ su {settings.broker_host}...")
    
    try:
        connection = await aio_pika.connect_robust(
            f"amqp://{settings.broker_username}:{settings.broker_password}@{settings.broker_host}:{settings.broker_port}/",
            heartbeat=120  
        )
        async with connection:
            channel = await connection.channel()
            
            # FIX 1: Alziamo il prefetch a 20. RabbitMQ invia fino a 20 messaggi in anticipo 
            # al worker, permettendo al semaforo (5) di lavorare a pieno regime senza tempi morti.
            await channel.set_qos(prefetch_count=20)

            queue_in = await channel.get_queue("da-indicizzare") 
            
            log.info("✓ Worker Ingestione attivo. In ascolto su 'da-indicizzare'.")
            
            async with queue_in.iterator() as queue_iter:
                async for message in queue_iter:
                    # FIX 2: create_task avvia l'elaborazione in background SENZA bloccare il ciclo di ricezione.
                    # Il ciclo preleverà istantaneamente i prossimi messaggi, scatenando la vera concorrenza.
                    background_tasks = set()
                    
                    task = asyncio.create_task(on_message_received(message))
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
    except Exception as e:
        log.error(f"Errore critico di connettività broker: {e}")
        raise

if __name__ == "__main__":
    log.info("=== START: Servizio Ingestione (Event-Driven) ===")
    try:
        asyncio.run(main_worker())
    except KeyboardInterrupt:
        log.info("Interruzione catturata. Spegnimento worker...")
    finally:
        log.info("=== STOP: Worker spento ===")