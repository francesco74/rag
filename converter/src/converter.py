import os
import json
import shutil
import logging
import asyncio
import io
from pathlib import Path
from PIL import Image

import fitz          # PyMuPDF
import pymupdf4llm   # Native PDF to Markdown
import aio_pika      # Asynchronous RabbitMQ client

# --- Dipendenze AI importate ---
from google import genai
from google.genai import types
from google.cloud import vision
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from dotenv import load_dotenv
import html
import re

load_dotenv()

# ==============================================================================
# 1. CONFIGURAZIONE E LOGGING
# ==============================================================================

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - CONVERTER - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
log = logging.getLogger("ConverterWorker")

# --- DIRECTORY SYSTEM ---
BASE_DIR = Path("./data")
DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", str(BASE_DIR)))
STAGING_DIR = DATA_FOLDER / "staging"
INGESTION_WATCH_DIR = DATA_FOLDER / "watch"
ERROR_DIR = DATA_FOLDER / "converter" / "error"
ARCHIVE_DIR = DATA_FOLDER / "converter" / "archive"

# Assicuriamo che le directory esistano allo startup
for d in [STAGING_DIR, INGESTION_WATCH_DIR, ERROR_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CROP_OCR_LIMIT = 0
OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "gemini-3-flash-preview")

BROKER_HOST = os.environ.get("BROKER_HOST", "rabbitmq-service.rag.svc.cluster.local")
BROKER_PORT = int(os.environ.get("BROKER_PORT", 5672))
BROKER_USERNAME = os.environ.get("BROKER_USERNAME", "guest")
BROKER_PASSWORD = os.environ.get("BROKER_PASSWORD", "guest")

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# Limiter asincrono: gestisce internamente la concorrenza garantendo max 10 chiamate/minuto
GEMINI_LIMITER = AsyncLimiter(max_rate=10, time_period=60)

# Inizializzazione Client AI
api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
vision_client = vision.ImageAnnotatorClient()


def safe_move(src: Path, dest: Path):
    """Sposta i file in modo sicuro con sovrascrittura se necessario."""
    if dest.exists(): 
        log.debug(f"Sovrascrittura file esistente in destinazione: {dest.name}")
        dest.unlink()
    shutil.move(str(src), str(dest))
    log.debug(f"File spostato: {src.name} -> {dest.parent.name}/")

# ==============================================================================
# 2. MOTORE IBRIDO DI ESTRAZIONE E OCR
# ==============================================================================

def _cloud_vision_fallback(image: Image.Image) -> str:
    """Esegue il fallback su Google Cloud Vision API in caso di blocco policy."""
    log.info("Esecuzione fallback su Google Cloud Vision API...")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    vision_image = vision.Image(content=img_byte_arr.getvalue())
    response = vision_client.document_text_detection(image=vision_image)
    if response.error.message:
        log.error(f"Errore Cloud Vision API: {response.error.message}")
        raise Exception(f"Cloud Vision API Error: {response.error.message}")
    log.debug("Fallback Cloud Vision completato con successo.")
    return response.full_text_annotation.text

@retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_random_exponential(multiplier=2, min=10, max=80),
    stop=stop_after_attempt(20)
)
async def async_ocr_generate(image_input, as_markdown=False) -> str:
    """Interroga Gemini Vision API rispettando i limiti di rate e gestendo i fallback."""
    async with GEMINI_LIMITER:
        prompt = "Transcribe the text in this image precisely. Format the output strictly as Markdown." if as_markdown else "Transcribe the text in this image precisely as raw text."
        
        log.debug("Chiamata a Gemini Vision API inviata.")
        try:
            response = await client.aio.models.generate_content(
                model=OCR_MODEL_NAME,
                contents=[prompt, image_input],
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)
                    ]
                )
            )

            if not response.candidates or not response.candidates[0].content.parts:
                reason_enum = response.candidates[0].finish_reason if response.candidates else None
                reason = reason_enum.name if reason_enum else "Unknown"
                
                if reason in ['SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', '3', '4']:
                    log.warning(f"Gemini API bloccata per policy (Reason: {reason}). Avvio fallback visivo.")
                    if not isinstance(image_input, Image.Image):
                        log.error("Fallback Cloud Vision richiede un PIL.Image. Tipo ricevuto: %s", type(image_input))
                        raise ValueError("Tipo immagine non supportato per il fallback.")
                    fallback_text = await asyncio.to_thread(_cloud_vision_fallback, image_input)
                    
                    return fallback_text.strip()
                raise ValueError(f"Gemini API Blocked Page. Exact Finish Reason: {reason}")
            
            log.debug("Risposta Gemini Vision ricevuta con successo.")
            return response.text.strip().replace("-\n", "")
            
        except Exception as e:
            log.error("Errore critico durante la generazione OCR con Gemini.", exc_info=True)
            raise ValueError(f"OCR Generation Failed: {e}")

async def extract_hybrid_markdown_from_pdf_async(file_path: Path, file_bytes: bytes = None) -> str:
    """Estrae il testo nativo dal PDF o innesca l'OCR per le pagine scansionate."""
    md_pages = []
    
    def evaluate_and_extract_page(doc, p_num):
        page = doc[p_num]
        rect = page.rect
        clip_rect = fitz.Rect(0, CROP_OCR_LIMIT, rect.width, rect.height - CROP_OCR_LIMIT) if rect.height > 200 else rect
        text = page.get_text(sort=True, clip=clip_rect).strip()
        
        char_count = len(text)
        if char_count < 50:
            log.debug(f"Pagina {p_num+1}: Rilevata bassa densità di testo ({char_count} chars). Tagging per elaborazione OCR.")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, clip=clip_rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return {"type": "image", "content": img}
        else:
            log.debug(f"Pagina {p_num+1}: Rilevato testo nativo sufficiente ({char_count} chars). Estrazione Markdown.")
            page_md = pymupdf4llm.to_markdown(doc, pages=[p_num])
            page_md = page_md.replace("<br>", "\n")
            
            return {"type": "markdown", "content": page_md}

    log.info(f"Avvio estrazione ibrida per: {file_path.name}")
    
    try:
        if file_bytes:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        else:
            doc = fitz.open(file_path)
            
        total_pages = len(doc)
        log.debug(f"Documento aperto. Pagine totali: {total_pages}")

        for page_num in range(total_pages):
            page_data = await asyncio.to_thread(evaluate_and_extract_page, doc, page_num)
            
            if page_data["type"] == "image":
                gemini_md = await async_ocr_generate(page_data["content"], as_markdown=True)
                md_pages.append(gemini_md)
            else:
                md_pages.append(page_data["content"])

        doc.close()
        log.info(f"Estrazione ibrida completata per {file_path.name}. Generati {len(md_pages)} blocchi.")
        return "\n\n---\n\n".join(md_pages)
        
    except Exception as e:
        log.error(f"Fallimento durante l'apertura/lettura del PDF {file_path.name}", exc_info=True)
        raise

# ==============================================================================
# 3. CORE LOGIC - GESTIONE MESSAGGI (EVENT-DRIVEN)
# ==============================================================================

async def process_single_job(payload: dict, channel: aio_pika.Channel):
    """
    Riceve il task decodificato. Cerca il file direttamente in STAGING_DIR.
    """
    source_type = payload.get("source_type")
    rel_path_str = payload.get("rel_path")
    
    if source_type != "json" or not rel_path_str:
        log.error(f"Payload invalido o non supportato: {payload}")
        return

    rel_path = Path(rel_path_str)
    
    # La risoluzione del percorso ora è immediata
    file_path = STAGING_DIR / rel_path
    
    if not file_path.exists():
        log.warning(f"Manifesto JSON non trovato sul filesystem: {file_path}. Ignorato.")
        return

    if file_path.suffix.lower() != ".json":
        log.error(f"Il Converter accetta solo file .json. Trovato: {file_path.name}")
        err_dir = ERROR_DIR / rel_path.parent
        err_dir.mkdir(parents=True, exist_ok=True)
        safe_move(file_path, err_dir / file_path.name)
        return

    base_name = file_path.stem
    dest_dir = INGESTION_WATCH_DIR / rel_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_target_dir = ARCHIVE_DIR / rel_path.parent
    archive_target_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Elaborazione manifesto JSON iniziata: {rel_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        attached_files = manifest.get("files", [])
        original_source = manifest.get("source", str(rel_path))
        current_json_dir = file_path.parent
        
        for filename in attached_files:
            attached_file_path = current_json_dir / filename
            
            if not attached_file_path.exists():
                log.warning(f"Allegato mancante {filename} nella cartella {current_json_dir.name}.")
                continue

            log.info(f"Conversione allegato: {filename}")
            attached_ext = attached_file_path.suffix.lower()
            md_content = None

            # Pipeline di conversione
            if attached_ext == ".pdf":
                md_content = await extract_hybrid_markdown_from_pdf_async(attached_file_path)
            elif attached_ext in [".txt", ".md"]:
                md_content = attached_file_path.read_text(encoding="utf-8", errors="ignore")
            elif attached_ext in IMG_EXTENSIONS:
                log.info(f"Rilevata immagine {filename}. Avvio OCR nativo...")
                with Image.open(attached_file_path) as img:
                    md_content = await async_ocr_generate(img, as_markdown=True)
            else:
                log.warning(f"Estensione {attached_ext} ignorata per il file {filename}.")
            
            if md_content:
                safe_stem = Path(filename).stem
                unique_base_name = f"{base_name}_{safe_stem}"
                
                final_md_path = dest_dir / f"{unique_base_name}.md"
                final_md_path.write_text(md_content, encoding="utf-8")
                
                child_manifest = manifest.copy()
                child_manifest["source"] = f"{original_source}::{filename}"
                child_manifest["files"] = [filename]
                
                final_json_path = dest_dir / f"{unique_base_name}.json"
                with open(final_json_path, "w", encoding="utf-8") as f:
                    json.dump(child_manifest, f, indent=2)
                
                # Notifica ad Ingest
                ingest_payload = {"json_manifest_path": str(final_json_path.relative_to(INGESTION_WATCH_DIR))}
                await channel.default_exchange.publish(
                    aio_pika.Message(body=json.dumps(ingest_payload).encode(), delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
                    routing_key="da-indicizzare"
                )
                log.info(f"✓ Notificato Ingest per file generato: {unique_base_name}.json")

            # Archiviazione allegato processato
            safe_move(attached_file_path, archive_target_dir / attached_file_path.name)

        # Archiviazione manifesto radice
        safe_move(file_path, archive_target_dir / file_path.name)
        log.info(f"✓ Manifesto JSON '{file_path.name}' completato e archiviato con successo.")

    except Exception as e:
        log.error(f"Fallimento durante l'esplosione del manifest {rel_path}: {e}", exc_info=True)
        err_dir = ERROR_DIR / rel_path.parent
        err_dir.mkdir(parents=True, exist_ok=True)
        safe_move(file_path, err_dir / file_path.name)

# ==============================================================================
# 4. ORCHESTRAZIONE MESSAGE BROKER E LOOP PRINCIPALE
# ==============================================================================

async def on_message_received(message: aio_pika.IncomingMessage, channel: aio_pika.Channel):
    """Callback triggered all'arrivo di ogni messaggio da RabbitMQ."""
    async with message.process(ignore_processed=True):  # ← non auto-ack
        try:
            payload = json.loads(message.body.decode())
            log.debug(f"Payload RabbitMQ Ricevuto: {payload}")
            await process_single_job(payload, channel)
            await message.ack()
        except json.JSONDecodeError as e:
            log.error(f"Impossibile decodificare il messaggio: {e}. Scartato in DLQ.")
            await message.reject(requeue=False)
        except Exception as e:
            log.error(f"Errore durante l'elaborazione: {e}", exc_info=True)
            await message.reject(requeue=False)

async def main_worker():
    
    
    log.info(f"Avvio Worker. Tentativo di connessione a RabbitMQ su {BROKER_HOST}...")
    
    try:
        connection = await aio_pika.connect_robust(f"amqp://{BROKER_USERNAME}:{BROKER_PASSWORD}@{BROKER_HOST}:{BROKER_PORT}/")
        
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)

            queue_in = await channel.get_queue("da-convertire")  # ← assume che esista già

            log.info("✓ Connessione stabilita. In ascolto sulla coda 'da-convertire'.")

            async with queue_in.iterator() as queue_iter:
                async for message in queue_iter:
                    await on_message_received(message, channel)
                    
    except Exception as e:
        log.error(f"Errore critico di connessione a RabbitMQ: {e}")
        raise

if __name__ == "__main__":
    log.info("=== START: Servizio Normalizzazione Documentale (Event-Driven) ===")
    try:
        asyncio.run(main_worker())
    except KeyboardInterrupt:
        log.info("Interruzione manuale ricevuta. Spegnimento gracefull in corso...")
    finally:
        log.info("=== STOP: Worker terminato ===")