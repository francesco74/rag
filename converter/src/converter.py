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

# --- Dipendenze AI importate da ingest.py ---
from google import genai
from google.genai import types
from google.cloud import vision
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from dotenv import load_dotenv

load_dotenv()

# Configurazione dinamica del logging
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - CONVERTER - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
log = logging.getLogger("ConverterWorker")

# --- CONFIGURAZIONI ---
BASE_DIR = Path("./data")
DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", str(BASE_DIR)))
STAGING_DIRECT = DATA_FOLDER / "staging" / "direct"
STAGING_JSON = DATA_FOLDER / "staging" / "json"
INGESTION_WATCH_DIR = DATA_FOLDER / "watch"
ERROR_DIR = BASE_DIR / "error"
ARCHIVE_DIR = DATA_FOLDER / "archive"

CROP_OCR_LIMIT = 0
OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "gemini-3-flash-preview")
GEMINI_LIMITER = AsyncLimiter(max_rate=10, time_period=60)

# Inizializzazione del nuovo client Google GenAI
api_key = os.environ.get("GOOGLE_API_KEY") 
client = genai.Client(api_key=api_key)

vision_client = vision.ImageAnnotatorClient()

def setup_dirs():
    for d in [STAGING_DIRECT, STAGING_JSON, INGESTION_WATCH_DIR, ERROR_DIR, ARCHIVE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log.debug(f"Directory di sistema verificate in: {BASE_DIR.absolute()}")

def safe_move(src: Path, dest: Path):
    if dest.exists(): 
        log.debug(f"Sovrascrittura file esistente in destinazione: {dest.name}")
        dest.unlink()
    shutil.move(str(src), str(dest))
    log.debug(f"File spostato: {src.name} -> {dest.parent.name}/")

# ==============================================================================
# LOGICA IBRIDA EREDITATA DA INGEST.PY
# ==============================================================================

def _cloud_vision_fallback(image: Image.Image) -> str:
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
    async with GEMINI_LIMITER:
        prompt = "Transcribe the text in this image precisely. Format the output strictly as Markdown." if as_markdown else "Transcribe the text in this image precisely as raw text."
        
        log.debug("Chiamata a Gemini Vision API inviata.")
        try:
            response = await client.aio.models.generate_content(
                model=OCR_MODEL_NAME,
                contents=[prompt, image_input],
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        )
                    ]
                )
            )

            if not response.candidates or not response.candidates[0].content.parts:
                reason_enum = response.candidates[0].finish_reason if response.candidates else None
                reason = reason_enum.name if reason_enum else "Unknown"
                
                if reason in ['SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', '3', '4']:
                    log.warning(f"Gemini API bloccata per policy (Reason: {reason}). Avvio fallback visivo.")
                    fallback_text = await asyncio.to_thread(_cloud_vision_fallback, image_input)
                    return fallback_text.strip()
                raise ValueError(f"Gemini API Blocked Page. Exact Finish Reason: {reason}")
            
            log.debug("Risposta Gemini Vision ricevuta con successo.")
            return response.text.strip().replace("-\n", "")
            
        except Exception as e:
            log.error("Errore critico durante la generazione OCR con Gemini.", exc_info=True)
            raise ValueError(f"OCR Generation Failed: {e}")

async def extract_hybrid_markdown_from_pdf_async(file_path: Path, file_bytes: bytes = None) -> str:
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
# LOGICA DI ROUTING E MERGE
# ==============================================================================

async def process_direct_folder():
    """Elabora file orfani in direct mantenendo l'alberatura delle directory."""
    log.info("--- Controllo coda DIRECT ---")
    
    files = list(STAGING_DIRECT.rglob("*.*"))
    if not files:
        return

    for file_path in files:
        if not file_path.is_file(): continue
        
        ext = file_path.suffix.lower()
        rel_path = file_path.relative_to(STAGING_DIRECT)
        base_name = rel_path.stem
        
        dest_dir = INGESTION_WATCH_DIR / rel_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        archive_target_dir = ARCHIVE_DIR / rel_path.parent
        archive_target_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Inizio processo file diretto: {rel_path}")
        
        try:
            target_text_file = None

            if ext in [".txt", ".md"]:
                target_text_file = dest_dir / file_path.name
                # Copia il file per il watcher, poi sposta l'originale in archive
                shutil.copy2(file_path, target_text_file)
                safe_move(file_path, archive_target_dir / file_path.name)
                
            elif ext == ".pdf":
                md_content = await extract_hybrid_markdown_from_pdf_async(file_path)
                if md_content:
                    target_text_file = dest_dir / f"{base_name}.md"
                    target_text_file.write_text(md_content, encoding="utf-8")
                # Archiviazione del file originale invece di .unlink()
                safe_move(file_path, archive_target_dir / file_path.name)
                
            else:
                log.warning(f"File diretto ignorato (estensione non valida per la conversione): {rel_path}")
                err_dir = ERROR_DIR / rel_path.parent
                err_dir.mkdir(parents=True, exist_ok=True)
                safe_move(file_path, err_dir / file_path.name)
                continue
                
            if target_text_file and target_text_file.exists():
                unique_source = f"direct://{str(rel_path.with_suffix('')).replace(os.sep, '/')}"
                
                json_manifest = {
                    "source": unique_source,
                    "files": [file_path.name],
                    "metadati": {
                        "percorso_originale": str(rel_path)
                    }
                }
                json_dest = dest_dir / f"{base_name}.json"
                with open(json_dest, "w", encoding="utf-8") as f:
                    json.dump(json_manifest, f, indent=2)
                log.info(f"File e JSON generati in {dest_dir.name}")

        except Exception as e:
            log.error(f"Errore irreversibile nell'elaborazione del file diretto {rel_path}", exc_info=True)
            err_dir = ERROR_DIR / rel_path.parent
            err_dir.mkdir(parents=True, exist_ok=True)
            safe_move(file_path, err_dir / file_path.name)

async def process_json_folder():
    """Elabora gruppi da Manifest JSON esplodendoli in file 1:1."""
    log.info("--- Controllo coda JSON (Manifests) ---")
    
    manifests = list(STAGING_JSON.rglob("*.json"))
    if not manifests:
        return

    for json_path in manifests:
        rel_path = json_path.relative_to(STAGING_JSON)
        base_name = rel_path.stem
        
        dest_dir = INGESTION_WATCH_DIR / rel_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        archive_target_dir = ARCHIVE_DIR / rel_path.parent
        archive_target_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Avvio esplosione manifesto: {rel_path}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            attached_files = manifest.get("files", [])
            original_source = manifest.get("source", str(rel_path))
            current_json_dir = json_path.parent 
            
            for filename in attached_files:
                file_path = current_json_dir / filename
                
                if not file_path.exists(): 
                    log.warning(f"Allegato mancante nella cartella {current_json_dir.name}: {filename}")
                    continue

                log.info(f"Estrazione singolo allegato: {filename}")
                ext = file_path.suffix.lower()
                md_content = None

                # Pipeline di conversione
                if ext == ".pdf":
                    md_content = await extract_hybrid_markdown_from_pdf_async(file_path)
                elif ext in [".txt", ".md"]:
                    md_content = file_path.read_text(encoding="utf-8", errors="ignore")
                else:
                    log.warning(f"Estensione ignorata durante la conversione: {ext} per il file {filename}")
                
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
                        
                # Archiviazione del file originale invece di .unlink()
                safe_move(file_path, archive_target_dir / file_path.name)

            # Archiviazione anche del JSON/manifest originario (raggruppatore)
            if json_path.exists():
                safe_move(json_path, archive_target_dir / json_path.name)

        except Exception as e:
            log.error(f"Fallimento critico nell'elaborazione del manifest {rel_path}", exc_info=True)
            err_dir = ERROR_DIR / rel_path.parent
            err_dir.mkdir(parents=True, exist_ok=True)
            safe_move(json_path, err_dir / json_path.name)

async def main_run():
    setup_dirs()
    log.info("=== START: Servizio Normalizzazione Documentale (Ibrido) ===")
    await process_direct_folder()
    await process_json_folder()
    log.info("=== STOP: Ciclo terminato ===")

if __name__ == "__main__":
    asyncio.run(main_run())