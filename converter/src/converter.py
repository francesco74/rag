import os
import json
import shutil
import logging
import asyncio
import io
import time
from pathlib import Path
from PIL import Image
from common.db_logger import MySQLLogHandler, init_db_pool

import fitz          # PyMuPDF
import pymupdf4llm   # Native PDF to Markdown
import aio_pika      # Asynchronous RabbitMQ client

# --- Dipendenze AI importate ---

from google.cloud import vision
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import concurrent.futures
import multiprocessing as mp
import queue

from dotenv import load_dotenv
from common.config import settings
import html
import re

load_dotenv()


def _detect_cgroup_cpu_quota() -> int:
    """
    Legge il limite CPU effettivo del cgroup (v2 o v1) e restituisce un
    numero intero di CPU utilizzabili (arrotondato per difetto, minimo 1).

    Serve a impostare correttamente intra_op_num_threads/inter_op_num_threads
    di ONNX Runtime: di default ONNX Runtime auto-rileva il numero di CPU
    del SISTEMA HOST (non del cgroup del container), quindi in un pod con
    CPU limit basso (es. 1-2 core) può creare molti più thread di quanti
    core siano realmente disponibili, causando thrashing e rallentamenti
    drastici e imprevedibili sotto contesa.
    """
    try:
        # cgroup v2: "/sys/fs/cgroup/cpu.max" -> "<quota> <period>" oppure "max <period>"
        cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max_path.exists():
            content = cpu_max_path.read_text().strip().split()
            quota, period = content[0], content[1]
            if quota != "max":
                return max(1, int(int(quota) / int(period)))

        # cgroup v1: "cpu.cfs_quota_us" / "cpu.cfs_period_us"
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_path.exists() and period_path.exists():
            quota = int(quota_path.read_text().strip())
            period = int(period_path.read_text().strip())
            if quota > 0:
                return max(1, quota // period)
    except Exception:
        pass

    # Nessun limite cgroup rilevato (o non containerizzato): usa le CPU del sistema.
    return max(1, os.cpu_count() or 1)


ONNX_THREAD_LIMIT = _detect_cgroup_cpu_quota()


def _limit_onnx_threads():
    """
    Monkey-patch di onnxruntime.SessionOptions: forza intra_op_num_threads e
    inter_op_num_threads sul limite CPU reale del cgroup (ONNX_THREAD_LIMIT)
    su OGNI sessione creata, incluse quelle interne di pymupdf.layout (il
    modulo AI di pymupdf4llm), senza dover modificare i file di quella
    libreria in site-packages (che verrebbero sovrascritti a ogni reinstall).
    Va chiamata una volta per processo, prima di invocare pymupdf4llm.to_markdown().
    """
    import onnxruntime as ort

    if getattr(ort.SessionOptions, "_thread_limit_patched", False):
        return  # già applicato in questo processo

    original_init = ort.SessionOptions.__init__

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.intra_op_num_threads = ONNX_THREAD_LIMIT
        self.inter_op_num_threads = ONNX_THREAD_LIMIT

    ort.SessionOptions.__init__ = _patched_init
    ort.SessionOptions._thread_limit_patched = True

# --- DIRECTORY SYSTEM ---
DATA_FOLDER = settings.data_folder
STAGING_DIR = DATA_FOLDER / "staging"
INGESTION_WATCH_DIR = DATA_FOLDER / "watch"
ERROR_DIR = DATA_FOLDER / "converter" / "error"
ARCHIVE_DIR = DATA_FOLDER / "converter" / "archive"

PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=2)

FAULTHANDLER_INTERVAL = 30  # secondi - intervallo di dump automatico dello stack se il worker resta bloccato

MP_CTX = mp.get_context("spawn")


def _killable_worker_entry(q, target_func, args):
    """
    Entry point eseguito nel processo dedicato: chiama target_func e mette
    il risultato in coda.

    Usa faulthandler.dump_traceback_later come "allarme" di auto-diagnosi:
    se il worker resta bloccato, ogni FAULTHANDLER_INTERVAL secondi stampa
    da solo su stderr lo stack trace Python corrente (file/riga/funzione
    esatti), finché non termina. Non richiede ptrace/SYS_PTRACE né processi
    esterni allegati: è il processo stesso che si auto-ispeziona, quindi
    funziona senza toccare la configurazione del container.
    """
    import faulthandler

    faulthandler.dump_traceback_later(FAULTHANDLER_INTERVAL, repeat=True, exit=False)
    try:
        result = target_func(*args)
        q.put(("ok", result))
    except Exception as e:
        q.put(("error", str(e)))
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_killable(target_func, args: tuple, timeout: float):
    """
    Esegue target_func(*args) in un PROCESSO DEDICATO (non nel PROCESS_POOL
    condiviso) con un timeout che, se scaduto, TERMINA DAVVERO il processo
    (SIGTERM, poi SIGKILL se necessario).

    Questo risolve un problema strutturale di ProcessPoolExecutor +
    asyncio.wait_for: se il worker va in hang, wait_for si limita a
    cancellare l'ATTESA lato asyncio, ma il processo del pool resta
    bloccato e occupa uno slot per sempre (con max_workers=2 bastano
    2 file "maledetti" per bloccare l'intero servizio).

    IMPORTANTE: aspettiamo il risultato leggendo DIRETTAMENTE dalla Queue
    (q.get(timeout=...)), NON con proc.join(timeout) seguito da q.get().
    Se il worker mette in coda un payload più grande del buffer della pipe
    del sistema operativo (es. immagini rasterizzate di pagine PDF), il
    processo figlio si blocca in q.put() aspettando che qualcuno legga.
    Se nel frattempo il genitore fosse bloccato solo su proc.join(), senza
    mai leggere la coda, si creerebbe un DEADLOCK vero e proprio: il figlio
    aspetta che il genitore legga, il genitore aspetta che il figlio finisca.
    Leggendo dalla coda fin da subito evitiamo questo scenario.

    Va chiamata da codice sincrono (gira in un thread separato via
    asyncio.to_thread, così non blocca l'event loop mentre aspetta).
    """
    q = MP_CTX.Queue()
    proc = MP_CTX.Process(target=_killable_worker_entry, args=(q, target_func, args))
    proc.start()

    try:
        status, payload = q.get(timeout=timeout)
    except queue.Empty:
        log.warning(f"Worker PID {proc.pid} non ha risposto entro {timeout}s: terminazione forzata (hang rilevato).")
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            log.warning(f"Worker PID {proc.pid} non risponde a SIGTERM: invio SIGKILL.")
            proc.kill()
            proc.join()
        raise TimeoutError(f"Worker terminato forzatamente dopo {timeout}s")

    # Il risultato è arrivato: diamo al processo il tempo di terminare in modo
    # pulito (ha già fatto q.put(), sta solo per uscire da run()).
    proc.join(5)
    if proc.is_alive():
        log.warning(f"Worker PID {proc.pid} ha consegnato il risultato ma non è terminato entro 5s: termino comunque.")
        proc.terminate()
        proc.join()

    if status == "error":
        raise RuntimeError(f"Errore nel processo worker: {payload}")
    return payload


# ==============================================================================
# 1. CONFIGURAZIONE E LOGGING
# ==============================================================================
init_db_pool()

log_level_str = settings.log_level
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - CONVERTER - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
log = logging.getLogger("ConverterWorker")

db_handler = MySQLLogHandler()
db_handler.setLevel(logging.WARNING) 
db_handler.setFormatter(logging.Formatter('%(asctime)s - CONVERTER - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
log.addHandler(db_handler)

# Assicuriamo che le directory esistano allo startup
for d in [STAGING_DIR, INGESTION_WATCH_DIR, ERROR_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CROP_OCR_LIMIT = 0
OCR_MODEL_NAME = settings.ocr_model_name
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]

EXTRACTION_TIMEOUT = 1200.0        # 20 minuti - estrazione ibrida (testo/vettori + OCR mirato)
FALLBACK_RENDER_TIMEOUT = 300.0    # 5 minuti - estrazione di fallback per-pagina (senza get_drawings, deve essere più rapida)
MAX_CHARS_FOR_AI_LAYOUT = 70_000   # oltre questa soglia una pagina è quasi certamente una planimetria/disegno tecnico, non testo normale: si salta il layout AI (ONNX) e si usa il testo grezzo
                                    # NB: soglia abbassata da 200_000 a 70_000 dopo aver osservato hang di pymupdf4llm.to_markdown()
                                    # (layout AI/ONNX) già a ~139k caratteri su una singola pagina (planimetrie CAD dense di etichette/quote).
VECTOR_COUNT_LAYOUT_SKIP = 500     # soglia secondaria: molti tracciati vettoriali + testo già cospicuo è un altro segnale forte
                                    # di disegno tecnico, indipendentemente dal superamento di MAX_CHARS_FOR_AI_LAYOUT.
VECTOR_COUNT_MIN_CHARS_SKIP = 50_000

# Limiter asincrono: gestisce internamente la concorrenza garantendo max 10 chiamate/minuto
GEMINI_LIMITER = AsyncLimiter(max_rate=10, time_period=60)

# Inizializzazione Client AI
api_llm_key = settings.api_llm_key
client = genai.Client(api_key=api_llm_key)
vision_client = vision.ImageAnnotatorClient()

def _cpu_heavy_pdf_extraction(file_path_str: str, file_bytes: bytes, crop_limit: int):
    """
    Funzione isolata per ProcessPoolExecutor.
    Gira in un processo OS separato, bypassando totalmente il GIL.
    """
    import fitz
    import pymupdf4llm

    # Manteniamo attivo il modulo di layout AI (qualità migliore su
    # tabelle/multi-colonna), ma limitiamo i thread ONNX Runtime al numero
    # di CPU realmente assegnate dal cgroup (vedi _detect_cgroup_cpu_quota):
    # di default ONNX Runtime auto-rileva le CPU dell'HOST, non quelle del
    # container, generando più thread di quanti core siano disponibili e
    # causando thrashing sotto contesa (probabile causa reale degli hang).
    _limit_onnx_threads()

    print(f"[CHILD PROCESS] Avvio estrazione per {file_path_str} (ONNX thread limit: {ONNX_THREAD_LIMIT})")
    
    
    if file_bytes:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    else:
        doc = fitz.open(file_path_str)

    extracted_pages = []
    print(f"[CHILD PROCESS] Estrazione PDF iniziata per {file_path_str} ({len(doc)} pagine).")
    # Teniamo traccia dello stato corrente di pymupdf4llm.use_layout() per evitare
    # toggle ridondanti: riattivare il layout AI richiama pymupdf.layout.activate(),
    # che ha un costo di inizializzazione non banale, quindi lo facciamo solo quando
    # lo stato desiderato cambia rispetto a quello corrente.
    layout_ai_active = True  # pymupdf4llm ha il layout AI attivo di default all'import
    for p_num in range(len(doc)):
        page_t0 = time.time()
        page = doc[p_num]
        try:
            # get_drawings() estrae i tracciati vettoriali (linee, poligoni, curve CAD)
            vector_count = len(page.get_drawings())
        except Exception:
            vector_count = 0
        
        rect = page.rect
        clip_rect = fitz.Rect(0, crop_limit, rect.width, rect.height - crop_limit) if rect.height > 200 else rect
        text = page.get_text(sort=True, clip=clip_rect).strip()
        
        char_count = len(text)
        if  (char_count < 50) or (vector_count > 1500 and char_count < 1000):
            print(f"[CHILD PROCESS] Pagina {p_num}: Testo nativo insufficiente o rilevata presenza eccessiva di elementi vettoriali({char_count} chars / {vector_count} vettori). OCR necessario.")
            #log.debug(f"Pagina {p_num}: Rilevato testo nativo non sufficiente ({char_count} chars). Estrazione mediante OCR.")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, clip=clip_rect)
            extracted_pages.append({"type": "image", "content": pix.tobytes("png")})
            print(f"[CHILD PROCESS] Pagina {p_num}: COMPLETATA (rasterizzazione per OCR) in {time.time() - page_t0:.1f}s.")
        else:
            # Estrazione Markdown
            skip_ai_layout = (
                char_count > MAX_CHARS_FOR_AI_LAYOUT
                or (vector_count > VECTOR_COUNT_LAYOUT_SKIP and char_count > VECTOR_COUNT_MIN_CHARS_SKIP)
            )
            if skip_ai_layout:
                # Quantità di testo abnorme e/o presenza massiccia di elementi
                # vettoriali (tipico di planimetrie/disegni tecnici con migliaia
                # di etichette di quote/coordinate, non un documento "normale").
                # Il modello di layout AI (ONNX/GNN) non scala su input di questo
                # tipo e può restare bloccato a lungo (confermato via faulthandler,
                # osservato hang già a ~139k caratteri su una singola pagina).
                #
                # Invece di usare testo grezzo (perdendo ogni struttura), usiamo
                # pymupdf4llm in "legacy mode" (use_layout(False)): produce comunque
                # vero Markdown (titoli via euristica su dimensione font, tabelle via
                # tabulate) ma SENZA invocare onnxruntime/pymupdf_layout, quindi
                # senza rischio di hang.
                print(f"[CHILD PROCESS] Pagina {p_num}: Quantità di testo abnorme e/o troppi vettori ({char_count} chars / {vector_count} vettori) — probabile planimetria/disegno tecnico. Salto il layout AI (legacy mode), uso pymupdf4llm senza ONNX.")
                if layout_ai_active:
                    pymupdf4llm.use_layout(False)
                    layout_ai_active = False
                page_md = pymupdf4llm.to_markdown(doc, pages=[p_num])
                page_md = page_md.replace("<br>", "\n")
                extracted_pages.append({"type": "markdown", "content": page_md})
                print(f"[CHILD PROCESS] Pagina {p_num}: COMPLETATA (legacy mode, no layout AI) in {time.time() - page_t0:.1f}s.")
            else:
                #log.debug(f"Pagina {p_num}: Rilevato testo nativo sufficiente ({char_count} chars). Estrazione Markdown.")
                print(f"[CHILD PROCESS] Pagina {p_num}: Testo sufficiente ({char_count} chars). Estrazione Markdown...")
                if not layout_ai_active:
                    pymupdf4llm.use_layout(True)
                    layout_ai_active = True
                page_md = pymupdf4llm.to_markdown(doc, pages=[p_num])
                page_md = page_md.replace("<br>", "\n")
                extracted_pages.append({"type": "markdown", "content": page_md})
                print(f"[CHILD PROCESS] Pagina {p_num}: COMPLETATA (layout AI) in {time.time() - page_t0:.1f}s.")
            
    doc.close()
    return extracted_pages


def _cpu_heavy_pdf_fallback_extraction(file_path_str: str, file_bytes: bytes):
    """
    Funzione di FALLBACK, usata solo quando _cpu_heavy_pdf_extraction va in
    timeout (chiamata via _run_killable, non più via ProcessPoolExecutor).

    NON usa get_drawings() né pymupdf4llm.to_markdown(): quest'ultima,
    pur sembrando "leggera", internamente richiama comunque get_drawings()
    per il rilevamento di tabelle/layout, quindi non evitava affatto il
    probabile collo di bottiglia. Qui usiamo solo get_text() grezzo, già
    calcolato per decidere se la pagina ha testo sufficiente, così non
    facciamo un secondo giro di analisi sulla pagina.
    """
    import fitz

    print(f"[CHILD PROCESS - FALLBACK] Estrazione per-pagina (solo get_text, no drawings/pymupdf4llm) per {file_path_str}")

    if file_bytes:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    else:
        doc = fitz.open(file_path_str)

    extracted_pages = []
    for p_num in range(len(doc)):
        page_t0 = time.time()
        page = doc[p_num]
        text = page.get_text(sort=True).strip()
        char_count = len(text)

        if char_count < 50:
            print(f"[CHILD PROCESS - FALLBACK] Pagina {p_num}: testo nativo insufficiente ({char_count} chars). OCR necessario.")
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            extracted_pages.append({"type": "image", "content": pix.tobytes("png")})
        else:
            print(f"[CHILD PROCESS - FALLBACK] Pagina {p_num}: testo sufficiente ({char_count} chars). Uso testo grezzo (no pymupdf4llm).")
            extracted_pages.append({"type": "markdown", "content": text})

        print(f"[CHILD PROCESS - FALLBACK] Pagina {p_num}: COMPLETATA in {time.time() - page_t0:.1f}s.")

    doc.close()
    print(f"[CHILD PROCESS - FALLBACK] Completato: {len(extracted_pages)} pagine elaborate per {file_path_str}")
    return extracted_pages


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
    # FIX: Cattura i Rate Limit (429) e gli Errori di Server (500/503) del nuovo SDK
    retry=retry_if_exception_type((genai_errors.APIError, genai_errors.ServerError, genai_errors.ClientError)),
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

async def _ocr_with_final_fallback(image: Image.Image, as_markdown: bool = True) -> str:
    """
    Wrapper di OCR a 2 tentativi (usato come 2°/3° livello della pipeline):
    1) Prova Gemini (async_ocr_generate, con i suoi retry interni e l'eventuale
       fallback automatico su Cloud Vision in caso di blocco policy).
    2) Se anche questo fallisce definitivamente (retry esauriti, errore non
       recuperabile, ecc.), esegue un ultimo tentativo diretto su Cloud Vision.
    Se pure questo fallisce, l'eccezione viene propagata al chiamante.
    """
    try:
        return await async_ocr_generate(image, as_markdown=as_markdown)
    except Exception as e:
        log.warning(f"OCR Gemini fallito definitivamente ({e}). Ultimo tentativo su Cloud Vision...")
        try:
            fallback_text = await asyncio.to_thread(_cloud_vision_fallback, image)
            return fallback_text.strip()
        except Exception as e2:
            log.error(f"Anche il fallback finale su Cloud Vision è fallito: {e2}", exc_info=True)
            raise

async def extract_hybrid_markdown_from_pdf_async(file_path: Path, file_bytes: bytes = None) -> str:
    """Estrae il testo nativo dal PDF o innesca l'OCR (Versione Multi-Processo Anti-Crash)."""
    log.info(f"Avvio estrazione ibrida MULTI-PROCESSO per: {file_path.name}")
    md_pages = []
    
    try:
        # Eseguiamo l'estrazione in un processo DEDICATO e killabile: se va in
        # timeout, il worker viene terminato per davvero (vedi _run_killable),
        # invece di restare appeso a occupare uno slot del PROCESS_POOL condiviso.
        try:
            pages_data = await asyncio.to_thread(
                _run_killable,
                _cpu_heavy_pdf_extraction,
                (str(file_path), file_bytes, CROP_OCR_LIMIT),
                EXTRACTION_TIMEOUT  # <--- TIMEOUT DI SICUREZZA (20 Minuti)
            )
        except TimeoutError:
            log.warning(
                f"TIMEOUT: l'estrazione ibrida di {file_path.name} ha impiegato più di "
                f"{int(EXTRACTION_TIMEOUT // 60)} minuti (probabile PDF vettoriale/CAD pesante). "
                f"Il worker è stato terminato forzatamente. Avvio fallback: estrazione per-pagina + OCR."
            )
            return await _fallback_full_gemini_ocr(file_path, file_bytes)

        # 2. Tornati nell'Event Loop Async, gestiamo le eventuali chiamate API di rete (OCR)
        for page_data in pages_data:
            if page_data["type"] == "image":
                # Ricostruiamo l'oggetto PIL dai bytes generati dal processo figlio
                img = Image.open(io.BytesIO(page_data["content"]))
                ocr_md = await _ocr_with_final_fallback(img, as_markdown=True)
                md_pages.append(ocr_md)
            else:
                md_pages.append(page_data["content"])

        log.info(f"Estrazione ibrida completata per {file_path.name}. Generati {len(md_pages)} blocchi.")
        return "\n\n---\n\n".join(md_pages)
        
    except Exception as e:
        log.error(f"Fallimento durante l'estrazione multi-processo del PDF {file_path.name}", exc_info=True)
        raise


async def _fallback_full_gemini_ocr(file_path: Path, file_bytes: bytes = None) -> str:
    """
    Fallback usato quando l'estrazione ibrida principale va in timeout.
    Rifà l'estrazione pagina per pagina evitando sia get_drawings() che
    pymupdf4llm.to_markdown() (che lo richiama internamente): usa solo
    get_text() grezzo se la pagina ha testo sufficiente, altrimenti la
    rasterizza per l'OCR via Gemini/Cloud Vision (_ocr_with_final_fallback).
    Quindi, ad esempio, la pagina 1 può restare testo nativo grezzo e la
    pagina 2 può finire in OCR, a seconda del contenuto di ciascuna.
    """
    try:
        pages_data = await asyncio.to_thread(
            _run_killable,
            _cpu_heavy_pdf_fallback_extraction,
            (str(file_path), file_bytes),
            FALLBACK_RENDER_TIMEOUT
        )
    except TimeoutError:
        log.error(
            f"TIMEOUT CRITICO: anche l'estrazione di fallback per {file_path.name} "
            f"ha superato {int(FALLBACK_RENDER_TIMEOUT)}s. Il worker è stato terminato forzatamente. File saltato."
        )
        raise ValueError(
            "PDF extraction timed out and fallback extraction also timed out "
            "(possible corrupted or extremely heavy file)"
        )

    md_pages = []
    for page_data in pages_data:
        if page_data["type"] == "image":
            img = Image.open(io.BytesIO(page_data["content"]))
            ocr_md = await _ocr_with_final_fallback(img, as_markdown=True)
            md_pages.append(ocr_md)
        else:
            md_pages.append(page_data["content"])

    log.info(f"Fallback completato per {file_path.name}. Generati {len(md_pages)} blocchi (misto testo nativo/OCR).")
    return "\n\n---\n\n".join(md_pages)

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

                with fitz.open(attached_file_path) as doc:
                    page_count = len(doc)
                
                if page_count > settings.max_allowed_pages:
                    # Lanciamo un errore specifico che interrompe il job ed evita sprechi di CPU
                    raise ValueError(
                        f"Rifiutato: Il file '{filename}' ha {page_count} pagine, "
                        f"superando il limite massimo di {settings.max_allowed_pages}."
                    )
                
                md_content = await extract_hybrid_markdown_from_pdf_async(attached_file_path)
            elif attached_ext in [".txt", ".md"]:
                md_content = attached_file_path.read_text(encoding="utf-8", errors="ignore")
            elif attached_ext in IMG_EXTENSIONS:
                log.info(f"Rilevata immagine {filename}. Avvio OCR nativo...")
                with Image.open(attached_file_path) as img:
                    md_content = await _ocr_with_final_fallback(img, as_markdown=True)
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

        raise e

# ==============================================================================
# 4. ORCHESTRAZIONE MESSAGE BROKER E LOOP PRINCIPALE
# ==============================================================================

async def on_message_received(message: aio_pika.IncomingMessage, channel: aio_pika.Channel):
    """Callback triggered all'arrivo di ogni messaggio da RabbitMQ."""
    try:
        payload = json.loads(message.body.decode())
        log.debug(f"Payload RabbitMQ Ricevuto: {payload}")
        await process_single_job(payload, channel)
        if not message.channel.is_closed:
            await message.ack()
        else:
            log.warning("Channel chiuso prima dell'ack — messaggio già processato con successo, nessun ack inviato.")
    except json.JSONDecodeError as e:
        log.error(f"Impossibile decodificare il messaggio: {e}. Scartato in DLQ.")
        if not message.channel.is_closed:
            await message.reject(requeue=False)
    except Exception as e:
        log.error(f"Errore durante l'elaborazione: {e}", exc_info=True)
        if not message.channel.is_closed:
            await message.reject(requeue=False)
        else:
            log.warning("Channel chiuso — impossibile fare reject. RabbitMQ re-accoda automaticamente al reconnect.")

async def main_worker():
    
    
    log.info(f"Avvio Worker. Tentativo di connessione a RabbitMQ su {settings.broker_host}...")
    
    try:
        connection = await aio_pika.connect_robust(
            f"amqp://{settings.broker_username}:{settings.broker_password}@{settings.broker_host}:{settings.broker_port}/",
            heartbeat=120,
        )
        
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