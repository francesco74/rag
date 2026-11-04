import os
import json
import shutil
import logging
import asyncio
import contextvars
import io
import time
from pathlib import Path
from PIL import Image
from common.db_logger import MySQLLogHandler, init_db_pool

import pymupdf          # PyMuPDF
import pymupdf4llm   # Native PDF to Markdown
import aio_pika      # Asynchronous RabbitMQ client

# --- Dipendenze AI importate ---

from google.cloud import vision
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import multiprocessing as mp
import queue

from dotenv import load_dotenv
from common.config import settings
from common.upload_status import FAILED, PROCESSING, mark_upload, record_ocr
from common.utility import CONVERSION_QUEUE, CONVERSION_QUEUE_ARGS, EXTERNAL_OCR_MARKER
from ocr_provider import OcrBlockedError, OcrTransientError, clean_transcription, get_external_fallback, get_ocr_provider
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
ARCHIVE_DIR = DATA_FOLDER / "processed"

CONCURRENCY_LIMIT = 2  # numero massimo di file elaborati in parallelo da QUESTA istanza.
                        # Ogni file "pesante" genera comunque un sottoprocesso dedicato (via _run_killable)
                        # con il proprio limite di thread ONNX (vedi _detect_cgroup_cpu_quota) — processarne
                        # più di 2-3 in parallelo nello stesso container rischia di saturare CPU/RAM.
                        # Per aumentare il throughput oltre questo, preferire più REPLICHE del worker
                        # (RabbitMQ distribuisce automaticamente i messaggi tra consumer concorrenti sulla
                        # stessa coda) piuttosto che alzare ulteriormente questo valore.

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
    Esegue target_func(*args) in un PROCESSO DEDICATO (non in un pool condiviso
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

EXTRACTION_TIMEOUT = 600.0         # 10 minuti - TENTATIVO 1: libreria CON layout AI (ONNX/GNN), intero documento
LEGACY_EXTRACTION_TIMEOUT = 300.0  # 5 minuti - TENTATIVO 2: libreria SENZA layout AI (legacy mode), intero documento.
                                    # NB: abbiamo verificato che la legacy mode NON è garantita più veloce
                                    # dell'AI layout su pagine molto dense di tracciati vettoriali (CAD) — può
                                    # anzi essere più lenta. Il vero paracadute è il TENTATIVO 3 (OCR), quindi
                                    # non ha senso tenere timeout alti sui primi due livelli: tanto vale arrivare
                                    # prima all'OCR, che funziona indipendentemente da quanto è "brutta" la pagina.
FALLBACK_OCR_TIMEOUT = 300.0       # 5 minuti - rasterizzazione di TUTTE le pagine per il fallback OCR (TENTATIVO 3)

# Limiter asincrono: gestisce internamente la concorrenza garantendo max OCR_RATE_PER_MINUTE chiamate/minuto
OCR_LIMITER = AsyncLimiter(max_rate=settings.ocr_rate_per_minute, time_period=60)

# Creato al primo uso: senza credenziali Google il converter parte comunque.
_vision_client = None


# Pagine del file corrente {"ocr", "total"}, da registrare a fine conversione.
PAGE_STATS = contextvars.ContextVar("page_stats", default=None)


def _count_pages(ocr=0, total=0):
    stats = PAGE_STATS.get()
    if stats is not None:
        stats["ocr"] += ocr
        stats["total"] += total


def inspect_page(page, crop_limit: int = CROP_OCR_LIMIT):
    """La regola che manda una pagina all'OCR: la usa anche rebuild_document_ocr.py."""
    try:
        # get_drawings() estrae i tracciati vettoriali (linee, poligoni, curve CAD)
        vector_count = len(page.get_drawings())
    except Exception:
        vector_count = 0

    rect = page.rect
    clip_rect = pymupdf.Rect(0, crop_limit, rect.width, rect.height - crop_limit) if rect.height > 200 else rect
    char_count = len(page.get_text(sort=True, clip=clip_rect).strip())
    return clip_rect, char_count, vector_count, (char_count < 50) or (vector_count > 1500 and char_count < 1000)


def _cpu_heavy_pdf_extraction(file_path_str: str, file_bytes: bytes, crop_limit: int, use_ai_layout: bool = True):
    """
    Funzione isolata per esecuzione in processo dedicato (via _run_killable).
    Gira in un processo OS separato, bypassando totalmente il GIL.

    use_ai_layout=True  -> TENTATIVO 1 della catena: pymupdf4llm con layout AI
                           (ONNX/GNN) attivo, qualità migliore su tabelle/multi-colonna.
    use_ai_layout=False -> TENTATIVO 2 della catena: pymupdf4llm in "legacy mode"
                           (pymupdf4llm.use_layout(False)): titoli via euristica
                           sulla dimensione del font, tabelle via tabulate, ma
                           SENZA invocare onnxruntime — quindi senza rischio di
                           hang, a scapito di qualità leggermente inferiore su
                           layout complessi (multi-colonna, tabelle articolate).
    """
    import pymupdf
    import pymupdf4llm

    # Limitiamo comunque i thread ONNX Runtime al numero di CPU realmente
    # assegnate dal cgroup (vedi _detect_cgroup_cpu_quota): utile quando
    # use_ai_layout=True, innocuo (non viene mai creata una sessione ONNX)
    # quando use_ai_layout=False.
    _limit_onnx_threads()
    pymupdf4llm.use_layout(use_ai_layout)

    mode_label = "layout AI (ONNX)" if use_ai_layout else "legacy mode (no ONNX)"
    print(f"[CHILD PROCESS] Avvio estrazione per {file_path_str} in modalità: {mode_label} (ONNX thread limit: {ONNX_THREAD_LIMIT})")

    if file_bytes:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    else:
        doc = pymupdf.open(file_path_str)

    extracted_pages = []
    print(f"[CHILD PROCESS] Estrazione PDF iniziata per {file_path_str} ({len(doc)} pagine).")
    for p_num in range(len(doc)):
        page_t0 = time.time()
        page = doc[p_num]
        clip_rect, char_count, vector_count, ocr_needed = inspect_page(page, crop_limit)
        if ocr_needed:
            print(f"[CHILD PROCESS] Pagina {p_num}: Testo nativo insufficiente o rilevata presenza eccessiva di elementi vettoriali({char_count} chars / {vector_count} vettori). OCR necessario.")
            #log.debug(f"Pagina {p_num}: Rilevato testo nativo non sufficiente ({char_count} chars). Estrazione mediante OCR.")
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False, clip=clip_rect)
            extracted_pages.append({"type": "image", "content": pix.tobytes("png")})
            print(f"[CHILD PROCESS] Pagina {p_num}: COMPLETATA (rasterizzazione per OCR) in {time.time() - page_t0:.1f}s.")
        else:
            print(f"[CHILD PROCESS] Pagina {p_num}: Testo sufficiente ({char_count} chars). Estrazione Markdown ({mode_label})...")
            page_md = pymupdf4llm.to_markdown(doc, pages=[p_num])
            page_md = page_md.replace("<br>", "\n")
            extracted_pages.append({"type": "markdown", "content": page_md})
            print(f"[CHILD PROCESS] Pagina {p_num}: COMPLETATA ({mode_label}) in {time.time() - page_t0:.1f}s.")
            
    doc.close()
    return extracted_pages


def _rasterize_all_pages(file_path_str: str, file_bytes: bytes):
    """
    TENTATIVO 3 della catena: usata solo quando SIA il tentativo con layout AI
    SIA quello in legacy mode sono andati in timeout (chiamata via _run_killable).

    Non usa né get_drawings() né pymupdf4llm.to_markdown(): a questo punto non ci
    fidiamo più di alcuna estrazione testuale nativa per l'intero documento, quindi
    rasterizziamo TUTTE le pagine (anche quelle con testo nativo abbondante) e le
    passiamo tutte all'OCR (OCR_PROVIDER).
    """
    import pymupdf

    print(f"[CHILD PROCESS - FALLBACK OCR] Rasterizzazione completa di tutte le pagine per {file_path_str}")

    if file_bytes:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    else:
        doc = pymupdf.open(file_path_str)

    rendered_pages = []
    for p_num in range(len(doc)):
        page_t0 = time.time()
        page = doc[p_num]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        rendered_pages.append(pix.tobytes("png"))
        print(f"[CHILD PROCESS - FALLBACK OCR] Pagina {p_num}: rasterizzata in {time.time() - page_t0:.1f}s.")

    doc.close()
    print(f"[CHILD PROCESS - FALLBACK OCR] Completato: {len(rendered_pages)} pagine rasterizzate per {file_path_str}")
    return rendered_pages


def safe_move(src: Path, dest: Path):
    """Sposta i file in modo sicuro con sovrascrittura se necessario."""
    if dest.exists(): 
        log.debug(f"Sovrascrittura file esistente in destinazione: {dest.name}")
        dest.unlink()
    shutil.move(str(src), str(dest))
    log.debug(f"File spostato: {src.name} -> {dest.parent.name}/")


# NOTA: la stessa identica esigenza si presenta in estrattore.py (che sbusta i
# file .p7m PRIMA che arrivino qui) — questa è una rete di sicurezza per gli
# eventuali allegati che, nonostante tutto, arrivano ancora incapsulati
# (mislabeling a monte da Sicr@Web: un allegato dichiarato ".pdf" il cui
# contenuto è in realtà ancora una busta CAdES/PKCS7 non sbustata, si veda la
# stessa funzione in estrattore.py per i dettagli). Duplicata qui invece di
# importata da un modulo condiviso solo perché converter.py ed estrattore.py
# sono attualmente due deployable indipendenti senza un pacchetto comune
# importabile da entrambi — da valutare un refactor futuro in common/ se la
# duplicazione diventa scomoda da mantenere.
def try_extract_pdf_from_pkcs7(file_bytes: bytes, filename: str = "") -> bytes | None:
    """
    Tenta di interpretare file_bytes come busta di firma PKCS7/CMS (CAdES) e,
    se lo è, ne estrae il contenuto incapsulato (il PDF originale).

    Usa asn1crypto per un parsing ASN.1 vero (controllo semantico del campo
    content_type)
    Restituisce:
    - None se file_bytes NON è una struttura PKCS7 valida (caso normale: PDF
      nativo già in chiaro) — il chiamante deve procedere con file_bytes
      originali, nessun errore da segnalare.
    - i bytes del PDF estratto, se file_bytes è una busta PKCS7 di tipo
      signed_data e l'estrazione riesce.

    Solleva ValueError SOLO se la struttura è riconosciuta come PKCS7
    signed_data ma l'estrazione del contenuto incapsulato fallisce
    (caso anomalo/da indagare, distinto da "non era affatto una busta").
    """
    from asn1crypto.cms import ContentInfo

    try:
        content_info = ContentInfo.load(file_bytes)
    except Exception:
        # Non è (o non è un valido) ASN.1: quasi certamente un PDF nativo normale.
        return None

    if content_info['content_type'].native != 'signed_data':
        # E' una struttura ASN.1 valida ma non una busta di firma: non il nostro caso.
        return None

    try:
        signed_data = content_info['content']
        encap_content_info = signed_data['encap_content_info']
        raw_content = encap_content_info['content'].native
        extracted_bytes = raw_content if isinstance(raw_content, bytes) else encap_content_info['content'].chosen.contents
        log.debug("Sbustamento completato per %s", filename)
        return extracted_bytes
    except Exception as e:
        log.error("Fallimento sbustamento P7M per %s: %s", filename, str(e))
        raise ValueError(f"Decodifica P7M fallita: {str(e)}")


# ==============================================================================
# 2. MOTORE IBRIDO DI ESTRAZIONE E OCR
# ==============================================================================

def vision_enabled() -> bool:
    """Vision si usa solo se OCR_VISION_FALLBACK lo ammette e le credenziali ci sono."""
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    return settings.ocr_vision_fallback and bool(credentials) and Path(credentials).is_file() and Path(credentials).stat().st_size > 0


def _cloud_vision_transcribe(image: Image.Image) -> str:
    """Trascrive la pagina con Google Cloud Vision: solo dal comando del revisore."""
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    log.info("Invio della pagina a Google Cloud Vision...")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    vision_image = vision.Image(content=img_byte_arr.getvalue())
    response = _vision_client.document_text_detection(image=vision_image)
    if response.error.message:
        log.error(f"Errore Cloud Vision API: {response.error.message}")
        raise Exception(f"Cloud Vision API Error: {response.error.message}")
    log.debug("Trascrizione con Cloud Vision completata.")
    return response.full_text_annotation.text

OCR_RETRY = retry(
    # Solo rate limit, sovraccarico e rete: gli altri errori non cambiano ritentando.
    retry=retry_if_exception_type(OcrTransientError),
    wait=wait_random_exponential(multiplier=2, min=10, max=80),
    stop=stop_after_attempt(settings.ocr_max_attempts),
    reraise=True,
)


async def _transcribe_page(provider, image_input, as_markdown) -> str:
    async with OCR_LIMITER:
        prompt = "Transcribe the text in this image precisely. Format the output strictly as Markdown." if as_markdown else "Transcribe the text in this image precisely as raw text."

        log.debug("Chiamata OCR inviata.")
        text = clean_transcription(await provider.transcribe(image_input, prompt))
        if not text:
            raise ValueError("OCR: risposta vuota dal modello")
        log.debug("Risposta OCR ricevuta con successo.")
        return text


@OCR_RETRY
async def async_ocr_generate(image_input, as_markdown=False) -> str:
    """Interroga il modello OCR principale rispettando i limiti di rate."""
    return await _transcribe_page(get_ocr_provider(), image_input, as_markdown)


@OCR_RETRY
async def external_ocr_generate(provider, image_input, as_markdown=False) -> str:
    """Interroga il modello esterno come ripiego di quello principale."""
    return await _transcribe_page(provider, image_input, as_markdown)


def publish_external_ocr_availability():
    """Solo il converter conosce chiavi e credenziali: segnala all'API se il comando del revisore è utilizzabile."""
    try:
        external = get_external_fallback() is not None
    except Exception as e:
        # Il pulsante resta visibile: il revisore vedrà l'errore invece di un comando sparito.
        log.error(f"Ripiego OCR esterno non utilizzabile: {e}")
        external = True
    marker = DATA_FOLDER / EXTERNAL_OCR_MARKER
    if external or vision_enabled():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    else:
        marker.unlink(missing_ok=True)
    log.info(f"OCR esterno su richiesta: modello esterno {'attivo' if external else 'spento'}, "
             f"Vision {'attivo' if vision_enabled() else 'spento'}.")


async def _external_transcribe_page(image: Image.Image) -> str:
    """Una pagina per il comando del revisore: il modello esterno, poi Vision se abilitato."""
    # Fuori dal try: una configurazione sbagliata ferma il lavoro, non lo manda a Vision.
    external = get_external_fallback()
    failure = "modello esterno non attivo"
    if external:
        try:
            return await external_ocr_generate(external, image, as_markdown=True)
        except Exception as e:
            failure = e
            log.warning(f"OCR esterno non riuscito ({e}).")
    if not vision_enabled():
        raise ValueError(f"{failure}; Google Cloud Vision non è abilitato.")
    return (await asyncio.to_thread(_cloud_vision_transcribe, image)).strip()


async def transcribe_externally(file_path: Path, filename: str) -> str:
    """Tutte le pagine all'OCR esterno, per scelta del revisore: non passa dal modello di OCR_PROVIDER."""
    if file_path.suffix.lower() in IMG_EXTENSIONS:
        with Image.open(file_path) as img:
            return await _external_transcribe_page(img)
    pdf_bytes = file_path.read_bytes()
    extracted = try_extract_pdf_from_pkcs7(pdf_bytes, filename)
    if extracted is not None:
        pdf_bytes = extracted
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if len(doc) > settings.max_allowed_pages:
            raise ValueError(f"Il file ha {len(doc)} pagine, oltre il limite di {settings.max_allowed_pages}.")
    pages = await asyncio.to_thread(_run_killable, _rasterize_all_pages, (str(file_path), pdf_bytes), FALLBACK_OCR_TIMEOUT)
    texts = []
    for png_bytes in pages:
        texts.append(await _external_transcribe_page(Image.open(io.BytesIO(png_bytes))))
    return "\n\n---\n\n".join(texts)


async def _ocr_page(image: Image.Image, as_markdown: bool = True) -> str:
    """Una pagina con il modello OCR configurato. Se fallisce, il documento va in errore: nessun ripiego automatico."""
    _count_pages(ocr=1)
    return await async_ocr_generate(image, as_markdown=as_markdown)


async def extract_hybrid_markdown_from_pdf_async(file_path: Path, file_bytes: bytes = None) -> str:
    """
    Estrae il testo nativo dal PDF o innesca l'OCR, seguendo questa catena di
    fallback (ogni livello scatta solo se il precedente va in TIMEOUT):

      1) pymupdf4llm CON layout AI (ONNX/GNN)      -> EXTRACTION_TIMEOUT (20 min)
      2) pymupdf4llm SENZA layout AI (legacy mode) -> LEGACY_EXTRACTION_TIMEOUT (10 min)
      3) OCR completo via OCR_PROVIDER (rasterizzazione di tutte le pagine)
      Se l'OCR di una pagina fallisce (errore o rifiuto), il documento va in errore.
    """
    log.info(f"Avvio estrazione ibrida MULTI-PROCESSO per: {file_path.name}")
    md_pages = []

    try:
        # --- TENTATIVO 1: layout AI (ONNX) ---
        try:
            pages_data = await asyncio.to_thread(
                _run_killable,
                _cpu_heavy_pdf_extraction,
                (str(file_path), file_bytes, CROP_OCR_LIMIT, True),
                EXTRACTION_TIMEOUT
            )
        except TimeoutError:
            log.warning(
                f"TIMEOUT: l'estrazione con layout AI di {file_path.name} ha impiegato più di "
                f"{int(EXTRACTION_TIMEOUT // 60)} minuti (probabile PDF vettoriale/CAD pesante). "
                f"Il worker è stato terminato forzatamente. Provo TENTATIVO 2: legacy mode (no ONNX)."
            )
            # --- TENTATIVO 2: legacy mode (senza ONNX) ---
            try:
                pages_data = await asyncio.to_thread(
                    _run_killable,
                    _cpu_heavy_pdf_extraction,
                    (str(file_path), file_bytes, CROP_OCR_LIMIT, False),
                    LEGACY_EXTRACTION_TIMEOUT
                )
            except TimeoutError:
                log.warning(
                    f"TIMEOUT: anche l'estrazione in legacy mode (no ONNX) di {file_path.name} ha impiegato più di "
                    f"{int(LEGACY_EXTRACTION_TIMEOUT // 60)} minuti. Il worker è stato terminato forzatamente. "
                    f"Provo TENTATIVO 3: OCR completo."
                )
                # --- TENTATIVO 3: OCR completo (OCR_PROVIDER) ---
                return await _fallback_full_ocr(file_path, file_bytes)

        # Tornati nell'Event Loop Async, gestiamo le eventuali chiamate API di rete (OCR
        # per le pagine senza testo nativo sufficiente, sia nel Tentativo 1 che nel 2)
        for page_data in pages_data:
            if page_data["type"] == "image":
                # Ricostruiamo l'oggetto PIL dai bytes generati dal processo figlio
                img = Image.open(io.BytesIO(page_data["content"]))
                ocr_md = await _ocr_page(img, as_markdown=True)
                md_pages.append(ocr_md)
            else:
                md_pages.append(page_data["content"])

        log.info(f"Estrazione ibrida completata per {file_path.name}. Generati {len(md_pages)} blocchi.")
        return "\n\n---\n\n".join(md_pages)
        
    except Exception as e:
        log.error(f"Fallimento durante l'estrazione multi-processo del PDF {file_path.name}", exc_info=True)
        raise


async def _fallback_full_ocr(file_path: Path, file_bytes: bytes = None) -> str:
    """
    TENTATIVO 3 della catena: usato quando SIA il tentativo con layout AI (ONNX)
    SIA quello in legacy mode (senza ONNX) sono andati in timeout.

    Rasterizza tutte le pagine del documento e le passa al modello OCR.
    """
    try:
        rendered_pages = await asyncio.to_thread(
            _run_killable,
            _rasterize_all_pages,
            (str(file_path), file_bytes),
            FALLBACK_OCR_TIMEOUT
        )
    except TimeoutError:
        log.error(
            f"TIMEOUT CRITICO: anche la rasterizzazione di fallback per {file_path.name} "
            f"ha superato {int(FALLBACK_OCR_TIMEOUT)}s. Il worker è stato terminato forzatamente. File saltato."
        )
        raise ValueError(
            "PDF extraction timed out at every fallback level (AI layout, legacy mode, "
            "and raw page rasterization) — possible corrupted or extremely heavy file"
        )

    md_pages = []
    for png_bytes in rendered_pages:
        img = Image.open(io.BytesIO(png_bytes))
        ocr_md = await _ocr_page(img, as_markdown=True)
        md_pages.append(ocr_md)

    log.info(f"Fallback OCR completo completato per {file_path.name}. Generati {len(md_pages)} blocchi.")
    return "\n\n---\n\n".join(md_pages)

# ==============================================================================
# 3. CORE LOGIC - GESTIONE MESSAGGI (EVENT-DRIVEN)
# ==============================================================================

async def convert_file_to_markdown(attached_file_path: Path, filename: str) -> str | None:
    """Testo Markdown di un file; None se il formato non è gestito."""
    attached_ext = attached_file_path.suffix.lower()
    if attached_ext == ".pdf":

        pdf_bytes = attached_file_path.read_bytes()

        try:
            extracted = try_extract_pdf_from_pkcs7(pdf_bytes, filename)
        except ValueError as e:
            # Riconosciuto come busta PKCS7 ma estrazione fallita: caso
            # anomalo, non ha senso procedere con i bytes ancora
            # incapsulati (PyMuPDF fallirebbe comunque poco dopo con un
            # errore meno chiaro). Meglio un fallimento esplicito qui.
            raise ValueError(f"File '{filename}': {e}")

        if extracted is not None:
            log.warning(
                f"Il file '{filename}' ha estensione .pdf ma il contenuto è in realtà "
                f"una busta di firma digitale PKCS7/CMS (.p7m) non sbustata a monte. "
                f"PDF incapsulato estratto con successo ({len(extracted)} bytes)."
            )
            pdf_bytes = extracted

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = len(doc)

        if page_count > settings.max_allowed_pages:
            # Lanciamo un errore specifico che interrompe il job ed evita sprechi di CPU
            raise ValueError(
                f"Rifiutato: Il file '{filename}' ha {page_count} pagine, "
                f"superando il limite massimo di {settings.max_allowed_pages}."
            )

        _count_pages(total=page_count)
        return await extract_hybrid_markdown_from_pdf_async(attached_file_path, file_bytes=pdf_bytes)
    elif attached_ext in [".txt", ".md"]:
        return attached_file_path.read_text(encoding="utf-8", errors="ignore")
    elif attached_ext in IMG_EXTENSIONS:
        log.info(f"Rilevata immagine {filename}. Avvio OCR nativo...")
        _count_pages(total=1)
        with Image.open(attached_file_path) as img:
            return await _ocr_page(img, as_markdown=True)
    log.warning(f"Estensione {attached_ext} ignorata per il file {filename}.")
    return None


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
        await asyncio.to_thread(mark_upload, original_source, PROCESSING)
        
        for filename in attached_files:
            # Risoluzione sul filesystem: usiamo solo il nome base di
            # 'filename', perché il file è co-locato nella stessa cartella
            # del JSON (anche quando 'filename' nel manifest è un percorso
            # relativo più lungo, es. "preliminari/gennaio/test.pdf" per i
            # manifest generati da direct.py — lì serve per file_name in DB,
            # non per la risoluzione fisica). Per i manifest Sicr@Web,
            # 'filename' è già un nome semplice: Path(filename).name
            # restituisce lo stesso valore, nessun cambio di comportamento.
            attached_file_path = current_json_dir / Path(filename).name
            
            if not attached_file_path.exists():
                log.warning(f"Allegato mancante {filename} nella cartella {current_json_dir.name}.")
                await asyncio.to_thread(mark_upload, original_source, FAILED, f"File non trovato: {filename}")
                continue

            log.info(f"Conversione allegato: {filename}")
            stats = {"ocr": 0, "total": 0}
            PAGE_STATS.set(stats)
            md_content = await convert_file_to_markdown(attached_file_path, filename)

            if md_content:
                safe_stem = Path(filename).stem
                # Se lo stem del JSON coincide già con quello dell'allegato
                # (sempre vero per i manifest a singolo file di direct.py,
                # dato che il JSON è nominato come f"{stem_del_file}.json"),
                # non concateniamo: eviterebbe solo una ripetizione inutile
                # (es. "test_test"). Nella pipeline Sicr@Web, dove un JSON
                # può referenziare più allegati diversi, i due stem sono
                # tipicamente diversi e la concatenazione resta necessaria
                # per disambiguare l'output dei vari allegati.
                unique_base_name = safe_stem if safe_stem == base_name else f"{base_name}_{safe_stem}"
                
                final_md_path = dest_dir / f"{unique_base_name}.md"
                final_md_path.write_text(md_content, encoding="utf-8")
                
                child_manifest = manifest.copy()
                # Se 'source' include già il nome/percorso dell'allegato per
                # intero (es. i manifest di direct.py, dove source =
                # "direct://topic/sub/preliminari/gennaio/test.pdf" e filename
                # = "preliminari/gennaio/test.pdf"), non appendiamo di nuovo
                # "::filename" — finirebbe duplicato (".../test.pdf::preliminari/gennaio/test.pdf").
                # Nella pipeline Sicr@Web, dove 'source' è solo l'identificativo
                # dell'atto (es. "sicraweb://{uid}", senza alcun nome file),
                # questa condizione è sempre falsa e il comportamento resta
                # quello originale.
                if original_source.endswith(filename):
                    child_manifest["source"] = original_source
                else:
                    child_manifest["source"] = f"{original_source}::{filename}"
                child_manifest["files"] = [filename]
                
                await asyncio.to_thread(record_ocr, child_manifest["source"], rel_path.parts[0], rel_path.parts[1],
                                        stats["ocr"], stats["total"])

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
            else:
                await asyncio.to_thread(mark_upload, original_source, FAILED, "Nessun testo ricavato dal file")

            # Archiviazione allegato processato. Usiamo Path(filename).name
            # (solo il nome base) perché archive_target_dir mirrora già
            # l'intera alberatura originale (deriva da rel_path.parent, che
            # con file e JSON co-locati è già profondo/univoco di suo).
            # Usare il 'filename' completo qui raddoppierebbe il sottopercorso
            # (es. .../preliminari/gennaio/preliminari/gennaio/test.pdf).
            archived_dest = archive_target_dir / Path(filename).name
            archived_dest.parent.mkdir(parents=True, exist_ok=True)
            safe_move(attached_file_path, archived_dest)

        # Archiviazione manifesto radice
        safe_move(file_path, archive_target_dir / file_path.name)
        log.info(f"✓ Manifesto JSON '{file_path.name}' completato e archiviato con successo.")

    except Exception as e:
        log.error(f"Fallimento durante l'esplosione del manifest {rel_path}: {e}", exc_info=True)
        if "original_source" in locals():
            await asyncio.to_thread(mark_upload, original_source, FAILED, e)
        err_dir = ERROR_DIR / rel_path.parent
        err_dir.mkdir(parents=True, exist_ok=True)
        safe_move(file_path, err_dir / file_path.name)

        raise e

# ==============================================================================
# 4. ORCHESTRAZIONE MESSAGE BROKER E LOOP PRINCIPALE
# ==============================================================================

async def process_conversion_job(job_id: str):
    """
    Lavoro delle API esterne (external_routes.py). Nella cartella del lavoro:
    request.json e il file li scrive l'API; qui si aggiornano status.json e
    result.md. Il file ricevuto si cancella a fine lavoro, comunque vada.
    """
    if not re.fullmatch(r"[0-9a-f]{32}", str(job_id)):
        log.error(f"Lavoro di conversione con id non valido: {job_id!r}")
        return
    job_dir = settings.data_folder / "conversions" / job_id
    if not (job_dir / "request.json").exists():
        log.warning(f"Lavoro di conversione {job_id} non trovato (già scaduto?). Ignorato.")
        return

    def write_status(**status):
        tmp = job_dir / "status.json.tmp"
        tmp.write_text(json.dumps(status), encoding="utf-8")
        tmp.replace(job_dir / "status.json")

    request_data = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    input_path = job_dir / request_data["input_file"]
    external_ocr = bool(request_data.get("external_ocr"))
    write_status(status="running")
    try:
        if external_ocr:
            md_content = await transcribe_externally(input_path, request_data["file_name"])
        else:
            md_content = await convert_file_to_markdown(input_path, request_data["file_name"])
        if not md_content or not md_content.strip():
            raise ValueError("Nessun testo estratto dal documento.")
        tmp = job_dir / "result.md.tmp"
        tmp.write_text(md_content, encoding="utf-8")
        tmp.replace(job_dir / "result.md")
        write_status(status="completed")
        log.info(f"Conversione {job_id} completata: {len(md_content)} caratteri.")
    except Exception as e:
        log.error(f"Conversione {job_id} fallita: {e}", exc_info=True)
        write_status(status="failed", code="conversion_failed",
                     error=f"Trascrizione non riuscita: {e}" if external_ocr else "Conversione del documento non riuscita.")
    finally:
        input_path.unlink(missing_ok=True)


async def on_conversion_message(message: aio_pika.IncomingMessage):
    """Gli errori del lavoro finiscono in status.json: il messaggio si conferma comunque."""
    async with message.process(requeue=False):
        await process_conversion_job(json.loads(message.body.decode()).get("job_id"))


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
            # prefetch_count = CONCURRENCY_LIMIT: RabbitMQ non consegna più di CONCURRENCY_LIMIT
            # messaggi non ancora "ack-ati" contemporaneamente, il che throttla naturalmente
            # l'iteratore qui sotto senza bisogno di ulteriore sincronizzazione esplicita.
            await channel.set_qos(prefetch_count=CONCURRENCY_LIMIT)

            queue_in = await channel.get_queue("da-convertire")  # ← assume che esista già
            # Conversioni delle API esterne: stesso limite di concorrenza dei documenti.
            queue_conversions = await channel.declare_queue(
                CONVERSION_QUEUE, durable=True, arguments=CONVERSION_QUEUE_ARGS)
            await queue_conversions.consume(on_conversion_message)

            log.info(f"✓ Connessione stabilita. In ascolto sulla coda 'da-convertire' (concorrenza: {CONCURRENCY_LIMIT} file in parallelo).")

            # Teniamo un riferimento ai task in corso per evitare che vengano
            # garbage-collected prematuramente (rischio noto di asyncio.create_task
            # senza riferimenti forti) e per loggare eventuali eccezioni non gestite.
            in_flight_tasks = set()

            def _on_task_done(task: asyncio.Task):
                in_flight_tasks.discard(task)
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    # Non dovrebbe succedere: on_message_received gestisce già le sue
                    # eccezioni internamente (ack/reject). Se arriva qui è un bug da correggere.
                    log.error(f"Task di elaborazione messaggio terminato con eccezione non gestita: {exc}", exc_info=exc)

            async with queue_in.iterator() as queue_iter:
                async for message in queue_iter:
                    task = asyncio.create_task(on_message_received(message, channel))
                    in_flight_tasks.add(task)
                    task.add_done_callback(_on_task_done)
                    
    except Exception as e:
        log.error(f"Errore critico di connessione a RabbitMQ: {e}")
        raise

if __name__ == "__main__":
    log.info("=== START: Servizio Normalizzazione Documentale (Event-Driven) ===")
    publish_external_ocr_availability()
    try:
        asyncio.run(main_worker())
    except KeyboardInterrupt:
        log.info("Interruzione manuale ricevuta. Spegnimento gracefull in corso...")
    finally:
        log.info("=== STOP: Worker terminato ===")