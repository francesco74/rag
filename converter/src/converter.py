import os
import json
import shutil
import logging
import asyncio
import re
from pathlib import Path
from common.db_logger import MySQLLogHandler, init_db_pool

import aio_pika      # Asynchronous RabbitMQ client
from google.cloud import vision

from dotenv import load_dotenv
from common.config import settings
from common.upload_status import FAILED, PROCESSING, mark_upload, record_ocr
from common.utility import CONVERSION_QUEUE, CONVERSION_QUEUE_ARGS, EXTERNAL_OCR_MARKER

# Conversione, OCR e controlli sui file: libreria condivisa con atti-pa
# (https://github.com/francesco74/pa-common)
from pa_common import ingresso, processo
from pa_common.conversione import FORMATI, Statistiche, Tempi, in_markdown
from pa_common.conversione.pdf import rasterizza
from pa_common.errori import ErroreOcr
from pa_common.ocr import Ocr, crea_ocr

load_dotenv()

# --- DIRECTORY SYSTEM ---
DATA_FOLDER = settings.data_folder
STAGING_DIR = DATA_FOLDER / "staging"
INGESTION_WATCH_DIR = DATA_FOLDER / "watch"
ERROR_DIR = DATA_FOLDER / "converter" / "error"
ARCHIVE_DIR = DATA_FOLDER / "processed"

CONCURRENCY_LIMIT = 2  # numero massimo di file elaborati in parallelo da QUESTA istanza.
                        # Ogni PDF e ogni foglio di calcolo genera comunque un sottoprocesso dedicato
                        # (pa_common.processo) con il proprio limite di thread ONNX — processarne
                        # più di 2-3 in parallelo nello stesso container rischia di saturare CPU/RAM.
                        # Per aumentare il throughput oltre questo, preferire più REPLICHE del worker
                        # (RabbitMQ distribuisce automaticamente i messaggi tra consumer concorrenti sulla
                        # stessa coda) piuttosto che alzare ulteriormente questo valore.


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
# i messaggi di pa-common (conversione, OCR) nello stesso registro
for _nome in ("pa_common",):
    logging.getLogger(_nome).addHandler(db_handler)

# Assicuriamo che le directory esistano allo startup
for d in [STAGING_DIR, INGESTION_WATCH_DIR, ERROR_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# Catena di estrazione dei PDF (pa_common.conversione.pdf): ogni livello scatta solo se il
# precedente supera il suo tempo massimo.
#   1) layout AI (ONNX), 10 minuti; 2) modalità semplice (senza ONNX), 5 minuti: non è
#   garantita più veloce sulle pagine piene di tracciati vettoriali (CAD), il vero paracadute
#   è il livello 3; 3) OCR di tutte le pagine, rasterizzate in al massimo 5 minuti.
# Lo stesso tempo della modalità semplice vale per un documento d'ufficio o un foglio di calcolo.
TEMPI_PDF = Tempi(layout_ai=600.0, semplice=300.0, rasterizza=300.0)
SEPARATORE_PAGINE = "\n\n---\n\n"

# Creato al primo uso: senza credenziali Google il converter parte comunque.
_vision_client = None


def safe_move(src: Path, dest: Path):
    """Sposta i file in modo sicuro con sovrascrittura se necessario."""
    if dest.exists(): 
        log.debug(f"Sovrascrittura file esistente in destinazione: {dest.name}")
        dest.unlink()
    shutil.move(str(src), str(dest))
    log.debug(f"File spostato: {src.name} -> {dest.parent.name}/")


# ==============================================================================
# 2. OCR
# ==============================================================================
# Il modello di OCR_PROVIDER (OCR_MODEL_NAME) trascrive le pagine scansionate: gemini,
# openai, anthropic, mistral oppure local, con LOCAL_LLM_BACKEND vllm (LOCAL_VLLM_BASE_URL)
# o ollama (LOCAL_LLM_HOST). Il modello deve leggere le immagini. Se fallisce, il documento
# va in errore: nessun ripiego automatico.

_OCR = None
_EXTERNAL_FALLBACK = None
EXTERNAL_PROVIDERS = ("gemini", "openai", "anthropic", "mistral")


def _ocr_config(provider: str, model: str, api_key: str) -> Ocr:
    """L'OCR di pa-common con i limiti del converter (OCR_RATE_PER_MINUTE, OCR_MAX_ATTEMPTS)."""
    opzioni = dict(al_minuto=settings.ocr_rate_per_minute, tentativi=settings.ocr_max_attempts)
    if provider == "local":
        if settings.local_llm_backend == "vllm":
            return crea_ocr("vllm", model, api_key=settings.local_vllm_api_key or "", base_url=settings.local_vllm_base_url, **opzioni)
        if settings.local_llm_backend == "ollama":
            return crea_ocr("ollama", model, base_url=settings.local_llm_host, **opzioni)
        raise ValueError(f"Unknown LOCAL_LLM_BACKEND='{settings.local_llm_backend}'. Valid options: ['ollama', 'vllm']")
    if provider not in EXTERNAL_PROVIDERS:
        raise ValueError(f"Unknown OCR_PROVIDER='{provider}'. Valid options: {list(EXTERNAL_PROVIDERS) + ['local']}")
    return crea_ocr(provider, model, api_key=api_key, **opzioni)


def get_ocr() -> Ocr:
    """Il modello OCR principale, creato al primo uso."""
    global _OCR
    if _OCR is None:
        if not settings.ocr_model_name:
            raise ValueError(f"OCR_PROVIDER='{settings.ocr_provider}' richiede OCR_MODEL_NAME: nessun modello impostato.")
        _OCR = _ocr_config(settings.ocr_provider, settings.ocr_model_name, settings.api_llm_key)
        log.info(f"OCR provider: {settings.ocr_provider} (modello {settings.ocr_model_name})")
    return _OCR


def get_external_fallback() -> Ocr | None:
    """Il modello esterno per il comando del revisore, None se OCR_EXTERNAL_FALLBACK non è attivo. Attivo ma incompleto è un errore."""
    global _EXTERNAL_FALLBACK
    if not settings.ocr_external_fallback:
        return None
    if _EXTERNAL_FALLBACK is None:
        if settings.ocr_external_provider not in EXTERNAL_PROVIDERS:
            raise ValueError(f"OCR_EXTERNAL_ALTERNATIVE_PROVIDER='{settings.ocr_external_provider}' non valido. "
                             f"Valori ammessi: {list(EXTERNAL_PROVIDERS)}")
        if not settings.ocr_external_model_name or not settings.ocr_external_api_key:
            raise ValueError("OCR_EXTERNAL_FALLBACK=true richiede OCR_EXTERNAL_ALTERNATIVE_MODEL_NAME e OCR_EXTERNAL_API_KEY.")
        _EXTERNAL_FALLBACK = _ocr_config(settings.ocr_external_provider, settings.ocr_external_model_name,
                                         settings.ocr_external_api_key)
        log.info(f"Ripiego OCR esterno: {settings.ocr_external_provider} (modello {settings.ocr_external_model_name})")
    return _EXTERNAL_FALLBACK


def vision_enabled() -> bool:
    """Vision si usa solo se OCR_VISION_FALLBACK lo ammette e le credenziali ci sono."""
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    return settings.ocr_vision_fallback and bool(credentials) and Path(credentials).is_file() and Path(credentials).stat().st_size > 0


def _cloud_vision_transcribe(png: bytes) -> str:
    """Trascrive la pagina con Google Cloud Vision: solo dal comando del revisore."""
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    log.info("Invio della pagina a Google Cloud Vision...")
    response = _vision_client.document_text_detection(image=vision.Image(content=png))
    if response.error.message:
        log.error(f"Errore Cloud Vision API: {response.error.message}")
        raise Exception(f"Cloud Vision API Error: {response.error.message}")
    log.debug("Trascrizione con Cloud Vision completata.")
    return response.full_text_annotation.text


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


async def _external_transcribe_page(png: bytes) -> str:
    """Una pagina per il comando del revisore: il modello esterno, poi Vision se abilitato."""
    # Fuori dal try: una configurazione sbagliata ferma il lavoro, non lo manda a Vision.
    external = get_external_fallback()
    failure = "modello esterno non attivo"
    if external:
        try:
            return await external.trascrivi(png, markdown=True)
        except Exception as e:
            failure = e
            log.warning(f"OCR esterno non riuscito ({e}).")
    if not vision_enabled():
        raise ValueError(f"{failure}; Google Cloud Vision non è abilitato.")
    return (await asyncio.to_thread(_cloud_vision_transcribe, png)).strip()


async def transcribe_externally(file_path: Path, filename: str) -> str:
    """Tutte le pagine all'OCR esterno, per scelta del revisore: non passa dal modello di OCR_PROVIDER."""
    if file_path.suffix.lower() in IMG_EXTENSIONS:
        return await _external_transcribe_page(file_path.read_bytes())
    pdf_bytes = ingresso.sbusta_p7m(file_path.read_bytes()) or file_path.read_bytes()
    pages = await asyncio.to_thread(processo.esegui, rasterizza, (pdf_bytes,), TEMPI_PDF.rasterizza)
    if len(pages) > settings.max_allowed_pages:
        raise ValueError(f"Il file ha {len(pages)} pagine, oltre il limite di {settings.max_allowed_pages}.")
    texts = []
    for png in pages:
        texts.append(await _external_transcribe_page(png))
    return SEPARATORE_PAGINE.join(texts)


# ==============================================================================
# 3. CORE LOGIC - GESTIONE MESSAGGI (EVENT-DRIVEN)
# ==============================================================================

def _ocr_o_niente() -> Ocr | None:
    """Il modello OCR, None se non è configurabile: i documenti scansionati vanno in errore
    con il motivo (OcrNonDisponibile), gli altri si convertono."""
    try:
        return get_ocr()
    except (ValueError, ErroreOcr, ImportError) as e:
        log.error(f"OCR non utilizzabile: {e}")
        return None


async def convert_file_to_markdown(attached_file_path: Path, filename: str, stats: Statistiche | None = None) -> str | None:
    """Testo Markdown di un file; None se il formato non è gestito.

    pa-common sbusta le firme (anche un ".pdf" che è ancora una busta CAdES non sbustata a
    monte da Sicr@Web), controlla che il contenuto corrisponda all'estensione, converte i PDF
    pagina per pagina con l'OCR delle pagine scansionate, i documenti d'ufficio con
    LibreOffice e i fogli di calcolo con una tabella per foglio.
    """
    attached_ext = ingresso.estensione(attached_file_path.name)
    if attached_ext not in FORMATI and attached_ext not in ingresso.BUSTE:
        log.warning(f"Estensione {attached_ext} ignorata per il file {filename}.")
        return None
    return await in_markdown(
        attached_file_path.read_bytes(),
        attached_file_path.name,
        ocr=_ocr_o_niente(),
        max_pagine=settings.max_allowed_pages,
        separatore_pagine=SEPARATORE_PAGINE,
        tempi=TEMPI_PDF,
        statistiche=stats,
    )


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
            stats = Statistiche()
            md_content = await convert_file_to_markdown(attached_file_path, filename, stats)

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
                                        stats.pagine_ocr, stats.pagine)

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