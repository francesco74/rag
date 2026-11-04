import json
import logging
import asyncio
import shutil
from pathlib import Path

import aio_pika
from common.config import settings
from common.file_intake import (INPUT_EXTENSIONS, SUPPORTED_EXTENSIONS, detect_extension_from_content,
                                try_extract_from_pkcs7, unwrapped_name)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - WRAPPER - %(levelname)s - %(message)s')
log = logging.getLogger("WrapperDiretti")

DATA_FOLDER = settings.data_folder
INPUT_DIR = DATA_FOLDER / "direct"  # Cartella dove l'utente/sistema carica i file crudi
STAGING_DIR = DATA_FOLDER / "staging"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

async def scan_and_wrap(channel: aio_pika.Channel):
    """
    Scansiona l'inbox. Ogni file DEVE trovarsi in
    INPUT_DIR/{topic_id}/{sub_topic_id}/{eventuali sottocartelle}/{nome_file},
    es. INPUT_DIR/TOPICX/SUBTOPICY/preliminari/gennaio/test.pdf.

    Se il file ha estensione .p7m, viene prima sbustato (PKCS7/CMS): il
    contenuto estratto viene poi verificato via magic bytes per controllare
    che sia uno dei formati ammessi (SUPPORTED_EXTENSIONS) — un file .p7m
    non dice nulla sul contenuto incapsulato finché non lo si apre. Se lo
    sbustamento fallisce, o il contenuto risultante non è tra i formati
    ammessi, il file viene ignorato con un warning esplicito (nessuna
    eccezione silenziosa).

    Il file e il relativo manifest JSON vengono spostati mantenendo
    l'INTERA alberatura originale: cambia solo la radice, da INPUT_DIR a
    STAGING_DIR (es. STAGING_DIR/TOPICX/SUBTOPICY/preliminari/gennaio/), e
    restano co-locati nella stessa sottocartella (nessun rischio di
    collisione: cartelle diverse per file diversi, anche con lo stesso nome
    base).

    "files" contiene il percorso relativo COMPLETO rispetto alla radice
    topic/sub_topic (es. "preliminari/gennaio/test.pdf"), che diventa il
    valore di file_name in DB — non solo il nome base. converter.py risolve
    comunque il file fisico correttamente perché usa solo il nome base di
    questo valore per la ricerca sul filesystem (il file è co-locato con il
    JSON), mantenendo però il percorso completo nel manifest per DB/Qdrant.
    """
    files = [f for f in INPUT_DIR.rglob("*.*") if f.is_file()]

    if not files:
        log.info("Nessun nuovo file trovato nella cartella 'inbox_diretti'. Esco pulito.")
        return

    log.info(f"Trovati {len(files)} file diretti da normalizzare.")

    for file_path in files:
        ext = file_path.suffix.lower()
        if ext not in INPUT_EXTENSIONS:
            log.warning(f"File non supportato, ignorato: {file_path.name}")
            continue

        rel_path = file_path.relative_to(INPUT_DIR)
        parts = rel_path.parts

        # Servono almeno topic_id, sub_topic_id, e il nome del file.
        if len(parts) < 3:
            log.warning(
                f"Percorso non conforme (attesi topic_id/sub_topic_id/.../file): "
                f"'{rel_path}'. Atteso INPUT_DIR/<topic_id>/<sub_topic_id>/.../file. Ignorato."
            )
            continue

        topic_id, sub_topic_id = parts[0], parts[1]
        subfolder_parts = parts[2:-1]  # eventuali sottocartelle tra sub_topic_id e il file

        # --- Sbustamento P7M (se applicabile) ---
        # unwrapped_bytes è None per i file già in chiaro (nessuna scrittura
        # esplicita: si usa shutil.move sul file originale, invariato).
        unwrapped_bytes = None
        final_filename = file_path.name

        if ext == ".p7m":
            raw_bytes = file_path.read_bytes()
            try:
                extracted = try_extract_from_pkcs7(raw_bytes, file_path.name)
            except ValueError as e:
                log.error(f"Sbustamento P7M fallito per '{rel_path}': {e}. Ignorato.")
                continue

            if extracted is None:
                log.warning(f"'{rel_path}' ha estensione .p7m ma non è una busta PKCS7/CMS valida. Ignorato.")
                continue

            detected_ext = detect_extension_from_content(extracted)
            if detected_ext is None:
                log.warning(
                    f"'{rel_path}': sbustamento riuscito ma il contenuto non corrisponde a "
                    f"nessun formato ammesso ({sorted(SUPPORTED_EXTENSIONS)}). Ignorato."
                )
                continue

            unwrapped_bytes = extracted

            # Nome finale: rimuove ".p7m" e usa l'estensione rilevata.
            # "fattura.pdf.p7m" -> "fattura.pdf" (già corretta)
            # "fattura.p7m" -> "fattura.pdf" (estensione mancante, aggiunta)
            final_filename = unwrapped_name(file_path.name, detected_ext)

            log.info(f"✓ Sbustato '{rel_path}' -> '{final_filename}' (rilevato: {detected_ext}).")

        # Percorso relativo alla radice topic/sub_topic, es.
        # "preliminari/gennaio/test.pdf" — usa final_filename (post-sbustamento
        # se applicabile). Diventa il valore di "files" (-> file_name in DB),
        # NON la posizione fisica in staging (quella mantiene l'intera
        # alberatura originale, vedi target_dir sotto).
        filename_rel = "/".join((*subfolder_parts, final_filename))

        base_name = Path(final_filename).stem

        # Il file mantiene l'INTERA alberatura originale: cambia solo la
        # radice, da INPUT_DIR a STAGING_DIR.
        target_dir = STAGING_DIR / rel_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        new_file_path = target_dir / final_filename
        # JSON co-locato con il file, stessa sottocartella: nessuna
        # collisione possibile, dato che due file con lo stesso nome ma in
        # sottocartelle diverse finiscono in target_dir diversi.
        json_path = target_dir / f"{base_name}.json"

        try:
            if new_file_path.exists():
                log.info(f"Sovrascrittura del file esistente in staging: {rel_path}")
                new_file_path.unlink()

            if unwrapped_bytes is not None:
                # Contenuto sbustato: scriviamo i bytes decifrati e rimuoviamo
                # l'originale .p7m dall'inbox (non c'è nulla da "spostare",
                # il contenuto scritto è diverso da quello letto).
                new_file_path.write_bytes(unwrapped_bytes)
                file_path.unlink()
            else:
                shutil.move(str(file_path), str(new_file_path))

            # Il manifest adotta il path originale esatto come chiave 'source'
            manifest_data = {
                "source": f"direct://{topic_id}/{sub_topic_id}/{filename_rel}",
                "files": [filename_rel],
                "metadati": {
                    "percorso_originale": str(rel_path)
                }
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            log.info(f"✓ Generato manifest {json_path.name} in {target_dir}/")

            rel_json_path = str(json_path.relative_to(STAGING_DIR))

            payload = {
                "source_type": "json",
                "rel_path": rel_json_path
            }

            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key="da-convertire"
            )

        except Exception as e:
            log.error(f"Errore durante il wrapping del file {rel_path}: {e}")

async def main_run():
    conn_string = f"amqp://{settings.broker_username}:{settings.broker_password}@{settings.broker_host}:{settings.broker_port}/"

    log.info("Avvio Wrapper Diretti (Modalità One-Shot CronJob)...")
    connection = await aio_pika.connect_robust(conn_string)
    
    async with connection:
        channel = await connection.channel()
        # get_queue() passivo, NON declare_queue(): la coda "da-convertire"
        # è creata dall'infrastruttura con argomenti specifici (es.
        # x-dead-letter-exchange) che qui non conosciamo. Una declare_queue()
        # senza gli stessi argomenti esatti fallisce con
        # ChannelPreconditionFailed se la coda esiste già con argomenti
        # diversi — stesso pattern già usato in converter.py.
        await channel.get_queue("da-convertire")
        
        await scan_and_wrap(channel)
        
    log.info("Esecuzione terminata. Arresto del container.")

if __name__ == "__main__":
    asyncio.run(main_run())