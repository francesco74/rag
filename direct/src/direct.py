import os
import json
import logging
import asyncio
import shutil
import uuid
from pathlib import Path

import aio_pika
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - WRAPPER - %(levelname)s - %(message)s')
log = logging.getLogger("WrapperDiretti")

BASE_DIR = Path("./data")
DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", str(BASE_DIR)))
INPUT_DIR = DATA_FOLDER / "direct"  # Cartella dove l'utente/sistema carica i file crudi
STAGING_DIR = DATA_FOLDER / "staging"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}

async def scan_and_wrap(channel: aio_pika.Channel):
    """Scansiona l'inbox, mantiene l'alberatura, genera i JSON in staging e notifica RabbitMQ."""
    files = [f for f in INPUT_DIR.rglob("*.*") if f.is_file()]
    
    if not files:
        log.info("Nessun nuovo file trovato nella cartella 'inbox_diretti'. Esco pulito.")
        return

    log.info(f"Trovati {len(files)} file diretti da normalizzare.")

    for file_path in files:
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            log.warning(f"File non supportato, ignorato: {file_path.name}")
            continue

        # Manteniamo il nome base e il nome completo intatti
        base_name = file_path.stem
        original_filename = file_path.name
        
        # Calcolo dell'alberatura rispetto alla cartella di input
        rel_parent = file_path.relative_to(INPUT_DIR).parent
        
        target_dir = STAGING_DIR / rel_parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Destinazioni finali usando i nomi originali
        new_file_path = target_dir / original_filename
        json_path = target_dir / f"{base_name}.json"

        try:
            # Se esiste già un file con lo stesso nome, shutil.move in ambienti POSIX lo sovrascrive
            if new_file_path.exists():
                log.info(f"Sovrascrittura del file esistente in staging: {original_filename}")
                new_file_path.unlink()

            shutil.move(str(file_path), str(new_file_path))
            
            # Il manifest adotta il path originale esatto come chiave 'source'
            manifest_data = {
                "source": f"direct://{str(rel_parent / base_name).replace(os.sep, '/')}",
                "files": [original_filename],
                "metadati": {
                    "percorso_originale": str(file_path.relative_to(INPUT_DIR))
                }
            }
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            log.info(f"✓ Generato manifest {json_path.name} in {target_dir.name}/")

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
            log.error(f"Errore durante il wrapping del file {original_filename}: {e}")

async def main_run():
    rabbitmq_host = os.environ.get("RABBITMQ_HOST", "rabbitmq-service.rag.svc.cluster.local")
    
    log.info("Avvio Wrapper Diretti (Modalità One-Shot CronJob)...")
    connection = await aio_pika.connect_robust(f"amqp://{rabbitmq_host}/")
    
    async with connection:
        channel = await connection.channel()
        await channel.declare_queue("da-convertire", durable=True)
        
        await scan_and_wrap(channel)
        
    log.info("Esecuzione terminata. Arresto del container.")

if __name__ == "__main__":
    asyncio.run(main_run())