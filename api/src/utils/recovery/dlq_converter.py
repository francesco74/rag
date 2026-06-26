# Questo script sposta i JSON dalla cartella error alla cartella staging e li riaccoda. Se il JSON è andato perduto, 
# segnala la situazione per ulteriori approfiondimenti.
import os
import json
import logging
import pika
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - DLQ CONVERTER - %(levelname)s - %(message)s')

BROKER_HOST = os.environ.get("BROKER_HOST", "rabbitmq-service.rag.svc.cluster.local")
BROKER_PORT = int(os.environ.get("BROKER_PORT", 5672))
BROKER_USERNAME = os.environ.get("BROKER_USERNAME", "guest")
BROKER_PASSWORD = os.environ.get("BROKER_PASSWORD", "guest")

DLQ_NAME = os.environ.get("CONVERTER_DLQ_NAME", "da-convertire-dlq")
TARGET_QUEUE = "da-convertire"

BASE_DIR = Path("./data")
DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", str(BASE_DIR)))
STAGING_DIR = DATA_FOLDER / "staging"
ERROR_DIR = DATA_FOLDER / "converter" / "error"

def safe_move(src: Path, dest: Path):
    if not src.exists(): return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): dest.unlink()
    shutil.move(str(src), str(dest))
    return True

def main():
    logging.info(f"Avvio elaborazione DLQ: {DLQ_NAME}")
    try:
        credentials = pika.PlainCredentials(BROKER_USERNAME, BROKER_PASSWORD)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=BROKER_HOST, port=BROKER_PORT, credentials=credentials))
        channel = connection.channel()
    except Exception as e:
        logging.error(f"Connessione fallita: {e}")
        return

    success_count, ghost_count = 0, 0

    while True:
        method_frame, _, body = channel.basic_get(queue=DLQ_NAME, auto_ack=False)
        if method_frame is None: break

        try:
            payload = json.loads(body.decode())
            rel_path = Path(payload.get("rel_path", ""))
            
            err_json_path = ERROR_DIR / rel_path
            staging_json_path = STAGING_DIR / rel_path

            # Tenta il ripristino da error a staging (o verifica se è già in staging)
            if safe_move(err_json_path, staging_json_path) or staging_json_path.exists():
                channel.basic_publish(
                    exchange='', 
                    routing_key=TARGET_QUEUE, 
                    body=body, 
                    properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent)
                )
                success_count += 1
                logging.info(f"✓ JSON ripristinato in staging e accodato: {rel_path.name}")
            else:
                # JSON Inesistente -> Alert CRITICAL, ma NESSUNA MODIFICA al filesystem
                ghost_count += 1
                logging.critical(f"✗ GHOST MESSAGE: JSON {rel_path.name} non trovato in error o staging. I binari associati restano intatti in staging per ispezione.")

            # Diamo ACK per smaltire il messaggio dalla DLQ (sia ripristinato, sia fantasma)
            channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            
        except Exception as e:
            logging.error(f"Errore gestione messaggio: {e}")
            # Se lo script in sé fallisce (es. formato JSON rotto), rimettiamo in coda DLQ
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)

    connection.close()
    logging.info(f"Recovery completato. Ripristinati: {success_count} | Messaggi Fantasma: {ghost_count}")

if __name__ == "__main__":
    main()