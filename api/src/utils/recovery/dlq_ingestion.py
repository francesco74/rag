# Riporta i file .md/.txt e il manifesto .json dalla cartella di errore alla cartella watch, 
# ripulendo il JSON dal tracciato d'errore iniettato originariamente.

import os
import json
import logging
import pika
import shutil
from pathlib import Path
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool


logging.basicConfig(level=settings.log_level, format='%(asctime)s - DLQ INGESTION - %(levelname)s - %(message)s')



DLQ_NAME = "da-indicizzare.dlq"
TARGET_QUEUE = "da-indicizzare"

DATA_FOLDER = settings.data_folder
WATCH_DIR = DATA_FOLDER / "watch"
ERROR_DIR = DATA_FOLDER / "ingestion" / "error"

def safe_move(src: Path, dest: Path):
    if not src.exists(): return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): dest.unlink()
    shutil.move(str(src), str(dest))
    return True

def main():
    logging.info(f"Avvio elaborazione DLQ: {DLQ_NAME}")
    try:
        credentials = pika.PlainCredentials(settings.broker_username, settings.broker_password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=settings.broker_host, port=settings.broker_password, credentials=credentials))
        channel = connection.channel()
    except Exception as e:
        logging.error(f"Connessione fallita: {e}")
        return

    success_count = 0

    while True:
        method_frame, _, body = channel.basic_get(queue=DLQ_NAME, auto_ack=False)
        if method_frame is None: break

        try:
            payload = json.loads(body.decode())
            rel_path = Path(payload.get("json_manifest_path", ""))
            
            err_json_path = ERROR_DIR / rel_path
            watch_json_path = WATCH_DIR / rel_path

            # 1. Ripristino file di testo (.md o .txt)
            for ext in [".md", ".txt"]:
                safe_move(err_json_path.with_suffix(ext), watch_json_path.with_suffix(ext))

            # 2. Ripristino JSON pulendo l'errore iniettato
            if err_json_path.exists():
                watch_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(err_json_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                
                manifest_data.pop("_ingestion_error", None)
                
                with open(watch_json_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest_data, f, indent=2)
                err_json_path.unlink()
                
                channel.basic_publish(exchange='', routing_key=TARGET_QUEUE, body=body, 
                                      properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent))
                success_count += 1
                logging.info(f"✓ Ripristinato e accodato in watch: {rel_path.name}")
            else:
                logging.warning(f"✗ Manifesto JSON {err_json_path.name} non trovato in error. Scartato.")

            channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            
        except Exception as e:
            logging.error(f"Errore gestione messaggio: {e}")
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)

    connection.close()
    logging.info(f"Recovery completato. Messaggi ripristinati: {success_count}")

if __name__ == "__main__":
    main()