# Se un documento ha superato correttamente l'estrazione ed ha il suo JSON pronto nella cartella staging, 
# ma l'evento RabbitMQ è andato perso o non è mai partito, questo script 
# lo individua e lo inietta direttamente nella pipeline del Converter.

import os
import json
import logging
import pika
from pathlib import Path
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool

logging.basicConfig(level=settings.log_level, format='%(asctime)s - STAGING PUBLISHER - %(levelname)s - %(message)s')

DATA_FOLDER = settings.data_folder
STAGING_DIR = DATA_FOLDER / "staging"
TARGET_QUEUE = "da-convertire"

def main():
    if not STAGING_DIR.exists():
        logging.error(f"Cartella staging non trovata: {STAGING_DIR}")
        return

    json_files = list(STAGING_DIR.rglob("*.json"))
    if not json_files:
        logging.warning("Nessun manifesto JSON orfano trovato nello staging.")
        return

    logging.info(f"Trovati {len(json_files)} file JSON. Connessione a RabbitMQ...")
    
    try:
        credentials = pika.PlainCredentials(settings.broker_username, settings.broker_password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=settings.broker_host, port=settings.broker_port, credentials=credentials))
        channel = connection.channel()
    except Exception as e:
        logging.error(f"Impossibile connettersi al broker: {e}")
        return

    success_count = 0
    for json_path in json_files:
        try:
            rel_path = str(json_path.relative_to(STAGING_DIR))
            payload = {
                "source_type": "json",
                "rel_path": rel_path
            }
            
            channel.basic_publish(
                exchange='',
                routing_key=TARGET_QUEUE,
                body=json.dumps(payload).encode(),
                properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent)
            )
            success_count += 1
        except Exception as e:
            logging.error(f"Errore accodamento {json_path.name}: {e}")

    connection.close()
    logging.info(f"Operazione completata! Accodati {success_count}/{len(json_files)} JSON dormienti.")

if __name__ == "__main__":
    main()