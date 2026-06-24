import os
import json
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import mysql.connector

# Setup base
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("Migrazione")

# Chiavi di sistema da NON salvare nel campo JSON "metadata" (copiate dal tuo ingest.py)
PROTECTED_KEYS = {"topic_id", "sub_topic_id", "source", "parent_id", "content", 
                  "parent_index", "child_index", "file_name", "_ingestion_error"}

DB_HOST = os.environ.get("MYSQL_SERVICE_HOST", "localhost")
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PORT = int(os.environ.get("MYSQL_SERVICE_PORT", 3306))
DB_PASS = os.environ.get("MYSQL_PASSWORD", "password")
DB_NAME = os.environ.get("MYSQL_DATABASE", "rag_system")

def migrate():
    # 1. Connessioni
    q_client = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"), 
        port=int(os.environ.get("QDRANT_PORT", 6333))
    )
    
    db_conn = mysql.connector.connect(
        host=DB_HOST,
            user=DB_USER,
            port=DB_PORT,
            password=DB_PASS,
            database=DB_NAME
    )
    cursor = db_conn.cursor()

    batch_size = 500
    offset = None
    total_migrated = 0

    log.info("Inizio migrazione da Qdrant a MySQL...")

    while True:
        # 2. Estrazione paginata da Qdrant (Scroll)
        # Ignoriamo i vettori (with_vectors=False) per risparmiare banda e RAM
        records, next_offset = q_client.scroll(
            collection_name="parent_documents",
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        if not records:
            break

        mysql_batch = []
        for point in records:
            payload = point.payload or {}
            
            # Ricostruiamo i metadati filtrando le chiavi di sistema
            metadata_dict = {k: v for k, v in payload.items() if k not in PROTECTED_KEYS}

            mysql_batch.append((
                str(point.id),  # L'ID del point di Qdrant è il nostro parent_id
                payload.get("topic_id", "UNKNOWN"),
                payload.get("sub_topic_id", "UNKNOWN"),
                payload.get("source", "UNKNOWN"),
                payload.get("file_name", None),
                payload.get("parent_index", 0),
                payload.get("content", ""),
                json.dumps(metadata_dict)
            ))

        # 3. Inserimento massivo in MySQL
        # Uso INSERT IGNORE: se lo script si interrompe, puoi rilanciarlo senza errori di duplicazione (Idempotenza)
        insert_query = """
            INSERT IGNORE INTO parent_documents 
            (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try:
            cursor.executemany(insert_query, mysql_batch)
            db_conn.commit()
            total_migrated += cursor.rowcount # Conta solo i record effettivamente inseriti
            log.info(f"Processati {len(records)} record. Offset attuale: {next_offset}")
        except Exception as e:
            db_conn.rollback()
            log.error(f"Errore durante l'inserimento su MySQL: {e}")
            break

        if next_offset is None:
            break
        
        offset = next_offset

    # 4. Cleanup Qdrant (Opzionale ma consigliato)
    cursor.close()
    db_conn.close()
    
    log.info(f"Migrazione completata. {total_migrated} record inseriti in MySQL.")
    log.info("Ricordati di cancellare la collection 'parent_documents' da Qdrant per liberare la RAM!")

if __name__ == "__main__":
    migrate()