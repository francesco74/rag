# parent_documents_qdrant2mysql.py
import os
import json
import asyncio
import logging
from qdrant_client import AsyncQdrantClient
import mysql.connector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - REIMPORT - %(levelname)s - %(message)s')
log = logging.getLogger("ReloadParents")

# ==============================================================================
# CONFIGURAZIONE AMBIENTE
# ==============================================================================
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant.rag.svc.cluster.local")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
OLD_COLLECTION_NAME = "parent_documents"

DB_HOST = os.environ.get("MYSQL_SERVICE_HOST", "localhost")
DB_PORT = int(os.environ.get("MYSQL_SERVICE_PORT", 3306))
DB_USER = os.environ.get("MYSQL_USER", "raguser")
DB_PASS = os.environ.get("MYSQL_PASSWORD", "")
DB_NAME = os.environ.get("MYSQL_DATABASE", "rag_db")


async def reload_parents_with_correct_source():
    log.info(f"Connessione a Qdrant su {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant_client = None
    
    try:
        qdrant_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        if not await qdrant_client.collection_exists(OLD_COLLECTION_NAME):
            log.error(f"× La collection '{OLD_COLLECTION_NAME}' non esiste in Qdrant. Impossibile ricaricare.")
            return

        log.info("Estrazione record da Qdrant...")
        all_records = []
        
        # Paginazione / Scroll dei dati da Qdrant
        records, next_page = await qdrant_client.scroll(
            collection_name=OLD_COLLECTION_NAME, limit=100, with_payload=True, with_vectors=False
        )
        all_records.extend(records)
        
        while next_page:
            records, next_page = await qdrant_client.scroll(
                collection_name=OLD_COLLECTION_NAME, limit=100, offset=next_page, with_payload=True, with_vectors=False
            )
            all_records.extend(records)

        total_found = len(all_records)
        log.info(f"Scaricate {total_found} entry da Qdrant. Connessione a MySQL...")
        
        try:
            conn = mysql.connector.connect(
                host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
            )
        except Exception as conn_err:
            log.error(f"× Connessione a MySQL fallita: {conn_err}")
            return

        try:
            with conn.cursor() as cursor:
                log.info("Inizio trasformazione e inserimento dei nuovi record...")
                inserted_count = 0
                
                for record in all_records:
                    p = record.payload
                    if not p:
                        continue
                    
                    # ----------------------------------------------------------
                    # TRASFORMAZIONE AL VOLO DEL CAMPO SOURCE (Il fulcro dello script)
                    # ----------------------------------------------------------
                    old_source = p.get("source", "unknown_source")
                    clean_source = old_source
                    
                    # Rimuoviamo l'estensione originale (.pdf, .txt, .md)
                    for ext in [".pdf", ".txt", ".md"]:
                        if clean_source.endswith(ext):
                            clean_source = clean_source[:-len(ext)]
                    
                    # Ricostruiamo il source con il formato esteso richiesto
                    correct_source = f"direct://greenlees/corrispondenza/{clean_source}"
                    # ----------------------------------------------------------

                    # Inserimento nel DB con il source corretto
                    cursor.execute("""
                        INSERT INTO parent_documents 
                        (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(record.id),
                        p.get("topic_id", "default_topic"),
                        p.get("sub_topic_id", "default_sub"),
                        correct_source,  # <--- Usiamo il valore trasformato
                        p.get("file_name", p.get("source", "unknown_file")),
                        p.get("parent_index", 0),
                        p.get("content", ""),
                        json.dumps(p)
                    ))
                    inserted_count += cursor.rowcount
                    
            conn.commit()
            log.info(f"✓ Ricaricamento completato con successo! Inseriti {inserted_count} record puliti su MySQL.")
            
        except Exception as db_err:
            conn.rollback()
            log.error(f"× Errore durante il caricamento su MySQL: {db_err}")
        finally:
            conn.close()

    except Exception as qdrant_err:
        log.error(f"× Errore Qdrant: {qdrant_err}")
    finally:
        if qdrant_client is not None:
            await qdrant_client.close()

if __name__ == "__main__":
    asyncio.run(reload_parents_with_correct_source())