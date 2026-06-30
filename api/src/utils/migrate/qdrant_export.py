# export_qdrant.py
import os
import asyncio
import requests
from qdrant_client import AsyncQdrantClient
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool

QDRANT_HTTP_URL = f"http://{settings.qdrant_host}:{settings.qdrant_port}"

# Inserisci qui le collection che vuoi esportare dal vecchio server
COLLECTIONS = ["document_chunks", "parent_documents"]

async def export_snapshots():
    print(f"Connessione a Qdrant su {settings.qdrant_host}:{settings.qdrant_port}...")
    
    # Inizializziamo a None per evitare UnboundLocalError nel blocco finally
    client = None
    try:
        # Inizializzazione vecchio stile, compatibile con tutte le versioni dell'SDK
        client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
        for col in COLLECTIONS:
            try:
                if await client.collection_exists(col):
                    print(f"\n[+] Generazione snapshot per '{col}'...")
                    snap = await client.create_snapshot(collection_name=col)
                    
                    file_path = f"{col}.snapshot"
                    print(f"[*] Download di {snap.name} in corso...")
                    
                    download_url = f"{QDRANT_HTTP_URL}/collections/{col}/snapshots/{snap.name}"
                    
                    r = requests.get(download_url, stream=True)
                    r.raise_for_status() 
                    
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"✓ Snapshot salvato localmente: {file_path}")
                else:
                    print(f"[-] Collection '{col}' non esistente sul server. Saltata.")
                    
            except Exception as e:
                print(f"× Errore durante l'export di '{col}': {e}")
                
    except Exception as init_error:
        print(f"× Errore critico di inizializzazione del client Qdrant: {init_error}")
    finally:
        # Chiusura sicura della sessione se il client è stato istanziato correttamente
        if client is not None:
            await client.close()
            print("\n[*] Connessione con Qdrant chiusa correttamente.")

if __name__ == "__main__":
    asyncio.run(export_snapshots())