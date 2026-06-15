import os
import asyncio
import logging
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("qdrant_setup")

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
PARENT_COLLECTION = "parent_documents"

async def create_collection_if_missing(client, name, size=768):
    """Crea la collection solo se non esiste."""
    if not await client.collection_exists(name):
        log.info(f"Creazione collection: {name}")
        await client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )
    else:
        log.info(f"Collection {name} già esistente.")

async def safe_create_payload_index(client, collection_name, field_name, field_schema):
    """Tenta di creare un indice. Se esiste già, ignora l'errore in modo sicuro."""
    try:
        await client.create_payload_index(
            collection_name=collection_name, 
            field_name=field_name, 
            field_schema=field_schema
        )
        log.debug(f"Indice verificato/creato per '{field_name}' su '{collection_name}'.")
    except Exception as e:
        log.warning(f"Impossibile creare indice '{field_name}' su '{collection_name}': {e}")

async def setup_infrastructure():
    try:
        client = AsyncQdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"), 
            port=int(os.environ.get("QDRANT_PORT", 6333)),
            timeout=60.0
        )
        log.info("Connessione a Qdrant stabilita.")

        # 1. CREAZIONE COLLECTION
        await create_collection_if_missing(client, QDRANT_COLLECTION)
        await create_collection_if_missing(client, PARENT_COLLECTION)
        await create_collection_if_missing(client, CACHE_COLLECTION)

        log.info("Inizio configurazione indici (Payload Indexes)...")

        # 2. INDICI PER DOCUMENT CHUNKS (Sistema Base + Nuovi Metadati)
        indexes_chunks = {
            "topic_id": models.PayloadSchemaType.KEYWORD,
            "sub_topic_id": models.PayloadSchemaType.KEYWORD,
            "source": models.PayloadSchemaType.KEYWORD,
            "parent_id": models.PayloadSchemaType.KEYWORD,
            "file_name": models.PayloadSchemaType.KEYWORD,
            "content": models.TextIndexParams(
                type="text", tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True
            ),
        }

        for field, schema in indexes_chunks.items():
            await safe_create_payload_index(client, QDRANT_COLLECTION, field, schema)

        # 3. INDICI PER PARENT DOCUMENTS
        await safe_create_payload_index(client, PARENT_COLLECTION, "source", models.PayloadSchemaType.KEYWORD)

        # 4. INDICI PER SEMANTIC CACHE (Aggiornati con i filtri)
        indexes_cache = {
            "topic_id": models.PayloadSchemaType.KEYWORD,
            "sub_topics_key": models.PayloadSchemaType.KEYWORD, # stringa generata "al volo" dal backend unendo in ordine alfabetico tutti i sub-topic che l'utente ha selezionato per una specifica ricerca
            "filters_key": models.PayloadSchemaType.KEYWORD # Indispensabile per isolare la cache dei filtri
        }
        for field, schema in indexes_cache.items():
            await safe_create_payload_index(client, CACHE_COLLECTION, field, schema)

        log.info("Configurazione architettura Vector DB completata con successo.")
        
    except Exception as e:
        log.critical(f"Errore critico durante il setup di Qdrant: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(setup_infrastructure())