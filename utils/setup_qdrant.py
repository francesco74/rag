"""
Crea le collection di Qdrant che mancano (document_chunks, semantic_cache,
conceptual_dictionary) e i loro indici sul payload. Non tocca le collection esistenti:
si può rilanciare quando si vuole. Esce con errore se Qdrant non risponde.

    python -m utils.setup_qdrant
"""
import asyncio
import logging
import sys

from qdrant_client import AsyncQdrantClient, models

from common.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("qdrant_setup")

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
CONCEPT_COLLECTION = "conceptual_dictionary"  

def collection_params(name, dimension):
    """Parametri di creazione di una collection, condivisi con utils/migrate_embeddings.py.

    document_chunks, con QDRANT_CHUNKS_QUANTIZATION (default), tiene i vettori
    pieni su disco e in RAM solo una copia int8 (un quarto dello spazio): la
    ricerca scorre la copia compressa e riordina i migliori risultati con i
    vettori pieni (rag/retrieval.py, QDRANT_OVERSAMPLING). Le altre collection
    sono piccole e restano in RAM.
    """
    params = {"vectors_config": models.VectorParams(size=dimension, distance=models.Distance.COSINE)}
    if name == QDRANT_COLLECTION and settings.qdrant_chunks_quantization:
        params["vectors_config"].on_disk = True
        params["quantization_config"] = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True))
    return params


async def create_collection_if_missing(client, name):
    """Crea la collection solo se non esiste."""
    if not await client.collection_exists(name):
        log.info(f"Creazione collection: {name} (dimensione {settings.embedding_dimension})")
        await client.create_collection(collection_name=name, **collection_params(name, settings.embedding_dimension))
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
    client = None
    try:
        client = AsyncQdrantClient(
            host=settings.qdrant_host, 
            port=settings.qdrant_port,
            timeout=60.0
        )
        log.info("Connessione a Qdrant stabilita.")

        # 1. CREAZIONE COLLECTION (Inclusa quella concettuale)
        await create_collection_if_missing(client, QDRANT_COLLECTION)
        await create_collection_if_missing(client, CACHE_COLLECTION)
        await create_collection_if_missing(client, CONCEPT_COLLECTION) # <-- Creazione della nuova collection

        log.info("Inizio configurazione indici (Payload Indexes)...")

        # 2. INDICI PER DOCUMENT CHUNKS (Sistema Base + Nuovi Metadati)
        indexes_chunks = {
            "topic_id": models.PayloadSchemaType.KEYWORD,
            "sub_topic_id": models.PayloadSchemaType.KEYWORD,
            "source": models.PayloadSchemaType.KEYWORD,
            "parent_id": models.PayloadSchemaType.KEYWORD,
            "file_name": models.PayloadSchemaType.KEYWORD,
            "percorso_originale": models.PayloadSchemaType.TEXT,
            "content_hash": models.PayloadSchemaType.KEYWORD,
            "content": models.TextIndexParams(
                type="text", tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True
            ),
            # Campi data usati per il filtro range temporale opzionale in chat.
            # Sono stringhe "YYYY-MM-DD" (vedi estrattore.py: clean_iso_date()),
            # quindi vanno indicizzate come DATETIME per usare models.DatetimeRange
            # in worker.py (retrieve_chunks) in modo efficiente e validato.
            "data": models.PayloadSchemaType.DATETIME,
        }

        for field, schema in indexes_chunks.items():
            await safe_create_payload_index(client, QDRANT_COLLECTION, field, schema)

        # 3. INDICI PER SEMANTIC CACHE (Aggiornati con i filtri)
        indexes_cache = {
            "topic_id": models.PayloadSchemaType.KEYWORD,
            "sub_topics_key": models.PayloadSchemaType.KEYWORD,
            "filters_key": models.PayloadSchemaType.KEYWORD 
        }
        for field, schema in indexes_cache.items():
            await safe_create_payload_index(client, CACHE_COLLECTION, field, schema)

        # 4. INDICI PER CONCEPTUAL DICTIONARY (Nuovo Dizionario Semantico)
        # Indicizziamo 'concept' e 'aliases' per permetterti, se necessario, 
        # di fare anche lookup testuali diretti o eliminazioni mirate via codice.
        indexes_concept = {
            "concept": models.PayloadSchemaType.KEYWORD,
            "aliases": models.PayloadSchemaType.KEYWORD
        }
        for field, schema in indexes_concept.items():
            await safe_create_payload_index(client, CONCEPT_COLLECTION, field, schema)

        log.info("Configurazione architettura Vector DB completata con successo.")
        
    except Exception as e:
        log.critical(f"Errore critico durante il setup di Qdrant: {e}")
        raise
    finally:
        if client:
            await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(setup_infrastructure())
    except Exception:
        sys.exit(1)