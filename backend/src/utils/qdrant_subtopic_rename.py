import asyncio
from qdrant_client import AsyncQdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

# Attenzione: nessuna virgola alla fine di questa riga!
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

async def migrate_qdrant():
    client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collection = "document_chunks"

    print("Aggiornamento payload: rinomina da 'carteggio' a 'corrispondenza'...")
    
    await client.set_payload(
        collection_name=collection,
        payload={"sub_topic_id": "corrispondenza"},  # Il nuovo valore da inserire
        points=models.Filter(
            must=[ 
                models.FieldCondition(
                    key="sub_topic_id",
                    match=models.MatchValue(value="carteggio") # Cerca esattamente questo valore
                )
            ]
        )
    )
    print("Migrazione completata.")

if __name__ == "__main__":
    asyncio.run(migrate_qdrant())