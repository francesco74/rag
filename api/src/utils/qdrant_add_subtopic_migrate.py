import asyncio
from qdrant_client import AsyncQdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_HOST=os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT=int(os.environ.get("QDRANT_PORT", 6333))

async def migrate_qdrant():
    client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collection = "document_chunks"

    print("Aggiornamento payload esistenti...")
    await client.set_payload(
        collection_name=collection,
        payload={"sub_topic_id": "carteggio"}, 
        points=models.Filter(
            must=[ 
                models.IsEmptyCondition(is_empty=models.PayloadField(key="sub_topic_id"))
            ]
        )
    )
    print("Migrazione completata.")

if __name__ == "__main__":
    asyncio.run(migrate_qdrant())