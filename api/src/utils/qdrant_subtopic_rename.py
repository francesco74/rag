import asyncio
from qdrant_client import AsyncQdrantClient, models
from dotenv import load_dotenv
import os
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool


async def migrate_qdrant():
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
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