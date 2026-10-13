import asyncio
from qdrant_client import AsyncQdrantClient, models
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool

async def migrate_qdrant():
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
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