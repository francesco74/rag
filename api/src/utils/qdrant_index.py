from qdrant_client import QdrantClient, models
import os
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool

# Adjust host/port to match your setup
client = QdrantClient(host= settings.qdrant_host, port= settings.qdrant_port)

client.create_payload_index(
    collection_name="document_chunks",
    field_name="content",
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        lowercase=True
    )
)
print("✅ Text index created on 'content' field.")