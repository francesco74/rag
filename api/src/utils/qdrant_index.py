from qdrant_client import QdrantClient, models
import os

# Adjust host/port to match your setup
client = QdrantClient(os.environ.get("QDRANT_HOST", "localhost"), port=6333)

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