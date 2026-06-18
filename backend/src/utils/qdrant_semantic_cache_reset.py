from qdrant_client import QdrantClient

# Connect to your local Qdrant instance
client = QdrantClient(host="localhost", port=6333)

# Delete the entire cache collection
client.delete_collection(collection_name="semantic_cache")
print("Qdrant semantic cache flushed successfully!")