import asyncio
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

load_dotenv()

async def wipe_qdrant_data():
    print("Connecting to Qdrant...")
    qdrant_client = AsyncQdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"), 
        port=int(os.environ.get("QDRANT_PORT", 6333))
    )
    
    # The collections defined in your ingestion script
    collections_to_delete = ["document_chunks", "semantic_cache"]
    
    for collection in collections_to_delete:
        try:
            # delete_collection returns True if successful, False if it didn't exist
            success = await qdrant_client.delete_collection(collection_name=collection)
            if success:
                print(f"✅ Successfully deleted collection: '{collection}'")
            else:
                print(f"⚠️ Collection '{collection}' did not exist (already clean).")
        except Exception as e:
            print(f"❌ Error deleting '{collection}': {e}")
            
    print("Qdrant wipe complete. You are ready to restart ingestion.")

if __name__ == "__main__":
    asyncio.run(wipe_qdrant_data())