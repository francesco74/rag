import asyncio
import os
import argparse
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

load_dotenv()

async def wipe_qdrant_data(collections_to_delete):
    print(f"Connecting to Qdrant to wipe: {', '.join(collections_to_delete)}...")
    qdrant_client = AsyncQdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"), 
        port=int(os.environ.get("QDRANT_PORT", 6333))
    )
    
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
            
    print("Qdrant wipe complete.")

if __name__ == "__main__":
    # Setup del parser per gli argomenti CLI
    parser = argparse.ArgumentParser(description="Wipe specific Qdrant collections.")
    parser.add_argument(
        "collections", 
        metavar="COLLECTION", 
        type=str, 
        nargs="+", # Richiede ALMENO un parametro, ma ne accetta multipli separati da spazio
        help="Il nome di una o più collection da cancellare (es. document_chunks parent_documents)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(wipe_qdrant_data(args.collections))