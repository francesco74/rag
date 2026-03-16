from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

print("Scanning entire database for 'Anderson'...")

# Fetch up to 1000 chunks (bypassing all filters)
res = client.scroll(
    collection_name="document_chunks", 
    limit=1000, 
    with_payload=True
)

found = False
for hit in res[0]:
    content = hit.payload.get("content", "").lower()
    if "anderson" in content:
        print("\n✅ DOCUMENT FOUND IN QDRANT!")
        print(f"-> Assigned Topic ID: {hit.payload.get('topic_id')}")
        print(f"-> Content excerpt: {hit.payload.get('content')[:150]}...")
        found = True

if not found:
    print("\n❌ DOCUMENT NOT FOUND. It was never successfully ingested.")