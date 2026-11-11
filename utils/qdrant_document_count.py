"""
Chunk e documenti indicizzati in Qdrant (document_chunks), in totale e per archivio.

    python -m utils.qdrant_document_count
"""

from collections import Counter

from qdrant_client import QdrantClient

from common.config import settings

COLLECTION = "document_chunks"


def main():
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=120)
    if not client.collection_exists(COLLECTION):
        raise SystemExit(f"La collection '{COLLECTION}' non esiste.")
    chunks, sources, offset = Counter(), {}, None
    while True:
        points, offset = client.scroll(COLLECTION, limit=1000, offset=offset,
                                       with_payload=["topic_id", "source"], with_vectors=False)
        for p in points:
            topic = p.payload.get("topic_id", "?")
            chunks[topic] += 1
            sources.setdefault(topic, set()).add(p.payload.get("source"))
        if offset is None:
            break
    print(f"{'archivio':30} {'documenti':>10} {'chunk':>10}")
    for topic in sorted(chunks):
        print(f"{topic:30} {len(sources[topic]):>10} {chunks[topic]:>10}")
    print(f"{'totale':30} {sum(len(s) for s in sources.values()):>10} {sum(chunks.values()):>10}")


if __name__ == "__main__":
    main()
