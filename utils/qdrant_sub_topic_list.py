"""
Archivi e sottocategorie presenti nei chunk di Qdrant, con il numero di chunk:
utile per confrontarli con la configurazione del sito di amministrazione.

    python -m utils.qdrant_sub_topic_list
"""

from collections import Counter

from qdrant_client import QdrantClient

from common.config import settings


def main():
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=120)
    if not client.collection_exists("document_chunks"):
        raise SystemExit("La collection 'document_chunks' non esiste.")
    pairs, offset = Counter(), None
    while True:
        points, offset = client.scroll("document_chunks", limit=1000, offset=offset,
                                       with_payload=["topic_id", "sub_topic_id"], with_vectors=False)
        pairs.update((p.payload.get("topic_id", "(senza archivio)"),
                      p.payload.get("sub_topic_id", "(senza sottocategoria)")) for p in points)
        if offset is None:
            break
    current = None
    for (topic, sub), n in sorted(pairs.items()):
        if topic != current:
            print(topic)
            current = topic
        print(f"  {sub:40} {n:>8} chunk")


if __name__ == "__main__":
    main()
