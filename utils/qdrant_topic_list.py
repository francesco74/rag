"""
Archivi (topic_id) presenti nei chunk di Qdrant: utile per confrontarli con quelli di
MySQL (python -m utils.manage_topics list).

    python -m utils.qdrant_topic_list
"""

from qdrant_client import QdrantClient

from common.config import settings


def main():
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=120)
    topics, offset = set(), None
    while True:
        points, offset = client.scroll("document_chunks", limit=1000, offset=offset,
                                       with_payload=["topic_id"], with_vectors=False)
        topics.update(p.payload.get("topic_id") for p in points if p.payload.get("topic_id"))
        if offset is None:
            break
    print(f"{len(topics)} archivi in document_chunks:")
    for t in sorted(topics):
        print(f"  {t}")


if __name__ == "__main__":
    main()
