"""
Svuota la cache semantica delle risposte (collection semantic_cache): dopo una
re-indicizzazione, una correzione estesa o un cambio di soglie le risposte salvate
potrebbero non corrispondere più ai documenti.

    python -m utils.qdrant_semantic_cache_reset                    # tutta la cache
    python -m utils.qdrant_semantic_cache_reset --topic greenlees   # solo le risposte di un archivio

Tutta la cache: la collection si cancella e si ricrea vuota, con dimensione e indici di
utils/setup_qdrant.py. Un archivio: si cancellano solo le sue risposte.
"""

import argparse
import asyncio

from qdrant_client import QdrantClient, models

from common.config import settings
from utils.setup_qdrant import CACHE_COLLECTION, setup_infrastructure


def main():
    ap = argparse.ArgumentParser(description="Svuota la cache semantica delle risposte.")
    ap.add_argument("--topic", help="solo le risposte di questo archivio")
    args = ap.parse_args()

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    if args.topic:
        if not client.collection_exists(CACHE_COLLECTION):
            print("La cache non esiste: niente da svuotare.")
            return
        flt = models.Filter(must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=args.topic))])
        n = client.count(CACHE_COLLECTION, count_filter=flt, exact=True).count
        client.delete(CACHE_COLLECTION, points_selector=models.FilterSelector(filter=flt), wait=True)
        print(f"✓ Eliminate {n} risposte dell'archivio '{args.topic}' dalla cache.")
        return

    n = client.count(CACHE_COLLECTION, exact=True).count if client.collection_exists(CACHE_COLLECTION) else 0
    client.delete_collection(CACHE_COLLECTION)
    asyncio.run(setup_infrastructure())     # ricrea la cache vuota con i suoi indici
    if not client.collection_exists(CACHE_COLLECTION):
        raise SystemExit("La cache non è stata ricreata: lanciare python -m utils.setup_qdrant e guardarne gli errori.")
    print(f"✓ Cache svuotata ({n} risposte eliminate) e ricreata.")


if __name__ == "__main__":
    main()
