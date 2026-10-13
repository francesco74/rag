"""
backfill_content_hash.py

Calcola e scrive il campo "content_hash" per tutti i punti già presenti
nella collection Qdrant "document_chunks", usando la STESSA identica
normalizzazione adottata in ingest.py:

    content_norm = " ".join(content.lower().split())
    content_hash = hashlib.md5(content_norm.encode()).hexdigest()

Questo garantisce che gli hash calcolati qui combacino esattamente con
quelli prodotti per i documenti ingeriti dopo l'introduzione del campo,
e con il fallback runtime in worker.py.

USO:
    python backfill_content_hash.py --dry-run          # solo conteggio, nessuna scrittura
    python backfill_content_hash.py                    # esegue il backfill
    python backfill_content_hash.py --topic attiprovincia   # limita a un topic_id
    python backfill_content_hash.py --batch-size 200    # dimensione batch scroll/update

Lo script è idempotente: rieseguirlo più volte non causa danni, i punti
che hanno già un content_hash coerente vengono semplicemente sovrascritti
con lo stesso valore (skippabili con --skip-existing per risparmiare tempo).
"""

import argparse
import hashlib
import logging
import sys
import time

from qdrant_client import QdrantClient, models

from common.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BACKFILL - %(levelname)s - %(message)s'
)
log = logging.getLogger("backfill_content_hash")

QDRANT_COLLECTION = "document_chunks"


def compute_content_hash(content: str) -> str:
    """Stessa normalizzazione usata in ingest.py e nel fallback di worker.py."""
    content_norm = " ".join(content.lower().split())
    return hashlib.md5(content_norm.encode()).hexdigest()


def build_scroll_filter(topic_id: str | None) -> models.Filter | None:
    if not topic_id:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))]
    )


def run_backfill(
    client: QdrantClient,
    batch_size: int,
    dry_run: bool,
    topic_id: str | None,
    skip_existing: bool,
):
    scroll_filter = build_scroll_filter(topic_id)

    total_scanned = 0
    total_missing_content = 0
    total_skipped_existing = 0
    total_updated = 0
    total_errors = 0

    next_offset = None
    start_time = time.time()

    log.info(
        f"Avvio backfill su collection '{QDRANT_COLLECTION}' "
        f"(topic_filter={topic_id or 'ALL'}, dry_run={dry_run}, batch_size={batch_size})"
    )

    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,  # non serve riscrivere i vettori, risparmia banda
            )
        except Exception as e:
            log.error(f"Scroll fallito a offset={next_offset}: {e}")
            total_errors += 1
            break

        if not points:
            break

        update_payload_points: list[tuple[str, str]] = []  # (point_id, content_hash)

        for p in points:
            total_scanned += 1
            payload = p.payload or {}
            content = payload.get("content", "")

            if not content or not content.strip():
                total_missing_content += 1
                log.debug(f"Punto {p.id}: campo 'content' mancante o vuoto, skip.")
                continue

            existing_hash = payload.get("content_hash")
            new_hash = compute_content_hash(content)

            if skip_existing and existing_hash == new_hash:
                total_skipped_existing += 1
                continue

            update_payload_points.append((p.id, new_hash))

        if update_payload_points and not dry_run:
            try:
                # set_payload con singolo valore non è bulk-friendly per valori
                # diversi per punto, quindi raggruppiamo per content_hash quando
                # possibile, altrimenti procediamo punto per punto in batch ridotti.
                # Qdrant supporta set_payload con points_selector su lista di ID,
                # ma il payload dev'essere lo stesso per tutti i punti selezionati,
                # quindi qui usiamo un raggruppamento per hash per ridurre le call.
                hash_to_ids: dict[str, list[str]] = {}
                for pid, h in update_payload_points:
                    hash_to_ids.setdefault(h, []).append(pid)

                for h, ids in hash_to_ids.items():
                    client.set_payload(
                        collection_name=QDRANT_COLLECTION,
                        payload={"content_hash": h},
                        points=ids,
                    )
                total_updated += len(update_payload_points)
            except Exception as e:
                log.error(f"set_payload fallito per batch a offset={next_offset}: {e}")
                total_errors += 1
        elif update_payload_points and dry_run:
            total_updated += len(update_payload_points)  # conteggio "would update"

        log.info(
            f"Progresso: scanned={total_scanned}, "
            f"updated{'(dry-run)' if dry_run else ''}={total_updated}, "
            f"skipped_existing={total_skipped_existing}, "
            f"missing_content={total_missing_content}, "
            f"errors={total_errors}"
        )

        if next_offset is None:
            break

    elapsed = time.time() - start_time
    log.info(
        f"=== Backfill completato in {elapsed:.1f}s ===\n"
        f"  Punti scansionati:        {total_scanned}\n"
        f"  Punti aggiornati:         {total_updated}{' (DRY RUN, nessuna scrittura)' if dry_run else ''}\n"
        f"  Skippati (hash invariato): {total_skipped_existing}\n"
        f"  Senza contenuto:          {total_missing_content}\n"
        f"  Errori:                   {total_errors}"
    )

    if total_errors > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Backfill del campo content_hash su Qdrant.")
    parser.add_argument("--dry-run", action="store_true", help="Calcola e conta senza scrivere su Qdrant.")
    parser.add_argument("--topic", type=str, default=None, help="Limita il backfill a un singolo topic_id.")
    parser.add_argument("--batch-size", type=int, default=200, help="Dimensione batch per scroll/update (default 200).")
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Forza la riscrittura anche dei punti che hanno già un content_hash coerente.",
    )
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60.0)

    try:
        run_backfill(
            client=client,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            topic_id=args.topic,
            skip_existing=args.skip_existing,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()