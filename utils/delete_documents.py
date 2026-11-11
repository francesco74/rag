"""
Elimina i documenti di una sottocategoria, o un solo documento, da tutto il sistema:

  - Qdrant: i chunk (document_chunks) e le risposte in cache dell'archivio (semantic_cache),
    che potrebbero citarli;
  - MySQL: parent (parent_documents), stato e blocchi di revisione, caricamenti
    (document_uploads) e pagine OCR (document_ocr). Lo storico della revisione
    (review_audit) resta;
  - file: i pacchetti in processed/ (originale, .md e manifest .json) vengono SPOSTATI in
    <DATA_FOLDER>/eliminati/<data_ora>/, non cancellati: per recuperarli basta riportarli in
    processed/ e re-indicizzarli (reindex_topic).

La sottocategoria resta (si elimina, vuota, dal sito di amministrazione), come l'archivio.

Senza --yes mostra solo cosa eliminerebbe. Dentro il pod dell'API o il pod strumenti:

    python -m utils.delete_documents greenlees carteggio                  # cosa eliminerebbe
    python -m utils.delete_documents greenlees carteggio --yes            # elimina
    python -m utils.delete_documents greenlees carteggio --source "SOURCE" --yes   # un solo documento

SOURCE è il campo "source" del documento (lo mostra l'elenco documenti della revisione e
il comando senza --yes).
"""

import argparse
import datetime
import json
import shutil
import sys

from qdrant_client import QdrantClient, models

from common.config import settings
from common.db_logger import get_db_connection, init_db_pool

CHUNKS, CACHE = "document_chunks", "semantic_cache"
PROCESSED = settings.data_folder / "processed"
# Tabelle con una riga per documento (source, topic_id, sub_topic_id).
TABLES = ("parent_documents", "review_status", "review_locks", "document_uploads", "document_ocr")


def _filter(topic, sub, source=None):
    must = [models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic)),
            models.FieldCondition(key="sub_topic_id", match=models.MatchValue(value=sub))]
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    return models.Filter(must=must)


def _where(topic, sub, source=None):
    sql, params = "topic_id = %s AND sub_topic_id = %s", [topic, sub]
    if source:
        sql, params = sql + " AND source = %s", params + [source]
    return sql, params


def _packages(topic, sub, source=None):
    """File in processed/ da spostare: tutta la cartella, o i file del pacchetto di quel source
    (manifest .json con quel "source" e i file con lo stesso nome)."""
    folder = PROCESSED / topic / sub
    if not folder.is_dir():
        return []
    if not source:
        return [p for p in folder.rglob("*") if p.is_file()]
    files = []
    for manifest in folder.rglob("*.json"):
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("source") != source:
                continue
        except (OSError, ValueError):
            continue
        files += [p for p in manifest.parent.glob(f"{manifest.stem}.*") if p.is_file()]
    return files


def main():
    ap = argparse.ArgumentParser(description="Elimina i documenti di una sottocategoria (o uno solo) da Qdrant, MySQL e processed/.")
    ap.add_argument("topic_id")
    ap.add_argument("sub_topic_id")
    ap.add_argument("--source", help="un solo documento (il suo campo source)")
    ap.add_argument("--yes", action="store_true", help="elimina davvero (senza: mostra solo cosa eliminerebbe)")
    args = ap.parse_args()
    topic, sub, source = args.topic_id, args.sub_topic_id, args.source

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=120)
    init_db_pool()
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile.")
    cursor = conn.cursor()
    where, params = _where(topic, sub, source)

    chunks = qdrant.count(CHUNKS, count_filter=_filter(topic, sub, source), exact=True).count
    cursor.execute(f"SELECT DISTINCT source FROM parent_documents WHERE {where} ORDER BY source", params)
    sources = [r[0] for r in cursor.fetchall()]
    rows = {}
    for table in TABLES:
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params)
        rows[table] = cursor.fetchone()[0]
    files = _packages(topic, sub, source)

    scope = f"'{topic}/{sub}'" + (f", documento '{source}'" if source else "")
    print(f"Eliminazione di {scope}:")
    print(f"  Qdrant      {chunks} chunk (più la cache delle risposte dell'archivio '{topic}')")
    for table, n in rows.items():
        print(f"  MySQL       {n:>6} righe in {table}")
    print(f"  processed/  {len(files)} file, da spostare in eliminati/")
    print(f"  documenti   {len(sources)}" + "".join(f"\n    {s}" for s in sources[:20])
          + (f"\n    ... e altri {len(sources) - 20}" if len(sources) > 20 else ""))
    if not (chunks or any(rows.values()) or files):
        print("Niente da eliminare.")
        return
    if not args.yes:
        print("\nNessuna modifica. Per eliminare: aggiungere --yes.")
        return

    # 1. Qdrant: senza i chunk il documento sparisce subito dalle risposte.
    qdrant.delete(CHUNKS, points_selector=models.FilterSelector(filter=_filter(topic, sub, source)), wait=True)
    if qdrant.collection_exists(CACHE):
        qdrant.delete(CACHE, wait=True, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic))])))
    print(f"✓ Qdrant: chunk eliminati, cache dell'archivio '{topic}' svuotata.")

    # 2. MySQL, in una transazione.
    for table in TABLES:
        cursor.execute(f"DELETE FROM {table} WHERE {where}", params)
    conn.commit()
    cursor.close()
    conn.close()
    print("✓ MySQL: righe eliminate (lo storico review_audit resta).")

    # 3. File: spostati, non cancellati.
    if files:
        dest_root = settings.data_folder / "eliminati" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for f in files:
            dest = dest_root / f.relative_to(PROCESSED)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
        print(f"✓ {len(files)} file spostati in {dest_root}.")


if __name__ == "__main__":
    main()
