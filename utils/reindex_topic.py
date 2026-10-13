"""
Re-indicizza i documenti di un archivio (topic) già indicizzati, ripassandoli
dall'ingestione con i pacchetti che sono già in processed/.

Serve quando cambia il modo in cui l'ingestione prepara il testo: per esempio
la pulizia dei rientri &nbsp; dell'OCR (common.utility.clean_ocr_text), che vale
solo per i documenti indicizzati dopo l'aggiornamento dell'immagine di ingest.

Per ogni pacchetto (manifest .json + testo .md/.txt con lo stesso nome) lo
script lo COPIA in watch/ nella stessa posizione e pubblica il messaggio sulla
coda "da-indicizzare", come fa la revisione quando si salva un testo corretto.
L'ingestione crea i nuovi parent e chunk, poi cancella i vecchi: il documento
non sparisce mai dalla ricerca. Stato di revisione e storico restano (sono
legati al documento, non ai parent); i file in processed/ vengono sovrascritti
dalla copia identica che l'ingestione vi rimette a fine lavoro.

Si saltano:
  - i pacchetti già presenti in watch/ (correzione della revisione o
    indicizzazione in corso: non vanno sovrascritti);
  - i documenti non presenti in parent_documents (mai indicizzati o falliti:
    non è compito di questo script);
  - senza --all, i documenti il cui testo INDICIZZATO (parent_documents) non
    contiene &nbsp; né <mark>: già puliti, o la pulizia non cambierebbe nulla
    di rilevante. Il criterio guarda l'indice e non il .md (che non viene mai
    pulito): rilanciando lo script si riprende da dove si era arrivati.

I messaggi partono a lotti: prima di ogni lotto si aspetta che in coda ne
restino al massimo --max-pending, così ingestione ed embedding non vengono
sommersi e l'avanzamento si vede.

Esempi (dentro il pod/container dell'API, che ha DATA_FOLDER, MySQL e RabbitMQ):
    python -m utils.reindex_topic greenlees --dry-run           # cosa verrebbe re-indicizzato
    python -m utils.reindex_topic greenlees                     # avvio, chiede conferma
    python -m utils.reindex_topic greenlees --sub-topic lettere --limit 20 --yes   # prova su pochi
    python -m utils.reindex_topic attiprovincia --all --batch 100 --max-pending 20

Dopo la re-indicizzazione conviene svuotare la cache semantica
(utils/qdrant_semantic_cache_reset.py): le risposte salvate sono state
generate dal contesto precedente.
"""

import argparse
import json
import pathlib
import shutil
import sys
import time

import pika

from common.config import settings
from common.db_logger import get_db_connection, init_db_pool

INGEST_QUEUE = "da-indicizzare"
PROCESSED_FOLDER = settings.data_folder / "processed"
WATCH_FOLDER = settings.data_folder / "watch"
# Ciò che la pulizia dell'OCR toglie davvero (spazi a fine riga e simili non
# valgono una re-indicizzazione), cercato nel testo già indicizzato.
DIRTY_MARKERS = ("&nbsp;", "&#160;", "\xa0", "<mark>")


def find_packages(topic_id, sub_topic_id=None):
    """Pacchetti (json, testo) in processed/<topic>[/<sub_topic>], a qualunque profondità."""
    root = PROCESSED_FOLDER / topic_id
    if sub_topic_id:
        root = root / sub_topic_id
    if not root.is_dir():
        sys.exit(f"Cartella non trovata: {root}")
    for json_path in sorted(root.rglob("*.json")):
        text_path = json_path.with_suffix(".md")
        if not text_path.exists():
            text_path = json_path.with_suffix(".txt")
        if not text_path.exists():
            continue  # non è un manifest di pacchetto (o il testo manca)
        try:
            source = json.loads(json_path.read_text(encoding="utf-8")).get("source")
        except Exception:
            continue
        if source:
            yield json_path, text_path, source


def indexed_sources(topic_id):
    """
    Documenti già indicizzati dell'archivio, come coppie (source, sub_topic_id):
    tutti, e quelli con almeno un parent che contiene ancora DIRTY_MARKERS.
    """
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile")
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source, sub_topic_id FROM parent_documents WHERE topic_id = %s",
                    (topic_id,))
        indexed = {(source, sub) for source, sub in cur.fetchall()}
        like = " OR ".join(["content LIKE %s"] * len(DIRTY_MARKERS))
        cur.execute(f"SELECT DISTINCT source, sub_topic_id FROM parent_documents "
                    f"WHERE topic_id = %s AND ({like})",
                    (topic_id, *[f"%{m}%" for m in DIRTY_MARKERS]))
        dirty = {(source, sub) for source, sub in cur.fetchall()}
        return indexed, dirty
    finally:
        conn.close()


def rabbit_channel():
    params = pika.ConnectionParameters(
        host=settings.broker_host,
        port=settings.broker_port,
        credentials=pika.PlainCredentials(settings.broker_username, settings.broker_password),
        connection_attempts=3,
        retry_delay=2,
        heartbeat=600,
    )
    connection = pika.BlockingConnection(params)
    return connection, connection.channel()


def pending_messages(channel):
    return channel.queue_declare(queue=INGEST_QUEUE, passive=True).method.message_count


def wait_for_queue(connection, channel, max_pending, poll_seconds=10):
    while (n := pending_messages(channel)) > max_pending:
        print(f"  coda '{INGEST_QUEUE}': {n} messaggi in attesa, aspetto...", flush=True)
        # connection.sleep e non time.sleep: tiene vivi gli heartbeat,
        # altrimenti RabbitMQ chiude la connessione durante le attese lunghe.
        connection.sleep(poll_seconds)


def main():
    ap = argparse.ArgumentParser(description="Re-indicizza i documenti già indicizzati di un archivio.")
    ap.add_argument("topic_id")
    ap.add_argument("--sub-topic", help="solo questa sottocategoria")
    ap.add_argument("--all", action="store_true",
                    help="tutti i documenti, non solo quelli con &nbsp; o <mark> nel testo")
    ap.add_argument("--limit", type=int, help="al massimo N documenti (per una prova)")
    ap.add_argument("--batch", type=int, default=200, help="messaggi per lotto (default 200)")
    ap.add_argument("--max-pending", type=int, default=50,
                    help="messaggi in coda oltre i quali si aspetta prima del lotto successivo (default 50)")
    ap.add_argument("--dry-run", action="store_true", help="elenca senza copiare né pubblicare")
    ap.add_argument("--yes", action="store_true", help="non chiedere conferma")
    args = ap.parse_args()

    init_db_pool()
    indexed, dirty = indexed_sources(args.topic_id)
    in_scope = {k for k in indexed if not args.sub_topic or k[1] == args.sub_topic}
    print(f"Archivio '{args.topic_id}'{f' / {args.sub_topic}' if args.sub_topic else ''}: "
          f"{len(in_scope)} documenti indicizzati, "
          f"{len({k for k in dirty if k in in_scope})} con &nbsp; o <mark> nel testo indicizzato.")

    todo, seen = [], set()
    skipped = {"già in watch/": [], "non indicizzati": [], "testo indicizzato già pulito": []}
    for json_path, text_path, source in find_packages(args.topic_id, args.sub_topic):
        rel = json_path.relative_to(PROCESSED_FOLDER)
        key = (source, rel.parts[1])
        seen.add(key)
        if key not in indexed:
            skipped["non indicizzati"].append(rel)
        elif (WATCH_FOLDER / rel).exists() or (WATCH_FOLDER / rel.with_suffix(text_path.suffix)).exists():
            skipped["già in watch/"].append(rel)
        elif not args.all and key not in dirty:
            skipped["testo indicizzato già pulito"].append(rel)
        else:
            todo.append((json_path, text_path, rel))
        if args.limit and len(todo) >= args.limit:
            break

    for reason, items in skipped.items():
        if items:
            print(f"Saltati ({reason}): {len(items)}")
            for rel in items[:10]:
                print(f"    {rel}")
            if len(items) > 10:
                print(f"    ... e altri {len(items) - 10}")
    print(f"Da re-indicizzare: {len(todo)}"
          f"{'' if args.all else ' (testo indicizzato con &nbsp; o <mark>)'}"
          f"{f', limite {args.limit}' if args.limit else ''}.")
    missing = in_scope - seen
    if missing and not args.limit:
        print(f"Senza pacchetto in processed/ (es. dati migrati), non ripassabili da qui: {len(missing)}")
        for source, sub in sorted(missing)[:10]:
            print(f"    {sub}: {source}")
    if not todo:
        return
    if args.dry_run:
        print("Documenti che verrebbero re-indicizzati:")
        for _, _, rel in todo[:20]:
            print(f"    {rel}")
        if len(todo) > 20:
            print(f"    ... e altri {len(todo) - 20}")
        print("Prova a vuoto: nulla copiato né pubblicato.")
        return
    if not args.yes and input(f"Avvio la re-indicizzazione di {len(todo)} documenti? [s/N] ").strip().lower() != "s":
        print("Annullato.")
        return

    connection, channel = rabbit_channel()
    sent = 0
    started = time.time()
    try:
        for i in range(0, len(todo), args.batch):
            wait_for_queue(connection, channel, args.max_pending)
            for json_path, text_path, rel in todo[i:i + args.batch]:
                dest_json = WATCH_FOLDER / rel
                dest_json.parent.mkdir(parents=True, exist_ok=True)
                # Prima il testo, poi il manifest: l'ingest parte dal manifest
                # e si aspetta di trovare già il testo accanto.
                shutil.copy2(text_path, dest_json.with_suffix(text_path.suffix))
                shutil.copy2(json_path, dest_json)
                channel.basic_publish(
                    exchange="",
                    routing_key=INGEST_QUEUE,
                    body=json.dumps({"json_manifest_path": str(rel)}).encode(),
                    properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
                )
                sent += 1
            elapsed = time.time() - started
            print(f"Inviati {sent}/{len(todo)} ({elapsed / 60:.1f} min)", flush=True)
        print("Attendo che la coda si svuoti...")
        wait_for_queue(connection, channel, 0, poll_seconds=15)
    except KeyboardInterrupt:
        print(f"\nInterrotto: {sent} documenti già inviati verranno comunque indicizzati. "
              f"Rilanciando lo script si riprende da dove si era arrivati.")
    finally:
        connection.close()
    print(f"Fatto: {sent} documenti inviati all'ingestione in {(time.time() - started) / 60:.1f} min "
          f"(gli ultimi possono essere ancora in lavorazione). Eventuali errori sono in "
          f"{settings.data_folder / 'ingestion' / 'error'}; rilanciando lo script con --dry-run "
          f"si vede quanti documenti hanno ancora testo da pulire.")


if __name__ == "__main__":
    main()
