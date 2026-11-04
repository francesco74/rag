"""
Allinea una tantum le date dei metadati al formato ISO AAAA-MM-GG, l'unico che
i filtri della chat e la revisione capiscono. Una data in un altro formato non
dà errori: il documento sparisce in silenzio dai filtri temporali.

Vale per le chiavi "data" e "data_*" (anche "data-*", "data.*", "data *").
Per ogni valore:
  - riconosciuto (con o senza ora, GG/MM/AAAA, GG-MM-AAAA, GG.MM.AAAA,
    AAAA/MM/GG, AAAAMMGG, "26 novembre 2025", anni a due cifre): riscritto in ISO.
    L'ordine è sempre giorno/mese, come negli atti italiani;
  - parziale (2024, 03/2024, 2024-03): la data non è determinabile, il campo
    viene tolto e l'anno finisce in "anno" se non c'è già;
  - sentinella (0001-01-01, anni prima del 1000) o non riconosciuto: il campo
    viene tolto. Un documento senza data resta visibile nei filtri temporali.

Si aggiornano, nell'ordine:
  1. i manifest in processed/ (così una re-indicizzazione non riporta i vecchi valori);
  2. per ogni documento di parent_documents, il payload dei chunk su Qdrant e
     poi i metadati su MySQL, che fanno da segnaposto: un documento interrotto
     a metà risulta ancora da fare e rilanciando lo script viene completato.

I valori originali restano nella tabella date_alignment_backup e l'elenco di
ciò che è stato tolto in <DATA_FOLDER>/date_alignment_report_<data_ora>.csv. Rilanciare lo
script non sovrascrive il backup e non cambia ciò che è già allineato.

Esempi (dentro il pod/container dell'API):
    python -m utils.align_dates --dry-run
    python -m utils.align_dates
    python -m utils.align_dates --topic greenlees

Alla fine conviene svuotare la cache semantica (utils/qdrant_semantic_cache_reset.py).
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

from common.config import settings
from common.db_logger import get_db_connection, init_db_pool

QDRANT_COLLECTION = "document_chunks"
PROCESSED_FOLDER = settings.data_folder / "processed"
# Un file per esecuzione: un rilancio, che non trova più nulla, non deve cancellare l'elenco.
REPORT_PATH = settings.data_folder / f"date_alignment_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"

DATE_KEY_RE = re.compile(r"^data($|[_\-. ])")
MONTHS = {name: i + 1 for i, name in enumerate(
    ("gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre").split())}
MONTHS.update({name[:3]: number for name, number in list(MONTHS.items())})

_YMD_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[t ].*)?$")
_DMY_RE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?:[t ].*)?$")
_COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_WORDS_RE = re.compile(r"^(\d{1,2})[°ºo]?\s+([a-zà]+)\.?\s+(\d{4})$")
_YEAR_RE = re.compile(r"^(\d{4})(?:\.0)?$")
_YEAR_MONTH_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})$")
_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})[-/.](\d{4})$")
_MONTH_WORD_YEAR_RE = re.compile(r"^([a-zà]+)\.?\s+(\d{4})$")


def _full_year(year):
    if len(year) == 4:
        return int(year)
    # Anni a due cifre: nel futuro non ci sono atti, quindi è il secolo scorso.
    return 2000 + int(year) if int(year) <= datetime.date.today().year % 100 else 1900 + int(year)


def _build_date(year, month, day):
    if year < 1000:
        return ("sentinel", None)
    try:
        return ("iso", datetime.date(year, month, day).isoformat())
    except ValueError:
        return ("invalid", None)


def parse_date(value):
    """
    Classifica un valore: ("iso", "AAAA-MM-GG"), ("partial", "AAAA"),
    ("sentinel", None), ("invalid", None) o ("empty", None).
    """
    if value is None or isinstance(value, bool):
        return ("empty", None) if value is None else ("invalid", None)
    text = str(value).replace("\xa0", " ").strip().lower()
    if not text:
        return ("empty", None)

    if m := _YMD_RE.match(text):
        return _build_date(int(m[1]), int(m[2]), int(m[3]))
    if m := _DMY_RE.match(text):
        return _build_date(_full_year(m[3]), int(m[2]), int(m[1]))
    if m := _COMPACT_RE.match(text):
        return _build_date(int(m[1]), int(m[2]), int(m[3]))
    if (m := _WORDS_RE.match(text)) and m[2] in MONTHS:
        return _build_date(int(m[3]), MONTHS[m[2]], int(m[1]))

    year = None
    if m := _YEAR_RE.match(text):
        year = m[1]
    elif (m := _YEAR_MONTH_RE.match(text)) and 1 <= int(m[2]) <= 12:
        year = m[1]
    elif (m := _MONTH_YEAR_RE.match(text)) and 1 <= int(m[1]) <= 12:
        year = m[2]
    elif (m := _MONTH_WORD_YEAR_RE.match(text)) and m[1] in MONTHS:
        year = m[2]
    if year:
        return ("partial", year) if int(year) >= 1000 else ("sentinel", None)
    return ("invalid", None)


def align_metadata(metadata):
    """
    Allinea un dizionario di metadati. Ritorna (nuovi_metadati, eventi), con un
    evento (chiave, valore_originale, esito) per ogni valore cambiato o tolto.
    Non modifica il dizionario ricevuto.
    """
    result = dict(metadata)
    events = []
    for key, value in metadata.items():
        if not DATE_KEY_RE.match(key) or key in settings.protected_keys:
            continue
        if isinstance(value, list):
            parsed = [parse_date(v) for v in value]
            dates = list(dict.fromkeys(iso for kind, iso in parsed if kind == "iso"))
            if dates != value:
                events.append((key, value, "list"))
                if dates:
                    result[key] = dates
                else:
                    del result[key]
            continue
        kind, parsed = parse_date(value)
        if kind == "iso":
            if parsed != value:
                result[key] = parsed
                events.append((key, value, "iso"))
            continue
        del result[key]
        events.append((key, value, kind))
        if kind == "partial" and not result.get("anno"):
            result["anno"] = parsed
    return result, events


def payload_changes(old, new):
    """Differenza tra due metadati: (chiavi da scrivere, chiavi da togliere)."""
    changed = {k: v for k, v in new.items() if k not in old or old[k] != v}
    removed = sorted(k for k in old if k not in new)
    return changed, removed


class Report:
    """Conteggi per esito e riga CSV per ogni valore che non è stato convertito in una data."""

    def __init__(self, path):
        self.counts = {}
        self.path = path
        self.rows = []

    def add(self, where, topic_id, sub_topic_id, source, events):
        for key, value, kind in events:
            self.counts[(where, kind)] = self.counts.get((where, kind), 0) + 1
            if kind != "iso":
                self.rows.append((where, topic_id, sub_topic_id, source, key, json.dumps(value, ensure_ascii=False), kind))

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dove", "topic_id", "sub_topic_id", "source", "chiave", "valore_originale", "esito"])
            writer.writerows(self.rows)

    def print(self):
        labels = {"iso": "convertiti in ISO", "partial": "parziali (tolti, anno conservato)",
                  "sentinel": "sentinelle (tolti)", "invalid": "non riconosciuti (tolti)",
                  "empty": "vuoti (tolti)", "list": "liste ripulite"}
        for where in ("manifest", "documento"):
            print(f"Valori nei {'manifest' if where == 'manifest' else 'documenti indicizzati'}:")
            found = False
            for kind, label in labels.items():
                if n := self.counts.get((where, kind)):
                    print(f"    {label}: {n}")
                    found = True
            if not found:
                print("    nessuna modifica")
        invalid = [r for r in self.rows if r[6] == "invalid"]
        for row in invalid[:30]:
            print(f"    non riconosciuto: {row[1]}/{row[2]} {row[3]} {row[4]}={row[5]}")
        if len(invalid) > 30:
            print(f"    ... e altri {len(invalid) - 30}")


def ensure_backup_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS date_alignment_backup (
            item_key CHAR(64) PRIMARY KEY,
            item TEXT NOT NULL,
            original JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()


def backup(cur, item, original):
    """Solo le chiavi data. INSERT IGNORE: rilanciando resta il primo originale."""
    subset = {k: v for k, v in original.items() if DATE_KEY_RE.match(k)}
    cur.execute("INSERT IGNORE INTO date_alignment_backup (item_key, item, original) VALUES (%s, %s, %s)",
                (hashlib.sha256(item.encode()).hexdigest(), item, json.dumps(subset, ensure_ascii=False)))


def align_manifests(conn, report, topic_id, dry_run):
    root = PROCESSED_FOLDER / topic_id if topic_id else PROCESSED_FOLDER
    if not root.is_dir():
        print(f"Cartella non trovata, manifest saltati: {root}")
        return
    cur = conn.cursor() if conn else None
    for path in root.rglob("*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"    manifest illeggibile, saltato: {path} ({e})")
            continue
        metadata = manifest.get("metadati") if isinstance(manifest, dict) else None
        if not isinstance(metadata, dict):
            continue
        new_metadata, events = align_metadata(metadata)
        if not events:
            continue
        rel = path.relative_to(PROCESSED_FOLDER)
        report.add("manifest", rel.parts[0], rel.parts[1] if len(rel.parts) > 1 else "",
                   manifest.get("source", str(rel)), events)
        if dry_run:
            continue
        backup(cur, f"manifest:{rel}", metadata)
        conn.commit()
        manifest["metadati"] = new_metadata
        # Scrittura atomica: un manifest a metà renderebbe il documento non re-indicizzabile.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    if cur:
        cur.close()


def document_filter(models, source, topic_id, sub_topic_id):
    return models.Filter(must=[
        models.FieldCondition(key="source", match=models.MatchValue(value=source)),
        models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
        models.FieldCondition(key="sub_topic_id", match=models.MatchValue(value=sub_topic_id)),
    ])


def align_document(conn, qdrant, models, report, source, topic_id, sub_topic_id, dry_run):
    """Ritorna True se il documento aveva qualcosa da allineare."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, metadata FROM parent_documents "
                    "WHERE source = %s AND topic_id = %s AND sub_topic_id = %s",
                    (source, topic_id, sub_topic_id))
        parents = []
        for parent_id, raw in cur.fetchall():
            metadata = json.loads(raw) if isinstance(raw, (str, bytes)) else (raw or {})
            new_metadata, events = align_metadata(metadata)
            if events:
                parents.append((parent_id, metadata, new_metadata, events))
        if not parents:
            return False

        # I metadati del manifest sono uguali in tutti i parent: per Qdrant, che
        # si aggiorna per documento, basta unire le differenze di ciascuno.
        changed, removed = {}, set()
        for _, old, new, _ in parents:
            parent_changed, parent_removed = payload_changes(old, new)
            changed.update(parent_changed)
            removed.update(parent_removed)
        removed -= set(changed)

        report.add("documento", topic_id, sub_topic_id, source, parents[0][3])
        if dry_run:
            return True

        for parent_id, old, _, _ in parents:
            backup(cur, f"parent:{parent_id}", old)
        conn.commit()

        doc_filter = document_filter(models, source, topic_id, sub_topic_id)
        if changed:
            qdrant.set_payload(collection_name=QDRANT_COLLECTION, payload=changed, points=doc_filter, wait=True)
        if removed:
            qdrant.delete_payload(collection_name=QDRANT_COLLECTION, keys=sorted(removed), points=doc_filter, wait=True)

        for parent_id, _, new, _ in parents:
            cur.execute("UPDATE parent_documents SET metadata = %s WHERE id = %s",
                        (json.dumps(new, ensure_ascii=False), parent_id))
        conn.commit()
        return True
    finally:
        cur.close()


def main():
    ap = argparse.ArgumentParser(description="Allinea una tantum le date dei metadati al formato ISO.")
    ap.add_argument("--topic", help="solo questo archivio")
    ap.add_argument("--dry-run", action="store_true", help="conta e scrive il report senza modificare nulla")
    args = ap.parse_args()

    init_db_pool()
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile")

    report = Report(REPORT_PATH)
    qdrant, models = None, None
    if not args.dry_run:
        from qdrant_client import QdrantClient, models
        qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=120)
        ensure_backup_table(conn)

    print("Manifest in processed/...", flush=True)
    align_manifests(conn, report, args.topic, args.dry_run)

    cur = conn.cursor()
    if args.topic:
        cur.execute("SELECT DISTINCT source, topic_id, sub_topic_id FROM parent_documents WHERE topic_id = %s", (args.topic,))
    else:
        cur.execute("SELECT DISTINCT source, topic_id, sub_topic_id FROM parent_documents")
    documents = cur.fetchall()
    cur.close()
    print(f"Documenti indicizzati: {len(documents)}", flush=True)

    touched, failed = 0, []
    for index, (source, topic_id, sub_topic_id) in enumerate(documents, 1):
        try:
            if align_document(conn, qdrant, models, report, source, topic_id, sub_topic_id, args.dry_run):
                touched += 1
        except Exception as e:
            conn.rollback()
            failed.append((topic_id, sub_topic_id, source, e))
        if index % 500 == 0:
            print(f"    {index}/{len(documents)} controllati, {touched} da allineare", flush=True)
    conn.close()

    report.write()
    report.print()
    print(f"Documenti {'da allineare' if args.dry_run else 'allineati'}: {touched} su {len(documents)}. "
          f"Report: {REPORT_PATH}")
    if failed:
        print(f"Errori su {len(failed)} documenti (rilanciando lo script vengono ritentati):")
        for topic_id, sub_topic_id, source, e in failed[:20]:
            print(f"    {topic_id}/{sub_topic_id} {source}: {e}")
        sys.exit(1)
    if args.dry_run:
        print("Prova a vuoto: nulla è stato modificato.")
    elif touched:
        print("Ora conviene svuotare la cache semantica (utils/qdrant_semantic_cache_reset.py).")


if __name__ == "__main__":
    main()
