"""
Ricostruisce document_ocr per i documenti indicizzati prima che il converter la
compilasse. Non trascrive nulla e non chiama modelli: rilegge i file archiviati
in processed/ con la stessa regola del converter. Va eseguito nel container del
converter, dopo la migrazione misc/migrations/document_ocr.sql.

    python rebuild_document_ocr.py                     # solo elenco, non scrive
    python rebuild_document_ocr.py --topic attiprovincia
    python rebuild_document_ocr.py --apply

Le righe già presenti non vengono mai toccate, quindi si può rilanciare. Il
risultato è una stima: un PDF finito a suo tempo nell'OCR completo per timeout
risulta con meno pagine di quelle trascritte. Esce con 1 se resta qualche
documento senza dato.
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pymupdf

from common.db_logger import get_db_connection, init_db_pool
from converter import ARCHIVE_DIR, IMG_EXTENSIONS, inspect_page, try_extract_pdf_from_pkcs7

TEXT_EXTENSIONS = (".txt", ".md", ".docx", ".odt", ".xlsx", ".ods")
FORK = mp.get_context("fork")


def _count_pdf_pages(path, pipe):
    try:
        data = path.read_bytes()
        data = try_extract_pdf_from_pkcs7(data, path.name) or data
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pipe.send((sum(inspect_page(page)[3] for page in doc), len(doc)))
    except Exception as e:
        pipe.send(e.__class__.__name__ + ": " + str(e))


def count_pdf_pages(path, timeout):
    """(pagine OCR, pagine totali), in un processo a parte perché un PDF pesante può non finire mai."""
    receiver, sender = FORK.Pipe(duplex=False)
    proc = FORK.Process(target=_count_pdf_pages, args=(path, sender))
    proc.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            raise TimeoutError(f"nessun risultato entro {timeout:g} secondi")
        try:
            result = receiver.recv()
        except EOFError:
            raise RuntimeError("analisi interrotta") from None
        if isinstance(result, str):
            raise ValueError(result)
        return result
    finally:
        proc.kill()
        proc.join()
        receiver.close()


def count_pages(path, timeout):
    """Come il converter: un'immagine passa sempre dall'OCR, un testo o un documento d'ufficio mai."""
    suffix = path.suffix.lower()
    if suffix in IMG_EXTENSIONS:
        return 1, 1
    if suffix in TEXT_EXTENSIONS:
        return 0, 0
    if suffix == ".pdf":
        return count_pdf_pages(path, timeout)
    return None


def archived_documents(root, topic=None):
    """Gli allegati dei manifest archiviati: ((source, archivio, serie), file), oppure l'errore del manifest."""
    for manifest_path in sorted((root / topic if topic else root).rglob("*.json")):
        parts = manifest_path.relative_to(root).parts
        if len(parts) < 3:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source, files = manifest["source"], manifest["files"]
            if not isinstance(source, str) or not isinstance(files, list):
                raise ValueError("source o files non validi")
        except (ValueError, KeyError, TypeError) as e:
            yield None, f"{manifest_path}: manifest non leggibile ({e})"
            continue
        for filename in files:
            # Lo stesso source che il converter assegna a ogni allegato.
            full_source = source if source.endswith(filename) else f"{source}::{filename}"
            yield (full_source, parts[0], parts[1]), manifest_path.parent / Path(filename).name


def _rows(query, params=()):
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile.")
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        conn.close()


def _save(key, ocr_pages, total_pages):
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("database non raggiungibile")
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO document_ocr (source, topic_id, sub_topic_id, ocr_pages, total_pages) "
                       "VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE total_pages = total_pages",
                       (*key, ocr_pages, total_pages))
        conn.commit()
    finally:
        conn.close()


def rebuild(root, documents, known, apply, timeout, topic=None):
    """Tratta i documenti indicizzati ancora senza dato; restituisce quanti ne ha sistemati e quanti ne restano."""
    done, seen, problems = 0, set(), []
    for key, target in archived_documents(root, topic):
        if key is None:
            problems.append(target)
            continue
        # Il manifest radice e quello dell'allegato portano allo stesso documento.
        if key in seen or key not in documents or key in known:
            continue
        seen.add(key)
        label = f"{key[1]}/{key[2]} {key[0]}"
        try:
            if not target.is_file():
                raise FileNotFoundError(f"file archiviato assente: {target}")
            pages = count_pages(target, timeout)
            if pages is None:
                raise ValueError(f"formato non gestito: {target.suffix}")
            if apply:
                _save(key, *pages)
        except Exception as e:
            problems.append(f"{label}: {e}")
            continue
        done += 1
        print(f"OCR {pages[0]}/{pages[1]}  {label}", flush=True)
    unmatched = documents - known - seen
    problems += [f"{t}/{s} {source}: nessun manifest in {root}" for source, t, s in sorted(unmatched)]
    return done, problems


def main():
    parser = argparse.ArgumentParser(description="Ricostruisce le pagine OCR dei documenti già indicizzati.")
    parser.add_argument("--apply", action="store_true", help="scrive nel database; senza, mostra soltanto il risultato")
    parser.add_argument("--topic", help="limita il lavoro a un archivio")
    parser.add_argument("--timeout", type=float, default=300, help="secondi massimi per ogni file (predefinito 300)")
    args = parser.parse_args()

    init_db_pool()
    where, params = (" WHERE topic_id = %s", (args.topic,)) if args.topic else ("", ())
    documents = set(_rows(f"SELECT DISTINCT source, topic_id, sub_topic_id FROM parent_documents{where}", params))
    known = set(_rows(f"SELECT source, topic_id, sub_topic_id FROM document_ocr{where}", params))

    done, problems = rebuild(ARCHIVE_DIR, documents, known, args.apply, args.timeout, args.topic)

    print(f"\nDocumenti indicizzati: {len(documents)}; con il dato già presente: {len(documents & known)}; "
          f"{'registrati' if args.apply else 'da registrare'}: {done}; senza dato: {len(problems)}.")
    for problem in problems:
        print(f"  {problem}")
    if not args.apply:
        print("Nessuna scrittura: per registrare rilancia con --apply.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
