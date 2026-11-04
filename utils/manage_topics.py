"""
Creazione ed eliminazione degli archivi (topic), solo da riga di comando dentro
il container dell'API. Il resto della configurazione e le sottocategorie si
gestiscono dal sito di amministrazione.

    python -m utils.manage_topics add mio_archivio --description "Descrizione" --prompt general
    python -m utils.manage_topics delete mio_archivio
    python -m utils.manage_topics list

L'eliminazione è rifiutata se l'archivio ha documenti o un sito, e chiede di
riscrivere l'identificativo (--yes per saltare la domanda).
"""

import argparse
import pathlib
import re
import sys

from common.db_logger import get_db_connection, init_db_pool

ARCHIVE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,99}$")
PROMPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "prompts"
RESERVED_PROMPTS = {"grader", "query_rewriter"}


def _conn():
    init_db_pool()
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile.")
    return conn


def cmd_add(args):
    if not ARCHIVE_ID_RE.match(args.topic_id):
        sys.exit(f"Identificativo non valido: '{args.topic_id}' (lettere minuscole, cifre, _ e -).")
    available = sorted(p.name for p in PROMPTS_DIR.iterdir() if p.is_file() and p.name not in RESERVED_PROMPTS)
    if args.prompt not in available:
        sys.exit(f"Prompt '{args.prompt}' inesistente. Disponibili: {', '.join(available)}.")
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM topics WHERE topic_id = %s", (args.topic_id,))
        if cur.fetchone():
            sys.exit(f"L'archivio '{args.topic_id}' esiste già: nessuna modifica.")
        cur.execute("INSERT INTO topics (topic_id, description, prompt, public_access) VALUES (%s, %s, %s, %s)",
                    (args.topic_id, args.description, args.prompt, args.public))
        conn.commit()
        print(f"Archivio '{args.topic_id}' creato.")
    finally:
        conn.close()


def cmd_delete(args):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM topics WHERE topic_id = %s", (args.topic_id,))
        if not cur.fetchone():
            sys.exit(f"Archivio '{args.topic_id}' inesistente.")
        cur.execute("SELECT 1 FROM parent_documents WHERE topic_id = %s LIMIT 1", (args.topic_id,))
        if cur.fetchone():
            sys.exit(f"L'archivio '{args.topic_id}' contiene documenti indicizzati: nessuna modifica.")
        cur.execute("SELECT hostname FROM chat_sites WHERE topic_id = %s", (args.topic_id,))
        row = cur.fetchone()
        if row:
            sys.exit(f"L'archivio '{args.topic_id}' ha il sito '{row[0]}': va eliminato prima, con utils.manage_sites delete.")
        if not args.yes and input(f"Riscrivi '{args.topic_id}' per confermare: ").strip() != args.topic_id:
            sys.exit("Annullato: nessuna modifica.")
        cur.execute("DELETE FROM review_user_topics WHERE topic_id = %s", (args.topic_id,))
        cur.execute("DELETE FROM topics WHERE topic_id = %s", (args.topic_id,))
        conn.commit()
        print(f"Archivio '{args.topic_id}' eliminato.")
    finally:
        conn.close()


def cmd_list(_args):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT topic_id, prompt, description FROM topics ORDER BY topic_id")
        for topic_id, prompt, description in cur.fetchall():
            print(f"{topic_id:30} {prompt or 'general':15} {description}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Archivi.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("topic_id")
    add.add_argument("--description", required=True)
    add.add_argument("--prompt", default="general")
    add.add_argument("--public", action="store_true", help="documenti originali scaricabili da chiunque")
    add.set_defaults(fn=cmd_add)
    delete = sub.add_parser("delete")
    delete.add_argument("topic_id")
    delete.add_argument("--yes", action="store_true")
    delete.set_defaults(fn=cmd_delete)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
