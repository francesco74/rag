"""
Creazione ed eliminazione dei siti della chat, solo da riga di comando (dentro il container dell'API).
Ogni archivio ha un solo sito, e l'archivio deve esistere già. Le modifiche
successive si fanno dal sito di amministrazione.

    python -m utils.manage_sites add archiviogreenlees.provincia.lucca.it --topic greenlees \
        --name "Archivio Ian Greenlees" --dir /tmp/sites/greenlees
    python -m utils.manage_sites add attiamministrativi.provincia.lucca.it --topic attiprovincia \
        --name "Atti amministrativi" --date-filter --dir /tmp/sites/attiamministrativi
    python -m utils.manage_sites delete archiviogreenlees.provincia.lucca.it
    python -m utils.manage_sites list

La cartella può contenere welcome_it.html, welcome_en.html, favicon.png e
buttons.json, una lista di {"label_it", "label_en", "url"}.
"add" non tocca mai un sito esistente: se il dominio o l'archivio sono già in uso
si ferma con un errore.
"""

import argparse
import json
import pathlib
import re
import sys

from common.db_logger import get_db_connection, init_db_pool
from common.site_home import clean_buttons, clean_html

HOSTNAME_RE = re.compile(r"^(?=.{1,255}$)[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$")


def _read(folder, name, binary=False):
    path = pathlib.Path(folder) / name if folder else None
    if not path or not path.is_file():
        return None
    return path.read_bytes() if binary else path.read_text(encoding="utf-8")


def _conn():
    init_db_pool()
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile.")
    return conn


def cmd_add(args):
    hostname = args.hostname.strip().lower()
    if not HOSTNAME_RE.match(hostname):
        sys.exit(f"Dominio non valido: '{args.hostname}' (es. archivio.ente.it, senza https:// né porta).")
    if args.dir and not pathlib.Path(args.dir).is_dir():
        sys.exit(f"Cartella non trovata: {args.dir}")
    welcome_it = clean_html(_read(args.dir, "welcome_it.html"))
    welcome_en = clean_html(_read(args.dir, "welcome_en.html"))
    try:
        buttons = clean_buttons(json.loads(_read(args.dir, "buttons.json") or "[]"))
    except ValueError as e:  # comprende il JSON non valido
        sys.exit(f"buttons.json non valido: {e}")
    icon = _read(args.dir, "favicon.png", binary=True)
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM topics WHERE topic_id = %s", (args.topic,))
        if not cur.fetchone():
            sys.exit(f"Archivio '{args.topic}' inesistente: va creato prima del sito.")
        cur.execute("SELECT topic_id FROM chat_sites WHERE hostname = %s", (hostname,))
        row = cur.fetchone()
        if row:
            sys.exit(f"Il dominio '{hostname}' è già il sito dell'archivio '{row[0]}': nessuna modifica.")
        cur.execute("SELECT hostname FROM chat_sites WHERE topic_id = %s", (args.topic,))
        row = cur.fetchone()
        if row:
            sys.exit(f"L'archivio '{args.topic}' ha già il sito '{row[0]}': nessuna modifica.")
        cur.execute(
            "INSERT INTO chat_sites (hostname, topic_id, name, allow_date_filter, welcome_it, welcome_en, home_buttons, icon) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (hostname, args.topic, args.name, args.date_filter, welcome_it, welcome_en,
             json.dumps(buttons, ensure_ascii=False), icon))
        conn.commit()
        loaded = [label for label, value in (("home IT", welcome_it), ("home EN", welcome_en), ("pulsanti", buttons), ("icona", icon))
                  if value]
        print(f"Sito '{hostname}' creato per l'archivio '{args.topic}'. Caricati: {', '.join(loaded) or 'nessun file'}.")
    finally:
        conn.close()


def cmd_delete(args):
    hostname = args.hostname.strip().lower()
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT topic_id FROM chat_sites WHERE hostname = %s", (hostname,))
        row = cur.fetchone()
        if not row:
            sys.exit(f"Sito '{hostname}' inesistente.")
        if not args.yes and input(f"Riscrivi '{hostname}' per confermare: ").strip() != hostname:
            sys.exit("Annullato: nessuna modifica.")
        cur.execute("DELETE FROM chat_sites WHERE hostname = %s", (hostname,))
        conn.commit()
        print(f"Sito '{hostname}' eliminato: l'archivio '{row[0]}' e i documenti restano.")
    finally:
        conn.close()


def cmd_list(_args):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT hostname, topic_id, name, active FROM chat_sites ORDER BY hostname")
        for hostname, topic_id, name, active in cur.fetchall():
            print(f"{hostname:45} {topic_id:20} {name}{'' if active else '  (disattivato)'}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Siti della chat.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("hostname")
    add.add_argument("--topic", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--date-filter", action="store_true")
    add.add_argument("--dir", help="cartella con welcome_it.html, welcome_en.html, favicon.png")
    add.set_defaults(fn=cmd_add)
    delete = sub.add_parser("delete")
    delete.add_argument("hostname")
    delete.add_argument("--yes", action="store_true")
    delete.set_defaults(fn=cmd_delete)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
