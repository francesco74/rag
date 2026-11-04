"""
Credenziali delle API per i sistemi esterni (dentro il container dell'API).

    python -m utils.manage_api_keys add gestionale --topic attiprovincia --topic greenlees
    python -m utils.manage_api_keys add conversioni --expires 2027-12-31     # solo conversione: nessun topic
    python -m utils.manage_api_keys allow gestionale delibere
    python -m utils.manage_api_keys deny gestionale greenlees
    python -m utils.manage_api_keys revoke gestionale                        # vale subito
    python -m utils.manage_api_keys enable gestionale
    python -m utils.manage_api_keys rotate gestionale                        # nuova chiave, la vecchia non vale più
    python -m utils.manage_api_keys list

La chiave si mostra una sola volta: nel database resta solo il suo hash.
"""

import argparse
import sys
from datetime import date

from common.db_logger import get_db_connection, init_db_pool
from common.utility import new_external_key


def _conn():
    init_db_pool()
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile.")
    return conn


def _key_id(cur, name):
    cur.execute("SELECT id FROM external_api_keys WHERE name = %s", (name,))
    row = cur.fetchone()
    if not row:
        sys.exit(f"Credenziale '{name}' non trovata.")
    return row[0]


def _check_topics(cur, topics):
    for topic in topics:
        cur.execute("SELECT 1 FROM topics WHERE topic_id = %s", (topic,))
        if not cur.fetchone():
            sys.exit(f"Topic '{topic}' inesistente.")


def _show_key(name, key):
    print(f"Chiave di '{name}' (si mostra solo ora, conservala in modo sicuro):\n\n    {key}\n")


def cmd_add(args):
    if args.expires:
        try:
            date.fromisoformat(args.expires)
        except ValueError:
            sys.exit("--expires deve essere una data AAAA-MM-GG.")
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM external_api_keys WHERE name = %s", (args.name,))
        if cur.fetchone():
            sys.exit(f"Credenziale '{args.name}' già esistente.")
        _check_topics(cur, args.topic or [])
        key, key_hash, prefix = new_external_key()
        cur.execute("INSERT INTO external_api_keys (name, key_hash, key_prefix, expires_at) VALUES (%s, %s, %s, %s)",
                    (args.name, key_hash, prefix, f"{args.expires} 23:59:59" if args.expires else None))
        key_id = cur.lastrowid
        for topic in args.topic or []:
            cur.execute("INSERT INTO external_api_key_topics (key_id, topic_id) VALUES (%s, %s)", (key_id, topic))
        conn.commit()
        _show_key(args.name, key)
    finally:
        conn.close()


def cmd_allow(args):
    conn = _conn()
    try:
        cur = conn.cursor()
        key_id = _key_id(cur, args.name)
        _check_topics(cur, args.topics)
        for topic in args.topics:
            cur.execute("INSERT IGNORE INTO external_api_key_topics (key_id, topic_id) VALUES (%s, %s)", (key_id, topic))
        conn.commit()
        print(f"'{args.name}' abilitata a: {', '.join(args.topics)}")
    finally:
        conn.close()


def cmd_deny(args):
    conn = _conn()
    try:
        cur = conn.cursor()
        key_id = _key_id(cur, args.name)
        for topic in args.topics:
            cur.execute("DELETE FROM external_api_key_topics WHERE key_id = %s AND topic_id = %s", (key_id, topic))
        conn.commit()
        print(f"'{args.name}' non più abilitata a: {', '.join(args.topics)}")
    finally:
        conn.close()


def _set_active(name, active):
    conn = _conn()
    try:
        cur = conn.cursor()
        key_id = _key_id(cur, name)
        cur.execute("UPDATE external_api_keys SET active = %s WHERE id = %s", (active, key_id))
        conn.commit()
        print(f"Credenziale '{name}' {'riattivata' if active else 'revocata'}.")
    finally:
        conn.close()


def cmd_rotate(args):
    conn = _conn()
    try:
        cur = conn.cursor()
        key_id = _key_id(cur, args.name)
        key, key_hash, prefix = new_external_key()
        cur.execute("UPDATE external_api_keys SET key_hash = %s, key_prefix = %s WHERE id = %s", (key_hash, prefix, key_id))
        conn.commit()
        _show_key(args.name, key)
    finally:
        conn.close()


def cmd_list(_args):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT k.name, k.key_prefix, k.active, k.expires_at, k.last_used_at, GROUP_CONCAT(t.topic_id ORDER BY t.topic_id) "
                    "FROM external_api_keys k LEFT JOIN external_api_key_topics t ON t.key_id = k.id "
                    "GROUP BY k.id ORDER BY k.name")
        for name, prefix, active, expires, used, topics in cur.fetchall():
            print(f"{name:<24} {prefix}…  {'attiva ' if active else 'revocata'}  scadenza={expires or '-'}  "
                  f"ultimo uso={used or 'mai'}  topic={topics or '-'}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Credenziali delle API per i sistemi esterni")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add", help="crea una credenziale")
    p.add_argument("name")
    p.add_argument("--topic", action="append", help="topic abilitato, ripetibile")
    p.add_argument("--expires", help="ultimo giorno di validità, AAAA-MM-GG")
    p.set_defaults(func=cmd_add)
    for name, func, help_ in (("allow", cmd_allow, "abilita a dei topic"), ("deny", cmd_deny, "toglie dei topic")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("name")
        p.add_argument("topics", nargs="+", metavar="topic")
        p.set_defaults(func=func)
    for name, func, help_ in (("revoke", lambda a: _set_active(a.name, False), "revoca, vale subito"),
                              ("enable", lambda a: _set_active(a.name, True), "riattiva"),
                              ("rotate", cmd_rotate, "genera una nuova chiave")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("name")
        p.set_defaults(func=func)
    sub.add_parser("list", help="elenca le credenziali").set_defaults(func=cmd_list)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
