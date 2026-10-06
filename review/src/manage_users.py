"""
Gestione degli utenti revisori da riga di comando.

Esempi (dentro il container del servizio review):
    python manage_users.py add mrossi --name "Mario Rossi"            # chiede la password
    python manage_users.py add admin --role admin
    python manage_users.py password mrossi                           # cambia password, chiude le sessioni
    python manage_users.py disable mrossi                            # disattiva e chiude le sessioni
    python manage_users.py enable mrossi
    python manage_users.py list
"""

import argparse
import getpass
import sys

from werkzeug.security import generate_password_hash

from common.db_logger import get_db_connection, init_db_pool

MIN_PASSWORD_LENGTH = 10


def _ask_password() -> str:
    while True:
        pwd = getpass.getpass("Password: ")
        if len(pwd) < MIN_PASSWORD_LENGTH:
            print(f"La password deve avere almeno {MIN_PASSWORD_LENGTH} caratteri.")
            continue
        if pwd != getpass.getpass("Ripeti password: "):
            print("Le password non coincidono.")
            continue
        return pwd


def _conn():
    init_db_pool()
    conn = get_db_connection()
    if not conn:
        sys.exit("Database non raggiungibile.")
    return conn


def cmd_add(args):
    pwd = _ask_password()
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO review_users (username, display_name, role, password_hash) VALUES (%s, %s, %s, %s)",
                (args.username, args.name or args.username, args.role, generate_password_hash(pwd)),
            )
        conn.commit()
        print(f"Utente '{args.username}' creato con ruolo '{args.role}'.")
    except Exception as e:
        sys.exit(f"Errore: {e}")
    finally:
        conn.close()


def _update(username, sql, params, message):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params + (username,))
            if cur.rowcount == 0:
                sys.exit(f"Utente '{username}' non trovato.")
        conn.commit()
        print(message)
    finally:
        conn.close()


def cmd_password(args):
    pwd = _ask_password()
    _update(args.username,
            "UPDATE review_users SET password_hash = %s, token_version = token_version + 1 WHERE username = %s",
            (generate_password_hash(pwd),), "Password aggiornata; le sessioni aperte sono state chiuse.")


def cmd_disable(args):
    _update(args.username,
            "UPDATE review_users SET active = 0, token_version = token_version + 1 WHERE username = %s",
            (), f"Utente '{args.username}' disattivato.")


def cmd_enable(args):
    _update(args.username, "UPDATE review_users SET active = 1 WHERE username = %s",
            (), f"Utente '{args.username}' riattivato.")


def cmd_list(_args):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT username, display_name, role, active, last_login_at FROM review_users ORDER BY username")
            for username, name, role, active, last in cur.fetchall():
                state = "attivo" if active else "DISATTIVATO"
                print(f"{username:20} {name or '':30} {role:8} {state:12} ultimo accesso: {last or '-'}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Gestione utenti del servizio di revisione")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="crea un utente")
    p.add_argument("username")
    p.add_argument("--name", help="nome visualizzato")
    p.add_argument("--role", choices=["revisore", "admin"], default="revisore")
    p.set_defaults(func=cmd_add)

    for name, func, help_ in (("password", cmd_password, "cambia la password"),
                              ("disable", cmd_disable, "disattiva un utente"),
                              ("enable", cmd_enable, "riattiva un utente")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("username")
        p.set_defaults(func=func)

    sub.add_parser("list", help="elenca gli utenti").set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
