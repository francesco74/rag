"""
Gestione degli utenti revisori da riga di comando.

Esempi (dentro il container del servizio review):
    python -m utils.manage_review_users add mrossi --name "Mario Rossi"            # chiede la password, ruolo revisore
    python -m utils.manage_review_users add admin --role admin
    python -m utils.manage_review_users add lbianchi --role correttore --role validatore
    python -m utils.manage_review_users grant mrossi admin                         # aggiunge uno o più ruoli
    python -m utils.manage_review_users revoke mrossi admin                        # toglie uno o più ruoli
    python -m utils.manage_review_users roles                                      # ruoli disponibili e permessi
    python -m utils.manage_review_users add lverdi --topic attiprovincia            # abilitato a un solo archivio
    python -m utils.manage_review_users allow mrossi attiprovincia delibere         # abilita ad altri archivi
    python -m utils.manage_review_users deny mrossi delibere                        # toglie l'abilitazione
    python -m utils.manage_review_users topics                                     # archivi esistenti, pubblici o riservati
    python -m utils.manage_review_users public attiprovincia                       # documenti scaricabili da chiunque
    python -m utils.manage_review_users private delibere                           # documenti solo per utenti abilitati
    python -m utils.manage_review_users password mrossi                           # cambia password, chiude le sessioni
    python -m utils.manage_review_users disable mrossi                            # disattiva e chiude le sessioni
    python -m utils.manage_review_users enable mrossi
    python -m utils.manage_review_users list
"""

import argparse
import getpass
import sys

from werkzeug.security import generate_password_hash

from common.db_logger import get_db_connection, init_db_pool
from common.review_permissions import PERM_ALL_TOPICS, ROLE_PERMISSIONS, ROLES, permissions_for

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
    roles = sorted(set(args.role or ["revisore"]))
    pwd = _ask_password()
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO review_users (username, display_name, password_hash) VALUES (%s, %s, %s)",
                (args.username, args.name or args.username, generate_password_hash(pwd)),
            )
            user_id = cur.lastrowid
            cur.executemany(
                "INSERT INTO review_user_roles (user_id, role) VALUES (%s, %s)",
                [(user_id, r) for r in roles],
            )
            topics = _existing_topics(cur, args.topic or [])
            cur.executemany(
                "INSERT INTO review_user_topics (user_id, topic_id) VALUES (%s, %s)",
                [(user_id, t) for t in topics],
            )
        conn.commit()
        print(f"Utente '{args.username}' creato con ruoli: {', '.join(roles)}.")
        print(f"Archivi: {_describe_topics(roles, topics)}")
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


def _user_id(cur, username):
    cur.execute("SELECT id FROM review_users WHERE username = %s", (username,))
    row = cur.fetchone()
    if not row:
        sys.exit(f"Utente '{username}' non trovato.")
    return row[0]


def _user_roles(cur, user_id):
    cur.execute("SELECT role FROM review_user_roles WHERE user_id = %s ORDER BY role", (user_id,))
    return [r for (r,) in cur.fetchall()]


# I ruoli si rileggono a ogni richiesta: grant e revoke valgono subito,
# senza chiudere le sessioni aperte.
def cmd_grant(args):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            user_id = _user_id(cur, args.username)
            current = set(_user_roles(cur, user_id))
            cur.executemany(
                "INSERT INTO review_user_roles (user_id, role) VALUES (%s, %s)",
                [(user_id, r) for r in sorted(set(args.roles) - current)],
            )
            roles = _user_roles(cur, user_id)
        conn.commit()
        print(f"Ruoli di '{args.username}': {', '.join(roles)}.")
    finally:
        conn.close()


def cmd_revoke(args):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            user_id = _user_id(cur, args.username)
            for role in args.roles:
                cur.execute("DELETE FROM review_user_roles WHERE user_id = %s AND role = %s", (user_id, role))
            roles = _user_roles(cur, user_id)
        conn.commit()
        print(f"Ruoli di '{args.username}': {', '.join(roles) or 'nessuno (non potrà fare nulla)'}.")
    finally:
        conn.close()


def _existing_topics(cur, topics):
    """Verifica che i topic esistano (un errore di battitura non abilita nulla)."""
    topics = sorted(set(topics))
    if not topics:
        return topics
    cur.execute("SELECT topic_id FROM topics")
    known = {t for (t,) in cur.fetchall()}
    unknown = [t for t in topics if t not in known]
    if unknown:
        sys.exit(f"Archivi inesistenti: {', '.join(unknown)}. Elenco con: topics")
    return topics


def _user_topics(cur, user_id):
    cur.execute("SELECT topic_id FROM review_user_topics WHERE user_id = %s ORDER BY topic_id", (user_id,))
    return [t for (t,) in cur.fetchall()]


def _describe_topics(roles, topics):
    if PERM_ALL_TOPICS in permissions_for(roles):
        return "tutti (per ruolo)"
    return ", ".join(topics) or "nessuno (non vedrà alcun documento)"


# Anche gli archivi si rileggono a ogni richiesta: allow e deny valgono subito.
def cmd_allow(args):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            user_id = _user_id(cur, args.username)
            current = set(_user_topics(cur, user_id))
            new = [t for t in _existing_topics(cur, args.topics) if t not in current]
            cur.executemany(
                "INSERT INTO review_user_topics (user_id, topic_id) VALUES (%s, %s)",
                [(user_id, t) for t in new],
            )
            roles, topics = _user_roles(cur, user_id), _user_topics(cur, user_id)
        conn.commit()
        print(f"Archivi di '{args.username}': {_describe_topics(roles, topics)}.")
    finally:
        conn.close()


def cmd_deny(args):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            user_id = _user_id(cur, args.username)
            for topic in args.topics:
                cur.execute("DELETE FROM review_user_topics WHERE user_id = %s AND topic_id = %s",
                            (user_id, topic))
            roles, topics = _user_roles(cur, user_id), _user_topics(cur, user_id)
        conn.commit()
        print(f"Archivi di '{args.username}': {_describe_topics(roles, topics)}.")
    finally:
        conn.close()


def cmd_topics(_args):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT topic_id, description, public_access FROM topics ORDER BY topic_id")
            for topic_id, description, public in cur.fetchall():
                print(f"{topic_id:30} {'pubblico' if public else 'riservato':10} {description or ''}")
    finally:
        conn.close()


# Accesso ai documenti originali (/files/...): pubblico per chiunque, oppure
# riservato agli utenti autenticati abilitati all'archivio. Vale subito.
def _set_public(args, public):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _existing_topics(cur, [args.topic])
            cur.execute("UPDATE topics SET public_access = %s WHERE topic_id = %s", (public, args.topic))
        conn.commit()
        print(f"Archivio '{args.topic}': documenti "
              f"{'pubblici' if public else 'riservati agli utenti autenticati e abilitati'}.")
    finally:
        conn.close()


def cmd_public(args):
    _set_public(args, True)


def cmd_private(args):
    _set_public(args, False)


def cmd_roles(_args):
    for role in ROLES:
        print(f"{role:14} {', '.join(sorted(ROLE_PERMISSIONS[role]))}")


def cmd_list(_args):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT u.id, u.username, u.display_name, u.active, u.last_login_at "
                "FROM review_users u ORDER BY u.username"
            )
            for user_id, username, name, active, last in cur.fetchall():
                roles, topics = _user_roles(cur, user_id), _user_topics(cur, user_id)
                state = "attivo" if active else "DISATTIVATO"
                print(f"{username:20} {name or '':30} {state:12} {','.join(roles) or '-':30} "
                      f"ultimo accesso: {last or '-'}")
                print(f"{'':20} archivi: {_describe_topics(roles, topics)}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Gestione utenti del servizio di revisione")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="crea un utente")
    p.add_argument("username")
    p.add_argument("--name", help="nome visualizzato")
    p.add_argument("--role", choices=ROLES, action="append",
                   help="ruolo da assegnare, ripetibile (predefinito: revisore)")
    p.add_argument("--topic", action="append",
                   help="archivio (topic) su cui può lavorare, ripetibile")
    p.set_defaults(func=cmd_add)

    for name, func, help_ in (("allow", cmd_allow, "abilita un utente ad archivi (topic)"),
                              ("deny", cmd_deny, "toglie l'abilitazione ad archivi (topic)")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("username")
        p.add_argument("topics", nargs="+", metavar="topic")
        p.set_defaults(func=func)

    sub.add_parser("topics", help="elenca gli archivi (topic)").set_defaults(func=cmd_topics)

    for name, func, help_ in (("public", cmd_public, "documenti dell'archivio scaricabili da chiunque"),
                              ("private", cmd_private, "documenti dell'archivio solo per utenti abilitati")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("topic")
        p.set_defaults(func=func)

    for name, func, help_ in (("grant", cmd_grant, "aggiunge ruoli a un utente"),
                              ("revoke", cmd_revoke, "toglie ruoli a un utente")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("username")
        p.add_argument("roles", nargs="+", choices=ROLES, metavar="ruolo")
        p.set_defaults(func=func)

    sub.add_parser("roles", help="elenca i ruoli e i permessi che concedono").set_defaults(func=cmd_roles)

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
