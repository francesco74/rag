"""
Ruoli e permessi del servizio di revisione.

Un utente può avere più ruoli (tabella review_user_roles): i suoi permessi
sono l'unione dei permessi dei ruoli assegnati. Gli endpoint verificano i
permessi, mai i nomi dei ruoli, e il frontend riceve l'elenco dei permessi
per decidere quali funzioni mostrare: per cambiare cosa può fare un ruolo
basta modificare ROLE_PERMISSIONS, senza toccare endpoint né frontend.

Modulo senza dipendenze: lo usa anche utils/manage_review_users.py.
"""

PERM_READ = "documenti.lettura"        # elenco, dettaglio, file originale, storico
PERM_TEXT = "documenti.testo"          # correzione del testo e re-indicizzazione
PERM_METADATA = "documenti.metadati"   # modifica dei metadati
PERM_STATUS = "documenti.stato"        # cambio dello stato di revisione
PERM_USERS = "utenti.gestione"         # amministrazione degli utenti
# Accesso a tutti gli archivi (topic) senza assegnazione. Gli altri utenti
# lavorano solo sui topic assegnati (tabella review_user_topics).
PERM_ALL_TOPICS = "topic.tutti"

ALL_PERMISSIONS = frozenset({PERM_READ, PERM_TEXT, PERM_METADATA, PERM_STATUS, PERM_USERS,
                             PERM_ALL_TOPICS})

ROLE_PERMISSIONS = {
    "lettore": frozenset({PERM_READ}),
    "correttore": frozenset({PERM_READ, PERM_TEXT}),
    "catalogatore": frozenset({PERM_READ, PERM_METADATA}),
    "validatore": frozenset({PERM_READ, PERM_STATUS}),
    "revisore": frozenset({PERM_READ, PERM_TEXT, PERM_METADATA, PERM_STATUS}),
    "admin": ALL_PERMISSIONS,
}

ROLES = tuple(ROLE_PERMISSIONS)


def permissions_for(roles) -> set[str]:
    """Unione dei permessi dei ruoli; i ruoli sconosciuti non concedono nulla."""
    perms = set()
    for role in roles:
        perms |= ROLE_PERMISSIONS.get(role, frozenset())
    return perms
