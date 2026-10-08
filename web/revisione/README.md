# Revisione documenti

Endpoint `/review` dell'API (`api/src/review_routes.py`, registrati in `app.py`) e interfaccia web (`web/revisione/`) per i revisori che controllano i documenti indicizzati nel RAG. Servono a:

- cercare i documenti per archivio, serie, stato o testo libero (oggetto, nome file, metadati);
- vedere **affiancati** il file originale (PDF, immagini) e il testo estratto dall'OCR;
- **correggere il testo**: il documento viene re-indicizzato in automatico;
- **aggiungere, modificare e rimuovere metadati**, che vengono aggiornati subito nell'archivio;
- segnare lo stato di revisione: *da revisionare*, *in revisione*, *revisionato*;
- consultare lo **storico**: chi ha modificato cosa e quando, con il confronto prima/dopo.

## Come funziona

Un documento è identificato dalla terna `(source, topic_id, sub_topic_id)`, la stessa che usa l'ingest.

| Cosa | Da dove viene letto | Come viene salvato |
|---|---|---|
| File originale | `DATA_FOLDER/processed/<topic>/<sub_topic>/<file_name>`, dove lo archivia il converter | Solo lettura |
| Testo | Il `.md` archiviato dall'ingest accanto al file originale. Se manca, il testo viene ricomposto dai `parent_documents` | Il `.md` corretto e il manifest vengono scritti in `watch/` e pubblicati sulla coda `da-indicizzare`. L'ingest, già idempotente, cancella i vecchi parent e chunk e crea i nuovi |
| Metadati | `parent_documents.metadata` del parent 0, escluse le chiavi `Header N` | Aggiornati su MySQL (in tutti i parent) e sul payload Qdrant (`set_payload` / `delete_payload`), senza ricalcolare gli embedding. Anche il manifest in `processed/` viene allineato |

Comportamenti da conoscere:

- **Re-indicizzazione.** Dopo una correzione il documento resta "in re-indicizzazione" finché l'ingest non ha rigenerato i parent. Nel frattempo i revisori vedono già il testo corretto e i metadati sono bloccati. Se l'ingest fallisce, l'errore che ha registrato in `ingestion/error/` viene mostrato nell'interfaccia.
- **Modifiche concorrenti.** Se due revisori modificano lo stesso testo, chi salva per secondo riceve un avviso (HTTP 409) invece di sovrascrivere il lavoro dell'altro.
- **Chiavi protette.** Le chiavi di sistema (`settings.protected_keys`) e le `Header N` non sono modificabili.
- **Date.** I campi il cui nome inizia per `data` devono essere nel formato `AAAA-MM-GG`, perché è quello che usano i filtri temporali della chat.

## Installazione

### 1. Database

Esegui la parte finale di `misc/mysql_schema.sql`, cioè le tabelle `review_users`, `review_status` e `review_audit`. È consigliato anche l'indice commentato su `parent_documents`.

### 2. API

Gli endpoint sono già dentro `app.py`: non c'è un nuovo servizio da pubblicare. Al pod/container dell'API servono però:

| Variabile / risorsa | Note |
|---|---|
| `AUTH_SECRET_KEY` | **Obbligatoria**, almeno 32 caratteri casuali (es. `openssl rand -hex 32`). Firma i token di sessione e i link temporanei; è pensata per essere condivisa dai futuri servizi con login (ognuno separa i propri token con un salt diverso). Va solo nell'API, mai nei frontend. Se manca, le rotte `/review` rispondono 503 e la chat continua a funzionare. Il vecchio nome `REVIEW_SECRET_KEY` è ancora accettato |
| `ALLOWED_ORIGINS` | Aggiungi l'URL del frontend di revisione, per il CORS |
| `REVIEW_SESSION_HOURS` | Durata della sessione, default 10 |
| `FILES_X_ACCEL_PREFIX` | Prefisso della location `internal` di nginx da cui inviare i documenti (in k8s `/_protected_files/`, vedi sotto). Vuoto: i file li invia l'API (sviluppo senza nginx) |
| `FILES_LOGIN_URL` | Pagina di login proposta a chi apre un documento riservato senza essere autenticato, default `/revisione/` |
| `SESSION_COOKIE_SECURE` | Default `true`: il cookie di sessione viaggia solo su HTTPS. `false` solo per lo sviluppo in http su un host diverso da localhost |
| `DATA_FOLDER` + volume | **Novità per l'API:** deve montare lo stesso volume dati di converter e ingest, in lettura/scrittura (servono `processed/`, `watch/` e `ingestion/error/`) |

Le rotte `/review/*` non usano `API_SECRET_KEY`: ogni revisore si autentica con le proprie credenziali. Allo stesso modo, la API key della chat non dà accesso alla revisione.

### 3. Utenti e ruoli

Ogni utente può avere più ruoli: i suoi permessi sono l'unione di quelli dei ruoli assegnati. Ogni endpoint verifica il permesso che gli serve (vedi la tabella [API](#api)) e l'interfaccia riceve i permessi dell'utente: le funzioni non concesse restano visibili in sola lettura (testo, metadati, stato), e chi non ha il permesso di lettura vede solo un avviso.

| Ruolo | Lettura | Testo | Metadati | Stato | Utenti |
|---|:-:|:-:|:-:|:-:|:-:|
| `lettore` | ✓ | | | | |
| `correttore` | ✓ | ✓ | | | |
| `catalogatore` | ✓ | | ✓ | | |
| `validatore` | ✓ | | | ✓ | |
| `revisore` | ✓ | ✓ | ✓ | ✓ | |
| `admin` | ✓ | ✓ | ✓ | ✓ | ✓ |

Ruoli e permessi sono definiti in `common/review_permissions.py`: per cambiare cosa può fare un ruolo, o aggiungerne uno, basta modificare quel file (il database salva solo i nomi dei ruoli).

```bash
docker exec -it <container-api> python -m utils.manage_review_users add mrossi --name "Mario Rossi"   # ruolo revisore
docker exec -it <container-api> python -m utils.manage_review_users add admin --role admin
docker exec -it <container-api> python -m utils.manage_review_users add lbianchi --role correttore --role validatore
docker exec -it <container-api> python -m utils.manage_review_users grant mrossi validatore   # aggiunge ruoli
docker exec -it <container-api> python -m utils.manage_review_users revoke mrossi validatore  # toglie ruoli
docker exec -it <container-api> python -m utils.manage_review_users roles    # ruoli disponibili e permessi
docker exec -it <container-api> python -m utils.manage_review_users list
docker exec -it <container-api> python -m utils.manage_review_users password mrossi   # chiude le sessioni aperte
docker exec -it <container-api> python -m utils.manage_review_users disable mrossi    # chiude le sessioni aperte
```

`grant` e `revoke` hanno effetto immediato, anche sulle sessioni già aperte: i ruoli vengono riletti a ogni richiesta.

### 4. Archivi assegnati

Oltre ai ruoli, ogni utente lavora solo sugli archivi (topic) che gli sono stati assegnati: vede solo quei topic e i loro documenti, e su ogni altro archivio lettura, file originale, storico e modifiche rispondono `403`. Chi ha il permesso `topic.tutti` (ruolo `admin`) vede tutti gli archivi senza assegnazione. Un utente senza archivi vede solo un avviso.

```bash
docker exec -it <container-api> python -m utils.manage_review_users topics                         # archivi esistenti
docker exec -it <container-api> python -m utils.manage_review_users add lverdi --topic attiprovincia
docker exec -it <container-api> python -m utils.manage_review_users allow mrossi attiprovincia delibere
docker exec -it <container-api> python -m utils.manage_review_users deny mrossi delibere
docker exec -it <container-api> python -m utils.manage_review_users list                           # ruoli e archivi di ogni utente
```

Anche `allow` e `deny` hanno effetto immediato.

**Database creati prima degli archivi assegnati:** eseguire una volta `misc/migrations/2026-10-07_review_user_topics.sql`. Crea la tabella `review_user_topics` e abilita gli utenti già esistenti a tutti gli archivi attuali, così nessuno perde l'accesso al rilascio; le restrizioni si applicano poi con `deny`.

**Database creati prima dei ruoli multipli:** eseguire una volta `misc/migrations/2026-10-06_review_user_roles.sql`, che sposta il ruolo di ogni utente nella nuova tabella `review_user_roles` e rimuove la colonna `review_users.role`.

### 5. Accesso ai documenti originali

Chat (attiamministrativi, archiviogreenlees) e revisione scaricano i documenti dallo stesso indirizzo dell'API:

```
/api/files/<topic>/<sub_topic>/<file>          (?download=1 per scaricarlo invece di aprirlo)
```

- **Archivio pubblico** (`topics.public_access = TRUE`): chiunque.
- **Archivio riservato** (`FALSE`, il default per i nuovi archivi): solo un utente autenticato con il permesso `documenti.lettura` e l'archivio assegnato. Al login l'API imposta un cookie di sessione (`rag_session`, HttpOnly), che il browser invia da solo anche aprendo il documento in una nuova scheda; chi apre il link senza essere autenticato vede una pagina con il link al login. Il cookie vale solo per scaricare documenti e per `/auth/me` e `/auth/logout`: le rotte `/review` richiedono comunque l'header `Authorization`.
- **Login dalla chat**: in alto nella chat c'è "Accedi" (stessi utenti della revisione). Cliccando una fonte, la chat controlla prima se il documento è consultabile: se è riservato spiega perché e propone l'accesso, poi lo apre; se l'utente non è abilitato a quell'archivio lo dice. La chat usa le rotte `/auth/login`, `/auth/me`, `/auth/logout` e non conserva il token: resta nel cookie HttpOnly.
- Si servono solo i file originali dei documenti indicizzati: testi estratti (`.md`), manifest (`.json`) e il resto di `processed/` non sono raggiungibili.
- L'API decide e basta: risponde con `X-Accel-Redirect` e il file lo invia nginx dalla location `internal` `/_protected_files/` (`k8s/nginx.yaml`), che il browser non può aprire direttamente.
- I vecchi link `/downloads/...` sono rimandati a `/api/files/...` (301).

```bash
docker exec -it <container-api> python -m utils.manage_review_users topics              # pubblico/riservato per archivio
docker exec -it <container-api> python -m utils.manage_review_users public attiprovincia
docker exec -it <container-api> python -m utils.manage_review_users private delibere
```

**Database esistenti:** eseguire una volta `misc/migrations/2026-10-07_files_access.sql`, che aggiunge `topics.public_access` lasciando pubblici gli archivi attuali (come oggi) e crea l'indice per la ricerca dei file.

Attenzione: rendere riservato un archivio protegge i **file originali**, non le risposte della chat, che continuano a usare il contenuto di tutti gli archivi interrogati.

### 6. Frontend

```bash
docker build -t rag-revisione web/revisione
docker run -e BASE_HREF=/revisione/ -e REVIEW_API_URL=/api/review -e PROJECT_NAME="Revisione documenti" -p 8080:80 rag-revisione
```

- `BASE_HREF` è il percorso sotto cui il reverse proxy pubblica l'app (es. `/revisione/`, come in `k8s/nginx.yaml`; predefinito `/`, la radice del sito). Si imposta all'avvio del container, non nella build: la stessa immagine funziona sotto qualunque percorso. Deve coincidere con la `location` del proxy: se è sbagliato il browser carica i file dell'app dalla radice del sito e si apre l'applicazione sbagliata.
- `FILES_URL` (facoltativa): indirizzo di `/files` dell'API; se assente si ricava da `REVIEW_API_URL` (`/api/review` → `/api/files`).
- `REVIEW_API_URL` può essere assoluto (`https://ia.ente.it/api/review`) o relativo al sito (`/api/review`): relativo, la stessa immagine funziona su più domini senza richieste CORS. Deve terminare con `/review`, il prefisso delle rotte di revisione.

## Test

```bash
pip install -r api/requirements.txt
python api/tests/test_review.py
```

Il test usa SQLite al posto di MySQL, Qdrant in memoria e intercetta la pubblicazione su RabbitMQ. Copre login, ricerca, lettura del testo e dell'originale, modifica dei metadati, correzione del testo con re-indicizzazione (incluso il caso di errore), stato, storico, invalidazione delle sessioni e permessi per ruolo (utenti con uno o più ruoli, revoca immediata).

## API

Tutti i percorsi hanno il prefisso `/review`. `/auth/login`, `/auth/me` e `/auth/logout` esistono anche senza prefisso (`/auth/...`, usati dalla chat e dai servizi futuri): stessi utenti, stessa sessione. Tutte le chiamate, tranne `/auth/login`, richiedono `Authorization: Bearer <token>`. Senza il permesso richiesto la risposta è `403`. Le rotte su documenti e storico rispondono `403` anche se il documento appartiene a un archivio non assegnato all'utente; `/documents` restituisce solo documenti degli archivi assegnati.

| Metodo | Percorso | Permesso | Descrizione |
|---|---|---|---|
| POST | `/auth/login` | — | `{username, password}` → `{token, user}` e cookie `rag_session` per i documenti; `user` contiene `roles`, `permissions` e `topics` (`null` = tutti gli archivi) |
| GET | `/auth/me` | — | Utente corrente, con `roles` e `permissions`; accetta anche il cookie di sessione |
| POST | `/auth/logout` | — | Accetta anche il cookie di sessione. Invalida il token lato server: chiude tutte le sessioni dell'utente e i link ai file già rilasciati |
| GET | `/topics` | `documenti.lettura` | Archivi e serie assegnati all'utente |
| GET | `/documents` | `documenti.lettura` | `topic_id`, `sub_topic_id`, `q`, `status`, `page`, `page_size` |
| GET | `/document` | `documenti.lettura` | `source`, `topic_id`, `sub_topic_id` → testo, metadati, stato e link al file originale |
| PUT | `/document/content` | `documenti.testo` | `{source, topic_id, sub_topic_id, content, base_hash, note, mark_reviewed}`; con `mark_reviewed` serve anche `documenti.stato` |
| PUT | `/document/metadata` | `documenti.metadati` | `{source, topic_id, sub_topic_id, metadata, note}`: insieme completo dei metadati |
| PUT | `/document/status` | `documenti.stato` | `{source, topic_id, sub_topic_id, status, note}` |
| GET | `/document/history` | `documenti.lettura` | Storico del documento |
| GET | `/document/history/<id>` | `documenti.lettura` | Voce di storico con i valori prima e dopo |
| GET | `/users` | `utenti.gestione` | Elenco utenti con i rispettivi ruoli e archivi |
