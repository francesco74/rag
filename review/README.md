# Revisione documenti

Servizio backend (`review/`) e interfaccia web (`web/revisione/`) per i revisori che controllano i documenti indicizzati nel RAG. Servono a:

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

### 2. Backend

```bash
docker build -f review/Dockerfile -t rag-review .      # dalla radice del repository
```

Variabili d'ambiente:

| Variabile | Note |
|---|---|
| `REVIEW_SECRET_KEY` | **Obbligatoria**, almeno 32 caratteri casuali (es. `openssl rand -hex 32`). Firma i token di sessione |
| `REVIEW_ALLOWED_ORIGINS` | URL del frontend, per il CORS (es. `https://revisione.ente.it`) |
| `REVIEW_SESSION_HOURS` | Durata della sessione, default 10 |
| `REVIEW_FILE_TOKEN_MINUTES` | Validità dei link al file originale, default 30 |
| `DATA_FOLDER` | **Lo stesso volume** di converter e ingest, montato in lettura/scrittura: servono `processed/`, `watch/` e `ingestion/error/` |
| `MYSQL_*`, `QDRANT_*`, `BROKER_*` | Gli stessi degli altri servizi |

Il servizio ascolta sulla porta 5001.

### 3. Utenti

```bash
docker exec -it <container-review> python manage_users.py add mrossi --name "Mario Rossi"
docker exec -it <container-review> python manage_users.py list
docker exec -it <container-review> python manage_users.py password mrossi   # chiude le sessioni aperte
docker exec -it <container-review> python manage_users.py disable mrossi    # chiude le sessioni aperte
```

### 4. Frontend

```bash
docker build -t rag-revisione web/revisione
docker run -e REVIEW_API_URL=https://ia.ente.it/review -e PROJECT_NAME="Revisione documenti" -p 8080:80 rag-revisione
```

## Test

```bash
pip install -r review/requirements.txt
python review/tests/test_review.py
```

Il test usa SQLite al posto di MySQL, Qdrant in memoria e intercetta la pubblicazione su RabbitMQ. Copre login, ricerca, lettura del testo e dell'originale, modifica dei metadati, correzione del testo con re-indicizzazione (incluso il caso di errore), stato, storico e invalidazione delle sessioni.

## API

Tutte le chiamate, tranne `/health`, `/auth/login` e `/document/file`, richiedono `Authorization: Bearer <token>`.

| Metodo | Percorso | Descrizione |
|---|---|---|
| POST | `/auth/login` | `{username, password}` → `{token, user}` |
| GET | `/auth/me` | Utente corrente |
| GET | `/topics` | Archivi e serie |
| GET | `/documents` | `topic_id`, `sub_topic_id`, `q`, `status`, `page`, `page_size` |
| GET | `/document` | `source`, `topic_id`, `sub_topic_id` → testo, metadati, stato e link al file originale |
| GET | `/document/file` | `token` (link firmato restituito da `/document`), `download=1` |
| PUT | `/document/content` | `{source, topic_id, sub_topic_id, content, base_hash, note, mark_reviewed}` |
| PUT | `/document/metadata` | `{source, topic_id, sub_topic_id, metadata, note}`: insieme completo dei metadati |
| PUT | `/document/status` | `{source, topic_id, sub_topic_id, status, note}` |
| GET | `/document/history` | Storico del documento |
| GET | `/document/history/<id>` | Voce di storico con i valori prima e dopo |
| GET | `/users` | Elenco utenti (solo ruolo `admin`) |
