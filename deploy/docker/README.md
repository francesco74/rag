# Installazione Docker di produzione

Il rag su un server con solo Docker, **a supporto di atti-pa**:
- API esterne: conversione, sessioni di documenti dell'assistente, consultazione degli archivi;
- ingestione dei documenti degli archivi, compresa l'estrazione da Sicr@Web;
- sito di amministrazione: archivi, utenti, revisione, credenziali dei sistemi esterni.

**I siti della chat non sono installati.** Il gateway risponde solo su `/api/` e
`/amministrazione/`.

| Servizio | Cosa fa |
|---|---|
| `mysql`, `redis`, `rabbitmq`, `qdrant` | database, code, vettori; nessuna porta aperta sul server |
| `setup` | crea collection e code a ogni avvio (idempotente) e termina |
| `api` | API (gunicorn) |
| `worker`, `celery-beat` | pulizie periodiche: cache semantica, sessioni di documenti scadute |
| `converter`, `ingestion` | conversione e OCR, indicizzazione |
| `revisione`, `gateway` | sito di amministrazione e unico punto di accesso |
| `direct`, `estrazione`, `phpmyadmin` | strumenti, solo quando servono (profili) |

## Requisiti

- Docker con il plugin compose.
- Il repository sul server, per esempio in `/opt/rag`: servono `misc/mysql_schema.sql` e
  `deploy/docker/nginx.conf`.
- Le immagini pubblicate con `deploy/immagini.sh VERSIONE`. Serve una versione con le sessioni di
  documenti e con `LLM_PROVIDER=vllm`, cioè costruita dopo il 12/11/2026.
- vLLM raggiungibile dal server: LLM, embedding, reranker. Il modello dell'OCR deve leggere le
  immagini.

## Installazione

```bash
git clone https://github.com/francesco74/rag /opt/rag
cd /opt/rag/deploy/docker
cp .env.example .env && chmod 600 .env     # poi compilare i valori
docker login registry.intranet.provincia.lucca    # se il registry lo richiede
docker compose pull
docker compose up -d
docker compose logs setup                  # collection e code create
docker compose ps
```

**Prima di avviare, controlla i nomi dei modelli** in `.env`: devono essere quelli che vLLM pubblica
(`curl http://10.1.1.132:8000/v1/models`). Un nome sbagliato non ferma l'avvio: le domande
ripiegano senza riscrittura e l'OCR fallisce solo al primo PDF scansionato.

Primo utente dell'amministrazione:
```bash
docker compose exec api python -m utils.manage_review_users add admin --role admin
```

Gli archivi (topic e sottocategorie) si creano dall'amministrazione o con
`docker compose exec api python -m utils.manage_topics` (vedi `COMANDI.md`).

## Collegare atti-pa

Chiave per atti-pa. Senza `--topic` vale per conversione e sessioni di documenti, che bastano
all'assistente; con `--topic ARCHIVIO` atti-pa può anche consultare quell'archivio.
```bash
docker compose exec api python -m utils.manage_api_keys add atti-pa
```
La chiave si vede solo ora: va inserita in atti-pa, Amministrazione → *Collegamento al RAG*.

**atti-pa sullo stesso server** (installazione `deploy/docker` di atti-pa): i container di atti-pa
entrano nella rete Docker `rag`, che il rag crea, e nessuna porta si apre all'esterno. atti-pa ha
già il file `deploy/docker/docker-compose.rag.yml`. Si attiva nel suo `.env` con
`COMPOSE_FILE=docker-compose.yml:docker-compose.rag.yml`, poi `docker compose up -d`
(manuale di atti-pa, `manuale/installazione_docker.md`, "Collegamento al rag"). In atti-pa
l'indirizzo del RAG è `http://rag-gateway:8080/api`. Il rag va avviato prima: senza la rete `rag`
atti-pa non parte.

**atti-pa su un altro server**: in `.env` imposta `RAG_INDIRIZZO` con l'indirizzo del server sulla
rete interna e limita la porta `RAG_PORTA` con il firewall al solo server di atti-pa. L'indirizzo
in atti-pa è `http://INDIRIZZO:8090/api`. Il traffico è in chiaro: sulla rete interna va bene,
altrimenti serve un proxy con TLS davanti al gateway.

## Amministrazione

Con `RAG_INDIRIZZO=127.0.0.1`, che è il valore predefinito, si raggiunge con un tunnel SSH:
```bash
ssh -L 8090:127.0.0.1:8090 utente@server
```
e poi <http://localhost:8090/amministrazione/>.

## Documenti degli archivi

Caricamento dall'amministrazione, oppure a mano: i file vanno in `/data/direct/<topic>/<sub_topic>/`
dentro il volume, poi si avvia `direct`.
```bash
docker compose exec api mkdir -p /data/direct/atti/determine
docker compose cp atto.pdf api:/data/direct/atti/determine/atto.pdf
docker compose --profile strumenti run --rm direct
```

Estrazione da Sicr@Web (`saga_extract/verifica.py`, gli argomenti vanno dopo il nome del
servizio), per esempio le determine degli ultimi 10 giorni:
```bash
docker compose --profile estrazione run --rm estrazione \
    --json-filters "{\"tipo_atto\": \"determina\", \"data_da\": \"$(date -d '10 days ago' +%d/%m/%Y)\", \"data_a\": \"$(date +%d/%m/%Y)\"}" \
    --recover --sicraweb-delay 5
```
Con il cron del server (ogni notte a mezzanotte), in uno script che contiene il comando qui sopra:
`0 0 * * * cd /opt/rag/deploy/docker && ./estrai-determine.sh >> /var/log/rag-estrazione.log 2>&1`.

## Aggiornamento

```bash
cd /opt/rag && git pull                         # schema, nginx.conf, migrazioni, guida
cd deploy/docker
./backup.sh                                     # prima di ogni aggiornamento
# migrazioni nuove di misc/migrations/ (idempotenti), una alla volta:
docker compose exec -T mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' < ../../misc/migrations/FILE.sql
# VERSIONE nuova in .env, poi
docker compose pull && docker compose up -d
docker compose logs setup && docker compose ps
```
Il servizio `setup` crea le collection e le code nuove.

Al cambio di modello di embedding segui `misc/migrations/MIGRAZIONE_EMBEDDING.md`, con
`docker compose exec api python -m utils.migrate_embeddings ...` al posto di `k exec strumenti`.

## Backup e ripristino

`backup.sh` salva:
- il database, senza le tabelle temporanee delle sessioni;
- uno snapshot per ogni collection di Qdrant: `document_chunks`, `conceptual_dictionary`,
  `boilerplate_phrases`;
- il volume dei dati, senza le cartelle di lavoro.

`./backup.sh --dry-run` mostra cosa farebbe. Da pianificare con il cron:
```
30 1 * * * /opt/rag/deploy/docker/backup.sh >> /var/log/rag-backup.log 2>&1
```

Ripristino, che **sostituisce i dati attuali**:
```bash
gunzip -c rag-DATA.sql.gz | docker compose exec -T mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" rag'
docker compose exec -T mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" rag' < ../../misc/migrations/document_sessions.sql
docker compose exec -T api tar xzf - -C /data < dati-DATA.tar.gz
# ogni collection (sostituisce quella esistente):
docker compose cp qdrant-document_chunks-DATA.snapshot qdrant:/qdrant/snapshots/ripristino.snapshot
docker compose exec api python -c "import urllib.request, json; urllib.request.urlopen(urllib.request.Request(
  'http://qdrant:6333/collections/document_chunks/snapshots/recover', method='PUT',
  data=json.dumps({'location': 'file:///qdrant/snapshots/ripristino.snapshot'}).encode(),
  headers={'Content-Type': 'application/json'}))"
docker compose up -d setup                      # ricrea session_chunks e la cache, se mancano
```

## Comandi utili

```bash
docker compose logs -f api worker converter ingestion
docker compose exec api python -m utils.migrate_embeddings stato       # collection, punti, dimensione
docker compose exec api python -m utils.manage_api_keys list
docker compose --profile strumenti up -d phpmyadmin                    # 127.0.0.1:8082
docker compose --profile strumenti stop phpmyadmin
```

## Note

- **Qdrant** è fissato a `QDRANT_VERSIONE` (predefinito `v1.18.2`, come la produzione su
  Kubernetes). Va aggiornato insieme a `qdrant-client` delle immagini: versioni lontane funzionano,
  ma il client segnala l'incompatibilità.
- **Siti della chat**: per attivarli si aggiungono il servizio `chat` (immagine `web-chat`) e la
  location `/` del gateway verso di esso, come in `docker-compose.yml` della radice e in
  `misc/local/nginx-routes.conf`.
