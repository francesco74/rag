# Sistema in locale con Docker Compose

Avvia tutto dal codice sorgente: MySQL, Redis, RabbitMQ, Qdrant, API, worker, converter, ingestione, due siti della chat, revisione e gateway.

Passi completi per la creazione di archivi, siti e utenti: sezione 12 di `README.MD`.

## Avvio

```bash
cp misc/local/env.example .env.local     # poi inserire API_LLM_KEY
docker compose --env-file .env.local up -d --build
docker compose --env-file .env.local exec api python -m utils.manage_review_users add admin --role admin
```

| Indirizzo | Cosa |
|---|---|
| http://greenlees.localhost:8081 | Chat Archivio Greenlees (archivio pubblico) |
| http://atti.localhost:8081 | Chat Atti amministrativi (archivio riservato) |
| http://localhost:8081/amministrazione/ | Amministrazione (revisione dei documenti, utenti, archivi, siti) |
| http://localhost:15672 | RabbitMQ |
| http://localhost:6333/dashboard | Qdrant |
| http://localhost:8082 | phpMyAdmin (utente `raguser` con `MYSQL_PASSWORD`, oppure `root` con `MYSQL_ROOT_PASSWORD`) |
| `127.0.0.1:3307` | MySQL |

La prima build è lunga: l'immagine dell'API contiene il reranker e le sue dipendenze.

## Dati e documenti

Il database parte con gli archivi di prova di `seed.sql` e senza documenti. Per aggiungere documenti (serve una `API_LLM_KEY` valida):

```bash
docker compose --env-file .env.local exec converter mkdir -p /data/direct/greenlees/lettere
docker compose --env-file .env.local cp mio.pdf converter:/data/direct/greenlees/lettere/mio.pdf
docker compose --env-file .env.local run --rm direct
```

Il ripiego su Google Cloud Vision richiede `GOOGLE_CREDENTIALS_DIR` in `.env.local`.

Il dizionario dei concetti si carica con `docker compose --env-file .env.local exec -w /app api python populate_concepts.py`.

## Comandi utili

```bash
docker compose --env-file .env.local logs -f worker | grep SUMMARY   # riepilogo di ogni domanda
docker compose --env-file .env.local up -d --build api worker        # dopo una modifica al backend
docker compose --env-file .env.local down                            # ferma, conserva i dati
docker compose --env-file .env.local down -v                         # ferma e CANCELLA tutti i dati
```

Lo schema SQL si applica solo alla prima creazione del volume MySQL. Per ripartire da uno schema aggiornato serve `down -v`, oppure si esegue la singola migrazione di `misc/migrations/`.
