# Sessioni di documenti (proposta)

Stato: **realizzata nel rag** (fasi 1-3), da mettere in produzione (§10). La parte di atti-pa è
da fare. Codice: `api/src/external_sessions.py`, `api/src/session_search.py`,
`common/document_sessions.py`, `common/splitting.py`; test: `api/tests/test_sessions.py`.

## 1. Scopo

L'assistente di atti-pa (la chat che sostituisce l'uso non autorizzato di ChatGPT e simili,
la "Shadow AI") deve lavorare su documenti caricati dall'utente: delibere, determine, leggi
intere. Metterli tutti nel contesto dell'LLM non funziona:

- non ci stanno (una legge lunga supera da sola il contesto di un modello locale);
- anche quando ci stanno, con molto testo l'LLM usa male la parte centrale del contesto
  ("lost in the middle").

Il rag sa già convertire, suddividere, calcolare gli embedding e cercare. Questa proposta
aggiunge **sessioni temporanee di documenti**: atti-pa carica i documenti di una
conversazione, il rag li indicizza in una collection a parte, e atti-pa ci cerca i passaggi
pertinenti o li legge in ordine, una sezione alla volta.

Confini: **il rag indicizza e cerca, non genera**. La risposta la scrive l'LLM dell'assistente
di atti-pa, con il suo prompt. I documenti di sessione non entrano mai negli archivi (topic),
nella revisione o nella cache semantica.

## 2. Concetti

- **Sessione**: contenitore temporaneo di documenti, creato da un sistema esterno (atti-pa)
  con la sua chiave (`external_api_keys`). Appartiene a quella chiave: con un'altra chiave
  non si vede (stessa risposta 404 di una sessione inesistente, come per le conversioni).
  Porta un `owner` opaco scelto dal sistema esterno (atti-pa: impronta di ente e utente), che
  il rag conserva e restituisce ma non interpreta.
- **Documento di sessione**: file caricato in una sessione; ha un `doc_id`, il nome del file e
  uno stato: `queued` → `converting` → `indexing` → `ready`, oppure `failed` con il motivo.
- **Sezione** (parent) e **pezzo** (child): come nell'ingestione degli archivi. Le sezioni
  stanno in MySQL con il loro indice d'ordine, i pezzi in Qdrant con il riferimento alla
  sezione.
- **Scadenza scorrevole**: ogni chiamata sulla sessione la sposta in avanti
  (`SESSION_TTL_MINUTES`, proposta 120). C'è anche un tetto assoluto dalla creazione
  (`SESSION_MAX_HOURS`, proposta 12). Scaduta la sessione, si cancella tutto.

## 3. Dati

### Qdrant: una sola collection `session_chunks`

**Non una collection per sessione.** Creare ed eliminare collection è un'operazione di
consenso in Qdrant (è andata in timeout durante la migrazione degli embedding del 12/11/2026)
e molte collection piccole costano più di una grande. Il modello multi-tenant consigliato da
Qdrant è una collection con il tenant nel payload:

| Campo del payload | Indice | Note |
|---|---|---|
| `session_id` | keyword, `is_tenant: true` | i punti di una sessione stanno vicini |
| `key_id` | keyword | chiave esterna proprietaria, filtro obbligatorio |
| `doc_id` | keyword | filtro sui documenti selezionati |
| `parent_index`, `chunk_index` | integer | posizione nel documento |
| `expires_at` | integer (epoch), range | pulizia |
| `content` | full-text, come in `document_chunks` | ricerca per parole chiave |

Vettori della stessa dimensione e dello stesso modello di `document_chunks`. La crea
`utils/setup_qdrant` come le altre. Con il cambio di modello di embedding **non si ricalcola**:
`migrate_embeddings` la ricrea vuota, come `semantic_cache`. Le sessioni aperte in quel momento
vanno perse, ed è accettabile.

### MySQL: `sessions`, `session_documents`, `session_parents`

- `sessions`: id, key_id, owner, created_at, expires_at.
- `session_documents`: id, session_id, file_name, status, error, tokens (stima del testo
  convertito, per l'indicatore nell'app), sections, created_at.
- `session_parents`: session_id, doc_id, parent_index, content, metadata (intestazioni
  Markdown). Serve per restituire la sezione intera intorno a un pezzo trovato e per la lettura
  in ordine.

Le tabelle nuove si creano con una migrazione SQL idempotente in `misc/migrations/`, con
backup e verifica come previsto per la produzione. **Restano fuori dai backup**
(`mysqldump --ignore-table`): sono dati temporanei per definizione.

## 4. API (in `external_sessions.py`, stessa autenticazione di `external_routes.py`)

```
POST   /external/v1/sessions                                crea: {owner} → {session_id, expires_at}
GET    /external/v1/sessions/<sid>                          stato: documenti, scadenza (la rinnova)
DELETE /external/v1/sessions/<sid>                          chiude e cancella tutto, subito
POST   /external/v1/sessions/<sid>/documents                carica un file (multipart) → 202 {doc_id}
GET    /external/v1/sessions/<sid>/documents/<did>          stato; a documento pronto: tokens, sections
DELETE /external/v1/sessions/<sid>/documents/<did>          toglie il documento dalla sessione
POST   /external/v1/sessions/<sid>/search                   {query, doc_ids, top_k} → passaggi
GET    /external/v1/sessions/<sid>/documents/<did>/sections ?from=&limit= → sezioni in ordine
```

- **search** gira nell'API (non nel worker Celery, che lavora un messaggio alla volta e farebbe
  aspettare la chat): vettori e parole chiave su `session_chunks` con il filtro su sessione,
  chiave e documenti, poi il reranker configurato. Con `vllm` il cross-encoder su server; con
  `onnx` il riordino lessicale (`lexical_reranker.py`), per non caricare il modello nell'API;
  con `lessicale` lo stesso. Se `vllm` non risponde si ripiega sul riordino lessicale. Restituisce per ogni passaggio
  `doc_id`, nome del file, `parent_index`, il testo della **sezione** (non il solo pezzo) e il
  punteggio. Niente query rewriting, grader o generazione: sono di atti-pa.
- **sections** restituisce le sezioni di un documento in ordine, a blocchi: atti-pa le usa per
  riassumere e confrontare un pezzo alla volta.
- **Formati e dimensioni** come per la conversione (`CONVERT_EXTENSIONS`,
  `CONVERT_MAX_UPLOAD_MB`).
- **Limiti per sessione**: `SESSION_MAX_DOCUMENTS` (proposta 10) e `SESSION_MAX_TOKENS`
  (proposta 500.000, circa due leggi lunghe), per non occupare vLLM e Qdrant senza misura.

## 5. Elaborazione

1. Il caricamento salva il file in `<DATA_FOLDER>/sessions/<sid>/<did>/` e pubblica sulla coda
   del converter, come la conversione esterna. Lo stato passa a `converting`.
2. A conversione finita il converter pubblica su una **coda propria**, `da-indicizzare-sessione`,
   e non su `da-indicizzare`: l'indicizzazione degli archivi non deve aspettare i documenti
   delle chat, e viceversa.
3. Un consumatore (nel servizio di ingestione, con concorrenza bassa) suddivide il testo con gli
   stessi splitter e le stesse dimensioni di `ingest.py`, calcola gli embedding, scrive sezioni e
   pezzi con `expires_at`, e mette lo stato a `ready`. Il file originale e il testo convertito si
   cancellano dal disco appena indicizzati: restano solo sezioni e pezzi, fino alla scadenza.

La logica di suddivisione si estrae da `process_single_file_async` in una funzione comune, così
archivi e sessioni non divergono.

## 6. Scadenza e pulizia

- Ogni chiamata su una sessione aggiorna `expires_at` in MySQL, che decide la scadenza.
- In Qdrant `expires_at` vale la creazione più `SESSION_MAX_HOURS` e **non si aggiorna mai**: serve
  solo a togliere i punti rimasti senza sessione (per esempio un documento indicizzato mentre la
  sessione veniva chiusa). Niente `set_payload` a ogni domanda.
- Un task di Celery Beat ogni 10 minuti cancella le sessioni scadute (punti Qdrant per
  `session_id`, righe MySQL, cartelle in `sessions/`) e i punti con `expires_at` passato. Lo stesso
  codice serve `DELETE`.
- I log registrano sessione, documento e dimensioni, **mai il testo**.

## 7. Sicurezza e dati personali

- Filtro su `session_id` **e** `key_id` sempre dal server; il client non sceglie altri filtri.
- I testi restano sul server fino alla scadenza: è una differenza rispetto alla conversione
  attuale (consegnato l'esito, non resta nulla). Va scritta nell'informativa dell'assistente e
  considerata nella DPIA dell'ente.
- Nessun backup delle tabelle e della collection di sessione.

## 8. Da verificare prima di iniziare

- Versione di `qdrant-client` allineata al server (oggi 1.20 contro 1.18.2) e creazione delle
  collection con timeout e nuovi tentativi.
- Carico di vLLM: embedding di una legge lunga mentre gira l'ingestione degli archivi.
- Dimensioni di sezioni e pezzi per i documenti delle chat: si parte con quelle degli archivi e
  si regolano con la batteria di prova di atti-pa.

## 9. Fasi

1. Tabelle, collection, API di sessione e documenti, coda e consumatore, pulizia. **Fatto.**
2. Ricerca con reranking e lettura delle sezioni in ordine. **Fatto.**
3. Test: isolamento tra chiavi, scadenza, cancellazione, limiti. **Fatto**
   (`api/tests/test_sessions.py`; converter e ingestione verificati a parte con conversione ed
   embedding simulati).

## 10. Messa in produzione

Il database è in produzione: backup e verifica prima e dopo. Le tabelle sono nuove e la
migrazione non tocca dati esistenti.

1. **Backup** di MySQL (`COMANDI.md`, Database).
2. **Tabelle**:
   ```bash
   k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' < misc/migrations/document_sessions.sql
   ```
   L'ultima istruzione elenca le tre tabelle: devono esserci tutte.
3. **Immagini nuove** di api (anche worker e celery-beat), converter e ingestion
   (`deploy/immagini.sh VERSIONE`), poi i manifest.
4. **Coda e collection** (si possono rilanciare):
   ```bash
   k exec strumenti -- python -m utils.setup_qdrant
   k exec strumenti -- env BROKER_PORT=15672 python -m utils.setup_rabbitmq
   ```
5. **Verifica**: `session_chunks` in `python -m utils.migrate_embeddings stato`; la coda
   `da-indicizzare-sessione` con un consumatore (l'ingestione) nella console di RabbitMQ; nei log
   di celery-beat il task `document-sessions-cleanup`.
6. **Prova** con una chiave esterna: creare una sessione, caricare un PDF, attendere `ready`,
   cercare, chiudere la sessione.

I valori di default (`SESSION_*`) non richiedono modifiche a `rag-config`.

