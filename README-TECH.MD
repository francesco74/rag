# Specifica Tecnica ed Architetturale: Sistema RAG Enterprise-Grade per l'Ente Locale

Questo documento descrive in modo formale e approfondito l'architettura, le strategie algoritmiche, i flussi di dati, lo schema del database, la mappatura di configurazione e i requisiti di sistema del modello **Retrieval-Augmented Generation (RAG)** avanzato sviluppato per l'Ente Locale. Il sistema adotta un approccio asincrono, resiliente, guidato dagli eventi (_Event-Driven_) e strutturato su una filiera multi-stadio protetta.

## 1. Introduzione e Obiettivi dell'Architettura

I sistemi RAG tradizionali (_Naïve RAG_) soffrono di limitazioni strutturali e algoritmiche che ne precludono l'adozione in scenari della Pubblica Amministrazione regolati da rigidi vincoli di conformità amministrativa e precisione:

- **Perdita di precisione semantica:** Blocchi di testo (_chunk_) troppo ampi diluiscono il segnale semantico dei vettori di embedding, portando a mancate corrispondenze.
- **Mancanza di contesto strutturale:** I documenti vengono frammentati in piccoli paragrafi isolati. Di conseguenza, il modello generativo (LLM) ha una visione parziale: vede solo le singole tessere del puzzle, perdendo il contesto logico-normativo circostante come preamboli d'atto, visti istituzionali o articoli di legge correlati.
- **Instabilità delle API esterne ed allucinazioni:** Fluttuazioni nelle quote delle API esterne e tendenza all'allucinazione in mancanza di vincoli tassativi di grounding documentale.

Questo sistema risolve tali sfide combinando un'architettura gerarchica dei dati **Parent-Child** (Parent Document Retriever) per conservare il contesto originario , una pipeline di ingestion guidata da eventi asincroni (**RabbitMQ**), un motore di retrieval basato su **Reranker Cross-Encoder locale** in ambiente Celery protetto da deadlock, un sistema di **Analisi Concettuale** per l'espansione dei sinonimi della Pubblica Amministrazione , un algoritmo di **Ripartizione e Redistribuzione Attiva delle Quote di Contesto**, e un **Ciclo di Autocorrezione con Grader CoT** (Chain of Thought) a feedback automatico.

Attorno al motore di risposta, un sottosistema di **revisione documentale** (applicazione web `revisione` e rotte `/review` dell'API) permette al personale abilitato di correggere i testi e i metadati indicizzati, con ruoli, archivi assegnati, storico e blocco in modifica; la stessa autenticazione governa l'accesso ai **file originali** (`/files`) degli archivi riservati (capitolo 6).

## 2. Topologia dei Servizi ed Event-Driven Workflow

Il sistema adotta una topologia a microservizi debolmente accoppiati, progettati per garantire scalabilità orizzontale, tolleranza ai guasti e isolamento dei carichi computazionali.

```
                      ┌─────────────────────────────────────────┐
                      │            Estrattore SOAP              │
                      │  (estrattore.py - leggi_atto_plus SOAP) │
                      └────────────────────┬────────────────────┘
                                           │
                                           ├─────────────────────────────────────┐
                                           │ (Download file .pdf / .p7m)         │
                                           ▼                                     ▼
 ┌─────────────────────────────────────────┴────────┐   ┌────────────────────────┴────────┐
 │               Wrapper Diretti                    │   │        Busta Crittografica      │
 │     (direct.py - Scan di data/direct)            │   │  (asn1crypto.cms - Extract RAM) │
 └──────────────────────┬───────────────────────────┘   └─────────────────┬───────────────┘
                        │                                                 │
                        │ (Generazione Manifest JSON)                     │ (File decodificato)
                        └───────────────────────────┬─────────────────────┘
                                                    │
                                                    ▼ (RabbitMQ: "da-convertire")
                              ┌─────────────────────────────────────────┐
                              │        Convertitore Documentale         │
                              │    (converter.py - Estrazione & OCR)    │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼ (RabbitMQ: "da-indicizzare")
                              ┌─────────────────────────────────────────┐
                              │              Worker Ingest              │
                              │  (ingest.py - Parent-Child Splitting)   │
                              └─────────┬──────────────────────┬────────┘
                                        │                      │
                                        ▼ (MySQL)              ▼ (Qdrant)
                              ┌──────────────────┐   ┌──────────────────┐
                              │ parent_documents │   │ document_chunks  │
                              └─────────▲────────┘   └─────────▲────────┘
                                        │                      │
                                        │ (Recupero Parent)    │ (Query Vettoriale)
                                        │                      │
┌─────────────────┐                     │                      │
│   API Gateway   ├─────────────────────┼──────────────────────┼─────────────────┐
│   (app.py)      │                     │                      │                 │
└────────┬────────┘                     │                      │                 │
         │                              │                      │                 │
         ▼ (Celery Queue - redis)       │                      │                 │
┌─────────────────┐                     │                      │                 │
│   Celery RAG    ├─────────────────────┴──────────────────────┴─────────────────┘
│   (worker.py)   │ ──► [Rewriter ──► Retrieval ──► Reranker ──► Grader ──► Cache]
└─────────────────┘

```

Frontend e accesso ai documenti (stessa API, stesso database):

```
 ┌──────────────────┐   ┌──────────────────┐        ┌────────────────────────────┐
 │  Chat (Flutter)  │   │ Revisione        │        │ nginx                      │
 │  attiamministr., │   │ (Flutter Web)    │        │  /api/*  ──► API Gateway   │
 │  archiviogreenl. │   │                  │        │  /_protected_files/ (int.) │
 └───┬──────────┬───┘   └────────┬─────────┘        └──────────────▲─────────────┘
     │ /chat    │ /files, /auth  │ /review, /auth, /files           │ X-Accel-Redirect
     ▼          ▼                ▼                                  │
 ┌───────────────────────────────────────────────────────────────┐  │
 │ API Gateway (app.py): blueprint review / auth / files ────────┼──┘
 └───────┬───────────────────────────────┬───────────────────────┘
         │ correzione del testo          │ metadati: MySQL + set_payload Qdrant
         ▼ (pacchetto in watch/ + RabbitMQ: "da-indicizzare")
 ┌─────────────────┐
 │  Worker Ingest  │  rigenera parent e chunk del documento
 └─────────────────┘
```

### I Componenti Centrali:

1. **API Gateway (`app.py` - Flask)**: Espone le interfacce REST della chat protette da Bearer Token (`API_SECRET_KEY`) e registra i blueprint `review` (`/review`), `auth` (`/auth`) e `files` (`/files`), che si autenticano da sé con le credenziali personali degli utenti (`SELF_AUTHENTICATED_PREFIXES`) e non accettano il token della chat. Gestisce il bootstrap caricando le rotte e i topic dinamici, convalida le richieste di chat, raccoglie i feedback e delega in modalità non bloccante la computazione pesante a Celery, rispondendo istantaneamente con un codice `202 Accepted` per l'avvio del polling.
2. **Celery Worker (`worker.py`)**: Il vero motore di ricerca e generazione. Gestisce la trasformazione semantica della query, la ricerca ibrida parallela, il riordinamento Cross-Encoder locale, la generazione strutturata e l'auditing qualitativo della risposta (Grader).
3. **Estrattore Massivo (`estrattore.py` - CLI / CronJob)**: Il ponte di collegamento con la suite Sicr@Web dell'Ente Locale. Invia richieste SOAP ai WS di ricerca ed estrazione, gestisce l'autenticazione applicativa ed estrae i flussi binari in sicurezza.
4. **Wrapper Diretti (`direct.py` - CLI / Daemon)**: Scansiona la cartella di ingresso locale per i documenti caricati manualmente, mantiene la gerarchia delle cartelle convertendola in copie `topic_id/sub_topic_id` e pubblica gli eventi su RabbitMQ.
5. **Convertitore Documentale (`converter.py` - Worker Async)**: Elabora asincronamente i PDF e le immagini. Esegue l'estrazione testuale nativa o l'OCR avanzato tramite Gemini Vision, con fallback asincrono su Google Cloud Vision API in caso di blocco delle policy di sicurezza.
6. **Worker Ingest (`ingest.py` - Worker Async)**: Frammenta i documenti normalizzati in strutture gerarchiche Parent-Child, genera gli embedding vettoriali e popola in modo transazionale MySQL e Qdrant.
7. **Revisione (`web/revisione` - Flutter Web, `review_routes.py`)**: Applicazione di correzione di testi, metadati e stato di revisione, con autenticazione, ruoli, archivi assegnati, storico e blocco in modifica (capitolo 6).
8. **Download dei documenti (`files_routes.py` + nginx)**: Unico punto di accesso ai file originali per chat e revisione. L'API decide l'accesso (archivio pubblico o utente autenticato e abilitato) e delega l'invio del file a nginx tramite `X-Accel-Redirect`.

## 3. Pipeline d'Ingestione Documentale End-to-End

[Inbound Act] ──► SOAP laws_atto_plus ──► RAM Decrypt p7m ──► [Manifest JSON]
│
▼ (RabbitMQ: da-convertire)
[Normalizzazione .md] ◄── [Gemini Vision OCR / Google Vision] ◄── Estrazione Ibrida (converter.py)
│
▼ (RabbitMQ: da-indicizzare)
[Parent Chunking] ──► MySQL (Atomic Delete-Insert)
│
▼ (Generazione Embeddings Batch)
[Child Chunks] ────► Qdrant (Upsert + GC per Ingestion ID)

### Fase 0: Estrazione SOAP e Sbustamento Crittografico (`estrattore.py`)

Il ciclo di vita di un documento ufficiale all'interno della Pubblica Amministrazione ha inizio con l'interrogazione del protocollo:

1. **Interrogazione SOAP**: L'estrattore esegue una ricerca basata sui parametri forniti nel JSON di input e scarica i file tramite l'operazione `leggi_atto_plus` del WS di Sicr@Web.
2. **Decodifica CAdES (P7M) in RAM**: I file firmati digitalmente terminanti in `.p7m` vengono caricati direttamente in memoria. La classe `ContentInfo` di `asn1crypto.cms` esegue il parsing della busta crittografica PKCS#7 ed estrae il payload binario originale (es. PDF nativo), evitando l'appoggio su tool di terze parti a livello OS.
3. **Deduplica dei File**: Se all'interno del medesimo atto coesistono il file originale firmato (`atto.pdf.p7m`) e la sua variante già de-crittografata (`atto.pdf`), il sistema scarta automaticamente l'involucro ridondante per prevenire elaborazioni duplicate dello stesso contenuto.
4. **Scrittura Atomica e Notifica**: Per evitare conflitti di lettura, l'estrattore scrive gli allegati e il manifest JSON con estensione volatile `.tmp` all'interno della cartella di staging. Solo a scrittura ultimata viene eseguita una rinomina atomica (`rename`) eliminando il `.tmp`. Successivamente, invia un evento persistente sulla coda RabbitMQ `da-convertire`.

### Fase A: Acquisizione Diretta e Staging (`direct.py`)

I documenti caricati manualmente dagli utenti vengono elaborati tramite il modulo `direct.py`:

- Esegue una scansione ricorsiva della directory `INPUT_DIR` (`data/direct`) per rilevare file `.pdf`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`.
- Preserva la struttura delle cartelle, identificando la directory padre come `topic_id` e la sottodirectory come `sub_topic_id`.
- Muove il file nativo in `data/staging` e genera un file manifest JSON sidecar contenente un identificativo URI logico come chiave `source` (es. `direct://percorso/nomefile`).
- Notifica RabbitMQ inviando un messaggio persistente sulla coda `da-convertire`.

### Fase B: Estrazione Ibrida, OCR Avanzato e Fallback Dinamico (`converter.py`)

Il modulo `converter.py` consuma il messaggio e avvia la normalizzazione testuale tramite un approccio asincrono guidato da `AsyncLimiter` (limite di 10 chiamate/minuto per Gemini):

1. **Analisi della Densità Informativa (PDF)**:
   - Estrae il testo di ogni pagina tramite PyMuPDF (`fitz`), applicando una maschera di ritaglio basata su `CROP_OCR_LIMIT` per escludere intestazioni e piè di pagina fissi.
   - Se il numero di caratteri nativi estratti da una pagina è inferiore a 50, la pagina viene catalogata come **scansionata** o **immagine** ed inviata alla pipeline OCR. Se è superiore, viene convertita direttamente in Markdown strutturato tramite `pymupdf4llm`.
2. **Generazione OCR tramite Gemini Vision**:
   - La pagina scansionata o l'immagine nativa viene passata a Gemini Vision (`OCR_MODEL_NAME`) per una trascrizione accurata forzata in formato Markdown.
3. **Bypass dei Filtri di Sicurezza (Google Cloud Vision Fallback)**:
   - Se la chiamata API a Gemini fallisce o viene bloccata a causa delle rigide policy di filtraggio (es. flag `SAFETY`, `RECITATION` o `BLOCKLIST`), il sistema cattura l'eccezione `ValueError` o `APIError`.
   - Per evitare l'arresto della catena di elaborazione (bloccante per documenti amministrativi contenenti falsi positivi), il sistema avvia un **fallback sincrono e immediato su Google Cloud Vision API** (`document_text_detection`).
4. **Scrittura in Watch**:
   - I flussi testuali derivati vengono fusi e scritti come file `.md` abbinati a un nuovo JSON manifesto figlio nella cartella `data/watch`. Il sistema pubblica un evento persistente sulla coda `da-indicizzare`.

### Fase C: Frammentazione Gerarchica e Upsert Atomico Idempotente (`ingest.py`)

Il worker di ingestion consuma l'evento sulla coda `da-indicizzare`, applicando logiche di transazionalità e protezione della memoria RAM:

1. **Risoluzione dei Parametri dal DB**:
   - Recupera i parametri di chunking interrogando la tabella MySQL `sub_topics`. Se non presenti, applica i default di sistema (`chunk_size = 500`, `chunk_overlap = 100`, `parent_chunk_size = 1500`, `use_markdown_splitter = True`).
   - Le chiavi di sistema (`settings.protected_keys`) eventualmente presenti tra i `metadati` del manifest vengono scartate con un WARNING (`split_protected_metadata`): i campi di sistema li imposta l'ingest.
2. **Pulizia del testo OCR (`clean_ocr_text`, `common/utility.py`)**:
   - Prima dello splitting il testo viene ripulito degli spazi con cui l'OCR imita l'impaginazione: rientri fatti con `&nbsp;` (decine per una firma allineata a destra) rimossi; sequenze di 3+ spazi dentro la riga (colonne allineate) sostituite da una tabulazione; doppi spazi ridotti a uno; imbottitura delle celle delle tabelle Markdown, spazi finali, righe vuote ripetute e tag `<mark>` eliminati. Rientri di soli spazi normali (elenchi annidati) e blocchi di codice restano invariati.
   - Il file `.md` su disco **non** viene modificato: la revisione mostra al revisore l'impaginazione originale dell'OCR. La pulizia vale solo per parent, chunk ed embedding.
   - I documenti indicizzati prima dell'introduzione della pulizia si ripassano con `utils/reindex_topic.py` (capitolo 10).
3. **Splitting Gerarchico Parent-Child**:
   - **Parent Splitting**: Se abilitato, divide il documento Markdown tramite `MarkdownHeaderTextSplitter` basandosi sugli header logici (`#` fino a `####`), catturando la gerarchia nei metadati. Successivamente, utilizza `RecursiveCharacterTextSplitter` con il tokenizer di Tiktoken (`cl100k_base`) per generare blocchi ampi coerenti (Parent).
   - **Child Splitting**: Ciascun blocco Parent viene frammentato in blocchi Child più piccoli e sovrapposti (`chunk_size` e `chunk_overlap`) per consentire una ricerca semantica millimetrica.
4. **Idempotenza Transazionale MySQL**:
   - Per garantire che la re-ingestione di un documento modificato non lasci record duplicati, il worker esegue un'operazione atomica in MySQL: prima esegue una `DELETE` su `parent_documents` basata sulla chiave `source`, `topic_id` e `sub_topic_id`, poi esegue una `executemany` per inserire massivamente i nuovi Parent Documents.
5. **Generazione Embedding in Batch ed Ingestione Qdrant**:
   - I blocchi Child vengono inviati in batch asincroni alle API Google GenAI per generare vettori a 768 dimensions via `gemini-embedding-001`. Il processo è protetto da un `AsyncLimiter` impostato a 1000 richieste/minuto e gestisce gli errori `ResourceExhausted` (429) e `ServiceUnavailable` (503) tramite retry con backoff esponenziale.
6. **Prevenzione OOM (Batch Flush su Qdrant)**:
   - Al fine di proteggere la memoria RAM durante l'ingestione di file Markdown di grandissime dimensioni, i punti Child vengono assemblati e accumulati in un buffer. Non appena il buffer supera i 500 punti, viene eseguito un _flush intermedio_ verso Qdrant (`qdrant_client.upsert`), svuotando la memoria prima di procedere.
7. **Qdrant Garbage Collection**:
   - Per rimuovere i "vettori fantasma" (i vecchi frammenti Child rimasti nel Vector DB dopo che il file originale è stato aggiornato), il sistema esegue una query di cancellazione mirata (`qdrant_client.delete`), rimuovendo tutti i punti associati allo stesso `source` ma dotati di un `_ingestion_id` obsoleto (diverso dall'ID univoco di sessione appena generato).

## 4. Motore di Retrieval Avanzato & Reranking

Il processo di recupero e ordinamento dei contesti all'interno di `worker.py` supera i limiti della semplice ricerca semantica (K-Nearest Neighbors) implementando una catena di elaborazione avanzata.

### A. Trasformazione delle Query ed Espansione Semantica

1. **Riscrittura Conversazionale**: Il modello configurato in `QUERY_REWRITER_MODEL_NAME`, invocato tramite l'adapter provider-agnostico, analizza fino a `MAX_HISTORY_ITEMS` messaggi di cronologia, riscrive la domanda in forma autoconsistente (`standalone_query`), la scompone in sotto-quesiti distinti (`search_queries`, fino a `MAX_SUB_QUERIES`), estrae parole chiave lessicali (`keywords`) e — per le domande aggregative — l'elenco delle entità che la risposta deve coprire (`coverage_targets`).
2. **Lookup del Concetto (`find_concepts`)**: avviene **prima** del rewriter, che riceve i concetti trovati come vocabolario verificato. Due livelli, uniti senza doppioni:
   - **Per nome** (`get_concepts_by_name`): un concetto il cui nome compare nella domanda corrente come parola intera, senza distinzione di maiuscole (es. "Versilia", "Piana di Lucca"), è preso con certezza. Gli alias da soli **non** fanno scattare il concetto: "scuole di Viareggio" non deve allargarsi a tutti i comuni della Versilia. Il dizionario è piccolo e si legge per intero a ogni domanda.
   - **Per similarità** (`get_concepts_by_similarity`): embedding (`SEMANTIC_SIMILARITY`) del testo di lookup — la domanda corrente più gli ultimi due turni della conversazione, per agganciare i follow-up brevi — confrontato con la collezione Qdrant `conceptual_dictionary`. Vengono considerati **tutti** i concetti sopra `QDRANT_CONCEPT_THRESHOLD` (`0.80`) fino a `QDRANT_CONCEPT_MAX_HITS` (`3`). Il confronto misura il significato dell'intera frase: una domanda lunga o una cronologia estesa può restare sotto soglia anche se il nome del concetto è presente, ed è il motivo del livello per nome.
   - Gli alias dei concetti verificati vengono **iniettati nel prompt del rewriter** tramite `format_verified_concepts()`, che li presenta come vocabolario disponibile invece di sostituirli meccanicamente nella query. È il rewriter a decidere come usarli: per un concetto geografico genera una sotto-query per **ogni** comune elencato, anche oltre `MAX_SUB_QUERIES` (unica eccezione al limite), con lo stesso tema richiesto dall'utente per ciascun luogo, e riempie `coverage_targets` con i comuni; i sinonimi di un concetto tematico non restringono il tema di un singolo luogo (prompt `query_rewriter`).
   - `collect_concept_aliases()` aggiunge gli stessi alias al set di keyword, e i concetti stessi vengono passati a `retrieve_chunks(concepts=...)`: nella ricerca sintattica gli alias di un concetto valgono come **alternative** (vedi 4.B).

### B. Ricerca Ibrida Parallela e Reranking Locale via ONNX Cross-Encoder

Per massimizzare il rendimento ed evitare colli di bottiglia, la ricerca dei frammenti candidati avviene in parallelo sfruttando un pool di thread concorrenti (`ThreadPoolExecutor`):

1. **Ricerca Semantica**: Query vettoriali parallele su Qdrant (`document_chunks`) per ciascuna sotto-query, con soglia `QDRANT_SEMANTIC_THRESHOLD` (`0.60`), raggruppamento per `content_hash` (deduplica dei chunk identici) e filtro su `topic_id` / `sub_topic_id` attivi. Il client Qdrant è istanziato con `timeout=10` secondi. Il margine è necessario perché `safe_vector_search` è _fail-soft_: in caso di eccezione registra l'errore e restituisce una lista vuota senza ritentare, quindi una search andata in timeout costa l'**intera** sotto-query e con essa la copertura dell'entità corrispondente.
2. **Ricerca Sintattica a Combinazioni (`safe_keyword_search`)**: In parallelo, una ricerca full-text (`MatchText` sull'indice testuale di `content`) che riempie fino a `QDRANT_SYNTACTIC_SIZE` posti con candidati unici (deduplica per `content_hash`). I risultati confluiscono tra i candidati della query principale (indice 0). Lo scroll di Qdrant **non ordina per pertinenza** (restituisce i punti in ordine di id, cioè UUID casuali), quindi l'ordine lo danno le combinazioni:
   - **Gruppi** (`build_keyword_groups`): gli alias di ogni concetto trovato formano **un gruppo in OR** (nome + alias), perché sono alternative e non requisiti — un pezzo che cita sia "Garfagnana" sia "Barga" è tipicamente un DUP che elenca tutti i comuni, non un atto più pertinente. Ogni altra keyword è un gruppo a sé; quelle che coincidono con un alias confluiscono nel suo gruppo.
   - **Conteggi** (`plan_keyword_combinations`): con `count` esatto (pochi ms ciascuno) si contano i pezzi di ogni combinazione in AND dei gruppi, livello per livello, per i `KEYWORD_MAX_GROUPS` (6, costante) gruppi più rari: al massimo 63 conteggi, in parallelo. Una combinazione che contiene una sotto-combinazione a zero risultati non si conta.
   - **Ordine di riempimento**: prima le combinazioni con più gruppi, poi, a parità, le più rare. Una combinazione con più di `KEYWORD_MAX_HITS` (1000) pezzi è troppo generica — lo scroll ne restituirebbe un campione casuale — e si salta: "lavori" da sola no, "lavori + scuole + {versilia | viareggio | …}" sì. Se nessuna combinazione è sotto soglia si usa la più specifica.
   - **Distribuzione geografica**: in una combinazione che contiene un concetto geografico i posti si dividono fra i luoghi del concetto, perché un comune con molti atti non li prenda tutti.
   - Log: `[KEYWORD] Gruppi: …(conteggi)` e `[KEYWORD] Combinazioni usate, in ordine: …`.
3. **Recupero condizionale dei vettori**: I vettori dei chunk vengono richiesti a Qdrant (`with_vectors=True`) **solo quando `mmr_lambda < 1.0`**, cioè solo quando MMR li userà davvero. Nel profilo del tentativo 0 la diversificazione è disattivata, quindi trasferire e mediare gli embedding sarebbe lavoro interamente sprecato. Il risparmio non è marginale: la serializzazione JSON dei vettori di 30 gruppi per sotto-query è la voce dominante della latenza delle search.
4. **Pool ONNX pre-caricato (Deadlock Protection)**:
   - Il Cross-Encoder locale (`mmarco-mMiniLMv2-L12-H384-v1`) viene inizializzato **una sola volta per processo, subito dopo il fork**, dentro l'hook `worker_process_init`. Il pre-caricamento tiene i ~7 secondi di init fuori dalla latenza percepita: il costo si paga all'avvio del processo, non sulla prima query che questo si trova a servire.
   - Il caricamento deve avvenire **dopo** il fork: una sessione ONNX non è ereditabile in sicurezza attraverso `fork()`, quindi non può essere costruita nel processo padre e condivisa.
   - **Vincolo di configurazione**: il tempo di init supera i 4 secondi che Celery concede di default a un processo forkato per segnalare "UP". `worker_proc_alive_timeout` va quindi tenuto ampiamente sopra il tempo di caricamento (valore consigliato `60.0`); un valore troppo basso produce un crash loop infinito, riconoscibile nei log dalla sequenza `Timed out waiting for UP message` → SIGKILL → nuovo fork.
   - Il parallelismo interno di ONNX Runtime resta disattivato per sicurezza in ambiente Celery:

```python
     sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
     sess_options.intra_op_num_threads = num_threads
     sess_options.inter_op_num_threads = 1
```

- **Dimensionamento del pool**: `get_reranker_pool()` alloca `RERANKER_POOL_SIZE` istanze indipendenti, una per thread di rerank concorrente, e assegna a ciascuna `num_threads = (CPU_LIMIT / CELERY_CONCURRENCY) / RERANKER_POOL_SIZE`. I core si leggono da `CPU_LIMIT` e **non** da `os.cpu_count()`: quando il processo gira sotto un limite di CPU imposto dal runtime di container (quota CFS), il numero di core dichiarato dal sistema operativo resta quello della macchina ospite e sovrastima le risorse realmente disponibili. La divisione per `CELERY_CONCURRENCY` è altrettanto necessaria, perché i core del pod si spartiscono tra **tutti** i processi Celery, non solo tra le istanze di uno.
- Il sistema calcola il punteggio sigmoideo assecondando la formula del modello:
  $$P = \sigma(x) = \frac{1}{1 + e^{-x}}$$
  I chunk con probabilità inferiore a `MIN_PROB_THRESHOLD` (`0.02`) vengono scartati, salvo mantenere sempre almeno due candidati per sotto-query. Sopravvivono al massimo `RERANK_SIZE` (`25`) chunk per sotto-query. Con la **ricerca approfondita** (opzione nei Filtri della chat, campo `deep_search` di `/chat`) il limite diventa `RERANK_MAX_SIZE` (`30`); queste risposte hanno una voce separata nella cache semantica.

### C. Selezione dei Parent, Diversificazione MMR e Ripartizione del Budget di Contesto

La selezione avviene in tre stadi distinti: **quali** documenti entrano, **quanto spazio** ciascuno riceve, e **quale porzione** di ciascuno viene mostrata.

#### C.1 Quota garantita e redistribuzione con MMR

1. **Quota per sotto-query**: ciascuna sotto-query ha diritto a `PARENTS_PER_QUERY` (`4`) documenti Parent distinti, risalendo dai propri migliori chunk. Il budget nominale è quindi `PARENTS_PER_QUERY × N_query`.
2. **Slot liberi**: se una sotto-query non produce risultati (nessun documento per quel comune, o una search fallita), la sua quota resterebbe inutilizzata:
   $$\text{Slot Liberi} = (\text{parents\_per\_query} \times N_{\text{queries}}) - N_{\text{selezionati}}$$
3. **Redistribuzione**: gli slot liberi vengono riassegnati attingendo alle riserve delle altre sotto-query tramite `select_with_mmr()`. Con `mmr_lambda = 1.0` (tentativo 0) la funzione va in **bypass** e si comporta come un ranking puro per score; con `lam < 1.0` penalizza i candidati troppo simili ai Parent già selezionati, confrontando i centroidi dei loro chunk con soglia `MMR_SIMILARITY_THRESHOLD` (`0.92`). La quota garantita non viene **mai** toccata da MMR: il centroide è un vettore-proxy, e non è abbastanza affidabile da giustificare l'esclusione di un documento rilevante.

4. **Posti extra per pertinenza** (`EXTRA_PARENTS`): la quota fissa per sotto-query taglierebbe gli elenchi ("tutti gli incarichi affidati a…") anche quando il reranker giudica pertinenti molti più documenti. Si aggiungono quindi i parent non ancora scelti con probabilità del reranker ≥ `EXTRA_PARENTS_MIN_PROB` (`0.5`), dal più pertinente, fino al tetto `MAX_SUB_QUERIES × PARENTS_PER_QUERY`; ogni extra va alla sotto-query che lo ha giudicato meglio. Log: `[EXTRA]`.
5. **Rilevanza dei parent** (`CONTEXT_SCORE_SOURCE`): per le quote di contesto, MMR e l'ordine dei core si usa di default la **probabilità del reranker** del miglior chunk del parent (`rerank`), non la similarità vettoriale (`vector`): quest'ultima è compressa in un intervallo stretto e non distingue un atto pertinente da uno che ne condivide solo il lessico. Log: `[SCORE]`.

#### C.2 Ripartizione del budget di caratteri (`allocate_context_budget`)

Il budget `MAX_CONTEXT_CHARS`, al netto del costo degli header, viene ripartito tra le sotto-query e poi tra i Parent con un algoritmo di _water-filling_ a cascata:

| Fase  | Comportamento                                                                                                                                                                                         |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Ogni Parent riceve almeno `need_min` (lo spazio per i suoi frammenti "core"), dentro la quota della propria sotto-query.                                                                              |
| **B** | L'avanzo delle sotto-query povere va a chi è ancora sotto il minimo vitale.                                                                                                                           |
| **C** | Se serve ancora, si sfora `MAX_CONTEXT_CHARS` fino a `CONTEXT_OVERFLOW_RATIO`, comunque entro l'hard cap.                                                                                             |
| **D** | Deficit residuo: il Parent riceve meno del minimo e `render_parent_window()` scarta il frammento meno rilevante, con WARNING esplicito.                                                               |
| **E** | Con il budget residuo i Parent crescono verso `need_full` (margini di contesto attorno ai core), prima dentro il gruppo, poi globalmente e nel rispetto del tetto anti-monopolio `CONTEXT_CAP_RATIO`. |
| **F** | Completamento: un documento che sfora di poco viene incluso **intero** invece che finestrato.                                                                                                         |

La politica di ripartizione tra sotto-query è governata da tre parametri, che senza toccare il codice producono comportamenti diversi:

| Politica              | FLOOR | CAP  | WEIGHT | Comportamento                                                        |
| --------------------- | ----- | ---- | ------ | -------------------------------------------------------------------- |
| Rigida                | 1.00  | 1.00 | none   | Parti uguali; nessuna sotto-query può essere sacrificata.            |
| **Morbida (default)** | 0.60  | 0.40 | best   | 60% diviso equamente, 40% pesato sulla rilevanza trovata.            |
| Anti-monopolio        | 0.00  | 0.35 | best   | Tutto sullo score, ma nessuna sotto-query oltre il 35% del contesto. |

> **Nota sulla pesatura**: gli score del Cross-Encoder **non** sono nativamente confrontabili tra sotto-query diverse — il reranker valuta la coppia query-documento, quindi formulazioni diverse producono distribuzioni su scale diverse. La pesatura va letta come euristica di priorità, non come misura assoluta di importanza. Se in `rag_metrics` le sotto-query in coda restano cronicamente sotto-servite, la mossa corretta è **alzare `CONTEXT_FLOOR_RATIO`**, non ritoccare il peso.

#### C.3 Hard cap derivato

`build_context()` applica un tetto invalicabile in fase di rendering, come rete di sicurezza finale. Il valore **non si configura in caratteri assoluti**: si esprime come frazione del budget tramite `CONTEXT_HARD_CAP_RATIO` (default `0.20`), e `load_settings()` deriva

```
context_hard_cap_chars = MAX_CONTEXT_CHARS × (1 + CONTEXT_HARD_CAP_RATIO)
```

Il tetto insegue così automaticamente il budget. La scelta protegge da un guasto silenzioso e costoso: se il tetto scendesse sotto `MAX_CONTEXT_CHARS`, il rendering scarterebbe i Parent eccedenti **dopo** che retrieval e reranking sono già stati pagati, e — seguendo l'ordine round-robin — taglierebbe la coda di tutte le sotto-query insieme, producendo buchi di copertura indistinguibili da una reale assenza di documenti in archivio. Con un rapporto al posto di una costante, questa configurazione non è nemmeno rappresentabile.

Un guardrail in `load_settings()` alza il ratio fino a `CONTEXT_OVERFLOW_RATIO` se impostato più basso, perché un hard cap inferiore all'espansione già concessa in fase di allocazione riprodurrebbe lo stesso difetto per un'altra strada.

#### C.4 Finestratura del Parent (`render_parent_window`)

Il Parent non viene troncato ciecamente: `compute_core_spans()` individua nel testo integrale le posizioni dei chunk che hanno vinto il rerank (tramite offset memorizzati o, in mancanza, ricerca testuale tollerante agli spazi), i blocchi contigui vengono fusi e attorno ai core si espande il contesto fino a esaurire il budget assegnato. Il taglio arretra fino al confine indicato da `CONTEXT_SNAP_BOUNDARY` (`sentence` | `paragraph` | `none`) per non spezzare il testo a metà frase. Le porzioni omesse sono segnalate al modello con un marcatore esplicito.

#### C.5 Testo ripulito e dati di catalogo nel contesto

- **Pulizia OCR**: all'assemblaggio del contesto il testo di ogni parent passa da `clean_ocr_text` (stesse regole dell'ingestione, vedi §3 Fase C), **prima** di calcolare fabbisogni e finestre: lo spazio recuperato va ad altri documenti. Anche i chunk usati per localizzare i core vengono ripuliti allo stesso modo e i loro offset scartati (si riferiscono al testo originale): la localizzazione passa dalla ricerca testuale tollerante agli spazi. Così la pulizia vale da subito anche per i documenti indicizzati prima della sua introduzione.
- **Metadati del documento** (`CONTEXT_METADATA`): il primo passaggio incluso di ogni documento porta una riga `Metadati del documento: …` con i suoi dati di catalogo (esclusi `CONTEXT_METADATA_EXCLUDE`, chiavi di sistema, `Header N` e campi con `_` iniziale; valori accorciati a `CONTEXT_METADATA_MAX_VALUE_CHARS`). Gli altri passaggi dello stesso documento portano solo il rimando `Stesso documento di: N`. I prompt dei topic spiegano al modello che i metadati descrivono l'intero atto e servono a citarlo con precisione, mentre fatti, importi e date di eventi vanno tratti dal testo. Il costo è contabilizzato nel budget e riportato nel log `[ALLOC]` (`metadati N chars`).

## 5. Generazione con Autovalutazione & Ciclo di Autocorrezione (Self-Correction Loop)

Una volta ricomposto il contesto mediante il recupero dei Parent Documents da MySQL, il sistema avvia un loop di generazione controllata e auto-correzione a più livelli:

1. **Generazione Strutturata (JSON Mode)**:
   - Il modello configurato in `ANSWER_GENERATOR_MODEL_NAME`, invocato tramite l'adapter provider-agnostico, genera la risposta basandosi sul template di prompt specifico del topic, caricato dinamicamente da MySQL.
   - Al generatore vengono passate anche le `search_queries` e i `coverage_targets` prodotti dal rewriter: senza l'elenco esplicito delle entità, la regola sul contesto parziale sarebbe inapplicabile, perché la `standalone_query` è deliberatamente generica e il modello non avrebbe nulla da confrontare col contesto per dichiarare le assenze.
   - L'output è forzato in JSON con le chiavi `is_found`, `is_general_knowledge` e `answer`.
2. **Budget di output senza tetto artificiale**:
   - `ANSWER_MAX_TOKENS` è **opzionale**. Se assente (default), il parametro `max_output_tokens` non viene passato affatto al provider e il tetto è quello del modello _per definizione_: nessuna costante di vendor da mantenere allineata a mano, e cambiare `ANSWER_GENERATOR_MODEL_NAME` non richiede di ricordarsi nulla.
   - Il troncamento è trattato come **allarme diagnostico anziché come condizione da ritentare**: operando già al soffitto del modello non esiste un budget più ampio da richiedere, e un secondo tentativo ripagherebbe per intero un contesto da decine di migliaia di token per ottenere lo stesso esito. Un ERROR riporta `finish_reason`, token di thinking, token di output visibile, dimensione del contesto e numero di Parent, e indica la causa probabile: se il thinking ha prodotto più token dell'output visibile la leva è `ANSWER_THINKING_LEVEL`, altrimenti il problema è a monte (troppi Parent in contesto, o un prompt poco stringente sulla sintesi).
3. **Parsing JSON Multi-Livello a Forza Bruta (`safe_json_parse`)**:
   - **Livello 1 (Standard)**: decodifica diretta con `json.loads`.
   - **Livello 2 (Markdown Strip)**: rimozione dei delimitatori Markdown (` ```json ... ``` `) via regex multi-linea.
   - **Livello 2b (Truncated Repair)**: recupero degli output troncati aggiungendo le chiusure mancanti.
   - **Livello 3 (Brute Force Regex)**: ricerca di qualsiasi contenuto tra graffe `{...}` con flag `re.DOTALL`.
4. **Logica Fail-Fast**: se il modello dichiara `is_found = False`, il Grader viene saltato e il flusso passa direttamente al ciclo di retry.
5. **Auditing tramite Grader LLM**:
   - Il valutatore (`GRADER_MODEL_NAME`) riceve la query, la risposta grezza e **il `context_str` realmente passato al generatore** — non i contenuti integrali. Giudicare su un testo più ampio di quello che l'autore aveva davanti produrrebbe sia falsi negativi (retry inutili su informazioni che erano state finestrate) sia falsi positivi (il grader trova la prova che mancava al generatore e valida un'affermazione non ancorata, cioè è cieco proprio sul caso che dovrebbe intercettare).
   - Restituisce un JSON strutturato `{verdict, reason, missing_targets, note}` con budget `GRADER_MAX_TOKENS` (`256`). Il margine serve perché sui modelli che contabilizzano il thinking sullo stesso budget di output un tetto troppo stretto tronca il JSON, rendendolo non interpretabile.
   - **Il Grader boccia solo due casi**: `grounding` (afferma come tratto dai documenti qualcosa che il contesto non sostiene) e `no_answer` (nessuna informazione utile). La **copertura parziale non è un fallimento**: se per un comune non risultano documenti, dirlo è la risposta corretta, non un difetto. `missing_targets` viene raccolto per sola diagnostica e finisce in `rag_metrics`, dove sui volumi distingue un problema di ingestion (documenti mai indicizzati) da uno di soglie (presenti ma non agganciati).
   - **Fail-open**: qualunque output non interpretabile, vuoto o eccezione vale PASS, coerentemente col trattamento delle eccezioni.
   - Il flag `is_general_knowledge` **non** esonera la risposta dal controllo: il Grader la esamina comunque. Il flag si alza anche quando la risposta è ancorata ai documenti e contiene un solo esempio suggerito in chiusura, e sono proprio queste risposte miste — le più esposte al rischio di confondere ciò che il documento dice con ciò che il modello sa — a richiedere la verifica. Il Grader distingue un suggerimento dichiarato come tale (legittimo) da un fatto attribuito ai documenti ma assente dal contesto (allucinazione).
6. **Ciclo di Retry con Profili Progressivi**:
   Il loop esegue fino a `MAX_MODEL_RETRIES` (`3`) tentativi, ciascuno con un profilo di retrieval più aggressivo del precedente:
   | Tentativo | `size_mult` | `threshold_mult` | `mmr_lambda` | Strategia |
   | --- | --- | --- | --- | --- |
   | 0 | 1.0 | 1.0 | 1.0 | Recall di base, MMR disattivato. |
   | 1 | 1.5 | 0.8 | 1.0 | "Forse non ho pescato abbastanza materiale, allargo la rete." |
   | 2 | 2.0 | 0.7 | 0.8 | "Ho materiale ma è un cluster ripetitivo, forzo diversità." |
   Il messaggio iniettato nella cronologia per il nuovo giro dipende dalla `reason` restituita dalla valutazione: riformulare in modo ampio ha senso quando la ricerca ha fallito (`not_found`), non quando i documenti c'erano ma erano insufficienti su un aspetto specifico. L'ultimo fallback esclude esplicitamente i `parent_id` già passati al generatore nei tentativi precedenti, per non riproporre materiale già rivelatosi inutile.
7. **Caching Semantico e Auto-Logging delle Query Fallite**:
   - **In caso di successo**: la risposta viene salvata nella collezione `semantic_cache` di Qdrant, con chiave che include topic, sotto-topic e filtri applicati per non inquinare la cache tra contesti diversi.
   - **In caso di fallimento definitivo**: il sistema salva la sessione in `chat_feedback` con voto `0` e dicitura `"Auto-Log: Risposta non trovata o non soddisfacente"`, abilitando analisi offline e fine-tuning dei prompt o della base documentale.

## 6. Revisione Documentale, Autenticazione e Accesso ai File

Il sottosistema di revisione vive nell'API (`review_routes.py`, `files_routes.py`) e nel frontend `web/revisione` (Flutter Web). Non c'è un servizio separato: le rotte sono blueprint di `app.py`, che condividono pool MySQL, client Qdrant e configurazione.

### A. Autenticazione

- **Utenti** in `review_users`, password con hash Werkzeug. Il login (`POST /review/auth/login`, anche come `/auth/login` per la chat) emette un token firmato con `itsdangerous` (`AUTH_SECRET_KEY`, salt `review-auth`) valido `REVIEW_SESSION_HOURS` ore e contenente `token_version`. Senza `AUTH_SECRET_KEY` le rotte rispondono 503 e la chat continua a funzionare.
- **Revoca**: logout, cambio password e disattivazione incrementano `token_version`, invalidando subito tutte le sessioni dell'utente (nessuna blacklist). Ruoli, archivi e stato dell'utente si rileggono a ogni richiesta, quindi una revoca vale senza nuovo login.
- **Trasporto**: la revisione usa l'header `Authorization: Bearer`; il login imposta inoltre il cookie HttpOnly `rag_session` (SameSite=Lax, `Secure` salvo `SESSION_COOKIE_SECURE=false`), accettato **solo** da `/files` e da `/auth/me|logout`. È così che la chat apre i documenti riservati senza mai leggere il token da JavaScript; le rotte `/review` che modificano dati non accettano il cookie (nessun rischio CSRF).
- **Separazione dalla chat**: le rotte `/review`, `/auth` e `/files` non accettano `API_SECRET_KEY`, e il token di un revisore non apre `/chat`.

### B. Ruoli, permessi e archivi

`common/review_permissions.py` definisce i permessi e i ruoli che li concedono; un utente può avere più ruoli (`review_user_roles`) e i suoi permessi sono l'unione. Endpoint e frontend verificano **permessi**, mai nomi di ruolo.

| Permesso | Consente |
| --- | --- |
| `documenti.lettura` | Elenco, dettaglio, storico, file originali |
| `documenti.testo` | Correzione del testo e re-indicizzazione |
| `documenti.metadati` | Modifica dei metadati |
| `documenti.stato` | Cambio dello stato di revisione |
| `utenti.gestione` | Elenco utenti, sblocco forzato dei documenti |
| `topic.tutti` | Tutti gli archivi senza assegnazione |

Ruoli: `lettore`, `correttore`, `catalogatore`, `validatore`, `revisore` (testo + metadati + stato), `admin` (tutto). Senza `topic.tutti` un utente lavora solo sugli archivi in `review_user_topics`: tutte le rotte sul singolo documento passano da `_doc_key_from_request`, che lo verifica.

### C. Rotte principali (`/review`)

| Rotta | Funzione |
| --- | --- |
| `GET /topics`, `GET /documents` | Archivi visibili; elenco documenti (un parent 0 per documento) con ricerca su source, nome file e metadati, filtro per stato, chi ha il documento aperto (`locked_by`). |
| `GET /document` | Testo, metadati, stato, file originale, stato di re-indicizzazione, blocco, `content_hash` e `metadata_hash`. |
| `PUT /document/content` | Correzione del testo (concorrenza ottimistica su `base_hash`). |
| `PUT /document/metadata` | Sostituzione dei metadati (concorrenza ottimistica su `base_metadata_hash`). |
| `PUT /document/status` | Stato: `da_revisionare` \| `in_revisione` \| `revisionato`. |
| `GET /document/history[/<id>]` | Storico da `review_audit`, con valore prima/dopo. |
| `POST` / `DELETE /document/lock` | Presa/rinnovo e rilascio del blocco in modifica (`force=1` con `utenti.gestione`). |
| `GET /users` | Utenti con ruoli e archivi (`utenti.gestione`). |

### D. Correzione del testo e re-indicizzazione

1. Il testo mostrato viene, in ordine, dalla correzione ancora in attesa di indicizzazione (`review_audit`), dal pacchetto `.md` in `processed/` (trovato tramite il manifest con lo stesso `source`), oppure — per i dati migrati senza pacchetto — dalla concatenazione dei parent.
2. Al salvataggio il pacchetto (`.md` + manifest `.json` con i metadati attuali da MySQL) viene scritto in `watch/` nella stessa posizione relativa che ha in `processed/` (scrittura atomica, prima il testo poi il manifest) e viene pubblicato `{"json_manifest_path": …}` sulla coda `da-indicizzare`. L'ingest, già idempotente, rigenera parent e chunk e riporta il pacchetto in `processed/`. Se RabbitMQ non è raggiungibile il pacchetto viene rimosso e la richiesta fallisce con 503: nulla resta a metà.
3. Lo stato di re-indicizzazione si ricava senza tabelle aggiuntive: `pending` finché l'ultima correzione in `review_audit` è più recente del `created_at` dei parent, `error` se il manifest compare in `ingestion/error/`. Il frontend interroga periodicamente e avvisa al termine.

### E. Metadati

I metadati vivono in `parent_documents.metadata` e nel payload dei chunk su Qdrant: si aggiornano entrambi (`set_payload` / `delete_payload` filtrati per `source`, `topic_id`, `sub_topic_id`; nessun embedding da ricalcolare) e si allinea il manifest in `processed/`, così una futura re-indicizzazione non riporta indietro i valori. Le chiavi di sistema e le `Header N` sono protette sia in lettura sia in scrittura; le date devono essere ISO `AAAA-MM-GG`, perché i filtri temporali della chat le confrontano come stringhe. Durante una re-indicizzazione in corso la modifica è rifiutata (409).

### F. Concorrenza e blocco in modifica

- **Blocco** (`review_locks`): chi ha almeno un permesso di modifica prende il blocco all'apertura del documento e il frontend lo rinnova ogni terzo di `REVIEW_LOCK_TTL_SECONDS`; senza rinnovi scade da solo. Gli altri utenti vedono il documento in sola lettura con il nome di chi lo revisiona. Il rilascio avviene all'uscita dal documento, alla chiusura della pagina (`fetch` con `keepalive` su `pagehide`) e al logout (tutti i blocchi dell'utente).
- **Presa atomica**: un'unica `INSERT … ON DUPLICATE KEY UPDATE`, che sfrutta l'ordine di valutazione delle assegnazioni di MySQL (`user_id` cambia per penultimo, `expires_at` per ultimo e solo se il blocco è ora del richiedente), senza `DELETE` preventiva dei blocchi scaduti, che con due richieste contemporanee produrrebbe un deadlock da gap lock. Gli orari sono quelli del database (`NOW()`), non dei pod.
- **Salvataggi**: testo, metadati e stato chiamano `_require_lock`, che prende il blocco se libero e mantiene la riga bloccata (`SELECT … FOR UPDATE`) fino al commit: due salvataggi simultanei sullo stesso documento si mettono in fila. Senza blocco la risposta è `409` con `reason: "locked"` e il titolare; con dati cambiati nel frattempo è `409` con `reason: "modified"`.

### G. Download dei documenti (`/files`)

`GET /files/<topic>/<sub_topic>/<path:file_name>[?download=1]` è l'unico accesso ai file originali, usato da chat e revisione:

- **Accesso**: archivio con `topics.public_access = TRUE` → chiunque; altrimenti utente autenticato (cookie o header) con `documenti.lettura` e l'archivio assegnato. L'autorizzazione precede la ricerca del documento, così un anonimo non può scoprire quali file esistano in un archivio riservato. Ai browser gli errori arrivano come pagina HTML (con il link di login `FILES_LOGIN_URL` sul 401), ai client come JSON.
- **Solo originali indicizzati**: il nome deve corrispondere a un documento in `parent_documents`; testi estratti e manifest non sono mai raggiungibili, e i percorsi fuori da `processed/` sono rifiutati.
- **Invio**: con `FILES_X_ACCEL_PREFIX` l'API risponde con `X-Accel-Redirect: <prefisso><percorso URL-encoded>` e il file lo invia nginx da una location `internal` (`/_protected_files/` → `alias /data/processed/`), senza occupare i worker Python; senza prefisso lo invia l'API (sviluppo). Nomi con spazi, accenti e caratteri come `%`, `+`, `#`, `?` sono gestiti lungo tutto il percorso (frontend → nginx → Flask → `X-Accel-Redirect`). I vecchi link `/downloads/...` delle chat sono rediretti a `/api/files/...`.
- **Cache**: `public, max-age=3600` per gli archivi pubblici, `private, no-store` per i riservati.

> **Limite**: la riservatezza di un archivio protegge l'apertura del file originale, non il contenuto delle risposte. La chat di un topic risponde usando il testo di tutti i suoi documenti; i documenti che non devono comparire nelle risposte vanno indicizzati in un topic (e una chat) separato.

### H. Frontend `revisione`

Flutter Web con configurazione a runtime (`docker-entrypoint.sh` → `env-config.js`: `REVIEW_API_URL`, `FILES_URL`, `BASE_HREF`, `PROJECT_NAME`), così la stessa immagine funziona sotto qualunque percorso e dominio. All'avvio verifica il `/health` dell'API (ricavato da `REVIEW_API_URL`: `/api/review` → `/api/health`) e mostra la pagina di manutenzione se non risponde. L'editor del testo lavora sui blocchi Markdown nella vista formattata (`MdDocument`, `RichBlockController`) riscrivendo solo il blocco modificato, con il sorgente disponibile su richiesta; il file originale è mostrato in un iframe accanto al testo. Installazione e variabili: `web/revisione/README.md`.

## 7. Schema del Database Relazionale (MySQL)

Il database adotta il motore di archiviazione **InnoDB** con set di caratteri **utf8mb4** per la piena conformità con i caratteri speciali e i metadati complessi estratti.

```

   ┌──────────────┐
   │    topics    │
   └──────┬───────┘
          │ 1
          │
          │ 1..N (Constraint FK CASCADE ON DELETE/UPDATE)
   ┌──────▼───────┐
   │  sub_topics  │
   └──────────────┘


┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  parent_docs     │      │  chat_feedback   │      │   system_logs    │      │   rag_metrics    │
└──────────────────┘      └──────────────────┘      └──────────────────┘      └──────────────────┘

   ┌──────────────┐  1..N  ┌───────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ review_users ├───────►│ review_user_roles │   │  review_status   │   │   review_audit   │
   └──────┬───┬───┘        └───────────────────┘   └──────────────────┘   └──────────────────┘
          │   │     1..N   ┌───────────────────┐   ┌──────────────────┐
          │   └───────────►│ review_user_topics│   │   review_locks   │◄── FK user_id
          │                └───────────────────┘   └──────────────────┘
          └──────────────────────────────────────────────────┘
```

### Dettaglio Tabelle, Vincoli e Indici composti:

1. **`topics`**: Mappa i macro-argomenti e associa i prompt di sistema personalizzati.
   - `topic_id` (VARCHAR(100) PRIMARY KEY)
   - `description` (TEXT, NOT NULL)
   - `aliases` (TEXT, NULL)
   - `prompt` (VARCHAR(100), NULL)
   - `public_access` (BOOLEAN, NOT NULL, DEFAULT FALSE) - _File originali scaricabili da chiunque (TRUE) o solo da utenti autenticati e abilitati all'archivio (FALSE), vedi §6.G_.
2. **`sub_topics`**: Configura a runtime i parametri di splitting per ciascun sotto-argomento.
   - `id` (INT AUTO_INCREMENT PRIMARY KEY)
   - `topic_id` (VARCHAR(100) NOT NULL)
   - `sub_topic_id` (VARCHAR(100) NOT NULL)
   - `description` (VARCHAR(255) NOT NULL)
   - `chunk_size` (INT, NULL)
   - `chunk_overlap` (INT, DEFAULT 50)
   - `parent_chunk_size` (INT, DEFAULT 1500)
   - `use_markdown_splitter` (TINYINT(1), DEFAULT 1)
   - _Vincoli_: `UNIQUE KEY unique_sub_topic (topic_id, sub_topic_id)`
   - _Chiave Esterna_: `CONSTRAINT fk_sub_topics_topic_id FOREIGN KEY (topic_id) REFERENCES topics (topic_id) ON DELETE CASCADE ON UPDATE CASCADE`
3. **`parent_documents`**: Contiene il testo strutturato esteso (Parent) dei documenti.
   - `id` (VARCHAR(36) PRIMARY KEY)
   - `topic_id` (VARCHAR(255) NOT NULL)
   - `sub_topic_id` (VARCHAR(255) NOT NULL)
   - `source` (VARCHAR(255) NOT NULL)
   - `file_name` (VARCHAR(255), NULL)
   - `parent_index` (INT, NULL)
   - `content` (LONGTEXT, NOT NULL)
   - `metadata` (JSON, NULL)
   - _Indici_: `INDEX idx_search (source, topic_id, sub_topic_id)` per la rimozione atomica e le query di retrieval.
4. **`chat_feedback`**: Registro per la raccolta dei feedback degli utenti e il tracciamento dei fallimenti automatici del Grader.
   - `id` (INT AUTO_INCREMENT PRIMARY KEY)
   - `topic_id` (VARCHAR(255), NULL)
   - `user_query` (TEXT, NULL)
   - `ai_response` (TEXT, NULL)
   - `rating` (INT, NULL) - _Il rating è impostato a 0 in caso di fallimento del Grader_.
   - `chat_history` (JSON, NULL)
   - `comment` (TEXT, NULL)
   - `reviewed` (TINYINT(1), DEFAULT 0)
5. **`system_logs`**: Auditing centralizzato delle anomalie e degli stacktrace, con l'identificativo dell'istanza che li ha generati.
   - `log_id` (BIGINT AUTO_INCREMENT PRIMARY KEY)
   - `log_level` (VARCHAR(15), NOT NULL)
   - `message` (TEXT, NOT NULL)
   - `file_name` (VARCHAR(255), NULL)
   - `line_no` (INT, NULL)
   - `pod_name` (VARCHAR(255), DEFAULT NULL)
   - _Indici_: `INDEX idx_log_level (log_level)`, `INDEX idx_created_at (created_at)`.
6. **`rag_metrics`**: Metriche per richiesta della chat.
   - `task_id`, `topic_id`, `query_preview`, `cache_hit`, `is_satisfactory`, `total_attempts`, `duration_ms`, `created_at`
   - `attempts_detail` (JSON): per tentativo, query riscritta, parametri di retrieval effettivi e conteggi di candidati e parent (vuoto per i cache hit). In `misc/mysql_schema.sql` sono raccolte le query di analisi tipiche.
7. **`review_users`**: Utenti di revisione e chat.
   - `id`, `username` (UNIQUE), `display_name`, `password_hash`, `active`, `created_at`, `last_login_at`
   - `token_version` (INT): incrementato a logout, cambio password e disattivazione; invalida i token già emessi.
8. **`review_user_roles`** (`user_id`, `role`, PK composta) e **`review_user_topics`** (`user_id`, `topic_id`, PK composta): ruoli e archivi assegnati, con FK `ON DELETE CASCADE` su `review_users`. I ruoli sono `VARCHAR` e non `ENUM`: un ruolo nuovo non richiede di modificare lo schema.
9. **`review_status`**: Stato di revisione per documento.
   - PK (`source`, `topic_id`, `sub_topic_id`); `status` ENUM(`da_revisionare`, `in_revisione`, `revisionato`); `note`, `updated_by`, `updated_at`. L'assenza di riga vale `da_revisionare`.
10. **`review_audit`**: Storico di ogni modifica.
    - `action` ENUM(`content`, `metadata`, `status`), utente, documento, `old_value` / `new_value` (testo completo, JSON dei metadati o stato), `note`, `details` (JSON).
    - _Indici_: `idx_doc (source, topic_id, sub_topic_id, created_at)`, `idx_user (username, created_at)`. È anche la fonte dello stato di re-indicizzazione (§6.D).
11. **`review_locks`**: Blocco in modifica.
    - PK (`source`, `topic_id`, `sub_topic_id`); `user_id` (FK `ON DELETE CASCADE`), `username`, `display_name`, `acquired_at`, `expires_at`. Contano solo le righe con `expires_at > NOW()`: quelle scadute restano finché qualcuno non riprende il documento.

Le modifiche allo schema dei database già installati sono in `misc/migrations/` (es. `review_locks.sql`, `parent_metadata_cleanup.sql`).

## 8. Mappatura Dettagliata delle Variabili di Configurazione (`config.py`)

La configurazione centralizzata raccoglie e convalida le variabili d'ambiente nella dataclass immutabile `Settings` (94 campi).

> ### ⚠ Precedenza delle fonti di configurazione
>
> `config.py` invoca `load_dotenv()` come prima istruzione. Le fonti attive, in ordine di precedenza decrescente, sono:
>
> ```
> Ambiente del processo  >  file .env  >  default in config.py
> ```
>
> L'ambiente del processo è popolato in modo diverso a seconda di come il sistema viene eseguito: variabili d'ambiente della shell in installazione diretta, direttive `environment`/`env_file` con Docker Compose, `docker service create --env` con Swarm, oggetti ConfigMap e Secret con Kubernetes. In tutti i casi vince sul `.env`.
>
> `load_dotenv()` **non sovrascrive** le variabili già presenti in `os.environ`, quindi un `.env` presente nell'immagine o nella working directory resta **invisibile** finché la stessa chiave è definita nell'ambiente — e riemerge nel momento in cui la si rimuove. Togliere una chiave dalla configurazione dell'orchestratore **non** fa quindi ricadere il sistema sul default del codice: scopre il valore sottostante.
>
> In un'immagine di produzione il `.env` non dovrebbe esistere: va escluso via `.dockerignore`, che Docker legge **solo** nella radice del contesto di build e non accanto al Dockerfile. Se contiene credenziali (`API_LLM_KEY`, `MYSQL_PASSWORD`, `WS_PASSWORD`) queste finiscono nei layer dell'immagine, leggibili da chiunque abbia accesso in pull al registry e non rimovibili dallo storico con una build successiva.
>
> Va infine ricordato che le variabili sono lette **all'avvio del processo**: modificare la configurazione non ha effetto finché il servizio non viene riavviato, con il comando proprio dell'ambiente in uso (`kubectl rollout restart`, `docker compose up -d`, `docker service update --force`, riavvio del servizio di sistema).

### A. Infrastruttura e Connettività

| Proprietà                             | Variabile d'Ambiente                               | Tipo          | Default                                             | Impatto                                                                                                                                                                         |
| ------------------------------------- | -------------------------------------------------- | :------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `http_timeout_seconds`                | `HTTP_TIMEOUT_SECONDS`                             | `int`         | `300`                                               | Soglia di blocco delle chiamate HTTP esterne verso Sicr@Web e le API LLM.                                                                                                       |
| `data_folder`                         | `DATA_FOLDER`                                      | `Path`        | dir di `config.py`                                  | Radice locale per `staging`, `watch`, `archive`, `error`.                                                                                                                       |
| `log_level`                           | `LOG_LEVEL`                                        | `str`         | `"INFO"`                                            | Verbosità del logger centralizzato. **`DEBUG` in produzione** genera un volume elevato (traffico httpcore, dump del contesto e delle risposte grezze).                          |
| `hostname`                            | `HOSTNAME`                                         | `str`         | `socket.gethostname()`                              | Identificativo dell'istanza nei log e nella tabella `system_logs`, per attribuire l'errore al container o alla macchina corretta in un'esecuzione distribuita.                  |
| `broker_host` / `broker_port`         | `BROKER_HOST` / `BROKER_PORT`                      | `str` / `int` | `"rabbitmq-service.rag.svc.cluster.local"` / `5672` | Endpoint RabbitMQ per la pipeline di **ingestion**. Non usato da `worker.py`, che costruisce il broker Celery da Redis.                                                         |
| `broker_username` / `broker_password` | `BROKER_USERNAME` / `BROKER_PASSWORD`              | `str`         | `"guest"` / `"guest"`                               | Credenziali RabbitMQ.                                                                                                                                                           |
| `broker_max_attempts`                 | `BROKER_MAX_ATTEMPTS`                              | `int`         | `30`                                                | Tentativi di connessione all'avvio prima di considerare fallita la partenza del servizio. Il margine serve quando i servizi si avviano insieme e il broker non è ancora pronto. |
| `redis_host` / `redis_port`           | `REDIS_HOST` / `REDIS_PORT`                        | `str` / `int` | `"localhost"` / `6379`                              | Broker **e** backend dei risultati per Celery.                                                                                                                                  |
| `qdrant_host` / `qdrant_port`         | `QDRANT_HOST` / `QDRANT_PORT`                      | `str` / `int` | `"localhost"` / `6333`                              | Endpoint del database vettoriale. Il client è istanziato con `timeout=10` (hardcoded in `worker.py`).                                                                           |
| `db_host` / `db_port`                 | `MYSQL_SERVICE_HOST` / `MYSQL_SERVICE_PORT`        | `str` / `int` | `"localhost"` / `3306`                              | Endpoint MySQL.                                                                                                                                                                 |
| `db_user` / `db_pass` / `db_name`     | `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | `str`         | `"raguser"` / `""` / `"rag_db"`                     | Credenziali e nome del database.                                                                                                                                                |
| `allowed_origins`                     | `ALLOWED_ORIGINS`                                  | `list[str]`   | `["*"]`                                             | Origini CORS ammesse dall'API Gateway.                                                                                                                                          |
| `api_secret_key`                      | `API_SECRET_KEY`                                   | `str`         | `"default_key"`                                     | Bearer Token a protezione degli endpoint Flask.                                                                                                                                 |
| `api_llm_key`                         | `API_LLM_KEY`                                      | `str`         | `"default_key"`                                     | Chiave per Google GenAI (generazione ed embedding).                                                                                                                             |

### B. Integrazione Sicr@Web (SOAP)

| Proprietà                                           | Variabile d'Ambiente                                                       | Tipo                  | Default                     | Impatto                                                           |
| --------------------------------------------------- | -------------------------------------------------------------------------- | :-------------------- | --------------------------- | ----------------------------------------------------------------- |
| `docws_ricerca_endpoint`                            | `DOCWS_RICERCA_ENDPOINT`                                                   | `str`                 | **Obbligatoria**            | URL SOAP per la federazione delle ricerche atti.                  |
| `docws_atti_endpoint`                               | `DOCWS_ATTI_ENDPOINT`                                                      | `str`                 | **Obbligatoria**            | URL SOAP per l'estrazione binaria degli atti.                     |
| `docws_codice_amministrazione` / `docws_codice_aoo` | `DOCWS_CODICE_AMMINISTRAZIONE` / `DOCWS_CODICE_AOO`                        | `str`                 | **Obbligatorie**            | Metadati di instradamento giuridico dell'Ente.                    |
| `ruolo_docws`                                       | `RUOLO_DOCWS`                                                              | `str`                 | **Obbligatoria**            | Livello di autorizzazione per la lettura del protocollo.          |
| `ws_username` / `ws_password`                       | `WS_USERNAME` / `WS_PASSWORD`                                              | `str`                 | `"sicraweb"` / `"sicraweb"` | Credenziali applicative per l'estrattore.                         |
| `verify_tls`                                        | `VERIFY_TLS`                                                               | `bool`                | `True`                      | Controllo dei certificati SSL verso Sicr@Web.                     |
| `soap_version`                                      | `SOAP_VERSION`                                                             | `str`                 | `"1.1"`                     | Versione del protocollo per la busta SOAP.                        |
| `id_tipo_iter_atteso`                               | `ID_TIPO_ITER_DECRETO_PRESIDENZIALE` / `ID_TIPO_ITER_DECRETO_DELIBERATIVO` | `dict[str, set[str]]` | `{"8"}` / `{"9","19"}`      | Codici di iter attesi per tipologia d'atto, usati in validazione. |
| `max_allowed_pages`                                 | `MAX_ALLOWED_PAGES`                                                        | `int`                 | `300`                       | Limite di pagine oltre il quale un documento non viene elaborato. |

### C. Modelli LLM e Provider

Le chiamate a modelli linguistici **non sono legate a un fornitore specifico**: passano tutte attraverso l'adapter `llm_provider.py`, che espone due sole operazioni (`generate_json`, `generate_text`) e normalizza dietro un'interfaccia unica le differenze fra gli SDK — eccezioni di rate-limit, modalità JSON nativa (assente su Anthropic, emulata via system prompt), `finish_reason` di troncamento (`MAX_TOKENS` su Gemini, `length` su OpenAI, `max_tokens` su Anthropic) e obbligatorietà di `max_tokens` (richiesto solo da Anthropic). Il provider si sceglie con `LLM_PROVIDER` e i tre ruoli — rewriter, generatore, grader — hanno ciascuno il proprio modello configurabile, potendo quindi differenziare capacità e costo per stadio della pipeline.

I default puntano a modelli Gemini, ma sono appunto default: `worker.py` non contiene alcun riferimento a un fornitore nelle fasi di riscrittura, generazione e grading.

> **Unica eccezione: gli embedding.** La generazione dei vettori **non** passa dall'adapter. `worker.py` istanzia direttamente un `genai.Client` e usa la costante `EMBEDDING_MODEL = "gemini-embedding-001"`, non configurabile da variabile d'ambiente. La conseguenza pratica: impostare `LLM_PROVIDER=openai` o `anthropic` cambia rewriter, generatore e grader, ma il sistema **continua a richiedere una `API_LLM_KEY` valida per Google**, poiché senza embedding non c'è né ricerca vettoriale né cache semantica. Il vincolo è deliberato — gli embedding devono restare coerenti con quelli già indicizzati in Qdrant, e cambiarne il modello richiederebbe la re-ingestione completa del corpus — ma va tenuto presente in caso di migrazione di fornitore.
>
> Il reranking è invece **interamente locale** (ONNX Cross-Encoder, nessuna chiamata esterna) e quindi indipendente da qualunque provider.

| Proprietà                     | Variabile d'Ambiente          | Tipo            | Default                    | Impatto                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------------- | ----------------------------- | :-------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llm_provider`                | `LLM_PROVIDER`                | `str`           | `"gemini"`                 | Adapter attivo: `gemini` \| `openai` \| `anthropic`. Non influenza gli embedding.                                                                                                                                                                                                                                                                                                                                                 |
| `query_rewriter_model_name`   | `QUERY_REWRITER_MODEL_NAME`   | `str`           | `"gemini-3.1-flash-lite"`  | De-contestualizzazione e scomposizione delle query.                                                                                                                                                                                                                                                                                                                                                                               |
| `answer_generator_model_name` | `ANSWER_GENERATOR_MODEL_NAME` | `str`           | `"gemini-3.5-flash"`       | Sintesi finale e composizione strutturata.                                                                                                                                                                                                                                                                                                                                                                                        |
| `grader_model_name`           | `GRADER_MODEL_NAME`           | `str`           | `"gemini-3.1-flash-lite"`  | Auditing e grading delle risposte.                                                                                                                                                                                                                                                                                                                                                                                                |
| `ocr_model_name`              | `OCR_MODEL_NAME`              | `str`           | `"gemini-3-flash-preview"` | Modello Vision per la trascrizione dei documenti non nativi (`converter.py`).                                                                                                                                                                                                                                                                                                                                                     |
| `answer_max_tokens`           | `ANSWER_MAX_TOKENS`           | `Optional[int]` | **`None`**                 | Tetto di output della risposta. **`None` = nessun tetto esplicito**: il parametro non viene passato al provider e vale il massimo del modello. Accetta vuoto, `0`, `auto`, `none`, `model` → `None`. Impostare un intero solo per una ragione specifica di lunghezza: **non** per contenere i costi, poiché i token si pagano a consumo e un troncamento costa _più_ di una risposta lunga (il primo tentativo è inutilizzabile). |
| `grader_max_tokens`           | `GRADER_MAX_TOKENS`           | `int`           | `256`                      | Tetto di output del Grader. Qui il tetto **non** è una capability da inseguire ma un _circuit breaker_: l'output è strutturalmente corto (JSON di 4 campi) e senza limite un loop di ripetizione genererebbe fino al soffitto del modello.                                                                                                                                                                                        |
| `answer_thinking_level`       | `ANSWER_THINKING_LEVEL`       | `str`           | `"LOW"`                    | Budget di ragionamento interno del generatore. Su Gemini è **condiviso** col budget di output: è la prima leva da abbassare quando i log segnalano un troncamento causato dal thinking.                                                                                                                                                                                                                                           |
| `grader_thinking_level`       | `GRADER_THINKING_LEVEL`       | `str`           | `"MINIMAL"`                | Idem per il Grader.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `max_model_retries`           | `MAX_MODEL_RETRIES`           | `int`           | `3`                        | Numero di tentativi del ciclo di autocorrezione (profili progressivi).                                                                                                                                                                                                                                                                                                                                                            |
| `allow_general_knowledge`     | `ALLOW_GENERAL_KNOWLEDGE`     | `bool`          | `True`                     | Se `False`, una risposta con `is_general_knowledge` attivo viene rifiutata e si torna nel loop con recall più ampia.                                                                                                                                                                                                                                                                                                              |

### D. Retrieval, Reranking e Concetti

| Proprietà                   | Variabile d'Ambiente        | Tipo       | Default                                        | Impatto                                                                                             |
| --------------------------- | --------------------------- | :--------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `max_history_items`         | `MAX_HISTORY_ITEMS`         | `int`      | `20`                                           | Finestra di cronologia passata al Query Rewriter.                                                   |
| `max_sub_queries`           | `MAX_SUB_QUERIES`           | `int`      | `5`                                            | Sotto-query massime generate dal Rewriter.                                                          |
| `qdrant_semantic_size`      | `QDRANT_SEMANTIC_SIZE`      | `int`      | `30`                                           | Candidati per ricerca vettoriale (moltiplicato dai profili di retry).                               |
| `qdrant_semantic_threshold` | `QDRANT_SEMANTIC_THRESHOLD` | `float`    | `0.60`                                         | Similarità minima sotto la quale i Child sono esclusi.                                              |
| `qdrant_syntactic_size`     | `QDRANT_SYNTACTIC_SIZE`     | `int`      | `20`                                           | Candidati unici estratti dalla ricerca full-text.                                                   |
| `keyword_max_hits`          | `KEYWORD_MAX_HITS`          | `int`      | `1000`                                         | Ricerca testuale a combinazioni: le parole chiave (alias di un concetto in OR) si combinano in AND, prima le combinazioni più ampie; quelle con più pezzi di così sono troppo generiche e si saltano (es. "lavori" da sola no, "lavori + scuole + versilia" sì). |
| `qdrant_concept_threshold`  | `QDRANT_CONCEPT_THRESHOLD`  | `float`    | `0.80`                                         | Soglia di attivazione dell'espansione dal dizionario concettuale.                                   |
| `qdrant_concept_max_hits`   | `QDRANT_CONCEPT_MAX_HITS`   | `int`      | `3`                                            | Concetti sopra soglia considerati (non solo il migliore).                                           |
| `rerank_size`               | `RERANK_SIZE`               | `int`      | `25`                                           | Chunk sopravvissuti al rerank per sotto-query.                                                      |
| `rerank_max_size`           | `RERANK_MAX_SIZE`           | `int`      | `30`                                           | Come `rerank_size`, ma con la ricerca approfondita richiesta dalla chat (`deep_search`).            |
| `rerank_truncate`           | `RERANK_TRUNCATE`           | `int`      | `1200`                                         | Caratteri massimi per chunk passati al Cross-Encoder.                                               |
| `rerank_batch_size`         | `RERANK_BATCH_SIZE`         | `int`      | `4`                                            | Ampiezza del batch di inferenza.                                                                    |
| `rerank_max_length`         | `RERANK_MAX_LENGTH`         | `int`      | `512`                                          | Token massimi del tokenizer del Reranker.                                                           |
| `min_prob_threshold`        | `MIN_PROB_THRESHOLD`        | `float`    | `0.02`                                         | Soglia sigmoidea sui logit; almeno 2 candidati sono sempre mantenuti.                               |
| `onnx_model_cache_path`     | `RERANKER_MODEL_PATH`       | `str`      | `"./model_cache/mmarco-mMiniLMv2-L12-H384-v1"` | Percorso di pesi e tokenizer del modello ONNX.                                                      |
| `parents_per_query`         | `PARENTS_PER_QUERY`         | `int`      | `4`                                            | Quota garantita di Parent per sotto-query.                                                          |
| `extra_parents`             | `EXTRA_PARENTS`             | `bool`     | `true`                                         | Posti extra per pertinenza: oltre alla quota, parent giudicati pertinenti dal reranker, fino a `MAX_SUB_QUERIES × PARENTS_PER_QUERY` parent in tutto.|
| `extra_parents_min_prob`    | `EXTRA_PARENTS_MIN_PROB`    | `float`    | `0.5`                                          | Probabilità minima del reranker perché un parent entri tra i posti extra.                           |
| `mmr_similarity_threshold`  | `MMR_SIMILARITY_THRESHOLD`  | `float`    | `0.92`                                         | Similarità oltre la quale MMR penalizza un candidato ridondante (attivo solo con `mmr_lambda < 1`). |
| `allow_subtopic_selection`  | `ALLOW_SUBTOPIC_SELECTION`  | `bool`     | `True`                                         | Abilita la selezione dei sotto-argomenti nel flusso d'interrogazione.                               |
| `protected_keys`            | _N/A_                       | `set[str]` | costante                                       | Metadati riservati (`parent_id`, `topic_id`, …) protetti da sovrascrittura in ingestion.            |

### E. Budget di Contesto

| Proprietà                | Variabile d'Ambiente     | Tipo    | Default                    | Impatto                                                                                                                                                                                          |
| ------------------------ | ------------------------ | :------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `max_context_chars`      | `MAX_CONTEXT_CHARS`      | `int`   | `30000`                    | Budget complessivo in caratteri del contesto passato al generatore.                                                                                                                              |
| `context_metadata`       | `CONTEXT_METADATA`       | `bool`  | `true`                     | Metadati del documento nel contesto: sul primo passaggio di ogni documento, rimando sugli altri.                                                                                                 |
| `context_metadata_exclude`| `CONTEXT_METADATA_EXCLUDE`| `frozenset`| `id_sicraweb,id_tipo_iter,percorso_originale,giorni_pubblicazione,classifica`| Campi tecnici esclusi dal contesto (oltre a chiavi di sistema, `Header N` e campi con `_` iniziale).                                                                                             |
| `context_metadata_max_value_chars`| `CONTEXT_METADATA_MAX_VALUE_CHARS`| `int`   | `200`                      | Valori dei metadati più lunghi vengono accorciati.                                                                                                                                               |
| `context_floor_ratio`    | `CONTEXT_FLOOR_RATIO`    | `float` | `0.60`                     | Frazione del budget divisa **equamente** tra le sotto-query. Alzarlo è la mossa corretta quando le sotto-query in coda restano sotto-servite.                                                    |
| `context_cap_ratio`      | `CONTEXT_CAP_RATIO`      | `float` | `0.40`                     | Tetto anti-monopolio: nessuna sotto-query oltre questa frazione del contesto.                                                                                                                    |
| `context_score_weight`   | `CONTEXT_SCORE_WEIGHT`   | `str`   | `"best"`                   | Criterio di pesatura per sotto-query: `best` (meno sensibile alla numerosità) \| `mean`.                                                                                                         |
| `context_score_source`   | `CONTEXT_SCORE_SOURCE`   | `str`   | `rerank`                   | Rilevanza dei parent per le quote di contesto, MMR e ordine dei core: `rerank` = probabilità del reranker, `vector` = similarità vettoriale (comportamento precedente).|
| `context_overflow_ratio` | `CONTEXT_OVERFLOW_RATIO` | `float` | `0.15`                     | Sforamento tollerato per includere un documento **intero** invece di finestrarlo per pochi caratteri. Non può mai servire ad aggiungere un Parent nuovo, solo a completarne uno già selezionato. |
| `context_hard_cap_ratio` | `CONTEXT_HARD_CAP_RATIO` | `float` | `0.20`                     | Tetto invalicabile espresso come **frazione** di `MAX_CONTEXT_CHARS`. Un guardrail lo allinea a `CONTEXT_OVERFLOW_RATIO` se impostato più basso.                                                 |
| `context_hard_cap_chars` | _derivato_               | `int`   | `MAX_CONTEXT_CHARS × 1.20` | **Non leggibile da env.** Calcolato in `load_settings()`; è il valore che `build_context()` usa come rete di sicurezza finale.                                                                   |
| `context_snap_boundary`  | `CONTEXT_SNAP_BOUNDARY`  | `str`   | `"sentence"`               | Confine su cui arretrare il taglio: `sentence` \| `paragraph` \| `none`. Costo zero, nessuna espansione.                                                                                         |

### F. Revisione, Autenticazione e Accesso ai File

| Proprietà | Variabile d'Ambiente | Tipo | Default | Impatto |
| --- | --- | :--- | --- | --- |
| `auth_secret_key` | `AUTH_SECRET_KEY` (o il vecchio `REVIEW_SECRET_KEY`) | `str` | `""` | Firma dei token di sessione. Almeno 32 caratteri casuali, solo nell'API. Vuota: rotte `/review`, `/auth` e download riservati rispondono 503, la chat funziona. |
| `session_cookie_secure` | `SESSION_COOKIE_SECURE` | `bool` | `true` | Cookie `rag_session` solo su HTTPS. `false` solo per lo sviluppo in http su un host diverso da localhost. |
| _N/A_ (`review_routes.py`) | `REVIEW_SESSION_HOURS` | `int` | `10` | Durata di una sessione. |
| `review_lock_ttl_seconds` | `REVIEW_LOCK_TTL_SECONDS` | `int` | `300` (min `60`) | Durata del blocco in modifica senza rinnovi; il frontend lo rinnova ogni terzo di questo tempo. |
| `files_x_accel_prefix` | `FILES_X_ACCEL_PREFIX` | `str` | `""` | Prefisso della location `internal` di nginx per `X-Accel-Redirect` (in k8s `/_protected_files/`). Vuoto: i file li invia l'API. |
| `files_login_url` | `FILES_LOGIN_URL` | `str` | `"/revisione/"` | Pagina di login proposta da `/files` a chi apre un documento riservato senza sessione. |

L'API deve inoltre montare in lettura/scrittura lo stesso volume dati di converter e ingest (`processed/`, `watch/`, `ingestion/error/`).

### G. Risorse di Calcolo e Concorrenza

Questi quattro parametri devono essere **coerenti tra loro e con le risorse effettivamente concesse al processo**: sono la causa più frequente di sovrasottoscrizione della CPU.

| Proprietà             | Variabile d'Ambiente  | Tipo  | Default          | Impatto                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------- | --------------------- | :---- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cpu_limit`           | `CPU_LIMIT`           | `int` | `os.cpu_count()` | Core effettivamente utilizzabili dal processo. **Il fallback su `os.cpu_count()` è affidabile solo in esecuzione diretta su una macchina dedicata**: in un container con un limite di CPU (`--cpus` di Docker, `cpus` in Compose, `resources.limits.cpu` in Kubernetes) il limite è applicato via quota CFS e _non_ modifica il numero di core che il sistema operativo dichiara, quindi il valore letto sovrastima — anche di parecchie volte su un nodo capiente. Va impostato esplicitamente ogni volta che il processo gira sotto un limite. |
| `celery_concurrency`  | `CELERY_CONCURRENCY`  | `int` | `1`              | Deve combaciare con `--concurrency` passato al comando Celery. Serve a `get_reranker_pool()` per sapere quanti processi condividono gli stessi core: senza, ogni processo dimensionerebbe il proprio pool assumendo di avere l'intera macchina per sé.                                                                                                                                                                                                                                                                                           |
| `reranker_pool_size`  | `RERANKER_POOL_SIZE`  | `int` | `1`              | Istanze ONNX indipendenti per processo. Ognuna è una copia completa del modello in RAM (~470 MB): è la leva più efficace sul footprint di memoria.                                                                                                                                                                                                                                                                                                                                                                                               |
| `max_reranker_thread` | `MAX_RERANKER_THREAD` | `int` | `2`              | Thread di rerank concorrenti (sotto-query processate in parallelo).                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

> **Guardrail attivo**: se `RERANKER_POOL_SIZE < MAX_RERANKER_THREAD`, `load_settings()` alza il pool e logga un WARNING. Il round-robin su `q_idx` assegnerebbe altrimenti la **stessa** `InferenceSession` (che è `ORT_SEQUENTIAL`) a più thread, che si serializzerebbero sullo stesso budget di calcolo — esattamente il problema per cui il pool esiste.
>
> Va però considerato che il pool **non aggiunge capacità di calcolo, la ridistribuisce**: con 2 core per processo, `pool=2` dà due istanze a 1 thread ciascuna, `pool=1` dà una istanza a 2 thread intra-op. Quando si è memory-bound, `pool=1` è una configurazione legittima e va valutata misurando, non esclusa a priori.

## 9. Funzionalità Mancanti e Sviluppi Possibili

Elenco delle capacità che il sistema oggi non offre e che rappresenterebbero un'estensione significativa. Non è un piano di lavoro: le voci sono indipendenti fra loro e ordinate per area, non per priorità.

### 9.1 Indipendenza dal fornitore di embedding

Riscrittura, generazione e grading sono già provider-agnostici (§8.C). La generazione dei vettori no: è vincolata a Google tramite una costante interna, quindi anche adottando integralmente un altro fornitore per gli LLM il sistema continua a richiedere credenziali Google.

Rendere l'embedding configurabile alla pari degli altri stadi consentirebbe di scegliere il fornitore in base a criteri propri — costo, residenza del dato, qualità sulla lingua italiana amministrativa — indipendentemente da quale modello genera le risposte. Per un ente pubblico la voce ha anche una valenza di sovranità del dato: gli embedding sono l'unica componente che richiede l'invio dell'**intero corpus** a un servizio esterno, non solo delle query.

Va tenuto presente che la dimensione del vettore è fissata alla creazione della collection Qdrant: cambiare modello comporta la re-indicizzazione completa dell'archivio. La configurabilità va quindi accompagnata da una verifica di coerenza fra modello configurato e indice esistente.

> _Nota documentale_: la dimensionalità dichiarata in §3 (768) va confermata leggendola dalla collection, poiché il modello in uso supporta dimensioni configurabili.

### 9.2 Esperienza di interrogazione

- **Risposta in streaming.** Oggi il client invia la domanda, riceve `202 Accepted` e interroga ciclicamente lo stato finché il task non è concluso: da trenta secondi a oltre un minuto di attesa senza alcun riscontro intermedio. Lo streaming incrementale dei token, o quantomeno la pubblicazione dello stato di avanzamento della pipeline ("riformulazione", "ricerca", "sintesi"), cambierebbe radicalmente la percezione del sistema a parità di tempo di calcolo.
- **Filtri espliciti per l'utente.** Il motore supporta già filtri su intervalli di date, valori esatti e liste di metadati, ma sono raggiungibili solo per via programmatica. Esporli nell'interfaccia — "solo determine del 2024", "solo atti del Settore Viabilità" — sposterebbe una capacità già presente dal codice all'utente finale.
- **Citazioni puntuali.** Le risposte citano il documento di provenienza per nome file. Poiché i frammenti conservano la posizione nel testo originale, sarebbe possibile portare l'utente **al punto esatto** del documento, evidenziando il passaggio da cui l'affermazione deriva. Per un contesto amministrativo, dove la verificabilità della fonte conta quanto la risposta, è probabilmente l'aggiunta di maggior valore percepito.
- **Continuità documentale nel dialogo.** La cronologia conversazionale trasporta solo il testo scambiato. I documenti recuperati al turno precedente non restano disponibili, quindi una domanda di approfondimento ("e cosa dice l'articolo 3 di quell'atto?") innesca una ricerca completamente nuova, che può non ripescare lo stesso documento.

### 9.3 Ricerca e qualità dei risultati

- **Fusione ibrida dei ranking.** Le ricerche semantica e lessicale coesistono, ma i risultati della seconda vengono semplicemente accodati ai candidati della query principale. Una fusione formale dei due ranking — Reciprocal Rank Fusion o equivalente — sfrutterebbe la complementarità dei due segnali invece di privilegiare arbitrariamente uno dei due percorsi.
- **Ricerca esatta su identificatori.** Domande su CIG, numeri di determina, protocolli e date sono ricerche di **identificatori**, non di concetti, e la similarità vettoriale è lo strumento sbagliato: un codice alfanumerico non ha vicini semantici significativi. Un percorso dedicato che estragga in ingestione i metadati strutturati e li interroghi per corrispondenza esatta darebbe risposte deterministiche dove oggi si affida alla fortuna del retrieval.
- **Deduplica dei contenuti quasi identici.** Il corpus amministrativo contiene ampie porzioni ricorrenti (formule di rito, riferimenti normativi, clausole standard). Capita che più frammenti pressoché identici occupino le prime posizioni del riordinamento, consumando budget di contesto senza aggiungere informazione. Una deduplica sui contenuti — non solo sull'identità esatta, già gestita — libererebbe spazio per materiale complementare.
- **Ricerca su tabelle e allegati strutturati.** Gli atti amministrativi contengono quadri economici, elenchi di interventi e piani particellari. La conversione in Markdown ne preserva l'aspetto ma non la struttura: una domanda che richieda di aggregare valori attraverso più tabelle non è oggi servibile in modo affidabile.

### 9.4 Valutazione e miglioramento continuo

- **Insieme di riferimento per la valutazione.** Il comportamento del sistema dipende da una decina di soglie numeriche (similarità, quote di contesto, probabilità minima, diversificazione). Senza un insieme di domande con i documenti attesi, ogni taratura è una scommessa: non è possibile sapere se una modifica migliora il sistema o sposta soltanto il problema. È la funzionalità mancante con il maggior effetto abilitante su tutte le altre.
- **Sfruttamento attivo del feedback.** I giudizi degli utenti e le sessioni fallite sono registrati, ma restano materiale di sola consultazione. Potrebbero alimentare un insieme di risposte verificate da preferire, o segnalare automaticamente le domande ricorrenti che il sistema non sa servire.
- **Analisi delle lacune di archivio.** Il valutatore registra, per ogni risposta, le entità richieste su cui non ha trovato nulla. Un'entità che ricorre sistematicamente in quell'elenco indica un buco di **ingestione**, non di ricerca. Un rapporto periodico su quel dato trasformerebbe un'informazione già raccolta in una segnalazione azionabile verso chi cura il corpus.
- **Cruscotto operativo.** Le metriche per richiesta sono persistite ma non presentate: tempi per stadio, tasso di riuso della cache, tentativi medi, frequenza dei troncamenti. Renderle visibili permetterebbe di accorgersi di un degrado prima che lo segnali un utente.

### 9.5 Esercizio, sicurezza e conformità

- **Verifiche di stato del servizio.** L'API espone `GET /health` (senza autenticazione), usato da chat e revisione per la pagina di manutenzione, ma verifica solo il pool MySQL: non l'indice vettoriale, né il worker Celery e l'avvenuto caricamento del riordinatore, né la presenza di `AUTH_SECRET_KEY`. Un supervisore — `HEALTHCHECK` di Docker, sonde di Kubernetes — non può quindi ancora distinguere un servizio pronto da uno che risponde ma non può servire una domanda.
- **Controllo degli accessi sul contenuto delle risposte.** Utenti, ruoli e archivi assegnati esistono (§6) e proteggono l'apertura dei file originali degli archivi riservati, ma la chat è ancora protetta da un unico token condiviso e risponde usando il testo di tutti i documenti del topic. Legare il retrieval all'identità dell'utente (filtrare i parent per archivio o per documento in base ai suoi permessi) è il passo necessario per servire con la stessa chat atti a diversa visibilità.
- **Tracciabilità delle risposte.** In ambito amministrativo può essere necessario ricostruire a posteriori che cosa il sistema ha risposto, a chi, sulla base di quali documenti e con quale versione dei prompt. Gli elementi sono in parte già registrati, ma non in una forma pensata per essere consultata come registro.
- **Espurgo e diritto all'oblio.** Non esiste un percorso per rimuovere selettivamente un documento e tutte le sue tracce — frammenti vettoriali, documenti padre, voci di cache semantica che potrebbero riprodurne il contenuto.

### 9.6 Ottimizzazioni note

Interventi che non aggiungono funzionalità ma riducono tempi o costi, elencati per completezza:

- **Avvio del riordinatore**: circa sette secondi per processo, riducibili sotto il secondo predisponendo gli artefatti del modello in forma già ottimizzata.
- **Modello quantizzato**: la variante a interi dimezzerebbe l'occupazione di memoria per istanza e accelererebbe l'inferenza, previa verifica della perdita di qualità.
- **Chiamate di embedding**: tre viaggi di rete sequenziali per richiesta, accorpabili.
- **Trasporto verso il database vettoriale**: il protocollo binario ridurrebbe la latenza sui profili che trasferiscono vettori.

### 9.7 Parità tra fornitori

L'adapter normalizza le differenze fra SDK, ma la copertura non è uniforme. Da considerare prima di un cambio di fornitore:

| Funzionalità                          |       Gemini        |   OpenAI    |          Anthropic           |
| ------------------------------------- | :-----------------: | :---------: | :--------------------------: |
| Modalità JSON nativa                  |         ✅          |     ✅      | ⚠️ emulata via system prompt |
| Livello di ragionamento configurabile |         ✅          | ❌ ignorato |         ❌ ignorato          |
| Tetto di output omettibile            |         ✅          |     ✅      |       ❌ obbligatorio        |
| Conteggio dei token di ragionamento   |         ✅          |     ❌      |              ❌              |
| Embedding                             | ✅ unico supportato |      —      |              —               |

La diagnostica del troncamento descritta in §5.2 distingue le cause grazie ai token di ragionamento, disponibili solo su Gemini: sugli altri fornitori degrada a una segnalazione generica.

## 10. Manutenzione, Monitoraggio e Scheduled Task

### Configurazione Celery (`worker.py`)

| Parametro                     | Valore       | Motivazione                                                                                                                                                                                                                                            |
| ----------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `worker_prefetch_multiplier`  | `1`          | Un task per volta per processo: i task sono lunghi e pesanti, il prefetch creerebbe code interne invisibili.                                                                                                                                           |
| `task_acks_late`              | `True`       | Il messaggio è confermato a task **completato**, non ricevuto.                                                                                                                                                                                         |
| `task_reject_on_worker_lost`  | `True`       | Un task interrotto dalla morte del processo torna in coda. Combinato con `acks_late`, un OOM a metà elaborazione causa una **riesecuzione completa**: è il motivo per cui conviene che a riciclare sia Celery (a fine task) e non il kernel (durante). |
| `worker_max_tasks_per_child`  | `100`        | Riciclo periodico contro la crescita lenta della memoria.                                                                                                                                                                                              |
| `worker_max_memory_per_child` | `2400000` KB | Soglia di riciclo per processo (~2,29 GiB). Confrontata da billiard con `ru_maxrss`, cioè il **picco storico**, che non decresce mai. Va tarata sopra `picco a riposo + consumo del task più pesante`.                                                 |
| `worker_proc_alive_timeout`   | `60.0`       | Secondi concessi a un processo forkato per segnalare "UP". Deve superare con margine il tempo di init di `worker_process_init`, che pre-carica il pool ONNX (~7s). Il default della libreria è `4.0`: insufficiente, e produce un crash loop infinito. |

### Dimensionamento della memoria

Il pool ONNX viene caricato **dopo il fork** (una sessione non è ereditabile in sicurezza), quindi le istanze **non sono condivise in copy-on-write** tra i processi Celery: ogni child paga la propria copia per intero.

Valori misurati su `RERANKER_POOL_SIZE=2` con `--concurrency=2`:

|                                             | RSS                |
| ------------------------------------------- | ------------------ |
| Processo padre                              | ~184 MiB           |
| Child (a riposo, 2 sessioni ONNX + runtime) | ~1,69 GiB ciascuno |
| **Totale a riposo**                         | **~3,56 GiB**      |

A cui va sommato il consumo per task (tra 223 e 613 MiB con un contesto da ~150k caratteri). Il servizio worker richiede quindi almeno **6 GiB** di memoria disponibile; con 4 GiB si satura e il riciclo dei processi diventa continuo. Il valore scala con `RERANKER_POOL_SIZE × concorrenza`: ogni istanza del riordinatore è una copia completa del modello.

**Diagnostica del riciclo continuo** — sintomo: un processo figlio riparte dopo ogni task, ripagando ~7s di caricamento del modello ogni volta. Le cause possibili sono due e si distinguono così:

```bash
# 1. Consumo reale dei processi (RSS in KB)
ps -o pid,rss,comm -e

# 2. Se il processo gira in un container, i contatori del cgroup dicono
#    se è stato il kernel a intervenire (cgroup v2):
cat /sys/fs/cgroup/memory.events
#   oom_kill > 0  -> OOM killer: serve più memoria al container, o un pool più piccolo
#   oom_kill = 0  -> è worker_max_memory_per_child: va alzato
```

I comandi vanno eseguiti nel contesto del processo worker: direttamente in installazione diretta, oppure con `docker exec <container>`, `docker service ps` più `docker exec` in Swarm, `kubectl exec deploy/<nome>` in Kubernetes.

Due avvertenze sulla lettura dei sintomi:

- **Un OOM del kernel non è sempre evidente.** A morire è un processo figlio, non il PID 1: il container resta in esecuzione, il suo stato non riporta alcun `OOMKilled` né codice di uscita 137, e Celery rimpiazza il processo in silenzio. Solo i contatori del cgroup o il log del kernel dell'host lo rivelano.
- **`memory.peak` non è sempre disponibile**: è stato introdotto nel kernel 6.8. Su kernel precedenti si usano `memory.current` e `memory.events`. Su cgroup v1 i percorsi sono diversi (`/sys/fs/cgroup/memory/memory.failcnt`, `memory.max_usage_in_bytes`).

### Altri presidi

- **Logging Centralizzato su Database (`system_logs`)**: handler `MySQLLogHandler` integrato con il logging standard. Le anomalie di livello `WARNING` o superiore (con stacktrace, file e riga) vengono inserite a database. Il recupero delle connessioni dal pool è protetto da un ping preventivo (`conn.ping(reconnect=True)`).
- **CronJob orario di pulizia cache semantica (`prune_semantic_cache`)**: task periodico di Celery Beat che rimuove da `semantic_cache` i record con timestamp superiore a `MAX_AGE_SECONDS` (24 ore), prevenendo l'obsolescenza dei contesti normativi.
- **Metriche per richiesta (`rag_metrics`)**: ogni task registra tentativi, cache hit, parametri di retrieval effettivi e `missing_targets` del Grader. Sui volumi, questi ultimi distinguono un problema di **ingestion** (documenti mai indicizzati per un'entità) da uno di **soglie** (presenti ma non agganciati dal retrieval).
- **Retry delle API con backoff e jitter**: le interazioni con API LLM, embedding e vector DB sono protette da `tenacity`. Su rate-limit (429) o indisponibilità (503) si applica un backoff esponenziale con jitter casuale, che impedisce fenomeni di risonanza tra pod concorrenti.
- **Script di amministrazione** (`utils/`, da eseguire nel pod dell'API con `python -m utils.<nome>`):
  - `manage_review_users`: utenti (`add`, `password`, `disable`/`enable`, `list`), ruoli (`grant`, `revoke`, `roles`), archivi assegnati (`allow`, `deny`) e accesso pubblico/riservato degli archivi (`topics`, `public`, `private`).
  - `reindex_topic <topic>`: ripassa dall'ingestione i documenti già indicizzati di un archivio, copiando i pacchetti da `processed/` a `watch/` e pubblicandoli su `da-indicizzare` a lotti (`--batch`, `--max-pending` sulla profondità della coda). Di default solo i documenti il cui testo **indicizzato** contiene ancora `&nbsp;` o `<mark>` (rilanciandolo riprende da dove era arrivato; `--all` per tutti); salta i pacchetti già in `watch/` (correzioni della revisione in sospeso) e segnala i documenti indicizzati senza pacchetto su disco. `--dry-run` per vedere cosa farebbe. Dopo una re-indicizzazione estesa conviene svuotare la cache semantica (`qdrant_semantic_cache_reset`).
- **Upgrade Infrastrutturale**: i miglioramenti di velocità sotto carico massivo richiederanno hardware dedicato (GPU/TPU) per svincolare l'Ente dalle code di elaborazione condivise.

## 11. Requisiti di Sistema e Dipendenze Python (`requirements.txt`)

Per consentire la corretta esecuzione del sistema in ambienti containerizzati (come Docker o Kubernetes), è essenziale mappare accuratamente tutte le librerie software utilizzate dai diversi moduli dell'applicazione. Di seguito viene fornito il contenuto del file `requirements.txt` seguito da un'analisi dell'utilizzo di ciascun pacchetto.

### A. File `requirements.txt` Completo

```text
# Core framework & Web API
Flask>=3.0.0
Flask-CORS>=4.0.0
python-dotenv>=1.0.0

# Task Queue & Async Operations
celery>=5.3.0
redis>=5.0.0
aio-pika>=9.0.0
pika>=1.3.0
aiolimiter>=1.1.0

# Database & Storage Connectors
mysql-connector-python>=8.2.0
qdrant-client>=1.18.0

# AI Models, LLM SDKs & Text Splitting
google-genai>=0.1.1
google-cloud-vision>=3.4.0
langchain-text-splitters>=0.0.1
typeguard>=4.1.0

# Local Reranking (ONNX & NLP)
onnxruntime>=1.16.0
transformers>=4.35.0
numpy>=1.24.0

# Document Extraction & Parsing
PyMuPDF>=1.23.0
pymupdf4llm>=0.0.12
asn1crypto>=1.5.1
pillow>=10.0.0

# Utilities & Resilience
tenacity>=8.2.0

```

### B. Mappatura Funzionale delle Dipendenze

Ogni pacchetto risponde a un preciso requisito algoritmico ed è strettamente accoppiato con uno o più moduli di codice:

1. **Estrattore SOAP e Decodifica (`estrattore.py`)**:

- `pika`: Consente l'invio sincrono dei messaggi di notifica degli atti estratti alla coda `da-convertire` di RabbitMQ.
- `asn1crypto`: Fornisce il parser nativo in RAM per decodificare le firme digitali in formato PKCS#7 / CAdES (`.p7m`), consentendo l'isolamento del PDF nativo senza appoggiarsi a tool di sistema esterni.

2. **Convertitore Documentale (`converter.py`)**:

- `PyMuPDF (fitz)` & `pymupdf4llm`: Consentono l'estrazione ad altissime prestazioni del testo nativo dai documenti PDF dell'Ente e la conversione strutturata in Markdown pulito.
- `pillow (PIL)`: Utilizzato per manipolare e inviare i buffer delle immagini estratte alle API di OCR.
- `google-cloud-vision`: Gestisce il fallback sincrono per l'estrazione visiva in caso di blocco di Gemini a causa delle policy di sicurezza.
- `aio-pika`: Consente il consumo asincrono dei messaggi e la notifica simultanea al modulo di ingestion.
- `aiolimiter`: Limita e regola la frequenza di chiamata delle API di Gemini Vision per evitare errori di timeout o saturazione delle quote.

3. **Ingestione Dati (`ingest.py`)**:

- `langchain-text-splitters`: Fornisce le classi `MarkdownHeaderTextSplitter` e `RecursiveCharacterTextSplitter` per lo splitting gerarchico Parent-Child.
- `qdrant-client`: Gestisce la connessione asincrona ad alta velocità verso il database vettoriale Qdrant for il caricamento dei Child Chunks.
- `typeguard`: Richiesto dal nuovo SDK di Google GenAI per la validazione dinamica dei tipi di dati passati in chiamata.

4. **Motore RAG e Reranking Locale (`worker.py` e `reranker.py`)**:

- `celery` & `redis`: Gestiscono la ricezione asincrona, la prioritizzazione e il tracciamento dei task della coda RAG.
- `onnxruntime`: Consente l'esecuzione efficiente e multi-thread limitata del modello Cross-Encoder locale, eliminando la necessità di GPU per il riordinamento di 25 candidati.
- `transformers`: Fornisce la classe `AutoTokenizer` compatibile con i modelli HuggingFace per la tokenizzazione locale della query e dei chunk.
- `mysql-connector-python`: Gestisce la persistenza e il retrieval dei Parent Documents e la scrittura asincrona dei log di sistema.
- `tenacity`: Fornisce i decoratori per implementare politiche difensive di retry con jitter casuale su tutte le chiamate esterne instabili.
