# Specifica Tecnica ed Architetturale: Sistema RAG Enterprise-Grade per l'Ente Locale

Questo documento descrive in modo formale e approfondito l'architettura, le strategie algoritmiche, i flussi di dati, lo schema del database, la mappatura di configurazione e i requisiti di sistema del modello **Retrieval-Augmented Generation (RAG)** avanzato sviluppato per l'Ente Locale. Il sistema adotta un approccio asincrono, resiliente, guidato dagli eventi (_Event-Driven_) e strutturato su una filiera multi-stadio protetta.

## 1. Introduzione e Obiettivi dell'Architettura

I sistemi RAG tradizionali (_Naïve RAG_) soffrono di limitazioni strutturali e algoritmiche che ne precludono l'adozione in scenari della Pubblica Amministrazione regolati da rigidi vincoli di conformità amministrativa e precisione:

- **Perdita di precisione semantica:** Blocchi di testo (_chunk_) troppo ampi diluiscono il segnale semantico dei vettori di embedding, portando a mancate corrispondenze.
- **Mancanza di contesto strutturale:** I documenti vengono frammentati in piccoli paragrafi isolati. Di conseguenza, il modello generativo (LLM) ha una visione parziale: vede solo le singole tessere del puzzle, perdendo il contesto logico-normativo circostante come preamboli d'atto, visti istituzionali o articoli di legge correlati.
- **Instabilità delle API esterne ed allucinazioni:** Fluttuazioni nelle quote delle API esterne e tendenza all'allucinazione in mancanza di vincoli tassativi di grounding documentale.

Questo sistema risolve tali sfide combinando un'architettura gerarchica dei dati **Parent-Child** (Parent Document Retriever) per conservare il contesto originario , una pipeline di ingestion guidata da eventi asincroni (**RabbitMQ**), un motore di retrieval basato su **Reranker Cross-Encoder locale** in ambiente Celery protetto da deadlock, un sistema di **Analisi Concettuale** per l'espansione dei sinonimi della Pubblica Amministrazione , un algoritmo di **Ripartizione e Redistribuzione Attiva delle Quote di Contesto**, e un **Ciclo di Autocorrezione con Grader CoT** (Chain of Thought) a feedback automatico.

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

### I Componenti Centrali:

1. **API Gateway (`app.py` - Flask)**: Espone le interfacce REST protette da Bearer Token. Gestisce il bootstrap caricando le rotte e i topic dinamici, convalida le richieste di chat, raccoglie i feedback e delega in modalità non bloccante la computazione pesante a Celery, rispondendo istantaneamente con un codice `202 Accepted` per l'avvio del polling.
2. **Celery Worker (`worker.py`)**: Il vero motore di ricerca e generazione. Gestisce la trasformazione semantica della query, la ricerca ibrida parallela, il riordinamento Cross-Encoder locale, la generazione strutturata e l'auditing qualitativo della risposta (Grader).
3. **Estrattore Massivo (`estrattore.py` - CLI / CronJob)**: Il ponte di collegamento con la suite Sicr@Web dell'Ente Locale. Invia richieste SOAP ai WS di ricerca ed estrazione, gestisce l'autenticazione applicativa ed estrae i flussi binari in sicurezza.
4. **Wrapper Diretti (`direct.py` - CLI / Daemon)**: Scansiona la cartella di ingresso locale per i documenti caricati manualmente, mantiene la gerarchia delle cartelle convertendola in copie `topic_id/sub_topic_id` e pubblica gli eventi su RabbitMQ.
5. **Convertitore Documentale (`converter.py` - Worker Async)**: Elabora asincronamente i PDF e le immagini. Esegue l'estrazione testuale nativa o l'OCR avanzato tramite Gemini Vision, con fallback asincrono su Google Cloud Vision API in caso di blocco delle policy di sicurezza.
6. **Worker Ingest (`ingest.py` - Worker Async)**: Frammenta i documenti normalizzati in strutture gerarchiche Parent-Child, genera gli embedding vettoriali e popola in modo transazionale MySQL e Qdrant.

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
2. **Splitting Gerarchico Parent-Child**:
   - **Parent Splitting**: Se abilitato, divide il documento Markdown tramite `MarkdownHeaderTextSplitter` basandosi sugli header logici (`#` fino a `####`), catturando la gerarchia nei metadati. Successivamente, utilizza `RecursiveCharacterTextSplitter` con il tokenizer di Tiktoken (`cl100k_base`) per generare blocchi ampi coerenti (Parent).
   - **Child Splitting**: Ciascun blocco Parent viene frammentato in blocchi Child più piccoli e sovrapposti (`chunk_size` e `chunk_overlap`) per consentire una ricerca semantica millimetrica.
3. **Idempotenza Transazionale MySQL**:
   - Per garantire che la re-ingestione di un documento modificato non lasci record duplicati, il worker esegue un'operazione atomica in MySQL: prima esegue una `DELETE` su `parent_documents` basata sulla chiave `source`, `topic_id` e `sub_topic_id`, poi esegue una `executemany` per inserire massivamente i nuovi Parent Documents.
4. **Generazione Embedding in Batch ed Ingestione Qdrant**:
   - I blocchi Child vengono inviati in batch asincroni alle API Google GenAI per generare vettori a 768 dimensions via `gemini-embedding-001`. Il processo è protetto da un `AsyncLimiter` impostato a 1000 richieste/minuto e gestisce gli errori `ResourceExhausted` (429) e `ServiceUnavailable` (503) tramite retry con backoff esponenziale.
5. **Prevenzione OOM (Batch Flush su Qdrant)**:
   - Al fine di proteggere la memoria RAM durante l'ingestione di file Markdown di grandissime dimensioni, i punti Child vengono assemblati e accumulati in un buffer. Non appena il buffer supera i 500 punti, viene eseguito un _flush intermedio_ verso Qdrant (`qdrant_client.upsert`), svuotando la memoria prima di procedere.
6. **Qdrant Garbage Collection**:
   - Per rimuovere i "vettori fantasma" (i vecchi frammenti Child rimasti nel Vector DB dopo che il file originale è stato aggiornato), il sistema esegue una query di cancellazione mirata (`qdrant_client.delete`), rimuovendo tutti i punti associati allo stesso `source` ma dotati di un `_ingestion_id` obsoleto (diverso dall'ID univoco di sessione appena generato).

## 4. Motore di Retrieval Avanzato & Reranking

Il processo di recupero e ordinamento dei contesti all'interno di `worker.py` supera i limiti della semplice ricerca semantica (K-Nearest Neighbors) implementando una catena di elaborazione avanzata.

### A. Trasformazione delle Query ed Espansione Semantica

1. **Riscrittura Conversazionale**: Il modello configurato in `QUERY_REWRITER_MODEL_NAME`, invocato tramite l'adapter provider-agnostico, analizza fino a `MAX_HISTORY_ITEMS` messaggi di cronologia, riscrive la domanda in forma autoconsistente (`standalone_query`), la scompone in sotto-quesiti distinti (`search_queries`, fino a `MAX_SUB_QUERIES`), estrae parole chiave lessicali (`keywords`) e — per le domande aggregative — l'elenco delle entità che la risposta deve coprire (`coverage_targets`).
2. **Lookup Semantico del Concetto**:
   - Il sistema esegue l'embedding della query standalone e interroga la collezione Qdrant `conceptual_dictionary`.
   - Vengono considerati **tutti** i concetti sopra `QDRANT_CONCEPT_THRESHOLD` (`0.80`) fino a un massimo di `QDRANT_CONCEPT_MAX_HITS` (`3`), non soltanto il migliore: una domanda può legittimamente insistere su più macro-aree.
   - Gli alias dei concetti verificati vengono **iniettati nel prompt del rewriter** tramite `format_verified_concepts()`, che li presenta come vocabolario disponibile invece di sostituirli meccanicamente nella query. È il rewriter a decidere come usarli, tipicamente generando una sotto-query per ciascun comune di una macro-area (es. "Versilia" → Viareggio, Forte dei Marmi, Pietrasanta, Camaiore, Massarosa).
   - `collect_concept_aliases()` aggiunge gli stessi alias al set di keyword usate dalla ricerca sintattica.

### B. Ricerca Ibrida Parallela e Reranking Locale via ONNX Cross-Encoder

Per massimizzare il rendimento ed evitare colli di bottiglia, la ricerca dei frammenti candidati avviene in parallelo sfruttando un pool di thread concorrenti (`ThreadPoolExecutor`):

1. **Ricerca Semantica**: Query vettoriali parallele su Qdrant (`document_chunks`) per ciascuna sotto-query, con soglia `QDRANT_SEMANTIC_THRESHOLD` (`0.60`), raggruppamento per `content_hash` (deduplica dei chunk identici) e filtro su `topic_id` / `sub_topic_id` attivi. Il client Qdrant è istanziato con `timeout=10` secondi. Il margine è necessario perché `safe_vector_search` è _fail-soft_: in caso di eccezione registra l'errore e restituisce una lista vuota senza ritentare, quindi una search andata in timeout costa l'**intera** sotto-query e con essa la copertura dell'entità corrispondente.
2. **Ricerca Sintattica**: In parallelo, una ricerca full-text lessicale (`MatchText`) sulle keyword, con scroll paginato e deduplica per `content_hash` fino a `QDRANT_SYNTACTIC_SIZE` candidati unici. I risultati confluiscono tra i candidati della query principale (indice 0).
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
  I chunk con probabilità inferiore a `MIN_PROB_THRESHOLD` (`0.02`) vengono scartati, salvo mantenere sempre almeno due candidati per sotto-query. Sopravvivono al massimo `RERANK_SIZE` (`25`) chunk per sotto-query.

### C. Selezione dei Parent, Diversificazione MMR e Ripartizione del Budget di Contesto

La selezione avviene in tre stadi distinti: **quali** documenti entrano, **quanto spazio** ciascuno riceve, e **quale porzione** di ciascuno viene mostrata.

#### C.1 Quota garantita e redistribuzione con MMR

1. **Quota per sotto-query**: ciascuna sotto-query ha diritto a `PARENTS_PER_QUERY` (`4`) documenti Parent distinti, risalendo dai propri migliori chunk. Il budget nominale è quindi `PARENTS_PER_QUERY × N_query`.
2. **Slot liberi**: se una sotto-query non produce risultati (nessun documento per quel comune, o una search fallita), la sua quota resterebbe inutilizzata:
   $$\text{Slot Liberi} = (\text{parents\_per\_query} \times N_{\text{queries}}) - N_{\text{selezionati}}$$
3. **Redistribuzione**: gli slot liberi vengono riassegnati attingendo alle riserve delle altre sotto-query tramite `select_with_mmr()`. Con `mmr_lambda = 1.0` (tentativo 0) la funzione va in **bypass** e si comporta come un ranking puro per score; con `lam < 1.0` penalizza i candidati troppo simili ai Parent già selezionati, confrontando i centroidi dei loro chunk con soglia `MMR_SIMILARITY_THRESHOLD` (`0.92`). La quota garantita non viene **mai** toccata da MMR: il centroide è un vettore-proxy, e non è abbastanza affidabile da giustificare l'esclusione di un documento rilevante.

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

## 6. Schema del Database Relazionale (MySQL)

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


┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  parent_docs     │      │  chat_feedback   │      │   system_logs    │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

### Dettaglio Tabelle, Vincoli e Indici composti:

1. **`topics`**: Mappa i macro-argomenti e associa i prompt di sistema personalizzati.
   - `topic_id` (VARCHAR(100) PRIMARY KEY)
   - `description` (TEXT, NOT NULL)
   - `aliases` (TEXT, NULL)
   - `prompt` (VARCHAR(100), NULL)
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

## 7. Mappatura Dettagliata delle Variabili di Configurazione (`config.py`)

La configurazione centralizzata raccoglie e convalida le variabili d'ambiente nella dataclass immutabile `Settings` (72 campi).

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
| `qdrant_concept_threshold`  | `QDRANT_CONCEPT_THRESHOLD`  | `float`    | `0.80`                                         | Soglia di attivazione dell'espansione dal dizionario concettuale.                                   |
| `qdrant_concept_max_hits`   | `QDRANT_CONCEPT_MAX_HITS`   | `int`      | `3`                                            | Concetti sopra soglia considerati (non solo il migliore).                                           |
| `rerank_size`               | `RERANK_SIZE`               | `int`      | `25`                                           | Chunk sopravvissuti al rerank per sotto-query.                                                      |
| `rerank_truncate`           | `RERANK_TRUNCATE`           | `int`      | `1200`                                         | Caratteri massimi per chunk passati al Cross-Encoder.                                               |
| `rerank_batch_size`         | `RERANK_BATCH_SIZE`         | `int`      | `4`                                            | Ampiezza del batch di inferenza.                                                                    |
| `rerank_max_length`         | `RERANK_MAX_LENGTH`         | `int`      | `512`                                          | Token massimi del tokenizer del Reranker.                                                           |
| `min_prob_threshold`        | `MIN_PROB_THRESHOLD`        | `float`    | `0.02`                                         | Soglia sigmoidea sui logit; almeno 2 candidati sono sempre mantenuti.                               |
| `onnx_model_cache_path`     | `RERANKER_MODEL_PATH`       | `str`      | `"./model_cache/mmarco-mMiniLMv2-L12-H384-v1"` | Percorso di pesi e tokenizer del modello ONNX.                                                      |
| `parents_per_query`         | `PARENTS_PER_QUERY`         | `int`      | `4`                                            | Quota garantita di Parent per sotto-query.                                                          |
| `mmr_similarity_threshold`  | `MMR_SIMILARITY_THRESHOLD`  | `float`    | `0.92`                                         | Similarità oltre la quale MMR penalizza un candidato ridondante (attivo solo con `mmr_lambda < 1`). |
| `allow_subtopic_selection`  | `ALLOW_SUBTOPIC_SELECTION`  | `bool`     | `True`                                         | Abilita la selezione dei sotto-argomenti nel flusso d'interrogazione.                               |
| `protected_keys`            | _N/A_                       | `set[str]` | costante                                       | Metadati riservati (`parent_id`, `topic_id`, …) protetti da sovrascrittura in ingestion.            |

### E. Budget di Contesto

| Proprietà                | Variabile d'Ambiente     | Tipo    | Default                    | Impatto                                                                                                                                                                                          |
| ------------------------ | ------------------------ | :------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `max_context_chars`      | `MAX_CONTEXT_CHARS`      | `int`   | `30000`                    | Budget complessivo in caratteri del contesto passato al generatore.                                                                                                                              |
| `context_floor_ratio`    | `CONTEXT_FLOOR_RATIO`    | `float` | `0.60`                     | Frazione del budget divisa **equamente** tra le sotto-query. Alzarlo è la mossa corretta quando le sotto-query in coda restano sotto-servite.                                                    |
| `context_cap_ratio`      | `CONTEXT_CAP_RATIO`      | `float` | `0.40`                     | Tetto anti-monopolio: nessuna sotto-query oltre questa frazione del contesto.                                                                                                                    |
| `context_score_weight`   | `CONTEXT_SCORE_WEIGHT`   | `str`   | `"best"`                   | Criterio di pesatura per sotto-query: `best` (meno sensibile alla numerosità) \| `mean`.                                                                                                         |
| `context_overflow_ratio` | `CONTEXT_OVERFLOW_RATIO` | `float` | `0.15`                     | Sforamento tollerato per includere un documento **intero** invece di finestrarlo per pochi caratteri. Non può mai servire ad aggiungere un Parent nuovo, solo a completarne uno già selezionato. |
| `context_hard_cap_ratio` | `CONTEXT_HARD_CAP_RATIO` | `float` | `0.20`                     | Tetto invalicabile espresso come **frazione** di `MAX_CONTEXT_CHARS`. Un guardrail lo allinea a `CONTEXT_OVERFLOW_RATIO` se impostato più basso.                                                 |
| `context_hard_cap_chars` | _derivato_               | `int`   | `MAX_CONTEXT_CHARS × 1.20` | **Non leggibile da env.** Calcolato in `load_settings()`; è il valore che `build_context()` usa come rete di sicurezza finale.                                                                   |
| `context_snap_boundary`  | `CONTEXT_SNAP_BOUNDARY`  | `str`   | `"sentence"`               | Confine su cui arretrare il taglio: `sentence` \| `paragraph` \| `none`. Costo zero, nessuna espansione.                                                                                         |

### F. Risorse di Calcolo e Concorrenza

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

## 8. Funzionalità Mancanti e Sviluppi Possibili

Elenco delle capacità che il sistema oggi non offre e che rappresenterebbero un'estensione significativa. Non è un piano di lavoro: le voci sono indipendenti fra loro e ordinate per area, non per priorità.

### 8.1 Indipendenza dal fornitore di embedding

Riscrittura, generazione e grading sono già provider-agnostici (§7.C). La generazione dei vettori no: è vincolata a Google tramite una costante interna, quindi anche adottando integralmente un altro fornitore per gli LLM il sistema continua a richiedere credenziali Google.

Rendere l'embedding configurabile alla pari degli altri stadi consentirebbe di scegliere il fornitore in base a criteri propri — costo, residenza del dato, qualità sulla lingua italiana amministrativa — indipendentemente da quale modello genera le risposte. Per un ente pubblico la voce ha anche una valenza di sovranità del dato: gli embedding sono l'unica componente che richiede l'invio dell'**intero corpus** a un servizio esterno, non solo delle query.

Va tenuto presente che la dimensione del vettore è fissata alla creazione della collection Qdrant: cambiare modello comporta la re-indicizzazione completa dell'archivio. La configurabilità va quindi accompagnata da una verifica di coerenza fra modello configurato e indice esistente.

> _Nota documentale_: la dimensionalità dichiarata in §3 (768) va confermata leggendola dalla collection, poiché il modello in uso supporta dimensioni configurabili.

### 8.2 Esperienza di interrogazione

- **Risposta in streaming.** Oggi il client invia la domanda, riceve `202 Accepted` e interroga ciclicamente lo stato finché il task non è concluso: da trenta secondi a oltre un minuto di attesa senza alcun riscontro intermedio. Lo streaming incrementale dei token, o quantomeno la pubblicazione dello stato di avanzamento della pipeline ("riformulazione", "ricerca", "sintesi"), cambierebbe radicalmente la percezione del sistema a parità di tempo di calcolo.
- **Filtri espliciti per l'utente.** Il motore supporta già filtri su intervalli di date, valori esatti e liste di metadati, ma sono raggiungibili solo per via programmatica. Esporli nell'interfaccia — "solo determine del 2024", "solo atti del Settore Viabilità" — sposterebbe una capacità già presente dal codice all'utente finale.
- **Citazioni puntuali.** Le risposte citano il documento di provenienza per nome file. Poiché i frammenti conservano la posizione nel testo originale, sarebbe possibile portare l'utente **al punto esatto** del documento, evidenziando il passaggio da cui l'affermazione deriva. Per un contesto amministrativo, dove la verificabilità della fonte conta quanto la risposta, è probabilmente l'aggiunta di maggior valore percepito.
- **Continuità documentale nel dialogo.** La cronologia conversazionale trasporta solo il testo scambiato. I documenti recuperati al turno precedente non restano disponibili, quindi una domanda di approfondimento ("e cosa dice l'articolo 3 di quell'atto?") innesca una ricerca completamente nuova, che può non ripescare lo stesso documento.

### 8.3 Ricerca e qualità dei risultati

- **Fusione ibrida dei ranking.** Le ricerche semantica e lessicale coesistono, ma i risultati della seconda vengono semplicemente accodati ai candidati della query principale. Una fusione formale dei due ranking — Reciprocal Rank Fusion o equivalente — sfrutterebbe la complementarità dei due segnali invece di privilegiare arbitrariamente uno dei due percorsi.
- **Ricerca esatta su identificatori.** Domande su CIG, numeri di determina, protocolli e date sono ricerche di **identificatori**, non di concetti, e la similarità vettoriale è lo strumento sbagliato: un codice alfanumerico non ha vicini semantici significativi. Un percorso dedicato che estragga in ingestione i metadati strutturati e li interroghi per corrispondenza esatta darebbe risposte deterministiche dove oggi si affida alla fortuna del retrieval.
- **Deduplica dei contenuti quasi identici.** Il corpus amministrativo contiene ampie porzioni ricorrenti (formule di rito, riferimenti normativi, clausole standard). Capita che più frammenti pressoché identici occupino le prime posizioni del riordinamento, consumando budget di contesto senza aggiungere informazione. Una deduplica sui contenuti — non solo sull'identità esatta, già gestita — libererebbe spazio per materiale complementare.
- **Ricerca su tabelle e allegati strutturati.** Gli atti amministrativi contengono quadri economici, elenchi di interventi e piani particellari. La conversione in Markdown ne preserva l'aspetto ma non la struttura: una domanda che richieda di aggregare valori attraverso più tabelle non è oggi servibile in modo affidabile.

### 8.4 Valutazione e miglioramento continuo

- **Insieme di riferimento per la valutazione.** Il comportamento del sistema dipende da una decina di soglie numeriche (similarità, quote di contesto, probabilità minima, diversificazione). Senza un insieme di domande con i documenti attesi, ogni taratura è una scommessa: non è possibile sapere se una modifica migliora il sistema o sposta soltanto il problema. È la funzionalità mancante con il maggior effetto abilitante su tutte le altre.
- **Sfruttamento attivo del feedback.** I giudizi degli utenti e le sessioni fallite sono registrati, ma restano materiale di sola consultazione. Potrebbero alimentare un insieme di risposte verificate da preferire, o segnalare automaticamente le domande ricorrenti che il sistema non sa servire.
- **Analisi delle lacune di archivio.** Il valutatore registra, per ogni risposta, le entità richieste su cui non ha trovato nulla. Un'entità che ricorre sistematicamente in quell'elenco indica un buco di **ingestione**, non di ricerca. Un rapporto periodico su quel dato trasformerebbe un'informazione già raccolta in una segnalazione azionabile verso chi cura il corpus.
- **Cruscotto operativo.** Le metriche per richiesta sono persistite ma non presentate: tempi per stadio, tasso di riuso della cache, tentativi medi, frequenza dei troncamenti. Renderle visibili permetterebbe di accorgersi di un degrado prima che lo segnali un utente.

### 8.5 Esercizio, sicurezza e conformità

- **Verifiche di stato del servizio.** Il sistema non espone alcun endpoint o comando che attesti la propria operatività: né l'API Gateway né il worker dichiarano di aver completato l'inizializzazione e di essere in grado di servire. Qualunque supervisore — l'`HEALTHCHECK` di Docker, le sonde di liveness e readiness di Kubernetes, il monitoraggio di un servizio di sistema — può quindi solo constatare che il processo esiste, non che funzioni. Un controllo che verifichi la raggiungibilità di database e indice vettoriale e l'avvenuto caricamento del riordinatore renderebbe visibili all'esterno i malfunzionamenti che oggi restano confinati ai log.
- **Controllo degli accessi per documento.** L'API è protetta da un unico token condiviso: chiunque possa interrogare il sistema vede l'intero corpus indicizzato. La segmentazione per argomento e sotto-argomento esiste già come struttura, ma non è legata a un'identità. Per un ente con atti a diversa visibilità è probabilmente il limite più rilevante in vista di un'apertura a una platea più ampia.
- **Tracciabilità delle risposte.** In ambito amministrativo può essere necessario ricostruire a posteriori che cosa il sistema ha risposto, a chi, sulla base di quali documenti e con quale versione dei prompt. Gli elementi sono in parte già registrati, ma non in una forma pensata per essere consultata come registro.
- **Espurgo e diritto all'oblio.** Non esiste un percorso per rimuovere selettivamente un documento e tutte le sue tracce — frammenti vettoriali, documenti padre, voci di cache semantica che potrebbero riprodurne il contenuto.

### 8.6 Ottimizzazioni note

Interventi che non aggiungono funzionalità ma riducono tempi o costi, elencati per completezza:

- **Avvio del riordinatore**: circa sette secondi per processo, riducibili sotto il secondo predisponendo gli artefatti del modello in forma già ottimizzata.
- **Modello quantizzato**: la variante a interi dimezzerebbe l'occupazione di memoria per istanza e accelererebbe l'inferenza, previa verifica della perdita di qualità.
- **Chiamate di embedding**: tre viaggi di rete sequenziali per richiesta, accorpabili.
- **Trasporto verso il database vettoriale**: il protocollo binario ridurrebbe la latenza sui profili che trasferiscono vettori.

### 8.7 Parità tra fornitori

L'adapter normalizza le differenze fra SDK, ma la copertura non è uniforme. Da considerare prima di un cambio di fornitore:

| Funzionalità                          |       Gemini        |   OpenAI    |          Anthropic           |
| ------------------------------------- | :-----------------: | :---------: | :--------------------------: |
| Modalità JSON nativa                  |         ✅          |     ✅      | ⚠️ emulata via system prompt |
| Livello di ragionamento configurabile |         ✅          | ❌ ignorato |         ❌ ignorato          |
| Tetto di output omettibile            |         ✅          |     ✅      |       ❌ obbligatorio        |
| Conteggio dei token di ragionamento   |         ✅          |     ❌      |              ❌              |
| Embedding                             | ✅ unico supportato |      —      |              —               |

La diagnostica del troncamento descritta in §5.2 distingue le cause grazie ai token di ragionamento, disponibili solo su Gemini: sugli altri fornitori degrada a una segnalazione generica.

## 9. Manutenzione, Monitoraggio e Scheduled Task

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
- **Upgrade Infrastrutturale**: i miglioramenti di velocità sotto carico massivo richiederanno hardware dedicato (GPU/TPU) per svincolare l'Ente dalle code di elaborazione condivise.

## 10. Requisiti di Sistema e Dipendenze Python (`requirements.txt`)

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
