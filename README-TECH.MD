
# Specifica Tecnica ed Architetturale: Sistema RAG Enterprise-Grade per l'Ente Locale

Questo documento descrive in modo formale e approfondito l'architettura, le strategie algoritmiche, i flussi di dati, lo schema del database, la mappatura di configurazione e i requisiti di sistema del modello **Retrieval-Augmented Generation (RAG)** avanzato sviluppato per l'Ente Locale. Il sistema adotta un approccio asincrono, resiliente, guidato dagli eventi (*Event-Driven*) e strutturato su una filiera multi-stadio protetta.

## 1. Introduzione e Obiettivi dell'Architettura

I sistemi RAG tradizionali (*Naïve RAG*) soffrono di limitazioni strutturali e algoritmiche che ne precludono l'adozione in scenari della Pubblica Amministrazione regolati da rigidi vincoli di conformità amministrativa e precisione:

* **Perdita di precisione semantica:** Blocchi di testo (*chunk*) troppo ampi diluiscono il segnale semantico dei vettori di embedding, portando a mancate corrispondenze.
* **Mancanza di contesto strutturale:** I documenti vengono frammentati in piccoli paragrafi isolati. Di conseguenza, il modello generativo (LLM) ha una visione parziale: vede solo le singole tessere del puzzle, perdendo il contesto logico-normativo circostante come preamboli d'atto, visti istituzionali o articoli di legge correlati.
* **Instabilità delle API esterne ed allucinazioni:** Fluttuazioni nelle quote delle API esterne e tendenza all'allucinazione in mancanza di vincoli tassativi di grounding documentale.

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
[Normalizzazione .md] ◄── [Gemini Vision OCR / Google Vision] ◄── Estrazione Ibrida     (converter.py)
    │
    ▼  (RabbitMQ: da-indicizzare)
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
* Esegue una scansione ricorsiva della directory `INPUT_DIR` (`data/direct`) per rilevare file `.pdf`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`.
* Preserva la struttura delle cartelle, identificando la directory padre come `topic_id` e la sottodirectory come `sub_topic_id`.
* Muove il file nativo in `data/staging` e genera un file manifest JSON sidecar contenente un identificativo URI logico come chiave `source` (es. `direct://percorso/nomefile`).
* Notifica RabbitMQ inviando un messaggio persistente sulla coda `da-convertire`.

### Fase B: Estrazione Ibrida, OCR Avanzato e Fallback Dinamico (`converter.py`)
Il modulo `converter.py` consuma il messaggio e avvia la normalizzazione testuale tramite un approccio asincrono guidato da `AsyncLimiter` (limite di 10 chiamate/minuto per Gemini):
1. **Analisi della Densità Informativa (PDF)**:
   * Estrae il testo di ogni pagina tramite PyMuPDF (`fitz`), applicando una maschera di ritaglio basata su `CROP_OCR_LIMIT` per escludere intestazioni e piè di pagina fissi.
   * Se il numero di caratteri nativi estratti da una pagina è inferiore a 50, la pagina viene catalogata come **scansionata** o **immagine** ed inviata alla pipeline OCR. Se è superiore, viene convertita direttamente in Markdown strutturato tramite `pymupdf4llm`.
2. **Generazione OCR tramite Gemini Vision**:
   * La pagina scansionata o l'immagine nativa viene passata a Gemini Vision (`OCR_MODEL_NAME`) per una trascrizione accurata forzata in formato Markdown.
3. **Bypass dei Filtri di Sicurezza (Google Cloud Vision Fallback)**:
   * Se la chiamata API a Gemini fallisce o viene bloccata a causa delle rigide policy di filtraggio (es. flag `SAFETY`, `RECITATION` o `BLOCKLIST`), il sistema cattura l'eccezione `ValueError` o `APIError`.
   * Per evitare l'arresto della catena di elaborazione (bloccante per documenti amministrativi contenenti falsi positivi), il sistema avvia un **fallback sincrono e immediato su Google Cloud Vision API** (`document_text_detection`).
4. **Scrittura in Watch**:
   * I flussi testuali derivati vengono fusi e scritti come file `.md` abbinati a un nuovo JSON manifesto figlio nella cartella `data/watch`. Il sistema pubblica un evento persistente sulla coda `da-indicizzare`.

### Fase C: Frammentazione Gerarchica e Upsert Atomico Idempotente (`ingest.py`)
Il worker di ingestion consuma l'evento sulla coda `da-indicizzare`, applicando logiche di transazionalità e protezione della memoria RAM:
1. **Risoluzione dei Parametri dal DB**:
   * Recupera i parametri di chunking interrogando la tabella MySQL `sub_topics`. Se non presenti, applica i default di sistema (`chunk_size = 500`, `chunk_overlap = 100`, `parent_chunk_size = 1500`, `use_markdown_splitter = True`).
2. **Splitting Gerarchico Parent-Child**:
   * **Parent Splitting**: Se abilitato, divide il documento Markdown tramite `MarkdownHeaderTextSplitter` basandosi sugli header logici (`#` fino a `####`), catturando la gerarchia nei metadati. Successivamente, utilizza `RecursiveCharacterTextSplitter` con il tokenizer di Tiktoken (`cl100k_base`) per generare blocchi ampi coerenti (Parent).
   * **Child Splitting**: Ciascun blocco Parent viene frammentato in blocchi Child più piccoli e sovrapposti (`chunk_size` e `chunk_overlap`) per consentire una ricerca semantica millimetrica.
3. **Idempotenza Transazionale MySQL**:
   * Per garantire che la re-ingestione di un documento modificato non lasci record duplicati, il worker esegue un'operazione atomica in MySQL: prima esegue una `DELETE` su `parent_documents` basata sulla chiave `source`, `topic_id` e `sub_topic_id`, poi esegue una `executemany` per inserire massivamente i nuovi Parent Documents.
4. **Generazione Embedding in Batch ed Ingestione Qdrant**:
   * I blocchi Child vengono inviati in batch asincroni alle API Google GenAI per generare vettori a 768 dimensions via `gemini-embedding-001`. Il processo è protetto da un `AsyncLimiter` impostato a 1000 richieste/minuto e gestisce gli errori `ResourceExhausted` (429) e `ServiceUnavailable` (503) tramite retry con backoff esponenziale.
5. **Prevenzione OOM (Batch Flush su Qdrant)**:
   * Al fine di proteggere la memoria RAM durante l'ingestione di file Markdown di grandissime dimensioni, i punti Child vengono assemblati e accumulati in un buffer. Non appena il buffer supera i 500 punti, viene eseguito un *flush intermedio* verso Qdrant (`qdrant_client.upsert`), svuotando la memoria prima di procedere.
6. **Qdrant Garbage Collection**:
   * Per rimuovere i "vettori fantasma" (i vecchi frammenti Child rimasti nel Vector DB dopo che il file originale è stato aggiornato), il sistema esegue una query di cancellazione mirata (`qdrant_client.delete`), rimuovendo tutti i punti associati allo stesso `source` ma dotati di un `_ingestion_id` obsoleto (diverso dall'ID univoco di sessione appena generato).

## 4. Motore di Retrieval Avanzato & Reranking

Il processo di recupero e ordinamento dei contesti all'interno di `worker.py` supera i limiti della semplice ricerca semantica (K-Nearest Neighbors) implementando una catena di elaborazione avanzata.

### A. Trasformazione delle Query ed Espansione Semantica
1. **Riscrittura Conversazionale**: Il modello `QUERY_REWRITER_MODEL_NAME` (`gemini-3.1-flash-lite`) si comporta come un agente intelligente dotato di pianificazione : analizza fino a 20 messaggi di cronologia, scompone richieste articolate in sotto-quesiti distinti (`search_queries` fino a 5) e mappa parole chiave lessicali (`keywords`).
2. **Lookup Semantico del Concetto**:
   * Il sistema esegue l'embedding della query standalone e interroga una collezione speciale in Qdrant denominata `conceptual_dictionary`.
   * Se la query ha un'affinità semantica superiore a `qdrant_concept_threshold` (`0.80`), estrae un array di **alias e sinonimi della Pubblica Amministrazione** (es. associando "tassa rifiuti" a "TARI" , o macro-aree geografiche come "Garfagnana" o "Mediavalle" ai singoli comuni componenti ).
   * Se vengono rilevati alias, viene invocata la funzione `expand_standalone_with_aliases` per rigenerare e potenziare le sotto-query di ricerca, garantendo che gerghi territoriali o normativi non esplicitati vengano catturati.

### B. Ricerca Ibrida Parallela e Reranking Locale via ONNX Cross-Encoder
Per massimizzare il rendimento ed evitare colli di bottiglia, la ricerca dei frammenti candidati avviene in parallelo sfruttando un pool di thread concorrenti (`ThreadPoolExecutor`):
1. **Ricerca Semantica**: Esegue query vettoriali parallele su Qdrant (`document_chunks`) per ciascuna sotto-query generata, applicando soglie di confidenza minime (`qdrant_semantic_threshold = 0.60`) e filtrando per i `topic_id` e `sub_topic_id` attivi.
2. **Ricerca Sintattica**: In parallelo alle query vettoriali, esegue una ricerca full-text lessicale basata sulle keyword estratte, sfruttando filtri logici `MatchText` su Qdrant per raccogliere corrispondenze esatte (codici, date, identificativi d'atto).
3. **Lazy Loading del Reranker ONNX (Deadlock Protection)**:
   * Per evitare i classici blocchi dei socket TCP e i deadlock causati dalle chiamate parallele multi-processo di Celery, il modello locale di Rerank Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`) viene caricato in modalità "Lazy Load" solo quando invocato all'interno del processo attivo del worker.
   * Il parallelismo interno di ONNX Runtime viene disattivato forzatamente per garantire la sicurezza del thread:
     ```python
     sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
     sess_options.intra_op_num_threads = num_threads
     sess_options.inter_op_num_threads = 1
     ```
   * Il sistema calcola il punteggio sigmoideo assecondando la formula del modello:
     $$P = \sigma(x) = \frac{1}{1 + e^{-x}}$$
     I chunk con una probabilità inferiore a `min_prob_threshold` (`0.02`) vengono immediatamente scartati dal pool dei risultati.

### C. Algoritmo di Ripartizione delle Quote e Redistribuzione Attiva
In presenza di forti asimmetrie nei dati (es. se alcuni uffici non hanno prodotto documentazione recente su un tema) , il sistema applica un'ottimizzazione ingegneristica per redistribuire in totale profondità la capacità di analisi sull'unico ufficio loquace:
1. **Quota Garantita per Sotto-query**: Ciascuna sotto-query ha diritto a una quota fissa di documenti Parent associati ai suoi migliori Child vettoriali, definita dal parametro `parents_per_query` (default 4).
2. **Allocazione Iniziale**: Il sistema scorre la lista riordinata dei risultati e assegna a ciascuna query i suoi primi 4 documenti Parent univoci.
3. **Calcolo degli Slot Liberi**:
   * Se alcune sotto-query non producono risultati, il sistema calcola la capacità residua non sfruttata per evitare risposte povere:
     $$\text{Slot Liberi} = (\text{parents\_per\_query} \times N_{\text{queries}}) - N_{\text{selezionati}}$$
   * Questi slot rimasti liberi vengono ridistribuiti in modalità **round-robin** a favore delle sotto-query che possiedono una riserva di documenti validi scartati dall'allocazione iniziale.
   * Una volta identificati i frammenti ideali, il sistema risale al "Documento Padre" (l'intero articolo o atto ufficiale)  per servire all'LLM un quadro informativo coerente privo di decontestualizzazioni. Il budget totale è protetto dal parametro `max_context_chars = 30000` caratteri.

## 5. Generazione con Autovalutazione & Ciclo di Autocorrezione (Self-Correction Loop)

Una volta ricomposto il contesto mediante il recupero dei Parent Documents da MySQL, il sistema avvia un loop di generazione controllata e auto-correzione a più livelli:

1. **Generazione Strutturata (JSON Mode)**:
   * Il modello `ANSWER_GENERATOR_MODEL_NAME` (`gemini-3.5-flash`) genera la risposta basandosi sul template di prompt specifico del topic caricato dinamicamente da MySQL.
   * L'output viene forzato in formato JSON contenente la chiave booleana `is_found` e la stringa testuale `answer`.
2. **Parsing JSON Multi-Livello a Forza Bruta (`safe_json_parse`)**:
   La risposta strutturata viene estratta applicando una strategia difensiva a quattro livelli sequenziali:
   * **Livello 1 (Standard)**: Tentativo standard di decodifica tramite `json.loads`.
   * **Livello 2 (Markdown Strip)**: Pulizia proattiva dei delimitatori Markdown (```` ```json ... ``` ````) tramite espressioni regolari multi-linea.
   * **Livello 2b (Truncated Repair)**: Tentativo di recupero in caso di output troncato dalle API esterne aggiungendo le chiusure mancanti (`rstrip() + '"}'`).
   * **Livello 3 (Brute Force Regex)**: Ricerca bruteforce di qualsiasi contenuto compreso tra le parentesi graffe `{...}` con flag `re.DOTALL`.
3. **Logica Fail-Fast**:
   * Se il modello imposta `is_found = False` (ammettendo esplicitamente che i documenti non contengono la risposta), il sistema salta il passaggio del Grader per risparmiare risorse e tempo di calcolo, indirizzando immediatamente il flusso verso il ciclo di retry.
4. **Auditing tramite Grader LLM (Chain of Thought)**:
   * Se `is_found = True`, il modello valutatore (`gemini-3.1-flash-lite`) si attiva in una fase di verifica : esamina la query, la risposta generata e il testo completo dei contesti passati per ordinare se la risposta è accurata e direttamente verificabile.
5. **Ciclo di Retry Attivo e Riformulazione**:
   * Se il Grader boccia la risposta (rilevando lacune) , il sistema formula una nuova strategia di ricerca  incrementando l'indice di tentativo (`MAX_MODEL_RETRIES = 2`).
   * Viene iniettato nella cronologia un messaggio fittizio utente che istruisce esplicitamente il query rewriter a cambiare angolazione semantica, allargare i filtri logici e utilizzare sinonimi differenti per estrarre nuovi documenti.
   * La pipeline riesegue il retrieval e produce una nuova risposta.
6. **Caching Semantico e Auto-Logging delle Query Fallite**:
   * **In caso di successo**: La risposta viene salvata nella collezione `semantic_cache` di Qdrant. Se un altro utente pone una richiesta affine applicando i medesimi filtri, la memoria intelligente abbatte i costi operativi restituendo l'output in millisecondi.
   * **In caso di fallimento definitivo**: Il sistema salva automaticamente la sessione nella tabella `chat_feedback` di MySQL con un voto di default pari a `0` e la dicitura `"Auto-Log: Risposta non trovata o non soddisfacente"`. Questo permette agli amministratori di effettuare analisi offline e sessioni di fine-tuning dei prompt o della base documentale.

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
   * `topic_id` (VARCHAR(100) PRIMARY KEY)
   * `description` (TEXT, NOT NULL)
   * `aliases` (TEXT, NULL)
   * `prompt` (VARCHAR(100), NULL)

2. **`sub_topics`**: Configura a runtime i parametri di splitting per ciascun sotto-argomento.
   * `id` (INT AUTO_INCREMENT PRIMARY KEY)
   * `topic_id` (VARCHAR(100) NOT NULL)
   * `sub_topic_id` (VARCHAR(100) NOT NULL)
   * `description` (VARCHAR(255) NOT NULL)
   * `chunk_size` (INT, NULL)
   * `chunk_overlap` (INT, DEFAULT 50)
   * `parent_chunk_size` (INT, DEFAULT 1500)
   * `use_markdown_splitter` (TINYINT(1), DEFAULT 1)
   * *Vincoli*: `UNIQUE KEY unique_sub_topic (topic_id, sub_topic_id)`
   * *Chiave Esterna*: `CONSTRAINT fk_sub_topics_topic_id FOREIGN KEY (topic_id) REFERENCES topics (topic_id) ON DELETE CASCADE ON UPDATE CASCADE`

3. **`parent_documents`**: Contiene il testo strutturato esteso (Parent) dei documenti.
   * `id` (VARCHAR(36) PRIMARY KEY)
   * `topic_id` (VARCHAR(255) NOT NULL)
   * `sub_topic_id` (VARCHAR(255) NOT NULL)
   * `source` (VARCHAR(255) NOT NULL)
   * `file_name` (VARCHAR(255), NULL)
   * `parent_index` (INT, NULL)
   * `content` (LONGTEXT, NOT NULL)
   * `metadata` (JSON, NULL)
   * *Indici*: `INDEX idx_search (source, topic_id, sub_topic_id)` per la rimozione atomica e le query di retrieval.

4. **`chat_feedback`**: Registro per la raccolta dei feedback degli utenti e il tracciamento dei fallimenti automatici del Grader.
   * `id` (INT AUTO_INCREMENT PRIMARY KEY)
   * `topic_id` (VARCHAR(255), NULL)
   * `user_query` (TEXT, NULL)
   * `ai_response` (TEXT, NULL)
   * `rating` (INT, NULL) - *Il rating è impostato a 0 in caso di fallimento del Grader*.
   * `chat_history` (JSON, NULL)
   * `comment` (TEXT, NULL)
   * `reviewed` (TINYINT(1), DEFAULT 0)

5. **`failed_queries`**: Tracciamento delle query non gestite o non instradabili correttamente.
   * `id` (INT AUTO_INCREMENT PRIMARY KEY)
   * `standalone_query` (TEXT, NOT NULL)
   * `failure_type` (VARCHAR(50), NOT NULL)
   * `topic_id_routed` (VARCHAR(255), NULL)

6. **`system_logs`**: Auditing centralizzato delle anomalie e stacktrace dei pod.
   * `log_id` (BIGINT AUTO_INCREMENT PRIMARY KEY)
   * `log_level` (VARCHAR(15), NOT NULL)
   * `message` (TEXT, NOT NULL)
   * `file_name` (VARCHAR(255), NULL)
   * `line_no` (INT, NULL)
   * `pod_name` (VARCHAR(255), DEFAULT NULL)
   * *Indici*: `INDEX idx_log_level (log_level)`, `INDEX idx_created_at (created_at)`.

## 7. Mappatura Dettagliata delle Variabili di Configurazione (`config.py`)

La configurazione centralizzata raccoglie e convalida le variabili d'ambiente. Di seguito viene dettagliato l'esatto ruolo logico, il tipo dato e l'impatto sul comportamento dei vari script Python:

| Proprietà dell'Oggetto Settings | Variabile d'Ambiente Corrispondente | Tipo Dato | Valore Default / Fallback | Impatto Algoritmico e Interazione tra Moduli |
| --- | --- | :--- | --- | --- |
| `http_timeout_seconds` | `HTTP_TIMEOUT_SECONDS` | `int` | `300` | Definisce la soglia di blocco delle chiamate HTTP esterne verso Sicr@Web e le API LLM. |
| `data_folder` | `DATA_FOLDER` | `Path` | Percorso relativo a `config.py` | Radice locale del filesystem utilizzata per mappare `staging`, `watch`, `archive` ed `error`. |
| `log_level` | `LOG_LEVEL` | `str` | `"INFO"` | Regola la verbosità del logger centralizzato (`logging.basicConfig`). |
| `broker_host` | `BROKER_HOST` | `str` | `"rabbitmq-service.rag.svc.cluster.local"` | Endpoint dell'istanza RabbitMQ per la comunicazione asincrona. |
| `broker_port` | `BROKER_PORT` | `int` | `5672` | Porta del broker RabbitMQ. |
|| `broker_username` / `broker_password` | `BROKER_USERNAME` / `BROKER_PASSWORD` | `str` / `str` | `"guest"` / `"guest"` | Credenziali di accesso per la pubblicazione e consumo degli eventi. |
| `broker_max_attempts` | `BROKER_MAX_ATTEMPTS` | `int` | `30` | Tentativi massimi di connessione asincrona tollerati all'avvio dei pod prima del crash. |
| `docws_ricerca_endpoint` | `DOCWS_RICERCA_ENDPOINT` | `str` | **Obbligatoria** | URL SOAP di Sicr@Web per la federazione delle ricerche atti (`estrattore.py`). |
| `docws_atti_endpoint` | `DOCWS_ATTI_ENDPOINT` | `str` | **Obbligatoria** | URL SOAP di Sicr@Web per l'estrazione binaria degli atti (`estrattore.py`). |
| `docws_codice_amministrazione` / `docws_codice_aoo` | `DOCWS_CODICE_AMMINISTRAZIONE` / `DOCWS_CODICE_AOO` | `str` | **Obbligatoria** | Metadati di instradamento giuridico e classificazione dell'Ente Locale. |
| `ws_username` / `ws_password` | `WS_USERNAME` / `WS_PASSWORD` | `str` / `str` | `"test"` / `"test"` | Credenziali applicative Sicr@Web per l'estrattore. |
| `ruolo_docws` | `RUOLO_DOCWS` | `str` | **Obbligatoria** | Livello di autorizzazione formale per la lettura del protocollo. |
| `verify_tls` | `VERIFY_TLS` | `bool` | `True` | Abilita o disabilita il controllo dei certificati SSL per i server Sicr@Web. |
| `soap_version` | `SOAP_VERSION` | `str` | `"1.1"` | Versione del protocollo per la composizione della busta SOAP. |
| `db_host` | `MYSQL_SERVICE_HOST` | `str` | `"localhost"` | Endpoint del server di database relazionale MySQL. |
| `db_port` | `MYSQL_SERVICE_PORT` | `int` | `3306` | Porta del server MySQL. |
| `db_user` / `db_pass` / `db_name` | `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | `str` | `"raguser"` / `""` / `"rag_db"` | Credenziali e nome del database per la persistenza e il logging. |
| `ocr_model_name` | `OCR_MODEL_NAME` | `str` | `"gemini-3-flash-preview"` | Modello Gemini Vision dedicato alla trascrizione dei file non digitali nativi. |
| `api_llm_key` | `API_LLM_KEY` | `str` | `"default_key"` | Chiave di autenticazione per le chiamate API di Google GenAI e i vettori di embedding. |
| `api_secret_key` | `API_SECRET_KEY` | `str` | `"default_key"` | Token di sicurezza Bearer a protezione degli endpoint esposti dal gateway Flask. |
| `hostname` | `HOSTNAME` | `str` | `socket.gethostname()` | Nome identificativo dell'host, fondamentale nei log per associare l'errore al pod Kubernetes. |
| `qdrant_host` / `qdrant_port` | `QDRANT_HOST` / `QDRANT_PORT` | `str` / `int` | `"localhost"` / `6333` | Endpoint e porta del database vettoriale Qdrant. |
| `allow_subtopic_selection` | `ALLOW_SUBTOPIC_SELECTION` | `bool` | `True` | Abilita la selezione dei sotto-argomenti nel flusso d'interrogazione dell'utente. |
| `allowed_origins` | `ALLOWED_ORIGINS` | `list` | `["*"]` | Origini consentite dal modulo CORS di Flask per proteggere l'API Gateway. |
| `max_history_items` | `MAX_HISTORY_ITEMS` | `int` | `20` | Finestra massima di messaggi di cronologia passati al Query Rewriter. |
| `redis_host` / `redis_port` | `REDIS_HOST` / `REDIS_PORT` | `str` / `int` | `"localhost"` / `6379` | Endpoint e porta dell'istanza Redis, broker e backend per Celery. |
| `allow_general_knowledge` | `ALLOW_GENERAL_KNOWLEDGE` | `bool` | `True` | Se impostata su `False`, forza il sistema a scartare le risposte non ancorate ai documenti interni. |
| `rerank_size` | `RERANK_SIZE` | `int` | `25` | Finestra massima di documenti Child passati al modulo ONNX Reranker locale. |
| `rerank_truncate` | `RERANK_TRUNCATE` | `int` | `1200` | Lunghezza massima in caratteri di ciascun chunk passato al Cross-Encoder per evitare OOM. |
| `rerank_batch_size` | `RERANK_BATCH_SIZE` | `int` | `4` | Ampiezza della finestra di inferenza parallela del Cross-Encoder locale. |
| `rerank_max_length` | `RERANK_MAX_LENGTH` | `int` | `512` | Token massimi gestiti dal tokenizer del Reranker. |
| `answer_max_tokens` | `ANSWER_MAX_TOKENS` | `int` | `4096` | Lunghezza massima dell'output generato dall'LLM per la risposta finale. |
| `qdrant_syntactic_size` | `QDRANT_SYNTACTIC_SIZE` | `int` | `20` | Numero massimo di candidati estratti dalla ricerca full-text per keyword. |
| `qdrant_semantic_size` | `QDRANT_SEMANTIC_SIZE` | `int` | `30` | Numero massimo di candidati estratti dalla ricerca vettoriale. |
| `qdrant_semantic_threshold` | `QDRANT_SEMANTIC_THRESHOLD` | `float` | `0.60` | Distanza minima vettoriale sotto la quale i Child vengono esclusi. |
| `qdrant_concept_threshold` | `QDRANT_CONCEPT_THRESHOLD` | `float` | `0.80` | Soglia di attivazione per l'espansione dei sinonimi dal dizionario concettuale. |
| `max_context_chars` | `MAX_CONTEXT_CHARS` | `int` | `30000` | Budget massimo in caratteri del contesto complessivo passato all'LLM. |
| `max_sub_queries` | `MAX_SUB_QUERIES` | `int` | `5` | Numero massimo di sotto-query generate dal Query Rewriter. |
| `parent_budget_ratio` | `PARENT_BUDGET_RATIO` | `float` | `0.80` | Coefficiente di protezione per il contenimento preventivo del testo dei parent. |
| `onnx_model_cache_path` | `RERANKER_MODEL_PATH` | `str` | `"./model_cache/mmarco..."` | Percorso locale sul disco dove risiedono i pesi e il tokenizer del modello ONNX. |
| `min_prob_threshold` | `MIN_PROB_THRESHOLD` | `float` | `0.02` | Soglia di accetazione sigmoidea calcolata sui logit del Cross-Encoder. |
| `parents_per_query` | `PARENTS_PER_QUERY` | `int` | `4` | Quota iniziale garantita di Parent Documents assegnata a ciascuna sotto-query. |
| `query_rewriter_model_name` | `QUERY_REWRITER_MODEL_NAME` | `str` | `"gemini-3.1-flash-lite"` | Modello preposto alla de-contestualizzazione e scomposizione delle query. |
| `answer_generator_model_name` | `ANSWER_GENERATOR_MODEL_NAME` | `str` | `"gemini-3.5-flash"` | Modello preposto alla sintesi finale e composizione strutturata. |
| `grader_model_name` | `GRADER_MODEL_NAME` | `str` | `"gemini-3.1-flash-lite"` | Modello valutatore incaricato dell'auditing e del grading delle risposte. |
| `protected_keys` | *N/A* | `set` | Metadati interni protetti | Set statico di metadati riservati (es. `parent_id`, `topic_id`) protetti da sovrascrittura. |
| `max_reranker_thread` | `MAX_RERANKER_THREAD` | `int` | `2` | Numero di core CPU dedicati all'inferenza sequenziale di ONNX Runtime. |
| `answer_thinking_level` | `ANSWER_THINKING_LEVEL` | `str` | `None` | Livello di computazione e ragionamento interno (*thinking_level*) per i modelli logici. |
| `reranker_pool_size` | `RERANKER_POOL_SIZE` | `int` | `1` | Numero di istanze ONNX allocate concorrentemente in memoria. |


## 8. Manutenzione, Monitoraggio e Scheduled Task

* **Logging Centralizzato su Database (`system_logs`)**:
  Il sistema implementa un handler di logging personalizzato (`MySQLLogHandler`) integrato con il modulo standard di Python. Tutte le anomalie con livello pari o superiore a `WARNING` (compresi gli stacktrace completi e i dettagli del file e della riga di codice) vengono intercettate ed inserite a database. Il recupero delle connessioni dal pool è protetto da un ping preventivo (`conn.ping(reconnect=True)`) che previene la perdita dei log in caso di timeout della connessione MySQL.
* **CronJob hourly di pulizia cache semantica (`prune_semantic_cache`)**:
  Per prevenire l'obsolescenza dei contesti aziendali e normativi, un task pianificato periodico di Celery Beat (`prune_semantic_cache`) viene eseguito ogni ora:
  * Invia una richiesta di rimozione vettoriale a Qdrant, eliminando tutti i record salvati all'interno della collezione `semantic_cache` il cui timestamp è superiore a `MAX_AGE_SECONDS` (24 ore).
* **Robusto meccanismo di retry delle API con Jitter casuale**:
  Tutte le interazioni esterne con le API LLM, i vettori di embedding e il database vettoriale sono avvolte da politiche di resilienza regolate tramite la libreria `tenacity`. In caso di errori di rate-limit (codice HTTP 429) o indisponibilità temporanea del servizio (codice HTTP 503), il sistema applica un algoritmo di **backoff esponenziale con jitter casuale** (fino a 20 tentativi per le API ad alta frequenza, con attese variabili tra 10s e 80s). Questo impedisce l'insorgenza di fenomeni di risonanza armonica e protegge i pod da crash bloccanti.
* **Upgrade Infrastrutturale**: I miglioramenti futuri relativi alla velocità sotto carico massivo richiederanno l'acquisto di hardware dedicato (GPU/TPU) per svincolare l'Ente dalle code di elaborazione condivise.


## 9. Requisiti di Sistema e Dipendenze Python (`requirements.txt`)

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
qdrant-client>=1.7.0

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
* `pika`: Consente l'invio sincrono dei messaggi di notifica degli atti estratti alla coda `da-convertire` di RabbitMQ.
* `asn1crypto`: Fornisce il parser nativo in RAM per decodificare le firme digitali in formato PKCS#7 / CAdES (`.p7m`), consentendo l'isolamento del PDF nativo senza appoggiarsi a tool di sistema esterni.


2. **Convertitore Documentale (`converter.py`)**:
* `PyMuPDF (fitz)` & `pymupdf4llm`: Consentono l'estrazione ad altissime prestazioni del testo nativo dai documenti PDF dell'Ente e la conversione strutturata in Markdown pulito.
* `pillow (PIL)`: Utilizzato per manipolare e inviare i buffer delle immagini estratte alle API di OCR.
* `google-cloud-vision`: Gestisce il fallback sincrono per l'estrazione visiva in caso di blocco di Gemini a causa delle policy di sicurezza.
* `aio-pika`: Consente il consumo asincrono dei messaggi e la notifica simultanea al modulo di ingestion.
* `aiolimiter`: Limita e regola la frequenza di chiamata delle API di Gemini Vision per evitare errori di timeout o saturazione delle quote.


3. **Ingestione Dati (`ingest.py`)**:
* `langchain-text-splitters`: Fornisce le classi `MarkdownHeaderTextSplitter` e `RecursiveCharacterTextSplitter` per lo splitting gerarchico Parent-Child.
* `qdrant-client`: Gestisce la connessione asincrona ad alta velocità verso il database vettoriale Qdrant for il caricamento dei Child Chunks.
* `typeguard`: Richiesto dal nuovo SDK di Google GenAI per la validazione dinamica dei tipi di dati passati in chiamata.


4. **Motore RAG e Reranking Locale (`worker.py` e `reranker.py`)**:
* `celery` & `redis`: Gestiscono la ricezione asincrona, la prioritizzazione e il tracciamento dei task della coda RAG.
* `onnxruntime`: Consente l'esecuzione efficiente e multi-thread limitata del modello Cross-Encoder locale, eliminando la necessità di GPU per il riordinamento di 25 candidati.
* `transformers`: Fornisce la classe `AutoTokenizer` compatibile con i modelli HuggingFace per la tokenizzazione locale della query e dei chunk.
* `mysql-connector-python`: Gestisce la persistenza e il retrieval dei Parent Documents e la scrittura asincrona dei log di sistema.
* `tenacity`: Fornisce i decoratori per implementare politiche difensive di retry con jitter casuale su tutte le chiamate esterne instabili.
