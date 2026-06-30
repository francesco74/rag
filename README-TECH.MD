# Specifica Tecnica ed Architetturale: Sistema RAG Enterprise-Grade

Questo documento descrive in modo formale e approfondito l'architettura, le strategie algoritmiche, i flussi di dati e i requisiti tecnologici del sistema **Retrieval-Augmented Generation (RAG)** avanzato implementato. Il sistema adotta un approccio asincrono, resiliente e guidato dagli eventi (*Event-Driven*) e si avvale di un frontend cross-platform orientato all'accessibilità, per garantire un'esperienza d'uso priva di barriere e rigorosamente allineata agli standard moderni.

## 1. Introduzione e Obiettivi dell'Architettura

I sistemi RAG tradizionali (*Naïve RAG*) soffrono di limitazioni intrinseche che ne precludono l'adozione in scenari aziendali regolati:
* **Perdita di precisione semantica:** Blocchi di testo (chunk) troppo ampi diluiscono il segnale semantico dei vettori di embedding.
* **Mancanza di contesto strutturale:** Blocchi di testo troppo piccoli privano l'LLM del contesto logico-narrativo circostante.
* **Instabilità delle API ed allucinazioni:** Fluttuazioni nelle quote di API esterne e generazione di informazioni plausibili ma non presenti nei dati sorgente.

Questo sistema risolve tali sfide combinando un'architettura dati gerarchica **Parent-Child**, una pipeline di ingestion asincrona basata su **RabbitMQ** e un motore di esecuzione con **Self-Correction Loop** (autocorrezione via Grader), **Reranking Cross-Encoder locale**, **Ripartizione Dinamica delle Quote** delle sotto-interrogazioni e un frontend asincrono reattivo.

## 2. Architettura di Sistema & Topologia dei Servizi

Il sistema si basa su una topologia a microservizi cooperanti e debolmente accoppiati (Event-Driven), strutturata per eliminare colli di bottiglia e garantire il disaccoppiamento dei carichi di lavoro computazionali.

```
              ┌─────────────────────────────────────────┐
              │          Wrapper Diretti               │
              │   (direct.py - Scan di data/direct)     │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼ (RabbitMQ: "da-convertire")
              ┌─────────────────────────────────────────┐
              │       Convertitore Documentale          │
              │   (converter.py - Estrazione & OCR)     │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼ (RabbitMQ: "da-indicizzare")
              ┌─────────────────────────────────────────┐
              │            Worker Ingest                │
              │ (ingest.py - Parent-Child Splitting)    │
              └─────────┬──────────────────────┬────────┘
                        │                      │
                        ▼ (MySQL)              ▼ (Qdrant)
              ┌──────────────────┐   ┌──────────────────┐
              │ parent_documents │   │ document_chunks  │
              └─────────▲────────┘   └─────────▲────────┘
                        │                      │

┌─────────────────┐        │                      │
│   API Gateway   ├────────┼──────────────────────┼─────────────────┐
│   (app.py)      │        │                      │                 │
└────────┬────────┘        │                      │                 │
│                 │                      │                 │
▼ (Celery Queue)  │                      │                 │
┌─────────────────┐        │                      │                 │
│   Celery RAG    ├────────┴──────────────────────┴─────────────────┘
│   (worker.py)   │ ──► [Rewriter ──► Retrieval ──► Reranker ──► Grader ──► Cache]
└─────────────────┘

```

### I Servizi Principali:
1. **API Gateway (`app.py` - Flask)**: Espone le interfacce REST protette tramite Bearer Token. Valida le richieste di chat, raccoglie i feedback strutturati, interroga dinamicamente le configurazioni dei sotto-argomenti ed effettua l'offloading asincrono dei task di ricerca e generazione sulla coda Celery, gestendo i tempi di risposta tramite polling client-side.
2. **Celery Worker (`worker.py`)**: Il motore di calcolo distribuito RAG. Gestisce la trasformazione semantica delle query, la ricerca ibrida, il riordinamento locale, la generazione strutturata e l'auditing qualitativo della risposta.
3. **Wrapper Diretti (`direct.py` - Daemon / CronJob)**: Scansiona le cartelle di arrivo dei documenti crudi, preserva la struttura dei topic, genera i manifesti strutturati in JSON e pubblica gli eventi di conversione su RabbitMQ.
4. **Convertitore Documentale (`converter.py` - Worker)**: Elabora asincronamente i PDF e le immagini. Esegue l'estrazione ibrida del testo e converte i formati complessi in file Markdown puliti e normalizzati.
5. **Worker Ingest (`ingest.py` - Worker)**: Frammenta i documenti Markdown normalizzati in strutture gerarchiche (Parent-Child), genera gli embedding tramite API esterne e popola in modo transazionale MySQL e Qdrant.
6. **Client SPA (`main.dart` - Flutter Web/Desktop)**: Applicazione client-side che interagisce asincronamente con l'API Gateway, implementando logiche di ottimizzazione UX, accessibilità aumentata e rendering strutturato del markup.


## 3. Pipeline di Ingestion Documentale (End-to-End)

Il viaggio di un documento, dalla sua immissione come file grezzo fino alla persistenza indiciata, è suddiviso in tre fasi rigidamente orchestrate:


```

[Inbound File] ──► (direct.py) ──► Generazione Manifest (.json) e Staging
│
▼ (RabbitMQ: da-convertire)
(converter.py) ──► Analisi Densità ──► [Testo Nativo / Gemini OCR / Vision Fallback] ──► Markdown
│
▼ (RabbitMQ: da-indicizzare)
(ingest.py) ──► Splitting Gerarchico ──► [Parent / Child] ──► Embedding Batch ──► Upsert Atomico

```

### Fase A: Rilevamento, Wrapping e Staging (`direct.py`)
* Il sistema esegue una scansione ricorsiva della directory `INPUT_DIR` (`data/direct`) per rilevare file `.pdf`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`.
* Mantiene intatta l'alberatura delle sottocartelle (che rappresentano logicamente la classification in `topic_id` e `sub_topic_id`).
* Muove il file nativo in `data/staging` e genera un manifesto JSON strutturato avente come chiave `source` l'URI logico del file (es. `direct://percorso/nomefile`).
* Notifica RabbitMQ inviando un messaggio persistente sulla coda `"da-convertire"`.

### Fase B: Estrazione Ibrida, OCR Avanzato e Fallback Dinamico (`converter.py`)
Il modulo `converter.py` consuma il messaggio e attiva una pipeline di normalizzazione testuale:
1. **Analisi della Densità Informativa (PDF)**:
   * Per i file PDF, estrae il testo di ogni pagina tramite `PyMuPDF` (filtrando margini e intestazioni non desiderate tramite un rettangolo di ritaglio configurato da `CROP_OCR_LIMIT`).
   * Se i caratteri estratti da una pagina sono inferiori a un limite critico (default 50 caratteri), la pagina viene catalogata come **scansionata** o a **bassa densità**.
2. **Generazione OCR tramite Gemini con Fallback Multi-Livello**:
   * Le pagine scansionate e le immagini native vengono convertite in un buffer PIL e passate a Gemini Vision (`gemini-3-flash-preview` o analogo definito in `OCR_MODEL_NAME`) con un prompt specifico che impone l'output in formato Markdown strutturato.
   * **Bypass dei Filtri di Safety**: Se la chiamata fallisce o viene bloccata a causa delle rigide policy di filtraggio contenuti di Google (es. flag `SAFETY`, `RECITATION` o `BLOCKLIST`), il sistema cattura l'eccezione e attiva istantaneamente un **fallback nativo e asincrono su Google Cloud Vision API** (`document_text_detection`). Questo impedisce l'arresto della catena di elaborazione per documenti che contengono falsi positivi per le policy di sicurezza LLM.
3. **Scrittura in Watch**:
   * I flussi testuali derivati (nativi o OCR) vengono fusi e scritti come file `.md` associati a un nuovo JSON manifesto figlio nella cartella `data/watch`. Il sistema pubblica un evento persistente sulla coda `"da-indicizzare"`.

### Fase C: Frammentazione Gerarchica e Upsert Atomico Idempotente (`ingest.py`)
Il worker d'ingestion si occupa di dividere fisicamente il documento Markdown e inviarlo alle memorie fisiche:
1. **Risoluzione dei Parametri di Chunking**:
   * Recupera i parametri specifici per il sotto-argomento interrogando la tabella MySQL `sub_topics`. Se non configurati, applica i valori di sistema predefiniti (es. `chunk_size = 500`, `chunk_overlap = 100`, `parent_chunk_size = 1500`, `use_markdown_splitter = True`).
2. **Splitting Gerarchico Parent-Child**:
   * **Parent Splitting**: Se abilitato, applica prima `MarkdownHeaderTextSplitter` per dividere il testo basandosi sugli header Markdown (da `#` fino a `####`), catturando la gerarchia del documento nei metadati. Successivamente, utilizza `RecursiveCharacterTextSplitter` con il tokenizer di Tiktoken (`cl100k_base`) per produrre blocchi coerenti di grandi dimensioni (Parent).
   * **Child Splitting**: Ciascun blocco Parent viene ulteriormente frammentato in blocchi Child più piccoli e sovrapposti.
3. **Embedding asincroni e Rate-Limiting**:
   * I blocchi Child vengono inviati in batch asincroni alle API Google GenAI per generare vettori ad alta dimensionalità (768 dimensioni via `gemini-embedding-001`).
   * Un limitatore asincrono (`AsyncLimiter` limitato a 100 richieste/minuto) e logiche di retry con backoff esponenziale casuale gestiscono i codici di errore `ResourceExhausted` (429) o `ServiceUnavailable` (503).
4. **Idempotenza e Conservazione della Transazionalità**:
   * **MySQL Upsert**: Viene eseguita una query di `DELETE` atomica sulla tabella `parent_documents` basata sull'indice combinato `idx_search (source, topic_id, sub_topic_id)`. Subito dopo, i nuovi Parent Documents vengono inseriti massivamente via `executemany`. Questo assicura che un reprocessing dello stesso documento elimini le versioni obsolete e inserisca quelle aggiornate senza lasciare record orfani o duplicati.
   * **Qdrant Upsert**: I Child Chunks vengono caricati in Qdrant includendo nel payload i metadati estratti, il `parent_id` di accoppiamento con MySQL e un `_ingestion_id` univoco.
   * **Qdrant Garbage Collection**: Per prevenire "vettori fantasma" (frammenti Child orfani rimasti in Qdrant dopo che il documento originale è stato modificato), il sistema esegue una query di cancellazione mirata (`qdrant_client.delete`) per rimuovere tutti i punti associati allo stesso `source` ma aventi un `_ingestion_id` non corrispondente a quello appena generato.

## 4. Motore di Retrieval Avanzato & Reranking

Il processo di recupero dei contesti all'interno di `worker.py` supera i limiti della semplice ricerca semantica (K-Nearest Neighbors) implementando una catena di elaborazione avanzata.

```
   Query Utente + History (fino a 20 messaggi)
                       │
                       ▼ (gemini-3.1-flash-lite)
           [ Standalone Query / Sub-Queries / Keywords ]
                       │
             ┌─────────┴─────────┐
             ▼ (Embedding)       ▼ (Dizionario Concettuale)
     [ Query Vector ]       [ Concept Lookup (Qdrant) ]
             │                   │
             │                   ▼ (Alias rilevati?)
             │              ┌────┴────┐
             │          (Sì)│     (No)│
             │              ▼         ▼
             │        Espansione   Continua
             │        con Aliases  Normale
             │              │         │
             └──────────────┼─────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Ricerca Ibrida Parallela  │
              │ (Vettoriale + Keyword)   │
              └─────────────┬────────────┘
                            │
                            ▼ (ONNX Reranker Locale)
              ┌──────────────────────────┐
              │   De-duplicazione e      │
              │   Reranking (Sigmoid)    │
              └─────────────┬────────────┘
                            │
                            ▼ (Algoritmo di Redistribuzione)
              ┌──────────────────────────┐
              │ Ripartizione delle Quote │
              │     (Parents Pool)       │
              └──────────────────────────┘

```

### A. Trasformazione delle Query ed Espansione Concettuale Semantica
1. **Query Rewriting Conversazionale**: Il modello `gemini-3.1-flash-lite` analizza gli ultimi `MAX_HISTORY_ITEMS` (fino a 20 messaggi di cronologia) e converte l'interrogazione dell'utente in una query decontestualizzata (`standalone_query`), una lista di sotto-interrogazioni focalizzate (`search_queries` fino a 5 sotto-query) e un elenco di parole chiave lessicali (`keywords`).
2. **Lookup Concettuale**:
   * Il sistema esegue l'embedding della query standalone e interroga una collezione speciale in Qdrant denominata `conceptual_dictionary`.
   * Se la query ha un'affinità semantica superiore a `QDRANT_CONCEPT_THRESHOLD` (default `0.80`) con un concetto mappato a database, estrae un array di **alias e sinonimi aziendali**.
   * Se vengono rilevati alias, il sistema invoca `expand_standalone_with_aliases` per rigenerare e potenziare le sotto-query di ricerca, garantendo che concetti normativi o gerghi tecnici vengano tradotti e ricercati con precisione assoluta.

### B. Ricerca Ibrida Parallela e Reranking Locale via ONNX Cross-Encoder
Per massimizzare le prestazioni ed evitare colli di bottiglia, il recupero dei frammenti candidati avviene in parallelo sfruttando un pool di thread concorrenti (`ThreadPoolExecutor`):
1. **Ricerca Semantica**: Esegue query vettoriali parallele su Qdrant (`document_chunks`) per ciascuna sotto-query generata, applicando soglie di confidenza minime (`QDRANT_THRESHOLD = 0.60`) e filtrando rigorosamente per i `topic_id` e `sub_topic_id` selezionati dall'utente.
2. **Ricerca Sintattica**: In parallelo alle query vettoriali, esegue una ricerca full-text lessicale basata sulle keyword estratte, sfruttando filtri logici `MatchText` su Qdrant per raccogliere corrispondenze esatte (codici, date, identificativi).
3. **Lazy Loading del Reranker ONNX**:
   * Per evitare conflitti di memoria e blocchi dei socket TCP causati dai fork di Celery nei sistemi operativi Unix-like, il Cross-Encoder locale viene istanziato esclusivamente in modalità "Lazy Load" all'interno del processo del worker (`worker_process_init.connect`).
   * Il modello `mmarco-mMiniLMv2-L12-H384-v1` esegue il riordinamento di tutti i candidati accumulati.
   * Il sistema calcola il punteggio sigmoideo assecondando la formula del modello:
     $$P = \sigma(x) = \frac{1}{1 + e^{-x}}$$
     I chunk con una probabilità inferiore a `MIN_PROB_THRESHOLD` (default `0.02`) vengono immediatamente scartati dal pool dei risultati.

### C. Algoritmo di Ripartizione delle Quote e Redistribuzione Attiva
In contesti multi-query, un problema tipico è la saturazione del contesto causata da una singola sotto-query molto loquace, a scapito di sotto-query con pochi documenti ma di vitale importanza. Questo sistema implementa un algoritmo proprietario per la gestione democratica del contesto:
1. **Quota Garantita per Sotto-query**: Ciascuna sotto-query ha diritto a una quota fissa di documenti Parent associati ai suoi migliori Child vettoriali, definita dal parametro `PARENTS_PER_QUERY` (default 4).
2. **Allocazione Iniziale**: Il sistema scorre la lista riordinata dei risultati e assegna a ciascuna query i suoi primi 4 documenti Parent univoci.
3. **Calcolo degli Slot Liberi (Redistribuzione)**:
   * Se alcune sotto-query non producono risultati (o ne producono meno della quota di 4), il sistema calcola la capacità residua non sfruttata:
     $$\text{Slot Liberi} = (\text{PARENTS\_PER\_QUERY} \times N_{\text{queries}}) - N_{\text{selezionati}}$$
   * Questi slot rimasti liberi vengono ridistribuiti in modalità **round-robin** a favore delle sotto-query che possiedono una riserva di documenti validi scartati dall'allocazione iniziale.
   * Questo meccanismo garantisce che il budget massimo di contesto impostato (`MAX_CONTEXT_CHARS = 30000` caratteri) non venga mai sprecato, ottimizzando la densità informativa offerta al modello generatore.

## 5. Architettura del Frontend e Strategie di Accessibilità (`main.dart`)

Il modulo client sviluppato in **Flutter** (`main.dart`) implementa pattern avanzati per la gestione dello stato reattivo, la conformità WCAG per l'accessibilità, il rendering strutturato e l'ottimizzazione UX per mitigare l'attesa computazionale del backend.

```
   [ App Start ] ──► Rilevamento Lingua Browser (html.window) ──► IT / EN
                                  │
                                  ▼
                [ Check First Run (SharedPreferences) ] 
              /                                        \
 (Mai visto) /                                          \ (Già visto)
           ▼                                             ▼
   [ Mostra Tutorial Interattivo ]                   Carica App Standard
   (TutorialCoachMark su TargetKey)
                    │
                    ▼
    [ Tema Attivo (ValueNotifier) ]
┌───────────────────┼───────────────────┐
▼                   ▼                   ▼
[ Tema Chiaro ]     [ Tema Scuro ]     [ Alto Contrasto ]
                                       (Nero, Giallo, Cyan)
                                       [Shortcut: Alt + T]

```

### A. Internazionalizzazione e Localizzazione Dinamica
1. **Rilevamento Nativo della Lingua**: All'avvio, l'applicazione interroga il runtime del browser tramite `html.window.navigator.language` per determinare l'impostazione locale del sistema operativo. 
2. **Aggiornamento Dinamico del DOM**: In base al valore rilevato (es. `it` o `en`), viene impostato il `ValueNotifier<AppLang>` (`langNotifier`). Il client aggiorna asincronamente il tag principale del documento HTML del browser (`html.document.documentElement?.lang = langCode`) per allinearsi perfettamente con le tecnologie assistive e gli screen reader dell'utente.
3. **Sincronizzazione della Chat**: Al cambio manuale della lingua effettuato dall'utente tramite l'AppBar, la cronologia della chat viene reinizializzata in modo pulito (`_clearChat()`) per evitare conflitti linguistici nella history conversazionale inviata all'LLM.

### B. Gestione dei Temi e Accessibilità WCAG-Compliant
L'interfaccia supporta un ciclo di tre temi grafici dinamici gestiti tramite `ValueNotifier<AppTheme>` ed ereditati dall'alberatura dei widget Flutter:
* **Tema Chiaro e Tema Scuro**: Basati sulla palette di colori dinamici ricavata dal seme `Colors.blue` tramite Material 3.
* **Tema ad Alto Contrasto (Accessibility-First)**: Configurato specificatamente per ipovedenti o utilizzi in condizioni di luce solare diretta estrema. Utilizza uno sfondo nero assoluto (`ColorScheme.highContrastDark`), testi ad altissima leggibilità in bianco puro, dettagli ad alta visibilità in giallo acceso (`Colors.yellowAccent`) e accenti di selezione in ciano (`Colors.cyanAccent`).
* **Scorciatoie hardware native**: Attraverso i widget `Shortcuts`, `Actions` e `Intents`, il sistema mappa la scorciatoia hardware `Alt + T` (o `Option + T` su MacOS) per cicliare i temi in qualsiasi punto dello schermo, senza la necessità di interagire fisicamente con i pulsanti a schermo.
* **Predisposizione Screen Reader (Semantics)**: L'intera interfaccia racchiude i widget critici (come i messaggi della chat e l'indicatore di caricamento) all'interno di elementi `Semantics` impostando `liveRegion: true` e `link: true`. Questo assicura che qualsiasi variazione asincrona dello schermo (come l'arrivo di una risposta o l'attivazione dei tempi di attesa) venga annunciata istantaneamente e con priorità dai motori di sintesi vocale di sistema.

### C. UX Onboarding Interattivo (`TutorialCoachMark`)
Per facilitare la comprensione delle funzioni a utenti non tecnici:
1. Al primo avvio, l'applicazione interroga le preferenze persistenti tramite `SharedPreferences` verificando la chiave `tutorial_seen`.
2. Se assente, viene istanziato un flusso di overlay interattivo pilotato da `TutorialCoachMark`.
3. Il tutorial aggancia sequenzialmente i `GlobalKey` associati agli elementi strutturali della pagina (`_filterKey`, `_langKey`, `_themeKey`, `_deleteKey`, `_sendKey`) oscurando il resto dello schermo con un velo opaco ad alto contrasto (`opacityShadow: 0.85`) e presentando schede esplicative localizzate in base alla lingua dell'utente.

### D. Rendering Ibrido del Contesto HTML e Gestione delle Fonti
L'LLM restituisce risposte contenenti tag strutturati HTML. Il frontend decodifica ed esegue il rendering grafico di tali contenuti in modo sicuro:
* **Integrazione `flutter_html`**: Utilizza la libreria per formattare paragrafi (`<p>`), elenchi puntati (`<ul>`/`<li>`) e codice. La stilizzazione adatta dinamicamente i margini e i colori del testo basandosi sul tema attivo correntemente nel BuildContext.
* **TagExtension personalizzate**: Gli elementi d'ancoraggio `<a>` dotati di classe CSS `.btn` vengono intercettati e renderizzati graficamente come pulsanti nativi di tipo `OutlinedButton`, con comportamento delegato a `url_launcher` per aprire in sicurezza le sorgenti ospitate sull'Nginx aziendale.
* **Gestione delle Fonti**: Sotto ogni risposta, viene inserito un menu collassabile (`AnimatedSize` con curve di transizione fluide `Curves.easeInOut`) che mostra il numero complessivo delle fonti utilizzate. L'espansione del menu rivela una serie di widget `ActionChip` cliccabili che reindirizzano l'utente al percorso esatto del documento sorgente parent (es. `/topic/subtopic/nomefile.pdf`).

### E. Polling Asincrono e Logica di Progressive Delay (UX Resiliente)
La generazione di una risposta RAG strutturata con grading e auto-correzione può richiedere tempi di calcolo variabili (da pochi secondi a oltre un minuto in caso di retry multipli). Il frontend implementa una strategia asincrona resiliente:
1. **Accettazione della Richiesta (202 Accepted)**: La chiamata verso l'endpoint `/chat` risponde immediatamente con stato `202` fornendo il `task_id` di Celery. L'app imposta `_isLoading = true` e avvia il polling verso `/status/<task_id>` ogni 2 secondi.
2. **Mitigazione dell'Ansia da Attesa (Progressive Delay Messaging)**:
   All'avvio del polling, viene inizializzato un `Timer` Dart programmato per scattare dopo **20 secondi**.
   * Se la risposta del backend non è ancora arrivata entro i 20 secondi, l'applicazione aggiorna asincronamente lo stato del caricamento presentando a schermo un messaggio di cortesia dinamico e localizzato (es. *"L'elaborazione sta richiedendo più tempo del solito. Grazie per l'attesa..."*).
   * Questo riduce l'abbandono della pagina da parte dell'utente finale, fornendo un feedback visivo immediato che assicura che il sistema stia lavorando attivamente in background.
3. **Resilienza alle interruzioni di rete**: Il loop di polling tollera fino a 3 errori consecutivi di rete prima di dichiarare la perdita di connessione, evitando interruzioni premature della sessione in caso di micro-disconnessioni del Wi-Fi del client.

### F. Circuit Breaking e Stato di Manutenzione (`_buildMaintenancePage()`)
All'inizializzazione dello schermo, l'applicazione invoca l'endpoint `/config` per caricare le configurazioni del topic e dei relativi sub-topic.
* Se la chiamata fallisce (es. server non raggiungibile, timeout del servizio o errore interno 500), il client blocca la visualizzazione e attiva una schermata nativa e pulita di **Manutenzione** (`_buildMaintenancePage()`).
* Questa pagina presenta un'icona di cantiere in corso, spiega chiaramente la situazione all'utente tramite messaggi localizzati ed espone un comodo pulsante di ripristino ("Riprova connessione") per rieseguire il bootstrap del sistema non appena i servizi tornano attivi.

## 6. Generazione con Autovalutazione & Ciclo di Autocorrezione (Self-Correction Loop)

Una volta ricomposto il contesto mediante il recupero bulk dei Parent Documents da MySQL, il sistema avvia un loop di generazione controllata e auto-correzione a più livelli:


```
              ┌──────────────────────────────┐
              │  Generazione in JSON Mode    │
              │   (is_found, answer, ecc.)   │
              └──────────────┬───────────────┘
                             │
                             ▼
                 [ is_found == True? ]
                 /                   \
          (Sì)  /                     \ (No)
               ▼                       ▼
   ┌────────────────────────┐        ┌────────────────────────┐
   │   Grader LLM (CoT)     │        │   Incrementa Retry.    │
   └───────────┬────────────┘        │  Inietta istruzioni    │
               │                     │  correttive in History │
       [ Risposta Valida? ]          │   e rigenera Query     │
       /                  \          └───────────┬────────────┘
(Sì)  /                    \ (No)                │
     ▼                      ▼                    │
┌─────────────────┐    ┌─────────────────┐       │
│ Salva in Cache  │    │ Forza Retry     │◄──────┘
│ Ritorna Risposta│    │ (fino a max 2)  │
└─────────────────┘    └────────┬────────┘
                                │
                                ▼ (Se esauriti i tentativi)
                       ┌─────────────────┐
                       │  Auto-Log DB    │
                       │ (chat_feedback) │
                       └─────────────────┘

```

1. **Generazione Strutturata (JSON Mode)**:
   * Il modello `gemini-3.5-flash` genera la risposta finale basandosi sul template di prompt specifico del topic caricato dinamicamente da MySQL.
   * L'output viene forzato in formato JSON contenente la chiave booleana `is_found` e la stringa testuale `answer`.
2. **Logica Fail-Fast**:
   * Se il modello imposta `is_found = False` (ammettendo esplicitamente che i documenti non contengono la risposta), il sistema salta il passaggio del Grader per risparmiare risorse e tempo di calcolo, indirizzando immediatamente il flusso verso il ciclo di retry.
3. **Auditing tramite Grader LLM (Chain of Thought)**:
   * Se `is_found = True`, il modello valutatore (`gemini-3.1-flash-lite`) esamina la query, la risposta generata e il testo completo dei contesti passati per determinare se la risposta è accurata e priva di allucinazioni.
4. **Ciclo di Retry Attivo e Riformulazione**:
   * Se il Grader boccia la risposta, il sistema avvia un secondo tentativo (`MAX_MODEL_RETRIES = 2`).
   * Viene iniettato nella cronologia un messaggio fittizio utente che istruisce esplicitamente il query rewriter a cambiare angolazione semantica, allargare i filtri logici e utilizzare sinonimi differenti per estrarre nuovi documenti.
   * La pipeline riesegue il retrieval e produce una nuova risposta.
5. **Caching Semantico e Auto-Logging delle Query Fallite**:
   * **In caso di successo**: La risposta viene salvata nella collezione `semantic_cache` di Qdrant. La chiave è sensibile e tiene traccia della query, dei sub-topic attivi e dei filtri applicati.
   * **In caso di fallimento definitivo**: Il sistema salva automaticamente la sessione nella tabella `chat_feedback` di MySQL con un voto di default pari a `0` e la dicitura `"Auto-Log: Risposta non trovata o non soddisfacente"`. Questo permette agli amministratori di effettuare analisi offline e sessioni di fine-tuning dei prompt o della base documentale.

## 7. Schema del Database MySQL & Integrazione con il Codice

Di seguito viene mappata l'interazione tra lo schema logico del database MySQL (da `mysql_schema.txt`) e le funzioni della pipeline RAG:

| Tabella MySQL | Ruolo nel RAG | Modulo Interagente | Operazioni Chiave |
| :--- | :--- | :--- | :--- |
| `parent_documents` | Memorizza il contesto strutturato esteso (Parent). | `ingest.py` (Scrittura)<br>`worker.py` (Lettura) | `executemany` per inserimento massivo.<br>`SELECT WHERE id IN (...)` per recupero. |
| `sub_topics` | Configurazione a runtime dei parametri di chunking per sub-topic. | `ingest.py` (Lettura)<br>`app.py` (Lettura) | `get_subtopic_config` per recuperare i parametri di splitting prima dell'ingestion. |
| `topics` | Associazione dei macro-argomenti ai rispettivi prompt di sistema. | `worker.py` (Lettura) | `get_topic_prompt` per caricare dinamicamente il template del prompt. |
| `chat_feedback` | Raccolta feedback utente e tracciamento fallimenti del Grader. | `app.py` (Scrittura)<br>`worker.py` (Scrittura) | `log_automatic_negative_feedback` per query insoddisfacenti con rating 0. |
| `system_logs` | Auditing centralizzato ed errori di esecuzione dei pod. | `ingest.py` / Moduli RAG | `MySQLLogHandler` cattura eccezioni e stack-trace salvandoli nel database. |
| `failed_queries` | Registro delle query non instradabili o prive di topic valido. | `app.py` (Gateway) | Tracciamento dei tentativi di accesso non autorizzati o query orfane. |

## 8. Manutenzione, Monitoraggio e Scheduled Task

* **Logging Centralizzato (`system_logs`)**:
  Il logger utilizza un handler MySQL personalizzato (`MySQLLogHandler`). In caso di eccezioni critiche (es. arresto improvviso di un worker, errori di rete con RabbitMQ, ecc.), lo stack trace completo, il file di origine, il numero di riga e il nome del pod Kubernetes vengono registrati in modo strutturato a database, semplificando le attività di debugging in ambienti cloud.
* **CronJob di Manutenzione della Cache**:
  Per evitare la conservazione di risposte obsolete, Celery Beat esegue un task pianificato periodico (`prune_semantic_cache`) che viene attivato ogni ora (`crontab(minute=0)`):
  * Esegue una query di cancellazione su Qdrant rimuovendo tutti i vettori di cache il cui timestamp di creazione è superiore a `MAX_AGE_SECONDS` (24 ore).


## 9. Requisiti Software, Dipendenze e Configurazione

Il sistema è pronto per essere eseguito in ambienti containerizzati e orchestrati (Docker, Kubernetes). Di seguito vengono definiti i requisiti software essenziali.

### Dipendenze di Sistema:
* **Database Relazionale**: MySQL 8.0+ (consigliato l'uso del motore InnoDB, charset `utf8mb4`).
* **Database Vettoriale**: Qdrant v1.7+ (con supporto per ricerche di affinità ed eliminazione asincrona).
* **Message Broker**: RabbitMQ v3.12+ (con supporto per code durevoli e messaggi persistenti).
* **Distributed Task Queue & Backend di Stato**: Redis v7.0+ (per Celery).

### Dipendenze Python Principali (`requirements.txt`):
```text
google-genai>=0.1.1
google-cloud-vision>=3.4.0
qdrant-client>=1.7.0
langchain-text-splitters>=0.0.1
pymupdf>=1.23.0
pymupdf4llm>=0.0.12
aio-pika>=9.0.0
aiolimiter>=1.1.0
celery>=5.3.0
redis>=5.0.0
Flask>=3.0.0
Flask-CORS>=4.0.0
mysql-connector-python>=8.2.0
onnxruntime>=1.16.0
tenacity>=8.2.0
typeguard>=4.1.0

```

### Variabili d'Ambiente Fondamentali:

```bash
# --- Autenticazione & API ---
API_LLM_KEY="your-gemini-and-embedding-key"
API_SECRET_KEY="secure-bearer-token-for-flask"
ALLOWED_ORIGINS="[https://your-frontend-domain.com](https://your-frontend-domain.com)"

# --- Broker & Code ---
BROKER_HOST="rabbitmq-service.rag.svc.cluster.local"
BROKER_PORT=5672
BROKER_USERNAME="guest"
BROKER_PASSWORD="guest"
REDIS_URL="redis://redis-service.rag.svc.cluster.local:6379/0"

# --- Database ---
MYSQL_SERVICE_HOST="mysql-service.rag.svc.cluster.local"
MYSQL_SERVICE_PORT=3306
MYSQL_USER="raguser"
MYSQL_PASSWORD="your-secure-db-password"
MYSQL_DATABASE="rag_db"

QDRANT_HOST="qdrant-service.rag.svc.cluster.local"
QDRANT_PORT=6333

# --- Modelli & Reranking ---
OCR_MODEL_NAME="gemini-3-flash-preview"
QUERY_REWRITER_MODEL_NAME="gemini-3.1-flash-lite"
ANSWER_GENERATOR_MODEL_NAME="gemini-3.5-flash"
GRADER_MODEL_NAME="gemini-3.1-flash-lite"
RERANKER_MODEL_PATH="./model_cache/mmarco-mMiniLMv2-L12-H384-v1"

# --- Parametri Algoritmici ---
LOG_LEVEL="INFO"
ALLOW_GENERAL_KNOWLEDGE="true"
ALLOW_SUBTOPIC_SELECTION="true"
RERANK_SIZE=25
MAX_CONTEXT_CHARS=30000

```
