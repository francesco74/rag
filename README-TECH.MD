# Specifica Tecnica ed Architetturale: Sistema RAG Enterprise-Grade

Questo documento descrive in modo formale e approfondito l'architettura, le strategie algoritmiche, i flussi di dati e i requisiti tecnologici del sistema **Retrieval-Augmented Generation (RAG)** avanzato implementato. Il sistema adotta un approccio asincrono, resiliente e guidato dagli eventi (*Event-Driven*) avanzata della fase di ricerca e generazione.

## 1. Panoramica dell'Architettura di Sistema

L'infrastruttura si articola su una serie di microservizi accoppiati in modo lasco attraverso code di messaggi (**RabbitMQ**) e gestiti in modo distribuito per la parte computazionale intensiva RAG tramite **Celery** (con broker di stato **Redis**). La persistenza è ibrida, combinando un database relazionale (**MySQL**) per i dati strutturati e i testi completi e un database vettoriale (**Qdrant**) per la ricerca semantica rapida su frammenti più piccoli (Child).

### Componenti Principali:

1. **Wrapper Diretti (`direct.py`)**: Uno script (eseguibile anche como CronJob) che rileva nuovi documenti raw preservando l'alberatura, genera manifesti JSON e notifica RabbitMQ.

2. **Normalizzatore Documentale (`converter.py`)**: Consumatore asincrono che trasforma file PDF, immagini o testi grezzi in formato Markdown standardizzato utilizzando motori OCR avanzati e logiche di fallback.

3. **L'Ingestore di Frammenti (`ingest.py`)**: Consumatore asincrono che effettua lo split dei testi, genera embedding vettoriali e popola in modo atomico MySQL (Parent) e Qdrant (Child).

4. **API Gateway (`app.py`)**: Espone le interfacce Flask protette tramite Bearer Token per la gestione del flusso conversazionale e la raccolta di feedback.

5. **RAG Worker (`worker.py`)**: Motore Celery che esegue la pipeline di query-transformation, caching semantico, ricerca ibrida, riordinamento (Reranking) locale tramite modelli ONNX e loop di auto-correzione.

## 2. Flusso dei Documenti (Ingestion Pipeline)

Prima che i dati siano pronti per essere interrogati nel database vettoriale, essi attraversano un ciclo di vita rigorosamente strutturato diviso in tre fasi principali: **Rilevamento e Wrapping**, **Normalizzazione/OCR**, ed **Ingestione Parent-Child**.


```

[Area Input: data/direct]
│
▼
┌───────────────┐
│   direct.py   │  ──► Genera Manifest JSON + Sposta in data/staging
└───────────────┘
│
▼ [RabbitMQ: coda "da-convertire"]
┌───────────────┐
│  converter.py │  ──► Estrazione Ibrida (PyMuPDF / Gemini OCR / Vision Fallback)
└───────────────┘
│
▼ [RabbitMQ: coda "da-indicizzare"]
┌───────────────┐
│   ingest.py   │  ──► Parent-Child Splitting + MySQL/Qdrant Atomic Upsert
└───────────────┘

```

### Fase 1: Rilevamento e Wrapping (`direct.py`)

* Il sistema esegue una scansione ricorsiva della directory `data/direct` alla ricerca di estensioni supportate (`.pdf`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`).

* Per ciascun file identificato:

  1. Viene calcolato il percorso relativo rispetto alla cartella di origine per preservare l'alberatura del topic.

  2. Il file viene spostato nella directory di `staging` (`data/staging`).

  3. Viene generato un file di manifesto JSON radice (avente lo stesso nome del file ma estensione `.json`). Il manifesto definisce un identificativo univoco per la sorgente (`source`), la lista dei file associati e i metadati originali.

  4. Viene pubblicato un messaggio persistente sulla coda RabbitMQ `"da-convertire"` contenente il percorso del manifesto.

### Fase 2: Normalizzazione Documentale (`converter.py`)

Questo modulo assicura che qualsiasi formato di input venga convertito in un formato testuale uniforme Markdown. Il processo è guidato da un sistema di estrazione ibrida PDF/OCR:

```

               ┌──────────────────────┐
              │ Ricezione file .json │
              └──────────┬───────────┘
                         │
                         ▼
             [ Tipo di File Allegato? ]
            /             │             \
    (PDF) /               │ (.txt/.md)   \ (Immagine)
          ▼               ▼               ▼
 ┌─────────────────┐    ┌──────────┐ ┌───────────────┐
 │Lettura Pagine   │    │ Lettura  │ │  Gemini OCR   │
 │con PyMuPDF      │    │ Diretta  │ │  (Markdown)   │
 └────────┬────────┘    └────┬─────┘ └───────┬───────┘
          │                  │               │
   [Densità Testo?]          │               │
  /                \         │               │
 / <50 crt.         \>=50    │               │
▼                    ▼       │               │
┌──────────────┐ ┌──────────┐│               │
│  Gemini OCR  │ │PyMuPDF4LL││               │
│(con fallback)│ │Markdown  ││               │
└──────┬───────┘ └────┬─────┘│               │
       │              │      │               │
       ▼              ▼      ▼               ▼
┌──────────────────────────────────────────────┐
│      Salvataggio .md + .json in data/watch   │
│        Notifica coda "da-indicizzare"        │
└──────────────────────────────────────────────┘

```

1. **Lettura del Manifesto**: Il Worker consuma l'evento ed ispeziona i file allegati in staging.

2. **Estrazione Ibrida da PDF**:

   * Per ciascuna pagina del PDF, viene calcolato il rect ed estratto il testo nativo.

   * **Valutazione della Densità**: Se la pagina contiene meno di 50 caratteri, viene considerata una pagina scansionata (immagine). Viene renderizzata in un'immagine PIL e inviata alle API di **Gemini Vision** (`gemini-3-flash-preview` o analogo configurato via `OCR_MODEL_NAME`) per l'OCR con output forzato in formato Markdown.

   * **Resilienza alle policy e Fallback**: Se le API di Gemini bloccano la trascrizione per motivi di policy di sicurezza (es. flag `SAFETY`, `RECITATION`), il sistema avvia istantaneamente un fallback asincrono su **Google Cloud Vision API** per evitare il blocco del flusso.

   * Se la pagina contiene testo nativo sufficiente, viene convertita direttamente in Markdown strutturato tramite `pymupdf4llm` (gestendo i tag di interruzione riga `<br>`).

3. **Gestione Immagini Standard**: I file immagine nativi vengono inviati direttamente alla pipeline OCR di Gemini.

4. **Scrittura in Watch & Notifica**: Una volta normalizzati i file allegati, vengono salvati file individuali `.md` e i rispettivi manifesti figli aggiornati (aventi la chiave `source` arricchita con la sintassi `sorgente::nomefile.pdf`) all'interno della directory `data/watch`. Il sistema notifica infine la coda `"da-indicizzare"`.

### Fase 3: Frammentazione e Ingestione Parent-Child (`ingest.py`)

Il modulo d'ingestione adotta un'architettura **Parent-Child** studiata per risolvere il classico trade-off dei sistemi RAG (passare al modello generatore contesti ampi e ricchi di dettagli, mantenendo al contempo gli embedding vettoriali focalizzati su segmenti piccoli e semanticamente densi).

1. **Risoluzione Dinamica delle Configurazioni**: Il sistema recupera dal DB MySQL i parametri di chunking impostati per lo specifico sub-topic (es. `chunk_size`, `chunk_overlap`, `parent_chunk_size`, `use_markdown_splitter`). Se non presenti, applica solidi valori di fallback.

2. **Generazione dei Parent Documents**:

   * Il testo Markdown completo viene elaborato. Se abilitato, viene impiegato il `MarkdownHeaderTextSplitter` per dividere il file basandosi sull'alberatura degli header (da `#` fino a `####`).

   * Successivamente, viene applicato un `RecursiveCharacterTextSplitter` basato sul tokenizer di Tiktoken (`cl100k_base`) per generare i **Parent Documents** (con una dimensione tipica di \~1500 token).

3. **Generazione dei Child Chunks**:

   * Ogni blocco Parent viene ulteriormente frammentato in blocchi **Child** più piccoli (es. 500 token di dimensione con 100 di sovrapposizione).

4. **Embedding Asincroni in Batch**:

   * I frammenti Child vengono impacchettati e inviati in batch alle API di Google GenAI utilizzando il modello `gemini-embedding-001`.

   * Per mitigare i limiti di quota di Google, la chiamata viene regolata da un limitatore asincrono (`AsyncLimiter` impostato a 100 richieste/minuto) e da un sistema di riesecuzione robusto (`tenacity`) con backoff esponenziale casuale in caso di errori `ResourceExhausted` o `ServiceUnavailable`.

5. **Persistenza e Idempotenza Transazionale**:

   * **MySQL**: I Parent Documents vengono inseriti massivamente nella tabella `parent_documents`. Per garantire l'idempotenza ed evitare la duplicazione dei dati in caso di reprocessing del documento, viene eseguita preventivamente una cancellazione atomica dei vecchi parent associati a quella specifica triade (`source`, `topic_id`, `sub_topic_id`).

   * **Qdrant**: I vettori dei Child vengono caricati massivamente in Qdrant (collezione `document_chunks`) includendo nel payload un identificativo di ingestione univoco (`_ingestion_id`).

   * **Cleanup Vettoriale**: Per prevenire frammenti orfani nel database vettoriale, subito dopo il successo del caricamento dei nuovi vettori viene eseguita un'operazione di eliminazione mirata per rimuovere tutti i punti associati allo stesso `source` ma aventi un `_ingestion_id` obsoleto.

## 3. Analisi delle Caratteristiche Avanzate del Sistema RAG

Il motore di esecuzione implementato all'interno di `worker.py` dimostra un design all'avanguardia che implementa i pattern più moderni del RAG avanzato.

### A. Architettura Parent-Child (MySQL + Qdrant)

Invece di cercare un compromesso rischioso sulla dimensione dei blocchi, il sistema divide nettamente le responsabilità:

* **Qdrant** ospita esclusivamente gli embedding e i payload dei frammenti piccoli (**Child**). Questo massimizza la precisione e la rilevanza semantica durante la fase di ricerca vettoriale (evitando l'effetto di diluizione del segnale semantico tipico dei testi lunghi).

* **MySQL** ospita il testo completo e strutturato del blocco più grande (**Parent**).

* Durante la fase di recupero, il sistema interroga Qdrant per identificare i Child più affini, ma estrae dai payload i relativi `parent_id`. Successivamente esegue una query bulk su MySQL per caricare i testi Parent corrispondenti. In questo modo, l'LLM generatore riceve un contesto ampio, coerente e non frammentato.

### B. Query Transformation & Espansione tramite Dizionario Concettuale

La query originale dell'utente, spesso ambigua o formulata in linguaggio naturale colloquiale, viene arricchita prima di toccare i database vettoriali:

1. **Query Rewriter**: Il modello `gemini-3.1-flash-lite` analizza lo storico della conversazione (fino a 20 elementi) e riformula la richiesta estraendone una query indipendente (`standalone_query`), un set di sotto-interrogazioni focalizzate (`search_queries`) e una lista di parole chiave fondamentali (`keywords`).

2. **Semantic Concept Lookup**: La query standalone viene convertita in un vettore per affinità semantica e confrontata con una collezione Qdrant dedicata chiamata `conceptual_dictionary`.

   * Se il punteggio supera la soglia (`QDRANT_CONCEPT_THRESHOLD`), significa che l'utente si sta riferendo ad una terminologia tecnica o ad un concetto normativo specifico che possiede sinonimi o alias noti nel sistema.

   * Gli alias individuati vengono fusi con la query standalone e usati per espandere le sotto-interrogazioni di ricerca in modo mirato (`expand_standalone_with_aliases`).

### C. Algoritmo di Retrieval Ibrido e Reranking Locale (ONNX)

Per catturare sia le sfumature di significato sia i riferimenti letterali (codici, date, codici identificativi), il sistema esegue una ricerca ibrida parallela:

* **Ricerca Semantica**: Esegue ricerche vettoriali parallele in Qdrant (sfruttando un pool di thread concorrenti) per ciascuna sotto-query generata.

* **Ricerca Sintattica**: Se sono state estratte parole chiave, esegue una ricerca full-text asincrona (scansione tramite filtri logici `MatchText` su Qdrant) per catturare corrispondenze lessicali precise.

* **Reranking Cross-Encoder Locale**: Tutti i frammenti candidati accumulati vengono de-duplicati e riordinati localmente tramite un modello di Reranking basato su ONNX Runtime (`mmarco-mMiniLMv2-L12-H384-v1`). L'operazione è altamente ottimizzata dal punto di vista computazionale (sfruttando il parallelismo a livello di thread della CPU) e converte i logit grezzi in probabilità reali tramite una funzione sigmoidea robusta. Solo i blocchi che superano la soglia di confidenza minima (`MIN_PROB_THRESHOLD`) vengono promossi.

### D. Ripartizione Dinamica delle Quote (Quota Allocation & Redistribution)

Uno dei problemi principali del RAG multi-query è che una query di ricerca potrebbe monopolizzare l'intero spazio di contesto disponibile, oscurando sotto-query meno loquaci ma ugualmente importanti. Per risolvere questo problema, il sistema implementa un algoritmo proprietario di ripartizione e redistribuzione delle risorse:

1. Ogni sotto-query ha una quota massima garantita di documenti Parent restituibili (`PARENTS_PER_QUERY`).

2. Il sistema scansiona i risultati ordinati dal Reranker e assegna gli slot.

3. Se una o più sotto-query non producono risultati sufficienti o non raggiungono la quota ideale (es. per carenza di documenti rilevanti sul database vettoriale), **gli slot rimanenti non vengono sprecati**.

4. Gli slot liberi vengono ridistribuiti in modalità *round-robin* tra le sotto-query che possiedono una riserva di documenti candidati extra. Questo assicura che il budget del contesto passato all'LLM (regolato da `MAX_CONTEXT_CHARS`) sia sempre sfruttato al massimo della sua utilità.

### E. Autovalutazione e Loop di Correzione (Self-Correction & Grader Loop)

Il sistema implementa un potente pattern ad agenti basato su un ciclo di auto-correzione prima di servire la risposta all'utente:


```
       ┌────────────────────────┐
       │ Generazione Risposta   │
       │ (gemini-3.5-flash)     │
       └───────────┬────────────┘
                   │
                   ▼
        [ Il modello segnala? ]
        /                     \
(is_found = False)              \ (is_found = True)
      ▼                           ▼
┌──────────────┐          ┌───────────────────┐
│ Forza Fallito│          │ Valutazione con   │
│ Senza Grader │          │ Grader LLM        │
└──────┬───────┘          └─────────┬─────────┘
       │                            │
       │                    [ Risposta valida? ]
       │                    /               \  
       │            (No)   /                 \ (Sì)
       ▼                  ▼                    ▼
┌────────────────────────────┐         ┌───────────────────┐
│ Incrementa Tentativi.      │         │ Salva in Cache    │
│ Inietta istruzione di      │         │ Ritorna Risposta  │
│ ri-formulazione e ri-esegui│         └───────────────────┘
│ pipeline di Retrieval      │
└────────────────────────────┘

```

1. **Generazione Strutturata**: L'LLM (`gemini-3.5-flash`) genera la risposta richiedendo esplicitamente un formato JSON strutturato contenente la chiave booleana `is_found`.

2. **Logica Fail-Fast**: Se `is_found` è falsato (il modello dichiara esplicitamente di non disporre di materiale attendibile nei documenti), il Grader viene evitato per risparmiare tempo e risorse API, classificando immediatamente il tentativo come insoddisfacente.

3. **Il Grader**: Se la risposta è strutturalmente valida, un modello valutatore (`gemini-3.1-flash-lite`) esegue un controllo di fedeltà (CoT - Chain of Thought) incrociando la risposta con i testi dei contesti estratti per identificare eventuali allucinazioni.

4. **Ciclo di Retry Attivo**: Se il Grader boccia la risposta (o se `is_found` è falso), il sistema avvia un nuovo tentativo (fino a `MAX_MODEL_RETRIES`):

   * Modifica la cronologia inserendo un'istruzione correttiva dinamica (esortando l'LLM ad allargare l'angolo semantico o a usare sinonimi alternativi).

   * Riesegue l'intera pipeline di query transformation, embedding vettoriali e retrieval per ottenere un set di contesti differente e più ampio.

5. **Auto-Logging degli Errori**: Se anche l'ultimo tentativo fallisce, la query originale viene salvata automaticamente nella tabella MySQL `chat_feedback` con un rating pari a `0` e un commento di auto-log. Questo consente agli amministratori del sistema di eseguire un'analisi offline dei fallimenti per migliorare i prompt o arricchire la base documentale.

### F. Cache Semantica e Filtri Dinamici sui Metadati

Per garantire tempi di risposta fulminei (sub-secondari) per quesiti ricorrenti, il sistema interroga una cache semantica memorizzata all'interno di Qdrant (`semantic_cache`):

* La chiave di cache è incredibilmente sensibile e sicura: viene calcolata in base al vettore della query principale, alla combinazione ordinata dei sub-topic attivi e ad un hash crittografico MD5 che rappresenta fedelmente lo stato dei filtri dei metadati applicati.

* Se la visualizzazione della query in cache supera la soglia dello `0.97`, la risposta salvata viene istantaneamente restituita, bypassando l'intero ciclo di recupero, riordinamento e chiamata all'LLM.

* Una routine periodica pianificata via Celery Beat (`prune_semantic_cache`) provvede ad eliminare i record di cache obsoleti (più vecchi di 24 ore) per evitare la persistenza di informazioni superate.

## 4. Analisi dei Punti di Forza dell'Architettura

* **Resilienza alle API Esterne**: L'uso combinato di code RabbitMQ per l'ingestione, la limitazione del rate tramite code asincrone (`AsyncLimiter`) e il fallback transizionale tra Gemini e Google Cloud Vision rendono la pipeline di digitalizzazione e ocrizzazione immune a blocchi improvvisi.

* **Strutturazione del Codice per Processi Sicuri**: L'inizializzazione dei client di rete (`QdrantClient`, pool MySQL) avviene rigorosamente in modalità post-fork all'interno dell'evento Celery `@worker_process_init.connect`. Questo azzera i rischi di corruzione dei socket o errori di Broken Pipe legati alla gestione dei thread in ambienti Unix.

* **Separazione dei Contesti (Topic/Sub-topic)**: La segmentazione logica dei dati su MySQL e Qdrant assicura che gli utenti o le interfacce API interroghino esclusivamente perimetri documentali coerenti, riducendo drasticamente il rumore informativo nel contesto.

* **Filtrazione Flessibile e Tipizzata**: Il sistema mappa nativamente costrutti complessi sui filtri vettoriali di Qdrant (come ricerche full-text mirate, range temporali numerici, selezioni multiple tramite `MatchAny` o valori puntuali).

