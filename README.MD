# Guida al Sistema RAG Agentico Avanzato per la Ricerca Documentale nell'Ente Locale

## Introduzione e Contesto Aziendale

Gli Enti Locali (Comuni, Province, Unioni di Comuni) gestiscono quotidianamente una imponente mole di documenti amministrativi: deliberazioni di Giunta e di Consiglio, determinazioni dirigenziali, ordinanze, regolamenti, piani territoriali e atti di protocollo. Spesso, la frammentazione di questo patrimonio informativo rallenta l'attività istruttoria degli uffici e impatta sulla qualità delle risposte fornite ai cittadini.

L'introduzione di un assistente intelligente basato su architettura **RAG** (*Retrieval-Augmented Generation*) rappresenta lo strumento ideale per la ricerca interna, capace di leggere e analizzare migliaia di pagine di documenti, contratti, manuali e normative in pochi secondi. Questa tecnologia non si limita a cercare semplici parole chiave, ma "comprende" il contesto profondo degli atti dell'Ente e formula risposte sintetiche, accurate e sicure basandosi esclusivamente su di essi, senza mai inventare nulla.

## 1. La Filosofia Operativa: "L'Esame a Libro Aperto"

Le intelligenze artificiali tradizionali rispondono alle domande basandosi solo sulle informazioni generiche su cui sono state originariamente addestrate. Questo approccio le porta ad "allucinare", ovvero a inventare risposte che sembrano credibili ma sono del tutto false o non verificate.

Il nostro sistema avanzato supera questo limite comportandosi come un istruttore amministrativo o uno studente brillante che affronta un **esame a libro aperto**:

* 
**Assenza di cieca fiducia:** Il sistema non risponde attingendo a una memoria astratta o generica.


* 
**Interrogazione dell'archivio:** Ricevuto il quesito, interroga istantaneamente l'archivio digitale e la biblioteca dell'Ente (delibere, regolamenti, determine).


* 
**Estrazione puntuale:** Identifica ed estrae i passaggi, i paragrafi e gli articoli esatti e più pertinenti al caso.


* 
**Sintesi documentata:** Presenta all'IA solo questi estratti ufficiali per farle redigere una sintesi chiara, accurata e direttamente verificabile.


* 
**Trasparenza sui limiti:** Se un'informazione non è presente all'interno dei documenti aziendali, l'assistente lo dichiara esplicitamente in chat evitando qualsiasi tipo di invenzione.


## 2. Il Viaggio dei Documenti: Caricamento e Archiviazione (Ingestion)

Prima di poter rispondere alle domande dei funzionari, il sistema deve "studiare" ed elaborare i file (PDF, relazioni, immagini, scansioni) attraverso un processo automatico suddiviso in tre passaggi chiave:

### Passo 1: Caricamento Veloce ed Elaborazione in Background

* I file vengono inseriti dall'utente o dai sistemi gestionali dell'Ente all'interno di una cartella condivisa.


* Il sistema rileva immediatamente i nuovi arrivi e avvia l'elaborazione dei dati in background, senza rallentare le normali attività lavorative degli uffici.



### Passo 2: Lettura Intelligente e OCR di Riserva

* Il sistema apre i documenti e ne traduce il contenuto in un testo pulito e uniforme.


* In presenza di scansioni o foto sbiadite, si attiva una "vista digitale" (OCR) per riconoscere in modo accurato i caratteri scritti nelle immagini.


* Qualora i filtri o le regole di sicurezza di un modulo IA dovessero bloccare la lettura di una pagina per un falso allarme, il sistema non interrompe il flusso, ma passa istantaneamente il compito a un "occhio di riserva" (Google Vision) per garantire il recupero di ogni singola riga del testo.



### Passo 3: Scomposizione "Genitore-Figlio" (Parent Document Retriever)

Per preservare la massima velocità di ricerca senza smarrire il filo logico a causa della frammentazione dei testi (*chunking*), il sistema implementa la tecnologia *Parent Document Retriever*, strutturando l'archivio su due livelli integrati:

* 
**I "Figli" (I frammenti di ricerca):** Il documento viene spezzato in piccolissimi frammenti di poche righe ciascuno, sui quali viene effettuata la ricerca semantica. Questi frammenti garantiscono una precisione millimetrica e fungono da indice analitico ultra-veloce.


* 
**I "Genitori" (Il Documento Padre):** Il sistema conserva l'intero atto amministrativo, la pagina completa o l'articolo di legge originale in un archivio sicuro.


* 
**La logica di ricongiungimento:** Quando l'utente pone una domanda, il sistema individua in un lampo il frammento "Figlio" rilevante e recupera automaticamente l'intero "Documento Padre" di riferimento da passare all'IA. Questo garantisce al modello un quadro informativo coerente, azzerando le interpretazioni errate dovute alla decontestualizzazione delle frasi.



## 3. Capacità dell'Architettura e Limiti dei Sistemi RAG Standard

### Cosa PUÒ Fare il Sistema per l'Ente

In un contesto regolato in modo stringente dal diritto amministrativo, la sicurezza e la precisione sono requisiti tassativi. L'architettura offre capacità verticali e sicure:

**Citazione rigorosa delle fonti:** Il sistema indica con assoluta precisione gli estremi del documento da cui ha tratto l'informazione (es. *"Determina Dirigenziale n. 145 del 12/04/2024"*), permettendo al funzionario una verifica immediata sull'atto ufficiale.

**Comprensione del linguaggio settoriale:** Sfruttando la ricerca semantica, l'assistente comprende nativamente sinonimi, concetti affini e costrutti tipici della Pubblica Amministrazione.
 
**Ricerca trasversale tra uffici diversi:** Consente di interrogare contemporaneamente basi documentali eterogenee, facendo emergere correlazioni o incongruenze regolamentari tra settori differenti dell'Ente.
 
**Gestione della riservatezza e GDPR:** Il sistema filtra i documenti in base alle specifiche autorizzazioni dell'utente; mostra i dati sensibili solo al personale autorizzato e formula le risposte per gli altri utenti basandosi unicamente sugli atti pubblici.
 
**Aggiornamento in tempo reale:** L'approvazione di un nuovo atto richiede il solo caricamento nell'archivio per renderlo subito disponibile per le risposte, azzerando i costi e i tempi di riaddestramento dei modelli.

### Cosa NON PUÒ Fare un RAG (Vincoli Amministrativi e Tecnologici)

Comprendere i limiti intrinseci della tecnologia è essenziale per scongiurare rischi di conformità o errori procedurali:

**Assenza di discrezionalità amministrativa:** Il sistema descrive e sintetizza unicamente le informazioni esistenti; non può e non deve assumere decisioni discrezionali o valutare l'opportunità e la legittimità di un provvedimento.

**Mancata risoluzione delle antinomie giuridiche:** Se nel database coesistono due regolamenti contrastanti (uno vecchio del 2015 e uno nuovo del 2023), il sistema li presenterà entrambi all'utente, non possedendo la capacità giuridica per stabilire quale norma debba prevalere.

**Verifica della vigenza:** Il sistema non può conoscere autonomamente se un atto è stato annullato in autotutela, a meno che lo stato di validità del documento non sia esplicitamente aggiornato nel database di partenza.

### Il Superamento delle Barriere di Aggregazione dei RAG Convenzionali

Un'architettura RAG convenzionale è progettata solo per recuperare frammenti isolati e mostra un fallimento strutturale di fronte a richieste di sintesi olistica. Il nostro progetto supera queste barriere attraverso moduli avanzati:

**Analisi territoriale e concettuale:** Un RAG standard fallisce se si cercano informazioni su macro-aree (es. *Garfagnana* o *Mediavalle*) qualora gli atti citino solo i singoli comuni (Barga, Castelnuovo, ecc.), poiché non possiede nozioni geografiche implicite. Il nostro sistema integra invece dizionari concettuali e ontologie gerarchiche della PA che espandono automaticamente la query a tutti i territori afferenti.

**Analisi quantitativa e statistica:** Richieste come *"Quante ordinanze di sgombero sono state firmate nell'ultimo anno?"* generano un blocco nei RAG tradizionali, che estraggono solo pochi frammenti rilevanti senza poter scorrere e contare l'intero archivio.
 
**Ricostruzione storico-evolutiva e Trend Analysis:** Senza una comprensione del tempo, del nesso di causalità amministrativa (es. abrogazione implicita) o la capacità di leggere centinaia di relazioni complessive (es. criticità nella gestione degli appalti), i sistemi standard forniscono risposte sconnesse, parziali o aneddotiche. Il nostro impianto supera questo limite combinando logiche agentiche e riordinamento delle priorità.


## 4. L'Architettura Avanzata: Come Risponde alle Domande? (Il Cervello del Sistema)

Quando un funzionario inserisce una domanda all'interno della chat (es. *"Ci sono sconti e agevolazioni per l'acquisto di vetture?"*), l'assistente si comporta come un "agente intelligente" attivo e attiva una catena di controllo qualità strutturata:

```
              [ Domanda ]
                   │
                   ▼
   [ 1. Traduttore e Ontologie PA ]  ───► Mappa sinonimi e relazioni
                   │                      territoriali (es. Garfagnana)
                   ▼
    [ 2. Scomposizione Agentica ]    ───► Divide i quesiti complessi
                   │                      in sotto-ricerche separate
                   ▼
     [ 3. Ricerca Doppia Ibrida ]    ───► Combina Significato Semantico
                   │                      e Codici/Parole Sintattiche
                   ▼
    [ 4. Riordinatore di Priorità ]  ───► Seleziona gli atti più recenti
                   │                      ed esclude i duplicati
                   ▼
    [ 5. Validazione del Grader ]    ───► Un secondo modulo IA indipendente
                   │                      verifica l'ancoraggio alle fonti
         ┌─────────┴─────────┐
    (Approvato)         (Bocciato)
         │                   │
         ▼                   ▼
[ Risposta in Chat ]   [ Riformulazione Automatica ]

```

1. **Comprensione del Significato ed Espansione Semantica:** Il sistema traduce i sinonimi (capisce che *auto*, *veicoli* e *vetture* indicano lo stesso concetto) ed espande la query. Se viene citata una macro-area territoriale, il modulo concettuale mappa la richiesta includendo i singoli comuni costituenti.


2. **Scomposizione e Ricerca Doppia (Ibrida):** L'agente scompone le domande complesse o multi-fase in sotto-quesiti distinti ed esegue ricerche separate. Interroga il database unendo la ricerca semantica (significato profondo) e la ricerca sintattica (parole esatte, date e numeri di protocollo).


3. **Riordinamento delle Priorità e Verifica delle Lacune:** Il sistema analizza i risultati, posiziona in cima gli atti più pertinenti o aggiornati ed esclude i doppioni. Valuta autonomamente se la documentazione recuperata è sufficiente; in caso di lacune informative, formula una nuova strategia di ricerca interrogando nuovamente il database con chiavi differenti prima di procedere.


4. **Controllo Qualità (Il Grader Indipendente):** Un secondo modulo di intelligenza artificiale indipendente (il validatore o *Grader*) prende la risposta e la confronta rigidamente con i documenti originali dell'Ente. Se la risposta è fedele e accurata viene mostrata in chat; se contiene imprecisioni o non è sufficientemente ancorata alle fonti, viene bocciata e l'assistente riformula la ricerca in background in totale autonomia.



## 5. L'Interfaccia Utente: La Scrivania Digitale del Funzionario

Il frontend dell'applicazione è stato interamente sviluppato per offrire ai dipendenti dell'Ente il massimo livello di controllo, accessibilità visiva e trasparenza operativa:

**Accoglienza Guidata (Il Mini-Tour):** Al primo accesso assoluto, l'utente viene accompagnato da un tour interattivo che illustra la funzione dei comandi principali (gestione lingue, filtri, temi visivi, invio e pulizia dei messaggi).

**Rilevamento Automatico della Lingua:** L'interfaccia rileva la lingua del browser dell'utente (es. italiano o inglese) e si adatta automaticamente, consentendo comunque la modifica manuale immediata tramite l'icona della bandierina nella barra superiore.

**Accessibilità e Benessere Visivo (3 Temi ed Alto Contrasto):** Per prevenire l'affaticamento visivo e garantire l'uso in ogni condizione ambientale, l'app offre tre configurazioni visive, modificabili anche tramite la scorciatoia da tastiera `Alt + T`:


*Tema Chiaro:* Ottimizzato per ambienti di lavoro molto illuminati.

*Tema Scuro:* Riposante, volto a minimizzare l'emissione di luce blu degli schermi.

*Tema ad Alto Contrasto:* Progettato per ipovedenti o per l'utilizzo sotto la luce solare diretta, caratterizzato da uno sfondo nero assoluto, testi bianchi e dettagli in giallo acceso per la massima leggibilità.


**Il Filtro dei Sotto-Argomenti:** Un pulsante dedicato consente di aprire una lista di controllo con cui circoscrivere la ricerca, includendo o escludendo specifiche categorie, uffici o annualità di documenti (es. analizzare solo i contratti del 2026 o i regolamenti tecnici).

**Trasparenza nei Tempi di Attesa:** Durante l'elaborazione, l'interfaccia mostra un indicatore visivo animato. Se l'analisi di una domanda particolarmente complessa supera i 20 secondi, l'app mostra un messaggio di cortesia automatico per confermare al funzionario che il sistema è regolarmente operativo dietro le quinte.

**Fonti Cliccabili e Verificabili:** Al di sotto di ogni risposta compare un menu a tendina espandibile con l'elenco delle fonti impiegate. L'utente può cliccare sui comodi pulsanti (*chip*) recanti i nomi dei file originali per scaricare o aprire direttamente il documento sorgente nel punto esatto da cui è stato estratto il testo.

**Circuiti di Feedback con Commenti:** Ogni risposta è corredata dai pulsanti "Mi piace" e "Non mi piace". Al clic, si apre una finestra per l'inserimento di un commento facoltativo, inviato istantaneamente al database centrale per abilitare processi di ottimizzazione continua.


## 6. Efficienza Operativa, Sostenibilità Economica e Gestione degli Imprevisti

L'architettura software è governata da logiche ingegneristiche avanzate che ottimizzano le risorse computazionali ed economiche dell'amministrazione:

**Ottimizzazione per Disparità Documentale (Redistribuzione Flessibile):** Se un'interrogazione coinvolge più uffici e si rileva che la maggior parte di essi non ha prodotto atti su quel tema specifico nell'ultimo anno, il sistema non restituisce una risposta povera e non disperde potenza di calcolo. La capacità di analisi viene ridiretta in automatico per esaminare in massima profondità l'unico settore che possiede abbondante documentazione sul tema, garantendo risposte dettagliate.

**Sostenibilità dei Costi di Calcolo (Cache Semantica Composta):** Per ridurre drasticamente i costi operativi legati all'uso degli LLM, la piattaforma introduce una memoria intelligente. Se due funzionari eseguono la stessa domanda (o quesiti semanticamente affini) applicando i medesimi filtri di visibilità, il sistema riconosce la richiesta e attinge in un lampo alla risposta già convalidata, abbattendone i tempi di rilascio da secondi a millisecondi.

**Ancoraggio a Filtri Amministrativi (Hybrid Search):** Al fine di evitare anacronismi o scambi tra atti storici e correnti, l'architettura permette di blindare i confini di ricerca applicando filtri amministrativi certi (intervalli temporali rigidi o tipologie d'atto tassative) entro i quali l'IA è obbligata a muoversi, azzerando gli errori procedurali.

**Schermata di Cortesia per Manutenzione:** In presenza di disconnessioni di rete o spegnimento programmato del server per aggiornamenti tecnici, l'interfaccia previene crash applicativi mostrando una schermata chiara e rilassante con un'icona di cantiere e un pulsante per riprovare la connessione.


## 7. Cicli di Feedback e Ottimizzazione Semantica Continua

Il motore semantico applicato al contesto pubblico non è statico, ma si evolve dinamicamente assimilando l'uso quotidiano dei funzionari attraverso canali di monitoraggio attivo e automatico:

**Gestione dei Feedback Utente (Circuiti Attivi):** Le segnalazioni e i commenti inseriti manualmente dal personale permettono di intercettare minuziose sfumature procedurali o dettagli normativi strettamente locali che potrebbero sfuggire agli algoritmi di base.

**Feedback Automatico di Non-Risposta (Auto-Logging Passive):** Se il sistema rileva l'assenza totale di risposte valide nel database, o se il modulo di validazione (*Grader*) scarta il testo ritenendolo non supportato dai documenti ufficiali, si attiva una procedura di log difensivo. L'interrogazione fallita viene registrata nel database centrale in modo del tutto automatico, senza che l'utente debba effettuare alcuna operazione manuale.

**Ampliamento delle Liste Semantiche e dei Dizionari:** Gli amministratori di sistema analizzano centralmente le casistiche accumulate da entrambi i circuiti (attivi e automatici) per aggiornare le liste semantiche e le ontologie gerarchiche. Inserendo nuovi sinonimi burocratici, modificando le mappe territoriali o correggendo i collegamenti concettuali, l'intervallo semantico dell'assistente si espande costantemente: una richiesta che oggi incontra un limite informativo genererà in futuro una risposta eccellente.


## 8. Confronto Operativo delle Architetture

La tabella sottostante riassume le differenze strutturali e funzionali che distanziano le soluzioni software implementate in questo progetto rispetto a un approccio RAG tradizionale:

| Esigenza di Ricerca | RAG Standard (Frammentato) | RAG Avanzato (Nostro Progetto) |
| --- | --- | --- |
| **Precisione e Contesto degli Atti** | **Bassa:** estrae solo frasi isolate perdendo gli estremi normativi o i preamboli fondamentali dell'atto. | **Alta:** individua il punto specifico ma fornisce all'IA l'intero articolo o documento di riferimento (*Parent Document*).|
| **Domande Complesse / Multi-fase** | **Incompleta:** esegue un'unica ricerca lineare e risponde solo a una parte della domanda dell'utente.| **Completa:** l'agente scompone la domanda, interroga più volte il database e assembla i vari pezzi in una risposta organica.|
| **Riferimenti Territoriali e Sinonimi** | <br>**Nulla:** non trova corrispondenze se si cercano macro-aree geografiche descritte solo per singoli comuni o termini differenti.|**Ottimizzata:** l'analisi concettuale e il modulo dei sinonimi espandono la query ai territori afferenti e ai termini burocratici correlati.|
| **Evoluzione e Apprendimento** | **Nulla:** il sistema rimane statico; gli angoli ciechi e gli errori rimangono ignoti fino a segnalazione manuale cartacea o e-mail.| **Dinamica:** integra circuiti di feedback attivi (commenti utente) e *auto-logging* automatico delle non-risposte per ampliare costantemente i dizionari.|
| **Trasparenza e Interfaccia** | **Limitata:** risposte testuali opache, prive di controllo sulla visibilità dei dati o tracciabilità dei tempi e delle fonti. | **Massima:** include mini-tour guidato, 3 temi di accessibilità visiva (`Alt+T`), messaggi di cortesia sui tempi, filtri attivi e fonti scaricabili.|
| **Efficienza e Costi** | **Variabile:** rallenta linearmente all'aumentare dell'archivio, senza una logica di ottimizzazione dei costi computazionali.| **Predisposta:** ottimizzata a livello software tramite cache semantica composta (risposte in millisecondi), filtri amministrativi certi e redistribuzione della capacità.|

## 9. Sintesi

La scelta di implementare una soluzione integrata che fonde la flessibilità logica del backend (RAG Agentico, *Parent Document Retriever* e moduli di analisi concettuale) con la trasparenza e l'accessibilità del frontend (interfaccia multi-tema, fonti scaricabili, gestione trasparente dei tempi e degli imprevisti) dota l'Ente di uno strumento di lavoro sicuro, affidabile e all'avanguardia.

Questa architettura affronta e risolve i limiti storici dell'intelligenza artificiale (frammentazione, perdita di contesto e allucinazioni) , garantendo al personale una scrivania digitale protetta e introducendo un ciclo di auto-apprendimento basato sull'uso reale che tutela la sostenibilità economica e l'efficienza amministrativa dell'Ente.