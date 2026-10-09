# Guida al Sistema RAG Agentico Avanzato per la Ricerca Documentale nell'Ente Locale

## Introduzione e Contesto Aziendale

Gli Enti Locali (Comuni, Province, Unioni di Comuni) gestiscono quotidianamente una imponente mole di documenti amministrativi: deliberazioni di Giunta e di Consiglio, determinazioni dirigenziali, ordinanze, regolamenti, piani territoriali e atti di protocollo. Spesso, la frammentazione di questo patrimonio informativo rallenta l'attività istruttoria degli uffici e impatta sulla qualità delle risposte fornite ai cittadini.

L'introduzione di un assistente intelligente basato su architettura **RAG** (*Retrieval-Augmented Generation*) rappresenta lo strumento ideale per la ricerca interna, capace di leggere e analizzare migliaia di pagine di documenti, contratti, manuali e normative in pochi secondi. Questa tecnologia non si limita a cercare semplici parole chiave, ma "comprende" il contesto profondo degli atti dell'Ente e formula risposte sintetiche, accurate e sicure basandosi esclusivamente su di essi, senza mai inventare nulla.

Accanto all'assistente, il progetto mette a disposizione degli uffici un'applicazione di **revisione**: il personale abilitato controlla e corregge i testi letti dall'IA, ne completa i dati di catalogo e ne tiene traccia, così che la qualità dell'archivio migliori nel tempo grazie al lavoro delle persone.

## 1. La Filosofia Operativa: "L'Esame a Libro Aperto"

Le intelligenze artificiali tradizionali rispondono alle domande basandosi solo sulle informazioni generiche su cui sono state originariamente addestrate. Questo approccio le porta ad "allucinare", ovvero a inventare risposte che sembrano credibili ma sono del tutto false o non verificate.

Il nostro sistema avanzato supera questo limite comportandosi come un istruttore amministrativo o uno studente brillante che affronta un **esame a libro aperto**:

**Assenza di cieca fiducia:** Il sistema non risponde attingendo a una memoria astratta o generica.

**Interrogazione dell'archivio:** Ricevuto il quesito, interroga istantaneamente l'archivio digitale e la biblioteca dell'Ente (delibere, regolamenti, determine).

**Estrazione puntuale:** Identifica ed estrae i passaggi, i paragrafi e gli articoli esatti e più pertinenti al caso.

**Sintesi documentata:** Presenta all'IA solo questi estratti ufficiali, insieme ai dati di catalogo di ciascun atto (numero, data, oggetto, ufficio), per farle redigere una sintesi chiara, accurata e direttamente verificabile.

**Trasparenza sui limiti:** Se un'informazione non è presente all'interno dei documenti dell'Ente, l'assistente lo dichiara esplicitamente in chat evitando qualsiasi tipo di invenzione.


## 2. Il Viaggio dei Documenti: Caricamento e Archiviazione (Ingestion)

Prima di poter rispondere alle domande, il sistema deve "studiare" ed elaborare i file (PDF, relazioni, immagini, scansioni) attraverso un processo automatico suddiviso in tre passaggi chiave:

### Passo 1: Caricamento Veloce ed Elaborazione in Background

* I file vengono inseriti dall'utente o dai sistemi gestionali dell'Ente all'interno di una cartella condivisa.

* Il sistema rileva immediatamente i nuovi arrivi e avvia l'elaborazione dei dati in background, senza rallentare le normali attività lavorative degli uffici.

### Passo 2: Lettura Intelligente e OCR di Riserva

* Il sistema apre i documenti e ne traduce il contenuto in un testo pulito e uniforme.

* In presenza di scansioni o foto sbiadite, si attiva una "vista digitale" (OCR) per riconoscere in modo accurato i caratteri scritti nelle immagini.

* Qualora i filtri o le regole di sicurezza di un modulo IA dovessero bloccare la lettura di una pagina per un falso allarme, il sistema non interrompe il flusso, ma passa istantaneamente il compito a un "occhio di riserva" (Google Vision) per garantire il recupero di ogni singola riga del testo.

* Per riprodurre l'impaginazione dell'originale (rientri, firme allineate a destra, colonne), l'OCR inserisce molti spazi: prima dell'archiviazione il testo viene ripulito di quegli spazi superflui, che all'IA non servono e le sottrarrebbero spazio utile, conservando però titoli, elenchi, tabelle e colonne di dati. Il testo originale dell'OCR resta comunque disponibile ai revisori.

### Passo 3: Scomposizione "Genitore-Figlio" (Parent Document Retriever)

Per preservare la massima velocità di ricerca senza smarrire il filo logico a causa della frammentazione dei testi (*chunking*), il sistema implementa la tecnologia *Parent Document Retriever*, strutturando l'archivio su due livelli integrati:

**I "Figli" (I frammenti di ricerca):** Il documento viene spezzato in piccolissimi frammenti di poche righe ciascuno, sui quali viene effettuata la ricerca semantica. Questi frammenti garantiscono una precisione millimetrica e fungono da indice analitico ultra-veloce.

**I "Genitori" (Il Documento Padre):** Il sistema conserva l'intero atto amministrativo, la pagina completa o l'articolo di legge originale in un archivio sicuro.

**La logica di ricongiungimento:** Quando l'utente pone una domanda, il sistema individua in un lampo il frammento "Figlio" rilevante e recupera automaticamente l'intero "Documento Padre" di riferimento da passare all'IA. Questo garantisce al modello un quadro informativo coerente, azzerando le interpretazioni errate dovute alla decontestualizzazione delle frasi.

**I dati di catalogo:** Insieme al testo, ogni atto porta con sé i propri dati di catalogo (numero, data, oggetto, ufficio proponente, dirigente). L'IA li riceve accanto ai passaggi estratti e li usa per citare l'atto con precisione anche quando legge solo un frammento del documento.


## 3. Capacità dell'Architettura e Limiti dei Sistemi RAG Standard

### Cosa PUÒ Fare il Sistema per l'Ente

In un contesto regolato in modo stringente dal diritto amministrativo, la sicurezza e la precisione sono requisiti tassativi. L'architettura offre capacità verticali e sicure:

**Citazione rigorosa delle fonti:** Il sistema indica con assoluta precisione gli estremi del documento da cui ha tratto l'informazione (es. *"Determina Dirigenziale n. 145 del 12/04/2024"*), permettendo una verifica immediata sull'atto ufficiale.

**Comprensione del linguaggio settoriale:** Sfruttando la ricerca semantica, l'assistente comprende nativamente sinonimi, concetti affini e costrutti tipici della Pubblica Amministrazione.

**Ricerca trasversale tra uffici diversi:** Consente di interrogare contemporaneamente basi documentali eterogenee, facendo emergere correlazioni o incongruenze regolamentari tra settori differenti dell'Ente.

**Controllo dell'accesso ai documenti originali:** Ogni archivio può essere pubblico o riservato. Per gli archivi riservati, il file originale si apre solo dopo l'accesso con credenziali personali e solo per gli utenti abilitati a quell'archivio (vedi il capitolo 7).

**Aggiornamento in tempo reale:** L'approvazione di un nuovo atto richiede il solo caricamento nell'archivio per renderlo subito disponibile per le risposte, azzerando i costi e i tempi di riaddestramento dei modelli. Allo stesso modo, una correzione fatta in revisione è usata dall'assistente poco dopo il salvataggio.

### Cosa NON PUÒ Fare un RAG (Vincoli Amministrativi e Tecnologici)

Comprendere i limiti intrinseci della tecnologia è essenziale per scongiurare rischi di conformità o errori procedurali:

**Assenza di discrezionalità amministrativa:** Il sistema descrive e sintetizza unicamente le informazioni esistenti; non può e non deve assumere decisioni discrezionali o valutare l'opportunità e la legittimità di un provvedimento.

**Mancata risoluzione delle antinomie giuridiche:** Se nel database coesistono due regolamenti contrastanti (uno vecchio del 2015 e uno nuovo del 2023), il sistema li presenterà entrambi all'utente, non possedendo la capacità giuridica per stabilire quale norma debba prevalere.

**Verifica della vigenza:** Il sistema non può conoscere autonomamente se un atto è stato annullato in autotutela, a meno che lo stato di validità del documento non sia esplicitamente aggiornato nel database di partenza (ad esempio dai revisori, tramite i dati di catalogo).

**Riservatezza del contenuto nelle risposte:** La riservatezza di un archivio protegge l'apertura del file originale, non le risposte: la chat di un archivio risponde usando il testo di tutti i suoi documenti. I documenti che non devono comparire nelle risposte a tutti vanno quindi collocati in un archivio (e in una chat) a parte.

### Il Superamento delle Barriere di Aggregazione dei RAG Convenzionali

Un'architettura RAG convenzionale è progettata solo per recuperare frammenti isolati e mostra un fallimento strutturale di fronte a richieste di sintesi olistica. Il nostro progetto supera queste barriere attraverso moduli avanzati:

**Analisi territoriale e concettuale:** Un RAG standard fallisce se si cercano informazioni su macro-aree (es. *Garfagnana* o *Versilia*) qualora gli atti citino solo i singoli comuni (Barga, Castelnuovo, Viareggio, ecc.), poiché non possiede nozioni geografiche implicite. Il nostro sistema integra invece un dizionario concettuale che riconosce la macro-area citata e svolge una ricerca dedicata per ciascun comune che ne fa parte; nella risposta dichiara anche per quali comuni non ha trovato atti.

**Analisi quantitativa e statistica:** Richieste come *"Quante ordinanze di sgombero sono state firmate nell'ultimo anno?"* generano un blocco nei RAG tradizionali, che estraggono solo pochi frammenti rilevanti senza poter scorrere e contare l'intero archivio.

**Ricostruzione storico-evolutiva e Trend Analysis:** Senza una comprensione del tempo, del nesso di causalità amministrativa (es. abrogazione implicita) o la capacità di leggere centinaia di relazioni complessive (es. criticità nella gestione degli appalti), i sistemi standard forniscono risposte sconnesse, parziali o aneddotiche. Il nostro impianto supera questo limite combinando logiche agentiche e riordinamento delle priorità.


## 4. L'Architettura Avanzata: Come Risponde alle Domande? (Il Cervello del Sistema)

Quando viene inserita una domanda all'interno della chat (es. *"Quali lavori sulle scuole sono stati fatti in Garfagnana?"*), l'assistente si comporta come un "agente intelligente" attivo e attiva una catena di controllo qualità strutturata:

```
              [ Domanda ]
                   │
                   ▼
   [ 1. Dizionario dei Concetti ]    ───► Riconosce macro-aree e temi
                   │                      (es. Garfagnana → i suoi comuni)
                   ▼
    [ 2. Scomposizione Agentica ]    ───► Una ricerca per ogni comune
                   │                      o aspetto della domanda
                   ▼
     [ 3. Ricerca Doppia Ibrida ]    ───► Significato + combinazioni
                   │                      di parole chiave esatte
                   ▼
    [ 4. Riordinatore di Priorità ]  ───► Ordina per pertinenza ed
                   │                      esclude i duplicati
                   ▼
   [ 5. Composizione del Contesto ]  ───► Atti completi, dati di catalogo,
                   │                      spazio ripartito tra le ricerche
                   ▼
    [ 6. Validazione del Grader ]    ───► Un secondo modulo IA indipendente
                   │                      verifica l'ancoraggio alle fonti
         ┌─────────┴─────────┐
    (Approvato)         (Bocciato)
         │                   │
         ▼                   ▼
[ Risposta in Chat ]   [ Riformulazione Automatica ]

```

1. **Comprensione del Significato e Dizionario dei Concetti:** Il sistema traduce i sinonimi (capisce che *auto*, *veicoli* e *vetture* indicano lo stesso concetto) e consulta un dizionario concettuale curato dall'Ente. Un concetto viene riconosciuto sia quando è citato per nome (*"in Versilia"*) sia quando la domanda ne parla in altri termini; una macro-area territoriale viene così mappata sui singoli comuni che la compongono. Nelle conversazioni, le domande di seguito (*"e perché è stato risolto il contratto?"*) vengono completate con quanto detto prima.

2. **Scomposizione e Ricerca Doppia (Ibrida):** L'agente scompone le domande complesse in sotto-quesiti distinti, ad esempio una ricerca per ciascun comune di una macro-area, ed esegue ricerche separate. Interroga il database unendo la ricerca semantica (significato profondo) e la ricerca sintattica (parole esatte, nomi, numeri di protocollo). La ricerca sintattica privilegia gli atti che contengono **più termini insieme** (es. *nome e cognome* di un incaricato insieme a *risoluzione anticipata*) e non si lascia distrarre dai termini troppo comuni (*atti*, *lavori*), che usa solo in combinazione con altri.

3. **Riordinamento delle Priorità e Verifica delle Lacune:** Un modulo di riordinamento rilegge ogni passaggio trovato e lo confronta con la domanda, posizionando in cima gli atti più pertinenti ed escludendo i doppioni. Valuta autonomamente se la documentazione recuperata è sufficiente; in caso di lacune informative, formula una nuova strategia di ricerca interrogando nuovamente il database con chiavi differenti prima di procedere.

4. **Composizione del Contesto:** A ogni sotto-ricerca è garantito uno spazio minimo, così nessun aspetto della domanda resta escluso; lo spazio restante va alle ricerche che hanno trovato gli atti più pertinenti. Quando molti atti risultano chiaramente pertinenti (ad esempio *"tutti gli incarichi affidati a…"*), il sistema ne aggiunge altri oltre la quota di base. Di ogni atto l'IA riceve la porzione utile del documento completo e i dati di catalogo.

5. **Controllo Qualità (Il Grader Indipendente):** Un secondo modulo di intelligenza artificiale indipendente (il validatore o *Grader*) prende la risposta e la confronta rigidamente con i documenti originali dell'Ente. Se la risposta è fedele e accurata viene mostrata in chat; se contiene imprecisioni o non è sufficientemente ancorata alle fonti, viene bocciata e l'assistente riformula la ricerca in background in totale autonomia.



## 5. L'Interfaccia della Chat

Il frontend dell'applicazione è stato interamente sviluppato per offrire ai dipendenti dell'Ente il massimo livello di controllo, accessibilità visiva e trasparenza operativa:

**Accoglienza Guidata (Il Mini-Tour):** Al primo accesso assoluto, l'utente viene accompagnato da un tour interattivo che illustra la funzione dei comandi principali (gestione lingue, filtri, temi visivi, invio e pulizia dei messaggi).

**Rilevamento Automatico della Lingua:** L'interfaccia rileva la lingua del browser dell'utente (es. italiano o inglese) e si adatta automaticamente, consentendo comunque la modifica manuale immediata tramite l'icona della bandierina nella barra superiore.

**Accessibilità e Benessere Visivo (3 Temi ed Alto Contrasto):** Per prevenire l'affaticamento visivo e garantire l'uso in ogni condizione ambientale, l'app offre tre configurazioni visive, modificabili anche tramite la scorciatoia da tastiera (`Alt + T`):

*Tema Chiaro:* Ottimizzato per ambienti di lavoro molto illuminati.

*Tema Scuro:* Riposante, volto a minimizzare l'emissione di luce blu degli schermi.

*Tema ad Alto Contrasto:* Progettato per ipovedenti o per l'utilizzo sotto la luce solare diretta, caratterizzato da uno sfondo nero assoluto, testi bianchi e dettagli in giallo acceso per la massima leggibilità.

L'interfaccia è inoltre utilizzabile da tastiera e con i lettori di schermo: pulsanti, fonti e messaggi sono descritti per le tecnologie assistive.

**Il Menu dei Filtri:** Un pulsante *"Filtri"* raccoglie le opzioni che circoscrivono o approfondiscono la ricerca:

*Serie documentali:* include o esclude specifiche categorie di atti (es. solo le determine, solo i decreti).

*Periodo:* limita la ricerca agli atti di un intervallo di date, scegliendo se includere o meno quelli privi di data.

*Ricerca in maniera approfondita:* fa esaminare all'assistente un numero maggiore di fonti, per le domande che richiedono un quadro più completo, al prezzo di una risposta un po' più lenta.

Un pallino rosso sul pulsante segnala che è attivo almeno un filtro, e il suo suggerimento elenca quali.

**Trasparenza nei Tempi di Attesa:** Durante l'elaborazione, l'interfaccia mostra un indicatore visivo animato. Se l'analisi di una domanda particolarmente complessa supera i 20 secondi, l'app mostra un messaggio di cortesia automatico per confermare che il sistema è regolarmente operativo dietro le quinte.

**Fonti Cliccabili e Verificabili:** Al di sotto di ogni risposta compare un elenco espandibile delle fonti impiegate. L'utente può cliccare sui pulsanti (*chip*) recanti i nomi dei file originali per aprire o scaricare direttamente il documento sorgente e verificarne il contenuto.

**Accesso e Documenti Riservati:** Nella barra superiore un pulsante consente di accedere con le proprie credenziali e di uscire. Se una fonte appartiene a un archivio riservato e l'utente non ha ancora effettuato l'accesso, la chat lo segnala e propone l'accesso; effettuato l'accesso, il documento si apre con un clic.

**Circuiti di Feedback con Commenti:** Ogni risposta è corredata dai pulsanti "Mi piace" e "Non mi piace". Al clic, si apre una finestra per l'inserimento di un commento facoltativo, inviato istantaneamente al database centrale per abilitare processi di ottimizzazione continua.


## 6. La Revisione dei Testi

La qualità delle risposte dipende dalla qualità dei testi: una scansione sbiadita letta male dall'OCR, o un atto privo dei dati di catalogo, rende l'informazione difficile da trovare. Per questo il progetto include un'applicazione web di **revisione**, con cui il personale abilitato controlla e migliora l'archivio senza alcuna competenza tecnica.

**Elenco dei documenti:** Il revisore sceglie l'archivio e la serie, cerca per nome del file o per dati di catalogo, e filtra per stato di revisione. Per ogni documento vede lo stato e, se qualcuno lo sta già revisionando, chi.

**Originale e testo affiancati:** Aprendo un documento, il file originale (PDF o immagine) compare accanto al testo letto dall'OCR, così il confronto è immediato.

**Correzione nella vista formattata:** Il testo si corregge direttamente come appare formattato (titoli, grassetti, elenchi, tabelle), con un clic sul paragrafo da modificare, senza dover conoscere il linguaggio di impaginazione sottostante. Chi lo preferisce può comunque lavorare sul testo sorgente.

**Re-indicizzazione automatica:** Al salvataggio, il documento corretto viene rielaborato in background: di solito bastano pochi secondi o minuti perché l'assistente usi la versione corretta. Un avviso nella pagina segue l'avanzamento e segnala l'eventuale esito negativo.

**Dati di catalogo:** Il revisore può aggiungere, correggere o rimuovere i dati di catalogo del documento (es. oggetto, numero, data, ufficio). Le modifiche valgono subito per le ricerche e per le risposte, senza rielaborare il testo. I campi tecnici del sistema sono protetti e non modificabili.

**Stato di revisione:** Ogni documento è *da revisionare*, *in revisione* o *revisionato*; lo stato si aggiorna a mano o, salvando una correzione, con l'opzione *"Segna come revisionato"*.

**Storico completo:** Ogni correzione del testo, modifica dei dati di catalogo e cambio di stato è registrata con autore, data e nota facoltativa, ed è consultabile in qualsiasi momento insieme alla versione precedente.

**Un revisore alla volta:** Un documento aperto da un revisore è riservato a lui: gli altri lo consultano in sola lettura, con l'indicazione di chi lo sta revisionando, e possono modificarlo non appena viene chiuso. Se un revisore chiude la pagina senza uscire, il documento torna libero da solo dopo pochi minuti; un amministratore può comunque liberarlo prima. Nessuna modifica può così sovrascrivere il lavoro di un collega.


## 7. Utenti, Ruoli e Accesso ai Documenti

Chat e revisione condividono la stessa gestione degli utenti: ogni persona ha le proprie credenziali, e una sola autenticazione vale per entrambe.

**Ruoli:** Ogni utente riceve uno o più ruoli, che stabiliscono cosa può fare:

| Ruolo | Funzioni |
| --- | --- |
| **Lettore** | Consulta i documenti, il loro storico e i file originali degli archivi assegnati. |
| **Correttore** | Corregge i testi. |
| **Catalogatore** | Modifica i dati di catalogo. |
| **Validatore** | Cambia lo stato di revisione. |
| **Revisore** | Tutte le funzioni di revisione: testo, dati di catalogo e stato. |
| **Amministratore** | Tutte le funzioni, su tutti gli archivi, più la gestione degli utenti. |

L'interfaccia mostra a ciascuno solo le funzioni concesse dai suoi ruoli, e il sistema le verifica comunque a ogni operazione.

**Archivi assegnati:** Ogni utente lavora solo sugli archivi che gli sono stati assegnati; chi non ne ha alcuno vede un avviso che lo invita a rivolgersi a un amministratore.

**Archivi pubblici e riservati:** Ogni archivio è pubblico o riservato. I file originali di un archivio pubblico si aprono liberamente dalla chat; quelli di un archivio riservato solo dopo l'accesso, e solo per gli utenti abilitati a quell'archivio. Ogni richiesta di un file passa da questo controllo, anche quando il link viene copiato e aperto altrove.

**Sessioni sicure:** L'uscita chiude la sessione su tutti i dispositivi e le schede in cui era aperta; la disattivazione di un utente o la modifica dei suoi ruoli hanno effetto immediato.


## 8. Efficienza Operativa, Sostenibilità Economica e Gestione degli Imprevisti

L'architettura software è governata da logiche ingegneristiche avanzate che ottimizzano le risorse computazionali ed economiche dell'amministrazione:

**Ottimizzazione per Disparità Documentale (Redistribuzione Flessibile):** Se un'interrogazione coinvolge più uffici o più comuni e si rileva che la maggior parte di essi non ha prodotto atti su quel tema, il sistema non restituisce una risposta povera e non disperde capacità di analisi: lo spazio non utilizzato viene ridiretto in automatico verso le ricerche che hanno trovato documentazione abbondante e pertinente, garantendo risposte dettagliate.

**Sostenibilità dei Costi di Calcolo (Cache Semantica Composta):** Per ridurre drasticamente i costi operativi legati all'uso degli LLM, la piattaforma introduce una memoria intelligente. Se due utenti eseguono la stessa domanda (o quesiti semanticamente affini) applicando i medesimi filtri, il sistema riconosce la richiesta e attinge in un lampo alla risposta già convalidata, abbattendone i tempi di rilascio da secondi a millisecondi.

**Uso Mirato dello Spazio di Analisi:** Gli spazi di impaginazione dell'OCR vengono tolti prima che il testo arrivi all'IA, così lo spazio a disposizione per ogni risposta è dedicato al contenuto degli atti e non alla loro formattazione.

**Ancoraggio a Filtri Amministrativi (Hybrid Search):** Al fine di evitare anacronismi o scambi tra atti storici e correnti, l'architettura permette di blindare i confini di ricerca applicando filtri amministrativi certi (intervalli temporali rigidi o tipologie d'atto tassative) entro i quali l'IA è obbligata a muoversi, azzerando gli errori procedurali.

**Schermata di Cortesia per Manutenzione:** In presenza di disconnessioni di rete o spegnimento programmato del server per aggiornamenti tecnici, sia la chat sia la revisione prevengono crash applicativi mostrando una schermata chiara con un'icona di cantiere e un pulsante per riprovare la connessione.


## 9. Cicli di Feedback e Ottimizzazione Semantica Continua

Il motore semantico applicato al contesto pubblico non è statico, ma si evolve dinamicamente assimilando l'uso quotidiano attraverso canali di monitoraggio attivo e automatico:

**Gestione dei Feedback Utente (Circuiti Attivi):** Le segnalazioni e i commenti inseriti manualmente dal personale permettono di intercettare minuziose sfumature procedurali o dettagli normativi strettamente locali che potrebbero sfuggire agli algoritmi di base.

**Feedback Automatico di Non-Risposta (Auto-Logging Passive):** Se il sistema rileva l'assenza totale di risposte valide nel database, o se il modulo di validazione (*Grader*) scarta il testo ritenendolo non supportato dai documenti ufficiali, si attiva una procedura di log difensivo. L'interrogazione fallita viene registrata nel database centrale in modo del tutto automatico, senza che l'utente debba effettuare alcuna operazione manuale.

**Ampliamento delle Liste Semantiche e dei Dizionari:** Gli amministratori di sistema analizzano centralmente le casistiche accumulate da entrambi i circuiti (attivi e automatici) per aggiornare le liste semantiche e le ontologie gerarchiche. Inserendo nuovi sinonimi burocratici, modificando le mappe territoriali o correggendo i collegamenti concettuali, l'intervallo semantico dell'assistente si espande costantemente: una richiesta che oggi incontra un limite informativo genererà in futuro una risposta eccellente.

**Correzione dell'Archivio alla Fonte:** Quando una risposta sbagliata o incompleta dipende da un testo letto male o da un atto privo di dati di catalogo, la revisione permette di correggerlo una volta per tutte: dal salvataggio in poi, ogni domanda su quell'atto ne beneficia.


## 10. Confronto Operativo delle Architetture

La tabella sottostante riassume le differenze strutturali e funzionali che distanziano le soluzioni software implementate in questo progetto rispetto a un approccio RAG tradizionale:

| Esigenza di Ricerca | RAG Standard (Frammentato) | RAG Avanzato (Nostro Progetto) |
| --- | --- | --- |
| **Precisione e Contesto degli Atti** | **Bassa:** estrae solo frasi isolate perdendo gli estremi normativi o i preamboli fondamentali dell'atto. | **Alta:** individua il punto specifico ma fornisce all'IA l'intero atto di riferimento (*Parent Document*) insieme ai suoi dati di catalogo.|
| **Domande Complesse / Multi-fase** | **Incompleta:** esegue un'unica ricerca lineare e risponde solo a una parte della domanda dell'utente.| **Completa:** l'agente scompone la domanda, interroga più volte il database e assembla i vari pezzi in una risposta organica, garantendo spazio a ogni aspetto.|
| **Nomi, Codici e Termini Esatti** | **Casuale:** una ricerca per parole chiave restituisce i primi risultati trovati, sommersi dai termini più comuni. | **Mirata:** privilegia gli atti che contengono più termini della domanda insieme e usa i termini comuni solo in combinazione.|
| **Riferimenti Territoriali e Sinonimi** | **Nulla:** non trova corrispondenze se si cercano macro-aree geografiche descritte solo per singoli comuni o termini differenti.|**Ottimizzata:** il dizionario concettuale riconosce la macro-area, svolge una ricerca per ciascun comune e dichiara quelli privi di risultati.|
| **Qualità dei Testi** | **Statica:** gli errori dell'OCR restano nell'archivio e penalizzano per sempre ricerca e risposte. | **Migliorabile:** i revisori correggono testi e dati di catalogo, con storico completo, e l'assistente usa subito la versione corretta.|
| **Accesso ai Documenti** | **Indifferenziato:** chi ha il link apre qualunque file. | **Controllato:** archivi pubblici o riservati, utenti con ruoli e archivi assegnati.|
| **Evoluzione e Apprendimento** | **Nulla:** il sistema rimane statico; gli angoli ciechi e gli errori rimangono ignoti fino a segnalazione manuale cartacea o e-mail.| **Dinamica:** integra circuiti di feedback attivi (commenti utente) e *auto-logging* automatico delle non-risposte per ampliare costantemente i dizionari.|
| **Trasparenza e Interfaccia** | **Limitata:** risposte testuali opache, prive di controllo sulla visibilità dei dati o tracciabilità dei tempi e delle fonti. | **Massima:** include mini-tour guidato, 3 temi di accessibilità visiva (`Alt+T`), messaggi di cortesia sui tempi, filtri con ricerca approfondita e fonti consultabili.|
| **Efficienza e Costi** | **Variabile:** rallenta linearmente all'aumentare dell'archivio, senza una logica di ottimizzazione dei costi computazionali.| **Predisposta:** ottimizzata a livello software tramite cache semantica composta (risposte in millisecondi), filtri amministrativi certi e redistribuzione della capacità.|

## 11. Sintesi

La scelta di implementare una soluzione integrata che fonde la flessibilità logica del backend (RAG Agentico, *Parent Document Retriever*, ricerca ibrida a combinazioni e moduli di analisi concettuale) con la trasparenza e l'accessibilità del frontend (interfaccia multi-tema, filtri, fonti consultabili, gestione trasparente dei tempi e degli imprevisti) dota l'Ente di uno strumento di lavoro sicuro, affidabile e all'avanguardia.

La revisione dei testi e la gestione degli utenti completano il quadro: l'archivio non è un deposito immutabile ma un patrimonio che il personale dell'Ente controlla, corregge e protegge, con ruoli chiari e la traccia di ogni intervento.
